#!/usr/bin/env python3
"""Pack Paula's Roy inter-chaotic-stim recordings (roy_stims.h5) into an
mlPack1-style H5 for NeuronInverter-style CNNs.

Mirrors format_bbp3_for_ML.py:
  * split: valid=[0,n10] test=[n10,n10] train=[2*n10,rest]  (10/10/80, contiguous
    after a seeded shuffle -- exp data arrives ordered by neuron, sim data is
    already random)
  * normalization follows aggregate_Kaustubh_feature.normalize_volts: ONE scalar
    mean/std computed on the TRAIN split only, applied to valid/test as well.
    Adapted per-channel because probe 0 is mV and probe 1 is pA -- a single
    scalar across both would be meaningless.  Constants stored in meta + Stats.npz.
  * keys: <dom>_volts_norm (N,4000,2,1) fp16, <dom>_unit_par zeros (no ground
    truth for experimental data), meta.JSON.
Probe layout requested for this pack: probe 0 = soma Vm, probe 1 = stimulus Im.
Time base: 50 kHz decimated x5 -> dt 0.1 ms, 4000 bins (identical to
build_roy_expH5.py and the 4k50kInterChaoticB training stim).

Provenance per sample: <dom>_neuron_id, <dom>_stim_family, <dom>_coverslip,
<dom>_src_file, plus raw traces <dom>_raw_volts_mV / <dom>_stim_pA.
QC: cells CS_06302026_ch1_c2 and CS_07022026_ch2_c2 are excluded (dead/lost
cells, clipped stimulus -- see roy_analysis/qc_flags.csv).

Run: python3 format_royexp_for_ML.py --royH5 <roy_stims.h5> --outPath <dir>
"""
import argparse
import json
import os

import h5py
import numpy as np

CHAOTIC = ["Roy100", "Roy500", "Roy1000", "Roy1500", "Roy2000"]
QC_EXCLUDE = ["CS_06302026_ch1_c2", "CS_07022026_ch2_c2"]
DECIM = 5
N_TBIN = 4000
N_PAR = 15      # dummy width, == len(input_meta.include) of the consuming models
SEED = 20260826

# --target ca3ft: pack for voltage-only fine-tuning of the CA3 jaxley models
# (ca3_supervised_interchaoticB_k128 warm start).  Differences vs generic:
#   * volts z-scored with the FIXED sim-side constants (toolbox/jaxley_utils.py
#     VOLT_NORM_MEAN/STD) so HybridLoss._normalize_volts and the pack share one
#     z-space,
#   * 1 leading edge-pad bin -> 4001 time bins (the k128 CNN hard-codes a 4001
#     reshape); pair with voltage_loss.sim_t_skip_ms: 99.9 for exact alignment,
#   * unit_par is (N,6) zeros with column 0 = stim-family index into CHAOTIC —
#     read by voltage_loss.stim_from_label (labels are otherwise masked),
#   * input_meta/simu_info blocks carry the CA3 parName + phys_par_range from
#     the k128 sum_train.yaml so Trainer.patch_h5meta and HybridLoss can run.
CA3_VOLT_XM = -60.0951997   # == jaxley_utils.VOLT_NORM_MEAN
CA3_VOLT_XS = 18.95055671   # == jaxley_utils.VOLT_NORM_STD
CA3_PAR_NAMES = ["CA3_g_leak", "CA3_gbar_na3", "CA3_gkdrbar_kdr",
                 "CA3_gkabar_kap", "CA3_gbar_km", "CA3_gkdbar_kd"]
CA3_PHYS_PAR_RANGE = [[3.9417e-05, 0.5, "S/cm^2"], [0.04, 0.5, "S/cm^2"],
                      [0.01, 0.5, "S/cm^2"], [0.04, 0.5, "S/cm^2"],
                      [0.00052, 0.5, "S/cm^2"], [0.00025, 0.5, "S/cm^2"]]
# rig-accurate per-family stims (v2 sessions): holding -0.0496 nA + fitted
# slope x 5k50kInterChaoticB, in /pscratch/sd/k/ktub1999/main/DL4neurons2/stims
CA3_STIM_STEMS = [f + "_icav2_5k" for f in CHAOTIC]
# --stimChannel fixed: probe 1 = recorded Im on a FIXED scale (nA / 0.25) so a
# CNN that takes the stim as an input channel sees the same space as the
# synthetic ms2ch packs (gen_ca3_sharded --per-sample-stims).  MUST equal
# toolbox/jaxley_utils.py STIM_NORM_SCALE_NA.
STIM_SCALE_NA = 0.25


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--royH5", default="/global/homes/k/ktub1999/Neuron/neuron4/"
                   "neuroninverter/exp_data_paula_v2/roy_analysis/roy_stims.h5")
    p.add_argument("--outPath", default="/pscratch/sd/k/ktub1999/RoyExpPack")
    p.add_argument("--outName", default="RoyExpChaotic")
    p.add_argument("--target", choices=["generic", "ca3ft"], default="generic",
                   help="ca3ft: CA3 fine-tune pack (fixed norm, 4001 bins, "
                        "6-wide labels with stim-family index in col 0)")
    p.add_argument("--splitByNeuron", action="store_true",
                   help="leakage-free split: whole neurons per domain")
    p.add_argument("--evalOnly", action="store_true",
                   help="ALL samples -> test (train/valid empty); for tiny subsets "
                        "that are only ever scored, never fine-tuned (roy_analysis_clean)")
    p.add_argument("--stimChannel", choices=["trainsplit", "fixed"],
                   default="trainsplit",
                   help="fixed: probe 1 = recorded Im in nA / %.2f (matches the "
                        "synthetic ms2ch packs); trainsplit: legacy z-score with "
                        "train-split scalars" % STIM_SCALE_NA)
    args = p.parse_args()
    for a in vars(args):
        print("myArg:", a, getattr(args, a))
    return args


def main():
    args = get_parser()
    os.makedirs(args.outPath, exist_ok=True)

    vmL, imL, nidL, famL, csL, srcL = [], [], [], [], [], []
    with h5py.File(args.royH5, "r") as f:
        for nid in sorted(f["neurons"].keys()):
            if nid in QC_EXCLUDE:
                print("QC-excluded:", nid)
                continue
            g = f["neurons"][nid]
            for fam in CHAOTIC:
                if fam not in g:
                    continue
                fg = g[fam]
                assert fg.attrs["rate_hz"] == 50000, (nid, fam)
                vm, im = fg["Vm"][:], fg["Im"][:]
                srcs = [s.decode() if isinstance(s, bytes) else s
                        for s in fg["source_files"][:]]
                for r in range(vm.shape[0]):
                    vd = vm[r][::DECIM][:N_TBIN]
                    cd = im[r][::DECIM][:N_TBIN]
                    assert vd.shape[0] == N_TBIN, (nid, fam, r, vd.shape)
                    vmL.append(vd)
                    imL.append(cd)
                    nidL.append(nid)
                    famL.append(fam)
                    csL.append(bool(g.attrs["coverslip"]))
                    srcL.append(srcs[r])

    volts = np.stack(vmL).astype(np.float64)       # (N,4000) mV
    stims = np.stack(imL).astype(np.float64)       # (N,4000) pA
    nid = np.array(nidL)
    fam = np.array(famL)
    cs = np.array(csL)
    src = np.array(srcL)
    totSamp = volts.shape[0]
    print("M: collected", totSamp, "samples,", len(set(nidL)), "neurons")

    # seeded shuffle, then the format_bbp3_for_ML contiguous split
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(totSamp)
    volts, stims, nid, fam, cs, src = (a[perm] for a in (volts, stims, nid, fam, cs, src))

    nval = totSamp // 10
    split_neurons = None
    if args.splitByNeuron:
        # leakage-free: whole neurons per domain, filled to ~10/10/80 by sample
        # count; reorder samples valid|test|train so split_index stays contiguous
        rng2 = np.random.RandomState(SEED + 1)
        uniq = sorted(set(nidL))
        rng2.shuffle(uniq)
        counts = {u: int((nid == u).sum()) for u in uniq}
        buckets, acc = {"valid": [], "test": [], "train": []}, 0
        for u in uniq:
            if acc < nval:
                buckets["valid"].append(u)
            elif acc < 2 * nval:
                buckets["test"].append(u)
            else:
                buckets["train"].append(u)
            acc += counts[u]
        order = np.concatenate([np.flatnonzero(np.isin(nid, buckets[d]))
                                for d in ["valid", "test", "train"]])
        volts, stims, nid, fam, cs, src = (a[order] for a in (volts, stims, nid, fam, cs, src))
        nva = sum(counts[u] for u in buckets["valid"])
        nte = sum(counts[u] for u in buckets["test"])
        split_index = {"valid": [0, nva], "test": [nva, nte],
                       "train": [nva + nte, totSamp - nva - nte]}
        split_neurons = {d: buckets[d] for d in buckets}
        print("M:splitByNeuron neurons v/t/tr = %d/%d/%d"
              % tuple(len(buckets[d]) for d in ["valid", "test", "train"]))
    else:
        split_index = {"valid": [0, nval], "test": [nval, nval],
                       "train": [2 * nval, totSamp - 2 * nval]}
    if args.evalOnly:
        split_index = {"valid": [0, 0], "test": [0, totSamp], "train": [totSamp, 0]}
        split_neurons = {"valid": [], "test": sorted(set(nidL)), "train": []}
        print("M:evalOnly -> all %d samples in test" % totSamp)
    print("M:split_index", split_index)

    # train-split scalar stats, per channel (aggregate_Kaustubh_feature style).
    # ca3ft overrides the VOLTS channel with the fixed sim-side constants so the
    # pack z-space matches HybridLoss._normalize_volts; stim channel is unused
    # by training (probsSelect 0) and keeps the train-split stats.
    i0, ln = split_index["train"]
    if args.evalOnly:            # no train split: stats over everything
        i0, ln = 0, totSamp
    xm_v, xs_v = float(volts[i0:i0 + ln].mean()), float(volts[i0:i0 + ln].std())
    xm_s, xs_s = float(stims[i0:i0 + ln].mean()), float(stims[i0:i0 + ln].std())
    print("M:train stats  volts xm=%.4f xs=%.4f   stim xm=%.4f xs=%.4f"
          % (xm_v, xs_v, xm_s, xs_s))
    if args.target == "ca3ft":
        xm_v, xs_v = CA3_VOLT_XM, CA3_VOLT_XS
        print("M:ca3ft fixed volts norm xm=%.4f xs=%.4f" % (xm_v, xs_v))
    np.savez(os.path.join(args.outPath, "Stats.npz"),
             Mean=xm_v, Std=xs_v, StimMean=xm_s, StimStd=xs_s)

    n_tout = N_TBIN + 1 if args.target == "ca3ft" else N_TBIN
    fam_idx = np.array([CHAOTIC.index(f_) for f_ in fam.astype(str)])
    bigD = {}
    for dom in ["train", "valid", "test"]:
        ioff, myLen = split_index[dom]
        sl = slice(ioff, ioff + myLen)
        X = np.zeros((myLen, n_tout, 2, 1), dtype=np.float16)
        X[:, -N_TBIN:, 0, 0] = ((volts[sl] - xm_v) / xs_v).astype(np.float16)
        if args.stimChannel == "fixed":
            X[:, -N_TBIN:, 1, 0] = (stims[sl] / 1000.0 / STIM_SCALE_NA).astype(np.float16)
        else:
            X[:, -N_TBIN:, 1, 0] = ((stims[sl] - xm_s) / xs_s).astype(np.float16)
        if n_tout > N_TBIN:      # leading edge-pad (k128 CNN hard-codes 4001)
            X[:, :n_tout - N_TBIN] = X[:, n_tout - N_TBIN:n_tout - N_TBIN + 1]
        bigD[dom + "_volts_norm"] = X
        if args.target == "ca3ft":
            U = np.zeros((myLen, len(CA3_PAR_NAMES)), dtype=np.float64)
            U[:, 0] = fam_idx[sl]           # stim-family index for stim_from_label
            bigD[dom + "_unit_par"] = U
        else:
            bigD[dom + "_unit_par"] = np.zeros((myLen, N_PAR), dtype=np.float64)
        bigD[dom + "_raw_volts_mV"] = volts[sl].astype(np.float16)
        bigD[dom + "_stim_pA"] = stims[sl].astype(np.float16)
        bigD[dom + "_neuron_id"] = nid[sl].astype("S")
        bigD[dom + "_stim_family"] = fam[sl].astype("S")
        bigD[dom + "_coverslip"] = cs[sl]
        bigD[dom + "_src_file"] = src[sl].astype("S")
        print("M:%s volts_norm %s  norm mean %+0.3f std %0.3f"
              % (dom, X.shape, float(X[:, :, 0, 0].astype(np.float32).mean()),
                 float(X[:, :, 0, 0].astype(np.float32).std())))

    # canonical stim waveforms kept unsplit, like format_bbp3 keeps stim_names
    for f_ in CHAOTIC:
        m = fam == f_
        if m.any():
            w = stims[m]
            bigD["stim_" + f_ + "_pA"] = np.median(
                w - np.median(w[:, :2000], axis=1, keepdims=True), axis=0
            ).astype(np.float32)

    meta = {
        "message": "Paula exp Roy inter-chaotic stim pack: probe0=soma Vm, probe1=stim Im",
        "source_h5": args.royH5,
        "stim_names": ["chaotic"],
        "probe_names": ["soma", "stimulus"],
        "num_total_samples": int(totSamp),
        "num_neurons": int(len(set(nidL))),
        "qc_excluded_neurons": QC_EXCLUDE,
        "pack_info": {"split_index": split_index, "pack_conf": 1,
                      "shuffle_seed": SEED,
                      "full_input_h5": os.path.basename(args.royH5)},
        "norm": {"mode": "train-split scalar, per channel",
                 "source": "aggregate_Kaustubh_feature.normalize_volts adapted",
                 "volts_xm": xm_v, "volts_xs": xs_v,
                 "stim_xm": xm_s, "stim_xs": xs_s},
        "dt_ms": 0.1, "decimation": DECIM,
        "stim_scales": CHAOTIC,
        "params_dummy": [0, N_PAR],
        "split_caveat": "sample-level split as in format_bbp3_for_ML: reps of one "
                        "neuron can land in different domains; use <dom>_neuron_id "
                        "to regroup for leakage-free evaluation",
    }
    if args.splitByNeuron:
        meta["split_caveat"] = "splitByNeuron: whole neurons per domain (leakage-free)"
        meta["split_neurons"] = split_neurons
    if args.evalOnly:
        meta["split_caveat"] = "evalOnly: ALL samples in test; train/valid empty"
        meta["split_neurons"] = split_neurons
    if args.target == "ca3ft":
        meta["message"] += " [ca3ft: fixed norm, 4001 bins, family idx in label col 0]"
        meta["params_dummy"] = [1, len(CA3_PAR_NAMES)]   # cols 1.. are dummy; col 0 = family idx
        meta["lead_pad_bins"] = n_tout - N_TBIN
        meta["norm"] = {"mode": "fixed sim-side constants (volts), train-split scalar (stim)",
                        "source": "toolbox/jaxley_utils.py VOLT_NORM_MEAN/STD",
                        "volts_xm": xm_v, "volts_xs": xs_v,
                        "stim_xm": xm_s, "stim_xs": xs_s}
        if args.stimChannel == "fixed":
            meta["norm"]["mode"] = "fixed sim-side constants (volts), fixed nA scale (stim)"
            meta["norm"]["stim_scale_nA"] = STIM_SCALE_NA
            meta["message"] += " [stim channel: recorded Im, nA/%.2f fixed scale]" % STIM_SCALE_NA
        meta["stim_from_label"] = {"label_col": 0, "stim_family_order": CHAOTIC,
                                   "stim_names": CA3_STIM_STEMS,
                                   "holding_nA": -0.0496,
                                   "slope_note": "fitted per-family medians vs "
                                                 "5k50kInterChaoticB; see Roy*_icav2_5k.csv"}
        meta["input_meta"] = {"parName": CA3_PAR_NAMES,
                              "phys_par_range": CA3_PHYS_PAR_RANGE,
                              "num_phys_par": len(CA3_PAR_NAMES),
                              "num_time_bins": n_tout,
                              "cell_name": "ca3_pyramidal"}
        meta["simu_info"] = {"probe_names": ["soma", "stimulus"],
                             "stim_names": ["RoyChaoticMixed"],
                             "source": "experimental (Paula v2 Roy recordings)"}

    outF = os.path.join(args.outPath, args.outName + ".mlPack1.h5")
    with h5py.File(outF, "w") as hf:
        for k, v in bigD.items():
            hf.create_dataset(k, data=v)
        hf.create_dataset("meta.JSON",
                          data=np.array([json.dumps(meta)],
                                        dtype=h5py.special_dtype(vlen=str)),
                          dtype=h5py.special_dtype(vlen=str))
    print("M:done", outF, "(%.1f MB)" % (os.path.getsize(outF) / 1e6))


if __name__ == "__main__":
    main()
