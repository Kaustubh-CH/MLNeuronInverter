#!/usr/bin/env python3
"""Pack the Roy* ABF sweeps into per-stim mlPack1 HDF5s for NeuronInverter.

Normalization follows packBBP3/aggregate_Kaustubh.py :: normalize_volts, which
overrides the per-probe statistics with the fixed training constants
    xm = -60.0951997 , xs = 18.95055671
so experimental data lands on exactly the scale the model was trained on.

Time base: the model wants 4000 bins at dt=0.1 ms (400 ms).  The recordings are
0.4 s at 50 kHz = 20000 samples, so we take every 5th sample -- the same thing
NEURON does when it reports at dt=0.1 ms.  The Roy stimulus is bit-for-bit
stims/4k50kInterChaoticB.csv (= 5k50kInterChaoticB[1000:5000]), the waveform the
model was trained on, so the 4000-bin window aligns with no time shift.

Run:  shifter --image=balewski/ubu20-neuron8:v5 python3 build_roy_expH5.py [nPar] [outDir]

nPar must equal the consuming model's outputSize -- 15 for the ALL_CELLS
probescan_exc models (which predict only the 15 `include` parameters), 19 for the
per-cell L5_TTPC1cADpyr0 models (which predict all of them).  A model reading an
h5 built for the other width dies with "All arrays must be of the same length",
so each width gets its own output directory.

IV mode:  shifter --image=balewski/ubu20-neuron8:v5 python3 build_roy_expH5.py iv [outDir]

Packs the held-out IV-step protocol 26422000.abf (30 sweeps, 10 kHz, 500 ms
steps) into IVsteps.mlPack1.h5 in the same directory as the Roy packs.  This
pack is VALIDATION-ONLY: the model's inputShape is [4000, 1] and these sweeps
are 10330 bins, so it can never be CNN input.  It exists so the scoring and
plotting side reads exp traces through one schema (raw_volts_mV / stim_pA).
The recording is already at dt 0.1 ms -- no decimation, no resampling.
"""
import json
import os
import re
import sys

import h5py
import numpy as np
import pyabf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "exp_data_paula")
PREFIX = "Roy"

# from aggregate_Kaustubh.py::normalize_volts (hardcoded training stats)
XM = -60.0951997
XS = 18.95055671

DECIM = 5          # 50 kHz -> 10 kHz  (dt 0.02 ms -> 0.1 ms)
N_TBIN = 4000      # model inputShape [4000, 1]
DEF_OUT = "/global/homes/k/ktub1999/ExperimentalData/PyForEphys/RoyPaula"

IV_MODE = len(sys.argv) > 1 and sys.argv[1] == "iv"
if IV_MODE:
    N_PAR = 15
    OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else DEF_OUT
    NORM = "fixed"
    BLANK_FROM_MS = None
    BLANK_MV = -60.0
else:
    # Dummy-target width must equal the model's outputSize, not len(parName): the
    # dataloader sets outputSize = test_unit_par.shape[1], and predictExp then zips
    # it against the *included* parameter names.  A mismatch raises
    # "All arrays must be of the same length" when the prediction CSV is built.
    N_PAR = int(sys.argv[1]) if len(sys.argv) > 1 else 15   # == len(input_meta.include)
    OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else DEF_OUT
    # 'fixed'  : the hardcoded constants in aggregate_Kaustubh.py::normalize_volts
    # 'persamp': per-sweep mean 0 / std 1, which is what the packs built through
    #            aggregate_Kaustubh_feature.py::normalize_volts approximate and what
    #            this model's own unitParamsExactinputVolts.hdf5 shows it was fed
    NORM = sys.argv[3] if len(sys.argv) > 3 else "fixed"
    # Optionally overwrite the tail of every sweep with a constant.  The Roy stimulus
    # is active over bins 501..3500 (50.1..350.0 ms), so blanking from 350 ms removes
    # only the post-stimulus tail and leaves the driven response untouched.  The
    # blank is applied to the mV trace before normalization, and to the raw_volts_mV
    # copy as well, so the plots show what the model actually saw.
    BLANK_FROM_MS = float(sys.argv[4]) if len(sys.argv) > 4 else None
    BLANK_MV = float(sys.argv[5]) if len(sys.argv) > 5 else -60.0

IV_ABF = "26422000.abf"
IV_N_TBIN = 10330  # sweeps are already at dt 0.1 ms; keep them whole


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    groups = {}
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".abf"):
            continue
        abf = pyabf.ABF(os.path.join(DATA_DIR, fname))
        if not abf.protocol.startswith(PREFIX):
            continue
        amp = int(re.match(r"Roy(\d+)", abf.protocol).group(1))
        abf.setSweep(0, channel=0)
        v = abf.sweepY.copy()
        abf.setSweep(0, channel=1)
        cur = abf.sweepY.copy()
        groups.setdefault(amp, []).append((fname, v, cur, abf.dataRate))

    for amp in sorted(groups):
        recs = groups[amp]
        nsamp = len(recs)
        volts_norm = np.zeros((nsamp, N_TBIN, 1, 1), dtype=np.float64)
        raw = np.zeros((nsamp, N_TBIN), dtype=np.float32)
        stim = np.zeros((nsamp, N_TBIN), dtype=np.float32)

        for k, (fname, v, cur, rate) in enumerate(recs):
            assert rate == 50000, f"{fname}: expected 50 kHz, got {rate}"
            vd = v[::DECIM][:N_TBIN].astype(np.float32)
            cd = cur[::DECIM][:N_TBIN]
            assert vd.shape[0] == N_TBIN, f"{fname}: got {vd.shape[0]} bins"
            if BLANK_FROM_MS is not None:
                b0 = int(round(BLANK_FROM_MS / 0.1))
                assert 0 <= b0 < N_TBIN, f"blank start {b0} out of range"
                vd[b0:] = BLANK_MV
            if NORM == "persamp":
                vf = vd
                volts_norm[k, :, 0, 0] = (vf - vf.mean()) / vf.std()
            else:
                volts_norm[k, :, 0, 0] = (vd - XM) / XS
            raw[k] = vd
            stim[k] = cd

        # targets are unknown for experimental data -- predictExp only uses them
        # to compute a meaningless "loss", exactly as in the reference mlPack1.
        unit_par = np.zeros((nsamp, N_PAR), dtype=np.float64)

        meta = {
            "message": "Paula experimental data, Roy chaotic stim (exp_data_paula)",
            "Stim": "4k50kInterChaoticB (= 5k50kInterChaoticB[1000:5000])",
            "stim_scale_name": f"Roy{amp}",
            "stim_peak_pA": float(np.abs(stim[0] - np.median(stim[0])).max()),
            "norm": ({"mode": "persamp", "source": "per-sweep mean0/std1"}
                     if NORM == "persamp" else
                     {"mode": "fixed", "xm": XM, "xs": XS,
                      "source": "aggregate_Kaustubh.normalize_volts"}),
            "abf_files": [r[0] for r in recs],
            "dt_ms": 0.1,
            "decimation": DECIM,
            "blank": (None if BLANK_FROM_MS is None else
                      {"from_ms": BLANK_FROM_MS, "value_mV": BLANK_MV}),
            "params": [0, N_PAR],
        }
        outF = os.path.join(OUT_DIR, f"Roy{amp}.mlPack1.h5")
        with h5py.File(outF, "w") as hf:
            hf.create_dataset("test_volts_norm", data=volts_norm)
            hf.create_dataset("test_unit_par", data=unit_par)
            hf.create_dataset("meta.JSON",
                              data=np.array([json.dumps(meta)],
                                            dtype=h5py.special_dtype(vlen=str)),
                              dtype=h5py.special_dtype(vlen=str))
            # provenance for the final input-vs-output plot; unused by the loader
            hf.create_dataset("raw_volts_mV", data=raw)
            hf.create_dataset("stim_pA", data=stim)
        print(f"{os.path.basename(outF)}: volts_norm {volts_norm.shape}  "
              f"norm mean {volts_norm.mean():+.3f} std {volts_norm.std():.3f}  "
              f"raw V [{raw.min():.1f},{raw.max():.1f}] mV  "
              f"stim peak {meta['stim_peak_pA']:.0f} pA  <- {[r[0] for r in recs]}")


def iv_main():
    os.makedirs(OUT_DIR, exist_ok=True)
    abf = pyabf.ABF(os.path.join(DATA_DIR, IV_ABF))
    assert abf.dataRate == 10000, f"expected 10 kHz, got {abf.dataRate}"
    assert abf.sweepCount == 30, f"expected 30 sweeps, got {abf.sweepCount}"
    abf.setSweep(0, channel=1)
    assert abf.sweepUnitsY.strip() == "pA", \
        f"channel 1 units {abf.sweepUnitsY!r}, adcNames={abf.adcNames}"

    nsamp = abf.sweepCount
    volts_norm = np.zeros((nsamp, IV_N_TBIN, 1, 1), dtype=np.float64)
    raw = np.zeros((nsamp, IV_N_TBIN), dtype=np.float32)
    stim = np.zeros((nsamp, IV_N_TBIN), dtype=np.float32)
    baseline = np.zeros(nsamp)
    step_raw = np.zeros(nsamp)
    for k in range(nsamp):
        abf.setSweep(k, channel=0)
        v = abf.sweepY[:IV_N_TBIN].astype(np.float32)
        abf.setSweep(k, channel=1)
        cur = abf.sweepY[:IV_N_TBIN].astype(np.float64)
        assert v.shape[0] == IV_N_TBIN, f"sweep {k}: got {v.shape[0]} bins"
        volts_norm[k, :, 0, 0] = (v - XM) / XS
        raw[k] = v
        stim[k] = cur
        baseline[k] = np.median(cur[0:1100])       # step onset ~bin 1162
        step_raw[k] = np.median(cur[1662:5662])    # step interior, skip edges
    rel = step_raw - baseline

    # Detect the step window on the largest-|rel| sweep, as in
    # DL4neurons2/stims/Stim_scipts/make_ivstep_stims.py.
    j = int(np.argmax(np.abs(rel)))
    d = np.diff(stim[j].astype(np.float64))
    idx = np.where(np.abs(d) > 0.5 * abs(rel[j]))[0]
    on, off_last = int(idx[0]) + 1, int(idx[-1])
    assert abs(on - 1162) <= 2 and abs(off_last - 6161) <= 2, \
        f"step window {on}..{off_last}, expected ~1162..6161"

    unit_par = np.zeros((nsamp, N_PAR), dtype=np.float64)
    meta = {
        "message": "Paula experimental data, IV-step protocol (exp_data_paula)",
        "protocol": abf.protocol,
        "validation_only": True,
        "norm": {"mode": "fixed", "xm": XM, "xs": XS,
                 "source": "aggregate_Kaustubh.normalize_volts"},
        "abf_files": [IV_ABF],
        "dt_ms": 0.1,
        "decimation": 1,
        "step_on_bin": on,
        "step_off_bin": off_last,
        "step_on_ms": on * 0.1,
        "step_off_ms": (off_last + 1) * 0.1,
        "zero_sweep_index": int(np.argmin(np.abs(rel))),
        "baseline_pA": baseline.round(3).tolist(),
        "step_pA_raw": step_raw.round(3).tolist(),
        "step_pA_rel": rel.round(3).tolist(),
        "blank": None,
        "params": [0, N_PAR],
    }
    outF = os.path.join(OUT_DIR, "IVsteps.mlPack1.h5")
    with h5py.File(outF, "w") as hf:
        hf.create_dataset("test_volts_norm", data=volts_norm)
        hf.create_dataset("test_unit_par", data=unit_par)
        hf.create_dataset("meta.JSON",
                          data=np.array([json.dumps(meta)],
                                        dtype=h5py.special_dtype(vlen=str)),
                          dtype=h5py.special_dtype(vlen=str))
        hf.create_dataset("raw_volts_mV", data=raw)
        hf.create_dataset("stim_pA", data=stim)
        hf.create_dataset("step_pA_raw", data=step_raw)
        hf.create_dataset("step_pA_rel", data=rel)
        hf.create_dataset("baseline_pA", data=baseline)

    spikes = [int(np.sum((raw[k][1:] >= -10) & (raw[k][:-1] < -10)))
              for k in range(nsamp)]
    print(f"{os.path.basename(outF)}: raw {raw.shape}  "
          f"V [{raw.min():.1f},{raw.max():.1f}] mV  "
          f"rel [{rel.min():.1f},{rel.max():.1f}] pA  "
          f"step {on * 0.1:.1f}..{(off_last + 1) * 0.1:.1f} ms")
    print(f"spike counts per sweep: {spikes}")


if __name__ == "__main__":
    iv_main() if IV_MODE else main()
