#!/usr/bin/env python
"""Score a dt-0.2 L5 nc2 model (the fp32/dt0.2 voltage-only pilot, its 200k successor, or an
experimental fine-tune of them) on Paula's Roy v2 recordings and overlay the re-simulated traces.

Same flow as plot_exp_overlay_royv2_l5.py (2026-08-26), adapted to models whose input is the
0.2 ms grid (2001 bins):
  1. CNN on the dt-0.2 twin of the exp pack (RoyExpPack_l5dt02: volts_norm[:, ::2] of the ca3ft
     pack, fixed-norm soma trace, 2001 bins = 1-bin lead pad + recording samples at t=0.1,0.3,..),
  2. unit -> physical via the MODEL's own input_meta.phys_par_range (19 par, "lin" rows honoured),
     tanh-clamped as trained (--noTanh to skip),
  3. jaxley re-sim with cell l5ttpc (L5TTPC_NCOMP from the env, default 2) at solver dt --simDt
     (0.2 = the trained physics) under each family's rig-recorded current Roy<amp>_icaRec_5k
     (5000 pts x 0.1 ms = 500 ms; the recording is its last 400 ms) at stim_scale --stimScale
     (1.0 = the real injected current; the module's 1.5 is a synthetic-training device),
  4. align sim bins [500:2500] (t = 100..499.8 ms) with the recording decimated to the same grid,
     score mse_fixed / mse_z / dtw_z / spike counts, write per-family PDFs, CSVs, roy_traces.npz
     (data/sim/rig-current per family, for scripts/rin_fit.py), a unit-param
     boxplot and a composite overlay grid PNG (rows = neurons, cols = amplitudes).
Requires JAX_ENABLE_X64=true (set below) for the fp64 re-simulation.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")   # before any jax import
os.environ.setdefault("L5TTPC_NCOMP", "2")
import argparse, importlib, json, time
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from evaluate_voltage import load_trained_model
from toolbox import JaxleyBridge
from toolbox.jaxley_utils import (phys_par_range_to_arrays, phys_par_range_linear_mask,
                                  unit_to_phys_torch, VOLT_NORM_MEAN, VOLT_NORM_STD, load_stim_csv)
from toolbox.soft_dtw import soft_dtw_loss

PACK_DEF = "/pscratch/sd/k/ktub1999/RoyExpPack_l5dt02/RoyExpChaotic.mlPack1.h5"
AMPS = [500, 1000, 1500, 2000]    # Roy100 dropped from the study (user, 2026-08-26)
SIM_BATCH = 8


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--modelPath", required=True, help="<run>/out of a dt-0.2 L5 model")
    p.add_argument("--packFile", default=PACK_DEF)
    p.add_argument("--dom", default="test", choices=["train", "valid", "test"])
    p.add_argument("--outDir", default=None, help="default <modelPath>/exp_royv2_l5dt02")
    p.add_argument("--cellName", default="l5ttpc")
    p.add_argument("--stimSet", default="icaRec", choices=["icaRec", "icav2", "ica"],
                   help="Roy<amp>_<set>_5k CSV family; icaRec = the exact recorded rig current")
    p.add_argument("--simDt", type=float, default=0.2, help="solver dt (ms); 0.2 = trained physics")
    p.add_argument("--stimScale", type=float, default=1.0,
                   help="multiplier on the stim CSV; 1.0 = real injected current")
    p.add_argument("--numOverlay", type=int, default=4)
    p.add_argument("--maxN", type=int, default=0, help="cap traces per family (0 = all)")
    p.add_argument("--noTanh", action="store_true", help="skip the tanh clamp on the CNN output")
    p.add_argument("--tag", default="", help="label in figure titles")
    return p.parse_args()


def zscore(v):
    return (v - v.mean(axis=1, keepdims=True)) / (v.std(axis=1, keepdims=True) + 1e-6)


def zfix(v):
    return (v - VOLT_NORM_MEAN) / VOLT_NORM_STD


def spikes_pos(v, thr=0.0):
    return ((v[:, 1:] > thr) & (v[:, :-1] <= thr)).sum(axis=1)


def sim_family(pred_phys, cell_name, stim_name, stim_scale):
    """Chunked NO-GRAD jaxley forward on the cached handle -> (N, T_sim) soma mV."""
    import jax.numpy as jnp
    handle = JaxleyBridge.get_handle(cell_name, stim_name, stim_scale=stim_scale)
    pp = jnp.asarray(pred_phys.detach().cpu().numpy())
    outs = []
    for i0 in range(0, pp.shape[0], SIM_BATCH):
        pg = pp[i0:i0 + SIM_BATCH]
        ng = pg.shape[0]
        if ng < SIM_BATCH:
            pg = jnp.concatenate(
                [pg, jnp.broadcast_to(pg[:1], (SIM_BATCH - ng,) + pg.shape[1:])], axis=0)
        v = handle.simulate_batch(pg)                 # (B, n_rec, T_out)
        outs.append(np.asarray(v[:ng, 0, :]))
    return np.concatenate(outs, axis=0), float(handle.out_dt)


def main():
    args = get_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, trainMD = load_trained_model(args.modelPath, device)
    outDir = args.outDir or os.path.join(args.modelPath, "exp_royv2_l5dt02")
    os.makedirs(outDir, exist_ok=True)

    # unit -> phys mapping: the box the LOSS used (voltage_loss.phys_par_range, present for
    # in-loop and exp fine-tune runs, where the pack meta is the exp pack's 6-par CA3 copy),
    # else the run's input_meta (supervised runs).  Names from the cell module.
    im = trainMD.get("input_meta") or {}
    vl = (trainMD.get("train_params") or {}).get("voltage_loss") or {}
    ppr = vl.get("phys_par_range") or im["phys_par_range"]
    mod = importlib.import_module(f"toolbox.jaxley_cells.{args.cellName}")
    par_names = list(mod.PARAM_KEYS)
    if len(par_names) != len(ppr):
        par_names = list(im.get("parName") or [f"p{i}" for i in range(len(ppr))])
    T_model = int(im.get("num_time_bins") or trainMD["train_params"]["model"]["inputShape"][0])
    centers, logspans = phys_par_range_to_arrays(ppr)
    centers_t = torch.tensor(centers, dtype=torch.float64, device=device)
    logspans_t = torch.tensor(logspans, dtype=torch.float64, device=device)
    linear_t = torch.tensor(phys_par_range_linear_mask(ppr), device=device)
    P = len(par_names)
    print(f"[royv2-l5dt02] unit->phys box from {'voltage_loss' if vl.get('phys_par_range') else 'input_meta'} "
          f"({P} params, linear rows {np.flatnonzero(phys_par_range_linear_mask(ppr)).tolist()})", flush=True)
    use_tanh = not args.noTanh

    with h5py.File(args.packFile, "r") as f:
        meta = json.loads(f["meta.JSON"][0])
        X = f[args.dom + "_volts_norm"][:, :, 0, 0].astype(np.float32)  # (N, T_pack) fixed-z
        raw = f[args.dom + "_raw_volts_mV"][:].astype(np.float32)       # (N, 4000) mV @0.1 ms
        famS = f[args.dom + "_stim_family"][:].astype(str)
        nids = f[args.dom + "_neuron_id"][:].astype(str)
    pack_dt = float((meta.get("timeAxis") or {}).get("step", 0.1))
    if X.shape[1] != T_model:
        raise SystemExit(f"pack has {X.shape[1]} bins at {pack_dt} ms but the model wants {T_model}; "
                         f"use the pack whose grid matches the model")

    mod._DT = float(args.simDt)
    mod._T_MAX = 500.0                     # the 5k CSVs: 5000 pts x 0.1 ms
    JaxleyBridge.clear_cache()
    # recording = last 400 ms of the 500 ms window, on the solver's output grid
    print(f"[royv2-l5dt02] model={args.modelPath} P={P} T_model={T_model} dom={args.dom} N={len(X)} "
          f"tanh={use_tanh} cell={args.cellName} ncomp={os.environ.get('L5TTPC_NCOMP')} "
          f"simDt={args.simDt} stimScale={args.stimScale} stims=Roy<amp>_{args.stimSet}_5k", flush=True)

    unit_by_amp, summary, per_sample, grid, traces_out = {}, [], [], {}, {}
    for amp in AMPS:
        fam = f"Roy{amp}"
        m = famS == fam
        if not m.any():
            continue
        sel = np.flatnonzero(m)
        if args.maxN:
            sel = sel[:args.maxN]
        x = X[sel]
        with torch.no_grad():
            xt = torch.from_numpy(x[:, :, None]).contiguous().to(device)
            pred_unit = model(xt).float().cpu()
        pu = pred_unit.double().to(device)
        if use_tanh:
            pu = torch.tanh(pu)
        pred_phys = unit_to_phys_torch(pu, centers_t, logspans_t, linear_t)
        unit_by_amp[amp] = pu.cpu().numpy()

        stim_name = f"{fam}_{args.stimSet}_5k"
        t0 = time.time()
        v_sim, out_dt = sim_family(pred_phys, args.cellName, stim_name, args.stimScale)  # (N, T_sim)
        k = int(round(out_dt / 0.1))                    # recording samples per sim sample
        i0 = int(round(100.0 / out_dt))                 # sim index of t = 100 ms
        n_al = int(round(400.0 / out_dt))               # 400 ms of samples
        v_al = v_sim[:, i0:i0 + n_al]
        data_raw = raw[sel][:, ::k][:, :n_al]           # recording on the same grid
        T_data = data_raw.shape[1]; v_al = v_al[:, :T_data]
        bad = ~np.isfinite(v_al).all(axis=1)
        if bad.any():
            print(f"[royv2-l5dt02] WARN {fam}: {bad.sum()} non-finite sims dropped from stats")
        ok = ~bad
        v_al_f = np.nan_to_num(v_al, nan=0.0, posinf=0.0, neginf=0.0)
        mse_fixed = ((zfix(v_al_f) - zfix(data_raw)) ** 2).mean(axis=1)
        zs, zd = zscore(v_al_f), zscore(data_raw)
        mse_z = ((zs - zd) ** 2).mean(axis=1)
        dtw_z = np.array([float(soft_dtw_loss(
            torch.tensor(zs[i:i + 1], dtype=torch.float64),
            torch.tensor(zd[i:i + 1], dtype=torch.float64),
            gamma=0.1, n_points=256, band_ms=8.0, dt_ms=out_dt)) for i in range(len(zs))])
        sp_sim, sp_data = spikes_pos(v_al_f), spikes_pos(data_raw)
        grid[amp] = (data_raw, v_al_f, sp_sim, sp_data, mse_z, nids[sel], out_dt)
        # rig current (UNSCALED, nA) on the same window/grid, for scripts/rin_fit.py (R per rig-nA)
        I_rig = load_stim_csv(mod._STIM_DIR / f"{stim_name}.csv").astype(np.float64)
        I_al = np.interp(100.0 + np.arange(T_data) * out_dt, np.arange(len(I_rig)) * 0.1, I_rig)
        traces_out[f"{fam}_data"] = data_raw.astype(np.float32)
        traces_out[f"{fam}_sim"] = v_al_f.astype(np.float32)
        traces_out[f"{fam}_sim_ok"] = ok
        traces_out[f"{fam}_I_rig"] = I_al.astype(np.float32)
        traces_out[f"{fam}_nid"] = nids[sel]
        traces_out["out_dt"] = out_dt
        if not ok.any():
            print(f"[royv2-l5dt02] {fam}: ALL {len(sel)} sims non-finite -- skipped")
            summary.append((amp, len(sel), int(bad.sum()), np.nan, np.nan, np.nan, np.nan,
                            np.nan, float(sp_data.mean())))
            continue
        summary.append((amp, len(sel), int(bad.sum()),
                        float(mse_fixed[ok].mean()), float(np.median(mse_fixed[ok])),
                        float(mse_z[ok].mean()), float(dtw_z[ok].mean()),
                        float(sp_sim[ok].mean()), float(sp_data.mean())))
        for i, nidx in enumerate(sel):
            per_sample.append((nids[nidx], fam, float(mse_fixed[i]), float(mse_z[i]),
                               float(dtw_z[i]), int(sp_sim[i]), int(sp_data[i]), bool(bad[i])))
        print(f"[royv2-l5dt02] {fam}: N={len(sel)} jaxley {time.time()-t0:.1f}s "
              f"MSE_fixed={mse_fixed[ok].mean():.3f} MSE_z={mse_z[ok].mean():.3f} "
              f"DTW_z={dtw_z[ok].mean():.3f} spikes sim/data={sp_sim[ok].mean():.1f}/{sp_data.mean():.1f}",
              flush=True)

        t_axis = np.arange(T_data) * out_dt
        with PdfPages(os.path.join(outDir, f"Roy{amp}_voltage_overlays.pdf")) as pdf:
            for idx in range(min(args.numOverlay, len(sel))):
                fig, axes = plt.subplots(2, 1, figsize=(11, 5.0), sharex=True)
                axes[0].plot(t_axis, data_raw[idx], "k", lw=0.8, label="experimental")
                axes[0].plot(t_axis, v_al_f[idx], "C3", lw=0.8, alpha=0.8,
                             label=f"L5 nc2 sim from predicted params {args.tag}")
                axes[0].set_ylabel("V (mV)"); axes[0].legend(loc="upper right", fontsize=8)
                axes[0].set_title(f"{fam} {nids[sel[idx]]}  MSE_fixed={mse_fixed[idx]:.3f} "
                                  f"MSE_z={mse_z[idx]:.3f} DTW_z={dtw_z[idx]:.3f}  "
                                  f"spikes sim/data={int(sp_sim[idx])}/{int(sp_data[idx])}", fontsize=9)
                axes[1].plot(t_axis, zfix(data_raw[idx:idx+1])[0], "k", lw=0.8)
                axes[1].plot(t_axis, zfix(v_al_f[idx:idx+1])[0], "C3", lw=0.8, alpha=0.8)
                axes[1].set_xlabel("time (ms)"); axes[1].set_ylabel("fixed-z V")
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    np.savez_compressed(os.path.join(outDir, "roy_traces.npz"), stim_scale=args.stimScale, **traces_out)
    with open(os.path.join(outDir, "roy_summary.csv"), "w") as fo:
        fo.write("amp,n,n_bad,mse_fixed_mean,mse_fixed_median,mse_z_mean,dtw_z_mean,spikes_sim,spikes_data\n")
        for row in summary:
            fo.write(",".join(str(r) for r in row) + "\n")
    with open(os.path.join(outDir, "roy_per_sample.csv"), "w") as fo:
        fo.write("neuron_id,family,mse_fixed,mse_z,dtw_z,spikes_sim,spikes_data,sim_nonfinite\n")
        for row in per_sample:
            fo.write(",".join(str(r) for r in row) + "\n")

    # composite grid: rows = first numOverlay traces, cols = amplitudes (raw mV)
    amps_done = [a for a in AMPS if a in grid]
    if amps_done:
        nrow = min(args.numOverlay, min(len(grid[a][0]) for a in amps_done))
        fig, axes = plt.subplots(nrow, len(amps_done), figsize=(4.2 * len(amps_done), 2.3 * nrow),
                                 sharex=True, squeeze=False)
        for c, a in enumerate(amps_done):
            data_raw, v_al_f, sp_sim, sp_data, mse_z, nid, out_dt = grid[a]
            t_axis = np.arange(data_raw.shape[1]) * out_dt
            for r in range(nrow):
                ax = axes[r][c]
                ax.plot(t_axis, data_raw[r], "k", lw=0.6)
                ax.plot(t_axis, v_al_f[r], "C3", lw=0.6, alpha=0.85)
                ax.set_title(f"Roy{a} {nid[r]}  spikes sim/data {int(sp_sim[r])}/{int(sp_data[r])}  "
                             f"mse_z {mse_z[r]:.2f}", fontsize=7)
                ax.tick_params(labelsize=7)
                if c == 0: ax.set_ylabel("mV", fontsize=8)
                if r == nrow - 1: ax.set_xlabel("ms", fontsize=8)
        fig.suptitle(f"Roy v2 recordings (black) vs L5 nc2 re-sim of predicted params (red)  "
                     f"{args.tag}  [{args.dom}, stim {args.stimSet} x{args.stimScale}, dt {args.simDt}]",
                     fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(outDir, "roy_overlay_grid.png"), dpi=110); plt.close(fig)

    tags = [f"Roy{a}" for a in AMPS if a in unit_by_amp]
    cols = 4; rows = int(np.ceil(P / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.0 * rows), squeeze=False)
    for pi in range(P):
        ax = axes[pi // cols][pi % cols]
        vals = [unit_by_amp[a][:, pi] for a in AMPS if a in unit_by_amp]
        ax.boxplot(vals, tick_labels=tags, showfliers=False)
        ax.axhspan(-1, 1, color="green", alpha=0.05); ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(par_names[pi], fontsize=8); ax.grid(alpha=0.3, axis="y")
        ax.tick_params(labelsize=7)
    for j in range(P, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"L5 nc2 unit params on Roy v2 recordings ({args.dom}) {args.tag}; "
                 f"green = trained [-1,1] box ({'tanh-clamped' if use_tanh else 'unclamped'})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outDir, "roy_unit_params.png"), dpi=120)
    print(f"[royv2-l5dt02] wrote {outDir}/roy_summary.csv, roy_per_sample.csv, roy_overlay_grid.png, "
          f"per-family overlays, unit params", flush=True)


if __name__ == "__main__":
    main()
