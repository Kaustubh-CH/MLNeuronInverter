#!/usr/bin/env python
"""Predict the Roy/Paula chaotic recordings (Apr-2026) with a DTW-era CA3 model.

The recordings (exp_data_paula, 26422001-26422025) are 5 sweeps at each of 5
amplitudes of the Roy chaotic stimulus, which is bit-for-bit
4k50kInterChaoticB (= 5k50kInterChaoticB[1000:5000], corr 0.999, lag 0) at
scale amp*0.2508/1000 of the sim waveform, plus a constant holding current of
-0.5995 nA.  Roy2000 therefore delivers HALF the training amplitude.

Per amplitude:
  1. leading-edge-pad the 4000-bin recording to the model's T (the first
     1000 bins of the 5k training stim are exactly zero current, so the model
     window = 100 ms rest + the recorded window; edge value ~ resting V),
  2. CNN -> unit params (tanh) -> physical conductances,
  3. re-simulate with jaxley under the stimulus the rig ACTUALLY delivered
     (Roy<amp>_ica5k.csv = holding + scale * 5k waveform; --simScale train
     uses the unscaled training stimulus instead),
  4. overlay sim[..., -4000:] vs recording (z-scored), score MSE_z + spikes.

  ./plot_exp_overlay_roy.py --modelPath <run>/out --outDir <run>/out/exp_roy
"""
import os, time, argparse, importlib
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

from evaluate_voltage import load_trained_model
from toolbox import JaxleyBridge
from toolbox import jaxley_cells, jaxley_utils as _jutils
from toolbox.jaxley_utils import phys_par_range_to_arrays
from toolbox.soft_dtw import soft_dtw_loss

AMPS = [500, 1000, 1500, 2000]   # Roy100 dropped from the study (user, 2026-08-26)
ROY_DIR_DEF = "/global/homes/k/ktub1999/ExperimentalData/PyForEphys/RoyPaula6"


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--modelPath", default="out")
    p.add_argument("--outDir", default=None, help="default <modelPath>/exp_roy")
    p.add_argument("--royDir", default=ROY_DIR_DEF)
    p.add_argument("--simScale", choices=["rig", "train"], default="rig",
                   help="simulate under the delivered (rig) or training (train) stimulus")
    p.add_argument("--numOverlay", type=int, default=5)
    return p.parse_args()


def zscore(v):
    return (v - v.mean(axis=1, keepdims=True)) / (v.std(axis=1, keepdims=True) + 1e-6)


def spikes_pos(v, thr=0.0):  # upward threshold crossings
    return ((v[:, 1:] > thr) & (v[:, :-1] <= thr)).sum(axis=1)


def main():
    args = get_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, trainMD = load_trained_model(args.modelPath, device)
    outDir = args.outDir or os.path.join(args.modelPath, "exp_roy")
    os.makedirs(outDir, exist_ok=True)

    vl         = trainMD["train_params"]["voltage_loss"]
    cell_name  = vl["cell_name_for_sim"]
    clamp_tanh = bool(vl.get("clamp_unit_tanh", False))
    # Fine-tune runs on exp packs nest the sim pack's meta one level down and
    # may omit parName/num_time_bins at the top level (e.g. royexp_ft_*).
    _im        = trainMD["input_meta"]
    _nested    = _im.get("input_meta") or {}
    par_names  = _im.get("parName") or _nested["parName"]
    T_model    = int(_im.get("num_time_bins") or _nested["num_time_bins"])
    P          = trainMD["train_params"]["model"]["outputSize"]
    phys_par_range = vl.get("phys_par_range")
    if phys_par_range is None:
        from toolbox.HybridLoss import _read_phys_par_range_from_h5
        phys_par_range = _read_phys_par_range_from_h5(trainMD["train_params"]["full_h5name"])
    centers, logspans = phys_par_range_to_arrays(phys_par_range)
    centers_t  = torch.tensor(centers,  dtype=torch.float64, device=device)
    logspans_t = torch.tensor(logspans, dtype=torch.float64, device=device)

    # T_MAX must follow the SIMULATED stim's length, not a fixed 500 ms: a
    # 4000-bin training stim (e.g. 4k50kInterChaoticB) upsampled onto a 500 ms
    # grid gets its last value HELD for 100 ms by np.interp, and the
    # last-T_data alignment below then plots sim spikes ~100 ms early.
    spec = jaxley_cells.get(cell_name)
    mod = importlib.import_module(f"toolbox.jaxley_cells.{cell_name}")

    def set_t_max_for(stim: str):
        n = len(np.loadtxt(os.path.join(str(spec.stim_dir), f"{stim}.csv")))
        t = n * spec.dt_stim                      # 5000 -> 500 ms, 4000 -> 400 ms
        if getattr(mod, "_T_MAX", None) != t:
            mod._T_MAX = t
            JaxleyBridge.clear_cache()

    set_t_max_for("Roy1000_ica5k" if args.simScale == "rig" else vl["stim_name"])
    print(f"[roy] model={args.modelPath} cell={cell_name} T_model={T_model} "
          f"simScale={args.simScale} trainStim={vl.get('stim_name')}")

    unit_by_amp, summary = {}, []
    for amp in AMPS:
        fn = os.path.join(args.royDir, f"Roy{amp}.mlPack1.h5")
        with h5py.File(fn, "r") as f:
            volts = f["test_volts_norm"][:]           # (N, 4000, 1, 1) fixed-norm
            raw   = f["raw_volts_mV"][:]              # (N, 4000) mV
        data = volts[:, :, 0, 0].astype(np.float32)
        N, T_data = data.shape

        x = data
        if T_data < T_model:                          # leading pad: 0-100 ms is rest
            x = np.pad(x, ((0, 0), (T_model - T_data, 0)), mode="edge")
        elif T_data > T_model:
            x = x[:, :T_model]
        with torch.no_grad():
            xt = torch.from_numpy(x[:, :, None]).contiguous().to(device)
            pred_unit = model(xt).float().cpu()
        pu = pred_unit.double().to(device)
        if clamp_tanh:
            pu = torch.tanh(pu)
        pred_phys = centers_t * torch.pow(
            torch.tensor(10.0, dtype=torch.float64, device=device), pu * logspans_t)
        unit_by_amp[amp] = pu.cpu().numpy()
        np.savetxt(os.path.join(outDir, f"Roy{amp}_unit_params.csv"),
                   unit_by_amp[amp], delimiter=",", header=",".join(par_names), comments="")

        stim_name = f"Roy{amp}_ica5k" if args.simScale == "rig" else vl.get("stim_name")
        t0 = time.time()
        v = JaxleyBridge.simulate_batch(pred_phys, cell_name, stim_name)
        v_sim = v[:, 0, :].cpu().numpy()              # (N, T_out) mV
        v_al = v_sim[:, -T_data:]                     # recording = last 400 ms of the 5k window
        v_sim_z, v_data_z = zscore(v_al), zscore(data)
        mse_z  = ((v_sim_z - v_data_z) ** 2).mean(axis=1)
        # timing-tolerant score: soft-DTW on z-scored traces, training-loss
        # hyper-params (gamma 0.1, 256 pts, 8 ms band); does not reward silence
        dtw_z = np.array([float(soft_dtw_loss(
            torch.tensor(v_sim_z[i:i + 1], dtype=torch.float64),
            torch.tensor(v_data_z[i:i + 1], dtype=torch.float64),
            gamma=0.1, n_points=256, band_ms=8.0)) for i in range(N)])
        sp_sim  = spikes_pos(v_al, 0.0)
        sp_data = spikes_pos(raw, 0.0)
        summary.append((amp, N, float(mse_z.mean()), float(dtw_z.mean()),
                        float(sp_sim.mean()), float(sp_data.mean())))
        print(f"[roy] Roy{amp}: N={N} jaxley {time.time()-t0:.1f}s stim={stim_name}  "
              f"MSE_z mean={mse_z.mean():.3f}  DTW_z mean={dtw_z.mean():.3f}  "
              f"spikes sim/data={sp_sim.mean():.1f}/{sp_data.mean():.1f}")

        dt = 0.1; t_axis = np.arange(T_data) * dt
        pdf = PdfPages(os.path.join(outDir, f"Roy{amp}_voltage_overlays.pdf"))
        for idx in range(min(args.numOverlay, N)):
            fig, axes = plt.subplots(2, 1, figsize=(11, 5.0), sharex=True)
            axes[0].plot(t_axis, raw[idx], "k", lw=0.8)
            axes[0].plot(t_axis, v_al[idx], "C3", lw=0.8, alpha=0.8)
            axes[0].set_ylabel("V (mV)")
            axes[0].set_title(f"Roy{amp} sweep #{idx}  MSE_z={mse_z[idx]:.3f} DTW_z={dtw_z[idx]:.3f}  "
                              f"spikes sim/data={int(sp_sim[idx])}/{int(sp_data[idx])}  "
                              f"[sim stim {stim_name}]", fontsize=9)
            axes[1].plot(t_axis, v_data_z[idx], "k",  lw=0.8, label="experimental (z)")
            axes[1].plot(t_axis, v_sim_z[idx],  "C3", lw=0.8, alpha=0.8, label="sim from pred params (z)")
            axes[1].set_xlabel("time (ms)"); axes[1].set_ylabel("z-scored V")
            axes[1].legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)
        pdf.close()
        print(f"[roy] wrote {outDir}/Roy{amp}_voltage_overlays.pdf")

    # Combined unit-parameter figure across amplitudes.
    tags = [f"Roy{a}" for a in AMPS if a in unit_by_amp]
    cols = 3; rows = int(np.ceil(P / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
    for pi in range(P):
        ax = axes[pi // cols][pi % cols]
        vals = [unit_by_amp[a][:, pi] for a in AMPS if a in unit_by_amp]
        ax.boxplot(vals, tick_labels=tags, showfliers=False)
        for xi, vv in enumerate(vals):
            ax.scatter(np.full_like(vv, xi + 1) + np.random.uniform(-.08, .08, len(vv)),
                       vv, s=10, alpha=0.5, color="C0")
        ax.axhspan(-1, 1, color="green", alpha=0.05)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(par_names[pi], fontsize=10); ax.grid(alpha=0.3, axis="y")
        ax.set_ylabel("unit param")
    for j in range(P, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Predicted CA3 unit params on Roy/Paula recordings "
                 "(green = trained [-1,1] range); param consistency across drive = good sign",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outDir, "roy_unit_params.png"), dpi=120)
    print(f"[roy] wrote {outDir}/roy_unit_params.png")

    with open(os.path.join(outDir, "roy_summary.csv"), "w") as f:
        f.write("amp,n_sweeps,mse_z_mean,dtw_z_mean,spikes_sim,spikes_data\n")
        for row in summary:
            f.write(",".join(str(r) for r in row) + "\n")
    print(f"[roy] wrote {outDir}/roy_summary.csv")


if __name__ == "__main__":
    main()
