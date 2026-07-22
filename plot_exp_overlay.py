#!/usr/bin/env python
"""Plot predicted unit-params + voltage overlays for EXPERIMENTAL data.

For each experimental dataset:
  1. run the trained CNN on the (padded-to-model-length) experimental traces
     -> predicted CA3 unit params -> physical conductances,
  2. re-simulate the CA3 cell (jaxley) under the model's fine-tune stimulus,
  3. overlay simulated vs experimental voltage (both z-scored), one page/sweep.
Also writes a combined figure of the predicted unit params across datasets.

Reuses evaluate_voltage.load_trained_model + the same unit->phys / simulate /
z-score machinery, but sources voltage from an experimental H5 instead of the
pack's test split (experimental data has no CA3 ground-truth params).

  ./plot_exp_overlay.py --modelPath out --outDir out/exp_plots
Outputs (in --outDir):
  exp_unit_params.png                 predicted unit params, all datasets
  <tag>_voltage_overlays.pdf          per-sweep sim-vs-experimental overlays
"""
import os, glob, time, argparse, importlib
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

from evaluate_voltage import load_trained_model
from toolbox import JaxleyBridge
from toolbox import jaxley_cells, jaxley_utils as _jutils
from toolbox.jaxley_utils import phys_par_range_to_arrays

# (tag, experimental dir, predictStim index, probe index) — the 4 datasets
# whose trace length (4000) is compatible with the 4k model.
EXP_ROOT = "/global/homes/k/ktub1999/ExperimentalData/PyForEphys"
DATASETS = [
    ("Exact",  f"{EXP_ROOT}/Data_Exact_TotalNorm",          5, 0),
    ("Reduce", f"{EXP_ROOT}/Data_Reduce_TotalNorm",         5, 0),
    ("Step",   f"{EXP_ROOT}/Step500_Data_Reduce_TotalNorm", 5, 0),
    ("Sim",    f"{EXP_ROOT}/Sim_NNrow_TotalNorm",           0, 0),
]


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--modelPath", default="out")
    p.add_argument("--outDir", default=None, help="default <modelPath>/exp_plots")
    p.add_argument("--numOverlay", type=int, default=11, help="max sweeps to overlay/dataset")
    return p.parse_args()


def zscore(v):  # per-trace mean0/std1, axis=time
    return (v - v.mean(axis=1, keepdims=True)) / (v.std(axis=1, keepdims=True) + 1e-6)


def main():
    args = get_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, trainMD = load_trained_model(args.modelPath, device)
    outDir = args.outDir or os.path.join(args.modelPath, "exp_plots")
    os.makedirs(outDir, exist_ok=True)

    vl          = trainMD["train_params"]["voltage_loss"]
    cell_name   = vl["cell_name_for_sim"]
    clamp_tanh  = bool(vl.get("clamp_unit_tanh", False))
    stim_name   = vl.get("stim_name")
    t_max_over  = vl.get("t_max_override")
    par_names   = trainMD["input_meta"]["parName"]
    T_model     = int(trainMD["input_meta"]["num_time_bins"])
    P           = trainMD["train_params"]["model"]["outputSize"]
    phys_par_range = vl.get("phys_par_range")
    if phys_par_range is None:
        from toolbox.HybridLoss import _read_phys_par_range_from_h5
        phys_par_range = _read_phys_par_range_from_h5(trainMD["train_params"]["full_h5name"])
    centers, logspans = phys_par_range_to_arrays(phys_par_range)
    centers_t  = torch.tensor(centers,  dtype=torch.float64, device=device)
    logspans_t = torch.tensor(logspans, dtype=torch.float64, device=device)

    # t_max (auto -> stim length * dt_stim), matching HybridLoss/evaluate_voltage.
    if t_max_over is not None:
        mod = importlib.import_module(f"toolbox.jaxley_cells.{cell_name}")
        if isinstance(t_max_over, str) and t_max_over.lower() in ("auto", "stim"):
            spec = jaxley_cells.get(cell_name)
            sn = stim_name or spec.default_stim_name
            stim_arr = _jutils.load_stim_csv(Path(spec.stim_dir) / f"{sn}.csv")
            t_max_over = float(len(stim_arr)) * float(spec.dt_stim)
        mod._T_MAX = float(t_max_over)
        JaxleyBridge.clear_cache()
        print(f"[exp] cell={cell_name} stim={stim_name} t_max={t_max_over} ms T_model={T_model}")

    unit_by_ds = {}   # tag -> (nsweep, P) predicted unit params

    for tag, ddir, pstim, probe in DATASETS:
        files = glob.glob(os.path.join(ddir, "*.mlPack1.h5"))
        if not files:
            print(f"[exp] {tag}: no h5 in {ddir}; skip"); continue
        with h5py.File(files[0], "r") as f:
            volts = f["test_volts_norm"][:]          # (N, T, probe, stim)
        data = volts[:, :, probe, pstim].astype(np.float32)   # (N, T) experimental, z-normed
        N, T_data = data.shape

        # CNN forward: model wants T_model (=4001, jaxley t_max/dt+1); experimental
        # is 4000 -> edge-pad one leading sample (negligible, within first pool).
        x = data
        if T_data < T_model:
            x = np.pad(x, ((0, 0), (T_model - T_data, 0)), mode="edge")
        elif T_data > T_model:
            x = x[:, :T_model]
        with torch.no_grad():
            xt = torch.from_numpy(x[:, :, None]).contiguous().to(device)   # (N, T, C=1)
            pred_unit = model(xt).float().cpu()
        pu = pred_unit.double().to(device)
        if clamp_tanh:
            pu = torch.tanh(pu)
        pred_phys = centers_t * torch.pow(torch.tensor(10.0, dtype=torch.float64, device=device),
                                          pu * logspans_t)
        unit_by_ds[tag] = pu.cpu().numpy()
        # dump predicted unit params (one row/sweep) for downstream unit->phys.
        np.savetxt(os.path.join(outDir, f"{tag}_unit_params.csv"),
                   unit_by_ds[tag], delimiter=",", header=",".join(par_names), comments="")

        # Simulate under the model's stim, z-score, align to experimental length.
        sim_bs, sim_chunks = 64, []
        t0 = time.time()
        for i in range(0, N, sim_bs):
            v = JaxleyBridge.simulate_batch(pred_phys[i:i + sim_bs], cell_name, stim_name)
            sim_chunks.append(v[:, probe, :].cpu())
        v_sim = torch.cat(sim_chunks, dim=0).numpy()          # (N, T_sim) mV
        # jaxley emits t_max/dt+1 points; drop the leading t=0 sample so the sim
        # aligns with the experimental trace (which starts at the first stim step).
        if v_sim.shape[1] == T_data + 1:
            v_sim = v_sim[:, 1:]
        T = min(v_sim.shape[1], T_data)
        v_sim_z  = zscore(v_sim[:, :T])
        v_data_z = zscore(data[:, :T])
        mse_z = ((v_sim_z - v_data_z) ** 2).mean(axis=1)
        sp_sim  = ((v_sim[:, 1:T] > 0)  & (v_sim[:, :T-1] <= 0)).sum(axis=1)
        sp_data = ((v_data_z[:, 1:] > 2.0) & (v_data_z[:, :-1] <= 2.0)).sum(axis=1)
        print(f"[exp] {tag}: N={N} jaxley {time.time()-t0:.1f}s  "
              f"MSE_z mean={mse_z.mean():.3f}  spikes sim/data={sp_sim.mean():.1f}/{sp_data.mean():.1f}")

        # Per-sweep overlay PDF.
        dt = 0.1; t_axis = np.arange(T) * dt
        pdf = PdfPages(os.path.join(outDir, f"{tag}_voltage_overlays.pdf"))
        for idx in range(min(args.numOverlay, N)):
            fig = plt.figure(figsize=(11, 3.0))
            plt.plot(t_axis, v_data_z[idx], "k",  lw=1.0, label="experimental (z)")
            plt.plot(t_axis, v_sim_z[idx],  "C3", lw=1.0, alpha=0.8, label="sim from pred params (z)")
            plt.xlabel("time (ms)"); plt.ylabel("z-scored V")
            plt.title(f"{tag} sweep #{idx}  MSE_z={mse_z[idx]:.3f}  "
                      f"spikes sim/data={int(sp_sim[idx])}/{int(sp_data[idx])}  "
                      f"[stim {stim_name}]", fontsize=9)
            plt.legend(loc="upper right", fontsize=8); plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)
        pdf.close()
        print(f"[exp] wrote {outDir}/{tag}_voltage_overlays.pdf")

    # Combined unit-parameter figure: one subplot/param, box+strip across datasets.
    if unit_by_ds:
        tags = list(unit_by_ds.keys())
        cols = 3; rows = int(np.ceil(P / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
        for pi in range(P):
            ax = axes[pi // cols][pi % cols]
            vals = [unit_by_ds[t][:, pi] for t in tags]
            ax.boxplot(vals, labels=tags, showfliers=False)
            for xi, v in enumerate(vals):
                ax.scatter(np.full_like(v, xi + 1) + np.random.uniform(-.08, .08, len(v)),
                           v, s=10, alpha=0.5, color="C0")
            ax.axhspan(-1, 1, color="green", alpha=0.05)   # in-range band
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_title(par_names[pi], fontsize=10); ax.grid(alpha=0.3, axis="y")
            ax.set_ylabel("unit param")
        for j in range(P, rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.suptitle("Predicted CA3 unit parameters on experimental data (green band = trained [-1,1] range)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(outDir, "exp_unit_params.png"), dpi=120)
        print(f"[exp] wrote {outDir}/exp_unit_params.png")


if __name__ == "__main__":
    main()
