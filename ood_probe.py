#!/usr/bin/env python
"""Out-of-range (extrapolation) probe for the best chaoticRamp CA3 model.

For each of the 6 CA3 conductances, sweep its TRUE value from unit -1.5 -> +1.5
(others held at center=0), crossing the trained [-1,1] boundary. Simulate the
CA3 cell under 5kChaoticRamp, normalize with the SAME fixed-scale constants used
in training, run the CNN, and plot predicted vs true. Shows what the model does
when the underlying ion-channel params fall OUTSIDE the range it was trained on.

  ./ood_probe.py --modelPath <best_chaoticRamp>/out --outDir <out>/ood
"""
import os, argparse, time
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_voltage import load_trained_model
from toolbox import JaxleyBridge, jaxley_cells, jaxley_utils as _jutils
from toolbox.jaxley_utils import phys_par_range_to_arrays, normalize_volts_fixed
from pathlib import Path
import importlib


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--modelPath", required=True)
    p.add_argument("--outDir", default=None)
    p.add_argument("--nsweep", type=int, default=31, help="points per param sweep")
    p.add_argument("--lo", type=float, default=-1.5)
    p.add_argument("--hi", type=float, default=1.5)
    p.add_argument("--mode", choices=["sweep", "joint"], default="sweep",
                   help="sweep: one param at a time (others=center); "
                        "joint: all 6 drawn ~U(lo,hi) together (realistic OOD)")
    p.add_argument("--njoint", type=int, default=400, help="samples for --mode joint")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = get_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, trainMD = load_trained_model(args.modelPath, device)
    outDir = args.outDir or os.path.join(args.modelPath, "ood")
    os.makedirs(outDir, exist_ok=True)

    vl         = trainMD["train_params"]["voltage_loss"]
    cell_name  = vl["cell_name_for_sim"]
    clamp_tanh = bool(vl.get("clamp_unit_tanh", False))
    stim_name  = vl.get("stim_name")
    t_max_over = vl.get("t_max_override")
    par_names  = trainMD["input_meta"]["parName"]
    T_model    = int(trainMD["input_meta"]["num_time_bins"])
    P          = len(par_names)

    ppr = vl.get("phys_par_range")
    if ppr is None:
        from toolbox.HybridLoss import _read_phys_par_range_from_h5
        ppr = _read_phys_par_range_from_h5(trainMD["train_params"]["full_h5name"])
    centers, logspans = phys_par_range_to_arrays(ppr)
    centers_t  = torch.tensor(centers,  dtype=torch.float64, device=device)
    logspans_t = torch.tensor(logspans, dtype=torch.float64, device=device)

    # t_max auto -> stim length * dt_stim
    mod = importlib.import_module(f"toolbox.jaxley_cells.{cell_name}")
    if isinstance(t_max_over, str) and t_max_over.lower() in ("auto", "stim"):
        spec = jaxley_cells.get(cell_name)
        sn = stim_name or spec.default_stim_name
        stim_arr = _jutils.load_stim_csv(Path(spec.stim_dir) / f"{sn}.csv")
        t_max_over = float(len(stim_arr)) * float(spec.dt_stim)
    mod._T_MAX = float(t_max_over)
    JaxleyBridge.clear_cache()
    print(f"[ood] cell={cell_name} stim={stim_name} t_max={t_max_over} T_model={T_model} "
          f"params={P}  norm=fixed(mean={_jutils.VOLT_NORM_MEAN:.3f},std={_jutils.VOLT_NORM_STD:.3f})")

    # Build param sets. sweep: one param over [lo,hi], others=center. joint:
    # all 6 drawn ~U(lo,hi) together (every sample is jointly OOD).
    if args.mode == "sweep":
        sweep = np.linspace(args.lo, args.hi, args.nsweep)
        units = []
        for i in range(P):
            for u in sweep:
                v = np.zeros(P); v[i] = u; units.append(v)
        units = np.array(units)                               # (P*nsweep, P)
    else:
        units = np.random.default_rng(args.seed).uniform(
            args.lo, args.hi, size=(args.njoint, P))          # (njoint, P)
    true_unit_t = torch.tensor(units, dtype=torch.float64, device=device)
    pred_phys = centers_t * torch.pow(torch.tensor(10.0, dtype=torch.float64, device=device),
                                      true_unit_t * logspans_t)

    # Simulate under the training stim.
    N = units.shape[0]; sim_bs = 64; chunks = []
    t0 = time.time()
    for k in range(0, N, sim_bs):
        v = JaxleyBridge.simulate_batch(pred_phys[k:k+sim_bs], cell_name, stim_name)
        chunks.append(v[:, 0, :].cpu())
    v_raw = torch.cat(chunks, dim=0).numpy()                  # (N, T_sim) mV
    nan_mask = ~np.isfinite(v_raw).all(axis=1)
    print(f"[ood] jaxley {time.time()-t0:.1f}s  N={N}  NaN/unstable sims={int(nan_mask.sum())}")

    # Normalize EXACTLY as training, align length, CNN forward.
    v_norm = normalize_volts_fixed(np.nan_to_num(v_raw, nan=0.0)).astype(np.float32)
    T = min(v_norm.shape[1], T_model)
    if v_norm.shape[1] < T_model:
        v_norm = np.pad(v_norm, ((0, 0), (T_model - v_norm.shape[1], 0)), mode="edge")
    else:
        v_norm = v_norm[:, :T_model]
    with torch.no_grad():
        x = torch.from_numpy(v_norm[:, :, None]).contiguous().to(device)
        raw_out = model(x).float().cpu().numpy()              # (N, P) pre-tanh
    post = np.tanh(raw_out) if clamp_tanh else raw_out         # physical unit space [-1,1]

    # Plot per-param: pred vs true, marking trained [-1,1].
    cols = 3; rows = int(np.ceil(P / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0*cols, 3.8*rows), squeeze=False)
    in_err, ood_err = [], []
    for i in range(P):
        ax = axes[i//cols][i%cols]
        if args.mode == "sweep":
            sl = slice(i*args.nsweep, (i+1)*args.nsweep)      # rows where param i swept
            tu = units[sl, i]; pp = post[sl, i]; rr = raw_out[sl, i]; nm = nan_mask[sl]
        else:
            tu = units[:, i]; pp = post[:, i]; rr = raw_out[:, i]; nm = nan_mask
        ax.axvspan(-1, 1, color="green", alpha=0.07, label="trained range")
        ax.plot([args.lo, args.hi], [args.lo, args.hi], "k--", lw=1, alpha=0.5, label="identity")
        ax.axhline(1, color="grey", lw=0.6, ls=":"); ax.axhline(-1, color="grey", lw=0.6, ls=":")
        if args.mode == "sweep":
            ax.plot(tu[~nm], pp[~nm], "o-", color="C0", ms=4, label="pred (post-tanh)")
            ax.plot(tu[~nm], rr[~nm], ".", color="C3", ms=3, alpha=0.5, label="pred (pre-tanh)")
        else:
            oor_pt = np.abs(tu) > 1
            ax.scatter(tu[~nm & ~oor_pt], pp[~nm & ~oor_pt], s=8, alpha=0.4, color="C0", label="in-range true")
            ax.scatter(tu[~nm &  oor_pt], pp[~nm &  oor_pt], s=8, alpha=0.4, color="C3", label="OOD true")
        if nm.any(): ax.plot(tu[nm], np.zeros(nm.sum()), "x", color="grey", ms=5, label="NaN sim")
        ax.set_title(par_names[i], fontsize=10); ax.set_xlabel("true unit"); ax.set_ylabel("pred unit")
        yv = rr[~nm] if (~nm).any() else np.array([args.lo, args.hi])
        ax.set_ylim(min(args.lo, yv.min())-0.2, max(args.hi, yv.max())+0.2)
        ax.grid(alpha=0.3)
        inr = (np.abs(tu) <= 1) & ~nm; oor = (np.abs(tu) > 1) & ~nm
        if inr.any(): in_err.append(np.abs(pp[inr]-tu[inr]).mean())
        if oor.any(): ood_err.append(np.abs(pp[oor]-np.clip(tu[oor],-1,1)).mean())  # vs best-possible (saturated)
        if i == 0: ax.legend(fontsize=7, loc="upper left")
    for j in range(P, rows*cols): axes[j//cols][j%cols].axis("off")
    ttl = ("sweep one param, others=center" if args.mode == "sweep"
           else f"all 6 params jointly ~U({args.lo},{args.hi}), N={args.njoint}")
    fig.suptitle(f"Out-of-range probe: CA3 chaoticRamp model, pred vs true ({ttl})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outPng = os.path.join(outDir, f"ood_{args.mode}.png")
    fig.savefig(outPng, dpi=120)
    print(f"[ood] mode={args.mode}  in-range mean|pred-true| = {np.mean(in_err):.3f}  "
          f"OOD mean|pred-clip(true)| = {np.mean(ood_err):.3f}")
    print(f"[ood] wrote {outPng}")


if __name__ == "__main__":
    main()
