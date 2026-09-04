#!/usr/bin/env python3
"""Step-0 bias audit for the voltage-only training objective.

WHY: on the CA3 (or any jaxley) cell, supervised param-MSE recovers the
conductances (R^2 ~ 0.99) but every voltage-space objective plateaus at
R^2 ~ 0.5, and voltage fine-tuning *degrades* the supervised solution.  On
self-consistent synthetic data (the simulator IS the generator) that can only
mean one of two things:

  (a) the voltage-loss minimum does NOT sit at the true theta  -> a *biased*
      objective (normalization / time-window / stim / tanh mismatch), OR
  (b) the minimum is at true theta but the surface is *cliffy* and SGD cannot
      descend it.

These need opposite fixes, so we test which one is true BEFORE changing the
optimizer or the loss.  This tool measures, using the EXACT training criterion
(`toolbox.HybridLoss` built from the run's `sum_train.yaml`):

  1. Residual loss at the TRUE parameters.  On self-consistent data this should
     be ~0 (up to solver/dt).  A large value == the objective is biased: the
     candidate trace at true theta does not match the stored data trace, so the
     global minimum is somewhere else.  This is the single most decisive number.

  2. Per-parameter offset sweep: hold every sample at its true theta, sweep an
     offset delta added to one parameter, and plot mean loss vs delta.  The
     minimum should sit at delta=0.  A minimum offset from 0 (or a flat / ragged
     valley) for a given channel localizes the bias / non-smoothness to that
     channel (expect gbar_na3, gkdrbar_kdr to be the ragged ones).

  3. tanh representational bias: if the run trained with `clamp_unit_tanh=True`
     but the data was generated WITHOUT tanh, feeding true theta through the
     criterion's tanh lands on the wrong physical conductance, so loss(true)
     with tanh >> loss(true) without tanh.  We report both.

  4. (optional) the trained CNN's own prediction loss + parameter recovery, to
     compare loss(theta_pred) against loss(theta_true).

Usage:
  python scripts/voltage_loss_bias_probe.py --modelPath <voltage_run>/out \
         [--numSamples 16] [--grid 21] [--span 1.0] [--split test] [--no-model]

The --modelPath run must have a `voltage_loss` block in its sum_train.yaml
(i.e. it was a voltage / HybridLoss run).  You can point it at a voltage run
even when auditing why a *different* (supervised) model degrades under voltage
fine-tuning; the objective being audited is defined by this config.
"""

import os, sys, time, argparse, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# JAX env must be set BEFORE any jax / jaxley import (mirrors evaluate_voltage.py).
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda" if os.environ.get("SLURM_JOBID") else "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox.HybridLoss import build_hybrid_loss
from toolbox.jaxley_utils import phys_par_range_to_arrays


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--modelPath", required=True,
                   help="a voltage run's out/ (sum_train.yaml must have a voltage_loss block)")
    p.add_argument("-n", "--numSamples", type=int, default=16,
                   help="how many split samples to average the loss over (a batch)")
    p.add_argument("--grid", type=int, default=21,
                   help="number of offset points per parameter sweep")
    p.add_argument("--span", type=float, default=1.0,
                   help="sweep offset delta over [-span, +span] in unit space")
    p.add_argument("--split", default="test", choices=["test", "valid", "train"],
                   help="which pack split to read")
    p.add_argument("--no-model", action="store_true",
                   help="skip loading the CNN (only measures loss at true theta + sweeps)")
    p.add_argument("--fp64", dest="fp64", action="store_true", default=None,
                   help="force the in-loop jaxley solve to fp64, overriding the run's "
                        "voltage_loss.fp64.  Use with --no-fp64 to A/B the solver precision: "
                        "on self-consistent data (fp64-generated) the fp32 loss at TRUE theta "
                        "is the solver-mismatch FLOOR the objective can never get below.")
    p.add_argument("--no-fp64", dest="fp64", action="store_false",
                   help="force the in-loop jaxley solve to fp32 (see --fp64).")
    p.add_argument("-o", "--outDir", default=None,
                   help="output dir (default: <modelPath>/bias_probe)")
    return p.parse_args()


def load_split_volts(trainMD, split, n):
    """Read `n` samples of (B, T, C) fixed-z voltages + (B, P) true unit params,
    reshaped exactly like Dataloader_H5's (serialize/valid) branch so the channel
    order matches what HybridLoss._voltage_loss indexes (`true_volts[..., ci]`).
    """
    tp    = trainMD["train_params"]
    h5    = tp["full_h5name"]
    dcf   = tp["data_conf"]
    probs = list(dcf["probs_select"])
    stims = list(dcf["stims_select"])
    with h5py.File(h5, "r") as f:
        # stored (N, T, mxProb, mxStim) fp16 -> select probes then stims
        v = f[f"{split}_volts_norm"][:n].astype(np.float32)      # (n, T, mxProb, mxStim)
        v = v[:, :, probs, :][:, :, :, stims]                    # (n, T, nProb, nStim)
        true_unit = f[f"{split}_unit_par"][:n].astype(np.float32)  # (n, P)
    n_, T, nProb, nStim = v.shape
    # probe-major, stim-inner -> (n, T, nProb*nStim); for CA3 (1 probe) == per-stim channels
    true_volts = v.reshape(n_, T, nProb * nStim)
    return true_volts, true_unit, (nProb, nStim)


@torch.no_grad()
def loss_at(crit, pred_unit, true_volts):
    """Mean voltage-loss (the training voltage term) for a (B, P) candidate."""
    return float(crit._voltage_loss(pred_unit, true_volts).detach().cpu())


def main():
    args = get_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    trainMD = read_yaml(os.path.join(args.modelPath, "sum_train.yaml"), verb=0)
    tp = trainMD["train_params"]
    vl = tp.get("voltage_loss")
    if not tp.get("use_voltage_loss") or vl is None:
        sys.exit("[bias] this run has no voltage_loss block; point --modelPath at a voltage run.")

    # Solver-precision override.  Mutates the config BEFORE build_hybrid_loss so
    # the criterion is built exactly as training would have built it at that
    # precision -- nothing else about the run changes.
    if args.fp64 is not None:
        was = bool(vl.get("fp64", False))
        vl["fp64"] = bool(args.fp64)
        print(f"[bias] fp64 OVERRIDE: run trained with fp64={was}, "
              f"probing with fp64={vl['fp64']}")

    clamp_tanh = bool(vl.get("clamp_unit_tanh", False))
    cell_name  = vl["cell_name_for_sim"]
    print(f"[bias] cell={cell_name} clamp_unit_tanh={clamp_tanh} "
          f"mse_w={vl.get('mse_weight',1.0)} efel_w={vl.get('efel_weight',0.0)} "
          f"stim={vl.get('stim_name')} multi={vl.get('stim_names_multi')}", flush=True)

    # Build the EXACT training criterion, plus a tanh-off twin for the physical
    # coordinate sweep (so delta=0 reproduces the true physical conductance).
    crit = build_hybrid_loss(tp).to(device).eval()
    crit_notanh = copy.deepcopy(crit)
    crit_notanh.clamp_unit_tanh = False

    P = int(crit._centers.numel())
    try:
        import importlib
        param_names = list(importlib.import_module(
            f"toolbox.jaxley_cells.{cell_name}").PARAM_KEYS)[:P]
    except Exception:
        param_names = [f"p{p}" for p in range(P)]

    true_volts_np, true_unit_np, (nProb, nStim) = load_split_volts(trainMD, args.split, args.numSamples)
    true_volts = torch.tensor(true_volts_np, device=device)
    true_unit  = torch.tensor(true_unit_np[:, :P], device=device, dtype=torch.float32)
    B = true_unit.shape[0]
    print(f"[bias] split={args.split} B={B} true_volts={tuple(true_volts.shape)} "
          f"nProb={nProb} nStim={nStim} P={P}", flush=True)

    # ── Metric 1: residual loss at the true parameters ─────────────────────────
    # tanh-off: delta=0 lands exactly on the true physical conductance.
    L_true_phys = loss_at(crit_notanh, true_unit, true_volts)
    # with the run's actual tanh setting (what the network is really optimizing):
    L_true_cfg  = loss_at(crit, true_unit, true_volts)
    print(f"\n[bias] ===== residual at TRUE theta =====")
    print(f"[bias] loss(true, physical/no-tanh) = {L_true_phys:.6f}   "
          f"<-- should be ~0 on self-consistent data; large == BIASED objective")
    if clamp_tanh:
        print(f"[bias] loss(true, config tanh)      = {L_true_cfg:.6f}   "
              f"<-- if >> above, tanh cannot represent the true params (tanh bias)")

    # ── Metric 2: per-parameter offset sweep (tanh-off, physical coordinate) ───
    deltas = np.linspace(-args.span, args.span, args.grid)
    curves = np.full((P, args.grid), np.nan)
    argmin_delta = np.zeros(P)
    print(f"\n[bias] ===== per-parameter offset sweep ({args.grid} pts, +/-{args.span}) =====")
    for p in range(P):
        for gi, d in enumerate(deltas):
            cand = true_unit.clone()
            cand[:, p] = cand[:, p] + float(d)
            try:
                curves[p, gi] = loss_at(crit_notanh, cand, true_volts)
            except Exception as e:
                curves[p, gi] = np.nan  # sim NaN'd out of range; leave as gap
        col = curves[p]
        if np.all(np.isnan(col)):
            print(f"[bias]   {param_names[p]:<18} all-NaN sweep")
            continue
        argmin_delta[p] = deltas[int(np.nanargmin(col))]
        flag = "" if abs(argmin_delta[p]) <= (deltas[1] - deltas[0]) else "  <-- MIN OFF-TRUTH"
        print(f"[bias]   {param_names[p]:<18} argmin delta = {argmin_delta[p]:+.3f}"
              f"   loss(min)={np.nanmin(col):.4f} loss(0)={col[args.grid//2]:.4f}{flag}")

    # ── Metric 3: the trained CNN's own prediction ─────────────────────────────
    pred_summary = {}
    if not args.no_model:
        try:
            from evaluate_voltage import load_trained_model
            model, _ = load_trained_model(args.modelPath, device)
            imgs = true_volts.to(next(model.parameters()).dtype)
            with torch.no_grad():
                try:
                    pred_unit = model(imgs)
                except Exception:
                    pred_unit = model(imgs.permute(0, 2, 1))  # (B,C,T) fallback
            pred_unit = pred_unit[:, :P].float()
            L_pred = loss_at(crit, pred_unit, true_volts)
            eff_pred = torch.tanh(pred_unit) if clamp_tanh else pred_unit
            param_mse = float(((eff_pred - true_unit) ** 2).mean().cpu())
            pred_summary = {"loss_pred": L_pred, "loss_true_phys": L_true_phys,
                            "pred_below_truth": bool(L_pred < L_true_phys),
                            "param_mse_eff": param_mse}
            print(f"\n[bias] ===== CNN prediction =====")
            print(f"[bias] loss(theta_pred, config) = {L_pred:.6f}  vs loss(true,phys) = {L_true_phys:.6f}")
            print(f"[bias] loss(pred) < loss(true) ? {L_pred < L_true_phys}   "
                  f"(True => optimizer sits at a lower point than truth = biased/cliffy min)")
            print(f"[bias] effective-param MSE(pred, true) = {param_mse:.4f}")
        except Exception as e:
            print(f"[bias] (CNN prediction skipped: {e})")

    # ── Verdict heuristic ──────────────────────────────────────────────────────
    dgrid = deltas[1] - deltas[0]
    off_truth = [param_names[p] for p in range(P) if abs(argmin_delta[p]) > dgrid]
    print(f"\n[bias] ===== VERDICT =====")
    if L_true_phys > 0.1:
        print(f"[bias] BIASED: loss at true theta = {L_true_phys:.4f} is not ~0 -> the sim at "
              f"true params does NOT match the stored data. Check VOLT_NORM constants vs the "
              f"pack's normalization, sim_t_skip_bins/time window, and stim_name alignment "
              f"BEFORE any optimizer/loss change.")
    elif off_truth:
        print(f"[bias] PARTIALLY BIASED: loss valley minimum is off-truth for {off_truth} -> "
              f"the objective does not reward the true value for these channels.")
    else:
        print(f"[bias] UNBIASED: minimum sits at true theta for all channels (loss(true)"
              f"={L_true_phys:.4f}). The plateau is a CLIFFY-LANDSCAPE / optimization problem "
              f"-> proceed to soft-DTW (L1) + randomized smoothing (O1).")
    if clamp_tanh and L_true_cfg > 5 * max(L_true_phys, 1e-6):
        print(f"[bias] NOTE: clamp_unit_tanh inflates loss at truth "
              f"({L_true_cfg:.4f} vs {L_true_phys:.4f}) -> tanh cannot represent boundary "
              f"params; the network must predict atanh(theta), a representational bias.")

    # ── Save curves + summary ──────────────────────────────────────────────────
    outDir = args.outDir or os.path.join(args.modelPath, "bias_probe")
    os.makedirs(outDir, exist_ok=True)
    ncol = min(3, P)
    nrow = int(np.ceil(P / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for p in range(P):
        ax = axes[p // ncol][p % ncol]
        ax.plot(deltas, curves[p], "-o", ms=3)
        ax.axvline(0.0, color="k", lw=0.8, ls="--", label="true")
        ax.axvline(argmin_delta[p], color="r", lw=0.8, ls=":", label="argmin")
        ax.set_title(param_names[p]); ax.set_xlabel("offset from true (unit)")
        ax.set_ylabel("voltage loss"); ax.legend(fontsize=7)
    for p in range(P, nrow * ncol):
        axes[p // ncol][p % ncol].axis("off")
    fig.suptitle(f"Voltage-loss bias sweep — {cell_name} ({args.split}, B={B})")
    fig.tight_layout()
    figP = os.path.join(outDir, "bias_sweep.png")
    fig.savefig(figP, dpi=130); plt.close(fig)

    summary = {
        "modelPath": args.modelPath, "cell_name": cell_name, "split": args.split,
        "B": B, "P": P, "clamp_unit_tanh": clamp_tanh,
        "loss_true_physical": L_true_phys, "loss_true_config_tanh": L_true_cfg,
        "argmin_delta_per_param": {param_names[p]: float(argmin_delta[p]) for p in range(P)},
        "min_off_truth_params": off_truth,
        **pred_summary,
    }
    write_yaml(summary, os.path.join(outDir, "bias_summary.yaml"))
    print(f"\n[bias] wrote {figP}")
    print(f"[bias] wrote {os.path.join(outDir, 'bias_summary.yaml')}")
    print(f"[bias] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
