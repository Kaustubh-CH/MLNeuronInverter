#!/usr/bin/env python3
"""Per-channel recovery metrics that CREDIT a good-but-offset diagonal, plus a
per-channel AFFINE RECALIBRATION fit on validation and applied at test.

MOTIVATION
----------
Raw R^2 = 1 - SS_res/SS_tot penalizes any systematic slope/bias even when the
scatter is a TIGHT, well-ranked diagonal.  The A3 dV/dt result is exactly this:
na3 became a tight diagonal with a ~-0.40 unit bias offset, so its correlation /
rank were good but R^2 scored -0.78 -- the metric MASKED a real win.  These
functions report the offset-invariant quantities (Pearson r, Spearman rho, OLS
slope+intercept, R^2 after removing a fitted affine map) so the win is visible,
and produce a recalibration that turns the offset diagonal into a usable
prediction.

Everything runs on the UNIT-space arrays predict.py already saves
(`the_data.npz`: actual_val = trueU (B,P), predicted_val = recoU (B,P)).

HONEST RECALIBRATION
--------------------
The affine map (a_p, b_p) per channel is fit on the VALIDATION split and applied
to the TEST split, so the reported recalibrated R^2 is not fit on its own test
labels.  recoU_recal[:,p] = a_p * recoU[:,p] + b_p, with (a_p,b_p) the least
squares fit of trueU_val ~ a*recoU_val + b.  For a tight diagonal this collapses
the offset and the recalibrated R^2 approaches the squared Pearson r ceiling.

CLI
---
  # 1) fit on validation, 2) apply+score on test, in one call:
  python -m toolbox.recal_metrics --valid <run>/out/the_data_valid.npz \
         --test <run>/out/the_data.npz --parName sum_train.yaml -o <run>/out
"""

import numpy as np


# --------------------------------------------------------------------------- #
# offset-invariant per-channel metrics
# --------------------------------------------------------------------------- #
def _rankdata(x):
    """Average-rank transform (Spearman helper) without scipy."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-30
    return 1.0 - ss_res / ss_tot


def channel_metrics(true_col, pred_col):
    """All per-channel scalars for one parameter column.

    Returns dict with:
      r        Pearson correlation (offset/scale invariant)
      spearman rank correlation (monotonic, invariant to any monotone map)
      slope    OLS slope of true ~ a*pred + b (1.0 = calibrated; <1 = shrinkage)
      intercept
      bias     mean(pred - true)  (the systematic offset R^2 punishes)
      r2       raw R^2 (what is reported today)
      r2_debias  R^2 after removing only the bias:  true vs (pred - bias)
      r2_affine  R^2 after the fitted affine map applied to pred (== r**2 in-sample;
                 the achievable ceiling once slope+bias are corrected)
    """
    t = np.asarray(true_col, float); p = np.asarray(pred_col, float)
    m = np.isfinite(t) & np.isfinite(p)
    t, p = t[m], p[m]
    out = dict(n=int(t.size))
    if t.size < 3 or p.std() < 1e-12:
        return {**out, "r": np.nan, "spearman": np.nan, "slope": np.nan,
                "intercept": np.nan, "bias": float(np.mean(p - t)) if t.size else np.nan,
                "r2": _r2(t, p) if t.size else np.nan,
                "r2_debias": np.nan, "r2_affine": np.nan}
    r = float(np.corrcoef(t, p)[0, 1])
    rho = float(np.corrcoef(_rankdata(t), _rankdata(p))[0, 1])
    a, b = np.polyfit(p, t, 1)                       # true ~ a*pred + b
    bias = float(np.mean(p - t))
    return {**out, "r": r, "spearman": rho, "slope": float(a), "intercept": float(b),
            "bias": bias, "r2": _r2(t, p),
            "r2_debias": _r2(t, p - bias),
            "r2_affine": _r2(t, a * p + b)}


# --------------------------------------------------------------------------- #
# affine recalibration: FIT on valid, APPLY on test
# --------------------------------------------------------------------------- #
def fit_affine(true_val, pred_val):
    """Per-channel (a_p, b_p) least-squares fit of trueU_val ~ a*predU_val + b.
    Inputs (B,P). Returns a (P,), b (P,)."""
    P = pred_val.shape[1]
    a = np.ones(P); b = np.zeros(P)
    for p in range(P):
        t = true_val[:, p]; q = pred_val[:, p]
        m = np.isfinite(t) & np.isfinite(q)
        if m.sum() >= 3 and q[m].std() > 1e-12:
            a[p], b[p] = np.polyfit(q[m], t[m], 1)
    return a, b


def apply_affine(pred, a, b):
    return pred * a[None, :] + b[None, :]


def metrics_table(trueU, recoU, parName, recoU_recal=None):
    """List of per-channel metric dicts (raw + optional recalibrated), keyed by name."""
    rows = []
    P = trueU.shape[1]
    for p in range(P):
        name = parName[p] if p < len(parName) else f"par{p}"
        row = {"param": name, **channel_metrics(trueU[:, p], recoU[:, p])}
        if recoU_recal is not None:
            rc = channel_metrics(trueU[:, p], recoU_recal[:, p])
            row["r2_recal"] = rc["r2"]        # honest: valid-fit map on test labels
            row["slope_recal"] = rc["slope"]
            row["bias_recal"] = rc["bias"]
        rows.append(row)
    return rows


def print_table(rows):
    cols = ["param", "n", "r", "spearman", "slope", "bias", "r2",
            "r2_debias", "r2_affine"]
    if "r2_recal" in rows[0]:
        cols += ["r2_recal"]
    w = {"param": 20}
    print("".join(c.rjust(w.get(c, 10)) for c in cols))
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c, np.nan)
            cells.append((v[:20].ljust(20) if c == "param" else
                          (f"{v:>10d}" if c == "n" else f"{v:>10.3f}")))
        print("".join(cells))


# --------------------------------------------------------------------------- #
def _load_npz(path):
    z = np.load(path)
    # predict.py saves actual_val / predicted_val; accept a few aliases.
    t = z["actual_val"] if "actual_val" in z else z["trueU"]
    p = z["predicted_val"] if "predicted_val" in z else z["recoU"]
    return np.asarray(t, float), np.asarray(p, float)


def main():
    import argparse, os
    from toolbox.Util_IOfunc import read_yaml, write_yaml
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True, help="test the_data.npz")
    ap.add_argument("--valid", default=None,
                    help="valid the_data.npz to FIT the affine recal (honest split)")
    ap.add_argument("--parName", default=None,
                    help="sum_train.yaml to read input_meta.parName (else par0..)")
    ap.add_argument("-o", "--outDir", default=".")
    args = ap.parse_args()

    trueU, recoU = _load_npz(args.test)
    P = trueU.shape[1]
    parName = [f"par{p}" for p in range(P)]
    if args.parName:
        md = read_yaml(args.parName, verb=0)
        parName = (md.get("input_meta", {}).get("parName") or parName)[:P]

    recoU_recal = None
    coeffs = None
    if args.valid:
        tv, pv = _load_npz(args.valid)
        a, b = fit_affine(tv, pv)
        recoU_recal = apply_affine(recoU, a, b)
        coeffs = {parName[p]: {"a": float(a[p]), "b": float(b[p])} for p in range(P)}

    rows = metrics_table(trueU, recoU, parName, recoU_recal)
    print_table(rows)

    os.makedirs(args.outDir, exist_ok=True)
    write_yaml({"metrics": rows, "affine_recal": coeffs},
               os.path.join(args.outDir, "recal_metrics.yaml"))
    if recoU_recal is not None:
        np.savez(os.path.join(args.outDir, "the_data_recal.npz"),
                 actual_val=trueU, predicted_val=recoU_recal)
    print(f"[recal] wrote recal_metrics.yaml (+ the_data_recal.npz) to {args.outDir}/")


if __name__ == "__main__":
    main()
