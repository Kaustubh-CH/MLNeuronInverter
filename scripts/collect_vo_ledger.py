#!/usr/bin/env python3
"""Collect CA3 voltage-only run metrics into one comparison CSV (the "vo ledger").

Read-only over each run's ``out/eval/summary.yaml`` (produced by
``evaluate_voltage.py``, which applies the tanh clamp so its per-channel R2 is the
authoritative headline). Does NOT re-simulate or re-derive metrics.

Usage
-----
    python scripts/collect_vo_ledger.py RUN_DIR [RUN_DIR ...] [-o results.csv]

Each RUN_DIR may be:
  * a run dir containing ``out/eval/summary.yaml``   (jaxley_ca3 / ablation layout)
  * a dir containing ``eval/summary.yaml``           (an ``out/`` dir passed directly)
  * a path straight to a ``summary.yaml``

One row per run keyed by ``design``; re-running merges/updates rows in the CSV so the
ledger accumulates across sessions. Columns (per-channel R2 in CA3 PARAM_KEYS order):

    design, mean_r2, r2_leak, r2_na3, r2_kdr, r2_kap, r2_km, r2_kd,
    voltage_mse_z_mean, spike_diff_abs, n_samples, summary_path
"""
import argparse
import csv
import os
import sys

import yaml

# CA3 channel name (as written in summary.yaml) -> short ledger column.
CHAN_MAP = {
    "CA3_g_leak":     "r2_leak",
    "CA3_gbar_na3":   "r2_na3",
    "CA3_gkdrbar_kdr": "r2_kdr",
    "CA3_gkabar_kap": "r2_kap",
    "CA3_gbar_km":    "r2_km",
    "CA3_gkdbar_kd":  "r2_kd",
}
CHAN_COLS = ["r2_leak", "r2_na3", "r2_kdr", "r2_kap", "r2_km", "r2_kd"]
FIELDS = (["design", "mean_r2"] + CHAN_COLS +
          ["voltage_mse_z_mean", "spike_diff_abs", "n_samples", "summary_path"])

DEFAULT_OUT = os.path.join(
    os.environ.get("SCRATCH", "."), "tmp_neuInv", "jaxley_ca3", "vo_ledger",
    "results.csv")


def resolve_summary(path):
    """Return the summary.yaml path for a run dir/out dir/summary path, or None."""
    if os.path.isfile(path) and path.endswith(".yaml"):
        return path
    for cand in (os.path.join(path, "out", "eval", "summary.yaml"),
                 os.path.join(path, "eval", "summary.yaml")):
        if os.path.isfile(cand):
            return cand
    return None


def design_name(run_arg, summary):
    """Prefer the run-dir basename; fall back to the summary's model_path."""
    base = os.path.basename(os.path.normpath(run_arg))
    if base in ("out", "eval") or base.endswith(".yaml"):
        mp = summary.get("model_path", "")
        # model_path like .../<design>/out -> take the <design> component
        parts = [p for p in os.path.normpath(mp).split(os.sep) if p]
        if "out" in parts:
            i = parts.index("out")
            if i > 0:
                return parts[i - 1]
        return parts[-1] if parts else base
    return base


def row_from_summary(run_arg, spath):
    with open(spath) as fh:
        s = yaml.safe_load(fh)
    row = {c: "" for c in FIELDS}
    row["design"] = design_name(run_arg, s)
    row["mean_r2"] = round(float(s.get("channel_r2_overall", float("nan"))), 4)
    for entry in (s.get("channel_per_param") or []):
        col = CHAN_MAP.get(entry.get("name"))
        if col is not None and entry.get("r2") is not None:
            row[col] = round(float(entry["r2"]), 4)
    if s.get("voltage_mse_z_mean") is not None:
        row["voltage_mse_z_mean"] = round(float(s["voltage_mse_z_mean"]), 4)
    if s.get("spike_count_diff_mean_abs") is not None:
        row["spike_diff_abs"] = round(float(s["spike_count_diff_mean_abs"]), 3)
    if s.get("n_samples") is not None:
        row["n_samples"] = int(s["n_samples"])
    row["summary_path"] = spath
    return row


def load_existing(out_path):
    rows = {}
    if os.path.isfile(out_path):
        with open(out_path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows[r["design"]] = r
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dir(s), out dir(s), or summary.yaml path(s)")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="ledger CSV (default: %(default)s)")
    args = ap.parse_args(argv)

    rows = load_existing(args.out)
    added = 0
    for run in args.runs:
        spath = resolve_summary(run)
        if spath is None:
            print(f"[skip] no summary.yaml under {run}", file=sys.stderr)
            continue
        r = row_from_summary(run, spath)
        rows[r["design"]] = r
        added += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ordered = sorted(rows.values(),
                     key=lambda r: (float(r["mean_r2"]) if r.get("mean_r2") not in ("", None) else -9,
                                    float(r["r2_kdr"]) if r.get("r2_kdr") not in ("", None) else -9),
                     reverse=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(ordered)

    # Pretty-print the ranked ledger.
    hdr = ["design", "mean_r2"] + CHAN_COLS + ["voltage_mse_z_mean", "spike_diff_abs"]
    widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in ordered)) if ordered else len(h)
              for h in hdr}
    print("  ".join(h.ljust(widths[h]) for h in hdr))
    for r in ordered:
        print("  ".join(str(r.get(h, "")).ljust(widths[h]) for h in hdr))
    print(f"\n[vo-ledger] {added} run(s) scored -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
