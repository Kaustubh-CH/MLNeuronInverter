#!/usr/bin/env python3
"""Resample every stimulus CSV to a fixed number of time steps.

Motivation: the raw DL4neurons2 stims have very different lengths (4k..23k
samples), so under the dt_stim=0.1 ms convention each is simulated over a
different window (400..2300 ms).  Interpolating them all to a common length N
puts the sensitivity sweep on equal footing — identical simulation window
(N*dt_stim ms) and one XLA compile for the whole battery.

For each *.csv in --srcDir it linearly interpolates the 1D current waveform
onto N evenly spaced points and writes <dstDir>/<name>.csv (one value/row).
Header'd files (e.g. the ChaoticB* 'time,scaled_data' CSVs) are parsed and
their current column is used.

Usage:
  python make_interpolated_stims.py --n 4000 \
     --srcDir /global/homes/k/ktub1999/mainDL4/DL4neurons2/stims \
     --dstDir /global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/stim_interpolated
"""

import os, argparse, glob
import numpy as np


def load_current(path):
    """Return a 1D float array of the stimulus current, robust to a header."""
    try:
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            return arr.astype(np.float64)
        # multi-column, no header -> take the last column (scaled current).
        return arr[:, -1].astype(np.float64)
    except ValueError:
        # Has a text header like 'time,scaled_data'. Parse with genfromtxt.
        arr = np.genfromtxt(path, delimiter=",", names=True)
        cols = arr.dtype.names
        # Prefer a column that looks like the current/data, else the last one.
        pick = None
        for c in cols:
            if "data" in c.lower() or "scaled" in c.lower() or "current" in c.lower():
                pick = c; break
        pick = pick or cols[-1]
        return np.asarray(arr[pick], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcDir", default="/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims")
    ap.add_argument("--dstDir", default="/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/stim_interpolated")
    ap.add_argument("--n", type=int, default=4000, help="target number of time steps")
    args = ap.parse_args()

    os.makedirs(args.dstDir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.srcDir, "*.csv")))
    ok, bad = 0, []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            cur = load_current(f)
            if cur.ndim != 1 or cur.size < 2 or not np.all(np.isfinite(cur)):
                raise ValueError(f"bad current shape/values {cur.shape}")
            # Interpolate native index [0, L-1] -> N evenly spaced points.
            L = cur.size
            x_old = np.linspace(0.0, 1.0, L)
            x_new = np.linspace(0.0, 1.0, args.n)
            cur_i = np.interp(x_new, x_old, cur).astype(np.float64)
            np.savetxt(os.path.join(args.dstDir, f"{name}.csv"), cur_i)
            ok += 1
        except Exception as e:
            bad.append((name, str(e)))
            print(f"[interp] SKIP {name}: {e}")
    print(f"[interp] wrote {ok} stims -> {args.dstDir}  (target N={args.n})")
    if bad:
        print(f"[interp] skipped {len(bad)}: {[b[0] for b in bad]}")


if __name__ == "__main__":
    main()
