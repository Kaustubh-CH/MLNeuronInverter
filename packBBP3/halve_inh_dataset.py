#!/usr/bin/env python3
"""
Halve each cell's contribution in (already-shuffled) ONTRA Inhibitory datasets.

Why this works WITHOUT per-cell logic
-------------------------------------
The Inh datasets are packed by aggregate_All65.py, which does a *global*
`np.random.shuffle` of all samples before writing (verified in code AND on disk:
the 3 thirds of ALL_CELLS_Inhibitory_Intrapolated.simRaw.h5 are statistically
identical -> uniformly mixed, NOT contiguous per-cell blocks).

Because every cell is uniformly scattered, keeping any contiguous fraction of the
samples keeps that same fraction of EVERY cell, in proportion. So to "reduce each
cell's contribution to x/2" we simply keep the first `fraction` rows of each
sample-indexed dataset. No per-sample cell label is needed (and none exists in the
packed file).

Handles two file layouts automatically:
  * mlPack  (<split>_<field>, e.g. train_volts_norm): halves WITHIN each
    train/valid/test split, preserving the 8/1/1 ratio.
  * simRaw  (flat phys_par / unit_par / volts / *_stim_adjust): halves the flat
    sample axis.

Non-sample datasets (stim traces like 5k50kInterChaoticB, scalars, meta.JSON) are
copied verbatim. Large datasets are streamed in row-chunks so peak RAM stays low.

Usage
-----
  # one or more dataset dirs; writes <dir><outSuffix>/ alongside each:
  python3 halve_inh_dataset.py --inDirs /pscratch/.../BBP_Ontra_Inhibitory_Exclude_NGC-DA

  # all Inh dataset dirs under a parent matching a glob:
  python3 halve_inh_dataset.py --parent /pscratch/sd/k/ktub1999\
          --dirGlob 'BBP_Ontra_Inhibitory_Exclude_*'

  # preview only:
  python3 halve_inh_dataset.py --inDirs ... --dryRun
"""

import os, glob, json, argparse, time
import numpy as np
import h5py

SPLIT_PREFIXES = ("train_", "valid_", "test_")


def get_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--inDirs", nargs="+", help="explicit list of dataset directories")
    g.add_argument("--parent", help="parent dir; combine with --dirGlob")
    p.add_argument("--dirGlob", default="BBP_Ontra_Inhibitory_Exclude_*",
                   help="glob (under --parent) selecting dataset dirs")
    p.add_argument("--fileGlob", default="*.mlPack1.h5",
                   help="which h5 files inside each dir to halve "
                        "(use '*.h5' to also halve the .simRaw.h5)")
    p.add_argument("--outSuffix", default="_half",
                   help="output dir = <inDir> + outSuffix")
    p.add_argument("--fraction", type=float, default=0.5,
                   help="fraction of each cell's samples to keep (default 0.5)")
    p.add_argument("--random", action="store_true",
                   help="keep a RANDOM fraction instead of the first rows "
                        "(data is already shuffled, so off by default)")
    p.add_argument("--seed", type=int, default=1234, help="seed when --random")
    p.add_argument("--chunk", type=int, default=2048, help="row-chunk for streaming copy")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing output files")
    p.add_argument("--dryRun", action="store_true", help="print plan, write nothing")
    return p.parse_args()


def keep_index(n, frac, rng):
    """Indices to keep out of n samples."""
    k = int(round(n * frac))
    if rng is None:
        return slice(0, k), k                      # contiguous first-k (fast)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    return idx, k


def stream_copy(src, dst, sel, chunk):
    """Copy src[sel] -> dst in row-chunks. sel is a slice or sorted index array."""
    if isinstance(sel, slice):
        rows = range(sel.start, sel.stop, chunk)
        for i0 in rows:
            i1 = min(i0 + chunk, sel.stop)
            dst[i0 - sel.start:i1 - sel.start] = src[i0:i1]
    else:
        for w0 in range(0, len(sel), chunk):
            w1 = min(w0 + chunk, len(sel))
            dst[w0:w1] = src[sel[w0:w1]]


def is_sample_dataset(name, ds, n_total, split_lens):
    """Return the sample-count this dataset is indexed by, or None if not sample-indexed."""
    if ds.ndim == 0:
        return None
    n = ds.shape[0]
    if any(name.startswith(p) for p in SPLIT_PREFIXES):
        return n if n in split_lens else None
    if n == n_total and n_total > 0:
        return n
    return None


def halve_file(inF, outF, frac, rng, chunk, dry):
    try:
        hin = h5py.File(inF, "r")
    except OSError as e:
        print(f"  (SKIP, cannot open -- truncated/still downloading?) {inF}\n     {e}")
        return
    md = None
    if "meta.JSON" in hin:
        try:
            md = json.loads(hin["meta.JSON"][0])
        except Exception:
            md = None

    n_total = md["simu_info"]["num_total_samples"] if md else -1
    split_lens = set()
    if md and "pack_info" in md and "split_index" in md["pack_info"]:
        split_lens = {v[1] for v in md["pack_info"]["split_index"].values()}

    # plan
    plan = {}
    for name in hin.keys():
        if name == "meta.JSON":
            continue
        ds = hin[name]
        sc = is_sample_dataset(name, ds, n_total, split_lens)
        if sc is None:
            plan[name] = ("copy", ds.shape)
        else:
            sel, k = keep_index(sc, frac, rng)
            plan[name] = ("halve", (k,) + ds.shape[1:], sel)

    print(f"\n=== {os.path.basename(inF)} -> {outF}")
    for name, info in sorted(plan.items()):
        if info[0] == "copy":
            print(f"    copy   {name:28s} {info[1]}")
        else:
            print(f"    halve  {name:28s} {hin[name].shape} -> {info[1]}")
    if dry:
        hin.close()
        return

    os.makedirs(os.path.dirname(outF), exist_ok=True)
    t0 = time.time()
    hout = h5py.File(outF, "w")

    # update meta to reflect new sizes
    if md is not None:
        new_md = json.loads(json.dumps(md))  # deep copy
        if split_lens and "pack_info" in new_md:
            off = 0
            for sp in ("valid", "test", "train"):
                if sp in new_md["pack_info"]["split_index"]:
                    L = new_md["pack_info"]["split_index"][sp][1]
                    newL = int(round(L * frac))
                    new_md["pack_info"]["split_index"][sp] = [off, newL]
                    off += newL
            new_md["simu_info"]["num_total_samples"] = off
        elif n_total > 0:
            new_md["simu_info"]["num_total_samples"] = int(round(n_total * frac))
        new_md["pack_info_halved"] = {"source_file": os.path.basename(inF),
                                      "fraction": frac, "random": rng is not None}
        dtvs = h5py.special_dtype(vlen=str)
        dset = hout.create_dataset("meta.JSON", (1,), dtype=dtvs)
        dset[0] = json.dumps(new_md)

    for name, info in plan.items():
        src = hin[name]
        if info[0] == "copy":
            hout.create_dataset(name, data=src[()], dtype=src.dtype)
        else:
            newshape, sel = info[1], info[2]
            dst = hout.create_dataset(name, shape=newshape, dtype=src.dtype)
            stream_copy(src, dst, sel, chunk)

    hout.close()
    hin.close()
    sz = os.path.getsize(outF) / 1048576
    print(f"    wrote {sz:.0f} MB in {time.time()-t0:.0f}s")


def main():
    a = get_parser()
    if a.inDirs:
        dirs = a.inDirs
    else:
        dirs = sorted(glob.glob(os.path.join(a.parent, a.dirGlob)))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise SystemExit("no input dataset dirs found")

    rng = np.random.default_rng(a.seed) if a.random else None
    print(f"datasets to process: {len(dirs)}  fraction={a.fraction}  "
          f"mode={'random' if a.random else 'first-rows'}")

    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, a.fileGlob)))
        if not files:
            print(f"  (skip, no {a.fileGlob}) {d}")
            continue
        outDir = d.rstrip("/") + a.outSuffix
        for f in files:
            outF = os.path.join(outDir, os.path.basename(f))
            if os.path.exists(outF) and not a.overwrite and not a.dryRun:
                print(f"  (exists, skip) {outF}")
                continue
            halve_file(f, outF, a.fraction, rng, a.chunk, a.dryRun)

    print("\nDONE")


if __name__ == "__main__":
    main()
