#!/usr/bin/env python3
"""Turn a JOINT multi-stim pack into a POOLED one by moving the battery from the
probe axis to the stim axis.  Pure reshape -- no new simulation, no GPU.

`gen_ca3_sharded.py` writes a K-stim battery on the PROBE axis:

    train_volts_norm  (N, T, K, 1)   num_probs=K  num_stims=1

which `Dataloader_H5` turns into a K-CHANNEL CNN input -- variant A ("stims as
separate channels"), where one sample shows the network all K traces at once.

Variant B wants each (param-draw, stim) pair to be an INDEPENDENT single-channel
sample.  `Dataloader_H5.openH5` already implements exactly that flatten, but it
keys off the STIM axis (the `serialize_stims` branch does
`moveaxis(-1,0).reshape(locSamp*numStim, T, -1)` and tiles `unit_par` to match).
So all variant B needs is the same bytes with the axes swapped:

    train_volts_norm  (N, T, 1, K)   num_probs=1  num_stims=K
    -> dataloader emits (N*K, T, 1) + unit_par tiled K times

The pooled sample count is K*N, but each sample is still ONE trace, so an epoch
costs the same N*K jaxley solves as variant A -- the A/B isolates input structure.

The per-sample stim identity is NOT stored in the H5: the flatten order is
deterministic (`stim_idx = flat_index // locSamp`, before the dataloader's
shuffle), so `Dataloader_H5` reconstructs it and hands it to `HybridLoss`, which
needs it to simulate each sample under its own stimulus.

Usage:
    python scripts/pool_multistim_pack.py --in <joint.mlPack1.h5> --out-dir <dir> \
           [--cell-name ca3_pyramidal_joint4_pooled]
"""
import argparse, json, os, sys
from pathlib import Path

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="joint mlPack1.h5 (num_probs=K)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cell-name", default=None,
                    help="H5 stem for the pooled pack; default <joint stem>_pooled")
    args = ap.parse_args()

    src = Path(args.inp)
    if not src.exists():
        sys.exit(f"missing input pack: {src}")
    stem = args.cell_name or (src.name.replace(".mlPack1.h5", "") + "_pooled")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{stem}.mlPack1.h5"

    with h5py.File(src, "r") as f:
        meta = json.loads(f["meta.JSON"][0])
        K = int(meta["num_probs"])
        if K < 2:
            sys.exit(f"input pack has num_probs={K}; nothing to pool")
        if int(meta.get("num_stims", 1)) != 1:
            sys.exit(f"expected num_stims=1 on a joint pack, got {meta['num_stims']}")
        stims = list(meta["stim_names"])
        if len(stims) != K:
            sys.exit(f"stim_names has {len(stims)} entries but num_probs={K}")

        print(f"pooling {src.name}: K={K} stims={stims}")
        with h5py.File(dst, "w") as g:
            for dom in ("train", "valid", "test"):
                key = f"{dom}_volts_norm"
                if key not in f:
                    continue
                v = f[key]                                   # (N, T, K, 1)
                N, T = v.shape[0], v.shape[1]
                assert v.shape[2] == K and v.shape[3] == 1, f"{key} shape {v.shape}"
                out = g.create_dataset(f"{dom}_volts_norm", shape=(N, T, 1, K),
                                       dtype=v.dtype, compression="gzip",
                                       compression_opts=4)
                # Stream in blocks: the full array is several GB at 80k x 5001 x 4.
                step = max(1, 2000)
                for i0 in range(0, N, step):
                    i1 = min(N, i0 + step)
                    out[i0:i1] = np.swapaxes(v[i0:i1], 2, 3)  # (n,T,K,1)->(n,T,1,K)
                for pk in (f"{dom}_unit_par", f"{dom}_phys_par"):
                    if pk in f:
                        g.create_dataset(pk, data=f[pk][...])
                print(f"  {dom}: {v.shape} -> {out.shape}")

            # Keep the stim CSV-stem placeholder datasets the joint pack carries.
            for sname in stims:
                if sname in f:
                    g.create_dataset(sname, data=np.zeros((0,), dtype=np.float32))

            meta["cell_name"] = stem
            meta["num_probs"] = 1
            meta["num_stims"] = K
            meta["probe_names"] = ["soma"]
            # stim_names order IS the stim-axis order, and therefore the vocabulary
            # the per-sample stim index points into. HybridLoss.pooled_stim_names
            # must match it exactly.
            meta["stim_names"] = stims
            meta.setdefault("simu_info", {})["pooled_from"] = str(src)
            meta["simu_info"]["pooled_axis"] = "stim"
            meta["simu_info"]["joint_per_sample_channels"] = False
            dt = h5py.special_dtype(vlen=str)
            ds = g.create_dataset("meta.JSON", (1,), dtype=dt)
            ds[0] = json.dumps(meta)

    print(f"wrote {dst} ({os.path.getsize(dst)/1e9:.2f} GB)")
    print(f"  num_probs=1 num_stims={K}; train with "
          f"--probsSelect 0 --stimsSelect {' '.join(str(i) for i in range(K))}")


if __name__ == "__main__":
    main()
