#!/usr/bin/env python3
"""Filter an existing Roy exp ca3ft mlPack down to ONE stimulus family,
preserving each sample's train/valid/test domain (so the neuron-level split —
and thus test-neuron identity — stays identical to the parent pack).

  python filter_pack_family.py --inH5 <parent.mlPack1.h5> --family Roy2000 \
         --outPath <dir> --outName RoyExp2000
"""
import argparse, json, os
import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--inH5", default="/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft/"
               "RoyExpChaotic.mlPack1.h5")
p.add_argument("--family", default="Roy2000",
               help="one family, or a comma-separated list (e.g. "
                    "'Roy500,Roy1000,Roy1500,Roy2000')")
p.add_argument("--outPath", required=True)
p.add_argument("--outName", required=True)
args = p.parse_args()

os.makedirs(args.outPath, exist_ok=True)
outF = os.path.join(args.outPath, args.outName + ".mlPack1.h5")
with h5py.File(args.inH5, "r") as fi, h5py.File(outF, "w") as fo:
    meta = json.loads(fi["meta.JSON"][0])
    families = [s for s in args.family.split(",") if s]
    counts = {}
    for dom in ["train", "valid", "test"]:
        m = np.isin(fi[dom + "_stim_family"][:].astype(str), families)
        counts[dom] = int(m.sum())
        for key in fi.keys():
            if key.startswith(dom + "_"):
                fo.create_dataset(key, data=fi[key][:][m])
    for key in fi.keys():   # unsplit stim waveforms etc.
        if not any(key.startswith(d + "_") for d in ["train", "valid", "test"]) \
                and key != "meta.JSON":
            fo.create_dataset(key, data=fi[key][:])
    meta["message"] += f" [family-filtered: {args.family} only, parent domains kept]"
    meta["family_filter"] = {"family": args.family, "parent": args.inH5,
                             "counts": counts}
    meta["pack_info"]["split_index"] = {
        "valid": [0, counts["valid"]], "test": [0, counts["test"]],
        "train": [0, counts["train"]]}
    dt = h5py.special_dtype(vlen=str)
    ds = fo.create_dataset("meta.JSON", (1,), dtype=dt)
    ds[0] = json.dumps(meta)
print("wrote", outF, counts)
