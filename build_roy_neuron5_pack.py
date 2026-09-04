#!/usr/bin/env python
"""Build the NEURON-level 5-sweep battery pack from RoyExpPack_ca3ft.

Option 2 (user, 2026-08-28): give the CNN ALL of a neuron's sweeps and demand
ONE parameter set per neuron.  Each output sample is one neuron represented by
one sweep per stimulus family, stacked on the PROBE axis (the joint4 'stims as
CNN channels' layout):

    <dom>_volts_norm : (M, 4001, 5, 1) fp16   channels = [Roy100..Roy2000]
    <dom>_raw_volts_mV: (M, 4000, 5)  f32     same sweeps in mV (for eval)
    <dom>_unit_par   : (M, 6) zeros           (voltage-only; labels unused)
    <dom>_neuron_id / _combo_idx / _src_rows  provenance

Sweep combinations are a data augmentation: per neuron we draw N_COMBO random
one-sweep-per-family assignments (seeded).  Neurons missing a family entirely
are dropped (reported).  Train with ca3_royexp_neuron5_joint.hpar.yaml:
stim_names_multi = the 5 icaRec stims, sim under stim i vs channel i.
"""
import json
import numpy as np, h5py

SRC = "/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft/RoyExpChaotic.mlPack1.h5"
OUT = "/pscratch/sd/k/ktub1999/RoyExpPack_neuron5/RoyExpNeuron5.mlPack1.h5"
FAMS = ["Roy100", "Roy500", "Roy1000", "Roy1500", "Roy2000"]
N_COMBO = {"train": 12, "valid": 8, "test": 8}
SEED = 0

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
rng = np.random.default_rng(SEED)

with h5py.File(SRC, "r") as fs, h5py.File(OUT, "w") as fo:
    meta = json.loads(fs["meta.JSON"][0])
    for dom in ("train", "valid", "test"):
        X = fs[dom + "_volts_norm"][:, :, 0, 0]          # (N,4001) fp16
        raw = fs[dom + "_raw_volts_mV"][:]               # (N,4000)
        famS = fs[dom + "_stim_family"][:].astype(str)
        nids = fs[dom + "_neuron_id"][:].astype(str)
        by = {}                                          # neuron -> fam -> [rows]
        for i, (n, f) in enumerate(zip(nids, famS)):
            by.setdefault(n, {}).setdefault(f, []).append(i)
        keep = sorted(n for n, d in by.items() if all(f in d for f in FAMS))
        drop = sorted(set(by) - set(keep))
        if drop:
            print(f"[{dom}] DROP neurons missing a family: {drop}")
        Vn, Rw, Nid, Cid, Src = [], [], [], [], []
        for n in keep:
            for k in range(N_COMBO[dom]):
                rows = [int(rng.choice(by[n][f])) for f in FAMS]
                Vn.append(np.stack([X[r] for r in rows], axis=1))     # (4001,5)
                Rw.append(np.stack([raw[r] for r in rows], axis=1))   # (4000,5)
                Nid.append(n); Cid.append(k); Src.append(rows)
        M = len(Vn)
        fo.create_dataset(dom + "_volts_norm",
                          data=np.asarray(Vn, np.float16)[..., None],
                          chunks=(1, 4001, 5, 1))
        fo.create_dataset(dom + "_raw_volts_mV", data=np.asarray(Rw, np.float32))
        fo.create_dataset(dom + "_unit_par", data=np.zeros((M, 6), np.float32))
        fo.create_dataset(dom + "_neuron_id",
                          data=np.asarray(Nid, dtype=h5py.string_dtype()))
        fo.create_dataset(dom + "_combo_idx", data=np.asarray(Cid, np.int32))
        fo.create_dataset(dom + "_src_rows", data=np.asarray(Src, np.int32))
        print(f"[{dom}] {len(keep)} neurons x {N_COMBO[dom]} combos = {M} samples")
    # recorded stims travel with the pack (provenance for icaRec)
    for k in fs:
        if k.startswith("stim_"):
            fo.create_dataset(k, data=fs[k][:])
    meta["cell_name"] = "RoyExpNeuron5"
    meta["num_probs"] = 5
    meta["probe_names"] = [f + "_sweep" for f in FAMS]
    # Trainer.patch_h5meta re-derives probe/stim names by INDEXING
    # simu_info['probe_names'][probsSelect] and ['stim_names'][stimsSelect]
    meta.setdefault("simu_info", {})
    meta["simu_info"]["probe_names"] = [f + "_sweep" for f in FAMS]
    meta["simu_info"]["stim_names"] = [f + "_icaRec_5k" for f in FAMS]
    meta["neuron5_info"] = {
        "source": SRC, "families": FAMS, "n_combo": N_COMBO, "seed": SEED,
        "layout": "probe axis = one sweep per family; joint stim_names_multi "
                  "= Roy<amp>_icaRec_5k, sim ch i vs data ch i"}
    # 1-element ARRAY, not a scalar: Dataloader_H5 reads meta.JSON[0]
    fo.create_dataset("meta.JSON",
                      data=np.asarray([json.dumps(meta)],
                                      dtype=h5py.string_dtype()))
print("wrote", OUT)
