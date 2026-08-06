#!/usr/bin/env python3
"""Integration check (needs the real ca3_joint4_v1 pack; run on a login node).

End-to-end check that pooled sample i really carries stim stim_idx[i].

Builds a tiny joint pack from the real one, pools it, runs the actual
Dataloader_H5 Dataset, and asserts every emitted (trace, stim_idx, params)
triple matches the ORIGINAL joint pack. A misalignment here would silently
train every sample against the wrong protocol.
"""
import json, subprocess, sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path("/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur")
sys.path.insert(0, str(REPO))
SCRATCH = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pooled_align_check")
SRC = "/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_joint4_v1/ca3_pyramidal_joint4.mlPack1.h5"

N = 40          # tiny joint pack: N rows, 4 stims
TCUT = 64       # truncate time axis to keep it fast
# Take the window from MID-trace: every stimulus is quiescent before ~87 ms, so an
# early window makes all four traces byte-identical and unverifiable by content.
T0 = 2000       # 200 ms in, inside every stim's active region

tiny_dir = SCRATCH / "tiny_joint"; tiny_dir.mkdir(parents=True, exist_ok=True)
tiny = tiny_dir / "tiny_joint.mlPack1.h5"

# ── build the tiny joint pack (same layout: num_probs=4, num_stims=1) ──────
with h5py.File(SRC, "r") as f:
    meta = json.loads(f["meta.JSON"][0])
    vol = f["train_volts_norm"][: N * 3, T0:T0 + TCUT]   # (3N, TCUT, 4, 1)
    up = f["train_unit_par"][: N * 3]
    pp = f["train_phys_par"][: N * 3]
meta["num_time_bins"] = TCUT
splits = {"train": slice(0, N), "valid": slice(N, 2 * N), "test": slice(2 * N, 3 * N)}
with h5py.File(tiny, "w") as g:
    for dom, s in splits.items():
        g.create_dataset(f"{dom}_volts_norm", data=vol[s])
        g.create_dataset(f"{dom}_unit_par", data=up[s])
        g.create_dataset(f"{dom}_phys_par", data=pp[s])
    for sname in meta["stim_names"]:
        g.create_dataset(sname, data=np.zeros((0,), dtype=np.float32))
    dt = h5py.special_dtype(vlen=str)
    ds = g.create_dataset("meta.JSON", (1,), dtype=dt); ds[0] = json.dumps(meta)
print(f"tiny joint pack: {tiny}  volts {vol[:N].shape}")

# ── pool it with the real script ──────────────────────────────────────────
pool_dir = SCRATCH / "tiny_pooled"
r = subprocess.run([sys.executable, str(REPO / "scripts/pool_multistim_pack.py"),
                    "--in", str(tiny), "--out-dir", str(pool_dir),
                    "--cell-name", "tiny_pool"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
assert r.returncode == 0, "pooling failed"
pooled = pool_dir / "tiny_pool.mlPack1.h5"

with h5py.File(pooled, "r") as g:
    pm = json.loads(g["meta.JSON"][0])
assert pm["num_probs"] == 1 and pm["num_stims"] == 4, (pm["num_probs"], pm["num_stims"])
assert pm["stim_names"] == meta["stim_names"], "stim vocabulary reordered by pooling!"
print(f"pooled meta OK: num_probs=1 num_stims=4 stim_names={pm['stim_names']}")

# ── run the REAL Dataset over the pooled pack ─────────────────────────────
from toolbox.Dataloader_H5 import Dataset_h5_neuronInverter   # noqa: E402

conf = {
    "h5name": str(pooled), "domain": "train", "world_rank": 0, "world_size": 1,
    "local_batch_size": 8, "name": "verify", "use_manual_features": False,
    "data_conf": {
        "probs_select": [0], "stims_select": [0, 1, 2, 3], "valid_stims_select": [2],
        "serialize_stims": True, "append_stim": False, "parallel_stim": False,
        "num_data_workers": 0, "pooled_stim_index": True,
    },
}
dset = Dataset_h5_neuronInverter(conf, verb=0)
print(f"pooled dataset: {len(dset)} samples (expect {N}*4={N*4}), "
      f"frames {dset.data_frames.shape}")
assert len(dset) == N * 4, f"expected {N*4} pooled samples, got {len(dset)}"
assert dset.data_frames.shape[-1] == 1, "pooled input must be single-channel"

# Ground truth straight from the tiny JOINT pack.
with h5py.File(tiny, "r") as f:
    joint = f["train_volts_norm"][...].astype(np.float32)   # (N, T, 4, 1)
    jpar = f["train_unit_par"][...]

# Map each trace back to its (row, stim) by exact match, then check the label.
lookup = {}
for r_ in range(N):
    for s_ in range(4):
        lookup[joint[r_, :, s_, 0].tobytes()] = (r_, s_)
assert len(lookup) == N * 4, "traces not unique; cannot verify by content"

mism_stim = mism_par = 0
for i in range(len(dset)):
    X, sidx, Y = dset[i]
    key = np.asarray(X[:, 0], dtype=np.float32).tobytes()
    assert key in lookup, f"sample {i}: trace not found in the joint pack"
    row, stim = lookup[key]
    if int(sidx) != stim:
        mism_stim += 1
    if not np.allclose(Y, jpar[row], atol=1e-6):
        mism_par += 1

print()
print(f"checked {len(dset)} pooled samples against the joint pack")
print(f"  stim-index mismatches : {mism_stim}")
print(f"  param   mismatches    : {mism_par}")
counts = np.bincount(dset.data_stimIdx, minlength=4)
print(f"  per-stim counts       : {counts.tolist()} (expect {[N]*4})")
assert mism_stim == 0, "STIM INDEX MISALIGNED -- would train against the wrong protocol"
assert mism_par == 0, "unit_par tiling misaligned"
assert (counts == N).all(), "pooled set is not balanced across stims"
print("\nPASS: every pooled sample carries its own trace, stim id and params")
