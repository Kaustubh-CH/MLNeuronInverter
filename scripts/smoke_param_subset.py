#!/usr/bin/env python3
"""Smoke test for a --vary subset pack + HybridLoss.param_subset (GPU, ~2-5 min).

Builds the voltage loss exactly as Trainer does (build_hybrid_loss on the design's
voltage_loss block + the pack), then evaluates it on B test samples at
  (a) theta_true (the pack's subset labels, fed pre-tanh as atanh(u)),
  (b) the cell default (all-zero unit vector),
  (c) random unit draws.
PASS iff loss(a) is well below (b) and (c): proves the 10->19 expansion puts every
predicted channel in the right slot and the pinned ones at the generator's value.
  python scripts/smoke_param_subset.py <pack.h5> <design.hpar.yaml> [B]
"""
import sys, json, yaml, numpy as np, torch, h5py
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toolbox.HybridLoss import build_hybrid_loss

h5, design = sys.argv[1], sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 16
p = yaml.safe_load(open(design))
with h5py.File(h5, "r") as f:
    meta = json.loads(f["meta.JSON"][0])
    u = torch.tensor(f["test_unit_par"][:B], dtype=torch.float32)
    v = torch.tensor(f["test_volts_norm"][:B].astype(np.float32))[..., 0]   # (B, T, probes)
si = meta["simu_info"]
print("varied :", si["varied_params"]); print("pinned :", si["pinned_params"])
print("unit_par", tuple(u.shape), "volts", tuple(v.shape), "std(u)=", u.std(0).numpy().round(3))
params = {"use_voltage_loss": True, "voltage_loss": p["voltage_loss"], "full_h5name": h5,
          "local_batch_size": B, "batch_size": B}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crit = build_hybrid_loss(params).to(dev)
assert crit.param_subset == si["param_subset_indices"], (crit.param_subset, si["param_subset_indices"])
if hasattr(crit, "set_epoch"):
    crit.set_epoch(0)
v = v.to(dev); u = u.to(dev)
res = {}
with torch.no_grad():
    for tag, pu in [("true", torch.atanh(u.clamp(-0.999, 0.999))),
                    ("default", torch.zeros_like(u)),
                    ("random", torch.atanh(torch.empty_like(u).uniform_(-0.95, 0.95)))]:
        res[tag] = float(crit(pu, u, v))
        print(f"loss[{tag:7s}] = {res[tag]:.4f}", flush=True)
ok = res["true"] < res["default"] and res["true"] < res["random"]
print("SMOKE", "PASS" if ok else "FAIL", res)
sys.exit(0 if ok else 1)
