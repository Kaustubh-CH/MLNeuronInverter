#!/usr/bin/env python3
"""TEMPORARY sensitivity sweep for ball_and_stick.

Question being answered: is Leak_gLeak (dendrite passive leak) unrecoverable
from a SOMA recording because it is genuinely non-identifiable (the soma
trace barely moves when it changes), or just under-trained?

Method: hold all 4 unit params at 0 (= physical defaults), then sweep ONE
param at a time across its full unit range [-1, 1] (= +/-0.5 decade in phys
space, the same range the data generator samples).  For each swept value run
the jaxley forward sim (same JaxleyBridge path training uses) and measure how
much each recording site's z-scored trace moves vs the baseline (unit=0).

A param that strongly moves a site's trace is identifiable from that site.
We record soma + 2 dendrite sites, so we can see directly whether Leak_gLeak
shows up at the dendrite even though it is invisible at the soma.

Run on a GPU node:
    python toolbox/tests/sens_sweep_ball.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda" if os.environ.get("SLURM_JOBID") else "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import torch
import jax
jax.config.update("jax_enable_x64", True)

from toolbox import JaxleyBridge
from toolbox.jaxley_cells import ball_and_stick as cell_mod

CELL = "ball_and_stick"
PROBES = cell_mod.PROBE_NAMES
KEYS = cell_mod.PARAM_KEYS
DEFAULTS = cell_mod._DEFAULTS
LOG_HALFSPAN = 0.5     # must match scripts/gen_ball_and_stick_data.py
N_SWEEP = 9            # sweep points across [-1, 1]


def unit_to_phys(unit_vec):
    """unit in [-1,1]^P -> phys, log-scaled around defaults (matches gen)."""
    centers = np.array([DEFAULTS[k] for k in KEYS], dtype=np.float64)
    return centers * np.power(10.0, LOG_HALFSPAN * np.asarray(unit_vec))


def zscore(v):
    """z-score along time axis. v: (..., T)."""
    m = v.mean(axis=-1, keepdims=True)
    s = v.std(axis=-1, keepdims=True) + 1e-12
    return (v - m) / s


def main():
    P = len(KEYS)
    sweep_units = np.linspace(-1.0, 1.0, N_SWEEP)

    # Build the full phys matrix: for each param p, for each sweep value,
    # a vector that is all-defaults except param p set to the swept value.
    rows = []
    tags = []   # (param_idx, sweep_value)
    for p in range(P):
        for u in sweep_units:
            unit_vec = np.zeros(P)
            unit_vec[p] = u
            rows.append(unit_to_phys(unit_vec))
            tags.append((p, u))
    phys = torch.from_numpy(np.asarray(rows)).double()   # (P*N_SWEEP, P)

    print(f"[sweep] cell={CELL}  params={KEYS}  probes={PROBES}")
    print(f"[sweep] running {phys.shape[0]} sims ...", flush=True)
    with torch.no_grad():
        v = JaxleyBridge.simulate_batch(phys, CELL)   # (B, n_rec, T)
    v = v.detach().cpu().numpy().astype(np.float64)
    B, n_rec, T = v.shape
    print(f"[sweep] sim out: B={B} n_rec={n_rec} T={T}")
    assert n_rec == len(PROBES), f"expected {len(PROBES)} recordings, got {n_rec}"

    vz = zscore(v)   # (B, n_rec, T)

    # Baseline = the all-defaults trace (any param's u=0 row; pick param 0's).
    base_idx = {p: None for p in range(P)}
    for i, (p, u) in enumerate(tags):
        if abs(u) < 1e-9:
            base_idx[p] = i
    # all params share the same u=0 trace, but indexing per-param is robust.

    # For each (param, site): max over sweep of RMSE_z( trace, baseline ).
    print("\n=== per-site z-scored trace RMSE swing (max over sweep range) ===")
    header = f"{'param':<14}" + "".join(f"{pr:>14}" for pr in PROBES)
    print(header)
    print("-" * len(header))
    result = {}
    for p in range(P):
        base = vz[base_idx[p]]                       # (n_rec, T)
        swings = np.zeros(n_rec)
        for i, (pp, u) in enumerate(tags):
            if pp != p:
                continue
            rmse = np.sqrt(((vz[i] - base) ** 2).mean(axis=-1))   # (n_rec,)
            swings = np.maximum(swings, rmse)
        result[KEYS[p]] = swings
        row = f"{KEYS[p]:<14}" + "".join(f"{s:>14.4f}" for s in swings)
        print(row)

    # Relative identifiability: each param's best-site swing vs the strongest.
    print("\n=== identifiability summary ===")
    best = {k: float(np.max(s)) for k, s in result.items()}
    soma = {k: float(s[0]) for k, s in result.items()}
    top = max(best.values())
    for k in KEYS:
        site = PROBES[int(np.argmax(result[k]))]
        print(f"  {k:<14} soma_swing={soma[k]:.4f}  best_swing={best[k]:.4f} "
              f"(@{site})  rel_to_strongest={best[k]/top:.3f}")
    print("\n[sweep] interpretation: a param whose soma_swing is ~0 but whose "
          "best_swing (at a dendrite site) is large is identifiable ONLY with "
          "dendrite recordings.")


if __name__ == "__main__":
    main()
