#!/usr/bin/env python3
"""Generate a synthetic mlPack1.h5 from a registered jaxley cell.

Phase 3+ of the jaxley voltage-loss plan: the training pack is generated
by the same jaxley cell that the HybridLoss uses as its forward simulator.
With data and sim matched, the inverse problem is well-posed and the CNN
can actually recover input parameters.

The default source cell is `ball_and_stick_bbp` (the original Phase 3
target), but `--source-cell <registered_name>` lets the same script
generate data for any cell that exposes `_DEFAULTS`, `PARAM_KEYS`, `_DT`,
`_DT_STIM`, `_T_MAX`, `_V_INIT` (e.g. `ca3_pyramidal`).

Pipeline
--------
1. Sample N unit-parameter vectors uniformly from [-1, 1]^12.
2. Map unit -> physical via phys = center * 10**(unit * log_halfspan)
   (centers from `ball_and_stick_bbp._DEFAULTS`, log_halfspan = 0.5).
3. Run jaxley `simulate_batch` in batches; record soma voltage.
4. Per-sample z-score along the time axis (matches Dataloader_H5 expectation).
5. Cast voltages to fp16, split 80/10/10 into train/valid/test, and write
   an `.mlPack1.h5` whose layout exactly mirrors the reference cADpyr pack
   (`<dom>_volts_norm` (N,T,P,S) fp16, `<dom>_unit_par` (N,P_cnn) f32,
   `meta.JSON` with `input_meta.phys_par_range`).

Usage
-----
    PYTHONNOUSERSITE=1 \\
    /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley/bin/python \\
        scripts/gen_ball_and_stick_data.py \\
        --out $SCRATCH/synthetic_bbp_data/ballBBP_synth_v1 \\
        --n 20000 --batch 128 --seed 0

The output file lives at <out>/<cell_name>.mlPack1.h5 where `cell_name`
defaults to `ball_and_stick_bbp_synth` so the design YAML can pass
`--cellName ball_and_stick_bbp_synth` and the dataloader will find it.

Performance: ~5-8 s per batch of 128 on 1 A100 in fp64.  20k samples in
156 batches ≈ 15-25 min; 65k ≈ 1-1.5 h.

"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# JAX env must be set BEFORE importing jax/jaxley.
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda" if (
    os.environ.get("SLURM_JOBID") or os.environ.get("CUDA_VISIBLE_DEVICES")
) else "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

# Make the repo root importable.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import h5py
import numpy as np
import torch

import jax
jax.config.update("jax_enable_x64", True)

from toolbox import JaxleyBridge
from toolbox.jaxley_utils import (
    load_stim_csv,
    phys_par_range_to_arrays,
    unit_to_phys_np,
)


def _load_source_cell(name: str):
    """Import a registered jaxley_cells module by name and validate its API."""
    import importlib
    mod = importlib.import_module(f"toolbox.jaxley_cells.{name}")
    required = ["_DEFAULTS", "PARAM_KEYS", "_DT", "_DT_STIM", "_T_MAX", "_V_INIT"]
    missing = [a for a in required if not hasattr(mod, a)]
    if missing:
        raise AttributeError(
            f"toolbox.jaxley_cells.{name} is missing required attributes: {missing}")
    return mod


# ─────────────────────────────────────────────────────────────────────────
# defaults
# ─────────────────────────────────────────────────────────────────────────

# `phys_par_range` matches `ballBBP_voltage_only.hpar.yaml`.  Same order as
# `ball_and_stick_bbp.PARAM_KEYS`.  Halfspan 0.5 -> unit∈[-1,1] maps to
# ~3.16x range around the BBP defaults.
_LOG_HALFSPAN = 0.5

# Cell-name token written into the meta + filename.  This is what the
# design YAML / SLR will pass as --cellName so the dataloader picks
# <cell_name>.mlPack1.h5.
_DEFAULT_CELL_NAME = "ball_and_stick_bbp_synth"

# Stim used for both data generation and the HybridLoss sim — must match
# the cell's `default_stim_name`.
_STIM_NAME = "5k50kInterChaoticB"


def _build_phys_par_range(param_keys, defaults, cell_mod=None):
    """Build the `[[center, log_halfspan, unit], ...]` list (per-key overrides
    from `cell_mod.PHYS_RANGE_OVERRIDES` when the module is given)."""
    if cell_mod is not None:
        from toolbox.jaxley_utils import build_phys_par_range
        return build_phys_par_range(cell_mod, _LOG_HALFSPAN)
    return [[float(defaults[k]), float(_LOG_HALFSPAN), "S/cm^2"]
            for k in param_keys]


# ─────────────────────────────────────────────────────────────────────────
# generation
# ─────────────────────────────────────────────────────────────────────────

def generate_voltages(unit_par, batch_size, src_cell_mod, src_cell_name,
                       fp64=True, verbose=True, centers=None, logspans=None,
                       checkpoint_lengths=None, linear_rows=None):
    """Run the jaxley sim on every row of `unit_par`.

    `centers`/`logspans` define the unit->phys map (phys = center·10^(unit·span)).
    If None, fall back to the cell's `_DEFAULTS` + uniform `_LOG_HALFSPAN` — but
    callers should pass the SAME mapping the training loss uses (e.g. from the
    design YAML's phys_par_range) so the generated data is self-consistent.

    Returns
    -------
    volts : (N, T, P) float32  - per-sample probe traces, NOT z-scored.
    """
    N, P = unit_par.shape
    if centers is None or logspans is None:
        centers = np.asarray(
            [src_cell_mod._DEFAULTS[k] for k in src_cell_mod.PARAM_KEYS],
            dtype=np.float64,
        )
        logspans = np.full(P, _LOG_HALFSPAN, dtype=np.float64)
    from toolbox.jaxley_utils import phys_par_range_linear_mask as _lm
    phys_all = unit_to_phys_np(unit_par.astype(np.float64), np.asarray(centers), np.asarray(logspans),
                               _lm(linear_rows) if linear_rows is not None else None)

    out = None
    n_batches = int(np.ceil(N / batch_size))
    t0 = time.time()
    for b in range(n_batches):
        i0 = b * batch_size
        i1 = min(N, i0 + batch_size)
        phys_b = torch.from_numpy(phys_all[i0:i1])
        if fp64:
            phys_b = phys_b.double()
        else:
            phys_b = phys_b.float()
        # `simulate_batch` returns (B, n_recorded, T_out).  n_recorded is
        # however many .record() calls the cell's _build made (1 for soma-
        # only cells, >1 for multi-probe cells like ball_and_stick).
        with torch.no_grad():
            # simulate_batch always builds jax.vjp; without checkpointing the
            # forward tape OOMs for the ~2000-comp L5 cell, so pass the same
            # checkpoint schedule the training loss uses.
            v = JaxleyBridge.simulate_batch(phys_b, src_cell_name,
                                            stim_name=_STIM_NAME,
                                            checkpoint_lengths=checkpoint_lengths)
        v_np = v.detach().cpu().numpy()  # (B, n_rec, T)
        if out is None:
            n_rec = v_np.shape[1]
            T = v_np.shape[-1]
            out = np.zeros((N, T, n_rec), dtype=np.float32)
            if verbose:
                print(f"[gen] T_sim={T} n_probes={n_rec} "
                      f"(per cell spec dt_stim={src_cell_mod._DT_STIM} ms)",
                      flush=True)
        # (B, n_rec, T) -> (B, T, n_rec)
        out[i0:i1] = np.moveaxis(v_np, 1, 2).astype(np.float32)
        if verbose and (b % 5 == 0 or b == n_batches - 1):
            dt = time.time() - t0
            done = i1
            rate = done / dt if dt > 0 else 0.0
            eta = (N - done) / rate if rate > 0 else float("inf")
            print(f"[gen] batch {b+1}/{n_batches}  samples {done}/{N}  "
                  f"{rate:.1f} samp/s  ETA {eta/60:.1f} min", flush=True)
    if verbose:
        print(f"[gen] all {N} sims done in {(time.time()-t0)/60:.1f} min", flush=True)
    return out


def normalize_volts_fixed_scale(volts):
    """Fixed-scale voltage normalization matching aggregate_Kaustubh.py.

    `volts` shape: (N, T, P).  Uses a single global mean/std
    (`toolbox.jaxley_utils.normalize_volts_fixed`) for every sample/probe, so
    the absolute voltage scale is preserved — the HybridLoss candidate side
    applies the identical transform.
    """
    from toolbox.jaxley_utils import normalize_volts_fixed
    return normalize_volts_fixed(volts).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# H5 writer
# ─────────────────────────────────────────────────────────────────────────

def write_h5(out_path, volts_norm, unit_par, splits, meta, cell_name):
    """Write the mlPack1 H5 file."""
    n_train, n_valid, n_test = splits
    assert n_train + n_valid + n_test == volts_norm.shape[0]

    # (N, T, P) -> (N, T, P, 1) so the dataloader's S axis is present
    volts4 = volts_norm[..., np.newaxis]   # (N, T, P, 1)

    # Phys params (just for record; the CNN trains in unit space)
    from toolbox.jaxley_utils import phys_par_range_linear_mask as _lm
    centers, logspans = phys_par_range_to_arrays(meta["input_meta"]["phys_par_range"])
    phys_par = unit_to_phys_np(unit_par.astype(np.float64), centers, logspans,
                               _lm(meta["input_meta"]["phys_par_range"])).astype(np.float32)

    sl_train = slice(0, n_train)
    sl_valid = slice(n_train, n_train + n_valid)
    sl_test  = slice(n_train + n_valid, n_train + n_valid + n_test)

    print(f"[gen] writing H5 -> {out_path}", flush=True)
    with h5py.File(out_path, "w") as f:
        for dom, sl in [("train", sl_train), ("valid", sl_valid), ("test", sl_test)]:
            f.create_dataset(f"{dom}_volts_norm", data=volts4[sl].astype(np.float16),
                             compression="gzip", compression_opts=4)
            f.create_dataset(f"{dom}_unit_par",   data=unit_par[sl].astype(np.float32))
            f.create_dataset(f"{dom}_phys_par",   data=phys_par[sl].astype(np.float32))

        # Stim sentinel — kept zero-length, mirrors the reference pack which
        # uses the stim-name dataset only as a marker (the actual stim is
        # loaded by the cell spec at sim time).
        f.create_dataset(_STIM_NAME, data=np.zeros((0,), dtype=np.float32))

        meta_blob = json.dumps(meta).encode("utf-8")
        dt = h5py.special_dtype(vlen=str)
        ds = f.create_dataset("meta.JSON", (1,), dtype=dt)
        ds[0] = meta_blob.decode("utf-8")
    print(f"[gen] wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)", flush=True)


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output dir; H5 written inside")
    ap.add_argument("--source-cell", default="ball_and_stick_bbp",
                    help="registered jaxley cell name (toolbox.jaxley_cells.<name>) "
                         "to use as the data-generating simulator. Default: %(default)s")
    ap.add_argument("--cell-name", default=None,
                    help="filename stem + meta.cell_name. Default: <source-cell>_synth")
    ap.add_argument("--n", type=int, default=20000, help="total samples")
    ap.add_argument("--split", nargs=3, type=int, default=[80, 10, 10],
                    metavar=("TRAIN", "VALID", "TEST"),
                    help="integer percentages summing to 100")
    ap.add_argument("--batch", type=int, default=128, help="jaxley batch size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp32", action="store_true",
                    help="run sims in fp32 (faster but may NaN at large t_max)")
    ap.add_argument("--ppr-yaml", default=None,
                    help="design .hpar.yaml whose voltage_loss.phys_par_range defines the "
                         "unit->phys map for BOTH generation and the H5 meta. Use the SAME "
                         "yaml the training run uses so data + loss are self-consistent.")
    ap.add_argument("--checkpoint", default=None,
                    help="comma-separated checkpoint_lengths for jx.integrate, e.g. 500,10 "
                         "(needed for the ~2000-comp L5 cell or simulate_batch's vjp OOMs)")
    args = ap.parse_args()

    if sum(args.split) != 100:
        raise ValueError(f"--split must sum to 100, got {args.split}")

    src_cell = _load_source_cell(args.source_cell)
    if args.cell_name is None:
        args.cell_name = f"{args.source_cell}_synth"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.cell_name}.mlPack1.h5"

    if out_path.exists():
        print(f"[gen] WARNING: {out_path} exists and will be overwritten", flush=True)

    rng = np.random.default_rng(args.seed)
    P = len(src_cell.PARAM_KEYS)
    print(f"[gen] source_cell={args.source_cell}  cell_name={args.cell_name}  "
          f"P={P} params  N={args.n}  batch={args.batch}  "
          f"seed={args.seed}  fp64={'no' if args.fp32 else 'yes'}", flush=True)
    print(f"[gen] PARAM_KEYS = {src_cell.PARAM_KEYS}", flush=True)

    # 0. resolve the unit->phys mapping (shared by gen + meta + training loss)
    if args.ppr_yaml:
        import yaml
        _yd = yaml.safe_load(open(args.ppr_yaml))
        phys_par_range = _yd["voltage_loss"]["phys_par_range"]
        assert len(phys_par_range) == P, \
            f"ppr-yaml has {len(phys_par_range)} entries but cell has {P} params"
        print(f"[gen] using phys_par_range from {args.ppr_yaml} (self-consistent map)", flush=True)
    else:
        phys_par_range = _build_phys_par_range(src_cell.PARAM_KEYS, src_cell._DEFAULTS, src_cell)
    gen_centers, gen_logspans = phys_par_range_to_arrays(phys_par_range)

    # 1. unit draws
    unit_par = rng.uniform(-1.0, 1.0, size=(args.n, P)).astype(np.float32)

    # 2-3. simulate
    ckpt = tuple(int(x) for x in args.checkpoint.split(",")) if args.checkpoint else None
    volts = generate_voltages(unit_par, batch_size=args.batch,
                               src_cell_mod=src_cell, src_cell_name=args.source_cell,
                               fp64=not args.fp32,
                               centers=gen_centers, logspans=gen_logspans, linear_rows=phys_par_range,
                               checkpoint_lengths=ckpt)
    if not np.isfinite(volts).all():
        n_nan = int(np.isnan(volts).sum())
        n_inf = int(np.isinf(volts).sum())
        raise RuntimeError(f"[gen] simulator produced non-finite values: "
                           f"{n_nan} NaN, {n_inf} Inf in {volts.size} elements")

    # 4. fixed-scale normalization (matches HybridLoss candidate side)
    volts_norm = normalize_volts_fixed_scale(volts)
    # Sanity: report the resulting global mean/std (NOT 0/1 — fixed scale
    # preserves absolute voltage, so these vary with the cell's dynamics).
    print(f"[gen] fixed-norm volts: global mean={volts_norm.mean():.3f} "
          f"std={volts_norm.std():.3f}", flush=True)

    # 5. split
    n = args.n
    n_train = n * args.split[0] // 100
    n_valid = n * args.split[1] // 100
    n_test  = n - n_train - n_valid
    print(f"[gen] split: train={n_train} valid={n_valid} test={n_test}", flush=True)

    # Shuffle once before splitting so each split is i.i.d.
    perm = rng.permutation(n)
    volts_norm = volts_norm[perm]
    unit_par = unit_par[perm]

    # 6. meta
    cell_spec = {
        "_DT": src_cell._DT,
        "_DT_STIM": src_cell._DT_STIM,
        "_T_MAX": src_cell._T_MAX,
        "_V_INIT": src_cell._V_INIT,
        "_STIM_SCALE": float(getattr(src_cell, "_STIM_SCALE", 1.0)),
        "_NCOMP": getattr(src_cell, "_NCOMP", None),
    }
    # phys_par_range already resolved above (from --ppr-yaml or _DEFAULTS)
    T = volts_norm.shape[1]
    n_probes = volts_norm.shape[2]
    probe_names = getattr(src_cell, "PROBE_NAMES", None)
    if probe_names is None:
        probe_names = ["soma"] if n_probes == 1 else [f"probe{i}" for i in range(n_probes)]
    assert len(probe_names) == n_probes, \
        f"PROBE_NAMES has {len(probe_names)} entries but sim returned {n_probes} probes"
    meta = {
        "cell_name":          args.cell_name,
        "source_cell":        args.source_cell,
        "num_phys_par":       P,
        "num_varied_phys_par": P,
        "num_probs":          int(n_probes),
        "num_stims":          1,
        "num_time_bins":      int(T),
        "parName":            list(src_cell.PARAM_KEYS),
        "probe_names":        list(probe_names),
        "stim_names":         [_STIM_NAME],
        "timeAxis":           {"step": float(src_cell._DT_STIM), "unit": "(ms)"},
        "phys_par_range":     phys_par_range,
        "input_meta":         {
            "parName":        list(src_cell.PARAM_KEYS),
            "phys_par_range": phys_par_range,
        },
        "simu_info": {
            "generator":     "scripts/gen_ball_and_stick_data.py",
            "stim_name":     _STIM_NAME,
            "log_halfspan":  _LOG_HALFSPAN,
            "fp64":          (not args.fp32),
            "seed":          args.seed,
            "cell_spec":     cell_spec,
            # `Trainer.patch_h5meta` reads probe/stim names from simu_info,
            # not from the top level — keep both for self-documentation.
            "probe_names":   list(probe_names),
            "stim_names":    [_STIM_NAME],
        },
        "pack_info": {
            "n_total":       int(n),
            "n_train":       int(n_train),
            "n_valid":       int(n_valid),
            "n_test":        int(n_test),
            "batch":         int(args.batch),
        },
    }

    # 7. write
    write_h5(out_path, volts_norm, unit_par,
             splits=(n_train, n_valid, n_test),
             meta=meta, cell_name=args.cell_name)

    # Final sanity
    with h5py.File(out_path, "r") as f:
        print("[gen] final H5 keys:", list(f.keys()), flush=True)
        print("[gen]   train_volts_norm:", f["train_volts_norm"].shape, f["train_volts_norm"].dtype,
              flush=True)
        print("[gen]   train_unit_par:  ", f["train_unit_par"].shape, f["train_unit_par"].dtype,
              flush=True)


if __name__ == "__main__":
    main()
