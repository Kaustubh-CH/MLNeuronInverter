#!/usr/bin/env python3
"""Generate a JOINT 3-stimulus mlPack1.h5 from a registered jaxley cell (EXP 3).

Multi-stimulus variant of scripts/gen_ball_and_stick_data.py.

Hypothesis (Exp 3): probing the SAME cell with THREE different stimuli, each
exciting different channel dynamics, gives the inverse CNN the info it needs to
recover more parameters than a single stim can.

Design: JOINT per-sample channels. Every training sample is the SAME 19-param
set simulated under ALL 3 stimuli; the 3 soma traces are stacked along the
probe/channel axis -> volts (N, T, 3). The CNN sees all 3 jointly; the physics
loss (HybridLoss.stim_names_multi) re-simulates pred_phys under each stim and
supervises each channel.

Channel order is fixed and MUST match the design YAML's
`voltage_loss.stim_names_multi` and the run's `--probsSelect 0 1 2`:

    STIM_LIST = ["5k50kInterChaoticB", "5k0step_500", "5k0chirp"]
                  channel 0 (chaotic)   channel 1       channel 2

H5 layout is BYTE-COMPATIBLE with the single-stim pack: the 3 stims live on the
PROBE axis (num_probs=3, num_stims=1) so the existing dataloader probe machinery
handles them unchanged.

Usage
-----
    L5TTPC_NCOMP=2 JAX_ENABLE_X64=true \\
    PYTHONNOUSERSITE=1 python scripts/gen_multistim_data.py \\
        --source-cell l5ttpc --cell-name L5TTPC_multistim \\
        --out /pscratch/sd/k/ktub1999/l5ttpc_multistim_data/ \\
        --ppr-yaml l5ttpc_multistim.hpar.yaml \\
        --n 20000 --batch 64 --seed 0 --checkpoint 500,10

Cost: ~3x the single-stim gen (3 sims per sample). ~1-1.5 h for 20k.
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
    phys_par_range_to_arrays,
    unit_to_phys_np,
)


# ─────────────────────────────────────────────────────────────────────────
# the 3 stimuli — order == channel order == design YAML stim_names_multi
# ─────────────────────────────────────────────────────────────────────────
STIM_LIST   = ["5k50kInterChaoticB", "5k0step_500", "5k0chirp"]
PROBE_NAMES = ["chaotic", "step500", "chirp"]

_LOG_HALFSPAN = 0.5


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


def _build_phys_par_range(param_keys, defaults):
    return [[float(defaults[k]), float(_LOG_HALFSPAN), "S/cm^2"]
            for k in param_keys]


# ─────────────────────────────────────────────────────────────────────────
# generation
# ─────────────────────────────────────────────────────────────────────────

def generate_voltages_one_stim(phys_all, batch_size, src_cell_name, stim_name,
                               fp64=True, verbose=True, checkpoint_lengths=None):
    """Run the jaxley sim for ONE stim on every row of `phys_all`.

    Returns
    -------
    volts : (N, T) float32  - per-sample SOMA trace (n_recorded==1), NOT z-scored.
    """
    N = phys_all.shape[0]
    out = None
    n_batches = int(np.ceil(N / batch_size))
    t0 = time.time()
    for b in range(n_batches):
        i0 = b * batch_size
        i1 = min(N, i0 + batch_size)
        phys_b = torch.from_numpy(phys_all[i0:i1])
        phys_b = phys_b.double() if fp64 else phys_b.float()
        with torch.no_grad():
            v = JaxleyBridge.simulate_batch(phys_b, src_cell_name,
                                            stim_name=stim_name,
                                            checkpoint_lengths=checkpoint_lengths)
        v_np = v.detach().cpu().numpy()  # (B, n_rec, T)
        n_rec = v_np.shape[1]
        if n_rec != 1:
            raise RuntimeError(
                f"[gen] expected soma-only (n_rec==1), got n_rec={n_rec} for "
                f"stim {stim_name}. This generator stacks 1 soma trace per stim.")
        if out is None:
            T = v_np.shape[-1]
            out = np.zeros((N, T), dtype=np.float32)
            if verbose:
                print(f"[gen]   stim={stim_name}: T_sim={T} n_rec={n_rec}", flush=True)
        out[i0:i1] = v_np[:, 0, :].astype(np.float32)
        if verbose and (b % 10 == 0 or b == n_batches - 1):
            dt = time.time() - t0
            done = i1
            rate = done / dt if dt > 0 else 0.0
            eta = (N - done) / rate if rate > 0 else float("inf")
            print(f"[gen]   {stim_name} batch {b+1}/{n_batches}  {done}/{N}  "
                  f"{rate:.1f} samp/s  ETA {eta/60:.1f} min", flush=True)
    if verbose:
        print(f"[gen]   stim={stim_name} done in {(time.time()-t0)/60:.1f} min", flush=True)
    return out


def normalize_volts_fixed_scale(volts):
    """Fixed-scale voltage normalization matching aggregate_Kaustubh.py.

    volts (N, T, P).  Single global mean/std for every sample/channel, so the
    absolute voltage scale is preserved and the transform matches the
    HybridLoss candidate side exactly.
    """
    from toolbox.jaxley_utils import normalize_volts_fixed
    return normalize_volts_fixed(volts).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# H5 writer
# ─────────────────────────────────────────────────────────────────────────

def write_h5(out_path, volts_norm, unit_par, splits, meta):
    """Write the mlPack1 H5 file.  volts_norm shape (N, T, P=3)."""
    n_train, n_valid, n_test = splits
    assert n_train + n_valid + n_test == volts_norm.shape[0]

    # (N, T, P) -> (N, T, P, 1) so the dataloader's S axis is present
    volts4 = volts_norm[..., np.newaxis]   # (N, T, 3, 1)

    centers, logspans = phys_par_range_to_arrays(meta["input_meta"]["phys_par_range"])
    phys_par = unit_to_phys_np(unit_par.astype(np.float64), centers, logspans).astype(np.float32)

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

        # Stim-name sentinels (zero-length markers; actual stim loaded at sim time).
        for sname in STIM_LIST:
            f.create_dataset(sname, data=np.zeros((0,), dtype=np.float32))

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
    ap.add_argument("--source-cell", default="l5ttpc",
                    help="registered jaxley cell name. Default: %(default)s")
    ap.add_argument("--cell-name", default="L5TTPC_multistim",
                    help="filename stem + meta.cell_name. Default: %(default)s")
    ap.add_argument("--n", type=int, default=20000, help="total samples")
    ap.add_argument("--split", nargs=3, type=int, default=[80, 10, 10],
                    metavar=("TRAIN", "VALID", "TEST"),
                    help="integer percentages summing to 100")
    ap.add_argument("--batch", type=int, default=64, help="jaxley batch size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp32", action="store_true",
                    help="run sims in fp32 (faster but may NaN at large t_max)")
    ap.add_argument("--ppr-yaml", default=None,
                    help="design .hpar.yaml whose voltage_loss.phys_par_range defines "
                         "the unit->phys map for BOTH generation and the H5 meta.")
    ap.add_argument("--checkpoint", default="500,10",
                    help="comma-separated checkpoint_lengths for jx.integrate. "
                         "Default: %(default)s")
    args = ap.parse_args()

    if sum(args.split) != 100:
        raise ValueError(f"--split must sum to 100, got {args.split}")

    src_cell = _load_source_cell(args.source_cell)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.cell_name}.mlPack1.h5"
    if out_path.exists():
        print(f"[gen] WARNING: {out_path} exists and will be overwritten", flush=True)

    rng = np.random.default_rng(args.seed)
    P = len(src_cell.PARAM_KEYS)
    print(f"[gen] EXP3 multi-stim  source_cell={args.source_cell}  "
          f"cell_name={args.cell_name}  P={P}  N={args.n}  batch={args.batch}  "
          f"seed={args.seed}  fp64={'no' if args.fp32 else 'yes'}", flush=True)
    print(f"[gen] STIM_LIST (channel order) = {STIM_LIST}", flush=True)
    print(f"[gen] PROBE_NAMES               = {PROBE_NAMES}", flush=True)
    print(f"[gen] L5TTPC_NCOMP={os.environ.get('L5TTPC_NCOMP','?')}", flush=True)

    # 0. resolve unit->phys mapping (shared by gen + meta + training loss)
    if args.ppr_yaml:
        import yaml
        _yd = yaml.safe_load(open(args.ppr_yaml))
        phys_par_range = _yd["voltage_loss"]["phys_par_range"]
        assert len(phys_par_range) == P, \
            f"ppr-yaml has {len(phys_par_range)} entries but cell has {P} params"
        print(f"[gen] using phys_par_range from {args.ppr_yaml} (self-consistent map)", flush=True)
    else:
        phys_par_range = _build_phys_par_range(src_cell.PARAM_KEYS, src_cell._DEFAULTS)
    gen_centers, gen_logspans = phys_par_range_to_arrays(phys_par_range)

    # 1. unit draws — ONCE; reused for all 3 stims (same params, different stim)
    unit_par = rng.uniform(-1.0, 1.0, size=(args.n, P)).astype(np.float32)
    phys_all = unit_to_phys_np(unit_par.astype(np.float64),
                               np.asarray(gen_centers), np.asarray(gen_logspans))

    # 2-3. simulate EACH stim over ALL N samples; stack along channel axis
    ckpt = tuple(int(x) for x in args.checkpoint.split(",")) if args.checkpoint else None
    t_all = time.time()
    chan_volts = []
    # Per-stim resume cache: each completed stim is saved to .npy so a walltime
    # kill only loses the in-progress stim, not all prior stims. Key = cell_name
    # + stim index (cell_name encodes the seed/shard), so parallel ranks don't
    # collide. A rerun with identical args reuses finished stims.
    for ci, sname in enumerate(STIM_LIST):
        cache_f = out_dir / f".gencache_{args.cell_name}_stim{ci}.npy"
        if cache_f.exists():
            v = np.load(cache_f)
            if v.shape[0] == args.n:
                print(f"[gen] === stim {ci}/{len(STIM_LIST)-1}: {sname} (RESUMED from {cache_f.name}, {v.shape}) ===", flush=True)
                chan_volts.append(v)
                continue
            print(f"[gen] cache {cache_f.name} shape {v.shape} != N={args.n}; regenerating", flush=True)
        print(f"[gen] === stim {ci}/{len(STIM_LIST)-1}: {sname} ===", flush=True)
        v = generate_voltages_one_stim(
            phys_all, batch_size=args.batch, src_cell_name=args.source_cell,
            stim_name=sname, fp64=not args.fp32, checkpoint_lengths=ckpt)
        if not np.isfinite(v).all():
            n_nan = int(np.isnan(v).sum()); n_inf = int(np.isinf(v).sum())
            raise RuntimeError(f"[gen] stim {sname} produced non-finite values: "
                               f"{n_nan} NaN, {n_inf} Inf in {v.size} elements")
        np.save(cache_f, v)   # persist completed stim for resume
        print(f"[gen]   cached stim {ci} -> {cache_f.name}", flush=True)
        chan_volts.append(v)   # (N, T)
    # all stims must share T (verified: all 5000 samples = 500ms @ 0.1ms)
    Ts = [v.shape[1] for v in chan_volts]
    assert len(set(Ts)) == 1, f"stim trace lengths differ: {dict(zip(STIM_LIST, Ts))}"
    volts = np.stack(chan_volts, axis=2)   # (N, T, 3)
    print(f"[gen] stacked volts {volts.shape}  ({(time.time()-t_all)/60:.1f} min total)", flush=True)

    # 4. fixed-scale normalization (matches HybridLoss candidate side)
    volts_norm = normalize_volts_fixed_scale(volts)
    print(f"[gen] fixed-norm volts: global mean={volts_norm.mean():.3f} "
          f"std={volts_norm.std():.3f}", flush=True)

    # 5. split
    n = args.n
    n_train = n * args.split[0] // 100
    n_valid = n * args.split[1] // 100
    n_test  = n - n_train - n_valid
    print(f"[gen] split: train={n_train} valid={n_valid} test={n_test}", flush=True)

    perm = rng.permutation(n)
    volts_norm = volts_norm[perm]
    unit_par = unit_par[perm]

    # 6. meta
    cell_spec = {
        "_DT": src_cell._DT, "_DT_STIM": src_cell._DT_STIM,
        "_T_MAX": src_cell._T_MAX, "_V_INIT": src_cell._V_INIT,
    }
    T = volts_norm.shape[1]
    n_probes = volts_norm.shape[2]
    assert n_probes == len(STIM_LIST) == len(PROBE_NAMES) == 3
    meta = {
        "cell_name":           args.cell_name,
        "source_cell":         args.source_cell,
        "num_phys_par":        P,
        "num_varied_phys_par": P,
        "num_probs":           int(n_probes),
        "num_stims":           1,
        "num_time_bins":       int(T),
        "parName":             list(src_cell.PARAM_KEYS),
        "probe_names":         list(PROBE_NAMES),
        "stim_names":          list(STIM_LIST),
        "timeAxis":            {"step": float(src_cell._DT_STIM), "unit": "(ms)"},
        "phys_par_range":      phys_par_range,
        "input_meta": {
            "parName":        list(src_cell.PARAM_KEYS),
            "phys_par_range": phys_par_range,
        },
        "simu_info": {
            "generator":     "scripts/gen_multistim_data.py",
            "stim_names":    list(STIM_LIST),
            "joint_per_sample_channels": True,
            "log_halfspan":  _LOG_HALFSPAN,
            "fp64":          (not args.fp32),
            "seed":          args.seed,
            "cell_spec":     cell_spec,
            "probe_names":   list(PROBE_NAMES),
        },
        "pack_info": {
            "n_total": int(n), "n_train": int(n_train),
            "n_valid": int(n_valid), "n_test": int(n_test), "batch": int(args.batch),
        },
    }

    # 7. write
    write_h5(out_path, volts_norm, unit_par,
             splits=(n_train, n_valid, n_test), meta=meta)

    # cleanup per-stim resume caches now that the full pack is written
    for ci in range(len(STIM_LIST)):
        cf = out_dir / f".gencache_{args.cell_name}_stim{ci}.npy"
        if cf.exists():
            cf.unlink()
            print(f"[gen]   removed resume cache {cf.name}", flush=True)

    # final sanity
    with h5py.File(out_path, "r") as f:
        print("[gen] final H5 keys:", list(f.keys()), flush=True)
        print("[gen]   train_volts_norm:", f["train_volts_norm"].shape,
              f["train_volts_norm"].dtype, flush=True)
        print("[gen]   train_unit_par:  ", f["train_unit_par"].shape,
              f["train_unit_par"].dtype, flush=True)
        _m = json.loads(f["meta.JSON"][0])
        print("[gen]   meta num_probs=%s probe_names=%s stim_names=%s num_stims=%s" %
              (_m["num_probs"], _m["probe_names"], _m["stim_names"], _m["num_stims"]),
              flush=True)


if __name__ == "__main__":
    main()
