#!/usr/bin/env python3
"""Sharded multi-GPU generation of a CA3 mlPack1.h5 (single- or multi-stim).

Splits N samples across SLURM ranks (4 nodes x 4 GPUs = 16). Two phases in one
job:

  WORKER (srun -n<world>):  each rank pins to its local GPU, simulates its
    contiguous slice of the N unit-param draws under every --stim, and writes the
    RAW (un-normalized) volts slice to <shard-dir>/shard_RRR.npy  (N_slice, T, S).

  MERGE  (srun -n1 --merge): concatenates the shards in rank order, applies the
    SAME fixed-scale normalization the single-process generators use
    (toolbox.jaxley_utils.normalize_volts_fixed -> matches HybridLoss), shuffles +
    splits 80/10/10, and writes the mlPack1 H5 + meta.

Layout matches scripts/gen_ball_and_stick_data.py (1 stim -> num_probs=1) and
scripts/gen_multistim_data.py (K stims -> num_probs=K on the PROBE axis,
num_stims=1) so the existing dataloader/HybridLoss paths work unchanged.

Determinism: the full unit_par array is drawn once from default_rng(seed) and is
reproduced identically in both worker and merge, so shard boundaries and the
final shuffle match a single-process run.
"""
import os, sys, json, time, argparse
from pathlib import Path

_IS_MERGE = "--merge" in sys.argv
# Pin each worker rank to its local GPU BEFORE importing jax (merge needs no GPU).
_LOCALID = os.environ.get("SLURM_LOCALID")
if not _IS_MERGE and _LOCALID is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _LOCALID
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("JAX_ENABLE_X64", "true")

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import h5py

# Half-decade each way by default (center*10^±0.5 ≈ ×3.16). Override with
# NEUINV_LOG_HALFSPAN to widen/narrow the sampled conductance range, e.g. 1.0
# = ±full decade (×10 each way). Recorded in pack meta -> model's phys_par_range.
_LOG_HALFSPAN = float(os.environ.get("NEUINV_LOG_HALFSPAN", 0.5))


def _phys_range(param_keys, defaults):
    return [[float(defaults[k]), float(_LOG_HALFSPAN), "S/cm^2"] for k in param_keys]


def _load_cell(name):
    import importlib
    return importlib.import_module(f"toolbox.jaxley_cells.{name}")


def _slice(rank, world, n):
    lo = rank * n // world
    hi = (rank + 1) * n // world
    return lo, hi


def _draw_unit_par(seed, n, P):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, P)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
def worker(args):
    import torch
    from toolbox import JaxleyBridge
    from toolbox.jaxley_utils import phys_par_range_to_arrays, unit_to_phys_np

    rank  = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    cell = _load_cell(args.source_cell)
    if args.t_max is not None:
        cell._T_MAX = float(args.t_max)   # set before first simulate -> handle uses it
        JaxleyBridge.clear_cache()
    P = len(cell.PARAM_KEYS)
    ppr = _phys_range(cell.PARAM_KEYS, cell._DEFAULTS)
    centers, logspans = phys_par_range_to_arrays(ppr)

    unit_all = _draw_unit_par(args.seed, args.n, P)
    lo, hi = _slice(rank, world, args.n)
    phys = unit_to_phys_np(unit_all[lo:hi].astype(np.float64), centers, logspans)
    print(f"[rank {rank}/{world}] GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"slice [{lo}:{hi}] ({hi-lo} samples) stims={args.stims}", flush=True)

    chans = []
    for sname in args.stims:
        out = None
        nb = int(np.ceil((hi - lo) / args.batch))
        t0 = time.time()
        for b in range(nb):
            i0 = b * args.batch
            i1 = min(hi - lo, i0 + args.batch)
            pb = torch.from_numpy(phys[i0:i1]).double()
            with torch.no_grad():
                v = JaxleyBridge.simulate_batch(pb, args.source_cell, stim_name=sname)
            v = v.detach().cpu().numpy()          # (b, n_rec, T)
            if v.shape[1] != 1:
                raise RuntimeError(f"expected soma-only, got n_rec={v.shape[1]}")
            if out is None:
                out = np.zeros((hi - lo, v.shape[-1]), dtype=np.float32)
            out[i0:i1] = v[:, 0, :]
        chans.append(out)
        print(f"[rank {rank}] stim {sname}: {out.shape} in {(time.time()-t0)/60:.1f} min",
              flush=True)
    volts = np.stack(chans, axis=2)               # (n_slice, T, S)
    if not np.isfinite(volts).all():
        raise RuntimeError(f"[rank {rank}] non-finite volts")
    shard = Path(args.shard_dir) / f"shard_{rank:03d}.npy"
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.save(shard, volts)
    print(f"[rank {rank}] wrote {shard} {volts.shape}", flush=True)


# ─────────────────────────────────────────────────────────────────────────
def merge(args):
    from toolbox.jaxley_utils import (normalize_volts_fixed, phys_par_range_to_arrays,
                                       unit_to_phys_np)
    cell = _load_cell(args.source_cell)
    if args.t_max is not None:
        cell._T_MAX = float(args.t_max)   # recorded in meta so training's t_max matches
    P = len(cell.PARAM_KEYS)
    ppr = _phys_range(cell.PARAM_KEYS, cell._DEFAULTS)
    world = args.world

    parts = []
    for r in range(world):
        f = Path(args.shard_dir) / f"shard_{r:03d}.npy"
        if not f.exists():
            raise FileNotFoundError(f"missing shard {f}")
        parts.append(np.load(f))
    volts = np.concatenate(parts, axis=0)         # (N, T, S)
    assert volts.shape[0] == args.n, f"got {volts.shape[0]} rows, expected N={args.n}"
    S, T = volts.shape[2], volts.shape[1]
    print(f"[merge] concatenated {volts.shape} from {world} shards", flush=True)

    volts_norm = normalize_volts_fixed(volts).astype(np.float32)
    print(f"[merge] fixed-norm: global mean={volts_norm.mean():.3f} "
          f"std={volts_norm.std():.3f}", flush=True)

    # Reproduce the single-process rng stream: draw unit_par (advances state),
    # then permute — identical ordering to a non-sharded run.
    rng = np.random.default_rng(args.seed)
    unit_par = rng.uniform(-1.0, 1.0, size=(args.n, P)).astype(np.float32)
    perm = rng.permutation(args.n)
    volts_norm = volts_norm[perm]
    unit_par = unit_par[perm]

    n = args.n
    n_train = n * 80 // 100
    n_valid = n * 10 // 100
    n_test = n - n_train - n_valid
    centers, logspans = phys_par_range_to_arrays(ppr)
    phys_par = unit_to_phys_np(unit_par.astype(np.float64), centers, logspans).astype(np.float32)
    volts4 = volts_norm[..., np.newaxis]          # (N, T, S, 1)

    probe_names = (["soma"] if S == 1 else list(args.stims))
    out_path = Path(args.out) / f"{args.cell_name}.mlPack1.h5"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    meta = {
        "cell_name": args.cell_name, "source_cell": args.source_cell,
        "num_phys_par": P, "num_varied_phys_par": P,
        "num_probs": int(S), "num_stims": 1, "num_time_bins": int(T),
        "parName": list(cell.PARAM_KEYS), "probe_names": probe_names,
        "stim_names": list(args.stims),
        "timeAxis": {"step": float(cell._DT_STIM), "unit": "(ms)"},
        "phys_par_range": ppr,
        "input_meta": {"parName": list(cell.PARAM_KEYS), "phys_par_range": ppr},
        "simu_info": {
            "generator": "scripts/gen_ca3_sharded.py",
            "stim_names": list(args.stims),
            "joint_per_sample_channels": bool(S > 1),
            "log_halfspan": _LOG_HALFSPAN, "fp64": True, "seed": args.seed,
            "cell_spec": {"_DT": cell._DT, "_DT_STIM": cell._DT_STIM,
                          "_T_MAX": cell._T_MAX, "_V_INIT": cell._V_INIT},
            "probe_names": probe_names, "stim_names_multi": list(args.stims),
            "world_shards": world,
        },
        "pack_info": {"n_total": n, "n_train": n_train, "n_valid": n_valid,
                      "n_test": n_test, "batch": args.batch},
    }
    sl = {"train": slice(0, n_train), "valid": slice(n_train, n_train + n_valid),
          "test": slice(n_train + n_valid, n)}
    print(f"[merge] writing {out_path}  split train/valid/test="
          f"{n_train}/{n_valid}/{n_test}  num_probs={S}", flush=True)
    with h5py.File(out_path, "w") as f:
        for dom, s in sl.items():
            f.create_dataset(f"{dom}_volts_norm", data=volts4[s].astype(np.float16),
                             compression="gzip", compression_opts=4)
            f.create_dataset(f"{dom}_unit_par", data=unit_par[s].astype(np.float32))
            f.create_dataset(f"{dom}_phys_par", data=phys_par[s].astype(np.float32))
        for sname in args.stims:
            f.create_dataset(sname, data=np.zeros((0,), dtype=np.float32))
        dt = h5py.special_dtype(vlen=str)
        ds = f.create_dataset("meta.JSON", (1,), dtype=dt)
        ds[0] = json.dumps(meta)
    print(f"[merge] wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)", flush=True)


# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-cell", default="ca3_pyramidal")
    ap.add_argument("--cell-name", required=True, help="H5 filename stem + meta.cell_name")
    ap.add_argument("--out", required=True, help="output dir for the final H5")
    ap.add_argument("--shard-dir", required=True, help="scratch dir for per-rank shards")
    ap.add_argument("--stims", required=True,
                    help="comma-separated stim names (1 -> single-stim, K -> K probe channels)")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-max", type=float, default=None,
                    help="override the cell's _T_MAX (ms). Needed for interpolated "
                         "stims: they are 4000 samples @0.1ms = 400 ms, not the "
                         "cell default 500 ms. Set BEFORE the first sim so the "
                         "cached jaxley handle uses it; recorded in meta.")
    ap.add_argument("--merge", action="store_true", help="run the merge phase")
    ap.add_argument("--world", type=int, default=None,
                    help="(merge) number of shards to expect; default SLURM_NTASKS")
    args = ap.parse_args()
    args.stims = [s for s in args.stims.split(",") if s]
    if args.world is None:
        args.world = int(os.environ.get("SLURM_NTASKS", 1))
    if args.merge:
        merge(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
