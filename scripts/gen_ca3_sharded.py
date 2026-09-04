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


def _fam_assign(n, K, offset):
    """--per-sample-stims: deterministic family index per sample.

    offset + i % (K-offset): round-robin over stims[offset:], exact n/(K-offset)
    per family.  `offset` lets the stem list carry unused leading entries (e.g.
    Roy100 at index 0) so the label indices match the exp ca3ft packs, whose
    family order is Roy100,Roy500,...,Roy2000.  Same call in worker and merge.
    """
    return (offset + np.arange(n) % (K - offset)).astype(np.int64)


def _vary_indices(cell, vary):
    """--vary keys -> (varied_idx, pinned_idx). None -> all varied."""
    P = len(cell.PARAM_KEYS)
    if not vary:
        return list(range(P)), []
    vidx = []
    for k in vary:
        if k not in cell.PARAM_KEYS:
            raise ValueError(f"--vary {k!r} not in PARAM_KEYS {cell.PARAM_KEYS}")
        vidx.append(cell.PARAM_KEYS.index(k))
    pin = [i for i in range(P) if i not in vidx]
    return vidx, pin


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
    _, pin = _vary_indices(cell, args.vary)
    if pin:                       # pinned params sit at unit 0 = cell default
        unit_all[:, pin] = 0.0
    lo, hi = _slice(rank, world, args.n)
    phys = unit_to_phys_np(unit_all[lo:hi].astype(np.float64), centers, logspans)
    print(f"[rank {rank}/{world}] GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"slice [{lo}:{hi}] ({hi-lo} samples) stims={args.stims}", flush=True)

    if args.per_sample_stims:
        # Each sample is simulated under ITS OWN family's stim (round-robin
        # assignment); the shard holds a single volts channel.  Chunks are
        # padded to --batch so XLA compiles ONE shape per family.
        fam = _fam_assign(args.n, len(args.stims), args.fam_offset)[lo:hi]
        out = None
        for fi in sorted(set(fam.tolist())):
            sname = args.stims[fi]
            idx = np.flatnonzero(fam == fi)
            t0 = time.time()
            for b0 in range(0, len(idx), args.batch):
                sel = idx[b0:b0 + args.batch]
                pbnp = phys[sel]
                ng = pbnp.shape[0]
                if ng < args.batch:
                    pbnp = np.concatenate(
                        [pbnp, np.repeat(pbnp[:1], args.batch - ng, axis=0)], axis=0)
                pb = torch.from_numpy(pbnp).double()
                with torch.no_grad():
                    v = JaxleyBridge.simulate_batch(pb, args.source_cell, stim_name=sname)
                v = v.detach().cpu().numpy()      # (batch, n_rec, T)
                if v.shape[1] != 1:
                    raise RuntimeError(f"expected soma-only, got n_rec={v.shape[1]}")
                vv = v[:ng, 0, args.trim_bins:] if args.trim_bins else v[:ng, 0, :]
                if out is None:
                    out = np.zeros((hi - lo, vv.shape[-1]), dtype=np.float32)
                out[sel] = vv
            print(f"[rank {rank}] fam {fi} ({sname}): {len(idx)} samples in "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
        volts = out[..., np.newaxis]              # (n_slice, T, 1)
    else:
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
                vv = v[:, 0, args.trim_bins:] if args.trim_bins else v[:, 0, :]
                if out is None:
                    out = np.zeros((hi - lo, vv.shape[-1]), dtype=np.float32)
                out[i0:i1] = vv
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
    vidx, pin = _vary_indices(cell, args.vary)
    if pin:
        unit_par[:, pin] = 0.0     # match the worker's pinning
    perm = rng.permutation(args.n)
    volts_norm = volts_norm[perm]
    unit_par = unit_par[perm]

    n = args.n
    n_train = n * 80 // 100
    n_valid = n * 10 // 100
    n_test = n - n_train - n_valid
    centers, logspans = phys_par_range_to_arrays(ppr)
    phys_par = unit_to_phys_np(unit_par.astype(np.float64), centers, logspans).astype(np.float32)
    # With --vary, the pack's label/param space covers ONLY the varied channels
    # (outputSize follows unit_par width; pinned params are a known constant).
    par_names_out = [cell.PARAM_KEYS[i] for i in vidx]
    ppr_out = [ppr[i] for i in vidx]
    unit_par = unit_par[:, vidx]
    phys_par = phys_par[:, vidx]

    if args.per_sample_stims:
        # Probe 1 = the delivered stimulus (ideal per-family CSV, nA / fixed
        # scale) on the SAME trimmed time grid as the volts; label col 0 = the
        # family index (consumed by voltage_loss.stim_from_label, exp-pack
        # convention).  Norm constant shared with format_royexp_for_ML
        # --stimChannel fixed via toolbox.jaxley_utils.STIM_NORM_SCALE_NA.
        from toolbox.jaxley_utils import normalize_stim_fixed
        fam_perm = _fam_assign(args.n, len(args.stims), args.fam_offset)[perm]
        T = volts_norm.shape[1]
        stim_dir = Path(cell._STIM_DIR)
        waves = np.zeros((len(args.stims), T), dtype=np.float32)
        for fi in sorted(set(fam_perm.tolist())):
            w = np.loadtxt(str(stim_dir / f"{args.stims[fi]}.csv")).astype(np.float32)
            w = np.concatenate([w, w[-1:]])       # stim grid 5000 -> sim grid 5001
            assert len(w) >= args.trim_bins + T, (len(w), args.trim_bins, T)
            waves[fi] = w[args.trim_bins:args.trim_bins + T]
        stim_chan = normalize_stim_fixed(waves)[fam_perm]          # (N, T)
        volts4 = np.concatenate(
            [volts_norm, stim_chan[..., np.newaxis]], axis=2)[..., np.newaxis]
        S = 2                                     # (N, T, 2, 1)
        unit_par = np.concatenate(
            [fam_perm[:, None].astype(np.float32), unit_par], axis=1)
        phys_par = np.concatenate(
            [fam_perm[:, None].astype(np.float32), phys_par], axis=1)
        par_names_out = ["stim_family_idx"] + par_names_out
        ppr_out = [[1.0, 0.0, "idx"]] + ppr_out
        probe_names = ["soma", "stimulus"]
    else:
        volts4 = volts_norm[..., np.newaxis]      # (N, T, S, 1)
        probe_names = (["soma"] if S == 1 else list(args.stims))
    out_path = Path(args.out) / f"{args.cell_name}.mlPack1.h5"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    meta = {
        "cell_name": args.cell_name, "source_cell": args.source_cell,
        "num_phys_par": len(par_names_out), "num_varied_phys_par": len(par_names_out),
        "num_probs": int(S), "num_stims": 1, "num_time_bins": int(T),
        "parName": par_names_out, "probe_names": probe_names,
        "stim_names": list(args.stims),
        "timeAxis": {"step": float(cell._DT_STIM), "unit": "(ms)"},
        "phys_par_range": ppr_out,
        "input_meta": {"parName": par_names_out, "phys_par_range": ppr_out},
        "simu_info": {
            "generator": "scripts/gen_ca3_sharded.py",
            "stim_names": list(args.stims),
            "joint_per_sample_channels": bool(S > 1),
            "log_halfspan": _LOG_HALFSPAN, "fp64": True, "seed": args.seed,
            "cell_spec": {"_DT": cell._DT, "_DT_STIM": cell._DT_STIM,
                          "_T_MAX": cell._T_MAX, "_V_INIT": cell._V_INIT},
            "probe_names": probe_names, "stim_names_multi": list(args.stims),
            "world_shards": world,
            "varied_params": par_names_out,
            "pinned_params": [cell.PARAM_KEYS[i] for i in pin],
            "full_param_keys": list(cell.PARAM_KEYS),
            "param_subset_indices": vidx,
            "trim_bins": int(args.trim_bins),
        },
        "pack_info": {"n_total": n, "n_train": n_train, "n_valid": n_valid,
                      "n_test": n_test, "batch": args.batch},
    }
    if args.per_sample_stims:
        from toolbox.jaxley_utils import STIM_NORM_SCALE_NA
        meta["stim_from_label"] = {
            "label_col": 0, "stim_names": list(args.stims),
            "stim_family_order": [s.split("_")[0] for s in args.stims],
            "fam_offset": args.fam_offset,
            "stim_channel": {"probe": 1, "scale_nA": STIM_NORM_SCALE_NA,
                             "source": "ideal per-family CSV on the trimmed sim grid"}}
        meta["simu_info"]["per_sample_stims"] = True
        meta["simu_info"]["joint_per_sample_channels"] = False   # probe1 is the stim, not another sim
        meta["simu_info"]["fam_offset"] = args.fam_offset
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
    ap.add_argument("--vary", default=None,
                    help="comma-separated PARAM_KEYS to vary; the rest are pinned "
                         "at unit 0 (= cell default). unit_par/parName/phys_par_range "
                         "in the pack then cover ONLY the varied params.")
    ap.add_argument("--per-sample-stims", action="store_true",
                    help="each sample gets ONE stim from --stims (round-robin "
                         "over stims[fam-offset:]); pack = 2 probes (volts, "
                         "fixed-scale stim waveform) + family idx in label col 0")
    ap.add_argument("--fam-offset", type=int, default=0,
                    help="(--per-sample-stims) first stim index actually used; "
                         "leading stems are indexed but never assigned, so label "
                         "indices can match the exp packs' Roy100..Roy2000 order")
    ap.add_argument("--trim-bins", type=int, default=0,
                    help="drop the first N time bins of each simulated trace before "
                         "packing (e.g. 1000 = a 100 ms holding preamble), so the "
                         "pack window matches a recording that starts at stim onset")
    args = ap.parse_args()
    args.stims = [s for s in args.stims.split(",") if s]
    args.vary = [s for s in args.vary.split(",") if s] if args.vary else None
    if args.world is None:
        args.world = int(os.environ.get("SLURM_NTASKS", 1))
    if args.merge:
        merge(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
