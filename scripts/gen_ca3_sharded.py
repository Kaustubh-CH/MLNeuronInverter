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

--per-sample-stims instead gives each sample ONE stim from --stims (round-robin
over stims[--fam-offset:]); the pack then has 2 probes (soma volts, fixed-scale
stim waveform) and the family index in label column 0 (voltage_loss.stim_from_label).

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


def _phys_range(cell_mod):
    from toolbox.jaxley_utils import build_phys_par_range
    return build_phys_par_range(cell_mod, _LOG_HALFSPAN)


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


def _nominal_stim_on_grid(cell, sname, trim, T):
    """--per-sample-stims probe 1: the stim CSV (nA, UNscaled -- the same units the
    exp packs' --stimChannel carries) on the pack's output grid, built exactly as
    JaxleyBridge._build_handle builds the delivered stim (upsample to _DT, decimate
    by downsample_step), then trimmed like the volts."""
    from toolbox.jaxley_utils import load_stim_csv, upsample_stim, downsample_step
    w = load_stim_csv(Path(cell._STIM_DIR) / f"{sname}.csv")
    w = upsample_stim(w, cell._DT_STIM, cell._DT, cell._T_MAX)
    w = w[::max(1, downsample_step(cell._DT, cell._DT_STIM))]
    # jaxley records t=0..t_max inclusive -> one more bin than the stim grid
    # (e.g. 5000 -> 5001): hold the last value, never more than one bin.
    short = trim + T - len(w)
    assert short <= 1, (sname, len(w), trim, T)
    if short > 0:
        w = np.concatenate([w, np.repeat(w[-1:], short)])
    return w[trim:trim + T].astype(np.float32)


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
    from toolbox.jaxley_utils import phys_par_range_to_arrays, phys_par_range_linear_mask, unit_to_phys_np

    rank  = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    cell = _load_cell(args.source_cell)
    if args.t_max is not None:
        cell._T_MAX = float(args.t_max)   # set before first simulate -> handle uses it
    if args.dt is not None:
        cell._DT = float(args.dt)         # coarser solver step (must match training's sim_dt_override)
    JaxleyBridge.clear_cache()
    if args.per_sample_stims and len(getattr(cell, "PROBE_NAMES", None) or ["soma"]) > 1:
        raise RuntimeError(f"--per-sample-stims needs a soma-only cell; {args.source_cell} "
                           f"records {len(cell.PROBE_NAMES)} probes {list(cell.PROBE_NAMES)}")
    P = len(cell.PARAM_KEYS)
    ppr = _phys_range(cell)
    centers, logspans = phys_par_range_to_arrays(ppr); linear = phys_par_range_linear_mask(ppr)

    unit_all = _draw_unit_par(args.seed, args.n, P)
    _, pin = _vary_indices(cell, args.vary)
    if pin:                       # pinned params sit at unit 0 (= cell default for log rows)
        unit_all[:, pin] = 0.0
    lo, hi = _slice(rank, world, args.n)
    phys = unit_to_phys_np(unit_all[lo:hi].astype(np.float64), centers, logspans, linear)
    if rank == 0:
        print("[gen] phys_par_range:", "; ".join(f"{k}: {r[0]:.4g} +-{r[1]:.3g} {r[3] if len(r) > 3 else 'log10'}" for k, r in zip(cell.PARAM_KEYS, ppr)), flush=True)
    print(f"[rank {rank}/{world}] GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"slice [{lo}:{hi}] ({hi-lo} samples) stims={args.stims}", flush=True)

    import jax.numpy as jnp

    def _sim_chunk(handle, pnp):
        # Forward-only on the cached jitted vmap (no jax.vjp graph): the
        # training bridge's VJP OOMs for the ~1000-2000-comp L5 cells at
        # batch 32-64.  Pad the chunk to --batch so XLA compiles one shape.
        pg = jnp.asarray(pnp); ng = pg.shape[0]
        if ng < args.batch:
            pg = jnp.concatenate([pg, jnp.broadcast_to(pg[:1], (args.batch - ng,) + pg.shape[1:])], axis=0)
        v = np.asarray(handle.simulate_batch(pg))[:ng]          # (ng, n_rec, T)
        return v[..., args.trim_bins:] if args.trim_bins else v

    if args.per_sample_stims:
        # Each sample is simulated under ITS OWN family's stim (round-robin
        # assignment); the shard holds a single volts channel (the merge adds the
        # stim waveform as probe 1).  One compiled handle/shape per family.
        fam = _fam_assign(args.n, len(args.stims), args.fam_offset)[lo:hi]
        out = None
        for fi in sorted(set(fam.tolist())):
            sname = args.stims[fi]
            handle = JaxleyBridge.get_handle(args.source_cell, sname)
            idx = np.flatnonzero(fam == fi)
            t0 = time.time()
            for b0 in range(0, len(idx), args.batch):
                sel = idx[b0:b0 + args.batch]
                v = _sim_chunk(handle, phys[sel])
                if v.shape[1] != 1:
                    raise RuntimeError(f"--per-sample-stims needs a soma-only cell, "
                                       f"{args.source_cell} records n_rec={v.shape[1]}")
                if out is None:
                    out = np.zeros((hi - lo, v.shape[-1], 1), dtype=np.float32)
                out[sel] = np.moveaxis(v, 1, 2)   # (b, T, 1)
            print(f"[rank {rank}] fam {fi} ({sname}): {len(idx)} samples in "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
        volts = out                               # (n_slice, T, 1)
    else:
        chans = []
        for sname in args.stims:
            handle = JaxleyBridge.get_handle(args.source_cell, sname)
            out = None
            nb = int(np.ceil((hi - lo) / args.batch))
            t0 = time.time()
            for b in range(nb):
                i0 = b * args.batch
                i1 = min(hi - lo, i0 + args.batch)
                v = _sim_chunk(handle, phys[i0:i1])             # (b, n_rec, T)
                if v.shape[1] != 1 and len(args.stims) > 1:
                    raise RuntimeError(f"multi-stim packs need a soma-only cell, got n_rec={v.shape[1]}")
                if out is None:
                    out = np.zeros((hi - lo, v.shape[-1], v.shape[1]), dtype=np.float32)
                out[i0:i1] = np.moveaxis(v, 1, 2)     # (b, T, n_rec)
            chans.append(out)
            print(f"[rank {rank}] stim {sname}: {out.shape} in {(time.time()-t0)/60:.1f} min",
                  flush=True)
        # Channel axis: K stims of a soma-only cell -> (n, T, K); one stim of a
        # P-probe cell -> (n, T, P).  Either way the dataloader sees it as the
        # PROBE axis with num_stims=1.
        volts = np.concatenate(chans, axis=2)     # (n_slice, T, S*n_rec)
    if not np.isfinite(volts).all():
        raise RuntimeError(f"[rank {rank}] non-finite volts")
    shard = Path(args.shard_dir) / f"shard_{rank:03d}.npy"
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.save(shard, volts)
    print(f"[rank {rank}] wrote {shard} {volts.shape}", flush=True)


# ─────────────────────────────────────────────────────────────────────────
def merge(args):
    from toolbox.jaxley_utils import (normalize_volts_fixed, phys_par_range_to_arrays,
                                       phys_par_range_linear_mask, unit_to_phys_np)
    cell = _load_cell(args.source_cell)
    if args.t_max is not None:
        cell._T_MAX = float(args.t_max)   # recorded in meta so training's t_max matches
    if args.dt is not None:
        cell._DT = float(args.dt)         # recorded in meta (cell_spec._DT)
    P = len(cell.PARAM_KEYS)
    ppr = _phys_range(cell)
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
    if not np.isfinite(volts_norm).all():
        raise RuntimeError("[merge] non-finite volts after concatenation")

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
    phys_par = unit_to_phys_np(unit_par.astype(np.float64), centers, logspans,
                               phys_par_range_linear_mask(ppr)).astype(np.float32)
    # With --vary the pack's label/param space covers ONLY the varied params; the
    # full 19-row range + subset indices go to simu_info so HybridLoss / evaluate_voltage
    # can expand the CNN's subset prediction back to the full cell vector.
    varied_names = [cell.PARAM_KEYS[i] for i in vidx]
    par_names_out = list(varied_names)
    ppr_out = [ppr[i] for i in vidx]
    unit_par = unit_par[:, vidx]
    phys_par = phys_par[:, vidx]

    if args.per_sample_stims:
        # Probe 1 = the stimulus (nominal per-family CSV in nA, fixed scale) on the
        # SAME trimmed output grid as the volts; label col 0 = the family index
        # (consumed by voltage_loss.stim_from_label, exp-pack convention).  Norm
        # constant shared with format_royexp_for_ML --stimChannel fixed via
        # toolbox.jaxley_utils.STIM_NORM_SCALE_NA.
        from toolbox.jaxley_utils import normalize_stim_fixed
        if S != 1:
            raise RuntimeError(f"--per-sample-stims expects 1 volts channel per shard, got S={S}")
        fam_perm = _fam_assign(args.n, len(args.stims), args.fam_offset)[perm]
        waves = np.zeros((len(args.stims), T), dtype=np.float32)
        for fi in sorted(set(fam_perm.tolist())):
            waves[fi] = _nominal_stim_on_grid(cell, args.stims[fi], args.trim_bins, T)
        stim_chan = normalize_stim_fixed(waves)[fam_perm]           # (N, T)
        volts4 = np.concatenate(
            [volts_norm, stim_chan[..., np.newaxis]], axis=2)[..., np.newaxis]
        S = 2                                     # (N, T, 2, 1)
        unit_par = np.concatenate([fam_perm[:, None].astype(np.float32), unit_par], axis=1)
        phys_par = np.concatenate([fam_perm[:, None].astype(np.float32), phys_par], axis=1)
        par_names_out = ["stim_family_idx"] + par_names_out
        ppr_out = [[1.0, 0.0, "idx"]] + ppr_out
        probe_names = ["soma", "stimulus"]
    else:
        volts4 = volts_norm[..., np.newaxis]      # (N, T, S, 1)
        if len(args.stims) > 1:
            probe_names = list(args.stims)        # K stims as channels
        else:
            probe_names = list(getattr(cell, "PROBE_NAMES", None) or (["soma"] if S == 1 else [f"probe{i}" for i in range(S)]))
    assert len(probe_names) == S, f"probe_names {probe_names} vs S={S}"
    out_path = Path(args.out) / f"{args.cell_name}.mlPack1.h5"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    meta = {
        "cell_name": args.cell_name, "source_cell": args.source_cell,
        "num_phys_par": len(par_names_out), "num_varied_phys_par": len(par_names_out),
        "num_probs": int(S), "num_stims": 1, "num_time_bins": int(T),
        "parName": par_names_out, "probe_names": probe_names,
        "stim_names": list(args.stims),
        "timeAxis": {"step": float(max(cell._DT, cell._DT_STIM)), "unit": "(ms)"},
        "phys_par_range": ppr_out,
        "input_meta": {"parName": par_names_out, "phys_par_range": ppr_out},
        "simu_info": {
            "generator": "scripts/gen_ca3_sharded.py",
            "stim_names": list(args.stims),
            "joint_per_sample_channels": bool(S > 1),
            # precision the traces were generated at (run_ca3_gen.sh GEN_X64; default fp64)
            "log_halfspan": _LOG_HALFSPAN, "seed": args.seed,
            "fp64": str(os.environ.get("JAX_ENABLE_X64", "true")).lower() in ("1", "true", "yes"),
            "cell_spec": {"_DT": cell._DT, "_DT_STIM": cell._DT_STIM,
                          "_T_MAX": cell._T_MAX, "_V_INIT": cell._V_INIT,
                          "_STIM_SCALE": float(getattr(cell, "_STIM_SCALE", 1.0)),
                          "_NCOMP": getattr(cell, "_NCOMP", None)},
            "probe_names": probe_names, "stim_names_multi": list(args.stims),
            "world_shards": world,
            "varied_params": varied_names,
            "pinned_params": [cell.PARAM_KEYS[i] for i in pin],
            "full_param_keys": list(cell.PARAM_KEYS),
            "full_phys_par_range": ppr,
            "param_subset_indices": vidx,
            # bins dropped from the start of every trace, on the OUTPUT grid
            # (timeAxis.step); training's voltage_loss.sim_t_skip_ms must match
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
                             "stim_scale_in_sim": float(getattr(cell, "_STIM_SCALE", 1.0)),
                             "source": "nominal (unscaled) per-family CSV on the trimmed output grid"}}
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
    ap.add_argument("--dt", type=float, default=None,
                    help="override the cell's solver step _DT (ms), e.g. 0.2; the pack is then "
                         "stored on that grid (T = t_max/dt) and training must use the same "
                         "voltage_loss.sim_dt_override.")
    ap.add_argument("--vary", default=None,
                    help="comma-separated PARAM_KEYS to vary; the rest are pinned at unit 0 "
                         "(= cell default). unit_par/parName/phys_par_range in the pack then "
                         "cover ONLY the varied params (full range kept in simu_info).")
    ap.add_argument("--per-sample-stims", action="store_true",
                    help="each sample gets ONE stim from --stims (round-robin "
                         "over stims[fam-offset:]); pack = 2 probes (volts, "
                         "fixed-scale stim waveform) + family idx in label col 0. "
                         "Soma-only cells only.")
    ap.add_argument("--fam-offset", type=int, default=0,
                    help="(--per-sample-stims) first stim index actually used; "
                         "leading stems are indexed but never assigned, so label "
                         "indices can match the exp packs' Roy100..Roy2000 order")
    ap.add_argument("--trim-bins", type=int, default=0,
                    help="drop the first N OUTPUT-grid time bins of each simulated "
                         "trace before packing (e.g. 1000 = a 100 ms holding preamble "
                         "at 0.1 ms), so the pack window matches a recording that "
                         "starts at stim onset")
    ap.add_argument("--merge", action="store_true", help="run the merge phase")
    ap.add_argument("--world", type=int, default=None,
                    help="(merge) number of shards to expect; default SLURM_NTASKS")
    args = ap.parse_args()
    args.stims = [s for s in args.stims.split(",") if s]
    args.vary = [s for s in args.vary.split(",") if s] if args.vary else None
    if args.per_sample_stims and not 0 <= args.fam_offset < len(args.stims):
        ap.error(f"--fam-offset {args.fam_offset} must be in [0, {len(args.stims)}) "
                 f"for {len(args.stims)} --stims")
    if args.trim_bins < 0:
        ap.error("--trim-bins must be >= 0")
    if args.world is None:
        args.world = int(os.environ.get("SLURM_NTASKS", 1))
    if args.merge:
        merge(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
