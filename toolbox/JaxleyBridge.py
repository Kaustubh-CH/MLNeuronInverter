"""Torch <-> Jaxley bridge.

Exposes:

    simulate_batch(params_phys: Tensor[B, P],
                   stim:        Tensor[T_sim] | None,
                   cell_name:   str,
                   ) -> Tensor[B, T_out]

`params_phys` is a batch of physical parameter vectors (S/cm^2 etc.).  The
bridge vmaps a jitted jaxley simulation over the batch dimension and is
differentiable w.r.t. `params_phys`.

Efficiency
----------
* The jaxley cell is built **once per process** per cell name, then cached
  (`_CELL_CACHE`).  The jitted+vmap'd simulate function is captured at
  build time.
* On first forward we also jit-compile a `jax.vjp` closure; subsequent
  backwards reuse it (no recompilation).
* DLPack is used to roundtrip tensors zero-copy when both frameworks live
  on the same device.  If jax and torch disagree on device we fall back to
  numpy.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from . import jaxley_cells
from . import jaxley_utils


# ═════════════════════════════════════════════════════════════════════════
# Per-cell cached handle
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class _CellHandle:
    cell_name:        str
    spec:             "jaxley_cells.CellSpec"
    cell:             object                     # jx.Cell
    default_params:   list                       # list of dicts (one per trainable)
    simulate_batch:   Callable                   # jit(vmap(simulate_one))
    downsample_step:  int
    sim_len:          int                        # length at internal dt
    out_len:          int                        # length after downsampling
    v_init:           float
    stim_scale:       float = 1.0                # multiplier applied to the stim CSV
    out_dt:           float = 0.1                # ms between output samples (dt * downsample_step)


_CELL_CACHE: dict = {}


def _build_handle(cell_name: str, stim_name: Optional[str] = None,
                  checkpoint_lengths: Optional[Tuple[int, ...]] = None,
                  solver: str = "bwd_euler",
                  stim_scale: Optional[float] = None) -> _CellHandle:
    """Build + compile the cell once.  Expensive; called on first use.

    `stim_scale`: multiplier on the stim waveform.  None -> the cell spec's
    own `stim_scale` (the production value); an explicit float overrides it
    (used by the stim-scale sweeps in scripts/stim_scale_scan.py).

    `checkpoint_lengths`: if given, passed to `jx.integrate` as
    `checkpoint_lengths=list(checkpoint_lengths)` to enable gradient
    checkpointing across the time axis (jaxley 0.13+).  Two-level form
    `(outer, inner)` makes the backward chain only `inner` stiff steps long
    per VJP segment, which keeps fp32 stable at large t_max.
    """
    import jax
    import jax.numpy as jnp
    import jaxley as jx

    spec = jaxley_cells.get(cell_name)
    cell, entry_to_cnn_idx = spec.build_fn()

    # Snapshot the structure of the trainable list *before* any simulate call.
    default_params = cell.get_parameters()
    if len(default_params) != len(entry_to_cnn_idx):
        raise RuntimeError(
            f"{cell_name}: build_fn returned {len(entry_to_cnn_idx)} CNN indices "
            f"but cell has {len(default_params)} trainable entries. Spec bug."
        )

    # Stimulus waveform — fixed per handle.  Changing stims means a new
    # handle (new jit).  That's intentional: stim length + content change
    # the computation graph, so caching per (cell, stim) is the right key.
    stim_name = stim_name or spec.default_stim_name
    stim_path = Path(spec.stim_dir) / f"{stim_name}.csv"
    stim_csv  = jaxley_utils.load_stim_csv(stim_path)
    scale     = float(getattr(spec, "stim_scale", 1.0)) if stim_scale is None else float(stim_scale)
    if scale != 1.0:
        stim_csv = (stim_csv * scale).astype(np.float32)
    stim_up   = jaxley_utils.upsample_stim(stim_csv, spec.dt_stim, spec.dt, spec.t_max)
    # dt_sim <= dt_stim: decimate back to the stim grid; dt_sim > dt_stim (coarse
    # solver step, e.g. 0.2 ms): keep every step -> output grid is dt_sim.
    step      = max(1, jaxley_utils.downsample_step(spec.dt, spec.dt_stim))

    # `data_stimulate` expects (n_injectors, T_sim).
    stim_jnp  = jnp.asarray(stim_up[np.newaxis, :])
    data_stim = spec.stim_attach_fn(cell, stim_jnp)

    # Build a pytree-replaceable "params" list.  Each entry is a dict with
    # one key and a jnp array of shape matching the trainable's compartment
    # selection.  `entry_to_cnn_idx[i]` tells us which CNN output value
    # feeds `default_params[i]`; the value is broadcast into the entry's
    # shape.  If multiple entries share a CNN index (L5TTPC Ih_dend ->
    # basal + apical) the backward pass naturally sums their gradients.
    def _key_for_entry(entry_dict):
        (k,) = entry_dict.keys()
        return k

    default_shapes = [e[_key_for_entry(e)].shape for e in default_params]

    ckpt = list(checkpoint_lengths) if checkpoint_lengths else None

    def _simulate_one(flat_phys):
        """Run one forward.  `flat_phys` shape: (P,)."""
        params = []
        for e, idx, shape in zip(default_params, entry_to_cnn_idx, default_shapes):
            k = _key_for_entry(e)
            val = jnp.broadcast_to(flat_phys[idx:idx + 1], shape)
            params.append({k: val})
        kw = dict(
            params=params,
            delta_t=spec.dt,
            t_max=spec.t_max,
            data_stimuli=data_stim,
            solver=solver,
        )
        if ckpt is not None:
            kw["checkpoint_lengths"] = ckpt
        v = jx.integrate(cell, **kw)
        # `v` shape: (n_recorded, T_sim).  Downsample time axis.
        v_ds = v[:, ::step]
        return v_ds

    simulate_batch_jit = jax.jit(jax.vmap(_simulate_one, in_axes=0))

    sim_len = len(stim_up)
    out_len = sim_len // step

    return _CellHandle(
        cell_name        = cell_name,
        spec             = spec,
        cell             = cell,
        default_params   = default_params,
        simulate_batch   = simulate_batch_jit,
        downsample_step  = step,
        sim_len          = sim_len,
        out_len          = out_len,
        v_init           = spec.v_init,
        stim_scale       = scale,
        out_dt           = float(spec.dt) * step,
    )


def get_handle(cell_name: str, stim_name: Optional[str] = None,
               checkpoint_lengths: Optional[Tuple[int, ...]] = None,
               solver: str = "bwd_euler",
               stim_scale: Optional[float] = None) -> _CellHandle:
    """Return a cached (cell, compiled simulate) pair, building on first use.

    `checkpoint_lengths`, `solver` and `stim_scale` are part of the cache key —
    distinct values get distinct compiled handles.  `stim_scale=None` means
    "the cell spec's production value".
    """
    ckpt_key = tuple(checkpoint_lengths) if checkpoint_lengths else None
    # Resolve None -> the spec's production scale BEFORE keying, so an explicit
    # `stim_scale=<spec value>` and `None` share one compiled handle.
    if stim_scale is None:
        stim_scale = getattr(jaxley_cells.get(cell_name), "stim_scale", 1.0)
    scale_key = float(stim_scale)
    key = (cell_name, stim_name or "__default__", ckpt_key, solver, scale_key)
    if key not in _CELL_CACHE:
        _CELL_CACHE[key] = _build_handle(cell_name, stim_name, ckpt_key, solver, scale_key)
    return _CELL_CACHE[key]


# ═════════════════════════════════════════════════════════════════════════
# Torch <-> JAX conversion
# ═════════════════════════════════════════════════════════════════════════

def _torch_to_jax(t: torch.Tensor):
    import jax
    if t.device.type == "cuda":
        # dlpack gives zero-copy on matching CUDA devices.
        try:
            return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t.contiguous()))
        except Exception:
            pass
    import jax.numpy as jnp
    return jnp.asarray(t.detach().cpu().numpy())


def _jax_to_torch(x, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    try:
        import jax
        cap = jax.dlpack.to_dlpack(x)
        return torch.utils.dlpack.from_dlpack(cap).to(device=device, dtype=dtype)
    except Exception:
        return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)


# ═════════════════════════════════════════════════════════════════════════
# torch.autograd.Function — the actual bridge
# ═════════════════════════════════════════════════════════════════════════

class _JaxleySimulate(torch.autograd.Function):
    """Backprop voltage-MSE gradients through a cached jaxley simulation."""

    @staticmethod
    def forward(ctx, params_phys: torch.Tensor, cell_name: str,
                stim_name: Optional[str],
                checkpoint_lengths: Optional[Tuple[int, ...]] = None,
                solver: str = "bwd_euler",
                stim_scale: Optional[float] = None) -> torch.Tensor:
        import jax
        handle = get_handle(cell_name, stim_name, checkpoint_lengths, solver, stim_scale)

        # (B, P) torch → jax
        params_j = _torch_to_jax(params_phys)

        # Capture vjp for backward.  `vjp_fn` is closed over params_j and
        # replays on grad_out.
        v_j, vjp_fn = jax.vjp(handle.simulate_batch, params_j)

        ctx.cell_name   = cell_name
        ctx.stim_name   = stim_name
        ctx.vjp_fn      = vjp_fn
        ctx.in_device   = params_phys.device
        ctx.in_dtype    = params_phys.dtype

        return _jax_to_torch(v_j, params_phys.device, params_phys.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_j = _torch_to_jax(grad_out.contiguous())
        (dparams_j,) = ctx.vjp_fn(grad_j)
        dparams = _jax_to_torch(dparams_j, ctx.in_device, ctx.in_dtype)
        # NaN-guard: some stimuli (e.g. ramps that sweep the cell across a spiking
        # bifurcation) yield a non-finite VJP for isolated samples even when the
        # forward trace is finite.  Left unguarded, one NaN gradient poisons the
        # whole batch and — under DDP — NCCL all-reduce spreads it to every rank,
        # killing the run.  Zero out non-finite per-parameter grads so those
        # samples simply contribute nothing to this step.
        dparams = torch.nan_to_num(dparams, nan=0.0, posinf=0.0, neginf=0.0)
        return dparams, None, None, None, None, None


def simulate_batch(params_phys: torch.Tensor,
                   cell_name: str,
                   stim_name: Optional[str] = None,
                   checkpoint_lengths: Optional[Tuple[int, ...]] = None,
                   solver: str = "bwd_euler",
                   stim_scale: Optional[float] = None) -> torch.Tensor:
    """Run a vmapped, differentiable jaxley forward on a batch of params.

    Parameters
    ----------
    params_phys : (B, P) torch.Tensor
        Physical conductance/capacitance vectors in the order defined by
        `jaxley_cells.get(cell_name).param_keys`.
    cell_name : str
        Key into the `toolbox.jaxley_cells` registry.
    stim_name : str, optional
        CSV stem under `spec.stim_dir`.  Defaults to `spec.default_stim_name`.
    stim_scale : float, optional
        Override the cell spec's stimulus multiplier (sweeps only; production
        callers leave it None so the spec value is used everywhere).

    Returns
    -------
    (B, n_recorded, T_out) torch.Tensor
        Voltage traces downsampled to the 10 kHz grid the CNN trains on.
    """
    if params_phys.ndim != 2:
        raise ValueError(f"params_phys must be 2D (B, P), got shape {tuple(params_phys.shape)}")
    ckpt = tuple(checkpoint_lengths) if checkpoint_lengths else None
    return _JaxleySimulate.apply(params_phys, cell_name, stim_name, ckpt, solver, stim_scale)


# ═════════════════════════════════════════════════════════════════════════
# Introspection helpers (used by HybridLoss + tests)
# ═════════════════════════════════════════════════════════════════════════

def output_shape(cell_name: str, stim_name: Optional[str] = None) -> Tuple[int, int]:
    """Return (n_recorded, T_out) without running a forward."""
    h = get_handle(cell_name, stim_name)
    # n_recorded may depend on how many .record() calls the spec made.
    # We can't know it cheaply without a forward; probe with a zero input.
    import jax.numpy as jnp
    p = jnp.zeros((1, len(h.spec.param_keys)), dtype=jnp.float32)
    v = h.simulate_batch(p[0])
    return (int(v.shape[0]), int(v.shape[1]))


def param_keys(cell_name: str) -> List[str]:
    return list(jaxley_cells.get(cell_name).param_keys)


def clear_cache() -> None:
    _CELL_CACHE.clear()
