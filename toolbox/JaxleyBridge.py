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


_CELL_CACHE: dict = {}


def _build_handle(cell_name: str, stim_name: Optional[str] = None) -> _CellHandle:
    """Build + compile the cell once.  Expensive; called on first use."""
    import jax
    import jax.numpy as jnp
    import jaxley as jx

    spec = jaxley_cells.get(cell_name)
    cell = spec.build_fn()

    # Snapshot the structure of the trainable list *before* any simulate call.
    default_params = cell.get_parameters()

    # Stimulus waveform — fixed per handle.  Changing stims means a new
    # handle (new jit).  That's intentional: stim length + content change
    # the computation graph, so caching per (cell, stim) is the right key.
    stim_name = stim_name or spec.default_stim_name
    stim_path = Path(spec.stim_dir) / f"{stim_name}.csv"
    stim_csv  = jaxley_utils.load_stim_csv(stim_path)
    stim_up   = jaxley_utils.upsample_stim(stim_csv, spec.dt_stim, spec.dt, spec.t_max)
    step      = jaxley_utils.downsample_step(spec.dt, spec.dt_stim)

    # `data_stimulate` expects (n_injectors, T_sim).
    stim_jnp  = jnp.asarray(stim_up[np.newaxis, :])
    data_stim = spec.stim_attach_fn(cell, stim_jnp)

    # Build a pytree-replaceable "params" list.  Each entry is a dict with
    # one key and a jnp array of shape matching the trainable's compartment
    # selection.  To vmap over a single scalar per param, we take the CNN
    # output value and broadcast to that shape.
    # param_keys[i] may be used more than once (e.g. L5TTPC Ih_dend applies
    # to basal + apical).  Walk default_params and remember which indices
    # of the flat CNN vector feed each entry.
    param_keys = list(spec.param_keys)
    key_to_idx = {k: i for i, k in enumerate(param_keys)}

    def _key_for_entry(entry_dict):
        # Each entry has exactly one top-level key — use it.
        (k,) = entry_dict.keys()
        return k

    entry_to_cnn_idx: List[int] = []
    for e in default_params:
        k = _key_for_entry(e)
        # For L5TTPC, `Ih_gbar` is registered once in PARAM_KEYS (as the
        # BBP name `gIhbar_Ih_dend`).  We key the flat vector by BBP name
        # via jaxley_cells spec.param_keys; the jaxley key stored on the
        # trainable entry is `Ih_gbar`.  For the ball-and-stick we used the
        # jaxley key directly as the BBP name — so for that cell the
        # lookup is one-to-one.
        if k in key_to_idx:
            entry_to_cnn_idx.append(key_to_idx[k])
        else:
            # fall back: try matching any BBP key whose jaxley key == k
            # (for L5TTPC the spec exposes BBP names, which differ).
            matches = [i for i, bbp in enumerate(param_keys)
                       if bbp.split("_")[-1] in k or k in bbp]
            if not matches:
                raise RuntimeError(
                    f"can't align jaxley trainable {k!r} with any CNN param key "
                    f"in {param_keys!r}. Fix the spec's PARAM_KEYS."
                )
            entry_to_cnn_idx.append(matches[0])

    default_shapes = [e[_key_for_entry(e)].shape for e in default_params]

    def _simulate_one(flat_phys):
        """Run one forward.  `flat_phys` shape: (P,)."""
        params = []
        for e, idx, shape in zip(default_params, entry_to_cnn_idx, default_shapes):
            k = _key_for_entry(e)
            val = jnp.broadcast_to(flat_phys[idx:idx + 1], shape)
            params.append({k: val})
        v = jx.integrate(
            cell,
            params=params,
            delta_t=spec.dt,
            t_max=spec.t_max,
            data_stimuli=data_stim,
            solver="bwd_euler",
        )
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
    )


def get_handle(cell_name: str, stim_name: Optional[str] = None) -> _CellHandle:
    """Return a cached (cell, compiled simulate) pair, building on first use."""
    key = (cell_name, stim_name or "__default__")
    if key not in _CELL_CACHE:
        _CELL_CACHE[key] = _build_handle(cell_name, stim_name)
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
                stim_name: Optional[str]) -> torch.Tensor:
        import jax
        handle = get_handle(cell_name, stim_name)

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
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None]:
        grad_j = _torch_to_jax(grad_out.contiguous())
        (dparams_j,) = ctx.vjp_fn(grad_j)
        dparams = _jax_to_torch(dparams_j, ctx.in_device, ctx.in_dtype)
        return dparams, None, None


def simulate_batch(params_phys: torch.Tensor,
                   cell_name: str,
                   stim_name: Optional[str] = None) -> torch.Tensor:
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

    Returns
    -------
    (B, n_recorded, T_out) torch.Tensor
        Voltage traces downsampled to the 10 kHz grid the CNN trains on.
    """
    if params_phys.ndim != 2:
        raise ValueError(f"params_phys must be 2D (B, P), got shape {tuple(params_phys.shape)}")
    return _JaxleySimulate.apply(params_phys, cell_name, stim_name)


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
