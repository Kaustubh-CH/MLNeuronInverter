"""Registry of jaxley cell builders for the hybrid voltage-loss path.

Each builder returns a `CellSpec`:

    CellSpec(cell, param_keys, stim_attach_fn, record_attach_fn,
             dt, t_max, v_init, default_stim_name, stim_dir)

The bridge (`toolbox.JaxleyBridge`) consumes this spec, calls
`cell.make_trainable(...)` for every key in `param_keys`, and builds a
jit+vmap simulate function that takes `(batch_params_phys, stim_waveform)`
and returns a batch of voltage traces.

Builders are intentionally additive — they do NOT duplicate the existing
NEURON-side biophysics code.  They mirror the canonical reference
implementations at:

    /pscratch/sd/k/ktub1999/Neuron_Jaxley/sim_jaxley.py          (ball-and-stick)
    /pscratch/sd/k/ktub1999/Neuron_Jaxley/sim_jaxley_L5TTPC1.py  (BBP L5 TTPC)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple


@dataclass
class CellSpec:
    """Everything JaxleyBridge needs to run a forward simulation.

    Attributes
    ----------
    build_fn
        Zero-arg callable that returns a fresh `jx.Cell` with channels
        inserted + defaults set + `make_trainable` called for every key in
        `param_keys`.  Called once per process, result cached.
    param_keys
        Ordered list of trainable-parameter names.  Index `i` corresponds
        to the CNN output position `i` and must match the order of
        `sum_train.yaml['input_meta']['parName']` for the target cell.
    stim_attach_fn
        Callable `(cell, stim_jnp_array) -> data_stimuli` that attaches the
        stimulus to the soma (typically `cell.branch(0).comp(0).data_stimulate(...)`
        for ball-and-stick, or `cell.soma.comp(0).data_stimulate(...)` for L5TTPC).
    record_fn
        Callable `(cell) -> None` that sets up voltage recording.
    dt, dt_stim, t_max, v_init
        Integration timestep, stimulus sampling period, simulation length
        (ms), and initial voltage.
    default_stim_name
        Stim to use when the bridge is called without an explicit stim.
    stim_dir
        Path to the CSV stim directory (the same one used by DL4neurons2).
    """
    build_fn:         Callable
    param_keys:       List[str]
    stim_attach_fn:   Callable
    record_fn:        Callable
    dt:               float
    dt_stim:          float
    t_max:            float
    v_init:           float
    default_stim_name: str
    stim_dir:         Path


_REGISTRY: dict = {}


def register(name: str, spec_factory: Callable[[], CellSpec]) -> None:
    """Register a cell-spec factory under `name`."""
    if name in _REGISTRY:
        raise ValueError(f"cell already registered: {name}")
    _REGISTRY[name] = spec_factory


def get(name: str) -> CellSpec:
    """Return the registered CellSpec for `name`.  Raises if unknown."""
    if name not in _REGISTRY:
        # Lazy-import builders so a missing jaxley dep doesn't blow up
        # callers that don't actually use jaxley.
        from . import ball_and_stick, l5ttpc  # noqa: F401
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown jaxley cell {name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def list_cells() -> List[str]:
    from . import ball_and_stick, l5ttpc  # noqa: F401
    return sorted(_REGISTRY)
