"""Ball-and-stick cell builder for the hybrid voltage-loss path.

Ported from the canonical reference at
/pscratch/sd/k/ktub1999/Neuron_Jaxley/sim_jaxley.py — soma HH + 5-comp
passive dendrite with NEURON-matched defaults.

Trainable parameters exposed to the CNN (in CNN-output order):

    0  HH_gNa     (soma)      S/cm^2
    1  HH_gK      (soma)      S/cm^2
    2  HH_gLeak   (soma)      S/cm^2
    3  Leak_gLeak (dendrite)  S/cm^2

Fixed: reversal potentials, axial resistivity, membrane capacitance, geometry.
"""

from pathlib import Path
from . import CellSpec, register

_STIM_DIR = Path("/pscratch/sd/k/ktub1999/main/DL4neurons2/stims")

_DT_STIM = 0.1     # ms
_DT      = 0.1     # ms (use 0.025 if spike-time precision <0.1 ms matters)
_T_MAX   = 500.0   # ms
_V_INIT  = -65.0   # mV

PARAM_KEYS = [
    "HH_gNa",      # soma
    "HH_gK",       # soma
    "HH_gLeak",    # soma
    "Leak_gLeak",  # dendrite
]

# Default physical values — match NEURON hh mechanism defaults.
_DEFAULTS = {
    "HH_gNa":     0.12,
    "HH_gK":      0.036,
    "HH_gLeak":   0.0003,
    "Leak_gLeak": 0.0001,
}


def _build():
    """Build the jaxley ball-and-stick cell and mark params trainable."""
    # Import jaxley lazily so the registry can be imported without jax.
    import jaxley as jx
    from jaxley.channels import HH, Leak

    soma_branch = jx.Branch([jx.Compartment()], ncomp=1)
    dend_branch = jx.Branch([jx.Compartment()], ncomp=5)
    cell = jx.Cell([soma_branch, dend_branch], parents=[-1, 0])

    cell.branch(0).set("length", 20.0)
    cell.branch(0).set("radius", 10.0)
    cell.branch(1).set("length", 40.0)
    cell.branch(1).set("radius", 1.0)

    cell.set("axial_resistivity", 35.4)
    cell.set("capacitance", 1.0)

    cell.branch(0).insert(HH())
    cell.branch(0).set("HH_gNa",   _DEFAULTS["HH_gNa"])
    cell.branch(0).set("HH_gK",    _DEFAULTS["HH_gK"])
    cell.branch(0).set("HH_gLeak", _DEFAULTS["HH_gLeak"])
    cell.branch(0).set("HH_eNa",    50.0)
    cell.branch(0).set("HH_eK",    -77.0)
    cell.branch(0).set("HH_eLeak", -54.3)

    cell.branch(1).insert(Leak())
    cell.branch(1).set("Leak_gLeak", _DEFAULTS["Leak_gLeak"])
    cell.branch(1).set("Leak_eLeak", _V_INIT)

    cell.set("v", _V_INIT)
    cell.init_states(delta_t=_DT)

    # Record soma voltage — must be set BEFORE integrate().
    cell.branch(0).comp(0).record()

    # Expose parameters to the bridge.  Each key maps to exactly one
    # compartment/branch so the trainable array has shape (1,) and the
    # CNN-output vector (B, P=4) vmaps cleanly.  Entry order == CNN order,
    # so entry_to_cnn_idx is just [0, 1, 2, 3].
    cell.branch(0).make_trainable("HH_gNa")
    cell.branch(0).make_trainable("HH_gK")
    cell.branch(0).make_trainable("HH_gLeak")
    cell.branch(1).make_trainable("Leak_gLeak")
    entry_to_cnn_idx = [0, 1, 2, 3]

    return cell, entry_to_cnn_idx


def _attach_stim(cell, stim_jnp):
    """Inject current at soma comp(0).  `stim_jnp` shape: (1, T_sim) in nA."""
    return cell.branch(0).comp(0).data_stimulate(stim_jnp)


def _attach_record(cell):
    cell.branch(0).comp(0).record()


def _spec() -> CellSpec:
    return CellSpec(
        build_fn          = _build,
        param_keys        = list(PARAM_KEYS),
        stim_attach_fn    = _attach_stim,
        record_fn         = _attach_record,
        dt                = _DT,
        dt_stim           = _DT_STIM,
        t_max             = _T_MAX,
        v_init            = _V_INIT,
        default_stim_name = "5k50kInterChaoticB",
        stim_dir          = _STIM_DIR,
    )


register("ball_and_stick", _spec)
