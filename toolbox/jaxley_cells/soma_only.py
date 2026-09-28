"""Single-compartment HH soma cell.

Bench-only cell: same biophysics as the soma of `ball_and_stick.py`, but
without the 5-comp passive dendrite.  Useful as the minimal sanity case
when sweeping solvers / dt / batch size — any performance floor that we
can't hit here we definitely won't hit on the bigger cells.

Trainable parameters (CNN-output order):

    0  HH_gNa     S/cm^2
    1  HH_gK      S/cm^2
    2  HH_gLeak   S/cm^2
"""

from pathlib import Path
from . import CellSpec, register

_STIM_DIR = Path("/pscratch/sd/k/ktub1999/main/DL4neurons2/stims")

_DT_STIM = 0.1     # ms
_DT      = 0.1     # ms (use 0.025 if spike-time precision <0.1 ms matters)
_T_MAX   = 500.0   # ms (spec default; bench overrides to 100)
_V_INIT  = -65.0   # mV
# Stimulus multiplier (see CellSpec.stim_scale).  The stim CSVs are 6-7 nA
# L5-scale templates; this 1.26e-5 cm^2 HH soma (12.6 pF, R_in ~265 MOhm)
# needs them scaled to its own rheobase.  Value chosen by
# scripts/stim_scale_scan.py (see docs/model_ladder/): 0.05 -> InterChaoticB peaks
# at 0.34 nA (8 spikes, +40 mV, AHP -76, Vmin -77 mV), chaoticRamp at 0.30 nA
# (19 spikes); over the +-0.5 decade box: 5-13 % silent, 0 % block, 0 % out of range.
_STIM_SCALE = 0.05

PARAM_KEYS = [
    "HH_gNa",
    "HH_gK",
    "HH_gLeak",
]

_DEFAULTS = {
    "HH_gNa":   0.12,
    "HH_gK":    0.036,
    "HH_gLeak": 0.0003,
}


def _build():
    import jaxley as jx
    from jaxley.channels import HH

    soma_branch = jx.Branch([jx.Compartment()], ncomp=1)
    cell = jx.Cell([soma_branch], parents=[-1])

    cell.branch(0).set("length", 20.0)
    cell.branch(0).set("radius", 10.0)

    cell.set("axial_resistivity", 35.4)
    cell.set("capacitance", 1.0)

    cell.branch(0).insert(HH())
    cell.branch(0).set("HH_gNa",   _DEFAULTS["HH_gNa"])
    cell.branch(0).set("HH_gK",    _DEFAULTS["HH_gK"])
    cell.branch(0).set("HH_gLeak", _DEFAULTS["HH_gLeak"])
    cell.branch(0).set("HH_eNa",    50.0)
    cell.branch(0).set("HH_eK",    -77.0)
    cell.branch(0).set("HH_eLeak", -54.3)

    cell.set("v", _V_INIT)
    cell.init_states(delta_t=_DT)

    cell.branch(0).comp(0).record()

    cell.branch(0).make_trainable("HH_gNa")
    cell.branch(0).make_trainable("HH_gK")
    cell.branch(0).make_trainable("HH_gLeak")
    entry_to_cnn_idx = [0, 1, 2]

    return cell, entry_to_cnn_idx


def _attach_stim(cell, stim_jnp):
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
        stim_scale        = _STIM_SCALE,
    )


register("single_comp", _spec)
# Alias under the MODULE name so generators that import
# `toolbox.jaxley_cells.<name>` and then call the bridge with the same name
# (scripts/gen_ca3_sharded.py --source-cell soma_only) resolve.
register("soma_only", _spec)
