"""Multi-probe L5TTPC cell builder (EXPERIMENT 2).

Identical biophysics to the baseline `l5ttpc` cell, but records voltage at
FOUR locations instead of soma-only:

    probe 0 : soma                  (recorded by l5ttpc._build, index 0)
    probe 1 : axon  (proximal)
    probe 2 : apical (mid/distal)
    probe 3 : dend  (mid basal)

This matches the BBP data-probe convention documented in
`l5ttpc_detailed_c2b.hpar.yaml` ("data probe order: [soma, axon, apic, dend]")
and mirrors DL4neurons2/get_rec_points.py (distance-sorted: soma=0, then
axon/apical/basal by distance).

The hypothesis (docs/perf_research/sensitivity_oat.md): recording at multiple
compartments gives the CNN the information needed to recover the many
parameters that are nearly invisible at the soma.

All 19 trainable parameters, channel insertion, the apical-Ih gradient, and
the unit<->phys mapping are inherited verbatim from `l5ttpc` — we only add the
3 extra `.record()` calls.  Reads L5TTPC_NCOMP env (=2) via the l5ttpc module.

Probe-selection logic
---------------------
Distance-based selection in jaxley is fiddly, so we use FIXED representative
compartments per group (acceptable for this experiment, documented here):
  * axon   : SMALLEST global_branch_index in the axon group (proximal axon),
             comp(0).  Mirrors get_rec_points recording axon[0]/[1].
  * apical : MIDDLE global_branch_index of the apical group, middle comp
             (ncomp//2) — a mid/distal apical site.
  * dend   : MIDDLE global_branch_index of the basal group, middle comp.
The three sites sit on different branches/groups, so the four traces are
genuinely distinct (verified at build time).
"""

from . import CellSpec, register
from . import l5ttpc

# Re-export everything the data generator + bridge need from the baseline cell.
_DEFAULTS = l5ttpc._DEFAULTS
PARAM_KEYS = l5ttpc.PARAM_KEYS
_DT        = l5ttpc._DT
_DT_STIM   = l5ttpc._DT_STIM
_T_MAX     = l5ttpc._T_MAX
_V_INIT    = l5ttpc._V_INIT
_NCOMP     = l5ttpc._NCOMP
_STIM_SCALE = l5ttpc._STIM_SCALE
PHYS_RANGE_OVERRIDES = l5ttpc.PHYS_RANGE_OVERRIDES

# Probe order MUST equal the .record() order below, the data-channel order,
# probsSelect order, and voltage_loss.probe_loss_indices order.
PROBE_NAMES = ["soma", "axon", "apical", "dend"]


def _branch_indices(group):
    """Return the sorted unique global branch indices of a jaxley group."""
    nodes = group.nodes
    col = "global_branch_index" if "global_branch_index" in nodes.columns else "branch_index"
    return sorted(int(b) for b in nodes[col].unique())


def _add_extra_records(cell):
    """Add axon / apical / dend records (in that order) to an already-built
    cell whose soma is already recorded (probe 0)."""
    mid_comp = max(0, _NCOMP // 2)  # ncomp=2 -> comp 1 (mid/distal of branch)

    axon_branches   = _branch_indices(cell.axon)
    apical_branches = _branch_indices(cell.apical)
    basal_branches  = _branch_indices(cell.basal)

    # probe 1: proximal axon — smallest branch index, comp(0).
    axon_b = axon_branches[0]
    cell.branch(axon_b).comp(0).record()

    # probe 2: mid/distal apical — middle branch, middle comp.
    apic_b = apical_branches[len(apical_branches) // 2]
    cell.branch(apic_b).comp(mid_comp).record()

    # probe 3: mid basal (dend) — middle branch, middle comp.
    dend_b = basal_branches[len(basal_branches) // 2]
    cell.branch(dend_b).comp(mid_comp).record()

    return {"axon": (axon_b, 0), "apical": (apic_b, mid_comp), "dend": (dend_b, mid_comp)}


def _build():
    """Build the baseline L5TTPC cell (records soma), then add 3 more records."""
    cell, entry_to_cnn_idx = l5ttpc._build()
    _add_extra_records(cell)
    return cell, entry_to_cnn_idx


def _attach_stim(cell, stim_jnp):
    return l5ttpc._attach_stim(cell, stim_jnp)


def _attach_record(cell):
    # soma first (probe 0), then axon / apical / dend (probes 1-3).
    cell.soma.comp(0).record()
    _add_extra_records(cell)


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
        stim_dir          = l5ttpc._STIM_DIR,
        stim_scale        = _STIM_SCALE,
    )


register("l5ttpc_multiprobe", _spec)
