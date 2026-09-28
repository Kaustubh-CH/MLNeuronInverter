"""L5TTPC cell builder for the hybrid voltage-loss path.

Ported from /pscratch/sd/k/ktub1999/Neuron_Jaxley/sim_jaxley_L5TTPC1.py and
bbp_channels_jaxley.py.  The 19 trainable parameters are aligned with the
BBP `parName` ordering stored in every `<cell>.mlPack1.h5`'s meta (also
baked into `predictExp.py`'s `param_names` list).

Phase 1 wires up the build function + param list.  Phase 5 runs batched
vmapped integration on GPU + benchmarks.  Until Phase 5 the cell builds but
is only exercised by shape tests — gradcheck continues to run on the
ball-and-stick cell, which is orders of magnitude cheaper to backprop.
"""

import os
from pathlib import Path
from . import CellSpec, register

# Canonical paths — owned by /pscratch/sd/k/ktub1999/Neuron_Jaxley/.
_NJ_ROOT   = Path("/pscratch/sd/k/ktub1999/Neuron_Jaxley")
_SWC_PATH  = _NJ_ROOT / "results_detailed_morphology" / "L5_TTPC1.swc"
_STIM_DIR  = Path("/pscratch/sd/k/ktub1999/main/DL4neurons2/stims")

_DT_STIM = 0.1
_DT      = 0.1
_T_MAX   = 500.0
_V_INIT  = -75.0
# Compartments per branch.  Env-overridable so candidate C1 can halve the
# spatial discretisation (L5TTPC_NCOMP=2) at runtime without a code fork.
# Read once at import; used by read_swc, the apical-Ih gradient loop, and
# swc_apical_branch_distances.  Default stays 4 (baseline unchanged).
_NCOMP   = int(os.environ.get("L5TTPC_NCOMP", "4"))
# Stimulus multiplier (see CellSpec.stim_scale).  The stim CSVs were designed
# for this cell (1.0 = the native BBP protocol), but at 1.0 the BBP-default cell
# fires only 2 spikes under 5k50kInterChaoticB (box median 2).  1.5 (chosen by
# scripts/stim_scale_scan.py, docs/model_ladder/) gives 7-8 spikes with +37..40 mV
# peaks, -80 mV AHPs, Vmin -90 and 15 spikes on 5kChaoticRamp, no block and 0 %
# out-of-range across the +-0.5-decade box for BOTH ncomp=2 and ncomp=4 -- in
# line with the other rungs of the model ladder.  Packs made before 2026-09-03
# used 1.0 (their meta has no _STIM_SCALE key).
_STIM_SCALE = 1.5

# Fixed (non-trainable) DENDRITIC passive properties.  The basal + apical tree carries almost
# all of the membrane area, so these -- not the trainable g_pas_somatic / g_pas_axonal -- set
# the cell's input resistance and hence its rheobase.  BBP biophysics.hoc: g_pas 3e-5 S/cm^2
# everywhere, cm 2 uF/cm^2 on dendrites (spine correction).  Env/attr-overridable for the
# excitability probe (scripts/l5_excitability_probe.py, 2026-09-23); defaults = BBP.
_DEND_GPAS = float(os.environ.get("L5TTPC_DEND_GPAS", "3e-5"))
_DEND_CM   = float(os.environ.get("L5TTPC_DEND_CM", "2.0"))

# Apical Ih distance-dependent gradient — biophysics.hoc convention for L5TTPC.
# gIh(d) = max(0, _IH_A + _IH_B * exp(d * _IH_K)) * ih_base   (S/cm²)
# Ported from /pscratch/sd/k/ktub1999/Neuron_Jaxley/sim_jaxley_L5TTPC1.py.
_IH_A    = -0.8696
_IH_B    =  2.087
_IH_K    =  0.0031   # µm⁻¹
_IH_BASE =  8e-5     # base gbar (S/cm²)

# Parameter ordering matches `sum_train.yaml['input_meta']['parName']` for a
# BBP cADpyr excitatory cell (identical to predictExp.py's inlined list).
# Each entry is (bbp_name, jaxley_group, jaxley_param_key).  "all" = cell.set.
_CSV_PARAM_MAP = [
    ("gNaTs2_tbar_NaTs2_t_apical",    "apical", "NaTs2_t_gbar"),
    ("gSKv3_1bar_SKv3_1_apical",      "apical", "SKv3_1_gbar"),
    ("gImbar_Im_apical",              "apical", "Im_gbar"),
    ("gIhbar_Ih_dend",                "basal",  "Ih_gbar"),       # also apical
    ("gNaTa_tbar_NaTa_t_axonal",      "axon",   "NaTa_t_gbar"),
    ("gK_Tstbar_K_Tst_axonal",        "axon",   "K_Tst_gbar"),
    ("gNap_Et2bar_Nap_Et2_axonal",    "axon",   "Nap_Et2_gbar"),
    ("gSK_E2bar_SK_E2_axonal",        "axon",   "CaComplex_gSK_E2"),
    ("gCa_HVAbar_Ca_HVA_axonal",      "axon",   "CaComplex_gCa_HVA"),
    ("gK_Pstbar_K_Pst_axonal",        "axon",   "K_Pst_gbar"),
    ("gCa_LVAstbar_Ca_LVAst_axonal",  "axon",   "CaComplex_gCa_LVAst"),
    ("g_pas_axonal",                  "axon",   "BBPLeak_gLeak"),
    ("cm_axonal",                     "axon",   "capacitance"),
    ("gSKv3_1bar_SKv3_1_somatic",     "soma",   "SKv3_1_gbar"),
    ("gNaTs2_tbar_NaTs2_t_somatic",   "soma",   "NaTs2_t_gbar"),
    ("gCa_LVAstbar_Ca_LVAst_somatic", "soma",   "CaComplex_gCa_LVAst"),
    ("g_pas_somatic",                 "soma",   "BBPLeak_gLeak"),
    ("cm_somatic",                    "soma",   "capacitance"),
    ("e_pas_all",                     "all",    "BBPLeak_eLeak"),
]

PARAM_KEYS    = [entry[0] for entry in _CSV_PARAM_MAP]
_PARAM_GROUPS = [entry[1] for entry in _CSV_PARAM_MAP]
_PARAM_JAX    = [entry[2] for entry in _CSV_PARAM_MAP]

# BBP base values, verbatim from DL4neurons2/.../NewBase2/L5Params.csv (the same
# CSV generate_L5_samples.py / get_random_params samples around). Used as the
# unit=0 centre when a caller does not supply an explicit phys_par_range (e.g.
# scripts/gen_ball_and_stick_data.py). Order matches PARAM_KEYS.
# Non-conductance entries get their OWN sampling range (see
# toolbox.jaxley_utils.build_phys_par_range), LINEAR exactly as DL4neurons2
# run.py:get_random_params does for e_pas_all / cm_*:  phys = mid + u*halfwidth
#   cm      1.25 +- 0.75 -> 0.5 .. 2.0 uF/cm^2   (BBP pack range, uniform)
#   e_pas   -75  +- 10   -> -85 .. -65 mV        (BBP pack range, uniform)
# Conductances use the caller's log_halfspan: 1.0 = the BBP +-1 decade
# (run.py UNIT_RANGES [-1, 1]); the model ladder used 0.5.
PHYS_RANGE_OVERRIDES = {
    "cm_axonal":  (1.25, 0.75, "uF/cm^2", "lin"),
    "cm_somatic": (1.25, 0.75, "uF/cm^2", "lin"),
    "e_pas_all":  (-75.0, 10.0, "mV", "lin"),
}

_DEFAULTS = {
    "gNaTs2_tbar_NaTs2_t_apical":    0.026145,
    "gSKv3_1bar_SKv3_1_apical":      0.004226,
    "gImbar_Im_apical":              0.000143,
    "gIhbar_Ih_dend":                8e-05,
    "gNaTa_tbar_NaTa_t_axonal":      3.137968,
    "gK_Tstbar_K_Tst_axonal":        0.089259,
    "gNap_Et2bar_Nap_Et2_axonal":    0.006827,
    "gSK_E2bar_SK_E2_axonal":        0.007104,
    "gCa_HVAbar_Ca_HVA_axonal":      0.00099,
    "gK_Pstbar_K_Pst_axonal":        0.973538,
    "gCa_LVAstbar_Ca_LVAst_axonal":  0.008752,
    "g_pas_axonal":                  3e-05,
    "cm_axonal":                     1.0,
    "gSKv3_1bar_SKv3_1_somatic":     0.303472,
    "gNaTs2_tbar_NaTs2_t_somatic":   0.983955,
    "gCa_LVAstbar_Ca_LVAst_somatic": 0.000333,
    "g_pas_somatic":                 3e-05,
    "cm_somatic":                    1.0,
    "e_pas_all":                     -75.0,
}


def _apply_apical_ih_gradient(cell, swc_path: str, ih_base: float = _IH_BASE) -> None:
    """Apply the BBP biophysics.hoc distance-dependent Ih gradient to apical compartments.

    gIh(d) = max(0, _IH_A + _IH_B * exp(d * _IH_K)) * ih_base
    """
    import sys
    import numpy as np
    if str(_NJ_ROOT) not in sys.path:
        sys.path.insert(0, str(_NJ_ROOT))
    from morphology_utils import swc_apical_branch_distances

    distances = swc_apical_branch_distances(swc_path, ncomp=_NCOMP)
    apical_nodes = cell.apical.nodes
    # jaxley 0.13.x uses `global_branch_index`; older trees used `branch_index`.
    branch_col = "global_branch_index" if "global_branch_index" in apical_nodes.columns else "branch_index"
    apical_branch_indices = apical_nodes[branch_col].unique()

    n_swc = len(distances)
    n_jax = len(apical_branch_indices)
    if n_swc != n_jax:
        # Fallback to the audit's flat-value sentinel — clearly visible to anyone
        # who runs the cell at default and finds the gradient missing.
        cell.apical.set("Ih_gbar", 9.74e-5)
        return

    for i, branch_idx in enumerate(apical_branch_indices):
        for j in range(_NCOMP):
            d    = float(distances[i, j])
            g_ih = max(0.0, (_IH_A + _IH_B * np.exp(d * _IH_K)) * ih_base)
            cell.branch(int(branch_idx)).comp(j).set("Ih_gbar", g_ih)


def _build():
    """Build the jaxley L5TTPC cell, insert BBP channels, and mark 19 params trainable."""
    import sys
    # The channel implementations live in /pscratch/sd/k/ktub1999/Neuron_Jaxley/
    # — import them as-is rather than duplicating the .mod transcription.
    if str(_NJ_ROOT) not in sys.path:
        sys.path.insert(0, str(_NJ_ROOT))
    import jaxley as jx
    from bbp_channels_jaxley import (
        NaTs2_t, NaTa_t, Nap_Et2, SKv3_1, K_Tst, K_Pst, Im, Ih, CaComplex, BBPLeak,
    )

    if not _SWC_PATH.exists():
        raise FileNotFoundError(
            f"L5TTPC SWC missing: {_SWC_PATH}. Regenerate with "
            f"`python {_NJ_ROOT}/sim_neuron_L5TTPC1.py` inside the Neuron_Jaxley env."
        )

    cell = jx.read_swc(str(_SWC_PATH), ncomp=_NCOMP, assign_groups=True)

    cell.set("axial_resistivity", 100.0)
    cell.set("capacitance", 1.0)
    cell.set("v", _V_INIT)
    try:
        cell.apical.set("capacitance", _DEND_CM)
        cell.basal.set("capacitance", _DEND_CM)
    except Exception:
        pass

    cell.soma.insert(NaTs2_t()); cell.soma.insert(SKv3_1()); cell.soma.insert(Ih())
    cell.soma.insert(CaComplex()); cell.soma.insert(BBPLeak())

    cell.axon.insert(NaTa_t()); cell.axon.insert(Nap_Et2()); cell.axon.insert(SKv3_1())
    cell.axon.insert(K_Tst()); cell.axon.insert(K_Pst())
    cell.axon.insert(CaComplex()); cell.axon.insert(BBPLeak())

    cell.basal.insert(Ih()); cell.basal.insert(BBPLeak())

    cell.apical.insert(NaTs2_t()); cell.apical.insert(SKv3_1()); cell.apical.insert(Im())
    cell.apical.insert(Ih()); cell.apical.insert(BBPLeak())

    # Defaults from biophysics.hoc (reversal potentials etc.)
    cell.set("BBPLeak_gLeak", 3e-5)
    cell.basal.set("BBPLeak_gLeak", _DEND_GPAS); cell.apical.set("BBPLeak_gLeak", _DEND_GPAS)
    cell.set("BBPLeak_eLeak", -75.0)
    cell.soma.set("NaTs2_t_ena", 50.0); cell.soma.set("SKv3_1_ek", -85.0)
    cell.soma.set("Ih_ehcn", -45.0); cell.soma.set("CaComplex_ek", -85.0)
    cell.soma.set("CaComplex_gamma", 0.000609); cell.soma.set("CaComplex_decay", 210.485284)
    cell.axon.set("NaTa_t_ena", 50.0); cell.axon.set("Nap_Et2_ena", 50.0)
    cell.axon.set("SKv3_1_ek", -85.0); cell.axon.set("K_Tst_ek", -85.0)
    cell.axon.set("K_Pst_ek", -85.0); cell.axon.set("CaComplex_ek", -85.0)
    cell.axon.set("CaComplex_gamma", 0.002910); cell.axon.set("CaComplex_decay", 287.198731)
    cell.basal.set("Ih_ehcn", -45.0)
    cell.apical.set("NaTs2_t_ena", 50.0); cell.apical.set("SKv3_1_ek", -85.0)
    cell.apical.set("Im_ek", -85.0); cell.apical.set("Ih_ehcn", -45.0)

    # Set the 19 CNN-facing parameters to their BBP base values (`_DEFAULTS`,
    # = L5Params.csv) so that a forward WITHOUT a CNN override -- benchmarks,
    # the cross-cell physiology scan, `cell.get_parameters()` -- is the real
    # BBP cell and not the channel classes' 1e-5 placeholder gbar.  The bridge
    # overwrites every one of these with the CNN prediction during training,
    # so this changes nothing there.  gIhbar_Ih_dend is applied below via the
    # apical distance gradient (basal uniform).
    groups_for_defaults = {"soma": cell.soma, "axon": cell.axon,
                           "basal": cell.basal, "apical": cell.apical}
    for name, group_key, jax_key in _CSV_PARAM_MAP:
        val = float(_DEFAULTS[name])
        if name == "gIhbar_Ih_dend":
            cell.basal.set(jax_key, val)        # apical: gradient below
        elif group_key == "all":
            cell.set(jax_key, val)
        else:
            groups_for_defaults[group_key].set(jax_key, val)

    # BBP biophysics.hoc applies a distance-dependent gradient to apical Ih.
    # Apply with `_IH_BASE` so the default-parameter forward matches NEURON.
    # NOTE: `cell.apical.make_trainable("Ih_gbar")` below makes the bridge
    # uniformly broadcast the CNN-predicted scalar across apical compartments,
    # which overwrites this gradient during training. Gradient is preserved
    # at default params (benchmarks, reference comparisons) and during inference
    # whenever no override is supplied. See docs/phase1/L5TTPC.md for the plan
    # to preserve the gradient under CNN-driven training.
    _apply_apical_ih_gradient(cell, str(_SWC_PATH), _IH_BASE)

    cell.init_states(delta_t=_DT)
    cell.soma.comp(0).record()

    # Mark each of the 19 CNN-facing params trainable.  The "basal + apical
    # both change" case for gIhbar_Ih_dend is handled by calling
    # make_trainable on both groups with the same key — both entries point
    # back to the same CNN index (3), so the CNN prediction broadcasts to
    # both branches and the backward pass naturally sums the gradients.
    groups = {"soma": cell.soma, "axon": cell.axon, "basal": cell.basal, "apical": cell.apical}
    entry_to_cnn_idx = []
    for cnn_idx, (name, group_key, jax_key) in enumerate(_CSV_PARAM_MAP):
        if name == "gIhbar_Ih_dend":
            groups["basal"].make_trainable(jax_key);   entry_to_cnn_idx.append(cnn_idx)
            groups["apical"].make_trainable(jax_key);  entry_to_cnn_idx.append(cnn_idx)
        elif group_key == "all":
            cell.make_trainable(jax_key);              entry_to_cnn_idx.append(cnn_idx)
        else:
            groups[group_key].make_trainable(jax_key); entry_to_cnn_idx.append(cnn_idx)

    return cell, entry_to_cnn_idx


def _attach_stim(cell, stim_jnp):
    return cell.soma.comp(0).data_stimulate(stim_jnp)


def _attach_record(cell):
    cell.soma.comp(0).record()


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


register("L5TTPC", _spec)
# Also register under the BBP short-name so `cell_name_for_sim: L5_TTPC1cADpyr0`
# in a design yaml resolves directly.
register("L5_TTPC1cADpyr0", _spec)
# And under the MODULE name `l5ttpc` — HybridLoss.build_hybrid_loss resolves
# `t_max_override` by importing `toolbox.jaxley_cells.<cell_name_for_sim>`, so
# `cell_name_for_sim` must equal the module name (the convention
# ball_and_stick_bbp follows). Designs use `cell_name_for_sim: l5ttpc`.
register("l5ttpc", _spec)
