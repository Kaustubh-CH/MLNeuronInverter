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
_NCOMP   = 4

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
        cell.apical.set("capacitance", 2.0)
        cell.basal.set("capacitance", 2.0)
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

    cell.init_states(delta_t=_DT)
    cell.soma.comp(0).record()

    # Mark each of the 19 CNN-facing params trainable.  The "basal + apical
    # both change" case for gIhbar_Ih_dend is handled by calling
    # make_trainable on both groups with the same key, then the bridge sums
    # the gradient contributions.
    groups = {"soma": cell.soma, "axon": cell.axon, "basal": cell.basal, "apical": cell.apical}
    for name, group_key, jax_key in _CSV_PARAM_MAP:
        if name == "gIhbar_Ih_dend":
            # Apply to basal + apical so gradient flows to both branches.
            groups["basal"].make_trainable(jax_key)
            groups["apical"].make_trainable(jax_key)
        elif group_key == "all":
            cell.make_trainable(jax_key)
        else:
            groups[group_key].make_trainable(jax_key)

    return cell


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
    )


register("L5TTPC", _spec)
# Also register under the BBP short-name so `cell_name_for_sim: L5_TTPC1cADpyr0`
# in a design yaml resolves directly.
register("L5_TTPC1cADpyr0", _spec)
