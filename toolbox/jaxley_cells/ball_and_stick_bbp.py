"""Ball-and-stick cell with BBP channels (multi-section, multi-channel).

Same geometry as `ball_and_stick.py` (1-comp soma + 5-comp dendrite), but
with the BBP channel set from L5TTPC inserted on both sections.  The
trainable surface (12 parameters) is a deliberate trim of L5TTPC's 19 —
we drop axon and the apical Ih distance gradient, keeping just the
soma/apical channel pairs that exercise the same biophysics with two
sections instead of four.

Channel layout:

    soma     : NaTs2_t, Nap_Et2, SKv3_1, K_Tst, K_Pst, CaComplex (HVA+LVA+SK), Ih (fixed), BBPLeak
    dendrite : NaTs2_t, K_Pst, Im, Ih, BBPLeak

SKv3_1 (the BBP somatic Kv3 repolariser, 0.30 S/cm^2 in L5TTPC's somatic
block) was added 2026-09-03: without it the soma carries 0.98 S/cm^2 NaTs2_t
against only K_Tst/K_Pst and fires ONE spike then sits in a -25 mV
depolarisation plateau under every stimulus (see
docs/model_ladder/physiology/cells_default_physiology.md / ball_and_stick_bbp_skv3_fix.png).  With it the
cell fires repetitively with +40 mV peaks and -80 mV AHPs.
Somatic Ih (fixed at the BBP somatic 8e-5 S/cm^2, as in l5ttpc.py; not
trainable) was added at the same time: the BBP somatic block carries Ih, and
without it the hyperpolarising lobes of InterChaoticB drive this high-R_in
cell below -105 mV at any stim scale that also makes it fire.

Reversal potentials, Ca decay/gamma, and leak reversal are copied from
`toolbox/jaxley_cells/l5ttpc.py:140-151` (somatic block for soma,
apical block for dendrite).  No Ih distance gradient — that lives in the
sibling Ih-fix worktree.

CNN-output order (13 params):

    0  gNaTs2_tbar_NaTs2_t_somatic   (soma)
    1  gNap_Et2bar_Nap_Et2_somatic   (soma)
    2  gSKv3_1bar_SKv3_1_somatic     (soma)   <- added 2026-09-03
    3  gK_Tstbar_K_Tst_somatic       (soma)
    4  gK_Pstbar_K_Pst_somatic       (soma)
    5  gCa_LVAstbar_Ca_LVAst_somatic (soma)
    6  gCa_HVAbar_Ca_HVA_somatic     (soma)
    7  g_pas_somatic                 (soma)
    8  gNaTs2_tbar_NaTs2_t_apical    (dendrite)
    9  gK_Pstbar_K_Pst_apical        (dendrite)
   10  gImbar_Im_apical              (dendrite)
   11  gIhbar_Ih_apical              (dendrite)
   12  g_pas_apical                  (dendrite)

Packs/models made before 2026-09-03 have the old 12-param order (no SKv3_1);
they are not compatible with this cell (and the cell they were made with
could not fire repetitively).
"""

from pathlib import Path
from . import CellSpec, register

# Channel implementations live alongside the L5TTPC reference.
_NJ_ROOT  = Path("/pscratch/sd/k/ktub1999/Neuron_Jaxley")
_STIM_DIR = Path("/pscratch/sd/k/ktub1999/main/DL4neurons2/stims")

_DT_STIM = 0.1     # ms
_DT      = 0.1     # ms (use 0.025 if spike-time precision <0.1 ms matters)
_T_MAX   = 500.0   # ms
_V_INIT  = -75.0   # mV  — BBP convention (matches L5TTPC), not HH ball-and-stick
# Stimulus multiplier (see CellSpec.stim_scale): the 6-7 nA L5 templates drive
# this ~14 pF two-section cell to -340 mV.  Chosen by scripts/stim_scale_scan.py
# (docs/model_ladder/): 0.07 -> InterChaoticB peak 0.48 nA (4 spikes, +45 mV,
# AHP -88, Vmin -100), chaoticRamp 0.42 nA (8 spikes); larger scales push the
# hyperpolarising lobes of InterChaoticB below -105 mV in this GOhm cell.
_STIM_SCALE = 0.07

# Each entry: (cnn_param_name, branch_index, jaxley_param_key).
# Branch 0 = soma, branch 1 = dendrite.
_PARAM_MAP = [
    ("gNaTs2_tbar_NaTs2_t_somatic",   0, "NaTs2_t_gbar"),
    ("gNap_Et2bar_Nap_Et2_somatic",   0, "Nap_Et2_gbar"),
    ("gSKv3_1bar_SKv3_1_somatic",     0, "SKv3_1_gbar"),
    ("gK_Tstbar_K_Tst_somatic",       0, "K_Tst_gbar"),
    ("gK_Pstbar_K_Pst_somatic",       0, "K_Pst_gbar"),
    ("gCa_LVAstbar_Ca_LVAst_somatic", 0, "CaComplex_gCa_LVAst"),
    ("gCa_HVAbar_Ca_HVA_somatic",     0, "CaComplex_gCa_HVA"),
    ("g_pas_somatic",                 0, "BBPLeak_gLeak"),
    ("gNaTs2_tbar_NaTs2_t_apical",    1, "NaTs2_t_gbar"),
    ("gK_Pstbar_K_Pst_apical",        1, "K_Pst_gbar"),
    ("gImbar_Im_apical",              1, "Im_gbar"),
    ("gIhbar_Ih_apical",              1, "Ih_gbar"),
    ("g_pas_apical",                  1, "BBPLeak_gLeak"),
]

PARAM_KEYS = [entry[0] for entry in _PARAM_MAP]

# Defaults — soma values mirror BBP biophysics.hoc somatic block (same as the
# L5TTPC bench row in `toolbox/tests/bench_jaxley_cells.py:69-89`); apical
# values mirror the apical block.  Conductances we don't have a direct BBP
# apical default for (K_Pst_apical) get a small plausible prior so the
# default-param forward stays well-behaved.
_DEFAULTS = {
    "gNaTs2_tbar_NaTs2_t_somatic":   0.983955,
    "gNap_Et2bar_Nap_Et2_somatic":   0.006827,
    "gSKv3_1bar_SKv3_1_somatic":     0.303472,   # BBP L5TTPC somatic value
    "gK_Tstbar_K_Tst_somatic":       0.089259,
    "gK_Pstbar_K_Pst_somatic":       0.100000,
    "gCa_LVAstbar_Ca_LVAst_somatic": 0.000333,
    "gCa_HVAbar_Ca_HVA_somatic":     0.000990,
    "g_pas_somatic":                 3.0e-5,
    "gNaTs2_tbar_NaTs2_t_apical":    0.026145,
    "gK_Pstbar_K_Pst_apical":        0.010000,
    "gImbar_Im_apical":              0.000143,
    "gIhbar_Ih_apical":              8.0e-5,
    "g_pas_apical":                  3.0e-5,
}


def _build():
    """Build the jaxley ball-and-stick cell with BBP channels and mark 13 params trainable."""
    import sys
    if str(_NJ_ROOT) not in sys.path:
        sys.path.insert(0, str(_NJ_ROOT))
    import jaxley as jx
    from bbp_channels_jaxley import (
        NaTs2_t, Nap_Et2, SKv3_1, K_Tst, K_Pst, Im, Ih, CaComplex, BBPLeak,
    )

    soma_branch = jx.Branch([jx.Compartment()], ncomp=1)
    dend_branch = jx.Branch([jx.Compartment()], ncomp=5)
    cell = jx.Cell([soma_branch, dend_branch], parents=[-1, 0])

    cell.branch(0).set("length", 20.0)
    cell.branch(0).set("radius", 10.0)
    cell.branch(1).set("length", 40.0)
    cell.branch(1).set("radius", 1.0)

    # BBP biophysics defaults — Ra, cm, init voltage.
    cell.set("axial_resistivity", 100.0)
    cell.set("capacitance", 1.0)
    cell.set("v", _V_INIT)

    # ── Soma channels ──────────────────────────────────────────────────────
    cell.branch(0).insert(NaTs2_t())
    cell.branch(0).insert(Nap_Et2())
    cell.branch(0).insert(SKv3_1())
    cell.branch(0).insert(K_Tst())
    cell.branch(0).insert(K_Pst())
    cell.branch(0).insert(CaComplex())
    cell.branch(0).insert(Ih())
    cell.branch(0).insert(BBPLeak())

    # ── Dendrite channels ──────────────────────────────────────────────────
    cell.branch(1).insert(NaTs2_t())
    cell.branch(1).insert(K_Pst())
    cell.branch(1).insert(Im())
    cell.branch(1).insert(Ih())
    cell.branch(1).insert(BBPLeak())

    # ── Default conductances (set on each branch before make_trainable) ────
    for cnn_name, branch_idx, jax_key in _PARAM_MAP:
        cell.branch(branch_idx).set(jax_key, _DEFAULTS[cnn_name])

    # ── Reversal potentials & Ca dynamics (copied from l5ttpc.py:140-151) ──
    cell.set("BBPLeak_eLeak", -75.0)

    # Soma: somatic biophysics block.
    cell.branch(0).set("NaTs2_t_ena",     50.0)
    cell.branch(0).set("Nap_Et2_ena",     50.0)
    cell.branch(0).set("SKv3_1_ek",      -85.0)
    cell.branch(0).set("K_Tst_ek",       -85.0)
    cell.branch(0).set("K_Pst_ek",       -85.0)
    cell.branch(0).set("CaComplex_ek",   -85.0)
    cell.branch(0).set("CaComplex_gamma", 0.000609)
    cell.branch(0).set("CaComplex_decay", 210.485284)
    cell.branch(0).set("Ih_gbar",         8.0e-5)    # BBP somatic Ih, fixed
    cell.branch(0).set("Ih_ehcn",        -45.0)

    # Dendrite (apical-like): apical biophysics block.
    cell.branch(1).set("NaTs2_t_ena",  50.0)
    cell.branch(1).set("K_Pst_ek",    -85.0)
    cell.branch(1).set("Im_ek",       -85.0)
    cell.branch(1).set("Ih_ehcn",     -45.0)

    cell.init_states(delta_t=_DT)

    # Record soma voltage — must be set BEFORE integrate().
    cell.branch(0).comp(0).record()

    # Mark each of the 13 CNN-facing params trainable.  Each entry maps to
    # exactly one branch, so entry_to_cnn_idx is the identity list.
    entry_to_cnn_idx = []
    for cnn_idx, (_, branch_idx, jax_key) in enumerate(_PARAM_MAP):
        cell.branch(branch_idx).make_trainable(jax_key)
        entry_to_cnn_idx.append(cnn_idx)

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
        stim_scale        = _STIM_SCALE,
    )


register("ball_and_stick_bbp", _spec)
