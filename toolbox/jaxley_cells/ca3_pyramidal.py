"""CA3 Pyramidal Neuron (single-comp soma) — Jaxley port.

Mirrors the NEURON model under
    /global/homes/k/ktub1999/mainDL4/DL4neurons2/Adapting CA3 Pyramidal Neuron/

Geometry (single soma, nseg=1):
    L = 50 µm,  diam = 50 µm,  cm = 1.41 µF/cm²,  Ra = 150 Ω·cm,
    celsius = 34 °C,  v_init = -65 mV.

Active channels (6 conductances):
    leak,  na3,  kdr,  kap,  km,  kd

Trainable surface (CNN-output order):
    0  CA3_g_leak       (leak)
    1  CA3_gbar_na3     (na3)
    2  CA3_gkdrbar_kdr  (kdr)
    3  CA3_gkabar_kap   (kap)
    4  CA3_gbar_km      (km)
    5  CA3_gkdbar_kd    (kd)

Reversal potentials (verbatim from morphology_mechanisms.hoc):
    ena = +55,  ek = -90,  e_leak = +93.9115 (⚠ likely typo in source).
"""

from pathlib import Path

from . import CellSpec, register

_STIM_DIR = Path("/pscratch/sd/k/ktub1999/main/DL4neurons2/stims")

_DT_STIM = 0.1     # ms
_DT      = 0.1     # ms (use 0.025 for spike-time precision <0.1 ms)
_T_MAX   = 500.0   # ms
_V_INIT  = -65.0   # mV (matches NEURON h.v_init)

# Geometry (verbatim from morphology_mechanisms.hoc).
_L    = 50.0       # µm
_DIAM = 50.0       # µm
_CM   = 1.41       # µF/cm²
_RA   = 150.0      # Ω·cm

PARAM_KEYS = [
    "CA3_g_leak",
    "CA3_gbar_na3",
    "CA3_gkdrbar_kdr",
    "CA3_gkabar_kap",
    "CA3_gbar_km",
    "CA3_gkdbar_kd",
]

# CNN-name -> (jaxley channel-param key)
_PARAM_MAP = [
    ("CA3_g_leak",       "Leak_CA3_g"),
    ("CA3_gbar_na3",     "Na3_gbar"),
    ("CA3_gkdrbar_kdr",  "Kdr_ca1_gkdrbar"),
    ("CA3_gkabar_kap",   "Kap_rox_gkabar"),
    ("CA3_gbar_km",      "Km_ca3_gbar"),
    ("CA3_gkdbar_kd",    "Kd_ca3_gkdbar"),
]

_DEFAULTS = {
    "CA3_g_leak":      3.9417e-5,
    "CA3_gbar_na3":    0.04,
    "CA3_gkdrbar_kdr": 0.01,
    "CA3_gkabar_kap":  0.04,
    "CA3_gbar_km":     5.2e-4,
    "CA3_gkdbar_kd":   2.5e-4,
}


def _build():
    import jaxley as jx
    from toolbox.jaxley_channels.ca3_channels import (
        Leak_CA3, Na3, Kdr_ca1, Kap_rox, Km_ca3, Kd_ca3,
    )

    soma = jx.Branch([jx.Compartment()], ncomp=1)
    cell = jx.Cell([soma], parents=[-1])

    # Geometry — radius = diam/2 = 25 µm.
    cell.branch(0).set("length", _L)
    cell.branch(0).set("radius", _DIAM / 2.0)
    cell.set("axial_resistivity", _RA)
    cell.set("capacitance", _CM)
    cell.set("v", _V_INIT)

    cell.branch(0).insert(Leak_CA3())
    cell.branch(0).insert(Na3())
    cell.branch(0).insert(Kdr_ca1())
    cell.branch(0).insert(Kap_rox())
    cell.branch(0).insert(Km_ca3())
    cell.branch(0).insert(Kd_ca3())

    # Default conductances + reversal potentials.
    cell.branch(0).set("Leak_CA3_g",      _DEFAULTS["CA3_g_leak"])
    cell.branch(0).set("Leak_CA3_e",      93.9115)            # verbatim from hoc
    cell.branch(0).set("Na3_gbar",        _DEFAULTS["CA3_gbar_na3"])
    cell.branch(0).set("Na3_ena",         55.0)
    cell.branch(0).set("Kdr_ca1_gkdrbar", _DEFAULTS["CA3_gkdrbar_kdr"])
    cell.branch(0).set("Kdr_ca1_ek",     -90.0)
    cell.branch(0).set("Kap_rox_gkabar",  _DEFAULTS["CA3_gkabar_kap"])
    cell.branch(0).set("Kap_rox_ek",     -90.0)
    cell.branch(0).set("Km_ca3_gbar",     _DEFAULTS["CA3_gbar_km"])
    cell.branch(0).set("Km_ca3_ek",      -90.0)
    cell.branch(0).set("Kd_ca3_gkdbar",   _DEFAULTS["CA3_gkdbar_kd"])
    cell.branch(0).set("Kd_ca3_ek",      -90.0)

    cell.init_states(delta_t=_DT)
    cell.branch(0).comp(0).record()

    # Mark all 6 conductances trainable.
    entry_to_cnn_idx = []
    for cnn_idx, (_, jax_key) in enumerate(_PARAM_MAP):
        cell.branch(0).make_trainable(jax_key)
        entry_to_cnn_idx.append(cnn_idx)

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
    )


register("ca3_pyramidal", _spec)
