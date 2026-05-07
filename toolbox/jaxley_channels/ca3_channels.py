"""Jaxley ports of the CA3 Pyramidal Neuron channel kinetics.

Source mod files (Migliore CA3 model):
    /global/homes/k/ktub1999/mainDL4/DL4neurons2/Adapting CA3 Pyramidal Neuron/mechanisms/
        leak.mod, na3n.mod, kdrca1.mod, kaprox.mod, km.mod, kd.mod, cacumm.mod

All rate equations are direct transcriptions of the .mod sources at celsius=34
(qt computed per-channel below).

Notes
-----
* na3 has a 3rd "slow inactivation" gate `s` driven by `ar`.  At the hoc default
  `ar=1` (no slow inactivation), `sinf` collapses to 1.0 and `s` saturates at
  1 — but we still integrate the gate so the temporal extension matches NEURON
  byte-for-byte if `ar` is later perturbed.
* kdrca1 vs kd are algebraically identical; we keep two classes for clarity
  (and so that ḡ-only sweeps train them as separate parameters).
* cacum (Ca buffer) is INTENTIONALLY OMITTED.  This CA3 model inserts no Ca
  channels (cal2/can2/cat/cagk are present in mechanisms/ but not in the
  hoc), so ica=0 always.  cacum's only effect is to track cai which would
  stay at cai0=0 — no current contribution.
* `e_leak = 93.9115` mV in the source hoc is suspicious (likely a typo —
  conventional CA3 leak reversal is ~-65 mV).  We port the value verbatim
  per user direction; the comparison test will surface any inconsistency.

Pattern follows /pscratch/sd/k/ktub1999/Neuron_Jaxley/bbp_channels_jaxley.py
(BBP L5_TTPC1 reference port).
"""
from typing import Optional

import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler


# ── thermodynamic constant used by Migliore-style "alpn/betn" rate forms ────
# NEURON .mod expression:    1e-3 * zeta * (v - vhalf - sh) * 9.648e4 / (8.315 * (273.16+celsius))
# At celsius=34: 1e-3 * 9.648e4 / (8.315 * 307.16) = 0.037783 / mV (per unit zeta).
_RT_FACTOR_34C = 1e-3 * 9.648e4 / (8.315 * (273.16 + 34.0))   # ≈ 0.037783


# ── singularity-safe trap0 used by na3 ─────────────────────────────────────
# NEURON expression:    a*(v-th)/(1 - exp(-(v-th)/q));  limit = a*q as v->th
def _trap0(v, th, a, q):
    dv = v - th
    return jnp.where(jnp.abs(dv) < 1e-6, a * q, a * dv / (1.0 - jnp.exp(-dv / q)))


# ═══════════════════════════════════════════════════════════════════════════
# Leak — passive
# ═══════════════════════════════════════════════════════════════════════════
class Leak_CA3(Channel):
    """Passive leak (NONSPECIFIC_CURRENT i = g*(v-e))."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        p = self._name
        self.channel_params = {
            f"{p}_g": 3.9417e-5,
            f"{p}_e": 93.9115,        # verbatim from CA3 hoc — see module docstring
        }
        self.channel_states = {}
        self.current_name = f"i_{p}"

    def update_states(self, states, dt, v, params):
        return {}

    def compute_current(self, states, v, params):
        p = self._name
        return params[f"{p}_g"] * (v - params[f"{p}_e"])

    def init_state(self, states, v, params, delta_t):
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# na3 — transient Na  (m^3 h s)
# ═══════════════════════════════════════════════════════════════════════════
class Na3(Channel):
    """na3n.mod port.  States: m, h, s (slow inact.).  q10=2 ref 24°C."""

    # qt at celsius=34, q10=2, T_ref=24:  2^((34-24)/10) = 2.0
    QT = 2.0 ** ((34.0 - 24.0) / 10.0)

    # Constants from PARAMETER block (verbatim).
    SH = 24.0
    THA, QA = -30.0, 7.2
    RA, RB = 0.4, 0.124
    THI1, THI2 = -45.0, -45.0
    QD, QG = 1.5, 1.5
    RD, RG = 0.03, 0.01
    THINF, QINF = -50.0, 4.0
    MMIN, HMIN = 0.02, 0.5
    # Slow-inactivation block.
    VHALFS = -60.0
    A0S = 0.0003
    ZETAS = 12.0
    GMS = 0.2
    SMAX = 10.0
    VVH, VVS = -58.0, 2.0
    AR = 1.0   # 1 = no slow inactivation; sinf saturates to 1

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        p = self._name
        self.channel_params = {
            f"{p}_gbar": 0.04,
            f"{p}_ena":  55.0,
        }
        self.channel_states = {
            f"{p}_m": 0.0,
            f"{p}_h": 1.0,
            f"{p}_s": 1.0,
        }
        self.current_name = f"i_{p}"

    def _rates(self, v):
        """Return (minf, mtau, hinf, htau, sinf, taus)."""
        # m gate
        a_m = _trap0(v,  self.THA + self.SH,  self.RA, self.QA)
        b_m = _trap0(-v, -self.THA - self.SH, self.RB, self.QA)
        mtau = 1.0 / (a_m + b_m) / self.QT
        mtau = jnp.maximum(mtau, self.MMIN)
        minf = a_m / (a_m + b_m)
        # h gate
        a_h = _trap0(v,  self.THI1 + self.SH, self.RD, self.QD)
        b_h = _trap0(-v, -self.THI2 - self.SH, self.RG, self.QG)
        htau = 1.0 / (a_h + b_h) / self.QT
        htau = jnp.maximum(htau, self.HMIN)
        hinf = 1.0 / (1.0 + jnp.exp((v - self.THINF - self.SH) / self.QINF))
        # s gate (slow inactivation)
        c = 1.0 / (1.0 + jnp.exp((v - self.VVH - self.SH) / self.VVS))
        sinf = c + self.AR * (1.0 - c)
        # taus = bets(v) / (a0s * (1 + alps(v)))
        # alps and bets share the celsius-dependent factor; at 34°C,
        # 9.648e4 / (8.315 * 307.16) = 37.783 / V = 0.037783 / mV.
        # alps argument: zetas*(v-vhalfs-sh)*0.037783; bets: same * gms
        kT = 0.037783
        zarg = self.ZETAS * kT * (v - self.VHALFS - self.SH)
        alps = jnp.exp(zarg)
        bets = jnp.exp(zarg * self.GMS)
        taus = bets / (self.A0S * (1.0 + alps))
        taus = jnp.maximum(taus, self.SMAX)
        return minf, mtau, hinf, htau, sinf, taus

    def update_states(self, states, dt, v, params):
        p = self._name
        minf, mtau, hinf, htau, sinf, taus = self._rates(v)
        new_m = exponential_euler(states[f"{p}_m"], dt, minf, mtau)
        new_h = exponential_euler(states[f"{p}_h"], dt, hinf, htau)
        new_s = exponential_euler(states[f"{p}_s"], dt, sinf, taus)
        return {f"{p}_m": new_m, f"{p}_h": new_h, f"{p}_s": new_s}

    def compute_current(self, states, v, params):
        p = self._name
        m, h, s = states[f"{p}_m"], states[f"{p}_h"], states[f"{p}_s"]
        g = params[f"{p}_gbar"] * m * m * m * h * s
        return g * (v - params[f"{p}_ena"])

    def init_state(self, states, v, params, delta_t):
        p = self._name
        minf, _, hinf, _, sinf, _ = self._rates(v)
        return {f"{p}_m": minf, f"{p}_h": hinf, f"{p}_s": sinf}


# ═══════════════════════════════════════════════════════════════════════════
# kdr (kdrca1.mod) — delayed rect K   (single state n)
# ═══════════════════════════════════════════════════════════════════════════
class Kdr_ca1(Channel):
    """kdrca1.mod port.  q10=1 (no temp scaling)."""

    QT = 1.0
    SH = 24.0
    VHALFN = 13.0
    A0N = 0.02
    ZETAN = -3.0
    GMN = 0.7
    NMAX = 2.0   # tau floor; weird "if (taun<nmax) {taun=nmax/qt}" — see source

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        p = self._name
        self.channel_params = {
            f"{p}_gkdrbar": 0.01,
            f"{p}_ek":     -90.0,
        }
        self.channel_states = {f"{p}_n": 0.0}
        self.current_name = f"i_{p}"

    def _rates(self, v):
        kT = _RT_FACTOR_34C
        a = jnp.exp(self.ZETAN * kT * (v - self.VHALFN - self.SH))
        b = jnp.exp(self.ZETAN * self.GMN * kT * (v - self.VHALFN - self.SH))
        ninf = 1.0 / (1.0 + a)
        taun = b / (self.QT * self.A0N * (1.0 + a))
        # NEURON: if (taun<nmax) {taun=nmax/qt}.  With qt=1, this floors taun at NMAX.
        taun = jnp.where(taun < self.NMAX, self.NMAX / self.QT, taun)
        return ninf, taun

    def update_states(self, states, dt, v, params):
        p = self._name
        ninf, taun = self._rates(v)
        return {f"{p}_n": exponential_euler(states[f"{p}_n"], dt, ninf, taun)}

    def compute_current(self, states, v, params):
        p = self._name
        return params[f"{p}_gkdrbar"] * states[f"{p}_n"] * (v - params[f"{p}_ek"])

    def init_state(self, states, v, params, delta_t):
        p = self._name
        ninf, _ = self._rates(v)
        return {f"{p}_n": ninf}


# ═══════════════════════════════════════════════════════════════════════════
# kap (kaprox.mod) — A-type K  (n*l)
# ═══════════════════════════════════════════════════════════════════════════
class Kap_rox(Channel):
    """kaprox.mod port.  q10=5 ref 24°C; qtl=1 for l gate."""

    QT = 5.0 ** ((34.0 - 24.0) / 10.0)   # = 5.0
    QTL = 1.0
    SH = 24.0
    VHALFN, VHALFL = 11.0, -56.0
    A0N, A0L = 0.05, 0.05
    ZETAN, ZETAL = -1.5, 3.0
    GMN, GML = 0.55, 1.0
    LMIN, NMIN = 2.0, 0.1
    PW, TQ, QQ = -1.0, -40.0, 5.0

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        p = self._name
        self.channel_params = {
            f"{p}_gkabar": 0.04,
            f"{p}_ek":    -90.0,
        }
        self.channel_states = {f"{p}_n": 0.0, f"{p}_l": 1.0}
        self.current_name = f"i_{p}"

    def _rates(self, v):
        kT = _RT_FACTOR_34C
        # n gate — voltage-dependent zeta.
        zeta_n = self.ZETAN + self.PW / (1.0 + jnp.exp((v - self.TQ - self.SH) / self.QQ))
        a_n = jnp.exp(zeta_n * kT * (v - self.VHALFN - self.SH))
        b_n = jnp.exp(zeta_n * self.GMN * kT * (v - self.VHALFN - self.SH))
        ninf = 1.0 / (1.0 + a_n)
        taun = b_n / (self.QT * self.A0N * (1.0 + a_n))
        taun = jnp.maximum(taun, self.NMIN)

        # l gate — fixed zeta.
        a_l = jnp.exp(self.ZETAL * kT * (v - self.VHALFL - self.SH))
        linf = 1.0 / (1.0 + a_l)
        taul = 0.26 * (v + 50.0 - self.SH) / self.QTL
        taul = jnp.maximum(taul, self.LMIN / self.QTL)

        return ninf, taun, linf, taul

    def update_states(self, states, dt, v, params):
        p = self._name
        ninf, taun, linf, taul = self._rates(v)
        new_n = exponential_euler(states[f"{p}_n"], dt, ninf, taun)
        new_l = exponential_euler(states[f"{p}_l"], dt, linf, taul)
        return {f"{p}_n": new_n, f"{p}_l": new_l}

    def compute_current(self, states, v, params):
        p = self._name
        n, l = states[f"{p}_n"], states[f"{p}_l"]
        return params[f"{p}_gkabar"] * n * l * (v - params[f"{p}_ek"])

    def init_state(self, states, v, params, delta_t):
        p = self._name
        ninf, _, linf, _ = self._rates(v)
        return {f"{p}_n": ninf, f"{p}_l": linf}


# ═══════════════════════════════════════════════════════════════════════════
# km — M-current  (single state m, exponent st=1)
# ═══════════════════════════════════════════════════════════════════════════
class Km_ca3(Channel):
    """km.mod port.  q10 computed but NEVER applied to tau (verbatim bug)."""

    # qt is computed in NEURON but never used; we replicate that behaviour.
    SH = 24.0
    VHALFL = -40.0
    KL = -10.0
    VHALFT = -42.0
    A0T = 0.003
    ZETAT = 7.0
    GMT = 0.4
    B0 = 60.0

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        p = self._name
        self.channel_params = {
            f"{p}_gbar": 5.2e-4,
            f"{p}_ek":  -90.0,
        }
        self.channel_states = {f"{p}_m": 0.0}
        self.current_name = f"i_{p}"

    def _rates(self, v):
        # NEURON expression for inf:    1/(1 + exp((v - vhalfl - sh)/kl)),  kl=-10
        inf = 1.0 / (1.0 + jnp.exp((v - self.VHALFL - self.SH) / self.KL))
        # alpt/bett use a different exponent prefactor (0.0378, NOT 0.037783):
        # in km.mod the rate has the literal "0.0378*zetat*(v-vhalft-sh)" form.
        alpt = jnp.exp(0.0378 * self.ZETAT * (v - self.VHALFT - self.SH))
        bett = jnp.exp(0.0378 * self.ZETAT * self.GMT * (v - self.VHALFT - self.SH))
        tau = self.B0 + bett / (self.A0T * (1.0 + alpt))
        return inf, tau

    def update_states(self, states, dt, v, params):
        p = self._name
        inf, tau = self._rates(v)
        return {f"{p}_m": exponential_euler(states[f"{p}_m"], dt, inf, tau)}

    def compute_current(self, states, v, params):
        p = self._name
        return params[f"{p}_gbar"] * states[f"{p}_m"] * (v - params[f"{p}_ek"])

    def init_state(self, states, v, params, delta_t):
        p = self._name
        inf, _ = self._rates(v)
        return {f"{p}_m": inf}


# ═══════════════════════════════════════════════════════════════════════════
# kd — D-type K  (single state n).  Same algebra as kdr_ca1 with sh=0 and
# inverted-sign zetan.  Kept separate so trainable ḡ stays disjoint.
# ═══════════════════════════════════════════════════════════════════════════
class Kd_ca3(Channel):
    """kd.mod port.  q10=1.  sh=0 (NOT 24 like kdr/kap)."""

    QT = 1.0
    SH = 0.0
    VHALFN = -33.0
    A0N = 0.01
    ZETAN = 3.0
    GMN = 0.7
    NMAX = 2.0

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        p = self._name
        self.channel_params = {
            f"{p}_gkdbar": 2.5e-4,
            f"{p}_ek":    -90.0,
        }
        self.channel_states = {f"{p}_n": 0.0}
        self.current_name = f"i_{p}"

    def _rates(self, v):
        kT = _RT_FACTOR_34C
        a = jnp.exp(self.ZETAN * kT * (v - self.VHALFN - self.SH))
        b = jnp.exp(self.ZETAN * self.GMN * kT * (v - self.VHALFN - self.SH))
        ninf = 1.0 / (1.0 + a)
        taun = b / (self.QT * self.A0N * (1.0 + a))
        taun = jnp.where(taun < self.NMAX, self.NMAX / self.QT, taun)
        return ninf, taun

    def update_states(self, states, dt, v, params):
        p = self._name
        ninf, taun = self._rates(v)
        return {f"{p}_n": exponential_euler(states[f"{p}_n"], dt, ninf, taun)}

    def compute_current(self, states, v, params):
        p = self._name
        return params[f"{p}_gkdbar"] * states[f"{p}_n"] * (v - params[f"{p}_ek"])

    def init_state(self, states, v, params, delta_t):
        p = self._name
        ninf, _ = self._rates(v)
        return {f"{p}_n": ninf}
