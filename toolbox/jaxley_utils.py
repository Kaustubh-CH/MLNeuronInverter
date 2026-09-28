"""Helpers shared across the jaxley voltage-loss path.

Currently hosts:

  * unit <-> physical param conversion, in pure-JAX so gradients flow
    through it.  Mirrors `toolbox/unitParamConvert.py` but is import-safe
    in environments without pandas/matplotlib.
  * stim loading + upsampling to internal dt (CPU numpy, called once per
    cell at setup).
"""

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# voltage normalization (fixed mean/std)
# ─────────────────────────────────────────────────────────────────────────
# Single source of truth for the fixed-scale voltage normalization, matching
# packBBP3/aggregate_Kaustubh.py:normalize_volts() (which hard-codes these two
# constants).  Unlike a per-sample z-score, a fixed global mean/std preserves
# the ABSOLUTE voltage scale — resting potential, spike height, subthreshold
# amplitude — which is exactly the information a per-trace z-score throws away.
# Both the stored data volts (data generators) and the simulated candidate
# volts (HybridLoss) must use this SAME transform, or the MSE compares two
# different spaces.
VOLT_NORM_MEAN = -60.0951997   # mV
VOLT_NORM_STD  = 18.95055671   # mV


def normalize_volts_fixed(volts):
    """(volts - VOLT_NORM_MEAN) / VOLT_NORM_STD.

    Works for numpy arrays and torch tensors alike (scalar arithmetic), so the
    same transform is applied to the pack volts at data-gen time and to the
    simulated candidate trace inside the physics loss.
    """
    return (volts - VOLT_NORM_MEAN) / VOLT_NORM_STD


# Fixed-scale normalization for a STIMULUS-current input channel (packs that
# feed the delivered stim to the CNN as probe 1).  A plain scale — no mean
# shift — keeps zero current at 0 in normalized space, so holding (-0.0496 nA
# -> -0.198) and the Roy amplitude ladder (Roy2000 peak ~0.5 nA -> ~2.0) stay
# interpretable.  Both the synthetic generator (ideal Roy*_icav2_5k CSV) and
# the exp packer (recorded Im, pA -> nA) must use this SAME constant.
STIM_NORM_SCALE_NA = 0.25   # nA


def normalize_stim_fixed(stim_nA):
    """stim (nA) -> stim / STIM_NORM_SCALE_NA; numpy or torch alike."""
    return stim_nA / STIM_NORM_SCALE_NA


# ─────────────────────────────────────────────────────────────────────────
# unit <-> physical
# ─────────────────────────────────────────────────────────────────────────

def phys_par_range_to_arrays(
    phys_par_range: Sequence[Sequence],
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the `[[center, log_halfspan, unit_str], ...]` list from
    `sum_train.yaml['input_meta']` into (centers, log_halfspans) float arrays.
    Units are discarded (jaxley assumes S/cm^2 and uF/cm^2).
    """
    centers  = np.asarray([row[0] for row in phys_par_range], dtype=np.float32)
    logspans = np.asarray([row[1] for row in phys_par_range], dtype=np.float32)
    return centers, logspans


def build_phys_par_range(cell_mod, log_halfspan: float = 0.5):
    """`[[center, log10_halfspan, unit], ...]` for `cell_mod.PARAM_KEYS`.

    Centres come from `cell_mod._DEFAULTS` and every parameter gets the same
    `log_halfspan` (0.5 = x3.16 each way, the CA3/ladder convention) UNLESS
    the cell module defines `PHYS_RANGE_OVERRIDES = {key: (center, span, unit[, "lin"])}`
    -- used by l5ttpc for the non-conductance entries, mirroring DL4neurons2
    run.py:get_random_params: a 4th field "lin" marks a LINEAR row
    (phys = center + u*span: cm 1.25 +- 0.75 uF/cm2, e_pas -75 +- 10 mV);
    3-field rows stay exponential (phys = center*10**(u*span)).  The 4th field
    travels into the pack meta, so the loss / evaluator invert the same map.
    Single source of truth for the data generators, the stim-scale sweep and
    the pack meta.  NOTE: the ladder L5 packs in HOME (2026-09-03/04) predate
    the "lin" rows -- their e_pas/cm rows are 3-field (geometric centre+span
    over the same bounds) and are still inverted exponentially, consistently.
    """
    over = getattr(cell_mod, "PHYS_RANGE_OVERRIDES", {}) or {}
    out = []
    for k in cell_mod.PARAM_KEYS:
        if k in over:
            row = list(over[k])
            out.append([float(row[0]), float(row[1]), str(row[2])] + ([str(row[3])] if len(row) > 3 else []))
        else:
            out.append([float(cell_mod._DEFAULTS[k]), float(log_halfspan), "S/cm^2"])
    return out


def unit_to_phys_torch(unit, centers_t, logspans_t, linear_t=None):
    """Torch twin of `unit_to_phys_np` (same row conventions).  `linear_t`: bool
    tensor from `phys_par_range_linear_mask`, or None for all-exponential."""
    import torch
    expo = centers_t * torch.pow(torch.tensor(10.0, dtype=centers_t.dtype, device=centers_t.device),
                                 unit * logspans_t)
    if linear_t is None or not bool(linear_t.any()):
        return expo
    return torch.where(linear_t, centers_t + unit * logspans_t, expo)


def phys_par_range_linear_mask(phys_par_range: Sequence[Sequence]) -> np.ndarray:
    """Bool array: True where a phys_par_range row carries the 4th field "lin".

    Row conventions (mirrors DL4neurons2 run.py:get_random_params, u ~ U(-1,1)):
        [center, log10_halfspan, unit]          exponential: phys = center * 10**(u*span)
        [mid,    halfwidth,      unit, "lin"]   linear:      phys = mid + u*halfwidth
    BBP samples e_pas_all and cm_* linearly (b + a*u over [-85,-65] mV, [0.5,2] uF/cm2)
    and every conductance log-uniformly over +-1 decade; the CA3 / HH cells have
    no linear parameters, so a 3-field row (the historical format) is exponential.
    """
    return np.asarray([len(row) > 3 and str(row[3]).lower().startswith("lin") for row in phys_par_range],
                      dtype=bool)


def unit_to_phys_np(unit: np.ndarray, centers: np.ndarray, logspans: np.ndarray,
                    linear: np.ndarray = None) -> np.ndarray:
    """Numpy twin of the JAX version — used for data prep + sanity checks.
    `linear` (bool, P) selects rows mapped as mid + u*halfwidth; None = all exponential."""
    expo = centers * np.power(10.0, unit * logspans)
    if linear is None or not np.any(linear):
        return expo
    return np.where(np.asarray(linear)[None, :] if unit.ndim == 2 else np.asarray(linear),
                    centers + unit * logspans, expo)


def unit_to_phys_jax(unit, centers_j, logspans_j, linear_j=None):
    """JAX version of the same mapping.  `unit` shape: (..., P)."""
    import jax.numpy as jnp
    expo = centers_j * jnp.power(10.0, unit * logspans_j)
    if linear_j is None:
        return expo
    return jnp.where(linear_j, centers_j + unit * logspans_j, expo)


# ─────────────────────────────────────────────────────────────────────────
# stim loading
# ─────────────────────────────────────────────────────────────────────────

def load_stim_csv(path: Path) -> np.ndarray:
    """Load a DL4neurons2-style stim CSV as a 1D float32 array (nA)."""
    return np.loadtxt(str(path)).astype(np.float32)


def upsample_stim(stim: np.ndarray, dt_stim: float, dt_sim: float, t_max: float) -> np.ndarray:
    """Linear-interpolate `stim` (sampled every dt_stim ms) onto the
    internal integration grid (dt_sim ms) over [0, t_max) ms."""
    t_stim = np.arange(len(stim)) * dt_stim
    t_sim  = np.arange(0, t_max, dt_sim)
    return np.interp(t_sim, t_stim, stim).astype(np.float32)


def downsample_step(dt_sim: float, dt_stim: float) -> int:
    """Integer decimation factor to map the internal trace back to the
    10 kHz recording grid the CNN is trained on."""
    return int(round(dt_stim / dt_sim))
