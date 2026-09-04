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


def unit_to_phys_np(unit: np.ndarray, centers: np.ndarray, logspans: np.ndarray) -> np.ndarray:
    """Numpy twin of the JAX version — used for data prep + sanity checks."""
    return centers * np.power(10.0, unit * logspans)


def unit_to_phys_jax(unit, centers_j, logspans_j):
    """JAX version of the same mapping.  `unit` shape: (..., P)."""
    import jax.numpy as jnp
    return centers_j * jnp.power(10.0, unit * logspans_j)


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
