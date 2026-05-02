"""Hybrid channel + voltage loss for the jaxley physics-supervised path.

When `params['use_voltage_loss']` is truthy, `Trainer` swaps `MSELoss`
for this module:

    loss = w_p * MSE(pred_unit, true_unit)
         + w_v * MSE(z(simulate(unit_to_phys(pred_unit))), z(true_volts_soma))

`w_p` and `w_v` come from the design YAML.  When `mask_channels=True`,
the channel term is skipped (used for fine-tuning on experimental data
where ground-truth params don't exist).

Voltage-space alignment
-----------------------
* The training pack stores per-sample-per-probe z-scored voltages
  (`format_bbp3_for_ML.py`).  We replay the same z-score on the simulated
  soma trace before MSE so the two sides live in the same normalized
  units.
* The simulated trace and the data trace can have different lengths
  (`spec.t_max / spec.dt` vs `T_data`).  We truncate to `min(T_sim, T_data)`
  along the time axis.  Phase 3 should reconcile `_T_MAX` with the data
  pack so the truncation is a no-op.
* Stim alignment: the cell's `default_stim_name` must match the stim
  used to generate the training data.  Phase 2 trusts the YAML; Phase 3
  generates ball-and-stick data with the canonical stim explicitly.
"""

from typing import Optional

import torch
import torch.nn as nn

from . import JaxleyBridge
from .jaxley_utils import phys_par_range_to_arrays


class HybridLoss(nn.Module):
    """See module docstring."""

    def __init__(
        self,
        cell_name: str,
        phys_par_range,
        channel_weight: float = 1.0,
        voltage_weight: float = 0.0,
        mask_channels: bool = False,
        stim_name: Optional[str] = None,
        soma_probe_index: int = 0,
        clamp_unit_tanh: bool = False,
        checkpoint_lengths=None,
        solver: str = "bwd_euler",
        fp64: bool = False,
    ):
        super().__init__()
        self.cell_name        = cell_name
        self.channel_weight   = float(channel_weight)
        self.voltage_weight   = float(voltage_weight)
        self.mask_channels    = bool(mask_channels)
        self.stim_name        = stim_name
        self.soma_probe_index = int(soma_probe_index)
        self.solver           = str(solver)
        # fp64=True pushes pred_phys to float64 before the bridge call.
        # JAX must have been started with `jax_enable_x64=True` (set
        # via JAX_ENABLE_X64=true env var in the slr) for this to take
        # effect.  Required for stable backward at t_max ≥ 250 ms with
        # the BBP channel set; otherwise fp32 NaN's the cumulative VJP.
        self.fp64             = bool(fp64)
        # `checkpoint_lengths`: e.g. [outer, inner] passed to jx.integrate.
        # Backward chains over `inner` stiff steps per VJP segment, which
        # keeps fp32 stable at large t_max (and bounds memory).  None = no
        # checkpointing (default).
        self.checkpoint_lengths = (
            tuple(checkpoint_lengths) if checkpoint_lengths else None
        )
        # When the CNN drives jaxley directly (voltage-only loss with no
        # channel anchor), an unbounded last-layer output can produce
        # unit values far outside [-1, 1]; via phys = center·10^(unit·logspan)
        # this lands on physiologically impossible conductances and the
        # integrator NaNs.  `clamp_unit_tanh=True` squashes pred_unit through
        # tanh first so the CNN can only ask for phys ∈ [center/10^logspan,
        # center·10^logspan] — keeping jaxley numerically stable.
        self.clamp_unit_tanh  = bool(clamp_unit_tanh)
        self._mse = nn.MSELoss()

        centers, logspans = phys_par_range_to_arrays(phys_par_range)
        # As buffers so .to(device) moves them with the module.
        self.register_buffer("_centers",  torch.from_numpy(centers))
        self.register_buffer("_logspans", torch.from_numpy(logspans))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _unit_to_phys(self, unit: torch.Tensor) -> torch.Tensor:
        """unit -> physical, in torch so gradients flow into pred_unit.

        Mirrors `toolbox.unitParamConvert` and `jaxley_utils.unit_to_phys_jax`:
            phys = center * 10**(unit * log_halfspan)
        """
        c = self._centers.to(unit.dtype)
        l = self._logspans.to(unit.dtype)
        return c * torch.pow(torch.tensor(10.0, dtype=unit.dtype, device=unit.device),
                             unit * l)

    @staticmethod
    def _zscore_time(x: torch.Tensor) -> torch.Tensor:
        """Per-sample z-score along the time axis (last)."""
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True) + 1e-6
        return (x - mean) / std

    def _voltage_loss(self, pred_unit: torch.Tensor, true_volts: torch.Tensor) -> torch.Tensor:
        """`pred_unit`  : (B, P) in unit-normalized space.
           `true_volts` : (B, T, C) with C = num_probes (after dataloader reshape)
                          or possibly (B, T, probes*stims) if multiple stims.
        Returns scalar voltage MSE (in z-scored mV space).
        """
        # Cast low-precision inputs (AMP fp16/bf16) up to fp32 before the
        # jaxley round-trip; preserve fp32/fp64 so the bridge's captured vjp
        # sees a matching grad dtype on backward.
        if pred_unit.dtype in (torch.float16, torch.bfloat16):
            pred_unit = pred_unit.float()
        if self.clamp_unit_tanh:
            pred_unit = torch.tanh(pred_unit)
        # fp64 promotion (only effective if JAX_ENABLE_X64=true at process
        # start).  Required for stable backward through stiff BBP dynamics.
        if self.fp64:
            pred_unit = pred_unit.double()
        pred_phys = self._unit_to_phys(pred_unit)
        v_sim = JaxleyBridge.simulate_batch(
            pred_phys, self.cell_name, self.stim_name,
            checkpoint_lengths=self.checkpoint_lengths,
            solver=self.solver,
        )
        # v_sim: (B, n_recorded, T_out).  Use first recorded compartment (soma).
        v_sim_soma = v_sim[:, 0, :]                                  # (B, T_out)
        v_sim_z    = self._zscore_time(v_sim_soma)                   # (B, T_out)

        # Pick the soma probe trace from the dataloader's z-scored voltages.
        # `true_volts` shape: (B, T, C).  `soma_probe_index` selects the
        # column corresponding to the soma trace — the dataloader keeps the
        # stim_select / probe_select order from the design YAML.
        v_true_soma = true_volts[..., self.soma_probe_index].to(v_sim_z.dtype)  # (B, T)

        # Truncate to overlap.  The dataloader pack may be 4000 bins (400 ms)
        # while the cell spec runs 5000 (500 ms), or vice-versa.  Until phase 3
        # makes these equal, take the leading prefix.
        T = min(v_sim_z.shape[-1], v_true_soma.shape[-1])
        mse = self._mse(v_sim_z[:, :T], v_true_soma[:, :T])
        # Cast back to fp32 for the rest of the training graph (so AMP /
        # GradScaler / Adam state stay in their original dtype).
        return mse.float() if self.fp64 else mse

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred_unit: torch.Tensor,
        true_unit: Optional[torch.Tensor],
        true_volts: torch.Tensor,
    ) -> torch.Tensor:
        ch = pred_unit.new_zeros(())
        if not self.mask_channels and self.channel_weight > 0:
            if true_unit is None:
                raise ValueError("channel loss enabled but true_unit is None")
            ch = self._mse(pred_unit, true_unit)

        v = pred_unit.new_zeros(())
        if self.voltage_weight > 0:
            v = self._voltage_loss(pred_unit, true_volts)

        return self.channel_weight * ch + self.voltage_weight * v


# ─────────────────────────────────────────────────────────────────────────
# adapter so Trainer can use a uniform 3-arg call site
# ─────────────────────────────────────────────────────────────────────────

class _ChannelOnlyAdapter(nn.Module):
    """Wrap MSELoss in a 3-arg signature `(pred, target, _images)` so the
    Trainer's call site is uniform whether or not voltage loss is enabled.
    Bit-equivalent to plain MSELoss when used as the criterion.
    """

    def __init__(self):
        super().__init__()
        self._mse = nn.MSELoss()

    def forward(self, pred_unit, true_unit, _images=None):
        return self._mse(pred_unit, true_unit)


# ─────────────────────────────────────────────────────────────────────────
# factory
# ─────────────────────────────────────────────────────────────────────────

def _log_jax_devices_once():
    """Print jax.devices() once per process and, in multi-rank runs, pin
    JAX's default device to this rank's local GPU.  Without this, every
    rank's JAX defaults to CudaDevice(id=0), so all jaxley sims pile on
    GPU 0 and the other GPUs sit idle — killing DDP scaling."""
    if getattr(_log_jax_devices_once, "_done", False):
        return
    try:
        import jax, os
        devices = jax.devices()
        local = int(os.environ.get("SLURM_LOCALID", 0))
        if len(devices) > 1 and 0 <= local < len(devices):
            jax.config.update("jax_default_device", devices[local])
            print(
                f"[HybridLoss] jax.devices()={devices} backend={jax.default_backend()} "
                f"-> pinned default to devices[{local}]={devices[local]}",
                flush=True,
            )
        else:
            print(
                f"[HybridLoss] jax.devices()={devices} backend={jax.default_backend()}",
                flush=True,
            )
    except Exception as e:
        print(f"[HybridLoss] jax probe failed: {e}", flush=True)
    _log_jax_devices_once._done = True


def _read_phys_par_range_from_h5(h5_path: str):
    """Read meta.JSON from the training pack on any rank (rank-agnostic)."""
    import h5py, json
    with h5py.File(h5_path, "r") as h5f:
        if "meta.JSON" not in h5f:
            raise RuntimeError(f"{h5_path}: missing meta.JSON dataset")
        blob = h5f["meta.JSON"][0]
    meta = json.loads(blob)
    rng = meta.get("input_meta", {}).get("phys_par_range") or meta.get("phys_par_range")
    if rng is None:
        raise RuntimeError(
            f"{h5_path}: meta.JSON has no input_meta.phys_par_range"
        )
    return rng


def build_hybrid_loss(params) -> nn.Module:
    """Return the criterion module appropriate for `params`.

    `params['use_voltage_loss']` truthy -> `HybridLoss`.
    Otherwise -> `_ChannelOnlyAdapter` (bit-equivalent to MSELoss).
    """
    if not params.get("use_voltage_loss"):
        return _ChannelOnlyAdapter()

    _log_jax_devices_once()
    vl = params.get("voltage_loss") or {}
    cell_name = vl.get("cell_name_for_sim")
    if cell_name is None:
        raise ValueError("voltage_loss.cell_name_for_sim is required when use_voltage_loss=True")

    # Optional t_max override.  Two modes:
    #   * a number  -> set _T_MAX directly (in ms).
    #   * "auto"    -> compute as dt_stim × len(stim_csv); makes the sim
    #                  span the full stim waveform exactly.
    # bwd_euler's fp32 adjoint can NaN over too many steps with stiff BBP
    # channels — if that happens, drop t_max manually or switch to fp64.
    t_max_override = vl.get("t_max_override")
    if t_max_override is not None:
        import importlib
        try:
            mod = importlib.import_module(f"toolbox.jaxley_cells.{cell_name}")
        except ImportError:
            raise RuntimeError(
                f"voltage_loss.t_max_override set but cell module "
                f"toolbox.jaxley_cells.{cell_name} cannot be imported"
            )
        if not hasattr(mod, "_T_MAX"):
            raise RuntimeError(
                f"voltage_loss.t_max_override set but {cell_name} has no _T_MAX module attr"
            )
        if isinstance(t_max_override, str) and t_max_override.lower() in ("auto", "stim"):
            # Resolve from the stim CSV: t_max = dt_stim × len(stim).
            from . import jaxley_cells, jaxley_utils as _jutils
            from pathlib import Path
            spec = jaxley_cells.get(cell_name)
            stim_name = vl.get("stim_name") or spec.default_stim_name
            stim_path = Path(spec.stim_dir) / f"{stim_name}.csv"
            stim_arr = _jutils.load_stim_csv(stim_path)
            t_max_override = float(len(stim_arr)) * float(spec.dt_stim)
            print(
                f"[HybridLoss] t_max_override=auto -> {t_max_override} ms "
                f"(from {stim_path.name}: {len(stim_arr)} samples × dt_stim={spec.dt_stim})",
                flush=True,
            )
        mod._T_MAX = float(t_max_override)
        from . import JaxleyBridge as _bridge
        _bridge.clear_cache()

    phys_par_range = vl.get("phys_par_range")
    if phys_par_range is None:
        # Fall back to the dataset's H5 meta, which Dataloader_H5 only sets on
        # rank 0 (`dataset.metaData`); reading the file ourselves is rank-safe.
        h5_path = params.get("full_h5name")
        if h5_path is None:
            raise RuntimeError(
                "build_hybrid_loss: need params['full_h5name'] (set by the "
                "dataloader) to read phys_par_range from the H5 pack"
            )
        phys_par_range = _read_phys_par_range_from_h5(h5_path)

    return HybridLoss(
        cell_name        = cell_name,
        phys_par_range   = phys_par_range,
        channel_weight   = float(vl.get("channel_weight", 1.0)),
        voltage_weight   = float(vl.get("voltage_weight", 0.0)),
        mask_channels    = bool(vl.get("mask_channels", False)),
        stim_name        = vl.get("stim_name"),
        soma_probe_index = int(vl.get("soma_probe_index", 0)),
        clamp_unit_tanh  = bool(vl.get("clamp_unit_tanh", False)),
        checkpoint_lengths = vl.get("checkpoint_lengths"),
        solver           = str(vl.get("solver", "bwd_euler")),
        fp64             = bool(vl.get("fp64", False)),
    )
