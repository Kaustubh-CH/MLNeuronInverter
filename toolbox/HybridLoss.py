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
* The training pack stores voltages normalized with a FIXED global
  mean/std (`VOLT_NORM_MEAN`/`VOLT_NORM_STD`, matching
  `packBBP3/aggregate_Kaustubh.py`).  We apply the same fixed normalization
  to the simulated candidate trace before MSE so the two sides live in the
  same normalized units — and, unlike a per-sample z-score, the absolute
  voltage scale (resting level, spike height) is preserved on both sides.
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
from .jaxley_utils import (
    phys_par_range_to_arrays, normalize_volts_fixed,
    VOLT_NORM_MEAN, VOLT_NORM_STD,
)
from .soft_efel import (
    soft_efel_features, FEATURE_SCALES, STRONG_FEATURES,
    ALL_FEATURES as _EFEL_FEATURES,   # eFEL surrogates + L3 dV/dt features
)
from .soft_dtw import soft_dtw_loss


class _GradScale(torch.autograd.Function):
    """Straight-through per-parameter gradient reweighting.

    Forward is the identity, so the loss VALUE is unchanged; backward
    multiplies the incoming gradient (B, P) column-wise by a fixed weight
    vector `w` (P,).  Used to precondition the voltage-loss gradient so that
    each output parameter receives an ~equal-magnitude signal instead of one
    scaled by its own sensitivity (∂V/∂θ_p).  See `HybridLoss._voltage_loss`.
    """

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(w)
        return x

    @staticmethod
    def backward(ctx, g):
        (w,) = ctx.saved_tensors
        return g * w.to(g.dtype), None


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
        sim_t_skip_ms: float = 0.0,
        sim_dt_ms: float = 0.1,
        pad_batch_size: Optional[int] = None,
        probe_loss_indices=None,
        stim_names_multi=None,
        efel_weight: float = 0.0,
        mse_weight: float = 1.0,
        efel_features=None,
        efel_huber_delta: float = 1.0,
        efel_k: float = 2.0,
        efel_thr: float = -20.0,
        efel_nmax: int = 64,
        grad_precond_weights=None,
        # ── L1 soft-DTW (timing-robust distance); dtw_weight=0 -> off ──
        dtw_weight: float = 0.0,
        dtw_gamma: float = 0.1,
        dtw_n_points: int = 200,
        dtw_band_ms: float = 8.0,
        # ── L2 low-pass MSE (subthreshold-envelope match); lowpass_ms=0 -> off ──
        lowpass_ms: float = 0.0,
        # ── O1 randomized smoothing of the objective; smooth_sigma=0 -> off ──
        smooth_sigma: float = 0.0,
        smooth_samples: int = 1,
        # ── O2/L2 curriculum: per-epoch schedule for any of the above knobs ──
        schedule=None,
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
        # `sim_t_skip_ms`: drop the first N ms of the simulated trace
        # before z-scoring + MSE.  Use this when the data H5 trimmed an
        # initial pre-stim window (e.g. L5_TTPC1 packs trim 100 ms, so
        # data bin 0 corresponds to sim t = 100 ms).  Without this, the
        # first 100 ms of jaxley's output is compared against data that
        # actually starts at sim t = 100 ms — a 100 ms misalignment.
        self.sim_dt_ms = float(sim_dt_ms)
        self.sim_t_skip_bins = max(0, int(round(float(sim_t_skip_ms) / float(sim_dt_ms))))
        # `pad_batch_size`: canonical per-rank batch size.  When a mini-batch is
        # shorter than this (the last batch of an epoch when drop_last is off),
        # `_voltage_loss` pads pred_phys up to this size before the jaxley call
        # and slices the result back — so XLA doesn't recompile the vmapped sim
        # for a smaller batch shape.  None disables padding.
        self.pad_batch_size = int(pad_batch_size) if pad_batch_size else None
        # ── Multi-channel voltage supervision (opt-in; default None = soma-only) ──
        # `probe_loss_indices` (Exp 2 multi-probe): list of SIM-probe indices
        #   (the cell's .record() order) to supervise.  The i-th sim probe is
        #   compared against data channel i (true_volts[..., i]), so the list
        #   MUST be in the same order as the data's probe axis (probsSelect order).
        #   e.g. [0,1,2,3] for a [soma, axon, apical, dend] cell + --probsSelect 0 1 2 3.
        # `stim_names_multi` (Exp 3 multi-stim): list of stim CSV stems.  The loss
        #   simulates the SAME pred_phys under each stim, takes the soma trace
        #   (sim probe 0), and compares it against data channel i — so the data
        #   pack's channel order MUST match this stim order.
        # COMBINED (Exp 1): if BOTH are set, the loss simulates every stim and
        #   supervises every probe of each — producing len(stims)*len(probes)
        #   channels in STIM-MAJOR, probe-inner order:
        #   [s0p0, s0p1, ..., s0pK, s1p0, ...]. The data pack's channel order
        #   MUST match (gen loops stim-outer, probe-inner).
        self.probe_loss_indices = (
            [int(k) for k in probe_loss_indices] if probe_loss_indices else None
        )
        self.stim_names_multi = (
            [str(s) for s in stim_names_multi] if stim_names_multi else None
        )
        self._mse = nn.MSELoss()

        # ── L1 soft-DTW / L2 low-pass / O1 randomized smoothing (all opt-in) ─────
        # These attack the "cliffy" (non-smooth) V(theta) landscape that makes
        # pointwise voltage-MSE plateau: soft-DTW aligns spikes in time before
        # penalising, low-pass matches the smooth subthreshold envelope, and
        # randomized smoothing averages the loss over Gaussian jitter of the
        # predicted params so the descent direction survives spike-timing cliffs.
        # `dtw_weight` and `smooth_sigma` may be mutated per-epoch by the Trainer
        # (curriculum); everything is a no-op at its default so behaviour is
        # bit-identical unless enabled.
        self.dtw_weight   = float(dtw_weight)
        self.dtw_gamma    = float(dtw_gamma)
        self.dtw_n_points = int(dtw_n_points)
        self.dtw_band_ms  = float(dtw_band_ms)
        self.lowpass_ms   = float(lowpass_ms)
        self.smooth_sigma   = float(smooth_sigma)
        self.smooth_samples = max(1, int(smooth_samples))
        # Per-epoch curriculum: {attr: {start, end, epochs}} linearly ramps the
        # named scalar attribute from `start` to `end` over `epochs` epochs, then
        # holds at `end`.  Applied by `set_epoch`, which the Trainer calls once
        # per epoch.  None -> weights are static (default).
        self._schedule = dict(schedule) if schedule else None

        # ── Differentiable soft-eFEL voltage loss (opt-in; default off) ──────────
        # The voltage term becomes a blend of the z-scored MSE anchor and a
        # feature-matching loss on the soft-eFEL features (toolbox/soft_efel.py):
        #     v = mse_weight * MSE(z(sim), z(data))
        #       + efel_weight * mean_f Huber( (soft_f(sim) - soft_f(data)) / scale_f )
        # `efel_weight=0` (default) -> pure MSE, bit-identical to prior behavior.
        # The soft features run on RAW mV: the sim channel is jaxley's raw output;
        # the data channel is de-normalized from the pack's fixed z-space back to
        # mV (v_raw = v_norm * VOLT_NORM_STD + VOLT_NORM_MEAN) so both sides sit in
        # the physical units the -20 mV spike threshold / amplitudes assume.
        # Only STRONG_FEATURES carry the gradient by default (r>=0.79 vs real eFEL);
        # the data-side features are a constant target (computed under no_grad).
        self.efel_weight = float(efel_weight)
        self.mse_weight  = float(mse_weight)
        self.efel_huber_delta = float(efel_huber_delta)
        self.efel_k    = float(efel_k)
        self.efel_thr  = float(efel_thr)
        self.efel_nmax = int(efel_nmax)
        feats = list(efel_features) if efel_features else list(STRONG_FEATURES)
        bad = [f for f in feats if f not in _EFEL_FEATURES]
        if bad:
            raise ValueError(f"unknown efel_features {bad}; valid: {_EFEL_FEATURES}")
        self.efel_features = feats
        self._efel_scales = {f: float(FEATURE_SCALES[f]) for f in feats}

        centers, logspans = phys_par_range_to_arrays(phys_par_range)
        # As buffers so .to(device) moves them with the module.
        self.register_buffer("_centers",  torch.from_numpy(centers))
        self.register_buffer("_logspans", torch.from_numpy(logspans))

        # ── Per-parameter gradient preconditioning (opt-in; default off) ─────────
        # `grad_precond_weights`: length-P multiplier applied to the VOLTAGE-loss
        # gradient w.r.t. each output parameter (via `_GradScale`).  Counteracts
        # the sensitivity-weighting of voltage-MSE — a channel that barely bends
        # the trace (small ∂V/∂θ_p) otherwise gets a starved gradient and never
        # converges even when it is fully identifiable.  Weights are typically
        # w_p ∝ sensitivity_p^(-exponent), normalized to unit geometric mean so
        # the overall step scale (and hence LR) is preserved.  None -> no-op
        # (raw voltage gradient, bit-identical to prior behavior).  The channel
        # term is left untouched — it is already well-conditioned (Hessian ~ I).
        if grad_precond_weights is not None:
            w = torch.as_tensor(grad_precond_weights, dtype=torch.float32)
            if w.ndim != 1:
                raise ValueError("grad_precond_weights must be a 1-D length-P vector")
            self.register_buffer("_grad_precond_w", w)
        else:
            self._grad_precond_w = None

        # Apply epoch-0 curriculum values now that every schedulable attr exists.
        if self._schedule:
            self.set_epoch(0)

    # ------------------------------------------------------------------
    # curriculum
    # ------------------------------------------------------------------

    _SCHEDULABLE = ("mse_weight", "efel_weight", "dtw_weight",
                    "lowpass_ms", "smooth_sigma", "channel_weight", "voltage_weight")

    def set_epoch(self, epoch: int) -> None:
        """Update scheduled loss-term weights for `epoch` (O2/L2 curriculum).

        Each entry `{attr: {start, end, epochs}}` linearly interpolates `attr`
        from `start` to `end` over `epochs` epochs, then holds at `end`.  Only the
        whitelisted scalar knobs in `_SCHEDULABLE` may be scheduled.  No-op when no
        schedule was provided.
        """
        if not self._schedule:
            return
        for attr, spec in self._schedule.items():
            if attr not in self._SCHEDULABLE:
                raise ValueError(f"schedule: '{attr}' is not schedulable; "
                                 f"choose from {self._SCHEDULABLE}")
            start = float(spec["start"]); end = float(spec["end"])
            n = max(1, int(spec.get("epochs", 1)))
            frac = min(max(epoch / n, 0.0), 1.0)
            setattr(self, attr, start + (end - start) * frac)

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
    def _normalize_volts(x: torch.Tensor) -> torch.Tensor:
        """Fixed-scale voltage normalization, matching the data pack.

        Uses the SAME global mean/std as the data generators
        (`normalize_volts_fixed`), so the candidate and target traces are in
        one space and absolute voltage scale is preserved on both sides.
        """
        return normalize_volts_fixed(x)

    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable moving-average low-pass of a (B, T) trace.

        Window = `lowpass_ms` at the sim dt.  Smooths out the sharp AP transients
        so the MSE early in a curriculum matches the (near-convex) subthreshold
        envelope instead of the cliffy spike-timing detail.  `lowpass_ms<=0`
        returns the input unchanged.
        """
        if self.lowpass_ms <= 0:
            return x
        w = max(1, int(round(self.lowpass_ms / float(self.sim_dt_ms))))
        if w <= 1:
            return x
        k = x.new_ones(1, 1, w) / float(w)
        pad = w // 2
        y = nn.functional.conv1d(x.unsqueeze(1), k, padding=pad)
        return y.squeeze(1)[:, : x.shape[-1]]

    def _efel_feat_loss(self, v_sim_raw: torch.Tensor, v_true_raw: torch.Tensor) -> torch.Tensor:
        """Soft-eFEL feature-matching loss between two RAW-mV (B, T) traces.

        Computes the differentiable soft features on the sim side (gradient
        flows) and on the data side under no_grad (constant target), then a
        scale-normalized Huber penalty per feature, averaged.  Only the
        `self.efel_features` subset is computed (cheaper backward graph).
        """
        fs = soft_efel_features(v_sim_raw, k=self.efel_k, thr=self.efel_thr,
                                dt_ms=float(self.sim_dt_ms), nmax=self.efel_nmax,
                                only=self.efel_features)
        with torch.no_grad():
            ft = soft_efel_features(v_true_raw, k=self.efel_k, thr=self.efel_thr,
                                    dt_ms=float(self.sim_dt_ms), nmax=self.efel_nmax,
                                    only=self.efel_features)
        loss = v_sim_raw.new_zeros(())
        for f in self.efel_features:
            s = self._efel_scales[f]
            loss = loss + nn.functional.huber_loss(
                fs[f] / s, ft[f] / s, delta=self.efel_huber_delta, reduction="mean"
            )
        return loss / len(self.efel_features)

    def _voltage_loss(self, pred_unit: torch.Tensor, true_volts: torch.Tensor) -> torch.Tensor:
        """Voltage term, optionally wrapped in O1 randomized smoothing.

        With `smooth_sigma>0` the objective becomes L_sigma(theta) =
        E_eps[L(theta + sigma*eps)] estimated with `smooth_samples` Gaussian
        draws, so the gradient is a Gaussian-smoothed (hence descendable) version
        of the cliffy voltage surface rather than the raw sensitivity-weighted
        one.  `smooth_sigma=0` (default) -> a single un-perturbed call, so the
        behaviour is bit-identical to the pre-smoothing code.
        """
        if self.smooth_sigma <= 0:
            return self._voltage_loss_core(pred_unit, true_volts)
        total = pred_unit.new_zeros(())
        for _ in range(self.smooth_samples):
            eps = torch.randn_like(pred_unit) * self.smooth_sigma
            total = total + self._voltage_loss_core(pred_unit + eps, true_volts)
        return total / float(self.smooth_samples)

    def _voltage_loss_core(self, pred_unit: torch.Tensor, true_volts: torch.Tensor) -> torch.Tensor:
        """`pred_unit`  : (B, P) in unit-normalized space.
           `true_volts` : (B, T, C) with C = num_probes (after dataloader reshape)
                          or possibly (B, T, probes*stims) if multiple stims.
        Returns scalar voltage MSE (in z-scored mV space).
        """
        # Per-parameter gradient preconditioning: reweight the gradient of the
        # voltage loss w.r.t. each output parameter BEFORE any transform, so the
        # scaling lands on ∂(voltage_loss)/∂pred_unit exactly.  Identity in the
        # forward pass (loss value unchanged); only the backward signal to the
        # network is rescaled.  Applied here (not in `forward`) so the channel
        # loss keeps its untouched, well-conditioned gradient.
        if self._grad_precond_w is not None:
            pred_unit = _GradScale.apply(pred_unit, self._grad_precond_w)
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
        # C3: pad a short final mini-batch up to the canonical batch size so
        # XLA does not recompile the vmapped jaxley sim for a smaller batch
        # shape on the last batch of an epoch.  Padded rows repeat row 0 — an
        # in-range, numerically-stable sample — and are sliced off before the
        # MSE, so they contribute nothing to the loss or to its gradient.
        cur_bs = pred_phys.shape[0]
        do_pad = self.pad_batch_size is not None and 0 < cur_bs < self.pad_batch_size
        if do_pad:
            reps = self.pad_batch_size - cur_bs
            pred_phys = torch.cat([pred_phys, pred_phys[:1].expand(reps, -1)], dim=0)

        def _sim(stim_name):
            """Run the bridge for one stim and drop any padding rows."""
            v = JaxleyBridge.simulate_batch(
                pred_phys, self.cell_name, stim_name,
                checkpoint_lengths=self.checkpoint_lengths,
                solver=self.solver,
            )                                                       # (B, n_recorded, T)
            return v[:cur_bs] if do_pad else v

        # Build (sim_channel, data_channel) pairs depending on the mode.
        #   * stim_names_multi -> Exp 3: one sim per stim, soma each, vs data ch i
        #   * probe_loss_indices -> Exp 2: one sim, probe k each, vs data ch i
        #   * else              -> default soma-only (unchanged behavior)
        if self.stim_names_multi and self.probe_loss_indices is not None:  # Exp 1 combined
            # For each stim, sim the multi-probe cell and supervise every probe.
            # Channels are stim-major/probe-inner: data channel index = running count.
            pairs = []
            for sname in self.stim_names_multi:
                v_sim = _sim(sname)                                # (B, n_probes, T)
                for k in self.probe_loss_indices:
                    pairs.append((v_sim[:, k, :], true_volts[..., len(pairs)]))
        elif self.stim_names_multi:                                # Exp 3 multi-stim
            pairs = []
            for ci, sname in enumerate(self.stim_names_multi):
                v_sim = _sim(sname)
                pairs.append((v_sim[:, 0, :], true_volts[..., ci]))
        elif self.probe_loss_indices is not None:                  # Exp 2 multi-probe
            v_sim = _sim(self.stim_name)
            pairs = [(v_sim[:, k, :], true_volts[..., ci])
                     for ci, k in enumerate(self.probe_loss_indices)]
        else:                                                      # soma-only default
            v_sim = _sim(self.stim_name)
            pairs = [(v_sim[:, 0, :], true_volts[..., self.soma_probe_index])]

        # For each (sim_channel, data_channel) pair, blend the z-scored MSE
        # anchor with the soft-eFEL feature-matching loss (efel_weight>0).
        total = pred_phys.new_zeros(())
        for v_sim_ch, v_true_ch in pairs:
            # Drop the first `sim_t_skip_bins` so the window matches the data
            # H5's pre-trimmed window (0 for the self-consistent packs).
            if self.sim_t_skip_bins > 0:
                v_sim_ch = v_sim_ch[:, self.sim_t_skip_bins:]
            v_true_ch = v_true_ch.to(v_sim_ch.dtype)               # (B, T)
            T = min(v_sim_ch.shape[-1], v_true_ch.shape[-1])
            v_sim_ch  = v_sim_ch[:, :T]                            # raw mV (jaxley)
            v_true_ch = v_true_ch[:, :T]                           # fixed-z-space (pack)
            # NaN-guard: drop samples whose simulated trace went non-finite (ramp
            # stimuli crossing a spiking bifurcation can NaN the jaxley forward).
            # A single NaN would otherwise make the whole (DDP-reduced) loss NaN.
            finite = torch.isfinite(v_sim_ch).all(dim=1)          # (B,)
            if not bool(finite.all()):
                v_sim_ch  = v_sim_ch[finite]
                v_true_ch = v_true_ch[finite]
            if v_sim_ch.shape[0] == 0:                             # whole batch bad
                continue
            pair_loss = pred_phys.new_zeros(())
            if self.mse_weight > 0 or self.dtw_weight > 0:
                v_sim_z = self._normalize_volts(v_sim_ch)          # into pack z-space
            if self.mse_weight > 0:
                # L2: optionally low-pass both sides before the MSE (curriculum).
                a = self._lowpass(v_sim_z); b = self._lowpass(v_true_ch)
                pair_loss = pair_loss + self.mse_weight * self._mse(a, b)
            if self.dtw_weight > 0:
                # L1: timing-robust soft-DTW in the SAME z-space as the MSE anchor.
                pair_loss = pair_loss + self.dtw_weight * soft_dtw_loss(
                    v_sim_z, v_true_ch, gamma=self.dtw_gamma,
                    n_points=self.dtw_n_points, band_ms=self.dtw_band_ms,
                    dt_ms=float(self.sim_dt_ms), orig_T=v_sim_z.shape[-1])
            if self.efel_weight > 0:
                # de-normalize the data channel back to raw mV so both sides
                # feed the soft-eFEL layer in the physical units it assumes.
                v_true_raw = v_true_ch * VOLT_NORM_STD + VOLT_NORM_MEAN
                pair_loss = pair_loss + self.efel_weight * self._efel_feat_loss(v_sim_ch, v_true_raw)
            total = total + pair_loss
        mse = total / len(pairs)
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


def _resolve_grad_precond_weights(vl, n_par):
    """Turn a `voltage_loss.grad_precond` YAML block into a length-P weight
    vector (or None if the block is absent / disabled).

    Accepted keys (all optional):
      * weights:     [w0..w_{P-1}]  -> used verbatim (skips sensitivity math).
      * sensitivity: [s0..s_{P-1}]  -> w_p = s_p ** (-exponent).
      * exponent:    float (default 1.0). 1.0 = "inverse sensitivity" (equalize
                     gradient magnitude); 2.0 = diagonal Gauss-Newton / natural
                     gradient; 0.5 = inverse-sqrt (Fisher-diag) — all tunable.
      * normalize:   'geomean' (default) | 'mean' | 'none'. Rescales the weight
                     vector so it does not change the overall LR scale.
      * enabled:     set False to keep the block in the YAML but switch it off.
    """
    import numpy as np
    gp = vl.get("grad_precond")
    if not gp or gp.get("enabled", True) is False:
        return None

    if gp.get("weights") is not None:
        w = np.asarray(gp["weights"], dtype=np.float64)
    else:
        sens = gp.get("sensitivity")
        if sens is None:
            raise ValueError(
                "voltage_loss.grad_precond needs either 'weights' or 'sensitivity'"
            )
        sens = np.asarray(sens, dtype=np.float64)
        # Guard non-positive sensitivities (a truly dead param) against div-by-0
        # by flooring at a small fraction of the median observed sensitivity.
        pos = sens[sens > 0]
        floor = 1e-8 if pos.size == 0 else max(1e-12, 1e-3 * float(np.median(pos)))
        sens = np.clip(sens, floor, None)
        exponent = float(gp.get("exponent", 1.0))
        w = sens ** (-exponent)

    if w.shape != (n_par,):
        raise ValueError(
            f"voltage_loss.grad_precond weight vector has length {w.shape} "
            f"but the model has {n_par} output parameters"
        )

    mode = str(gp.get("normalize", "geomean")).lower()
    if mode == "geomean":
        w = w / np.exp(np.mean(np.log(w)))
    elif mode == "mean":
        w = w / np.mean(w)
    elif mode not in ("none", "off"):
        raise ValueError(f"grad_precond.normalize must be geomean|mean|none, got {mode!r}")
    return w.astype(np.float32)


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

    grad_precond_weights = _resolve_grad_precond_weights(vl, len(phys_par_range))
    if grad_precond_weights is not None:
        print(f"[HybridLoss] grad_precond weights (per-param) = "
              f"{[round(float(x), 4) for x in grad_precond_weights]}", flush=True)

    return HybridLoss(
        cell_name        = cell_name,
        phys_par_range   = phys_par_range,
        grad_precond_weights = grad_precond_weights,
        channel_weight   = float(vl.get("channel_weight", 1.0)),
        voltage_weight   = float(vl.get("voltage_weight", 0.0)),
        mask_channels    = bool(vl.get("mask_channels", False)),
        stim_name        = vl.get("stim_name"),
        soma_probe_index = int(vl.get("soma_probe_index", 0)),
        clamp_unit_tanh  = bool(vl.get("clamp_unit_tanh", False)),
        checkpoint_lengths = vl.get("checkpoint_lengths"),
        solver           = str(vl.get("solver", "bwd_euler")),
        fp64             = bool(vl.get("fp64", False)),
        sim_t_skip_ms    = float(vl.get("sim_t_skip_ms", 0.0)),
        sim_dt_ms        = float(vl.get("sim_dt_ms", 0.1)),
        # Canonical per-rank batch size for C3 last-batch padding.  train_dist
        # pops 'batch_size' into 'local_batch_size' before Trainer builds the
        # criterion; fall back to 'batch_size' for non-DDP / test call sites.
        pad_batch_size   = params.get("local_batch_size", params.get("batch_size")),
        # Multi-channel supervision (opt-in; None = soma-only). See HybridLoss.
        probe_loss_indices = vl.get("probe_loss_indices"),
        stim_names_multi   = vl.get("stim_names_multi"),
        # Soft-eFEL feature-matching voltage loss (opt-in; efel_weight=0 = off).
        efel_weight      = float(vl.get("efel_weight", 0.0)),
        mse_weight       = float(vl.get("mse_weight", 1.0)),
        efel_features    = vl.get("efel_features"),
        efel_huber_delta = float(vl.get("efel_huber_delta", 1.0)),
        efel_k           = float(vl.get("efel_k", 2.0)),
        efel_thr         = float(vl.get("efel_thr", -20.0)),
        efel_nmax        = int(vl.get("efel_nmax", 64)),
        # L1 soft-DTW / L2 low-pass / O1 randomized smoothing (all opt-in).
        dtw_weight       = float(vl.get("dtw_weight", 0.0)),
        dtw_gamma        = float(vl.get("dtw_gamma", 0.1)),
        dtw_n_points     = int(vl.get("dtw_n_points", 200)),
        dtw_band_ms      = float(vl.get("dtw_band_ms", 8.0)),
        lowpass_ms       = float(vl.get("lowpass_ms", 0.0)),
        smooth_sigma     = float(vl.get("smooth_sigma", 0.0)),
        smooth_samples   = int(vl.get("smooth_samples", 1)),
        schedule         = vl.get("schedule"),
    )
