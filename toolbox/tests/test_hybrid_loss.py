"""Phase 2 tests for toolbox.HybridLoss.

Run inside the neuroninverter_jaxley conda env:

    python -m toolbox.tests.test_hybrid_loss
    # or a single test:
    python -m toolbox.tests.test_hybrid_loss zero_recovers_mse

Forces JAX to CPU and fp64 so backward gradients are deterministic.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import time
import traceback

import numpy as np
import torch

import jax
jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

from toolbox import JaxleyBridge                                         # noqa: E402
from toolbox.HybridLoss import (                                          # noqa: E402
    HybridLoss, _ChannelOnlyAdapter, build_hybrid_loss,
)
from toolbox.jaxley_cells.ball_and_stick import _DEFAULTS, PARAM_KEYS     # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────────

# A synthetic phys_par_range matching ball_and_stick PARAM_KEYS:
#   center, log10_halfspan, unit
# unit=0 -> phys=center; unit=1 -> phys=center*10**0.5 (~3.16x)
_PHYS_PAR_RANGE = [[_DEFAULTS[k], 0.5, "S/cm^2"] for k in PARAM_KEYS]


def _shrink_t_max(value=5.0):
    """Force a fresh handle with a tiny t_max so backward is fast."""
    JaxleyBridge.clear_cache()
    import toolbox.jaxley_cells.ball_and_stick as bas
    bas._T_MAX = value


def _restore_t_max(old):
    import toolbox.jaxley_cells.ball_and_stick as bas
    bas._T_MAX = old
    JaxleyBridge.clear_cache()


# ─────────────────────────────────────────────────────────────────────────
# tests
# ─────────────────────────────────────────────────────────────────────────

def test_zero_recovers_mse():
    """voltage_weight=0 must give bitwise-identical loss vs plain MSE."""
    loss = HybridLoss(
        cell_name      = "ball_and_stick",
        phys_par_range = _PHYS_PAR_RANGE,
        channel_weight = 1.0,
        voltage_weight = 0.0,
    )
    rng = np.random.default_rng(0)
    pred  = torch.from_numpy(rng.standard_normal((4, len(PARAM_KEYS))))
    true  = torch.from_numpy(rng.standard_normal((4, len(PARAM_KEYS))))
    volts = torch.from_numpy(rng.standard_normal((4, 200, 3)))
    out_h = loss(pred, true, volts)
    out_m = torch.nn.MSELoss()(pred, true)
    diff = (out_h - out_m).abs().item()
    print(f"  hybrid(w_v=0) vs MSE: |diff| = {diff:.3e}")
    assert diff < 1e-12, f"hybrid(w_v=0) != MSE by {diff}"


def test_adapter_is_mse():
    """_ChannelOnlyAdapter must equal plain MSELoss."""
    adapter = _ChannelOnlyAdapter()
    rng = np.random.default_rng(1)
    pred  = torch.from_numpy(rng.standard_normal((4, len(PARAM_KEYS))))
    true  = torch.from_numpy(rng.standard_normal((4, len(PARAM_KEYS))))
    volts = torch.from_numpy(rng.standard_normal((4, 200, 3)))
    out_a = adapter(pred, true, volts)
    out_m = torch.nn.MSELoss()(pred, true)
    diff = (out_a - out_m).abs().item()
    print(f"  adapter vs MSE: |diff| = {diff:.3e}")
    assert diff < 1e-12


def test_voltage_forward_finite():
    """voltage_weight>0 — forward returns scalar finite value."""
    old = None
    try:
        import toolbox.jaxley_cells.ball_and_stick as bas
        old = bas._T_MAX
        _shrink_t_max(5.0)

        loss = HybridLoss(
            cell_name      = "ball_and_stick",
            phys_par_range = _PHYS_PAR_RANGE,
            channel_weight = 1.0,
            voltage_weight = 0.1,
        )
        B = 2
        P = len(PARAM_KEYS)
        pred = torch.zeros(B, P)                            # unit=0 -> defaults
        true = torch.zeros(B, P)
        volts = torch.randn(B, 200, 3) * 0.1
        L = loss(pred, true, volts)
        print(f"  forward: loss = {L.item():.6e}, finite={torch.isfinite(L).item()}")
        assert L.dim() == 0
        assert torch.isfinite(L).item()
        # voltage term should make the loss strictly nonzero (channel term is 0
        # since pred==true), so we're really exercising the voltage path.
        assert L.item() > 0, "voltage loss should be positive when v_sim != v_true"
    finally:
        if old is not None:
            _restore_t_max(old)


def test_voltage_grad_flows():
    """voltage-only forward must produce finite, nonzero grad w.r.t. params."""
    old = None
    try:
        import toolbox.jaxley_cells.ball_and_stick as bas
        old = bas._T_MAX
        _shrink_t_max(5.0)

        loss = HybridLoss(
            cell_name      = "ball_and_stick",
            phys_par_range = _PHYS_PAR_RANGE,
            channel_weight = 0.0,    # voltage-only
            voltage_weight = 1.0,
        )
        B = 2
        P = len(PARAM_KEYS)
        pred = torch.zeros(B, P, requires_grad=True)
        true = torch.zeros(B, P)
        volts = torch.randn(B, 200, 3) * 0.1
        L = loss(pred, true, volts)
        L.backward()
        g = pred.grad
        gn = g.norm().item()
        print(
            f"  voltage-only grad: shape={tuple(g.shape)} "
            f"finite={torch.isfinite(g).all().item()} norm={gn:.3e}"
        )
        assert torch.isfinite(g).all(), "non-finite grad from voltage loss"
        assert gn > 0, "voltage loss should produce nonzero grad"
    finally:
        if old is not None:
            _restore_t_max(old)


def test_mask_channels_skips_channel_loss():
    """mask_channels=True + voltage_weight=0 -> loss == 0 regardless of pred/true."""
    loss = HybridLoss(
        cell_name      = "ball_and_stick",
        phys_par_range = _PHYS_PAR_RANGE,
        channel_weight = 1.0,
        voltage_weight = 0.0,
        mask_channels  = True,
    )
    rng = np.random.default_rng(2)
    pred  = torch.from_numpy(rng.standard_normal((2, len(PARAM_KEYS))))
    volts = torch.from_numpy(rng.standard_normal((2, 200, 3)))
    L = loss(pred, None, volts)
    print(f"  mask_channels(w_v=0): loss = {L.item():.3e}")
    assert L.item() == 0.0


def test_unit_to_phys_matches_numpy():
    """Torch unit_to_phys path must agree with the numpy reference."""
    loss = HybridLoss(
        cell_name      = "ball_and_stick",
        phys_par_range = _PHYS_PAR_RANGE,
        channel_weight = 1.0,
        voltage_weight = 0.0,
    )
    from toolbox.jaxley_utils import phys_par_range_to_arrays, unit_to_phys_np
    centers, logspans = phys_par_range_to_arrays(_PHYS_PAR_RANGE)
    rng = np.random.default_rng(3)
    unit_np = rng.uniform(-1, 1, size=(3, len(PARAM_KEYS))).astype(np.float64)
    phys_np = unit_to_phys_np(unit_np, centers.astype(np.float64), logspans.astype(np.float64))
    phys_t  = loss._unit_to_phys(torch.from_numpy(unit_np)).numpy()
    diff = np.abs(phys_np - phys_t).max()
    print(f"  unit_to_phys (torch vs numpy): max|diff| = {diff:.3e}")
    assert diff < 1e-10


def test_factory_channel_only_passthrough():
    """build_hybrid_loss(use_voltage_loss=False) -> _ChannelOnlyAdapter."""
    crit = build_hybrid_loss({"use_voltage_loss": False})
    assert isinstance(crit, _ChannelOnlyAdapter), type(crit)
    rng = np.random.default_rng(4)
    pred  = torch.from_numpy(rng.standard_normal((2, 4)))
    true  = torch.from_numpy(rng.standard_normal((2, 4)))
    out_a = crit(pred, true, None)
    out_m = torch.nn.MSELoss()(pred, true)
    diff = (out_a - out_m).abs().item()
    print(f"  factory(channel-only) vs MSE: |diff| = {diff:.3e}")
    assert diff < 1e-12


def test_factory_builds_hybrid_with_explicit_range():
    """build_hybrid_loss reads phys_par_range from voltage_loss when given."""
    crit = build_hybrid_loss({
        "use_voltage_loss": True,
        "voltage_loss": {
            "cell_name_for_sim": "ball_and_stick",
            "channel_weight":    1.0,
            "voltage_weight":    0.0,
            "phys_par_range":    _PHYS_PAR_RANGE,
        },
    })
    assert isinstance(crit, HybridLoss), type(crit)
    print(f"  factory built HybridLoss for ball_and_stick, "
          f"P={crit._centers.shape[0]}")
    assert crit._centers.shape[0] == len(PARAM_KEYS)


# ─────────────────────────────────────────────────────────────────────────
# runner
# ─────────────────────────────────────────────────────────────────────────

TESTS = [
    ("zero_recovers_mse",        test_zero_recovers_mse),
    ("adapter_is_mse",           test_adapter_is_mse),
    ("unit_to_phys_matches",     test_unit_to_phys_matches_numpy),
    ("mask_channels",            test_mask_channels_skips_channel_loss),
    ("factory_channel_only",     test_factory_channel_only_passthrough),
    ("factory_hybrid",           test_factory_builds_hybrid_with_explicit_range),
    ("voltage_forward_finite",   test_voltage_forward_finite),
    ("voltage_grad_flows",       test_voltage_grad_flows),
]


def main(argv):
    if len(argv) > 1:
        names = set(argv[1:])
        chosen = [(n, f) for (n, f) in TESTS if n in names]
        if len(chosen) != len(names):
            unknown = names - {n for (n, _) in chosen}
            print("unknown test names:", sorted(unknown), file=sys.stderr)
            return 2
    else:
        chosen = TESTS

    failed = []
    for name, fn in chosen:
        print(f"[{name}]")
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            print(f"[{name}] OK ({dt:.2f}s)")
        except Exception:
            traceback.print_exc()
            print(f"[{name}] FAIL")
            failed.append(name)
    print()
    print(f"summary: {len(chosen) - len(failed)}/{len(chosen)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
