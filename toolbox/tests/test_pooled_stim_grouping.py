#!/usr/bin/env python3
"""Pooled multi-stim (variant B) grouping logic, with the jaxley solve stubbed out.

Checks the part that can silently corrupt a run without ever raising: that each
sample is simulated under ITS OWN stimulus, that the pooled loss is the per-sample
mean (not the per-group mean, which would up-weight whichever stim happened to be
rare in the batch), that padding is quantized, and that no sample is dropped or
double-counted.

Run:  python -m toolbox.tests.test_pooled_stim_grouping
"""
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from toolbox.HybridLoss import HybridLoss   # noqa: E402


STIMS = ["5kChaoticRamp", "5k0chaotic4", "BBP_Exp_Step1000_i4k", "chirp23a_i4k"]


def _make_loss(local_batch=128, quantum=8):
    """HybridLoss with every jaxley-touching knob off; only grouping is exercised."""
    return HybridLoss(
        cell_name="ca3_pyramidal",
        phys_par_range=[[1e-4, 0.5, "S/cm^2"]] * 6,
        channel_weight=0.0, voltage_weight=1.0, mask_channels=True,
        pad_batch_size=local_batch,
        pooled_stim_names=STIMS,
        pooled_pad_quantum=quantum,
    )


def test_grouping():
    torch.manual_seed(0)
    loss = _make_loss()
    B, P, T = 128, 6, 32
    pred = torch.randn(B, P, requires_grad=True)
    volts = torch.randn(B, T, 1)
    stim_idx = torch.randint(0, len(STIMS), (B,))

    seen = {}          # stim name -> list of per-sample voltage "signatures"
    calls = []

    def fake_core(pred_unit, true_volts, s_idx=None, stim_name_override=None, pad_to=None):
        assert s_idx is None, "pooled path must not recurse"
        assert stim_name_override is not None, "pooled path must name a stim"
        n = pred_unit.shape[0]
        calls.append((stim_name_override, n, pad_to))
        seen.setdefault(stim_name_override, []).extend(
            true_volts[:, 0, 0].tolist())
        # Return something that depends on the group so weighting is observable.
        return true_volts.mean()

    loss._voltage_loss_core = fake_core
    out = loss._pooled_voltage_loss(pred, volts, stim_idx)

    # 1. every sample went to the stim it was labelled with, exactly once
    total_seen = sum(len(v) for v in seen.values())
    assert total_seen == B, f"saw {total_seen} samples, expected {B}"
    for si, sname in enumerate(STIMS):
        want = sorted(volts[stim_idx == si][:, 0, 0].tolist())
        got = sorted(seen.get(sname, []))
        assert want == got, f"{sname}: wrong samples routed to this stimulus"

    # 2. one call per present stim, padded up to a multiple of the quantum
    assert len(calls) == len(torch.unique(stim_idx)), f"unexpected call count {len(calls)}"
    for sname, n, pad_to in calls:
        assert pad_to % 8 == 0 or pad_to == 128, f"{sname}: pad_to={pad_to} not quantized"
        assert pad_to >= n, f"{sname}: pad_to={pad_to} < n={n}"
        assert pad_to - n < 8, f"{sname}: padded {pad_to-n} rows, more than one quantum"

    # 3. result is the PER-SAMPLE mean, not the per-group mean
    want = volts.mean()
    assert torch.allclose(out, want, atol=1e-6), f"pooled loss {out} != per-sample mean {want}"

    # 4. gradient still flows to the network output
    out2 = loss._pooled_voltage_loss(pred, volts * pred[:, :1].unsqueeze(-1), stim_idx)
    out2.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all(), "no finite grad"
    print(f"OK grouping: {len(calls)} groups, "
          f"{[(s, n, p) for s, n, p in calls]}")


def test_single_stim_matches_reference():
    """All-one-stim pooled batch == the plain single-stim path on the same data."""
    loss = _make_loss()
    B, T = 64, 16
    pred = torch.randn(B, 6)
    volts = torch.randn(B, T, 1)
    stim_idx = torch.full((B,), 2, dtype=torch.long)

    got = {}

    def fake_core(pred_unit, true_volts, s_idx=None, stim_name_override=None, pad_to=None):
        got["stim"] = stim_name_override
        got["n"] = pred_unit.shape[0]
        return true_volts.mean()

    loss._voltage_loss_core = fake_core
    out = loss._pooled_voltage_loss(pred, volts, stim_idx)
    assert got["stim"] == STIMS[2], f"wrong stim {got['stim']}"
    assert got["n"] == B, f"lost samples: {got['n']} != {B}"
    assert torch.allclose(out, volts.mean(), atol=1e-6)
    print("OK single-stim pooled batch reduces to the plain path")


def test_weighting_is_not_per_group():
    """A lopsided batch must not let a 1-sample stim outvote a 63-sample one."""
    loss = _make_loss()
    B, T = 64, 16
    pred = torch.randn(B, 6)
    volts = torch.zeros(B, T, 1)
    stim_idx = torch.zeros(B, dtype=torch.long)
    stim_idx[0] = 1
    volts[0] = 100.0                       # the lone outlier sample

    loss._voltage_loss_core = lambda p, v, s=None, stim_name_override=None, pad_to=None: v.mean()
    out = loss._pooled_voltage_loss(pred, volts, stim_idx)

    per_sample = volts.mean()              # 100*16 / (64*16) = 1.5625
    per_group = (100.0 + 0.0) / 2          # what naive group-averaging would give
    assert torch.allclose(out, per_sample, atol=1e-6), \
        f"got {out.item()}, per-sample {per_sample.item()}, per-group {per_group}"
    print(f"OK weighting: {out.item():.4f} == per-sample {per_sample.item():.4f} "
          f"(per-group would be {per_group})")


def _make_loss_norm(local_batch=64, quantum=8):
    """Like _make_loss but with the opt-in per-stim EMA normalization ON."""
    return HybridLoss(
        cell_name="ca3_pyramidal",
        phys_par_range=[[1e-4, 0.5, "S/cm^2"]] * 6,
        channel_weight=0.0, voltage_weight=1.0, mask_channels=True,
        pad_batch_size=local_batch,
        pooled_stim_names=STIMS,
        pooled_pad_quantum=quantum,
        pooled_stim_norm="ema",
    )


def _two_group_setup(B=64, T=16):
    """Half the batch on stim 0 (raw loss 10x), half on stim 1 (raw loss 1x)."""
    stim_idx = torch.cat([torch.zeros(B // 2), torch.ones(B // 2)]).long()
    volts = torch.zeros(B, T, 1)
    scales = {STIMS[0]: 10.0, STIMS[1]: 1.0}

    def fake_core(p, v, s=None, stim_name_override=None, pad_to=None):
        # Graph-carrying loss whose value AND gradient scale with the stim's factor.
        return scales[stim_name_override] * p.abs().mean()

    return stim_idx, volts, fake_core


def test_stim_norm_off_is_identical():
    """pooled_stim_norm absent/None -> bitwise-identical to the pre-norm behavior."""
    stim_idx, volts, fake_core = _two_group_setup()
    pred = torch.ones(len(stim_idx), 6)
    outs = []
    for mk in (_make_loss, lambda: _make_loss(local_batch=64)):
        loss = mk()
        assert loss.pooled_stim_norm is None
        loss._voltage_loss_core = fake_core
        outs.append(loss._pooled_voltage_loss(pred, volts, stim_idx))
    # raw weighted mean: 0.5*10*1 + 0.5*1*1 = 5.5, and both ctors agree exactly
    assert outs[0].item() == outs[1].item() == 5.5, [o.item() for o in outs]
    print("OK norm off: default ctor unchanged, raw weighted mean 5.5")


def test_stim_norm_ema_equalizes():
    """With norm on, groups whose raw losses differ 10x contribute equally."""
    stim_idx, volts, fake_core = _two_group_setup()
    loss = _make_loss_norm()
    loss._voltage_loss_core = fake_core
    pred = torch.ones(len(stim_idx), 6, requires_grad=True)

    out = loss._pooled_voltage_loss(pred, volts, stim_idx)
    # EMA initializes at the group's own loss -> the FIRST step is exactly scale-1:
    # each normalized group loss = 1.0, total = 0.5*1 + 0.5*1 = 1.0 (raw is 5.5).
    assert abs(out.item() - 1.0) < 1e-6, out.item()
    for _ in range(5):   # constant losses: EMA stays put, total stays 1.0
        out = loss._pooled_voltage_loss(pred, volts, stim_idx)
    assert abs(out.item() - 1.0) < 1e-6, out.item()

    out.backward()
    g = pred.grad
    assert g is not None and torch.isfinite(g).all(), "no finite grad through norm"
    g0 = g[stim_idx == 0].abs().mean().item()
    g1 = g[stim_idx == 1].abs().mean().item()
    # un-normalized the ratio would be 10; normalized the groups match
    assert abs(g0 / g1 - 1.0) < 1e-6, (g0, g1)
    print(f"OK norm on: total {out.item():.4f} (raw 5.5), grad ratio {g0/g1:.4f} (raw 10)")


def test_stim_norm_gated_off_in_validation():
    """Under torch.no_grad() the norm must NOT apply and the EMA must NOT advance,
    so the plateau scheduler tracks the raw, stationary per-sample-mean metric."""
    stim_idx, volts, fake_core = _two_group_setup()
    loss = _make_loss_norm()
    loss._voltage_loss_core = fake_core
    pred = torch.ones(len(stim_idx), 6)
    with torch.no_grad():
        out = loss._pooled_voltage_loss(pred, volts, stim_idx)
    assert abs(out.item() - 5.5) < 1e-6, out.item()      # raw weighted mean
    assert loss._pooled_ema == {}, loss._pooled_ema      # state untouched
    print("OK norm gate: no_grad -> raw 5.5, EMA state empty")


if __name__ == "__main__":
    test_grouping()
    test_single_stim_matches_reference()
    test_weighting_is_not_per_group()
    test_stim_norm_off_is_identical()
    test_stim_norm_ema_equalizes()
    test_stim_norm_gated_off_in_validation()
    print("\nall pooled-grouping tests passed")
