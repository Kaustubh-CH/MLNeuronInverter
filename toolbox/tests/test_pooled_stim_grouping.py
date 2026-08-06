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


if __name__ == "__main__":
    test_grouping()
    test_single_stim_matches_reference()
    test_weighting_is_not_per_group()
    print("\nall pooled-grouping tests passed")
