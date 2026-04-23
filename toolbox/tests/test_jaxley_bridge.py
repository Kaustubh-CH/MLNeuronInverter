"""Phase 1 tests for toolbox.JaxleyBridge.

Run (inside shifter with jaxley installed):

    python -m toolbox.tests.test_jaxley_bridge
    # or a single test:
    python -m toolbox.tests.test_jaxley_bridge shapes

Exits non-zero on first failure so it's easy to wire into a SLURM job.
"""

import os
# Keep JAX off the GPU for the gradcheck — bwd_euler at fp64 gradchecks
# deterministically on CPU, and we don't want to contend with torch for VRAM.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import time
import traceback

import numpy as np
import torch

# fp64 everywhere so gradcheck is meaningful.
import jax
jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

from toolbox import JaxleyBridge, jaxley_cells  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# tests
# ─────────────────────────────────────────────────────────────────────────

def test_registry_lists_both_cells():
    cells = set(jaxley_cells.list_cells())
    assert "ball_and_stick" in cells, f"ball_and_stick missing: {cells}"
    assert "L5TTPC" in cells, f"L5TTPC missing: {cells}"
    print("  registered cells:", sorted(cells))


def test_shapes():
    spec = jaxley_cells.get("ball_and_stick")
    P = len(spec.param_keys)
    B = 3

    # Use actual defaults so the HH gating is reasonable.
    from toolbox.jaxley_cells.ball_and_stick import _DEFAULTS, PARAM_KEYS
    defaults = torch.tensor(
        [[_DEFAULTS[k] for k in PARAM_KEYS]] * B, dtype=torch.float64
    )
    v = JaxleyBridge.simulate_batch(defaults, "ball_and_stick")
    assert v.ndim == 3, f"expected 3D (B, n_rec, T_out), got {tuple(v.shape)}"
    assert v.shape[0] == B, f"batch dim: {v.shape}"
    assert v.shape[1] >= 1, f"no recorded compartments: {v.shape}"
    assert v.shape[2] > 100, f"time axis too short: {v.shape}"
    assert torch.isfinite(v).all(), "non-finite voltages"
    print(f"  shapes OK: params {tuple(defaults.shape)} -> volts {tuple(v.shape)}")


def test_cache_hit_no_recompile():
    """Second forward must be notably faster than the first (no recompile)."""
    from toolbox.jaxley_cells.ball_and_stick import _DEFAULTS, PARAM_KEYS
    p = torch.tensor([[_DEFAULTS[k] for k in PARAM_KEYS]], dtype=torch.float64)

    JaxleyBridge.clear_cache()
    t0 = time.time(); _ = JaxleyBridge.simulate_batch(p, "ball_and_stick"); t1 = time.time()
    t2 = time.time(); _ = JaxleyBridge.simulate_batch(p, "ball_and_stick"); t3 = time.time()
    cold = t1 - t0
    warm = t3 - t2
    print(f"  forward: cold={cold:.3f}s  warm={warm:.3f}s  ratio={cold/max(warm,1e-6):.1f}x")
    # Cold is dominated by tracing + compilation; warm should be >> faster.
    # Threshold loose — we want to flag "every call recompiles".
    assert cold > 2.0 * warm, (
        f"suspected recompile on second call (cold={cold:.2f}s warm={warm:.2f}s)"
    )


def test_vmap_matches_serial_loop():
    """Batched vmap must agree with a per-sample Python loop."""
    from toolbox.jaxley_cells.ball_and_stick import _DEFAULTS, PARAM_KEYS
    defaults = torch.tensor([_DEFAULTS[k] for k in PARAM_KEYS], dtype=torch.float64)

    # 3 random scales of the defaults — stays in a physiological range.
    rng = np.random.default_rng(0)
    scales = torch.tensor(rng.uniform(0.7, 1.3, size=(3,)), dtype=torch.float64)
    batch = defaults[None] * scales[:, None]

    v_batch = JaxleyBridge.simulate_batch(batch, "ball_and_stick")
    v_one_at_a_time = torch.stack([
        JaxleyBridge.simulate_batch(batch[i:i+1], "ball_and_stick")[0]
        for i in range(batch.shape[0])
    ], dim=0)
    diff = (v_batch - v_one_at_a_time).abs().max().item()
    print(f"  vmap vs loop max |diff| = {diff:.3e}")
    assert diff < 1e-6, f"vmap-vs-loop mismatch: {diff}"


def test_gradcheck_tiny():
    """Finite-difference gradcheck on 2 params to catch wiring bugs.

    We deliberately shrink t_max to keep the check cheap; a full check at
    t_max=500ms is too slow for this loop.  We do this by monkey-patching
    the spec before the cell is cached.
    """
    # Force a fresh handle with a tiny t_max.
    JaxleyBridge.clear_cache()
    import toolbox.jaxley_cells.ball_and_stick as bas
    old_tmax = bas._T_MAX
    try:
        bas._T_MAX = 5.0  # 5 ms, ~200 steps
        from toolbox.jaxley_cells.ball_and_stick import _DEFAULTS, PARAM_KEYS
        # Gradcheck across all 4 params; pick mild perturbation.
        p0 = torch.tensor([[_DEFAULTS[k] for k in PARAM_KEYS]],
                          dtype=torch.float64, requires_grad=True)

        def f(p):
            v = JaxleyBridge.simulate_batch(p, "ball_and_stick")
            # gradcheck wants a scalar-ish function — reduce to mean.
            return v.mean()

        # Analytical grad via autograd
        out = f(p0)
        (g_ana,) = torch.autograd.grad(out, p0, retain_graph=False)

        # Finite-difference grad — central differences, 1% eps of value.
        g_fd = torch.zeros_like(p0)
        for i in range(p0.shape[1]):
            eps = 1e-3 * abs(p0[0, i].item())
            pp = p0.detach().clone(); pp[0, i] += eps
            pm = p0.detach().clone(); pm[0, i] -= eps
            g_fd[0, i] = (f(pp) - f(pm)) / (2 * eps)

        rel = (g_ana - g_fd).abs() / (g_fd.abs().clamp_min(1e-10))
        print("  analytical grad:", g_ana.detach().cpu().numpy().ravel())
        print("  finite-diff grad:", g_fd.detach().cpu().numpy().ravel())
        print("  relative error :", rel.detach().cpu().numpy().ravel())
        # 5% tolerance — bwd_euler is not symplectic; we're mostly catching
        # wiring errors (wrong sign, missing broadcast, etc.).
        assert (rel < 5e-2).all(), f"gradcheck failed, rel err = {rel}"
    finally:
        bas._T_MAX = old_tmax
        JaxleyBridge.clear_cache()


def test_l5ttpc_registers_but_do_not_build():
    """Just confirm the spec is available; building the 19-param L5 cell
    + jit is Phase 5's problem."""
    spec = jaxley_cells.get("L5TTPC")
    assert len(spec.param_keys) == 19, f"expected 19 params, got {len(spec.param_keys)}"
    # No build() call here — Phase 5.


# ─────────────────────────────────────────────────────────────────────────
# runner
# ─────────────────────────────────────────────────────────────────────────

TESTS = [
    ("registry",      test_registry_lists_both_cells),
    ("shapes",        test_shapes),
    ("cache",         test_cache_hit_no_recompile),
    ("vmap_vs_loop",  test_vmap_matches_serial_loop),
    ("gradcheck",     test_gradcheck_tiny),
    ("l5_registers",  test_l5ttpc_registers_but_do_not_build),
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
        try:
            fn()
            print(f"[{name}] OK")
        except Exception:
            traceback.print_exc()
            print(f"[{name}] FAIL")
            failed.append(name)
    print()
    print(f"summary: {len(chosen) - len(failed)}/{len(chosen)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
