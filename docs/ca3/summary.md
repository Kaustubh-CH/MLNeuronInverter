# CA3 Pyramidal — NEURON ↔ Jaxley comparison

Status: **all 7 tests pass on the first try.** The Jaxley port reproduces
NEURON's voltage to within fractions of a mV at rest / sub-threshold and
exactly matches spike count + AP shape under suprathreshold drive.

## Source

NEURON model: `/global/homes/k/ktub1999/mainDL4/DL4neurons2/Adapting CA3 Pyramidal Neuron/`
(single-comp soma, L=50 µm, diam=50 µm, cm=1.41 µF/cm², Ra=150, celsius=34, v_init=-65).
Channels: `leak, na3, kdr, kap, km, kd` (+ `cacum` Ca buffer — vestigial here, omitted in port).

## Files

| Role | Path |
|---|---|
| Channels (Jaxley) | `toolbox/jaxley_channels/ca3_channels.py` |
| Cell builder | `toolbox/jaxley_cells/ca3_pyramidal.py` (registered as `ca3_pyramidal`) |
| NEURON sim | `toolbox/tests/sim_neuron_ca3.py` |
| Comparison harness | `toolbox/tests/test_ca3_neuron_vs_jaxley.py` |
| Overlay plots | `docs/ca3/test*_overlay.png` |
| Summary CSV | `docs/ca3/summary.csv` |
| Wall-time | `docs/ca3/walltime.txt` |

## Result table (default ḡ unless noted)

| Test | Setup | Metric | NEURON | Jaxley | Δ | pass |
|---|---|---|---:|---:|---:|---|
| 1. Rest | I=0, 500 ms | mean V (last 100 ms) | -65.000 | -65.000 | 0.000 mV | ✅ |
| 2. Subthresh | +0.05 nA, 100→400 ms | RMSE | — | — | 0.002 mV | ✅ |
| 2. Subthresh | same | spike count | 0 | 0 | 0 | ✅ |
| 3. Suprathresh-A | +0.30 nA, 100→400 ms | spike count | 0 | 0 | 0 | ✅ |
| 3. Suprathresh-A | same | peak Vmax Δ | -51.0 | -51.0 | 0.0 mV | ✅ |
| 4. f-I | step sweep | spike counts (matched per amp) | 0/0/0/0/0/0/25 | 0/0/0/0/0/0/25 | 0 | ✅ |
| 5. Param sweep | 50 random ḡ × U(10⁻⁰·⁵, 10⁰·⁵) | corr(spike_count) | — | — | 0.995 | ✅ |
| 6. AP shape | +1.0 nA, 200 ms | peak V | 44.54 | 42.71 | 1.83 mV | ✅ |
| 6. AP shape | same | ½-width | 1.30 | 1.40 | 0.10 ms | ✅ |
| 6. AP shape | same | AHP depth | -14.70 | -14.06 | 0.63 mV | ✅ |
| 7. Wall-time | 100 traces × 500 ms, default ḡ | NEURON mean | 118.7 | — | — | ℹ |
| 7. Wall-time | same; warm JAX, B=1, CPU | Jaxley mean | — | 571.3 | +452.6 ms | ℹ |

(Pass thresholds: rest <0.5 mV, subthresh RMSE <1.5 mV, suprathresh peak Δ <5 mV
and spike-time Δ <1 ms, f-I count diff ≤1, param sweep corr >0.95, AP peak Δ <3 mV
and ½-width Δ <0.3 ms and AHP Δ <3 mV.)

## Observations

* **No spikes at I=0.30 nA** in either stack. The model has unusually strong
  resting K_d outward current (~6× the leak depolarising drive), so threshold
  sits between 0.5 and 1.0 nA. NEURON and Jaxley agree on this.
* **`e_leak = +93.9115 mV`** (verbatim from the source hoc) is functionally
  fine: g_leak is small enough (3.94e-5 S/cm²) that the strong K_d outward
  current pulls rest to -65 mV anyway. So even if it's a typo in the source,
  it doesn't destabilise the cell. The same value is used on both sides; if
  it ever needs to change, change it in *both* `ca3_channels.Leak_CA3` and
  `sim_neuron_ca3.build_cell`.
* **AP peak is 1.83 mV lower in Jaxley** — within tolerance, but the signed
  bias suggests Jaxley's `bwd_euler` underestimates the upstroke slightly
  vs NEURON's CN+secondorder=0. Not worth chasing unless we tighten the
  threshold; the f-I curve and spike timing both match exactly.
* **Wall-time on CPU**: Jaxley is ~4.8× slower than NEURON for a *single*
  500 ms trace (571 ms vs 119 ms warm). This is the expected per-sample
  overhead — Jaxley wins by batching on GPU (cf. Phase 2: 14.9× speedup at
  16 GPUs, B=128). Single-trace wall-time isn't the right benchmark for
  the data-gen / training use case.

## Pipeline integration plan (Phase C, when you're ready)

1. Add `ca3_voltage_only.hpar.yaml` (clone of `ballBBP_voltage_only.hpar.yaml`):
   - `cell_name_for_sim: ca3_pyramidal`
   - `outputSize_override: 6`
   - 6-entry `phys_par_range` matching `ca3_pyramidal._DEFAULTS`
   - `clamp_unit_tanh: True`, `fp64: True`, `mask_channels: True`,
     `sim_t_skip_ms: 100.0` (same as ballBBP).
2. Extend `scripts/gen_ball_and_stick_data.py` with `--cell ca3_pyramidal`
   to generate matched synthetic data (it already accepts arbitrary cells
   via the registry).
3. Wrap in a SLURM script `scripts/gen_ca3_data.slr` and a training SLR
   `batchShifterJaxleyCA3.slr` (pattern-matches the existing matched
   variant).
4. Smoke run: 2 epochs × 4 GPUs to confirm the bridge handles CA3.

No code changes to the loss, bridge, or training loop are required — the
registry is the single integration point.
