# PLAN — Differentiable Physics Loss via Jaxley

Implementation plan for wiring a differentiable jaxley simulator into the existing CNN training loop so the model can be supervised on **voltage** (physics) in addition to — or instead of — **channel parameters**.

## Ground rules (per user)

- **Don't restructure existing code.** `toolbox/Model*.py`, `toolbox/Trainer.py`, `toolbox/Dataloader_H5.py` keep their shapes. New functionality goes in new files; existing files only get additive hooks (e.g. reading one new config key, swapping a loss object). No renames, no moved methods.
- **Efficient jaxley.** Build the cell topology once at startup, `jax.jit` + `jax.vmap` the forward pass over a batch of channel parameter vectors. Never rebuild the compartmental tree per step.
- **Runs on NERSC Perlmutter.** Every test runs inside a `shifter` container on an interactive GPU allocation.
- **Commit + push after each phase.** One commit per phase (or small sub-phases where a phase is long). See "open questions" re: branch strategy.

## Decisions (answered by user 2026-04-23)

1. **Push target:** `origin/CNN_Jaxley` (current branch). `master` untouched.
2. **Jaxley env:** install on top of `nersc/pytorch:ngc-21.08-v2` via `pip install --user`. Pin the deps in `requirements-jaxley.txt` so the same env can be reconstructed elsewhere.
3. **Ball-and-stick:** port from `/pscratch/sd/k/ktub1999/Neuron_Jaxley/sim_jaxley.py` (soma HH + dendrite Leak, 5-comp dendrite).
4. **L5TTPC:** port from `/pscratch/sd/k/ktub1999/Neuron_Jaxley/sim_jaxley_L5TTPC1.py` + `bbp_channels_jaxley.py`. SWC comes from `/pscratch/sd/k/ktub1999/Neuron_Jaxley/results_detailed_morphology/L5_TTPC1.swc`. Parameter mapping is the `CSV_PARAM_MAP` dict in that file.
5. **Stim:** `5k50kInterChaoticB`. Waveform file: `/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/5k50kInterChaoticB.csv` (also stored inside each `.mlPack1.h5`).
6. **"Push":** `git push origin CNN_Jaxley` after each phase.

---

## Phase 0 — Repository onboarding & architecture mapping

**Goal:** produce a machine-refreshable map of the code so future phases (and future Claude instances) don't re-derive it each time.

**Deliverables**
- `structure.md` — sections:
  - **Data flow:** `NEURON sim → packBBP3/aggregate_*.py → *.simRaw.h5 → format_bbp3_for_ML*.py → *.mlPack1.h5 → toolbox/Dataloader_H5.py → Trainer → sum_train.yaml → predict*.py → unitParamConvert*.py`. Include shapes at each stage (e.g. `volts: (N, 4000, 3, 5) fp16`).
  - **Module hierarchy:** which file owns the model (`toolbox/Model.py`, `Model_Multi.py`, `Transformer_Model.py`), the loss (inline `torch.nn.MSELoss` in `Trainer.py`), the optimizer + scheduler, the dataloader.
  - **Config surface:** every knob in a `<design>.hpar.yaml`, annotated.
  - **Jaxley integration points (placeholder):** where Phase 1–5 will attach — one paragraph per phase naming the exact file.
- `toolbox/refresh_structure.py` — reads the repo, regenerates the Data-flow + Module-hierarchy sections. Deterministic (no LLM call).
- `.git/hooks/post-commit` (or a `scripts/install_hooks.sh` that installs it) — runs `refresh_structure.py` when tracked `*.py` files change. Chosen as post-commit rather than pre-commit so it never blocks a commit.

**Tests (interactive, no GPU needed)**
```bash
salloc -C cpu -q interactive -t30:00 -A m2043 -N 1
module load pytorch
python toolbox/refresh_structure.py --check   # exits non-zero if structure.md is stale
python toolbox/refresh_structure.py           # regenerates
diff structure.md structure.md.bak            # stable output
```

**Commit:** `phase 0: structure.md + refresh hook`

---

## Phase 1 — Jaxley simulation wrapper

**Goal:** a function `simulate_batch(channel_params_batch, stim_batch) → voltage_batch` that is differentiable w.r.t. `channel_params_batch` and efficient.

**Deliverables**
- `toolbox/jaxley_cells/__init__.py` — registry that maps `cell_name → build_cell()` callable.
- `toolbox/jaxley_cells/ball_and_stick.py` — jaxley ball-and-stick builder with a small fixed parameter list (pending Q3).
- `toolbox/jaxley_cells/l5ttpc.py` — stub for Phase 5 (loads morphology + channels, parameter names aligned to `sum_train.yaml`'s `parName`).
- `toolbox/JaxleyBridge.py`:
  - Holds a module-level `_CELL_CACHE` so the jaxley cell is built + jitted once per process.
  - `class JaxleySimulate(torch.autograd.Function)` — forward does `torch → (dlpack) → jax`, calls jitted `vmap(simulate)`, returns `torch.Tensor`. Backward uses a jitted `vjp` closure captured at first forward.
  - `simulate_batch(params: Tensor[B, P], stim: Tensor[B, T] or Tensor[T], cell_name: str) → Tensor[B, T, num_probes]`.
- `toolbox/jaxley_utils.py` — unit-param → physical-param conversion that mirrors `toolbox/unitParamConvert.py` but runs inside JAX (so gradients flow).

**Tests**
```bash
# GPU node, shifter image with JAX (pending Q2)
salloc -C gpu -q interactive -t1:00:00 --gpus-per-task=1 \
       --image=<chosen-image> -A m2043_g --ntasks-per-node=1 -N 1
srun -n1 shifter python -m toolbox.tests.test_jaxley_bridge
```
Test script asserts:
- Forward output shape matches expectation.
- Two calls on the same process don't recompile (check with `jax.jit`'s cache hit, or timing).
- `torch.autograd.gradcheck` passes at fp64 on a 2-compartment cell (use `torch.set_default_dtype(torch.float64)` for the check, 2 samples).
- `vmap` over batch of 8 param sets scales sub-linearly compared to a Python loop.

**Commit:** `phase 1: jaxley bridge + ball-and-stick cell + gradcheck test`

---

## Phase 2 — Hybrid loss

**Goal:** drop-in replacement for the inline MSE in `Trainer.py` that can compute channel-MSE, voltage-MSE, or a weighted sum, toggled via the hpar yaml.

**Deliverables**
- `toolbox/HybridLoss.py`:
  ```python
  class HybridLoss(nn.Module):
      def __init__(self, cell_name, channel_weight, voltage_weight,
                   mask_channels=False): ...
      def forward(self, pred_params, true_params, stim, true_volts) -> (loss, parts_dict)
  ```
  When `mask_channels=True` (experimental mode), channel component is skipped and `true_params` may be `None`.
- **Additive** hook in `toolbox/Trainer.py` (one-line change): after constructing the default `MSELoss`, if `params.get('use_voltage_loss')` is truthy, replace `self.criterion` with `HybridLoss(...)`. The existing loss call site does not move.
- Config additions in a new design yaml `m8lay_vs3_jaxley.hpar.yaml`:
  ```yaml
  use_voltage_loss: true
  voltage_loss:
      cell_name_for_sim: ball_and_stick   # or L5_TTPC1cADpyr0 in Phase 5
      channel_weight: 1.0
      voltage_weight: 0.1                 # annealable
      mask_channels: false
  ```
- `toolbox/tests/test_hybrid_loss.py` — synthetic forward + backward numerical check; asserts that setting `voltage_weight=0` recovers bitwise-identical loss vs the current MSE path.

**Tests**
```bash
srun -n1 shifter python -m toolbox.tests.test_hybrid_loss
# End-to-end smoke: 5 steps of training, ball-and-stick, batch=8
srun -n1 shifter python -u train_dist.py --design m8lay_vs3_jaxley \
     --cellName L5_TTPC1cADpyr0 --numGlobSamp 40 --epochs 1 \
     --data_path_temp /pscratch/sd/k/ktub1999/Apr26ExcNoNoise_24733338/
```
Assert: training loss decreases; GPU memory stable; each step < 2× the channel-only step time on batch=8 ball-and-stick.

**Commit:** `phase 2: hybrid channel + voltage loss, config toggle, regression test`

---

## Phase 3 — Ball-and-stick validation

**Goal:** show the voltage-loss path actually trains on a simple cell and produces correct channel predictions + voltage overlays.

**Deliverables**
- `scripts/gen_ball_and_stick_data.py` — packs a `.mlPack1.h5` with the same layout the existing dataloader expects, populated by jaxley ground-truth simulations over a parameter sweep. (Reuses `packBBP3/format_bbp3_for_ML.py` normalization so downstream code is unchanged.)
- `plotJaxleyValidation.py` — given a run_dir + a test h5:
  - bar chart of predicted vs ground-truth channels per parameter (with error bars),
  - overlaid voltage traces for N random test samples,
  - text summary: per-parameter RMSE + explained variance, voltage RMSE + explained variance.
- SLURM wrapper `batchShifterJaxley.slr` — analog of `batchShifter.slr` but runs the jaxley-enabled design end-to-end.

**Tests**
- Generate 16 k ball-and-stick samples (≤10 min).
- Train 20 epochs on 1 GPU.
- Assert per-parameter explained variance > 0.85 on validation.
- Assert voltage RMSE (normalized units) < 0.3 on validation.
- Commit the plots as PNGs under `docs/phase3/` (not in `.gitignore`-excluded paths) so they're visible in the commit.

**Commit:** `phase 3: ball-and-stick end-to-end + validation plots`

---

## Phase 4 — Experimental data & fine-tuning

**Goal:** fine-tune a pretrained simulation model on real voltage traces using voltage-loss only.

**Deliverables**
- Extension to `HybridLoss`: `mask_channels=True` path (already sketched in Phase 2) — zero out channel component, skip gradient on `true_params`.
- `predict_finetuneExp.py` or extension of `batchShifterFinetune.slr`: load `blank_model.pth` + `ckpt.pth`, swap in hybrid loss with `channel_weight=0`, train on experimental dataset dir (e.g. `/global/homes/k/ktub1999/ExperimentalData/PyForEphys/BBP_cADpyr_Step_last4/`).
- Normalization audit doc (appended to `structure.md`): confirm experimental voltage traces get the same per-sample-per-probe z-score that simulation data gets in `format_bbp3_for_ML.py`. If not, add the preprocessing step.
- Sanity script: after fine-tune, run the jaxley forward on the predicted channel params for each experimental trace and overlay with the measured trace.

**Tests**
- Fine-tune 10 epochs on a small experimental dir.
- Assert voltage loss decreases monotonically.
- Plot overlays for 20 held-out traces — visual check + numerical RMSE.
- Assert no backward pass references the (now absent) channel ground truth (i.e. removing that tensor from the batch doesn't raise).

**Commit:** `phase 4: experimental fine-tune with voltage-only loss + overlays`

---

## Phase 5 — Scale to L5TTPC

**Goal:** same pipeline, full biophysical complexity, reasonable training time.

**Phase 1 measurements (A100, stim=5k50kInterChaoticB, no param overrides):**
- Single L5TTPC forward: 15.8 s (including JIT); B=4 OOMs with a 556 MB allocation.
- Single ball-and-stick forward: 2.8 s; B=64 fits and scales (23 sims/s).
- Voltage trace vs `Neuron_Jaxley/sim_jaxley_L5TTPC1.py` reference DIVERGES (max|Δ|=93 mV) because we deliberately skip the apical Ih distance gradient (CNN emits one scalar per conductance → must broadcast uniformly). Either regenerate the reference without the gradient or accept the drift.

Those numbers are the concrete targets for Phase 5 (memory + throughput) before hybrid training on L5TTPC is tractable.

**Deliverables**
- `toolbox/jaxley_cells/l5ttpc.py` filled in — matches the 19 parameter names in `parName` exactly, in the same order, so `pred_params[:, i]` maps to the right channel.
- Output-layer size already matches (19 for cADpyr). No model surgery needed.
- Batched simulation: `jax.vmap(simulate)` over the full local batch, benchmark memory + step time, reduce `local_batch_size` if needed.
- If a single jaxley L5TTPC simulation is too slow, implement:
  - shorter simulated windows during training (subsample the 4000-bin window),
  - `jax.checkpoint` in the time-stepping loop,
  - optionally run voltage loss on a subset of steps per epoch (mixed schedule).
- Final `structure.md` refresh + a `docs/phase5/summary.md` with:
  - training curves,
  - per-parameter RMSE on simulated test split,
  - voltage RMSE on experimental test traces,
  - wall-clock per training step (channel-only vs hybrid).

**Tests**
- Single-step forward on batch=4 L5TTPC: shape + finite values.
- One-epoch training (reduced samples) fits in 8 GPUs × 30 min.
- Final experimental-trace overlays committed to `docs/phase5/`.

**Commit:** `phase 5: L5TTPC hybrid training + final report`

---

## Execution plan (after open questions are resolved)

Per phase:
1. Allocate interactive node (commands above).
2. Write code.
3. Run the phase's tests inside `shifter`.
4. If green → `git add` the new files + the minimal edits → commit with the phase tag → push (branch TBD).
5. Report results + next-phase readiness to user before starting next phase.

I will not push without your answer on Q1 + Q6.
