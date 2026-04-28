# ball_and_stick_bbp — multi-section BBP-channel cell

**Branch:** `phase1/ball-stick-bbp-channels`
**Off of:** `CNN_Jaxley` @ `784be8c`
**Goal:** A new cell, registered alongside the existing 4-param HH
`ball_and_stick`, that uses real BBP channels at multiple sections so the
trainable param surface looks like a slice of L5TTPC's. The simple HH
ball-and-stick stays untouched — it's the bridge gradcheck cell.

## Cell layout (proposed)

Same morphology as `ball_and_stick`: soma branch + dendrite branch
(treated as the "apical-like" section). Two sections × six channels means
**up to 12 trainable params per section pair**, but we trim where it
doesn't make sense (e.g. NaTa_t lives only on axon in BBP, but we don't
have an axon — leave it out).

| section  | channels inserted                                                  |
|----------|--------------------------------------------------------------------|
| soma     | NaTs2_t, Nap_Et2, K_Tst, K_Pst, CaComplex (HVA + LVA + SK), BBPLeak |
| dendrite | NaTs2_t, K_Pst, Im, Ih, BBPLeak                                    |

Reversal potentials and Ca decay/gamma constants come from L5TTPC defaults
(`toolbox/jaxley_cells/l5ttpc.py:140-151`). No distance gradient on
dendritic Ih in this cell — it's left uniform so the Ih-fix branch can
solve that issue independently.

## Trainable param list (proposed, BBP-style names)

CNN-output order:

| idx | param                         | section  | jaxley key            |
|----:|-------------------------------|----------|-----------------------|
|   0 | gNaTs2_tbar_NaTs2_t_somatic   | soma     | NaTs2_t_gbar          |
|   1 | gNap_Et2bar_Nap_Et2_somatic   | soma     | Nap_Et2_gbar          |
|   2 | gK_Tstbar_K_Tst_somatic       | soma     | K_Tst_gbar            |
|   3 | gK_Pstbar_K_Pst_somatic       | soma     | K_Pst_gbar            |
|   4 | gCa_LVAstbar_Ca_LVAst_somatic | soma     | CaComplex_gCa_LVAst   |
|   5 | gCa_HVAbar_Ca_HVA_somatic     | soma     | CaComplex_gCa_HVA     |
|   6 | g_pas_somatic                 | soma     | BBPLeak_gLeak         |
|   7 | gNaTs2_tbar_NaTs2_t_apical    | dendrite | NaTs2_t_gbar          |
|   8 | gK_Pstbar_K_Pst_apical        | dendrite | K_Pst_gbar            |
|   9 | gImbar_Im_apical              | dendrite | Im_gbar               |
|  10 | gIhbar_Ih_apical              | dendrite | Ih_gbar               |
|  11 | g_pas_apical                  | dendrite | BBPLeak_gLeak         |

12 params total. (Final list reviewed during implementation.)

## Tasks

### Implementation

- [ ] Create `toolbox/jaxley_cells/ball_and_stick_bbp.py`. Mirror the
      shape of `ball_and_stick.py`: build soma + dendrite branches,
      `assign_groups`-style sectioning via `cell.branch(0)` / `cell.branch(1)`.
- [ ] Insert BBP channels per the layout table above. Pull channel
      classes from `/pscratch/sd/k/ktub1999/Neuron_Jaxley/bbp_channels_jaxley.py`
      (already imported by `l5ttpc.py`).
- [ ] Set reversal potentials, Ca decay/gamma, leak reversal — copy from
      `l5ttpc.py:140-151` for the soma values; pick BBP apical defaults
      for the dendrite values.
- [ ] Set trainable defaults to plausible BBP values (use values from the
      L5TTPC reference CSV under `/pscratch/sd/k/ktub1999/Neuron_Jaxley/`
      as the prior).
- [ ] `make_trainable` each entry; `entry_to_cnn_idx` is the identity list.
- [ ] Register as `ball_and_stick_bbp` in `toolbox/jaxley_cells/__init__.py`.
- [ ] Update `toolbox/tests/bench_jaxley_cells.py::_default_params_tensor`
      so `bench_gpu_l5ttpc` can run the new cell.

### Verification

- [ ] **Build smoke test:** `python -c "from toolbox import jaxley_cells;
      c = jaxley_cells.get('ball_and_stick_bbp').build_fn()[0]; print(c)"`.
- [ ] **Forward at default params:** trace soma voltage for the
      `5k50kInterChaoticB` stim; sanity-check it spikes (or doesn't, if
      defaults are sub-threshold) and is bounded in [-90, +50] mV.
- [ ] **Param count:** `len(spec.param_keys) == len(cell.get_parameters())`.
- [ ] **vmap-jit forward:** B=16 vmap'd `jx.integrate` runs without
      shape errors.
- [ ] **Bench:** add `ball_and_stick_bbp` to the GPU bench
      (`toolbox/tests/bench_gpu_l5ttpc.py`), run B in {1,4,16,64,128}
      fwd + fwd_bwd, save under `/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_gpu_bench_bbp/bench_gpu.csv`.

### Out of scope for this branch

- Distance gradient on Ih (lives in the sibling Ih-fix worktree).
- Adding axon (different morphology — that's the dropped #3).
- Phase 2 hybrid loss / trainer hook.

## Reference paths

- Existing simple cell: `toolbox/jaxley_cells/ball_and_stick.py`
- L5TTPC for channel/reversal defaults: `toolbox/jaxley_cells/l5ttpc.py`
- Channel impls:        `/pscratch/sd/k/ktub1999/Neuron_Jaxley/bbp_channels_jaxley.py`
- BBP CSV parameter prior values: `/pscratch/sd/k/ktub1999/Neuron_Jaxley/results_detailed_morphology/*.csv`
