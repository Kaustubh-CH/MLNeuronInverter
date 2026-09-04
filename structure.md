# structure.md

Living map of the neuroninverter codebase. The hand-written sections are meant to be edited by humans (or by Claude at the start of each phase); the auto-generated appendix at the bottom is refreshed by `toolbox/refresh_structure.py` (pre-commit hook installs via `scripts/install_hooks.sh`).

## 1. Purpose (one paragraph)

Given voltage traces from a biophysically-detailed neuron simulation (BBP cell models, run via NEURON + custom MPI infra) or real patch-clamp recordings, a 1D-CNN predicts the 19 underlying ion-channel conductances + capacitances + leak reversal. Training data lives in sharded HDF5 "packs" on `$PSCRATCH`. Current loss is parameter-space MSE. The `CNN_Jaxley` branch adds a physics-based voltage loss computed by running a differentiable `jaxley` forward simulation on the predicted parameters and comparing V-traces.

## 2. Data flow

```
NEURON sims (DL4neurons2, external repo)
   └─ raw output: /pscratch/.../<jid>/<cell>/L*_*-v3-*-c0.h5
                  per-sample: 19 phys_par + 5 stim waveforms + (T=4000, probes=3, stims=5) volts

packBBP3/aggregate_Kaustubh*.py        [login node, ~2 min/cell]
   └─ aggregates 1000 raw h5s → <cell>.simRaw.h5
      shapes: phys_par (512000, 19) f32 | unit_par (512000, 19) f32
              volts (512000, 4000, 3, 5) f16 | stim waveforms (4000,) f32 each

packBBP3/format_bbp3_for_ML*.py        [CPU node, ~100–400 GB RAM, ~8 min/cell]
   └─ per-sample per-probe z-score of voltage → cast f16
      8/1/1 train/valid/test split → <cell>.mlPack1.h5
      shapes: train_volts_norm (409600, 4000, 3, 5) f16
              train_unit_par (409600, 19) f32
              <stim_name> (4000,) f32  (kept top-level so jaxley can use them)

toolbox/Dataloader_H5.py               [GPU node, one shard per rank]
   └─ reads full pack into RAM; serialize_stims=True treats each stim as
      an independent sample; optional efel-derived extras (AP count) when
      use_manual_features=True
      batch: (volts, [extras,] unit_par) → Trainer

toolbox/Trainer.py
   └─ DDP training loop over toolbox/Model.py (or Model_Multi, Transformer_Model)
      inline MSELoss on unit_par
      writes blank_model.pth, checkpoints/ckpt.pth, sum_train.yaml, tb_logs/

predict.py / predictExp.py
   └─ load blank+ckpt, run infer on test split or experimental h5 dir,
      dump sum_pred_*.yaml + MLoutput.h5 + PNG plots via toolbox/Plotter.py

toolbox/unitParamConvert{,Hdf5}.py
   └─ uses sum_train.yaml['input_meta']['phys_par_range'] to invert the
      log-scaled unit→physical mapping and emit S/cm², µF/cm², mV values
```

## 3. Module hierarchy

| Concern                     | File                                         | Key symbols |
|----------------------------|----------------------------------------------|-------------|
| Entry (train)              | `train_dist.py`                              | `get_parser`, `__main__` — reads `<design>.hpar.yaml`, wires DDP |
| Entry (predict)            | `predict.py`, `predictExp.py`, `predictExpIDX.py` | `load_model`, `model_infer`, `compute_residual` |
| Trainer (loop, AMP, ckpt)  | `toolbox/Trainer.py`                         | `Trainer.__init__`, `Trainer.train`, `self.criterion = MSELoss` |
| Model (1D CNN + FC)        | `toolbox/Model.py`                           | `MyModel` — stack of `Conv1d/MaxPool1d/BN/Dropout/FC` |
| Model (parallel-stim)      | `toolbox/Model_Multi.py`, `Model_Multi_Stim.py` | alternate `MyModel` selected when `parallel_stim: True` |
| Model (transformer)        | `toolbox/Transformer_Model.py`               | `TransformerEncoder` — selected when `model_type: Transformers` |
| Dataloader (single cell)   | `toolbox/Dataloader_H5.py`                   | `get_data_loader`, `Dataset_h5_neuronInverter` |
| Dataloader (ONTRA multi)   | `toolbox/Dataloader_multiH5.py`              | same API, reads multiple H5s |
| HPO                        | `RayTune.py`                                 | `Raytune`, `trainable(params)` — ASHA + Optuna search |
| Plotting                   | `toolbox/Plotter.py`, `Plotter_Backbone.py`  | `Plotter_NeuronInverter` |
| Param unit↔physical        | `toolbox/unitParamConvert.py`, `unitParamConvertHdf5.py` | reads `sum_train.yaml['input_meta']['phys_par_range']` |
| IO helpers                 | `toolbox/Util_IOfunc.py`, `Util_H5io3.py`    | `read_yaml`, `write_yaml`, `write3_data_hdf5` |
| Experiment utilities       | `toolbox/Util_Experiment.py`                 | efel-based feature extraction |
| SLURM entry                | `batchShifter.slr`, `batchShifter_ray.slr`, `batchShifterMultiCell.slr`, `batchShifterOntraInh.slr`, `batchShifterFinetune.slr` | each `cp -rp $codeList $wrkDir; cd $wrkDir; srun shifter ...` |
| Data packing               | `packBBP3/aggregate_*.py`, `format_bbp3_for_ML*.py` | see `packBBP3/Readme-packing` |
| HF publish                 | `upload_to_hf.py`, `predict_from_hf.py`      | blank_model.pth + ckpt.pth + sum_train.yaml → HF repo |
| Predicted-param evaluation | `predicted_data_analysis/{submit_matrix.sh, score_cells.py, rank_cells.py}` | 5×5 cross-amplitude matrix over 13 excitatory cells, σ-normalised eFEL scoring, cell ranking (see its README.md) |

### 3.a `<design>.hpar.yaml` config surface

Single source of truth for model shape + training. Examples: `m8lay_vs3.hpar.yaml` (current default), `L5TTPCRay.hpar.yaml`, `InhOntraRay.hpar.yaml`, `MultuStim.hpar.yaml`. Keys consumed:

- `data_path.<facility>` — per-facility root dir (Perlmutter/CoriGPU/Summit)
- `data_conf.{serialize_stims, append_stim, parallel_stim, num_data_workers, max_glob_samples_per_epoch}`
- `max_epochs`, `batch_size`, `const_local_batch`, `validation_period`, `log_freq_per_epoch`
- `use_manual_features` (bool) — if True, dataloader yields extras tensor; model concatenates after flatten
- `tb_show_graph`, `save_checkpoint`, `resume_checkpoint`
- `do_ray` (bool) — triggers `RayTune.py` path instead of plain Trainer
- `model_type: CNN | Transformers`
- `opt_pytorch.{amp, apex, autotune, zerograd}`
- `train_conf.{warmup_epochs, optimizer: [name, initLR], LRsched: {plateau_patience, reduceFactor}}`
- `model.{myId, num_cnn_blocks, conv_block.{filter, kernel, pool}, fc_block.{dims, dropFrac}, instance_norm_slot, layer_norm, batch_norm_cnn_slot, batch_norm_flat}`

CLI flags in `train_dist.py` override a subset: `--cellName --probsSelect --stimsSelect --validStimsSelect --initLR --epochs --data_path_temp --do_fine_tune --fine_tune-{blank_model,checkpoint_name}`.

### 3.b SLURM → training contract

Every `batchShifter*.slr` does: `cp -rp $codeList $wrkDir && cd $wrkDir && srun shifter bash toolbox/driveOneTrain.sh`. `$codeList` is a literal whitespace-separated list inside each .slr. **Any new file needed at runtime must be added to that list**, otherwise it won't exist in `$wrkDir` and the shifter command will fail.

## 4. Jaxley integration (phase plan)

Reference implementations live at `/pscratch/sd/k/ktub1999/Neuron_Jaxley/`:
- `sim_jaxley.py` — ball-and-stick (HH soma + 5-comp passive dendrite)
- `sim_jaxley_L5TTPC1.py` + `bbp_channels_jaxley.py` — BBP L5 pyramidal with `NaTs2_t, NaTa_t, Nap_Et2, SKv3_1, K_Tst, K_Pst, Im, Ih, CaComplex, BBPLeak`
- SWC morphology: `/pscratch/sd/k/ktub1999/Neuron_Jaxley/results_detailed_morphology/L5_TTPC1.swc`
- Stim CSVs: `/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/5k*.csv`. Phase 3+ uses `5k50kInterChaoticB.csv`.

### Phase attachment points in this repo

| Phase | New files                                                      | Existing files touched (additive only) |
|-------|----------------------------------------------------------------|----------------------------------------|
| 0     | `structure.md`, `toolbox/refresh_structure.py`, `scripts/hooks/pre-commit`, `scripts/install_hooks.sh`, `PLAN.md` | — |
| 1     | `toolbox/jaxley_cells/{__init__, ball_and_stick, l5ttpc}.py`, `toolbox/JaxleyBridge.py`, `toolbox/jaxley_utils.py`, `toolbox/tests/test_jaxley_bridge.py`, `environment.yml`, `scripts/phase1_tests.slr` | — |
| 2     | `toolbox/HybridLoss.py`, `m8lay_vs3_jaxley.hpar.yaml`, `toolbox/tests/test_hybrid_loss.py` | `toolbox/Trainer.py` (one `if params.get('use_voltage_loss'):` swap of `self.criterion`), `batchShifter.slr` `$codeList` (+1 entry) |
| 3     | `scripts/gen_ball_and_stick_data.py`, `plotJaxleyValidation.py`, `batchShifterJaxley.slr`, `docs/phase3/*.png` | — |
| 4     | `predict_finetuneExp.py` (or edits to `batchShifterFinetune.slr`), `docs/phase4/*.png` | `toolbox/HybridLoss.py` (mask_channels path) |
| 5     | `docs/phase5/summary.md`, `docs/phase5/*.png` | `toolbox/jaxley_cells/l5ttpc.py` (fill in) |

### Unit→physical mapping (shared across phases)

The jaxley cell expects conductances in S/cm² and capacitance in µF/cm². The CNN outputs are `unit_par ∈ ℝ^19` in the normalized log-range defined by `sum_train.yaml['input_meta']['phys_par_range']` (each row is `[center, log10_halfspan, unit]`). The conversion in `toolbox/unitParamConvert.py` is `phys = center * 10**(unit * half_log)`. `toolbox/jaxley_utils.py` (Phase 1) will mirror this inside JAX so gradients flow from voltage back through the CNN.

## 5. Quick commands

```bash
# activate the conda env (canonical runtime for the CNN_Jaxley branch)
module load conda
conda activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
# (rebuild from scratch: `conda env create --prefix <prefix> -f environment.yml`)

# run the Phase 1 bridge tests (gradcheck + vmap + cache; CPU, ~20 s)
python -m toolbox.tests.test_jaxley_bridge

# regenerate auto-sections of this file
python toolbox/refresh_structure.py

# check staleness (exits 1 if structure.md differs from regenerated)
python toolbox/refresh_structure.py --check

# install the pre-commit hook that re-runs refresh on staged .py changes
bash scripts/install_hooks.sh
```

<!-- =============================================================== -->
<!-- auto:begin  DO NOT EDIT by hand below this line.                  -->
## Appendix - auto-generated file inventory

_Regenerate with `python toolbox/refresh_structure.py`._

### Top-level scripts

- `All_probes_allCells.py`
    - defs: read_and_plot_etype_data
- `RayTune.py`
    - defs: threadTrain, trainable, Raytune
- `build_roy_expH5.py` - Pack the Roy* ABF sweeps into per-stim mlPack1 HDF5s for NeuronInverter.
    - defs: main, iv_main
- `chaoticramp_variance_probe.py` - Look directly at the ChaoticRamp trace ensemble per channel and test which
    - defs: smooth, spike_rate, main
- `convertPred.py`
    - defs: extract_efel_features_from_volts, normalize_volts, resample_by_interpolation
- `evaluate_voltage.py` - Evaluate a HybridLoss/voltage-trained model.
    - defs: get_parser, load_trained_model, load_test_data, main
- `feature_channel_sensitivity.py` - Feature x Channel sensitivity matrix for the CA3 (or any jaxley-cell) inverse.
    - defs: get_parser, resolve_phys_range, operating_points, feature_list, main
- `findSpikesExpC.py` - identify spikes in experimental wave forms
    - defs: get_parser, score_me, pack_scores, M_build_metaData, M_save_tspike
- `formatExpC4ML.py` - select subset of experimental waveforms and re-pack them for ML-predictions
    - defs: get_parser, Plotter
- `formatPredRoy4ML.py` - 2nd iteration of predictions
- `formatRawExpC.py` - agregate and format Roy's experimental data for one cell
    - defs: get_parser, find_exp_log, survey_routines, extract_routines, QA_one
- `formatSimB4ML.py` - select subset of simulated waveforms and re-pack them for ML-predictions
    - defs: get_parser
- `formatSimV8kHz.py` - formats packed Vyassa simu (aka production) for bbp153
    - defs: get_parser
- `make_interpolated_stims.py` - Resample every stimulus CSV to a fixed number of time steps.
    - defs: load_current, main
- `ood_probe.py` - Out-of-range (extrapolation) probe for the best chaoticRamp CA3 model.
    - defs: get_parser, main
- `plot2Dtester.py`
    - defs: args1
- `plotConductDrift.py` - plot scores and waveforms w/ spikes
    - defs: get_parser, Plotter, process_one
- `plotExpCSurvey.py` - inspect formated  experiment
    - defs: get_parser, print_exp_summary, Plotter
- `plotJaxleyValidation.py` - Phase 3 validation plotter.
    - defs: get_parser, load_trained_model, load_test_data, cnn_forward, jaxley_forward, per_param_metrics, voltage_metrics, plot_param_scatter, plot_voltage_overlay, plot_voltage_summary, main
- `plotPredSurvey.py` - plot scores and waveforms w/ spikes
    - defs: get_parser, Plotter
- `plotPredWave_fromRoy.py` - overlays  wavformes predicted by Roy to ML pred coductances
    - defs: get_parser, Plotter
- `plotRawExpC.py` - plot raw experimental data collected by Roy, a single input file
    - defs: get_parser, Plotter
- `plotRawOverlayExpC.py` - plot overaly of raw experimental data collected by Roy
    - defs: get_parser, Plotter
- `plotSpikesExpC.py` - plot scores and waveforms w/ spikes
    - defs: get_parser, Plotter, M_save_summary
- `plotSurveyExpC.py` - inspect formated  experiment
    - defs: get_parser, print_exp_summary, Plotter
- `plot_all_stims_paula.py` - Plot every recording in exp_data_paula: one page per ABF file.
    - defs: style, n_spikes, minmax_decimate, new_page, finish_page, page_iv, page_roy, page_gapfree, main
- `plot_ball_variation.py` - Characterize voltage-trace variation across the ball-and-stick synthetic dataset.
- `plot_ca3_default_trace.py` - CA3 pyramidal default-parameter voltage trace vs the ca3_synth_v2 dataset.
- `plot_ca3_multistim.py` - CA3 pyramidal multi-stim per-page PDF.
- `plot_ca3_traces.py` - Plot 100 voltage traces (raw mV) from the ca3_synth_v2 dataset.
- `plot_default_trace.py` - Ball-and-stick voltage trace at DEFAULT ion-channel values vs the dataset.
- `plot_efel_vs_channel.py` - Distribution of soft-eFEL features vs ion-channel value for the ball-and-stick pack.
- `plot_exp_overlay.py` - Plot predicted unit-params + voltage overlays for EXPERIMENTAL data.
    - defs: get_parser, zscore, main
- `plot_exp_overlay_royv2.py` - Voltage-error evaluation of a CA3 model on Paula's Roy v2 experimental pack.
    - defs: get_parser, zscore, zfix, spikes_pos, sim_family, main
- `plot_halved_vs_full_inh.py` - Compare ALL_CELLS_Inhibitory error (same-cell / intrapolation / extrapolation)
    - defs: load
- `plot_ion_channels_compare.py` - Side-by-side ion-channel comparison figure (truth-vs-pred 2D density per channel),
    - defs: section_cmap, load_domain, panel, main
- `plot_roy_input_vs_output.py` - Overlay the measured Roy traces (model input) against the NEURON traces
    - defs: load_jobs, jobs_for, n_spikes, spike_label, load_sim, style, main
- `plot_roy_predicted_params.py` - Plot the ion-channel / passive parameters the probescan_exc model predicted
    - defs: load, style, main
- `plot_roy_traces.py` - Visualize the "Roy*" voltage traces in exp_data_paula as a multi-page PDF.
    - defs: load_roy_traces, n_spikes, style, main
- `plot_stage_overlays.py` - Overlay experimental voltage traces with the sims from EVERY model stage.
    - defs: get_parser, zfix, spikes_pos, sim_family, main
- `plot_stims_pdf.py` - Plot every stimulus CSV in a directory into one multipage PDF (4 per page).
    - defs: load_current, main
- `plot_voltage_traces.py` - Plot 100 voltage traces (raw mV) from the ball-and-stick synthetic pack.
- `plot_voltage_traces_pdf.py` - Multi-page PDF: one voltage trace per page (raw mV) for the ball-and-stick pack.
- `predict.py` - read trained net : model+weights
    - defs: get_parser, load_model, model_infer, compute_residual
- `predictExp.py` - PREDiction Kaustubh
    - defs: get_parser, load_model, model_infer, compute_dummy_residual
- `predictExpIDX.py` - PREDiction Kaustubh
    - defs: get_parser, load_model, model_infer
- `predictTestPlot.py` - read trained net : model+weights
    - defs: get_parser, load_model, model_infer, compute_residual
- `predict_exp.py` - read trained net : model+weights
    - defs: get_parser, model_infer_exper, M_get_phys_packing
- `predict_from_hf.py` - predict_from_hf.py
    - defs: load_model_from_hf, load_traces_from_h5, predict, get_parser, main
- `roy_unit_to_phys.py` - Convert the Roy* predicted unit parameters to physical parameters, in three
    - defs: convert, main
- `sensitivity_analysis.py` - Identifiability / sensitivity analysis for the CA3 (or any jaxley-cell) inverse problem.
    - defs: get_parser, main
- `sensitivity_variation.py` - One-at-a-time (OAT) voltage-variation sensitivity analysis for jaxley cells.
    - defs: get_parser, compute_efel_variation, compute_metric_variation, compute_variance_metrics, resolve_phys_range, discover_stims, build_unit_samples, simulate_stim, main, write_outputs, write_efel_outputs, write_scalar_metric_outputs, _plot_su...
- `sensitivity_variation_compare.py` - Compare two sensitivity_variation runs (e.g. native vs interpolated stims),
    - defs: read_matrix, spearman, align, parse_kv, main
- `sensitivity_variation_merge.py` - Merge chunked sensitivity_variation.py outputs into one combined result.
    - defs: read_matrix, main
- `train_dist.py` - Not running on CPUs !
    - defs: get_parser
- `upload_best_per_cell_to_hf.py` - upload_best_per_cell_to_hf.py
    - defs: load_summary, is_complete, find_best_per_cell, slim_checkpoint, build_config, write_cell_extras, root_readme, upload_all, get_parser, main
- `upload_to_hf.py` - upload_to_hf.py
    - defs: load_summary, make_repo_name, make_model_card, upload_model, get_parser, main
- `yamlMaker.py`
    - defs: get_parser

### toolbox/

- `toolbox/Dataloader_H5.py`
    - defs: _safe_genfromtxt, get_data_loader, Dataset_h5_neuronInverter
- `toolbox/Dataloader_multiH5.py`
    - defs: get_data_loader, Dataset_multiH5_neuronInverter
- `toolbox/HybridLoss.py` - Hybrid channel + voltage loss for the jaxley physics-supervised path.
    - defs: _GradScale, HybridLoss, _ChannelOnlyAdapter, _log_jax_devices_once, _read_phys_par_range_from_h5, _resolve_grad_precond_weights, build_hybrid_loss
- `toolbox/JaxleyBridge.py` - Torch <-> Jaxley bridge.
    - defs: _CellHandle, _build_handle, get_handle, _torch_to_jax, _jax_to_torch, _JaxleySimulate, simulate_batch, output_shape, param_keys, clear_cache
- `toolbox/Model.py`
    - defs: MyModel
- `toolbox/Model2d.py`
    - defs: MyModel
- `toolbox/Model_Multi.py`
    - defs: MyModel
- `toolbox/Model_Multi_Stim.py`
    - defs: CnnBlock, FcBlock, MyModel
- `toolbox/Plotter.py`
    - defs: get_arm_color, get_hist_color_map, Plotter_NeuronInverter
- `toolbox/Plotter_Backbone.py`
    - defs: roys_fontset, Plotter_Backbone
- `toolbox/Trainer.py`
    - defs: patch_h5meta, average_gradients, Trainer
- `toolbox/Transformer_Model.py`
    - defs: PositionalEncoding, TransformerEncoder
- `toolbox/Util_Experiment.py`
    - defs: rebin_data1D, rebin_data2D, md5hash, id_generator, SpikeFinder
- `toolbox/Util_H5io3.py`
    - defs: write3_data_hdf5, append_data_hdf5, read3_data_hdf5
- `toolbox/Util_IOfunc.py`
    - defs: read_yaml, write_yaml, write_data_hdf5, read_data_hdf5, read_one_csv, write_one_csv, dateT2Str, dateStr2T, md5hash, build_name_hash, expand_dash_list
- `toolbox/aggregate_loss.py`
    - defs: get_parser
- `toolbox/jaxley_utils.py` - Helpers shared across the jaxley voltage-loss path.
    - defs: normalize_volts_fixed, normalize_stim_fixed, phys_par_range_to_arrays, unit_to_phys_np, unit_to_phys_jax, load_stim_csv, upsample_stim, downsample_step
- `toolbox/recal_metrics.py` - Per-channel recovery metrics that CREDIT a good-but-offset diagonal, plus a
    - defs: _rankdata, _r2, channel_metrics, fit_affine, apply_affine, metrics_table, print_table, _load_npz, main
- `toolbox/refresh_structure.py` - Regenerate the auto-appendix section of structure.md.
    - defs: FileSummary, tracked_files, summarize, collect, render, splice, main
- `toolbox/soft_dtw.py` - Differentiable soft-DTW distance for voltage-trace matching.
    - defs: _softmin3, soft_dtw, _downsample, soft_dtw_loss, _reference_soft_dtw
- `toolbox/soft_efel.py` - Differentiable (PyTorch) surrogate for 11 eFEL electrophysiology features.
    - defs: _softplus_max, _softplus_min, _soft_select, _spike_times, soft_efel_features, real_efel_features, _pearson, _self_test
- `toolbox/trace_metrics.py` - Differentiable full-trace distance metrics for voltage traces (torch, GPU).
    - defs: gaussian_kernel, multiscale_blurred_mse, decimate, _softmin3, _sdtw_raw, soft_dtw, gaussian_smooth, smoothed_ensemble_std, parameter_explained_var, soft_dtw_to_ref
- `toolbox/unitParamConvert.py`
    - defs: get_parser
- `toolbox/unitParamConvertHdf5.py`
    - defs: get_parser

### toolbox/jaxley_cells/

- `toolbox/jaxley_cells/__init__.py` - Registry of jaxley cell builders for the hybrid voltage-loss path.
    - defs: CellSpec, register, get, list_cells
- `toolbox/jaxley_cells/ball_and_stick.py` - Ball-and-stick cell builder for the hybrid voltage-loss path.
    - defs: _build, _attach_stim, _attach_record, _spec
- `toolbox/jaxley_cells/ball_and_stick_bbp.py` - Ball-and-stick cell with BBP channels (multi-section, multi-channel).
    - defs: _build, _attach_stim, _attach_record, _spec
- `toolbox/jaxley_cells/ca3_pyramidal.py` - CA3 Pyramidal Neuron (single-comp soma) — Jaxley port.
    - defs: _build, _attach_stim, _attach_record, _spec
- `toolbox/jaxley_cells/l5pc_jaxley.py` - Self-contained L5PC cell builder using an *existing* jaxley morphology.
    - defs: _build, _attach_stim, _attach_record, _spec
- `toolbox/jaxley_cells/l5ttpc.py` - L5TTPC cell builder for the hybrid voltage-loss path.
    - defs: _apply_apical_ih_gradient, _build, _attach_stim, _attach_record, _spec
- `toolbox/jaxley_cells/l5ttpc_multiprobe.py` - Multi-probe L5TTPC cell builder (EXPERIMENT 2).
    - defs: _branch_indices, _add_extra_records, _build, _attach_stim, _attach_record, _spec
- `toolbox/jaxley_cells/soma_only.py` - Single-compartment HH soma cell.
    - defs: _build, _attach_stim, _attach_record, _spec

### toolbox/tests/

- `toolbox/tests/__init__.py`
- `toolbox/tests/bench_fwd_bwd.py` - Forward + backward throughput across all jaxley cells in the repo.
    - defs: _sync, _time, bench_cell, main
- `toolbox/tests/bench_gpu_ca3.py` - GPU bench for CA3 Pyramidal — t_max=500 ms apples-to-apples vs NEURON.
    - defs: _build, _time, main
- `toolbox/tests/bench_gpu_l5ttpc.py` - GPU bench for the jaxley voltage-loss path.
    - defs: _build_loss_fn, _default_phys_jnp, _time_fn, _peak_mem_mb, _reset_peak_mem, run_combo, main
- `toolbox/tests/bench_jaxley_cells.py` - Correctness + throughput benchmark for the Phase 1 jaxley cells.
    - defs: _print, _default_params_tensor, _count_spikes, check_against_reference, bench_throughput, main
- `toolbox/tests/bench_solvers.py` - Solver + cell + batch sweep bench at dt=0.1 ms, t_max=100 ms.
    - defs: _default_params, build_simulate, time_warm, _extract_soma_trace, run_matrix, save_csv, save_text_summary, save_trace_plots, save_bars, cmd_main, cmd_scaling, build_parser, main
- `toolbox/tests/phase2_demo.py` - End-to-end Phase 2 demo.
- `toolbox/tests/plot_bench_bar_b128.py` - Grouped bar chart: sims/sec vs batch size, across every GPU config + CPU.
    - defs: _gpu_value, _cpu_value, _plot_panel, main
- `toolbox/tests/plot_bench_comparison.py` - Pull every completed bench CSV into a single comparison figure.
    - defs: _load_gpu, _load_cpu, _plot_panel, main
- `toolbox/tests/sens_sweep_ball.py` - TEMPORARY sensitivity sweep for ball_and_stick.
    - defs: unit_to_phys, zscore, main
- `toolbox/tests/sim_neuron_ca3.py` - NEURON-side reference simulation for the CA3 Pyramidal model.
    - defs: _load_neuron, build_cell, run_protocol, _resample_to_grid, _make_step, _save, run_test1_rest, run_test2_subthresh, run_test3_suprathresh, run_test4_fI, run_test5_paramsweep, run_test6_apshape, run_test7_walltime, main
- `toolbox/tests/test_ca3_neuron_vs_jaxley.py` - CA3 NEURON-vs-Jaxley comparison harness.
    - defs: _jaxley_runner, detect_spikes, coincidence_fraction, half_width, ahp_depth, Summary, _load_neuron, _plot_overlay, _gbar_from_npz, run_test1_rest, run_test2_subthresh, run_test3_suprathresh, run_test4_fI, run_test5_paramsweep, run_test6_a...
- `toolbox/tests/test_hybrid_loss.py` - Phase 2 tests for toolbox.HybridLoss.
    - defs: _shrink_t_max, _restore_t_max, test_zero_recovers_mse, test_adapter_is_mse, test_voltage_forward_finite, test_voltage_grad_flows, test_mask_channels_skips_channel_loss, test_unit_to_phys_matches_numpy, test_factory_channel_only_passthrou...
- `toolbox/tests/test_jaxley_bridge.py` - Phase 1 tests for toolbox.JaxleyBridge.
    - defs: test_registry_lists_both_cells, test_shapes, test_cache_hit_no_recompile, test_vmap_matches_serial_loop, test_gradcheck_tiny, test_fresh_state_per_call, test_l5ttpc_registers_but_do_not_build, main

### scripts/

- `scripts/gen_ball_and_stick_data.py` - Generate a synthetic mlPack1.h5 from a registered jaxley cell.
    - defs: _load_source_cell, _build_phys_par_range, generate_voltages, normalize_volts_fixed_scale, write_h5, main
- `scripts/gen_ca3_sharded.py` - Sharded multi-GPU generation of a CA3 mlPack1.h5 (single- or multi-stim).
    - defs: _phys_range, _load_cell, _slice, _draw_unit_par, _fam_assign, _vary_indices, worker, merge, main
- `scripts/gen_multistim_data.py` - Generate a JOINT 3-stimulus mlPack1.h5 from a registered jaxley cell (EXP 3).
    - defs: _load_source_cell, _build_phys_par_range, generate_voltages_one_stim, normalize_volts_fixed_scale, write_h5, main
- `scripts/voltage_loss_bias_probe.py` - Step-0 bias audit for the voltage-only training objective.
    - defs: get_parser, load_split_volts, loss_at, main
- `scripts/gen_best_stims_all.sh`
- `scripts/install_hooks.sh`
- `scripts/run_ca3_gen.sh`

### packBBP3/

- `packBBP3/Agg2.py` - re-pack samll hd5 NEURON output to one  6k-samples HD5 files
    - defs: get_parser, normalize_volts, get_h5_list, assemble_MD, import_stims_from_CVS, read_all_h5, clear_NaN_samples
- `packBBP3/Agg3.py` - re-pack samll hd5 NEURON output to one  6k-samples HD5 files
    - defs: get_parser, normalize_volts, get_h5_list, assemble_MD, import_stims_from_CVS, read_all_h5, clear_NaN_samples
- `packBBP3/aggregate_All65.py`
    - defs: get_parser, get_h5_list, clear_NaN_samples, fill_bigD_single_cell, import_stims_from_CVS, assemble_MD
- `packBBP3/aggregate_All65_noIndex.py` - re-pack samll hd5 NEURON output to one  6k-samples HD5 files
    - defs: get_parser, write3_data_hdf5_partial, append_data_hdf5_index, normalize_volts, get_only_directories, get_h5_list, assemble_MD, import_stims_from_CVS, read_all_h5, clear_NaN_samples
- `packBBP3/aggregate_Kaustubh.py` - re-pack samll hd5 NEURON output to one  6k-samples HD5 files
    - defs: get_parser, write3_data_hdf5_partial, append_data_hdf5_index, normalize_volts, get_h5_list, assemble_MD, import_stims_from_CVS, read_all_h5, clear_NaN_samples
- `packBBP3/aggregate_Kaustubh2.py`
- `packBBP3/aggregate_Kaustubh_feature.py` - re-pack samll hd5 NEURON output to one  6k-samples HD5 files
    - defs: get_parser, normalize_volts, get_h5_list, assemble_MD, import_stims_from_CVS, read_all_h5, clear_NaN_samples
- `packBBP3/exclude_params_inplace.py` - In-place column drop on already-packed mlPack1 H5 files.
    - defs: trim_h5, main
- `packBBP3/filter_pack_family.py` - Filter an existing Roy exp ca3ft mlPack down to ONE stimulus family,
- `packBBP3/format_bbp3_for_ML.py` - format samples for ML training
    - defs: get_parser, format_raw, read_meta_json
- `packBBP3/format_bbp3_for_ML_paralelly.py` - format samples for ML training
    - defs: get_parser, get_normal_stim, _process_efel_chunk, extract_efel_features_from_volts, format_raw, write_meta_json_hdf5, append_data_hdf5_index, read_meta_json_hdf5, read3_only_data_hdf5
- `packBBP3/format_bbp3_for_ML_paralelly_cell_wise.py` - format samples for ML training
    - defs: get_parser, get_normal_stim, format_raw, write_meta_json_hdf5, append_data_hdf5_index, read_meta_json_hdf5, read3_only_data_hdf5
- `packBBP3/format_bbp3_for_ML_paralelly_only_test.py` - format samples for ML training
    - defs: get_parser, get_normal_stim, format_raw, write_meta_json_hdf5, append_data_hdf5_index, read_meta_json_hdf5, read3_only_data_hdf5
- `packBBP3/format_royexp_for_ML.py` - Pack Paula's Roy inter-chaotic-stim recordings (roy_stims.h5) into an
    - defs: get_parser, main
- `packBBP3/format_vyassa_for_ML.py` - format samples for ML training
    - defs: get_parser, rebuildMD, addStim, format_raw
- `packBBP3/halve_inh_dataset.py` - Halve each cell's contribution in (already-shuffled) ONTRA Inhibitory datasets.
    - defs: get_parser, keep_index, stream_copy, is_sample_dataset, halve_file, main
- `packBBP3/ontraInhAllSubmit.py`
    - defs: get_parser
- `packBBP3/plotBaseVolts.py` - plot BBP3 simulation data
    - defs: get_parser, Plotter, import_stims_from_CVS
- `packBBP3/vet_volts.py` - plot BBP3 simulation: soma volts

<!-- auto:end -->
