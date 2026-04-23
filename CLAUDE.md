# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Neuron-Inverter: given voltage traces from a simulated (BBP) or experimental neuron, a CNN predicts the underlying biophysical conductance/capacitance parameters (the "inverse" of a NEURON simulation). Training data is produced externally by running NEURON simulations on BBP cell models; this repo handles packing that data into H5 shards, training the CNN on NERSC Perlmutter GPUs, inferring on simulated test sets and real experimental data, and converting predicted unit-normalized parameters back to physical units.

## Target environment

This code runs on NERSC Perlmutter via SLURM + Shifter containers. It is **not expected to run on a laptop** and training will not run on CPU. Dataset paths (`/pscratch/...`, `/global/homes/k/ktub1999/...`) and `shifter --image=nersc/pytorch:...` invocations are hard-coded throughout the SLURM scripts and `.hpar.yaml` files. When editing, keep these absolute paths — they are load-bearing, not examples.

## Commands

### Interactive training (from `Readme.perlmutter`)
```bash
salloc -C gpu -q interactive -t4:00:00 --gpus-per-task=1 \
       --image=nersc/pytorch:ngc-21.08-v2 -A m2043_g \
       --ntasks-per-node=4 -N 1
export MASTER_ADDR=`hostname`
export MASTER_PORT=8881
srun -n1 shifter python -u ./train_dist.py \
       --cellName L6_TPC_L1cADpyr1 --numGlobSamp 5120 \
       --probsSelect 0 1 2 --stimsSelect 0 --design m8lay_vs
```

### Batch training
- `sbatch batchShifter.slr <data_path> [design]` — primary 8-node training job; after training it chains `predict.py` over every stim index, `predictExp.py` over several experimental datasets, and `unitParamConvert*.py` to map unit→physical parameters.
- `sbatch batchShifter_ray.slr <data_path>` — Ray Tune HPO on a single node (4 GPUs/task), driven by `RayTune.py`.
- `sbatch batchShifterMultiCell.slr <data_path> <design> <cellName>` — single-cell training, used by `MultiCellBatchShiferSubmit.sh` to fan out a training job per cell in a list.
- `sbatch batchShifterOntraInh.slr` / `ontraInhSubmit.sh` — "ONTRA" (One-Network-To-Rule-all) runs over inhibitory cell populations.

### Inference
Always 1 GPU (or CPU). Do not distribute.
```bash
./predict.py --modelPath <run_dir>/out            # simulated test set
./predictExp.py --modelPath <run_dir>/out \
                --expFile <dir of experimental h5>  # real ephys data
```
`<run_dir>` is the per-job directory under `$SCRATCH/tmp_neuInv/bbp3/<cellName>/<wrkSufix>/` that `batchShifter.slr` creates by copying the code + hpar yaml into place.

### HuggingFace upload / remote inference
- `python upload_to_hf.py --model_dir <run_dir>/out --hf_user <user> --token ...` — packages `blank_model.pth`, `checkpoints/ckpt.pth`, and `sum_train.yaml` into an HF repo.
- `python predict_from_hf.py --repo <user>/neuron-<cell>-<design> --input <h5>` — round-trip inference from a published model.

### Data packing (see `packBBP3/Readme-packing`)
Two stages, both run outside shifter:
1. **Aggregate** 1000 raw NEURON `*.h5` into one `<cell>.simRaw.h5` — fast (~2 min), runs on login node:
   ```bash
   module load pytorch
   ./aggregate_Kaustubh.py --simPath <raw_dir> --outPath <out_dir> --jid <slurm_jid>_1
   ```
2. **Format for ML** — needs ~100–400 GB RAM, runs on a dedicated CPU node; splits 512k samples 8/1/1 into `train/valid/test`, normalizes voltages per-sample-per-probe to mean 0 / std 1, casts to fp16, writes `<cell>.mlPack1.h5`:
   ```bash
   salloc -C cpu -q interactive -t4:00:00 -A m2043 -N 1
   module load pytorch
   ./format_bbp3_for_ML.py --dataPath <dir> --cellName <cell>
   ```
`bigPacker*.sh` and `bigDom*.sh` wrap these over cell/job lists. "Ontra" variants pack many cells into one shared ALL_CELLS h5.

### TensorBoard
TB logs are written to `<run_dir>/out/tb_logs/`. View by SSHing to Perlmutter, `module load pytorch`, `tensorboard --port 9600 --logdir=out`, then port-forwarding.

## Architecture

### Training flow (`train_dist.py` → `toolbox/Trainer.py`)
1. `train_dist.py` reads a `<design>.hpar.yaml` (e.g. `m8lay_vs3.hpar.yaml`, `L5TTPCRay.hpar.yaml`, `InhOntraRay.hpar.yaml`, `MultuStim.hpar.yaml`) — this YAML is the single source of truth for model shape, optimizer, scheduler, data path per facility, AMP/APEX flags, validation period, and whether Ray Tune is enabled.
2. CLI flags override a subset of YAML keys (LR, epochs, cell name, probes/stims, data path, Ray Tune min-loss threshold, fine-tune checkpoint). `--facility perlmutter|corigpu|summit` selects the correct `data_path` inside the YAML.
3. DDP is spun up via `torch.distributed` using `SLURM_PROCID`/`SLURM_NTASKS` for rank/world-size; `const_local_batch: True` means batch size per GPU is fixed and global BS scales (LR effectively scales too).
4. `Trainer` picks a model class from `toolbox/Model.py` / `Model_Multi.py` / `Transformer_Model.py` based on `params['model_type']` and `params['data_conf']['parallel_stim']`. `use_manual_features` (off by default; added recently) enables a second input tensor of hand-engineered features (e.g. AP count from `efel`) concatenated after the CNN flatten.
5. Saves: `blank_model.pth` (whole `nn.Module`) and `checkpoints/ckpt.pth` (state dict + optimizer). `sum_train.yaml` is the per-run metadata bundle written alongside — every downstream script (`predict.py`, `predictExp.py`, `unitParamConvert*.py`, `upload_to_hf.py`) consumes this.

### Data loading (`toolbox/Dataloader_H5.py`)
Reads the whole `<cell>.mlPack1.h5` shard into RAM once per rank (no distributed sampler). Stim/probe selection happens via `stims_select` / `probs_select` index lists. `serialize_stims: True` treats each stim as an independent sample (multiplies dataset size); `append_stim: True` concatenates along the time axis; `parallel_stim: True` stacks along a new axis (requires `Model_Multi`). `Dataloader_multiH5.py` is the variant that reads multiple H5s for ONTRA runs.

### Ray Tune (`RayTune.py`)
Activated when `do_ray: True` in the design YAML. The search space overwrites `params['model'][...]` with `tune.choice(...)` distributions covering CNN depth/filters/kernels/pools, FC depth/dims, dropFrac, batch size, and optimizer LR. ASHA scheduler + Optuna search. Trials below `--minLoss-RayTune` have their `params['model']` block dumped to `--minLoss-RayTune-yamlPath/<loss>_<jobid>.yaml` — these YAMLs can then be fed back into a normal `train_dist.py` run. `batchShifter_ray.slr` runs `yaml_check.sh` in parallel to launch full-data training jobs for any such promoted YAML it finds.

### Inference + parameter conversion
- `predict.py` — inference on held-out test split of the ML pack; writes `the_data.npz`, `sum_pred_<proj><cell>.yaml`, optional `MLoutput.h5` (true + predicted + input voltages), and plots via `toolbox/Plotter.py`.
- `predictExp.py` / `predictExpIDX.py` — inference on experimental voltage directories; dumps unit parameters as CSV.
- `toolbox/unitParamConvert.py` (CSV) and `toolbox/unitParamConvertHdf5.py` (H5) — consume `sum_train.yaml['input_meta']['phys_par_range']` to invert the `unit → log-scaled conductance` mapping and emit physical parameters (`S/cm²`, `μF/cm²`, `mV`). Without running these, predictions are only in normalized "unit" space.

### Output parameter naming
19 biophysical parameters for cADpyr excitatory cells. Canonical order is in the `parName` field inside `sum_train.yaml['input_meta']` (and in `packBBP3/aa.yaml`). `predictExp.py` has this list inlined as `param_names`. `toolbox/BiophysicalMeaningExcParams.csv` maps each parameter to a human-readable meaning.

## Design conventions worth knowing

- **`<design>` flag drives everything**: `--design m8lay_vs3` loads `m8lay_vs3.hpar.yaml` — never hard-code architecture changes; add a new `<design>.hpar.yaml` instead, and the SLURM scripts will automatically copy it alongside the run via `codeList` in the slr files.
- **SLURM scripts copy code into `$wrkDir`** (`cp -rp $codeList $wrkDir; cd $wrkDir`) so `srun` operates on a frozen snapshot of the tree. If a new Python file must be available at runtime, add it to `codeList` inside the relevant `.slr` file.
- **`hpar.yaml` data paths are keyed per facility** (`perlmutter:`, `corigpu:`, sometimes `summit:`). `train_dist.py` picks the right one using `--facility`. Don't collapse them.
- **Cell name format**: `<layer>_<type><e-type><clone>` e.g. `L5_TTPC1cADpyr0`, `L23_BPcAC0`. Special sentinels: `ALL_CELLS`, `ALL_CELLS_Inhibitory`, `ALL_CELLS_interpolated`, `AllCellsTestOnly` — used for ONTRA runs over pooled populations.
- **Two backbones**: `toolbox/Model.py` (1D-CNN→FC, the common case), `toolbox/Model_Multi.py` + `Model_Multi_Stim.py` (parallel-stim variants), `toolbox/Transformer_Model.py` (selected via `model_type: Transformers`). `Model2d.py` is legacy.
- **`.gitignore` is aggressive**: `*yaml`, `*h5`, `*csv`, `*txt`, `logs/*`, `out/`, `L*` are all ignored. New design YAMLs must be `git add -f`'d or the ignore pattern relaxed.
