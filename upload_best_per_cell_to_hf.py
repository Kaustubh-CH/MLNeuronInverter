#!/usr/bin/env python
"""
upload_best_per_cell_to_hf.py
─────────────────────────────
Upload the *best* trained neuroninverter model per cell to a single PRIVATE
Hugging Face mono-repo, one `<cell>/` subfolder per model.

Why this exists (vs upload_to_hf.py):
  * The real on-disk layout is three levels deep:
        <scratch_root>/<cell>/<runID>/out/{blank_model.pth,checkpoints/ckpt.pth,sum_train.yaml}
    upload_to_hf.py --scan_dir only looks one level down, so it finds nothing.
  * There are many runs per cell. We keep only the run with the lowest
    `loss_valid` from sum_train.yaml.
  * Checkpoints are slimmed to inference-only (optimizer_state_dict dropped),
    cutting each ckpt.pth from ~140 MB to ~25 MB.
  * Everything lands in ONE private repo as <cell>/… subfolders, so you can
    pull any cell from anywhere (see predict_from_hf.py --subfolder).

This script imports no jaxley/jax, so it runs under EITHER runtime.

  # (a) shifter pytorch image (has huggingface_hub 0.16.4):
  export HF_TOKEN=<your hf write token>
  shifter --image=nersc/pytorch:ngc-21.08-v2 \
      python upload_best_per_cell_to_hf.py --hf_user ktub1999 --dry_run

  # (b) or the conda env (huggingface_hub 1.11.0):
  module load conda && conda activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
  python upload_best_per_cell_to_hf.py --hf_user ktub1999 --dry_run

Examples (prefix with `shifter --image=nersc/pytorch:ngc-21.08-v2` to use the image):
  # See what would be uploaded, no network:
  python upload_best_per_cell_to_hf.py --hf_user ktub1999 --dry_run

  # Upload just one cell to smoke-test the full path:
  python upload_best_per_cell_to_hf.py --hf_user ktub1999 --only L5_TTPC1cADpyr0

  # Upload best-per-cell for every cell:
  python upload_best_per_cell_to_hf.py --hf_user ktub1999
"""

import os
import sys
import json
import shutil
import argparse
import tempfile
from pathlib import Path

import yaml
import torch

DEFAULT_SCRATCH_ROOT = '/pscratch/sd/k/ktub1999/tmp_neuInv/bbp3'


# ──────────────────────────────────────────────
# Discovery: best complete run per cell
# ──────────────────────────────────────────────

def load_summary(out_dir: Path) -> dict:
    yaml_path = out_dir / 'sum_train.yaml'
    if not yaml_path.exists():
        return {}
    with open(yaml_path) as fh:
        return yaml.safe_load(fh) or {}


def is_complete(out_dir: Path) -> bool:
    return (
        (out_dir / 'blank_model.pth').exists()
        and (out_dir / 'checkpoints' / 'ckpt.pth').exists()
        and (out_dir / 'sum_train.yaml').exists()
    )


def find_best_per_cell(scratch_root: Path, only=None):
    """
    Return {cell_name: {'out_dir': Path, 'run_id': str, 'loss_valid': float,
                        'summary': dict}} keeping the lowest-loss complete run.
    """
    best = {}
    skipped_cells = []
    for cell_dir in sorted(p for p in scratch_root.iterdir() if p.is_dir()):
        cell = cell_dir.name
        if only and cell not in only:
            continue

        rankable = []
        for run_dir in sorted(p for p in cell_dir.iterdir() if p.is_dir()):
            out_dir = run_dir / 'out'
            if not is_complete(out_dir):
                continue
            summary = load_summary(out_dir)
            loss = summary.get('loss_valid')
            if loss is None:
                continue
            rankable.append((float(loss), run_dir.name, out_dir, summary))

        if not rankable:
            skipped_cells.append(cell)
            continue

        rankable.sort(key=lambda t: t[0])
        loss, run_id, out_dir, summary = rankable[0]
        best[cell] = {
            'out_dir': out_dir, 'run_id': run_id,
            'loss_valid': loss, 'summary': summary,
            'n_candidates': len(rankable),
        }
    return best, skipped_cells


# ──────────────────────────────────────────────
# Checkpoint slimming (inference-only)
# ──────────────────────────────────────────────

def slim_checkpoint(src_ckpt: Path, dst_ckpt: Path):
    """Keep model_state (+epoch/iters); drop optimizer_state_dict."""
    ckpt = torch.load(str(src_ckpt), map_location='cpu')
    slim = {'model_state': ckpt['model_state']}
    for k in ('epoch', 'iters'):
        if k in ckpt:
            slim[k] = ckpt[k]
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, str(dst_ckpt))


# ──────────────────────────────────────────────
# Per-cell config.json + model card
# ──────────────────────────────────────────────

def build_config(summary: dict) -> dict:
    tp = summary.get('train_params', {})
    keys = ['design', 'cell_name', 'max_epochs', 'model',
            'train_conf', 'data_conf']
    config = {k: tp[k] for k in keys if k in tp}
    config['val_loss'] = summary.get('loss_valid')
    config['num_ranks'] = summary.get('num_ranks')
    return config


CELL_CARD = """\
# {cell} — neuroninverter model

Best run for `{cell}` (lowest `loss_valid` of {n} candidate run(s)).

| Field | Value |
|-------|-------|
| Cell | `{cell}` |
| Design | `{design}` |
| Run id | `{run_id}` |
| Validation loss | `{loss}` |
| Epochs | `{epochs}` |

## Load this model

```python
import torch
from huggingface_hub import hf_hub_download

REPO, SUB = "{repo_id}", "{cell}"
blank = hf_hub_download(REPO, "blank_model.pth", subfolder=SUB)
ckpt  = hf_hub_download(REPO, "checkpoints/ckpt.pth", subfolder=SUB)

model = torch.load(blank, map_location="cpu")
sd = torch.load(ckpt, map_location="cpu")["model_state"]
sd = {{k.replace("module.", "", 1): v for k, v in sd.items()}}
model.load_state_dict(sd); model.eval()
```

Or with the repo helper:
`python predict_from_hf.py --repo {repo_id} --subfolder {cell} --input traces.h5`
"""


def write_cell_extras(staging: Path, cell: str, info: dict, repo_id: str):
    summary = info['summary']
    tp = summary.get('train_params', {})
    with open(staging / 'config.json', 'w') as fh:
        json.dump(build_config(summary), fh, indent=2, default=str)
    card = CELL_CARD.format(
        cell=cell, repo_id=repo_id, run_id=info['run_id'],
        design=tp.get('design', 'unknown'),
        loss=info['loss_valid'], epochs=tp.get('max_epochs', 'N/A'),
        n=info['n_candidates'],
    )
    with open(staging / 'README.md', 'w') as fh:
        fh.write(card)


def root_readme(repo_id: str, best: dict) -> str:
    rows = "\n".join(
        f"| `{cell}` | `{info['run_id']}` | `{info['loss_valid']:.5f}` |"
        for cell, info in sorted(best.items())
    )
    return f"""\
---
license: mit
tags: [neuroscience, neuroninverter, pytorch, ion-channel]
---

# neuroninverter models (best per cell)

Private collection of trained neuroninverter CNNs — one `<cell>/` subfolder per
cell, each the lowest-`loss_valid` run. Each subfolder holds `blank_model.pth`,
`checkpoints/ckpt.pth` (inference-only / slimmed), `sum_train.yaml`, `config.json`.

```python
python predict_from_hf.py --repo {repo_id} --subfolder <cell> --input traces.h5
```

| Cell | Run id | loss_valid |
|------|--------|-----------|
{rows}
"""


# ──────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────

def upload_all(best: dict, repo_id: str, token: str, private: bool):
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(repo_id, token=token, private=private, exist_ok=True,
                repo_type='model')
    print(f"✓ Repo ready: https://huggingface.co/{repo_id} (private={private})")

    uploaded = []
    for i, (cell, info) in enumerate(sorted(best.items()), 1):
        out_dir = info['out_dir']
        print(f"\n[{i}/{len(best)}] {cell}  (run {info['run_id']}, "
              f"loss_valid={info['loss_valid']:.5f})")
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            shutil.copy2(out_dir / 'blank_model.pth', staging / 'blank_model.pth')
            print("    slimming checkpoint …")
            slim_checkpoint(out_dir / 'checkpoints' / 'ckpt.pth',
                            staging / 'checkpoints' / 'ckpt.pth')
            shutil.copy2(out_dir / 'sum_train.yaml', staging / 'sum_train.yaml')
            write_cell_extras(staging, cell, info, repo_id)

            print(f"    uploading → {repo_id}/{cell} …")
            api.upload_folder(
                folder_path=str(staging),
                path_in_repo=cell,
                repo_id=repo_id,
                token=token,
                commit_message=f"Add best model for {cell} (run {info['run_id']})",
            )
        uploaded.append(cell)
        print(f"    ✓ done")

    # Top-level index README
    print("\nWriting top-level README …")
    api.upload_file(
        path_or_fileobj=root_readme(repo_id, best).encode(),
        path_in_repo='README.md',
        repo_id=repo_id,
        token=token,
        commit_message='Update model index',
    )
    return uploaded


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def get_parser():
    p = argparse.ArgumentParser(
        description='Upload best-per-cell neuroninverter models to a private HF mono-repo')
    p.add_argument('--scratch_root', default=DEFAULT_SCRATCH_ROOT,
                   help=f'Parent dir of <cell>/<runID>/out/ (default: {DEFAULT_SCRATCH_ROOT})')
    p.add_argument('--hf_user', required=True,
                   help='Your Hugging Face username / organisation')
    p.add_argument('--repo_name', default='neuroninverter-models',
                   help='Repo name; repo_id = <hf_user>/<repo_name>')
    p.add_argument('--token', default=None,
                   help='HF write token (fallback: $HF_TOKEN env var)')
    p.add_argument('--public', action='store_true', default=False,
                   help='Make the repo public (default: private)')
    p.add_argument('--only', nargs='+', default=None,
                   help='Restrict to these cell names (e.g. L5_TTPC1cADpyr0)')
    p.add_argument('--dry_run', action='store_true', default=False,
                   help='Print the best-per-cell selection and exit (no upload)')
    return p


def main():
    args = get_parser().parse_args()
    scratch_root = Path(args.scratch_root)
    if not scratch_root.is_dir():
        print(f"ERROR: scratch_root not found: {scratch_root}")
        sys.exit(1)

    only = set(args.only) if args.only else None
    best, skipped = find_best_per_cell(scratch_root, only=only)

    print(f"Selected {len(best)} cell(s); {len(skipped)} cell(s) had no "
          f"rankable complete run.")
    for cell, info in sorted(best.items()):
        print(f"  {cell:32s} -> run {info['run_id']:>12s}  "
              f"loss_valid={info['loss_valid']:.5f}  "
              f"({info['n_candidates']} candidate(s))")
    if skipped:
        print("\nSkipped (no complete run with loss_valid):")
        print("  " + ", ".join(skipped))

    if args.dry_run:
        print("\n[dry_run] nothing uploaded.")
        return
    if not best:
        print("\nNothing to upload.")
        return

    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print("ERROR: provide --token or set the HF_TOKEN environment variable")
        sys.exit(1)

    repo_id = f"{args.hf_user}/{args.repo_name}"
    uploaded = upload_all(best, repo_id, token, private=not args.public)
    print(f"\n─── Done: uploaded {len(uploaded)} cell(s) to "
          f"https://huggingface.co/{repo_id} ───")


if __name__ == '__main__':
    main()
