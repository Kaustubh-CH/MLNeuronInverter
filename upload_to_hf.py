#!/usr/bin/env python
"""
upload_to_hf.py
───────────────
Upload a trained neuroninverter model to Hugging Face Hub so it can be
downloaded and used for inference later.

Saved artefacts expected under <model_dir>:
  blank_model.pth          – full torch.nn.Module (architecture + random weights)
  checkpoints/ckpt.pth     – {iters, epoch, model_state, optimizer_state_dict}
  sum_train.yaml           – training summary / metadata

Usage
─────
  python upload_to_hf.py \
      --model_dir /pscratch/sd/k/ktub1999/tmp_neuInv/TB_logs/<run_dir> \
      --hf_user   <your-hf-username> \
      --token     <your-hf-write-token>        # or set HF_TOKEN env var

Optional:
  --repo_name  override auto-generated repo name  (default: neuron-<cell>-<design>)
  --private    make the HF repo private
  --scan_dir   instead of a single --model_dir, scan this parent directory for
               all sub-folders that contain both blank_model.pth and sum_train.yaml
"""

import os
import sys
import argparse
import json
import shutil
import tempfile
import yaml
import torch
from pathlib import Path

# ──────────────────────────────────────────────
# Helper: load training summary
# ──────────────────────────────────────────────

def load_summary(model_dir: str) -> dict:
    yaml_path = os.path.join(model_dir, 'sum_train.yaml')
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path) as fh:
        return yaml.safe_load(fh) or {}


# ──────────────────────────────────────────────
# Helper: build a repo name from the summary
# ──────────────────────────────────────────────

def make_repo_name(summary: dict, hf_user: str, override: str = None) -> str:
    if override:
        return f"{hf_user}/{override}"
    tp = summary.get('train_params', {})
    cell   = tp.get('cell_name', 'unknown_cell').replace('/', '-')
    design = tp.get('design',    'unknown_design').replace('/', '-')
    return f"{hf_user}/neuron-{cell}-{design}"


# ──────────────────────────────────────────────
# Helper: generate a markdown model card
# ──────────────────────────────────────────────

MODEL_CARD_TEMPLATE = """\
---
license: mit
tags:
  - neuroscience
  - neuroninverter
  - pytorch
  - ion-channel
cell_name: {cell_name}
design: {design}
val_loss: {val_loss}
epochs: {epochs}
---

# Neuroninverter model — {cell_name}

A CNN trained with the **neuroninverter** framework to invert
neural simulations and predict biophysical ion-channel parameters
from membrane voltage traces.

## Training details

| Field | Value |
|-------|-------|
| Cell name | `{cell_name}` |
| Design / architecture | `{design}` |
| Final validation loss | `{val_loss}` |
| Epochs trained | `{epochs}` |
| Host | `{host}` |
| Num ranks (GPUs) | `{num_ranks}` |

## Files in this repository

| File | Description |
|------|-------------|
| `blank_model.pth` | Full `torch.nn.Module` saved with `torch.save(model, ...)` |
| `checkpoints/ckpt.pth` | Best-epoch checkpoint: `model_state`, `optimizer_state_dict`, `epoch`, `iters` |
| `sum_train.yaml` | Complete training summary / hyper-parameters |
| `config.json` | Key training params in JSON (for easy browsing) |

## How to load and predict

```python
from huggingface_hub import hf_hub_download
import torch

REPO = "{repo_id}"

# 1. Download artefacts
blank_path = hf_hub_download(REPO, "blank_model.pth")
ckpt_path  = hf_hub_download(REPO, "checkpoints/ckpt.pth")

# 2. Reconstruct model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = torch.load(blank_path, map_location=device)
ckpt   = torch.load(ckpt_path,  map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 3. Predict  (shape depends on your model's inputShape)
# voltage_trace: torch.Tensor  shape (batch, time_bins, n_channels)
with torch.no_grad():
    predictions = model(voltage_trace.to(device))
```
"""

def make_model_card(summary: dict, repo_id: str) -> str:
    tp       = summary.get('train_params', {})
    cell     = tp.get('cell_name', 'unknown')
    design   = tp.get('design',    'unknown')
    val_loss = summary.get('loss_valid', 'N/A')
    epochs   = tp.get('max_epochs', 'N/A')
    host     = summary.get('host_name', 'N/A')
    ranks    = summary.get('num_ranks', 1)
    return MODEL_CARD_TEMPLATE.format(
        cell_name=cell, design=design,
        val_loss=val_loss, epochs=epochs,
        host=host, num_ranks=ranks,
        repo_id=repo_id,
    )


# ──────────────────────────────────────────────
# Core upload function
# ──────────────────────────────────────────────

def upload_model(model_dir: str, hf_user: str, token: str,
                 repo_name_override: str = None, private: bool = False):
    from huggingface_hub import HfApi, create_repo

    model_dir = str(Path(model_dir).resolve())
    blank_path = os.path.join(model_dir, 'blank_model.pth')
    ckpt_path  = os.path.join(model_dir, 'checkpoints', 'ckpt.pth')
    yaml_path  = os.path.join(model_dir, 'sum_train.yaml')

    # ── Validate required files ──────────────────────────────────────────
    missing = [p for p in [blank_path, ckpt_path] if not os.path.exists(p)]
    if missing:
        print(f"  [SKIP] {model_dir} — missing: {missing}")
        return None

    summary  = load_summary(model_dir)
    repo_id  = make_repo_name(summary, hf_user, repo_name_override)
    api      = HfApi()

    # ── Create (or reuse) the repo ───────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Repo   : {repo_id}")
    print(f"  Source : {model_dir}")
    try:
        create_repo(repo_id, token=token, private=private, exist_ok=True)
        print(f"  ✓ Repo ready on Hugging Face")
    except Exception as e:
        print(f"  ✗ Could not create repo: {e}")
        return None

    # ── Stage files in a temp directory ──────────────────────────────────
    with tempfile.TemporaryDirectory() as staging:
        # blank model
        shutil.copy2(blank_path, os.path.join(staging, 'blank_model.pth'))

        # checkpoint (preserve sub-folder structure)
        ckpt_staging = os.path.join(staging, 'checkpoints')
        os.makedirs(ckpt_staging, exist_ok=True)
        shutil.copy2(ckpt_path, os.path.join(ckpt_staging, 'ckpt.pth'))

        # training summary YAML
        if os.path.exists(yaml_path):
            shutil.copy2(yaml_path, os.path.join(staging, 'sum_train.yaml'))

        # JSON config (human-readable subset)
        tp = summary.get('train_params', {})
        config_keys = ['design', 'cell_name', 'max_epochs', 'model',
                       'train_conf', 'data_conf']
        config = {k: tp[k] for k in config_keys if k in tp}
        config['val_loss']  = summary.get('loss_valid')
        config['num_ranks'] = summary.get('num_ranks')
        with open(os.path.join(staging, 'config.json'), 'w') as fh:
            json.dump(config, fh, indent=2, default=str)

        # model card
        readme = make_model_card(summary, repo_id)
        with open(os.path.join(staging, 'README.md'), 'w') as fh:
            fh.write(readme)

        # ── Upload all staged files ───────────────────────────────────────
        print(f"  Uploading files …")
        for root, dirs, files in os.walk(staging):
            for fname in files:
                local_path  = os.path.join(root, fname)
                remote_path = os.path.relpath(local_path, staging)
                try:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=remote_path,
                        repo_id=repo_id,
                        token=token,
                    )
                    print(f"    ✓ {remote_path}")
                except Exception as e:
                    print(f"    ✗ {remote_path} — {e}")

    print(f"  → https://huggingface.co/{repo_id}")
    return repo_id


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def get_parser():
    p = argparse.ArgumentParser(description='Upload neuroninverter model to Hugging Face')
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--model_dir',  type=str,
                     help='Path to a single trained model output directory')
    grp.add_argument('--scan_dir',   type=str,
                     help='Parent directory; scan for all subdirs that contain a trained model')
    p.add_argument('--hf_user',   type=str, required=True,
                   help='Your Hugging Face username / organisation')
    p.add_argument('--token',     type=str, default=None,
                   help='HF write token (fallback: $HF_TOKEN env var)')
    p.add_argument('--repo_name', type=str, default=None,
                   help='Override the auto-generated repo name (only valid with --model_dir)')
    p.add_argument('--private',   action='store_true', default=False,
                   help='Make the HF repo private')
    return p


def main():
    args   = get_parser().parse_args()
    token  = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print("ERROR: provide --token or set the HF_TOKEN environment variable")
        sys.exit(1)

    if args.model_dir:
        upload_model(args.model_dir, args.hf_user, token,
                     repo_name_override=args.repo_name,
                     private=args.private)
    else:
        # scan for all subdirs that look like trained model dirs
        parent = Path(args.scan_dir)
        candidates = [
            d for d in sorted(parent.iterdir())
            if d.is_dir()
            and (d / 'blank_model.pth').exists()
            and (d / 'checkpoints' / 'ckpt.pth').exists()
        ]
        print(f"Found {len(candidates)} model(s) under {parent}")
        for d in candidates:
            upload_model(str(d), args.hf_user, token, private=args.private)

    print("\n─── All done ───")


if __name__ == '__main__':
    main()
