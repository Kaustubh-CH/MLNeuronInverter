#!/usr/bin/env python
"""
predict_from_hf.py
──────────────────
Download a neuroninverter model from Hugging Face and run inference.

Usage
─────
  python predict_from_hf.py \
      --repo    <hf-username>/neuron-L23_PCcADpyr2-m16lay \
      --input   /path/to/voltage_traces.h5 \
      --output  predictions.npy

  # or pass a token if the repo is private:
      --token   <your-hf-read-token>
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path


# ──────────────────────────────────────────────
# Download artefacts from HF Hub
# ──────────────────────────────────────────────

def load_model_from_hf(repo_id: str, token: str = None,
                       cache_dir: str = None,
                       subfolder: str = None) -> torch.nn.Module:
    """
    Download blank_model.pth + checkpoints/ckpt.pth from `repo_id`,
    reconstruct the model, load the best-epoch weights, and return it
    in eval mode on the best available device.

    `subfolder` selects a per-cell folder inside a mono-repo (e.g.
    'L5_TTPC1cADpyr0'); leave None for repos that store artefacts at root.
    """
    from huggingface_hub import hf_hub_download

    where = f"{repo_id}" + (f"/{subfolder}" if subfolder else "")
    print(f"Downloading model from: https://huggingface.co/{where}")

    blank_path = hf_hub_download(
        repo_id, filename='blank_model.pth', subfolder=subfolder,
        token=token, cache_dir=cache_dir,
    )
    ckpt_path = hf_hub_download(
        repo_id, filename='checkpoints/ckpt.pth', subfolder=subfolder,
        token=token, cache_dir=cache_dir,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    # Load full model object (architecture included)
    model = torch.load(blank_path, map_location=device)

    # Load best-epoch weights
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model_state']

    # Handle DDP prefix if present
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', '', 1): v
                      for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    ep = ckpt.get('epoch', '?')
    it = ckpt.get('iters', '?')
    print(f"  ✓ Loaded weights  (epoch={ep}, iters={it})")
    return model, device


# ──────────────────────────────────────────────
# Optional: load voltage traces from HDF5
# (matches the format produced by the BBP sim pipeline)
# ──────────────────────────────────────────────

def load_traces_from_h5(h5_path: str, probs_select=None):
    """
    Load voltage traces from an H5 file.
    Returns a float32 numpy array shape (N, time_bins, n_channels).
    Adjust the dataset key below to match your actual H5 layout.
    """
    import h5py
    with h5py.File(h5_path, 'r') as f:
        # Typical key used by the neuroninverter data pipeline
        data = f['voltage'][:]          # shape: (N, time_bins, n_channels)
    if probs_select is not None:
        data = data[:, :, probs_select]
    return data.astype(np.float32)


# ──────────────────────────────────────────────
# Prediction helper
# ──────────────────────────────────────────────

def predict(model: torch.nn.Module, traces: np.ndarray,
            device: torch.device, batch_size: int = 64) -> np.ndarray:
    """
    Run the model over `traces` in mini-batches.
    traces : np.ndarray  shape (N, time_bins, n_channels)
    returns: np.ndarray  shape (N, n_ion_channels)
    """
    results = []
    n = len(traces)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = torch.from_numpy(traces[start: start + batch_size]).to(device)
            out   = model(batch)
            results.append(out.cpu().numpy())
    return np.concatenate(results, axis=0)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def get_parser():
    p = argparse.ArgumentParser(
        description='Load a neuroninverter model from HF Hub and predict')
    p.add_argument('--repo',       required=True,
                   help='HF repo id, e.g. myuser/neuron-L23_PCcADpyr2-m16lay')
    p.add_argument('--input',      default=None,
                   help='Path to voltage-trace HDF5 file (optional for quick test)')
    p.add_argument('--output',     default='predictions.npy',
                   help='Where to save the prediction array (.npy)')
    p.add_argument('--token',      default=None,
                   help='HF read token (fallback: $HF_TOKEN env var)')
    p.add_argument('--cache_dir',  default=None,
                   help='Local directory to cache downloaded HF files')
    p.add_argument('--subfolder',  default=None,
                   help='Per-cell subfolder inside a mono-repo, e.g. L5_TTPC1cADpyr0')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--probs',      type=int, nargs='+', default=None,
                   help='Probe indices to select from the H5 file')
    return p


def main():
    args  = get_parser().parse_args()
    token = args.token or os.environ.get('HF_TOKEN')

    model, device = load_model_from_hf(
        args.repo, token=token, cache_dir=args.cache_dir,
        subfolder=args.subfolder)

    if args.input:
        print(f"Loading traces from: {args.input}")
        traces = load_traces_from_h5(args.input, probs_select=args.probs)
        print(f"  traces shape: {traces.shape}")

        preds = predict(model, traces, device, batch_size=args.batch_size)
        print(f"  predictions shape: {preds.shape}")

        np.save(args.output, preds)
        print(f"✓ Predictions saved → {args.output}")
    else:
        # Quick sanity check with random input using config.json
        from huggingface_hub import hf_hub_download
        import json
        cfg_path = hf_hub_download(args.repo, 'config.json',
                                   subfolder=args.subfolder,
                                   token=token, cache_dir=args.cache_dir)
        with open(cfg_path) as fh:
            cfg = json.load(fh)

        shape = cfg.get('model', {}).get('inputShape')
        if shape:
            dummy = torch.randn(4, *shape).to(device)
            with torch.no_grad():
                out = model(dummy)
            print(f"  Dummy forward pass: input {tuple(dummy.shape)} → output {tuple(out.shape)}")
            print("✓ Model works correctly")
        else:
            print("No input shape found in config.json — pass --input to run inference")


if __name__ == '__main__':
    main()
