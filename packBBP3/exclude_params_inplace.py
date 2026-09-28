#!/usr/bin/env python3
"""In-place column drop on already-packed mlPack1 H5 files.

Trims `*_unit_par` and `*_phys_par` to remove a set of named ion-channel
parameters and rewrites `meta.JSON` so that `include` / `num_phys_par`
match the new column count.  `parName`, `phys_par_range`, and
`base_values` are left at their canonical 19-entry length (the
project's convention: `include` indexes into the canonical list and
length(include) == data column count).

Idempotent: if the file is already trimmed to the requested set, it is
left untouched.

Usage:
    python3 exclude_params_inplace.py <h5_path> [<h5_path> ...]
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np


EXCLUDE_NAMES = [
    'gK_Tstbar_K_Tst_axonal',
    'g_pas_axonal',
    'gCa_LVAstbar_Ca_LVAst_somatic',
    'g_pas_somatic',
]

DOMAINS = ['train', 'valid', 'test']
PAR_DSETS = ['unit_par', 'phys_par']


def trim_h5(path: str, dry_run: bool = False) -> None:
    print(f'\n== {path} ==')
    if not os.path.exists(path):
        print('  SKIP: missing file')
        return

    mode = 'r' if dry_run else 'r+'
    with h5py.File(path, mode) as f:
        if 'meta.JSON' not in f:
            print('  SKIP: no meta.JSON')
            return
        meta = json.loads(f['meta.JSON'][0])
        par_name = list(meta['parName'])
        canonical_n = len(par_name)

        try:
            exclude_idx = sorted(par_name.index(n) for n in EXCLUDE_NAMES)
        except ValueError as e:
            print(f'  SKIP: parName missing entry ({e})')
            return
        keep_idx = [i for i in range(canonical_n) if i not in exclude_idx]
        n_keep = len(keep_idx)

        first_dset = f.get(f'{DOMAINS[0]}_{PAR_DSETS[0]}')
        if first_dset is None:
            print('  SKIP: no train_unit_par dataset')
            return
        cur_cols = first_dset.shape[1]

        cur_include = list(meta.get('include', list(range(canonical_n))))

        if cur_cols == n_keep and cur_include == keep_idx:
            print(f'  already trimmed: cols={cur_cols} include={cur_include}')
            return

        if cur_cols != canonical_n:
            print(f'  WARN: unexpected column count {cur_cols} '
                  f'(expected canonical {canonical_n}); aborting this file')
            return

        print(f'  exclude indices: {exclude_idx}')
        print(f'  keeping {n_keep} of {canonical_n} cols')
        print(f'  new include: {keep_idx}')

        if dry_run:
            return

        for dom in DOMAINS:
            for pname in PAR_DSETS:
                key = f'{dom}_{pname}'
                if key not in f:
                    continue
                old = f[key]
                old_shape = old.shape
                old_dtype = old.dtype
                # Read full, slice columns
                data = old[:, keep_idx]
                # h5py can't resize an existing dataset's dim that wasn't
                # created with maxshape=None, so delete + recreate.
                del f[key]
                f.create_dataset(key, data=data, dtype=old_dtype,
                                 chunks=True)
                print(f'  rewrote {key}: {old_shape} -> {data.shape}')

        meta['include'] = keep_idx
        meta['num_phys_par'] = n_keep
        # linearParIdx (canonical-19 indices) — drop any that got excluded
        if 'linearParIdx' in meta:
            meta['linearParIdx'] = [i for i in meta['linearParIdx']
                                    if i not in exclude_idx]

        # meta.JSON is a (1,) dataset of variable-length string; replace it
        del f['meta.JSON']
        meta_str = json.dumps(meta)
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('meta.JSON', data=np.array([meta_str], dtype=object),
                         dtype=dt)
        print(f"  meta.JSON updated (include len={n_keep}, num_phys_par={n_keep})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('paths', nargs='+', help='mlPack1.h5 file paths')
    p.add_argument('--dry-run', action='store_true',
                   help='Report what would change but do not modify files')
    args = p.parse_args()
    for path in args.paths:
        trim_h5(path, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
