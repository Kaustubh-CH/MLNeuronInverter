#!/usr/bin/env python3
"""
Side-by-side ion-channel comparison figure (truth-vs-pred 2D density per channel),
built from SAVED predictions only -- no inference needed.

Layout (matches CNNPaper Figure6 style):
  - The single column of ion channels is split into two blocks placed side by side.
  - Block A (left)  = axonal channels, Block B (right) = all other sections.
  - Each channel row shows 3 sub-panels for domain comparison:
        col 1 = new cell      (test / held-out test split)
        col 2 = new clone     (ALL_CELLS_interpolated  -> predictionlOntraNewClone)
        col 3 = new cell extr (AllCellsTestOnly         -> predictionlOntraNewCell)

Output: a PDF sized exactly 3.5 in wide x 5.5 in tall.

Usage:
  ./plot_ion_channels_compare.py --modelDir <run_dir>
"""
import os, csv, argparse
import numpy as np
import h5py, yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MAXU = 1.1
NBIN = 30


def section_cmap(section):
    violet = LinearSegmentedColormap.from_list(
        'violet', [(1, 1, 1), (101/255, 45/255, 144/255)], N=256)
    blue = LinearSegmentedColormap.from_list(
        'blue', [(1, 1, 1), (0, 173/255, 238/255)], N=256)
    green = LinearSegmentedColormap.from_list(
        'green', [(1, 1, 1), (0, 165/255, 81/255)], N=256)
    return {'apical': violet, 'axonal': blue, 'somatic': green,
            'dend': violet, 'all': 'Greys'}.get(section, 'Greys')


def load_domain(h5path, ncol_expect):
    with h5py.File(h5path, 'r') as f:
        U = f['ground_truth_upar'][:]   # truth
        Z = f['predict_upar'][:]        # predicted
    assert U.shape[1] == ncol_expect, (h5path, U.shape)
    return U, Z


def panel(ax, u, z, cmap):
    bins = np.linspace(-MAXU, MAXU, NBIN)
    ax.hist2d(z, u, bins=bins, cmin=1, cmap=cmap)
    ax.plot([0, 1], [0, 1], color='magenta', linestyle='--',
            linewidth=0.4, transform=ax.transAxes)
    ax.set_xlim(-MAXU, MAXU)
    ax.set_ylim(-MAXU, MAXU)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', length=0, labelbottom=False, labelleft=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.4)
    ax.spines['bottom'].set_linewidth(0.4)
    mse = float(np.mean((u - z) ** 2))
    ax.text(0.5, 0.02, '%.3f' % mse, transform=ax.transAxes,
            ha='center', va='bottom', size=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modelDir', default='.', help='run dir containing out/sum_train.yaml')
    ap.add_argument('--outName', default='ion_channels_compare.pdf')
    args = ap.parse_args()

    md = yaml.safe_load(open(os.path.join(args.modelDir, 'out/sum_train.yaml')))['input_meta']
    parName = md['parName']
    include = md['include']
    predNames = [parName[i] for i in include]          # 15 names, model output order
    sections = [n.split('_')[-1] for n in predNames]
    nPar = len(predNames)

    bio = {}
    with open(os.path.join(args.modelDir, 'toolbox/BiophysicalMeaningExcParams.csv')) as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                bio[row[0]] = row[1]
    disp = [bio.get(n, n) for n in predNames]

    # 3 domains, column order = test, interpolated, extrapolated
    domains = [
        ('test',   os.path.join(args.modelDir, 'predictionTest/MLoutput.h5')),
        ('interp', os.path.join(args.modelDir, 'predictionlOntraNewClone/MLoutput.h5')),
        ('extrap', os.path.join(args.modelDir, 'predictionlOntraNewCell/MLoutput.h5')),
    ]
    domHdr = ['new\ncell', 'new\nclone', 'extra\npolate']
    data = [load_domain(p, nPar) for _, p in domains]

    # split: block A = axonal, block B = the rest (original order preserved)
    blockA = [i for i in range(nPar) if sections[i] == 'axonal']
    blockB = [i for i in range(nPar) if sections[i] != 'axonal']
    nRow = max(len(blockA), len(blockB))

    fig = plt.figure(figsize=(3.5, 5.5), facecolor='white')
    # 7 cols: [A0 A1 A2 | gap | B0 B1 B2]; +1 header row at top
    gs = fig.add_gridspec(
        nRow + 1, 7,
        width_ratios=[1, 1, 1, 0.35, 1, 1, 1],
        height_ratios=[0.35] + [1] * nRow,
        wspace=0.18, hspace=0.20,
        left=0.02, right=0.98, top=0.97, bottom=0.02)

    blocks = [(blockA, 0), (blockB, 4)]   # (channel list, starting column)
    for chans, c0 in blocks:
        # column headers
        for d in range(3):
            hax = fig.add_subplot(gs[0, c0 + d])
            hax.axis('off')
            hax.text(0.5, 0.0, domHdr[d], ha='center', va='bottom', size=4.5)
        for r, ch in enumerate(chans):
            cmap = section_cmap(sections[ch])
            for d in range(3):
                ax = fig.add_subplot(gs[r + 1, c0 + d])
                panel(ax, data[d][0][:, ch], data[d][1][:, ch], cmap)
                # channel name centered over the middle (2nd) sub-panel
                if d == 1:
                    ax.set_title(disp[ch], size=6, pad=1)

    out = os.path.join(args.modelDir, 'out', args.outName)
    fig.savefig(out, format='pdf')                 # no bbox_inches -> keeps exact 3.5x5.5
    png = out[:-4] + '.png'
    fig.savefig(png, dpi=300)
    w, h = fig.get_size_inches()
    print('saved %s  size=%.2f x %.2f in' % (out, w, h))
    print('saved %s' % png)


if __name__ == '__main__':
    main()
