#!/usr/bin/env python3
"""Stimulus catalog (2026-09-24): scan the DL4neurons2 stim CSVs that JaxleyBridge loads
(one current value per line, nA, dt 0.1 ms), measure each, group them into families, and record
which design yamls in this repo use them.

  python scripts/stim_catalog.py                     # table -> docs/stims/stim_table.csv (login OK)
  srun -n1 python scripts/stim_catalog.py --gallery  # + docs/stims/stim_gallery.pdf (one page per family)

Flags: PA_AS_NA = max|I| > 20 nA (the 1000x unit bug: pA values loaded as nA; load_stim_csv has no
guard, see memory project_ca3_stim_pA_unit_bug).
"""
import argparse, csv, glob, os, re
from collections import defaultdict
import numpy as np

STIM_DIR = "/pscratch/sd/k/ktub1999/main/DL4neurons2/stims"
DT_MS = 0.1
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "stims")


def family(stem):
    """Strip amplitude digits and grid/unit suffixes -> a family key."""
    s = re.sub(r"_(i4k|5k|50khz|icaRec|icav2c?|ica)\b", "", stem)
    s = re.sub(r"^(\d+k)(\d*k)?", "", s)        # 5k / 4k50k / 5k50k grid prefixes
    s = re.sub(r"\d+(\.\d+)?", "#", s)
    return s.strip("_") or stem


def used_by():
    """stem -> designs whose yaml names it under any key containing "stim" (stim_name, pooled_stim_names,
    stim_names, stim_names_multi, stim_from_label.stim_names, ...)."""
    import yaml
    hits = defaultdict(set)

    def walk(node, design, under_stim=False):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, design, under_stim or "stim" in str(k).lower())
        elif isinstance(node, list):
            for v in node:
                walk(v, design, under_stim)
        elif under_stim and isinstance(node, str):
            hits[node].add(design)

    for y in glob.glob(os.path.join(REPO, "*.hpar.yaml")):
        try:
            walk(yaml.safe_load(open(y)), os.path.basename(y).replace(".hpar.yaml", ""))
        except Exception:
            pass
    return hits


def load(path):
    try:
        return np.loadtxt(path, dtype=np.float64).ravel()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimDir", default=STIM_DIR)
    ap.add_argument("--gallery", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    hits = used_by()
    rows, waves = [], {}
    for p in sorted(glob.glob(os.path.join(a.stimDir, "*.csv"))):
        stem = os.path.basename(p)[:-4]
        I = load(p)
        if I is None or I.size < 10:
            rows.append(dict(stem=stem, family=family(stem), n=0, t_ms="", min_nA="", max_nA="",
                             mean_nA="", std_nA="", end_nA="", flag="UNREADABLE", used_by=""))
            continue
        waves[stem] = I
        flag = "PA_AS_NA" if np.abs(I).max() > 20 else ""
        users = sorted(hits.get(stem, set()))
        rows.append(dict(stem=stem, family=family(stem), n=I.size, t_ms=f"{I.size * DT_MS:.0f}",
                         min_nA=f"{I.min():.3f}", max_nA=f"{I.max():.3f}", mean_nA=f"{I.mean():.3f}",
                         std_nA=f"{I.std():.3f}", end_nA=f"{I[-1]:.3f}", flag=flag,
                         used_by=f"{len(users)}: " + ",".join(users[:4]) + (" ..." if len(users) > 4 else "")
                         if users else ""))
    rows.sort(key=lambda r: (r["family"], r["stem"]))
    with open(os.path.join(OUT, "stim_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} stims, {len({r['family'] for r in rows})} families, "
          f"{sum(r['flag'] == 'PA_AS_NA' for r in rows)} PA_AS_NA -> {OUT}/stim_table.csv")

    if a.gallery:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        fams = defaultdict(list)
        for r in rows:
            if r["stem"] in waves:
                fams[r["family"]].append(r)
        with PdfPages(os.path.join(OUT, "stim_gallery.pdf")) as pdf:
            for fam, rs in sorted(fams.items()):
                fig, ax = plt.subplots(len(rs), 1, figsize=(10, 1.6 * len(rs) + 0.8), squeeze=False)
                for k, r in enumerate(rs):
                    I = waves[r["stem"]]
                    ax[k][0].plot(np.arange(I.size) * DT_MS, I, lw=0.5,
                                  color="C3" if r["flag"] else "C0")
                    ax[k][0].set_title(f"{r['stem']}  n={r['n']}  [{r['min_nA']}, {r['max_nA']}] nA  "
                                       f"{r['flag']}", fontsize=8, loc="left")
                    ax[k][0].tick_params(labelsize=7); ax[k][0].set_ylabel("nA", fontsize=7)
                ax[-1][0].set_xlabel("ms (dt 0.1)", fontsize=8)
                fig.suptitle(f"family: {fam}", fontsize=10)
                fig.tight_layout(rect=[0, 0, 1, 0.97]); pdf.savefig(fig); plt.close(fig)
        print(f"wrote {OUT}/stim_gallery.pdf ({len(fams)} pages)")


if __name__ == "__main__":
    main()
