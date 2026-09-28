#!/usr/bin/env python3
"""Convert the Roy* predicted unit parameters to physical parameters, in three
out-of-range treatments.

The conversion itself is the one in toolbox/unitParamConvert.py, re-derived here
so the sweep ordering is deterministic (sorted, not os.listdir order) and so the
three variants come out of a single pass.  It is verified exact against the
training data: reconstructing phys_par from unit_par in
  OntraExcRawFeb5thNoNoise/runs2/35504015_1/L4_SScADpyr2/*.h5
reproduces all 19 stored physical values to 1.3e-07 relative error.

  conductances (log)   P = base_i * 10**((lb+ub)/2) * 10**((ub-lb)/2 * u)
  cm_*, e_pas (linear) P = (lb+ub)/2 + (ub-lb)/2 * u

base_i is input_meta.base_values, bit-identical to
sensitivity_analysis/NewBase2/MeanParams0.csv.  It is a single base shared by all
46 training cells -- the parameterisation is cell-agnostic, which is why the same
base correctly reconstructs an L4_SS cell above.

Variants, for the 15 predicted parameters (the other 4 always take base):
  exact    u as predicted, no clipping
  minmax   u clipped to [-1, +1]  -- out-of-range pinned to the trained edge
  default  u out of [-1, +1] -> that parameter takes its base physical value

NOTE on `default`: unitParamConvert.py implements this as u := 0.  For 17 of the
19 parameters u=0 is exactly the base value, but cm_somatic and cm_axonal have an
asymmetric range [0.5, 2.0], so u=0 gives 1.25 uF/cm2 -- 1.25x their base of 1.0.
Here `default` assigns the base physical value directly, which is what "default"
means; set U0_SEMANTICS='zero' to reproduce the older u=0 behaviour instead.

Run: shifter --image=balewski/ubu20-neuron8:v5 python3 roy_unit_to_phys.py [modelDir] [dsetSuffix]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

ALL_CELLS = "/pscratch/sd/k/ktub1999/tmp_neuInv/bbp3/ALL_CELLS"
MODEL_DIR = (sys.argv[1] if len(sys.argv) > 1
             else os.path.join(ALL_CELLS, "probescan_exc_p0_56351363"))
DSET = sys.argv[2] if len(sys.argv) > 2 else ""   # e.g. "Blank350"
PRED_ROOT = os.path.join(MODEL_DIR, "predict_royPaula" + DSET)
YAML = os.path.join(MODEL_DIR, "out", "sum_train.yaml")
AMPS = [100, 500, 1000, 1500, 2000]
LINEAR = {"cm_somatic", "cm_axonal", "cm_all", "e_pas_all"}
VARIANTS = ["exact", "minmax", "default"]
U0_SEMANTICS = "base"          # 'base' (correct) or 'zero' (legacy u:=0)


def convert(u_row, par, rng, base, include, variant):
    """One predicted unit row (len 15) -> full physical row (len 19)."""
    phys = np.array(base, dtype=np.float64)     # non-included keep base
    for j, i in enumerate(include):
        u = float(u_row[j])
        lb, ub, _ = rng[i]
        oor = (u < -1) or (u > 1)
        if variant == "minmax":
            u = min(1.0, max(-1.0, u))
        elif variant == "default" and oor:
            if U0_SEMANTICS == "base":
                continue                        # leave phys[i] == base[i]
            u = 0.0
        if par[i] in LINEAR:
            phys[i] = (lb + ub) / 2 + (ub - lb) / 2 * u
        else:
            # log-space with a float64 guard: single-cell models can predict
            # |u| in the hundreds on out-of-distribution data, and the naive
            # product overflows.  The capped value is unsimulatable either
            # way -- 'exact' just has to report it finitely.
            e = np.log10(base[i]) + (lb + ub) / 2 + (ub - lb) / 2 * u
            phys[i] = 10.0 ** min(e, 300.0)
    return phys


def main():
    md = yaml.safe_load(open(YAML))["input_meta"]
    par, rng, include = md["parName"], md["phys_par_range"], md["include"]
    # some sum_train.yaml carry base values as strings ('8e-05'), coerce first
    base = [float(x) for x in md["base_values"]]
    names = [par[i] for i in include]

    print(f"model {os.path.basename(MODEL_DIR)}")
    print(f"{len(par)} parameters, {len(include)} predicted, "
          f"{len(par) - len(include)} held at base")
    print(f"default variant semantics: {U0_SEMANTICS}\n")

    print("%-9s %7s %8s %s" % ("amp", "sweeps", "n_oor", "out-of-range parameters"))
    for amp in AMPS:
        d = os.path.join(PRED_ROOT, f"Roy{amp}")
        files = sorted(glob.glob(os.path.join(d, "unitParam*.csv")))
        U = np.array([pd.read_csv(f)["unit_params_predict"].values for f in files])

        out = {v: np.array([convert(r, par, rng, base, include, v) for r in U])
               for v in VARIANTS}
        for v in VARIANTS:
            # named so BBP_sbatch_RoyParamFile.sh finds it as <prefix>1.csv
            np.savetxt(os.path.join(d, f"{v}Converted1.csv"), out[v])

        oor = np.abs(U) > 1
        hit = sorted({names[j] for j in np.where(oor.any(axis=0))[0]})
        print("%-9s %7d %8d %s" % (f"Roy{amp}", U.shape[0], oor.sum(),
                                   ", ".join(hit) if hit else "-"))

        # sanity: variants may only differ where a prediction went out of range
        for v in ("minmax", "default"):
            diff = ~np.isclose(out[v], out["exact"], rtol=1e-9)
            allowed = np.zeros_like(diff)
            allowed[:, include] = oor
            assert not (diff & ~allowed).any(), f"{v} changed an in-range param"

    print(f"\nwrote {{{','.join(VARIANTS)}}}Converted1.csv in each "
          f"{PRED_ROOT}/Roy*/")


if __name__ == "__main__":
    main()
