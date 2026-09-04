#!/usr/bin/env python
"""Build the STIM-AS-CHANNEL single-sweep pack from RoyExpPack_ca3ft.

User (2026-08-28): "Rerun the single sweep model but include information of the
stim by passing it as a channel."  Same samples as the ca3ft pack (one sweep =
one sample), but volts_norm gains a second PROBE channel carrying the family's
RECORDED stimulus current in nA:

    <dom>_volts_norm : (N, 4001, 2, 1) fp16
        ch0 = fixed-norm soma voltage (unchanged)
        ch1 = recorded stim, nA (stim_Roy<amp>_pA/1000, lead-padded 4000->4001
              with its first value; range -0.75..+3.4 ~ comparable to volt z)
    everything else copied verbatim (labels keep the family index in col 0 for
    stim_from_label; raw mV, neuron ids, recorded stims, meta).

Train the 2-ch arm with NEUINV_PROBS="0 1" and the 1-ch A/B control with
NEUINV_PROBS="0" on this SAME pack (channel selection is probsSelect).
The loss targets soma_probe_index=0 only, so ch1 is input-only.
"""
import json, os
import numpy as np, h5py

SRC = "/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft/RoyExpChaotic.mlPack1.h5"
OUT = "/pscratch/sd/k/ktub1999/RoyExpPack_stimch/RoyExpStimCh.mlPack1.h5"
FAMS = ["Roy100", "Roy500", "Roy1000", "Roy1500", "Roy2000"]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with h5py.File(SRC, "r") as fs, h5py.File(OUT, "w") as fo:
    meta = json.loads(fs["meta.JSON"][0])
    stim_nA = {}
    for fam in FAMS:
        r = fs[f"stim_{fam}_pA"][:].astype(np.float32) / 1000.0     # (4000,) nA
        stim_nA[fam] = np.concatenate([[r[0]], r])                  # lead-pad -> 4001
    for dom in ("train", "valid", "test"):
        X = fs[dom + "_volts_norm"][:, :, 0, 0].astype(np.float32)  # (N,4001)
        famS = fs[dom + "_stim_family"][:].astype(str)
        I = np.stack([stim_nA[f] for f in famS], axis=0)            # (N,4001)
        V2 = np.stack([X, I], axis=2)[..., None]                    # (N,4001,2,1)
        fo.create_dataset(dom + "_volts_norm", data=V2.astype(np.float16),
                          chunks=(1, 4001, 2, 1))
        for key in ("_raw_volts_mV", "_unit_par", "_stim_family",
                    "_neuron_id", "_coverslip", "_src_file", "_stim_pA"):
            k = dom + key
            if k in fs:
                fo.create_dataset(k, data=fs[k][:])
        print(f"[{dom}] {len(X)} sweeps -> 2-channel")
    for k in fs:
        if k.startswith("stim_"):
            fo.create_dataset(k, data=fs[k][:])
    meta["cell_name"] = "RoyExpStimCh"
    meta["num_probs"] = 2
    meta["probe_names"] = ["soma_V_fixednorm", "stim_I_nA"]
    meta.setdefault("simu_info", {})
    meta["simu_info"]["probe_names"] = ["soma_V_fixednorm", "stim_I_nA"]
    meta["simu_info"].setdefault("stim_names", [f + "_icaRec_5k" for f in FAMS])
    meta["stimch_info"] = {"source": SRC, "ch1": "recorded stim nA, lead-padded",
                           "note": "loss targets soma_probe_index=0 only"}
    fo.create_dataset("meta.JSON",
                      data=np.asarray([json.dumps(meta)], dtype=h5py.string_dtype()))
print("wrote", OUT)
