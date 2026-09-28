"""Physiological-plausibility measures for soma voltage traces (mV).

Shared by scripts/scan_cells_default_physiology.py, scripts/scan_ca3_packs_physiology.py
and scripts/stim_scale_scan.py so every table in the repo uses ONE definition of
"spike", "width", "AHP", "block" and "out of range".

    spike       upward crossing of 0 mV
    peak        max of the trace between consecutive spike onsets
    width       time above the midpoint between -20 mV and the peak (ms)
    AHP         minimum in the 20 ms after the peak (mV)
    rest        median of the first 50 ms (pre-stimulus for the 5k stims)
    block       any run >= 50 ms continuously above -20 mV (depolarisation block)
    oor         samples above +60 mV or below -120 mV (not seen in real cells)

`verdict()` turns the per-trace / per-population numbers into PASS/FAIL against
the criteria documented in docs/model_ladder/README.md.
"""
import numpy as np


def spike_stats(v, dt):
    v = np.asarray(v, dtype=float)
    T = len(v)
    up = np.where((v[1:] > 0) & (v[:-1] <= 0))[0] + 1
    n = len(up)
    peaks, widths, ahps = [], [], []
    for i, s in enumerate(up):
        e = up[i + 1] if i + 1 < n else T
        seg = v[s:e]
        if len(seg) < 3:
            continue
        pk = int(np.argmax(seg)); peak = float(seg[pk])
        half = (peak - 20.0) / 2.0
        above = np.where(seg > half)[0]
        widths.append(float((above[-1] - above[0] + 1) * dt) if len(above) else np.nan)
        peaks.append(peak)
        win = seg[pk:pk + int(20 / dt)]
        ahps.append(float(win.min()) if len(win) else np.nan)
    rest = float(np.median(v[: int(50 / dt)]))
    hi = v > -20.0
    run = 0; block = False
    for h in hi:
        run = run + 1 if h else 0
        if run * dt >= 50.0:
            block = True; break
    return dict(rest=rest, vmin=float(np.nanmin(v)), vmax=float(np.nanmax(v)), n=n,
                peak=float(np.mean(peaks)) if peaks else np.nan,
                width=float(np.nanmean(widths)) if widths else np.nan,
                ahp=float(np.nanmean(ahps)) if ahps else np.nan, block=block,
                nonfinite=int((~np.isfinite(v)).sum()),
                oor=int(((v > 60) | (v < -120)).sum()))


def scan_traces(V, dt):
    """Population summary of a (N, T) block of mV traces."""
    rows = [spike_stats(v, dt) for v in V]
    A = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    with np.errstate(all="ignore"):
        return dict(n_tr=len(rows), rest=float(np.nanmedian(A["rest"])),
                    vmin=float(np.nanmin(A["vmin"])), vmax=float(np.nanmax(A["vmax"])),
                    silent=100 * float(np.mean(A["n"] == 0)),
                    spk_med=float(np.nanmedian(A["n"])), spk_p95=float(np.nanpercentile(A["n"], 95)),
                    spk_max=float(np.nanmax(A["n"])),
                    peak=float(np.nanmedian(A["peak"])) if np.isfinite(A["peak"]).any() else np.nan,
                    width=float(np.nanmedian(A["width"])) if np.isfinite(A["width"]).any() else np.nan,
                    ahp=float(np.nanmedian(A["ahp"])) if np.isfinite(A["ahp"]).any() else np.nan,
                    block=100 * float(np.mean(A["block"])),
                    nonfinite=100 * float(np.mean(A["nonfinite"] > 0)),
                    oor=100 * float(np.mean(A["oor"] > 0)), per=A)


# ── PASS/FAIL criteria (docs/model_ladder/README.md) ─────────────────────────
DEFAULT_CRIT = dict(rest=(-85.0, -55.0), spikes_min=3, peak=(0.0, 55.0), width=(0.2, 4.0),
                    ahp_max=-45.0, vmin_min=-105.0, vmax_max=60.0)
BOX_CRIT = dict(silent_max=40.0, block_max=10.0, oor_max=5.0, nonfinite_max=0.0, spk_med_min=2.0)


def verdict_default(st, crit=DEFAULT_CRIT):
    """Reasons a single default-parameter trace fails; [] == PASS."""
    r = []
    if not (crit["rest"][0] <= st["rest"] <= crit["rest"][1]): r.append(f"rest {st['rest']:.0f}")
    if st["n"] < crit["spikes_min"]: r.append(f"{st['n']} spikes")
    if st["n"] > 0 and not (crit["peak"][0] <= st["peak"] <= crit["peak"][1]): r.append(f"peak {st['peak']:.0f}")
    if st["n"] > 0 and not (crit["width"][0] <= st["width"] <= crit["width"][1]): r.append(f"width {st['width']:.1f}ms")
    if st["n"] > 0 and not (st["ahp"] <= crit["ahp_max"]): r.append(f"AHP {st['ahp']:.0f}")
    if st["block"]: r.append("block")
    if st["vmin"] < crit["vmin_min"]: r.append(f"Vmin {st['vmin']:.0f}")
    if st["vmax"] > crit["vmax_max"]: r.append(f"Vmax {st['vmax']:.0f}")
    if st["nonfinite"]: r.append("non-finite")
    return r


def verdict_box(st, crit=BOX_CRIT):
    """Reasons a parameter-box population fails; [] == PASS."""
    r = []
    if st["silent"] > crit["silent_max"]: r.append(f"silent {st['silent']:.0f}%")
    if st["block"] > crit["block_max"]: r.append(f"block {st['block']:.0f}%")
    if st["oor"] > crit["oor_max"]: r.append(f"oor {st['oor']:.0f}%")
    if st["nonfinite"] > crit["nonfinite_max"]: r.append(f"non-finite {st['nonfinite']:.1f}%")
    if st["spk_med"] < crit["spk_med_min"]: r.append(f"median {st['spk_med']:.0f} spikes")
    return r
