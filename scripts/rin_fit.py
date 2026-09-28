#!/usr/bin/env python3
"""Apparent input resistance + membrane tau from a (V, I) trace pair, identical metric for
recordings and simulations: fit V(t) = V0 + R * (k_tau * I)(t) by least squares over the
SUB-THRESHOLD samples (drop 3 ms before .. 15 ms after every upward -20 mV crossing and any
V > -50 mV), grid-searching tau.  k_tau = causal unit-area exponential filter (single RC).
Returns R in MOhm (mV/nA), tau in ms, V0 in mV, r2 of the fit, n spikes."""
import numpy as np

TAUS = np.array([2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 65, 80], float)


def spikes(v, thr=-20.0):
    return np.flatnonzero((v[:-1] < thr) & (v[1:] >= thr))


def rc_filter(i, dt, tau):
    a = np.exp(-dt / tau)
    y = np.empty_like(i); acc = 0.0
    for k, x in enumerate(i):               # y' = (i - y)/tau  (unit DC gain)
        acc = a * acc + (1 - a) * x; y[k] = acc
    return y


def fit_rin(v, i_nA, dt):
    v = np.asarray(v, float); i_nA = np.asarray(i_nA, float)
    sp = spikes(v)
    keep = v < -50.0
    pre, post = int(3 / dt), int(15 / dt)
    for s in sp:
        keep[max(0, s - pre):s + post] = False
    best = None
    for tau in TAUS:
        f = rc_filter(i_nA, dt, tau)
        A = np.stack([np.ones(keep.sum()), f[keep]], 1)
        coef, *_ = np.linalg.lstsq(A, v[keep], rcond=None)
        res = v[keep] - A @ coef
        sse = float(res @ res)
        if best is None or sse < best[0]:
            best = (sse, tau, coef)
    sse, tau, (v0, R) = best
    var = float(((v[keep] - v[keep].mean()) ** 2).sum())
    return dict(R=float(R), tau=float(tau), V0=float(v0), r2=1 - sse / max(var, 1e-9),
                n_spk=int(len(sp)), frac_used=float(keep.mean()))
