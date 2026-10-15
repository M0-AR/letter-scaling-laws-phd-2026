"""Scaling-law fits: floor search + power law, joint N/D law, forecasts, IsoFLOP optimum.

Video claims:
  L(N) = floor + gap0 * (N/N0)^-k, 4x size keeps ~0.75 gap (k~0.207).
  10x keeps ~0.60; halving needs ~24x... but joint fit says 72% per 4x -> check.
  L(N,D) = E + A*N^-a + B*D^-b ; C ~= 6*N*D.
"""
from __future__ import annotations
import math
import numpy as np


def fit_floor_power(N: np.ndarray, L: np.ndarray, floors: np.ndarray | None = None):
    """Grid-search floor in [0, min(L)); for each, linear fit log(gap) vs log(N).

    Returns dict(floor, slope k, intercept, r2, pred_fn).
    Best = max R^2 (most constant proportional shrink per step).
    """
    N = np.asarray(N, dtype=float)
    L = np.asarray(L, dtype=float)
    if floors is None:
        floors = np.arange(0.0, float(L.min()) - 0.05, 0.01)
    best = None
    logN = np.log(N)
    for f in floors:
        gap = L - float(f)
        if np.any(gap <= 0):
            continue
        y = np.log(gap)
        A = np.vstack([logN, np.ones_like(logN)]).T
        coef, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
        slope, icept = float(coef[0]), float(coef[1])
        yhat = slope * logN + icept
        ss_tot = float(((y - y.mean()) ** 2).sum())
        ss_res = float(((y - yhat) ** 2).sum())
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        # want negative slope (gap shrinks)
        score = r2 if slope < 0 else -1
        if best is None or score > best["r2"]:
            k = -slope
            per4x = float(4.0 ** slope)  # gap multiplier per 4x size
            per10x = float(10.0 ** slope)
            half_mult = float(0.5 ** (1.0 / slope)) if slope < 0 else float("inf")
            best = {"floor": float(f), "slope": slope, "k": k, "intercept": icept,
                    "r2": r2, "per4x": per4x, "per10x": per10x,
                    "halving_mult": half_mult}
    return best


def predict(best: dict, Nnew: np.ndarray, N0: float | None = None) -> np.ndarray:
    Nnew = np.asarray(Nnew, dtype=float)
    # gap(N) = exp(intercept) * N^slope
    return np.asarray(best["floor"] + np.exp(best["intercept"]) * (Nnew ** best["slope"]))


def fit_joint(N: np.ndarray, D: np.ndarray, L: np.ndarray):
    """Fit L = E + A*N^-a + B*D^-b via coarse grid + least squares on A,B.

    Returns dict(E, a, b, A, B, mae, maxe, per4x_N, per4x_D).
    """
    from itertools import product
    N = np.asarray(N, dtype=float); D = np.asarray(D, dtype=float); L = np.asarray(L, dtype=float)
    best = None
    for E in np.arange(1.0, 2.0, 0.02):
        for a in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            for b in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                X = np.vstack([N ** (-a), D ** (-b)]).T
                y = L - E
                if np.any(y <= 0):
                    continue
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                A, B = float(coef[0]), float(coef[1])
                if A <= 0 or B <= 0:
                    continue
                pred = E + A * N ** (-a) + B * D ** (-b)
                mae = float(np.abs(pred - L).mean()); mx = float(np.abs(pred - L).max())
                if best is None or mae < best["mae"]:
                    best = {"E": float(E), "a": a, "b": b, "A": A, "B": B, "mae": mae,
                            "maxe": mx, "per4x_N": float(4.0 ** (-a)), "per4x_D": float(4.0 ** (-b))}
    return best


def isoflop_optimum(C: float, joint: dict, n_grid: int = 400) -> dict:
    """Minimize L(N, D=C/6N) over N for fixed compute C=6ND."""
    Ns = np.logspace(2, 7, n_grid)
    Ds = C / (6.0 * Ns)
    L = joint["E"] + joint["A"] * Ns ** (-joint["a"]) + joint["B"] * Ds ** (-joint["b"])
    i = int(np.argmin(L))
    return {"N": float(Ns[i]), "D": float(Ds[i]), "L": float(L[i])}


def scaling_exponents_from_optima(Cs: np.ndarray, joint: dict):
    """Fit N_opt ~ C^p, D_opt ~ C^q across budgets."""
    Ns, Ds = [], []
    for C in Cs:
        o = isoflop_optimum(float(C), joint)
        Ns.append(o["N"]); Ds.append(o["D"])
    Ns, Ds = np.array(Ns), np.array(Ds)
    p = float(np.polyfit(np.log(Cs), np.log(Ns), 1)[0])
    q = float(np.polyfit(np.log(Cs), np.log(Ds), 1)[0])
    return {"p": p, "q": q, "Ns": Ns, "Ds": Ds}
