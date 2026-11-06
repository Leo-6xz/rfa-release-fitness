"""
make_test_corpus.py -- small stand-in corpora for testing the workflow's code.

NOT for the paper. The paper runs on the IEEE DataPort release, the synthetic
corpus from scripts/09_generate_detector_dataset.py, and PVDAQ.

    clean         R independent runs per class, random onset, 1 kHz
    release_like  one run per class, fixed onset, attack files sharing the Normal
                  solve before onset, DoS from a different solve, and divergent
                  values in one channel of six of the seven files -- the
                  release's documented defects, in miniature
"""
import argparse
import os

import numpy as np
import pandas as pd

STRATA = ["Normal", "Additive", "Ramp", "SlowRamp", "Sinusoid", "Stealth", "DoS"]
CH = (["V1"] + [f"{q}DG{i}" for i in range(1, 5) for q in ("P", "Q", "f")])


def base(rng, n, dt):
    t = np.arange(n) * dt
    X = np.zeros((n, len(CH)))
    e = rng.normal(0, 1, (n, len(CH)))
    for k in range(1, n):
        X[k] = 0.995 * X[k - 1] + 0.05 * e[k]
    X[:, 0] = 311 * np.sin(2 * np.pi * 50 * t) + X[:, 0]
    for i in range(4):
        X[:, 1 + 3 * i] += 0.5        # P
        X[:, 2 + 3 * i] += 0.2        # Q
        X[:, 3 + 3 * i] += 50.0       # f
    return t, X


def attack(X, t, s, onset, rng, scale=1.0):
    k = t >= onset
    tt = t[k] - onset
    j = int(rng.integers(4))
    P, F = 1 + 3 * j, 3 + 3 * j
    if s == "Additive":
        X[k, P] += 0.3 * scale; X[k, F] += 0.05 * scale
    elif s == "Ramp":
        X[k, F] += 0.4 * scale * tt
    elif s == "SlowRamp":
        X[k, 2 + 3 * j] += 0.08 * scale * tt
    elif s == "Sinusoid":
        X[k, P] += 0.3 * scale * np.sin(2 * np.pi * 2 * tt)
    elif s == "Stealth":
        X[k, F] += 0.03 * scale * np.sin(2 * np.pi * 0.7 * tt)
        X[k, (F + 3) % len(CH) or 3] += 0.03 * scale * np.sin(2 * np.pi * 0.7 * tt + 1)
    elif s == "DoS":
        X[k, F] = X[np.argmax(k), F]
    return k.astype(int)


def write(df_x, t, y, path):
    df = pd.DataFrame(df_x, columns=CH)
    df.insert(0, "time", t)
    df["label"] = y
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def clean(out, runs, seconds, dt, seed):
    rng = np.random.default_rng(seed)
    n = int(seconds / dt)
    for s in STRATA:
        for r in range(runs):
            t, X = base(rng, n, dt)
            y = np.zeros(n, int)
            if s != "Normal":
                y = attack(X, t, s, rng.uniform(0.25, 0.70) * seconds, rng, rng.uniform(0.4, 1.8))
            write(X, t, y, os.path.join(out, s, f"run_{r:03d}.csv"))


def release_like(out, seconds, dt, seed):
    rng = np.random.default_rng(seed)
    n = int(seconds / dt)
    t, X0 = base(rng, n, dt)
    for i, s in enumerate(STRATA):
        if s == "DoS":
            t, X = base(np.random.default_rng(seed + 99), n, dt)     # a different solve
        else:
            X = X0.copy()                                             # same solve as Normal
        y = np.zeros(n, int)
        if s != "Normal":
            y = attack(X, t, s, 0.5 * seconds, np.random.default_rng(i))
        if s != "Normal":                                             # divergence in 6 of 7 files
            k = t >= (0.6 + 0.04 * i) * seconds
            X[k, 2] = 8.2e152 * (1 + 0.1 * i)
        write(X, t, y, os.path.join(out, s, "run_000.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=["clean", "release_like"], required=True)
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.kind == "clean":
        clean(a.out, a.runs, a.seconds, a.dt, a.seed)
    else:
        release_like(a.out, a.seconds, a.dt, a.seed)
    print("written", a.out)
