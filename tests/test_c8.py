"""C8 gate on constructed records. Run from the repository root:
    python tests/test_c8.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rfa"))
import rfa_core as R

def unit(stratum, D, onset=None):
    y = np.zeros(len(D), "int8")
    if onset is not None:
        y[onset:] = 1
    return {"stratum": stratum, ("_dec", 1.0): (D, y, D)}

def run(units):
    ds = {"channels": ["a"], "units": units, "strata": ["Normal", "Attack"]}
    desc = {"physical_dt_s": 1.0, "differential": {"baseline_stratum": "Normal", "channels": ["a"]}}
    return R.c8_differential(ds, desc)["status"]

rng = np.random.default_rng(0); n, o = 100, 50
base = rng.normal(size=(n, 1))
ev = base.copy(); ev[o:] += 1.0
nanbase = base.copy(); nanbase[30:o] = np.nan
noisy = base + 1e-3 * rng.normal(size=(n, 1)); noisy[o:] += 1.0
cases = [
    ("identical event and baseline", [unit("Normal", base), unit("Attack", base.copy(), o)], "FAIL"),
    ("exact pre-onset match, then divergence", [unit("Normal", base), unit("Attack", ev, o)], "PASS"),
    ("undefined first baseline, exact second", [unit("Normal", nanbase), unit("Normal", base), unit("Attack", ev, o)], "PASS"),
    ("small pre-onset mismatch, large divergence", [unit("Normal", base), unit("Attack", noisy, o)], "PASS"),
]
bad = 0
for name, units, want in cases:
    got = run(units); bad += got != want
    print(f"{'ok ' if got == want else 'BAD'} {name}: {got} (expected {want})")
sys.exit(1 if bad else 0)
