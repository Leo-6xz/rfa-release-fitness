"""
inject_defects.py -- does the workflow catch what it claims to catch?

Loads a CLEAN labelled corpus once, then builds six variants, each with exactly
one planted defect, and runs every check on the clean corpus and on each
variant. A defect is DETECTED when the check aimed at it moves to a worse
verdict than it has on the clean corpus. Any OTHER check that changes verdict is
reported as an off-target effect, not hidden.

    D1  integrity      2% of rows in every unit get a divergent value (1e30) in
                       one channel -- spread over all classes, so it is not a
                       class shortcut
    D2  metadata       the descriptor declares a sampling interval 10x the real one
    D3  sample size    one unit per class is kept
    D4  independence   30% of units are duplicated under new identifiers
    D5  shortcut       every class's records are truncated to a class-specific
                       length (80% + 3% x class index), after the event
    D6  class floor    in one event class, all but ~20 physical-timescale event
                       samples are relabelled as normal

C8 is not exercised here: the synthetic corpus has no paired baseline runs, so
it is NA on the clean corpus and on every variant. It is exercised on the
microgrid release, where it excludes denial of service.

    python inject_defects.py --descriptor descriptors\\microgrid_synthetic.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rfa_core as C        # noqa: E402
import rfa_spec as S        # noqa: E402

CHECKS = ["C1", "C2", "C3", "C4", "C5", "C5g", "C6", "C7", "C8"]
TARGET = {"D1": "C1", "D2": "C2", "D3": "C3", "D4": "C4", "D5": "C6", "D6": "C7"}
NAMES = {"D1": "divergent values", "D2": "wrong declared interval", "D3": "one unit per class",
         "D4": "duplicated units", "D5": "class-dependent record length",
         "D6": "event class below floor"}
WORSE = {"PASS": 0, "CONDITIONAL": 1, "FAIL": 2, "NOT_TESTABLE": 2, "NA": -1}


def d1(ds, desc, rng):
    ds = C.clone(ds)
    j = int(rng.integers(len(ds["channels"])))
    for u in ds["units"]:
        X = u["X"].copy()
        rows = rng.random(len(X)) < 0.02
        X[rows, j] = 1.0e30
        u["X"] = X
    return ds, desc, {"channel": ds["channels"][j]}


def d2(ds, desc, rng):
    d = dict(desc)
    d["sample_interval_s"] = float(desc["sample_interval_s"]) * 10
    return C.clone(ds), d, {"declared_interval_s": d["sample_interval_s"]}


def d3(ds, desc, rng):
    ds = C.clone(ds)
    keep, seen = [], set()
    for u in ds["units"]:
        if u["stratum"] not in seen:
            keep.append(u); seen.add(u["stratum"])
    ds["units"] = keep
    return ds, desc, {"units_kept": len(keep)}


def d4(ds, desc, rng):
    ds = C.clone(ds)
    n = len(ds["units"])
    pick = rng.choice(n, size=int(round(0.30 * n)), replace=False)
    for i in pick:
        u = dict(ds["units"][i])
        u = {k: v for k, v in u.items() if not (isinstance(k, tuple) and k[0] == "_dec")}
        u["id"] = u["id"] + "#dup"
        ds["units"].append(u)
    return ds, desc, {"duplicated": int(len(pick))}


def d5(ds, desc, rng):
    ds = C.clone(ds)
    cut = {}
    for u in ds["units"]:
        k = ds["strata"].index(u["stratum"])
        f = 0.80 + 0.03 * k
        n = len(u["X"])
        first = int(np.argmax(u["y"] > 0)) if u["y"].max() > 0 else 0
        m = max(int(f * n), first + 1)          # never cut away the whole event
        u["X"], u["t"], u["y"] = u["X"][:m], u["t"][:m], u["y"][:m]
        cut[u["stratum"]] = round(f, 2)
    return ds, desc, {"length_fraction_by_class": cut}


def d6(ds, desc, rng, target_positives=20):
    ds = C.clone(ds)
    null = desc.get("null_stratum", ds["strata"][0])
    events = [s for s in ds["strata"] if s != null]
    victim = events[int(rng.integers(len(events)))]
    units = [u for u in ds["units"] if u["stratum"] == victim and u["y"].max() > 0]
    f = C.factor_for(units[0], desc)
    budget = target_positives * f          # raw samples, one physical sample = f raw samples
    for u in units:
        y = u["y"].copy()
        pos = np.flatnonzero(y)
        keep = min(len(pos), max(0, budget))
        # keep whole physical blocks so the count lands on the target
        keep = (keep // f) * f
        y[pos[keep:]] = 0
        budget -= keep
        u["y"] = y
    return ds, desc, {"class": victim, "target_positives": target_positives}


DEFECTS = {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptor", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeats", type=int, default=None, help="testing only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated defect ids, testing only")
    a = ap.parse_args()

    desc = json.load(open(a.descriptor))
    out = a.out or os.path.join("results", desc["name"] + "_defects")
    os.makedirs(out, exist_ok=True)
    print(f"spec {S.SPEC_VERSION} (hash {S.spec_hash()}); clean corpus '{desc['name']}'")
    base = C.load_dataset(desc)

    rows, details = [], {}
    print("\n[clean]")
    clean = C.run_checks(C.clone(base), desc, repeats=a.repeats)
    rows.append({"variant": "clean", **{c: clean[c]["status"] for c in CHECKS}})
    details["clean"] = {c: clean[c]["metric"] for c in CHECKS}

    todo = [d for d in DEFECTS if not a.only or d in a.only.split(",")]
    summary = []
    for did in todo:
        rng = np.random.default_rng(a.seed)
        ds, d, info = DEFECTS[did](base, desc, rng)
        print(f"\n[{did}] {NAMES[did]}  {info}")
        res = C.run_checks(ds, d, repeats=a.repeats)
        st = {c: res[c]["status"] for c in CHECKS}
        rows.append({"variant": did, **st})
        details[did] = {"info": info, **{c: res[c]["metric"] for c in CHECKS}}
        tgt = TARGET[did]
        detected = WORSE[st[tgt]] > WORSE[clean[tgt]["status"]]
        off = [c for c in CHECKS if c != tgt and st[c] != clean[c]["status"]]
        summary.append({"defect": did, "planted": NAMES[did], "target_check": tgt,
                        "clean_status": clean[tgt]["status"], "defect_status": st[tgt],
                        "detected": detected, "off_target_changes": ", ".join(
                            f"{c}:{clean[c]['status']}->{st[c]}" for c in off)})

    mat = pd.DataFrame(rows)
    summ = pd.DataFrame(summary)
    mat.to_csv(os.path.join(out, "defect_matrix.csv"), index=False)
    summ.to_csv(os.path.join(out, "defect_summary.csv"), index=False)
    json.dump({"spec_version": S.SPEC_VERSION, "spec_hash": S.spec_hash(), "seed": a.seed,
               "repeats_override": a.repeats, "metrics": details},
              open(os.path.join(out, "defect_metrics.json"), "w"), indent=2, default=str)
    pd.set_option("display.width", 220)
    print("\nverdict by variant\n" + mat.to_string(index=False))
    print("\ndetection\n" + summ.to_string(index=False))
    print(f"\n{int(summ['detected'].sum())} of {len(summ)} planted defects detected by the intended check")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
