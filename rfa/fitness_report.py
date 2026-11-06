"""
fitness_report.py -- run the release-fitness assessment on one dataset.

    python fitness_report.py --descriptor descriptors\\microgrid_release.json

Writes, under --out (default results\\<dataset name>):
    report.json         every check with its metric, verdict and evidence,
                        the claim verdicts, the spec version and hash, and the
                        descriptor used
    checks.csv          one row per check
    fitness_matrix.csv  one row per claim: PASS / CONDITIONAL / FAIL and why

Every number for the paper's fitness tables comes from this command.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rfa_core as C        # noqa: E402
import rfa_spec as S        # noqa: E402

ORDER = ["C1", "C1d", "C2", "C3", "C4", "C5", "C5g", "C6", "C7", "C8"]


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptor", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeats", type=int, default=None,
                    help="override split repeats (testing only; the paper uses the spec value)")
    a = ap.parse_args()

    desc = json.load(open(a.descriptor))
    out = a.out or os.path.join("results", desc["name"])
    os.makedirs(out, exist_ok=True)
    print(f"release-fitness spec {S.SPEC_VERSION} (hash {S.spec_hash()}), dataset '{desc['name']}'")
    if a.repeats:
        print(f"  NOTE: repeats overridden to {a.repeats}; not a paper run")

    ds = C.load_dataset(desc)
    res = C.run_checks(ds, desc, repeats=a.repeats)
    claims = C.verdicts(res, desc.get("claims", list(S.CLAIMS)))

    print("\nchecks")
    for cid in ORDER:
        r = res.get(cid)
        if r is None:
            continue
        print(f"  {cid:<4s} {r['status']:<13s} {fmt(r['metric']):>12s}  {r['metric_name']}")
    print("\nfitness by claim")
    for k, v in claims.items():
        why = "; ".join(v["because"]) if v["because"] else ""
        print(f"  {k} {v['claim']:<42s} {v['verdict']:<12s} {why}")

    report = {
        "dataset": desc["name"],
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "spec_version": S.SPEC_VERSION, "spec_hash": S.spec_hash(),
        "descriptor": desc,
        "descriptor_hash": hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest()[:16],
        "python": platform.python_version(),
        "repeats_override": a.repeats,
        "checks": res, "claims": claims,
    }
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2, default=str)
    pd.DataFrame([{"check": cid, "name": S.CHECKS.get(cid[:2], {}).get("name", cid),
                   "status": res[cid]["status"], "metric": res[cid]["metric"],
                   "metric_name": res[cid]["metric_name"]} for cid in ORDER if cid in res]) \
        .to_csv(os.path.join(out, "checks.csv"), index=False)
    pd.DataFrame([{"claim": k, "name": v["claim"], "verdict": v["verdict"],
                   "because": "; ".join(v["because"])} for k, v in claims.items()]) \
        .to_csv(os.path.join(out, "fitness_matrix.csv"), index=False)
    print(f"\nwritten: {os.path.join(out, 'report.json')}, checks.csv, fitness_matrix.csv")


if __name__ == "__main__":
    main()
