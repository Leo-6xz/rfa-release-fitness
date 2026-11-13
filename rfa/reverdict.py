"""
reverdict.py -- recompute claim verdicts from stored check results under the
current specification, without rerunning any check.

Used when only the verdict rules change (spec 1.1.0 moved C6 from hard to soft
for K1-K3). Check results are read from results/<dataset>/checks.csv; nothing
is retrained. Prints old and new verdict per claim and writes
results/<dataset>/fitness_matrix_v<version>.csv.

    python rfa/reverdict.py
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rfa_spec as S  # noqa: E402

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
print(f"release-fitness spec {S.SPEC_VERSION} (hash {S.spec_hash()})")
changed = 0
for d in sorted(glob.glob(os.path.join(root, "*", "checks.csv"))):
    name = os.path.basename(os.path.dirname(d))
    st = {r["check"]: r["status"] for r in csv.DictReader(open(d))}
    old = {r["claim"]: r["verdict"] for r in csv.DictReader(open(os.path.join(os.path.dirname(d), "fitness_matrix.csv")))}
    rows = []
    for k in old:
        v, why = S.claim_verdict(k, st)
        changed += v != old[k]
        print(f"{name:22s} {k}: {old[k]:12s} -> {v:12s} {'; '.join(why)}")
        rows.append({"claim": k, "name": S.CLAIMS[k]["name"], "verdict": v, "because": "; ".join(why)})
    out = os.path.join(os.path.dirname(d), f"fitness_matrix_v{S.SPEC_VERSION}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["claim", "name", "verdict", "because"])
        w.writeheader(); w.writerows(rows)
print(f"verdicts changed: {changed}")
