"""
pvdaq_funnel.py

Regenerate, from the catalogue and the bucket listings alone, every number in
the PVDAQ effective-sample-size funnel -- so the table in the paper is one
command, not a reconstruction from a working session.

    catalogued systems
    -> pass QA, >= 3 years, catalogue-implied interval <= 20 min
    -> of which PVOutput.org  (verified on a random sample: daily aggregates only)
    -> non-PVOutput candidates
    -> with two consecutive years of >= 330 daily intraday files
    -> physically independent at d km, for several d

Independence is reported as the EXACT maximum number of candidate sites that
are mutually at least d km apart (a maximum independent set on the "closer
than d" graph), not a greedy count, so the number is a property of the data
and not of the order the sites happened to be visited in. With ~25 sites this
is instant.

Listings are cached in --cache, so a rerun is fast and reproducible offline;
delete the cache to re-query the bucket.

Usage (PowerShell, one line):
    python pvdaq_funnel.py
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd

from fetch_metadata import haversine_km
from inventory_sites import list_site, summarise

MIN_PER_YEAR = 525_600.0


def cached_summary(sid, cache):
    key = str(sid)
    if key not in cache:
        cache[key] = summarise(list_site(sid))
    return cache[key]


def usable_years(summary, min_days, n_years):
    good = {r["year"] for r in summary
            if r["kind"] in ("intraday", "mixed") and r["days"] >= min_days}
    for y in sorted(good, reverse=True):
        if all(v in good for v in range(y - n_years + 1, y + 1)):
            return list(range(y - n_years + 1, y + 1))
    return None


def max_independent(adj):
    """Exact maximum independent set size on a small graph (branch and bound)."""
    n = len(adj)
    nbrs = [set(np.flatnonzero(adj[i])) for i in range(n)]
    best = [0]

    def rec(cands, size):
        if size + len(cands) <= best[0]:
            return
        if not cands:
            best[0] = max(best[0], size)
            return
        v = max(cands, key=lambda u: len(nbrs[u] & cands))
        rec(cands - {v} - nbrs[v], size + 1)      # take v
        if nbrs[v] & cands:                        # skipping v only helps if it has neighbours
            rec(cands - {v}, size)

    rec(set(range(n)), 0)
    return best[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=r"pvdaq\systems.csv")
    ap.add_argument("--cache", default=r"pvdaq\listing_cache.json")
    ap.add_argument("--min-years", type=float, default=3.0)
    ap.add_argument("--max-interval", type=float, default=20.0)
    ap.add_argument("--min-days", type=int, default=330)
    ap.add_argument("--n-years", type=int, default=2)
    ap.add_argument("--pvoutput-sample", type=int, default=50,
                    help="random PVOutput systems listed to verify they are daily-only")
    ap.add_argument("--distances", default="200,100,50,25,10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=r"pvdaq\funnel.csv")
    a = ap.parse_args()

    cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}

    cat = pd.read_csv(a.catalogue)
    cat = cat.loc[:, ~cat.columns.str.startswith("Unnamed")]
    for c in ("years", "number_records", "available_sensor_channels", "latitude", "longitude"):
        cat[c] = pd.to_numeric(cat[c], errors="coerce")
    cat["est_interval_min"] = MIN_PER_YEAR / (
        cat["number_records"] / cat["years"] / cat["available_sensor_channels"])
    is_pvo = cat["system_public_name"].astype(str).str.startswith("Pvoutput")
    base = (cat["qa_status"].astype(str).str.strip().str.lower().eq("pass")
            & (cat["years"] >= a.min_years)
            & cat["latitude"].notna() & cat["longitude"].notna())

    funnel = [("catalogued systems", len(cat), "")]
    eligible = cat[base & (cat["est_interval_min"] <= a.max_interval)]
    funnel.append((f"pass QA, >= {a.min_years:g} years, catalogue-implied interval "
                   f"<= {a.max_interval:g} min", len(eligible), ""))
    pvo = eligible[is_pvo.loc[eligible.index]]
    funnel.append(("  of which PVOutput.org", len(pvo), ""))

    # Verify the PVOutput claim on a random sample rather than asserting it.
    random.seed(a.seed)
    sample = random.sample(sorted(pvo["system_id"].astype(int)), min(a.pvoutput_sample, len(pvo)))
    print(f"listing {len(sample)} random PVOutput systems ...")
    n_daily_only = 0
    for sid in sample:
        s = cached_summary(sid, cache)
        if s and all(r["kind"] == "daily" for r in s):
            n_daily_only += 1
    funnel.append((f"    sampled PVOutput systems that are daily aggregates only",
                   n_daily_only, f"of {len(sample)} sampled (seed {a.seed})"))

    non = eligible[~is_pvo.loc[eligible.index]]
    funnel.append(("  non-PVOutput candidates", len(non), ""))
    print(f"listing {len(non)} non-PVOutput candidates ...")
    good, empty, daily_only = [], 0, 0
    for _, r in non.iterrows():
        sid = int(r["system_id"])
        s = cached_summary(sid, cache)
        if not s:
            empty += 1
            continue
        if all(x["kind"] == "daily" for x in s):
            daily_only += 1
        yrs = usable_years(s, a.min_days, a.n_years)
        if yrs:
            good.append((sid, float(r["latitude"]), float(r["longitude"]), yrs))
    json.dump(cache, open(a.cache, "w"))

    funnel.append(("    daily aggregates only", daily_only, ""))
    funnel.append(("    empty prefix", empty, ""))
    funnel.append((f"    with {a.n_years} consecutive years >= {a.min_days} intraday days",
                   len(good), ""))

    if good:
        la = np.radians([g[1] for g in good])
        lo = np.radians([g[2] for g in good])
        dist = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
        for dkm in [float(x) for x in a.distances.split(",")]:
            adj = (dist < dkm) & ~np.eye(len(good), dtype=bool)
            funnel.append((f"      max sites mutually >= {dkm:g} km apart",
                           max_independent(adj), "exact maximum independent set"))

    out = pd.DataFrame(funnel, columns=["step", "count", "note"])
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 90)
    print("\n" + out.to_string(index=False))
    print(f"\nwritten: {a.out}   listing cache: {a.cache}")


if __name__ == "__main__":
    main()
