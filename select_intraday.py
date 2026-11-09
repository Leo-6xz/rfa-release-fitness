"""
select_intraday.py

Choose sites for the leakage probe by checking the data lake itself, not the
catalogue or the metadata.

What went wrong before, and what this does instead
--------------------------------------------------
1. PVOutput.org systems -- ~95% of the systems the catalogue marks as 10-20
   minute data -- are stored in the public CSV lake as one annual file of
   DAILY aggregates per year. All 30 downloaded sites, 365 site-years, were
   that. They are excluded here by name.

2. The NREL-sourced systems that DO have one file per day have EMPTY Metrics
   blocks in their metadata JSON, so a metadata-based "reports AC power"
   filter removed exactly the systems with intraday data. Metadata is not used
   to decide anything here.

3. The catalogue's first/last timestamps cannot see gaps inside a record.

So each candidate is accepted only after its own listing shows two consecutive
years with at least --min-days daily files each. Candidates are visited in
order of climate round-robin and record length, and a candidate must also be at
least --min-km from every site already accepted -- distance, not site_id, is
the unit of independence.

Resolution: anything at or finer than --max-interval minutes is accepted. The
probe averages every series to a common 15-minute grid, so a 1- or 5-minute
system is usable; what must be avoided is MIXING resolutions, not fine ones.

Usage (PowerShell, one line):
    python select_intraday.py
"""

import argparse
import sys

import numpy as np
import pandas as pd

from fetch_metadata import haversine_km
from inventory_sites import list_site, summarise

MIN_PER_YEAR = 525_600.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=r"pvdaq\systems.csv")
    ap.add_argument("--min-years", type=float, default=3.0)
    ap.add_argument("--max-interval", type=float, default=20.0,
                    help="minutes; coarser systems are excluded, finer ones are resampled")
    ap.add_argument("--min-km", type=float, default=100.0)
    ap.add_argument("--min-days", type=int, default=330)
    ap.add_argument("--n-years", type=int, default=2)
    ap.add_argument("--n-sites", type=int, default=30)
    ap.add_argument("--per-climate", type=int, default=6)
    ap.add_argument("--out", default=r"pvdaq\plan_intraday.csv")
    a = ap.parse_args()

    cat = pd.read_csv(a.catalogue)
    cat = cat.loc[:, ~cat.columns.str.startswith("Unnamed")]
    for c in ("years", "number_records", "available_sensor_channels",
              "latitude", "longitude", "dataset_size_mb"):
        cat[c] = pd.to_numeric(cat[c], errors="coerce")
    cat["est_interval_min"] = MIN_PER_YEAR / (
        cat["number_records"] / cat["years"] / cat["available_sensor_channels"])

    name = cat["system_public_name"].astype(str)
    steps = [
        ("not PVOutput.org (daily aggregates only)", ~name.str.startswith("Pvoutput")),
        ("qa_status == pass", cat["qa_status"].astype(str).str.strip().str.lower().eq("pass")),
        (f"years >= {a.min_years:g}", cat["years"] >= a.min_years),
        (f"interval <= {a.max_interval:g} min", cat["est_interval_min"] <= a.max_interval),
        ("has coordinates", cat["latitude"].notna() & cat["longitude"].notna()),
    ]
    keep = pd.Series(True, index=cat.index)
    print(f"catalogue: {len(cat)}")
    for label, m in steps:
        keep &= m.fillna(False)
        print(f"  after {label:<42s} {int(keep.sum()):>5d}")
    cand = cat[keep].copy()
    if cand.empty:
        sys.exit("no candidates")

    lat = np.radians(cand["latitude"].to_numpy(float))
    lon = np.radians(cand["longitude"].to_numpy(float))
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(d, np.inf)
    print(f"  of which have another candidate within {a.min_km:g} km: {int((d < a.min_km).any(1).sum())}")

    # A blank climate class must NOT remove a candidate. groupby() drops NaN keys
    # by default, and the first run lost 91 of 120 candidates that way -- 8
    # checked + 21 skipped for spacing accounted for only 29.
    n_blank = int(cand["kg_climate"].isna().sum())
    cand["kg_climate"] = cand["kg_climate"].fillna("unknown")
    if n_blank:
        print(f"  candidates with no climate class (kept, grouped as 'unknown'): {n_blank}")
    cand = cand.sort_values(["kg_climate", "years"], ascending=[True, False])
    queues = {c: list(g.index) for c, g in cand.groupby("kg_climate", dropna=False)}
    # 'unknown' is not a climate, so it must not be capped as if it were one.
    uncapped = {"unknown"}
    accepted, lat_p, lon_p = [], [], []
    per_clim = {c: 0 for c in queues}
    checked = no_intraday = too_close = 0

    print(f"\nchecking listings (each accepted site has {a.n_years} consecutive years "
          f">= {a.min_days} days) ...")
    progress = True
    while len(accepted) < a.n_sites and progress:
        progress = False
        for c in queues:
            if len(accepted) >= a.n_sites or (per_clim[c] >= a.per_climate and c not in uncapped):
                continue
            while queues[c]:
                idx = queues[c].pop(0)
                row = cand.loc[idx]
                la, lo = np.radians(row["latitude"]), np.radians(row["longitude"])
                if lat_p and haversine_km(la, lo, np.array(lat_p), np.array(lon_p)).min() < a.min_km:
                    too_close += 1
                    continue
                checked += 1
                sid = int(row["system_id"])
                try:
                    rows = summarise(list_site(sid))
                except Exception as e:
                    print(f"  {sid}: listing failed ({e}); skipped")
                    continue
                good = [r["year"] for r in rows
                        if r["kind"] in ("intraday", "mixed") and r["days"] >= a.min_days]
                pick = None
                for y in sorted(good, reverse=True):
                    run = list(range(y - a.n_years + 1, y + 1))
                    if all(v in good for v in run):
                        pick = run
                        break
                if pick is None:
                    no_intraday += 1
                    continue
                accepted.append({"system_id": sid, "years": "-".join(map(str, pick)),
                                 "climate": c, "name": row["system_public_name"],
                                 "latitude": row["latitude"], "longitude": row["longitude"],
                                 "est_interval_min": row["est_interval_min"],
                                 "record_years": row["years"]})
                lat_p.append(la); lon_p.append(lo); per_clim[c] += 1
                progress = True
                print(f"  + {sid:>6d} {c:<4s} {str(row['system_public_name'])[:36]:<36s} {pick[0]}-{pick[-1]}")
                break

    out = pd.DataFrame(accepted)
    never = len(cand) - checked - too_close
    print(f"\nlistings checked: {checked}   rejected for no usable intraday years: {no_intraday}   "
          f"skipped for spacing: {too_close}   not reached: {never}")
    # Every candidate must be accounted for. If this fails, candidates were
    # dropped without being examined, which is exactly the bug that hid 91 of
    # them in the first run.
    if len(accepted) < a.n_sites and never > 0:
        print(f"WARNING: {never} candidates were never examined although the target "
              f"was not reached; the per-climate cap ({a.per_climate}) may be binding")
    if out.empty:
        sys.exit("no site has usable intraday data; PVDAQ cannot support the intraday probe")
    print(f"accepted: {len(out)} sites across {out['climate'].nunique()} climates")
    la, lo = np.radians(out["latitude"].to_numpy(float)), np.radians(out["longitude"].to_numpy(float))
    if len(out) > 1:
        dd = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
        np.fill_diagonal(dd, np.inf)
        print(f"closest pair: {dd.min():.0f} km")
    print(f"year pairs: {out['years'].value_counts().to_dict()}")
    if len(out) < 12:
        print(f"\nWARNING: {len(out)} sites is thin for a site-held-out split. Try --min-km 50, "
              f"and weigh that against the weaker independence it buys.")
    out.to_csv(a.out, index=False)
    print(f"\nwritten: {a.out}")
    print("next:  python download_sites.py --plan " + a.out)


if __name__ == "__main__":
    main()
