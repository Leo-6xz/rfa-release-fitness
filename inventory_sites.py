"""
inventory_sites.py

List what actually exists in the data lake for each chosen site, year by year,
BEFORE deciding what to download. Nothing is downloaded.

Why this step exists
--------------------
download_sites.py chose years from the catalogue and assumed one file per day.
The catalogue describes each system's WHOLE record: its record count, and
therefore the 10-20 minute interval estimate, averages over every year. For
many systems the recent years are stored as a single annual file of DAILY
aggregates (daily max, daily mean, daily energy), so the two years that were
downloaded held 365 rows each and no intraday data at all. The catalogue's
first and last timestamps also cannot see gaps inside a record: system 10003
had no 2023 file and a 2024 file starting on 5 June.

So the listing is the source of truth. For every site and year this reports:

    kind      intraday  -- one file per day (system_<id>__date_<Y>_<MM>_<DD>.csv)
              daily     -- one annual file of daily aggregates (<id>_ac_<start>_<end>.csv)
              mixed     -- both present
    files     number of objects
    days      distinct days covered by intraday files
    MB        total size

and then picks, per site, the latest two CONSECUTIVE years that are intraday
with at least --min-days days each.

Usage (PowerShell, one line):
    python inventory_sites.py
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict

import pandas as pd

PREFIX = "s3://oedi-data-lake/pvdaq/csv/pvdata"
DAILY_FILE = re.compile(r"__date_(\d{4})_(\d{2})_(\d{2})\.csv$")
ANNUAL_FILE = re.compile(r"_ac_(\d{8})_(\d{8})\.csv$")
YEAR_DIR = re.compile(r"/year=(\d{4})/")


def list_site(sid: int):
    """Return (year, key, size) for every object under the system's prefix."""
    cmd = [sys.executable, "-m", "awscli", "s3", "ls", "--no-sign-request",
           "--recursive", f"{PREFIX}/system_id={sid}/"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode not in (0, 1):          # 1 = empty prefix
        raise RuntimeError(res.stderr.strip()[:300])
    out = []
    for line in res.stdout.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        size, key = int(parts[2]), parts[3]
        m = YEAR_DIR.search(key)
        if m:
            out.append((int(m.group(1)), key, size))
    return out


def summarise(objs):
    by_year = defaultdict(lambda: {"intraday": 0, "annual": 0, "days": set(), "bytes": 0})
    for year, key, size in objs:
        y = by_year[year]
        y["bytes"] += size
        m = DAILY_FILE.search(key)
        if m:
            y["intraday"] += 1
            y["days"].add(key)
        elif ANNUAL_FILE.search(key):
            y["annual"] += 1
    rows = []
    for year, y in sorted(by_year.items()):
        kind = ("mixed" if y["intraday"] and y["annual"]
                else "intraday" if y["intraday"] else "daily" if y["annual"] else "other")
        rows.append({"year": year, "kind": kind, "files": y["intraday"] + y["annual"],
                     "days": len(y["days"]), "MB": y["bytes"] / 1e6})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default=r"pvdaq\sites.csv")
    ap.add_argument("--min-days", type=int, default=330,
                    help="intraday days a year must have to count as usable")
    ap.add_argument("--n-years", type=int, default=2)
    ap.add_argument("--out", default=r"pvdaq\inventory.csv")
    a = ap.parse_args()

    sites = pd.read_csv(a.sites)["system_id"].astype(int).tolist()
    table, plan = [], []
    for i, sid in enumerate(sites, 1):
        try:
            rows = summarise(list_site(sid))
        except Exception as e:
            print(f"[{i}/{len(sites)}] {sid}: listing failed: {e}")
            continue
        for r in rows:
            table.append({"system_id": sid, **r})
        good = [r["year"] for r in rows
                if r["kind"] in ("intraday", "mixed") and r["days"] >= a.min_days]
        # latest run of n consecutive usable years
        pick = None
        for y in sorted(good, reverse=True):
            run = list(range(y - a.n_years + 1, y + 1))
            if all(v in good for v in run):
                pick = run
                break
        intr = [r["year"] for r in rows if r["kind"] in ("intraday", "mixed")]
        span = f"{min(intr)}-{max(intr)}" if intr else "none"
        print(f"[{i}/{len(sites)}] {sid}: intraday years {span:<10s} "
              f"usable (>= {a.min_days} d): {len(good):>2d}   pick: {pick}")
        plan.append({"system_id": sid, "years": "-".join(map(str, pick)) if pick else ""})

    inv = pd.DataFrame(table)
    inv.to_csv(a.out, index=False)
    pl = pd.DataFrame(plan)
    ok = pl[pl["years"] != ""]
    pl.to_csv(a.out.replace("inventory", "download_plan_intraday"), index=False)

    print(f"\nsites with {a.n_years} consecutive usable intraday years: {len(ok)} of {len(pl)}")
    if not ok.empty:
        common = ok["years"].value_counts()
        print("most common year pairs:")
        print(common.head(5).to_string())
    if not inv.empty:
        print("\nstorage kinds across all site-years:")
        print(inv["kind"].value_counts().to_string())
    print(f"\nwritten: {a.out}  and  {a.out.replace('inventory', 'download_plan_intraday')}")


if __name__ == "__main__":
    main()
