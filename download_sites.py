"""
download_sites.py

Download two complete calendar years of daily CSVs for each site chosen by
fetch_metadata.py.

Layout of the data lake (read from a real listing, not assumed):

    pvdaq/csv/pvdata/system_id=<id>/year=<Y>/month=<M>/day=<D>/
        system_<id>__date_<Y>_<MM>_<DD>.csv

one file per day. `aws s3 sync` with --exclude "*" and one --include per year
copies only the two chosen years and is restartable: rerun after an
interruption and it fetches only what is missing.

Which two years: the last two COMPLETE calendar years before each system's
last timestamp. Complete years, not "the last 730 days", so that a year-held-out
split holds out exactly one full seasonal cycle for every site.

Usage (PowerShell, one line each):
    python download_sites.py --dry-run
    python download_sites.py
"""

import argparse
import importlib.util
import os
import subprocess
import sys

import pandas as pd

BUCKET = "s3://oedi-data-lake/pvdaq/csv/pvdata"


def years_for(first_ts: str, last_ts: str, n: int = 2):
    """The last `n` complete calendar years inside [first_ts, last_ts]."""
    first = pd.to_datetime(first_ts, errors="coerce")
    last = pd.to_datetime(last_ts, errors="coerce")
    if pd.isna(first) or pd.isna(last):
        return None
    # A year is complete if the record covers Jan 1 through Dec 31 of it.
    last_full = last.year if (last.month, last.day) == (12, 31) else last.year - 1
    first_full = first.year if (first.month, first.day, first.hour, first.minute) == (1, 1, 0, 0) \
        else first.year + 1
    if last_full - first_full + 1 < n:
        return None
    return list(range(last_full - n + 1, last_full + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default=r"pvdaq\sites.csv")
    ap.add_argument("--catalogue", default=r"pvdaq\systems.csv")
    ap.add_argument("--dest", default=r"pvdaq\data")
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--plan", default=None,
                    help="CSV with system_id and years (e.g. 2018-2019) from "
                         "select_intraday.py. Years there come from the bucket listing "
                         "itself, so prefer this over --sites")
    ap.add_argument("--min-days", type=int, default=330,
                    help="per-year daily-file count below which a site is flagged")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the commands without downloading")
    a = ap.parse_args()

    # Run the AWS CLI through THIS interpreter, not as a bare "aws". pip installs
    # it on Windows as aws.cmd, a batch file: shutil.which("aws") finds it via
    # PATHEXT, so the old check passed, but subprocess cannot launch a .cmd
    # without a shell and failed with WinError 2 on the first site. The module
    # form sidesteps PATH, PATHEXT and shell quoting of the "*" patterns alike.
    if importlib.util.find_spec("awscli") is None:
        sys.exit("awscli is not installed in this environment: pip install awscli")
    aws = [sys.executable, "-m", "awscli"]

    cat = pd.read_csv(a.catalogue)
    cat = cat.loc[:, ~cat.columns.str.startswith("Unnamed")]
    rows, skipped = [], []
    if a.plan:
        pl = pd.read_csv(a.plan).merge(cat[["system_id", "dataset_size_mb", "years"]]
                                       .rename(columns={"years": "record_years_cat"}),
                                       on="system_id", how="left")
        for _, r in pl.iterrows():
            yrs = [int(y) for y in str(r["years"]).split("-")]
            yrs = list(range(yrs[0], yrs[-1] + 1))
            est_mb = r["dataset_size_mb"] / r["record_years_cat"] * len(yrs)
            rows.append((int(r["system_id"]), yrs, est_mb))
    else:
        sites = pd.read_csv(a.sites)
        plan = sites[["system_id"]].merge(
            cat[["system_id", "first_timestamp", "last_timestamp", "dataset_size_mb", "years"]],
            on="system_id", how="left")
        for _, r in plan.iterrows():
            yrs = years_for(r["first_timestamp"], r["last_timestamp"], a.years)
            if yrs is None:
                skipped.append(int(r["system_id"]))
                continue
            est_mb = r["dataset_size_mb"] / r["years"] * a.years
            rows.append((int(r["system_id"]), yrs, est_mb))

    total = sum(mb for _, _, mb in rows)
    print(f"{len(rows)} sites to download, ~{total:.0f} MB (estimate from the whole record), "
          f"~{sum(365 * len(y) for _, y, _ in rows):,} daily files expected")
    if skipped:
        print(f"skipped (fewer than {a.years} complete calendar years): {skipped}")

    short = []
    for i, (sid, yrs, mb) in enumerate(rows, 1):
        dest = os.path.join(a.dest, f"system_id={sid}")
        cmd = aws + ["s3", "sync", "--no-sign-request", "--only-show-errors",
               f"{BUCKET}/system_id={sid}/", dest, "--exclude", "*"]
        for y in yrs:
            cmd += ["--include", f"year={y}/*"]
        print(f"[{i}/{len(rows)}] system {sid}  years {yrs[0]}-{yrs[-1]}  ~{mb:.0f} MB")
        if a.dry_run:
            print("   " + " ".join(cmd))
            continue
        os.makedirs(dest, exist_ok=True)
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"   WARNING: awscli exited {res.returncode} for {sid}; rerun to resume")
        # Verify rather than report. The first run printed "2 files on disk" for
        # every site and carried on; the count is only useful if it is CHECKED
        # against what the plan requires.
        for y in yrs:
            yd = os.path.join(dest, f"year={y}")
            days = sum(1 for _, _, fs in os.walk(yd) for f in fs if "__date_" in f)
            flag = "" if days >= a.min_days else f"   <-- SHORT: needs >= {a.min_days}"
            if flag:
                short.append((sid, y, days))
            print(f"   {y}: {days} daily files{flag}")

    if not a.dry_run:
        if short:
            print(f"\n{len(short)} site-years are short of {a.min_days} daily files:")
            for sid, y, dd in short:
                print(f"   {sid} {y}: {dd}")
        else:
            print(f"\nall {sum(len(y) for _, y, _ in rows)} site-years have >= {a.min_days} daily files")

    # Record exactly which years were taken, so the probe and the paper can
    # state it without reconstructing it from folder names.
    pd.DataFrame([{"system_id": s, "years": "-".join(map(str, y))} for s, y, _ in rows]) \
        .to_csv(os.path.join(os.path.dirname(a.sites) or ".", "download_plan.csv"), index=False)
    print("\nplan written to download_plan.csv")


if __name__ == "__main__":
    main()
