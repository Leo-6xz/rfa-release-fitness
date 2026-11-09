"""
select_systems.py

Shortlist PVDAQ systems for the site-held-out leakage probe, using only the
catalogue (pvdaq/systems.csv), so nothing large is downloaded to decide.

Filters, and why each one is there
----------------------------------
* qa_status == pass. The catalogue already carries NREL's own QA verdict. A
  failing system would bring an integrity problem into the leakage comparison
  and confound the two -- they are separate checks in the protocol and should
  stay separate in the demonstration.

* years >= --min-years (default 3), so two clean contiguous years are likely to
  survive after gaps. A time-held-out split needs a full year held out to be
  seasonal rather than a slice of one season.

* Estimated sampling interval between --min-interval and --max-interval
  minutes (default 10-20, i.e. the 15-minute systems). The catalogue does not
  state resolution, so it is estimated as

      interval_min = 525,600 / (number_records / years / channels)

  which is minutes per year over readings per channel per year. It is an
  estimate -- channels are not all logged for the whole record -- and the real
  interval is checked from the data itself after download. Resolution has to
  match across sites: mixing 1-minute and 15-minute systems would make the
  site split partly a resolution split.

* available_sensor_channels >= --min-channels (default 4). The model needs AC
  power plus irradiance and temperature at minimum. Which channels actually
  exist is confirmed from each system's metadata in the next step, not assumed.

* At most --per-climate systems per Koppen-Geiger class. A site-held-out gap
  only means something if the sites differ; ten systems from Golden, CO would
  measure nothing.

Usage (PowerShell, one line):
    python select_systems.py
"""

import argparse
import sys

import pandas as pd

MIN_PER_YEAR = 525_600.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=r"pvdaq\systems.csv")
    ap.add_argument("--min-years", type=float, default=3.0)
    ap.add_argument("--min-interval", type=float, default=10.0,
                    help="minutes; lower bound of the estimated sampling interval")
    ap.add_argument("--max-interval", type=float, default=20.0,
                    help="minutes; upper bound of the estimated sampling interval")
    ap.add_argument("--min-channels", type=int, default=4)
    ap.add_argument("--per-climate", type=int, default=3)
    ap.add_argument("--max-systems", type=int, default=20)
    ap.add_argument("--out", default=r"pvdaq\shortlist.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.catalogue)
    # The catalogue has trailing empty columns; drop them.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    need = {"system_id", "years", "number_records", "dataset_size_mb",
            "available_sensor_channels", "qa_status", "kg_climate"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"catalogue is missing columns {sorted(missing)}; "
                 f"have {list(df.columns)}")

    for c in ("years", "number_records", "dataset_size_mb",
              "available_sensor_channels", "dc_capacity_kW"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    per_ch_year = df["number_records"] / df["years"] / df["available_sensor_channels"]
    df["est_interval_min"] = MIN_PER_YEAR / per_ch_year
    df["mb_per_year"] = df["dataset_size_mb"] / df["years"]

    steps = [
        ("qa_status == pass",
         df["qa_status"].astype(str).str.strip().str.lower().eq("pass")),
        (f"years >= {a.min_years:g}",
         df["years"] >= a.min_years),
        (f"interval {a.min_interval:g}-{a.max_interval:g} min",
         df["est_interval_min"].between(a.min_interval, a.max_interval)),
        (f"channels >= {a.min_channels}",
         df["available_sensor_channels"] >= a.min_channels),
    ]
    keep = pd.Series(True, index=df.index)
    print(f"catalogue: {len(df)} systems")
    for label, mask in steps:
        keep &= mask.fillna(False)
        print(f"  after {label:<24s} {int(keep.sum()):>5d}")
    cand = df[keep].copy()
    if cand.empty:
        sys.exit("\nno systems survive the filters; relax --min-years or the "
                 "interval bounds and rerun")

    # Spread across climates; within a climate, smallest download first.
    cand = cand.sort_values(["kg_climate", "mb_per_year"])
    picked = cand.groupby("kg_climate", group_keys=False).head(a.per_climate)
    picked = picked.sort_values(["kg_climate", "mb_per_year"]).head(a.max_systems)

    cols = [c for c in ("system_id", "system_public_name", "kg_climate", "tracking",
                        "type", "dc_capacity_kW", "years", "est_interval_min",
                        "mb_per_year", "available_sensor_channels",
                        "first_timestamp", "last_timestamp") if c in picked]
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 34)
    pd.set_option("display.float_format", lambda v: f"{v:.1f}")

    climates = sorted(cand["kg_climate"].dropna().astype(str).unique())
    print(f"\nclimates available after filtering: {len(climates)}  ({', '.join(climates)})")
    print(f"shortlist: {len(picked)} systems across "
          f"{picked['kg_climate'].nunique()} climates, "
          f"~{(picked['mb_per_year'] * 2).sum():.0f} MB for two years each\n")
    print(picked[cols].to_string(index=False))

    if picked["kg_climate"].nunique() < 3:
        print("\nWARNING: fewer than three climates. A site-held-out split across "
              "near-identical sites will understate the leakage gap. Raise "
              "--per-climate only if no other climates are available.")

    picked[cols].to_csv(a.out, index=False)
    print(f"\nwritten: {a.out}")
    print("Next: confirm each shortlisted system actually has AC power, irradiance "
          "and temperature channels (metadata) before downloading any data.")


if __name__ == "__main__":
    main()
