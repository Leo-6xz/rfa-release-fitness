"""
fetch_metadata.py

Download the small per-system metadata JSON (~2 KB each) for every catalogue
system that passes the basic filters, read what each one actually measures, and
choose the sites for the leakage probe.

Why this replaces the channel-count filter in select_systems.py
---------------------------------------------------------------
That filter cut 793 fifteen-minute systems down to 14, and the 14 were the wrong
kind of diverse: three were buildings on one Las Vegas campus, and two were
2023 Solar Data Prize systems (9069 alone is ~10 GB, and prize systems are stored
in a different layout). What the channel filter was really selecting for was
irradiance and temperature sensors. Most PVDAQ systems report AC power only.

So the probe's task changes to one that needs only AC power: short-horizon
forecasting of AC power from its own recent history plus time-of-day and
day-of-year. That opens hundreds of sites across many climates, and it makes
the leakage point cleanly -- a random split of an autocorrelated 15-minute
series puts adjacent intervals on both sides of the split.

Two corrections this script makes that the catalogue alone cannot:

* It groups by SITE, not by system. Several systems can share one site_id and
  therefore one weather record; holding out one while training on its
  neighbour is not a held-out site. Choosing the unit of independence is itself
  one of the protocol's checks.
* It excludes Solar Data Prize systems (data_prize flag), which use a separate
  storage layout and are far larger than the rest.

Usage (PowerShell, one line each):
    python fetch_metadata.py
    python fetch_metadata.py --n-sites 30 --per-climate 4
"""

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

BASE = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/system_metadata"
MIN_PER_YEAR = 525_600.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; all angles in radians."""
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def fetch(sid: int, outdir: str) -> str | None:
    """Download one metadata JSON, skipping it if already cached."""
    path = os.path.join(outdir, f"{sid}_system_metadata.json")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    try:
        with urllib.request.urlopen(f"{BASE}/{sid}_system_metadata.json", timeout=30) as r:
            data = r.read()
    except Exception:
        return None
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def parse(path: str) -> dict:
    """What a system is and what it measures, read from its own metadata."""
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    system, site = m.get("System", {}), m.get("Site", {})
    metrics = m.get("Metrics", {}) or {}
    names = [str(v.get("common_name", "")).lower() for v in metrics.values()]
    live = [str(v.get("common_name", "")).lower() for v in metrics.values()
            if not v.get("offline", False)]
    return {
        "system_id": int(system.get("system_id")),
        "site_id": str(system.get("site_id", site.get("site_id", ""))),
        "data_prize": str(system.get("data_prize", "f")).lower() in ("t", "true", "1"),
        "climate": site.get("climate_type"),
        "n_metrics": len(names),
        "has_ac_power": any(n == "ac power" for n in live),
        "has_irradiance": any("irradiance" in n for n in live),
        "has_temperature": any("temperature" in n for n in live),
        "metric_names": "; ".join(sorted(set(names))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=r"pvdaq\systems.csv")
    ap.add_argument("--meta-dir", default=r"pvdaq\meta")
    ap.add_argument("--min-years", type=float, default=3.0)
    ap.add_argument("--min-interval", type=float, default=10.0)
    ap.add_argument("--max-interval", type=float, default=20.0)
    ap.add_argument("--n-sites", type=int, default=30)
    ap.add_argument("--per-climate", type=int, default=4)
    ap.add_argument("--min-km", type=float, default=100.0,
                    help="minimum distance between any two picked systems. At 100 km "
                         "two sites rarely share a cloud field at 15-minute resolution")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=r"pvdaq\sites.csv")
    a = ap.parse_args()
    os.makedirs(a.meta_dir, exist_ok=True)

    cat = pd.read_csv(a.catalogue)
    cat = cat.loc[:, ~cat.columns.str.startswith("Unnamed")]
    for c in ("years", "number_records", "dataset_size_mb", "available_sensor_channels"):
        cat[c] = pd.to_numeric(cat[c], errors="coerce")
    cat["est_interval_min"] = MIN_PER_YEAR / (
        cat["number_records"] / cat["years"] / cat["available_sensor_channels"])
    cat["mb_per_year"] = cat["dataset_size_mb"] / cat["years"]

    base = cat[
        cat["qa_status"].astype(str).str.strip().str.lower().eq("pass")
        & (cat["years"] >= a.min_years)
        & cat["est_interval_min"].between(a.min_interval, a.max_interval)
    ].copy()
    ids = base["system_id"].astype(int).tolist()
    print(f"catalogue {len(cat)} -> {len(ids)} pass QA, >= {a.min_years:g} years, "
          f"{a.min_interval:g}-{a.max_interval:g} min")
    print(f"fetching {len(ids)} metadata files (~{len(ids) * 2 / 1024:.1f} MB, cached after the first run) ...")

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        paths = list(ex.map(lambda s: fetch(s, a.meta_dir), ids))
    got = [p for p in paths if p]
    if len(got) < len(ids):
        print(f"  {len(ids) - len(got)} metadata files could not be fetched and are skipped")
    if not got:
        sys.exit("no metadata could be fetched; check the network and rerun")

    meta = pd.DataFrame([parse(p) for p in got])
    df = base.merge(meta, on="system_id", how="inner")

    print(f"\nwith metadata:              {len(df):>5d}")
    df = df[~df["data_prize"]]
    print(f"  not a Solar Data Prize:   {len(df):>5d}")
    df = df[df["has_ac_power"]]
    print(f"  reports AC power:         {len(df):>5d}")
    print(f"  distinct site_id values:  {df['site_id'].nunique():>5d}")

    # site_id IS NOT A UNIT OF INDEPENDENCE. The first version of this script
    # grouped by it and reported "0 sites with >1 system", which was true and
    # meant nothing: the three Andre Agassi buildings on one Las Vegas campus
    # carry three different site_ids, and the first shortlist put four Fort
    # Lauderdale homes, three Honolulu homes and three Phoenix homes into a
    # "site-held-out" split. Systems a few km apart share their weather, so
    # holding one out while training on its neighbour is not a held-out site.
    # Independence is decided by DISTANCE, not by an identifier.
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
    lat = np.radians(df["latitude"].to_numpy(float))
    lon = np.radians(df["longitude"].to_numpy(float))
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(d, np.inf)
    near = (d < a.min_km).sum(axis=1)
    print(f"  with another eligible system within {a.min_km:g} km: "
          f"{int((near > 0).sum()):>5d}  (site_id cannot see these)")

    clim_col = "climate" if df["climate"].notna().any() else "kg_climate"
    # Within each climate: prefer irradiance + temperature (keeps a physics-input
    # task possible later), then the longest record. Download size is NOT a
    # ranking key any more: ranking by it selected only the smallest residential
    # systems and made the sample a function of file size.
    df = df.sort_values([clim_col, "has_irradiance", "has_temperature", "years"],
                        ascending=[True, False, False, False])
    queues = {c: g for c, g in df.groupby(clim_col)}

    # Greedy round-robin over climates, accepting a system only if it is at
    # least --min-km from every system already picked.
    picked_idx, lat_p, lon_p, per_clim = [], [], [], {c: 0 for c in queues}
    progress = True
    while len(picked_idx) < a.n_sites and progress:
        progress = False
        for c, g in queues.items():
            if len(picked_idx) >= a.n_sites or per_clim[c] >= a.per_climate:
                continue
            for idx, row in g.iterrows():
                if idx in picked_idx:
                    continue
                la, lo = np.radians(row["latitude"]), np.radians(row["longitude"])
                if lat_p and haversine_km(la, lo, np.array(lat_p), np.array(lon_p)).min() < a.min_km:
                    continue
                picked_idx.append(idx); lat_p.append(la); lon_p.append(lo)
                per_clim[c] += 1; progress = True
                break
    per_site = df
    picked = df.loc[picked_idx].sort_values([clim_col, "system_id"])
    if len(picked) < a.n_sites:
        print(f"\nonly {len(picked)} systems satisfy the {a.min_km:g} km spacing with at most "
              f"{a.per_climate} per climate; lower --min-km or raise --per-climate if that is too few")

    cols = ["system_id", "site_id", clim_col, "system_public_name", "years",
            "est_interval_min", "mb_per_year", "has_irradiance", "has_temperature",
            "latitude", "longitude"]
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 32)
    pd.set_option("display.float_format", lambda v: f"{v:.1f}")
    print(f"\nclimates among eligible sites: {per_site[clim_col].nunique()}   "
          f"picked: {len(picked)} sites across {picked[clim_col].nunique()} climates, "
          f"~{(picked['mb_per_year'] * 2).sum():.0f} MB for two years each")
    if len(picked) > 1:
        la, lo = np.radians(picked["latitude"].to_numpy(float)), np.radians(picked["longitude"].to_numpy(float))
        dd = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
        np.fill_diagonal(dd, np.inf)
        print(f"closest pair of picked systems: {dd.min():.0f} km apart\n")
    print(picked[cols].to_string(index=False))

    rich = df[df["has_irradiance"] & df["has_temperature"]]
    print(f"\nsites with AC power + irradiance + temperature: {len(rich)} "
          f"(enough for a second, physics-input task only if this is >= ~8)")

    picked[cols + ["metric_names"]].to_csv(a.out, index=False)
    per_site[cols + ["metric_names"]].to_csv(a.out.replace(".csv", "_all.csv"), index=False)
    print(f"\nwritten: {a.out}  and  {a.out.replace('.csv', '_all.csv')}")


if __name__ == "__main__":
    main()
