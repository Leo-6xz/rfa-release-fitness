"""
diagnose_rejected.py

Why were 94 of 99 intraday candidates rejected? Before concluding that PVDAQ
has only five usable independent sites, check what those systems actually hold.
A "rejection" only means no year matched the two filename patterns the
inventory knows (one file per day, or one annual file of daily aggregates). If
some systems use a third layout, the inventory calls those years "other" and
rejects them -- a filter quietly selecting the wrong population, again.

For every candidate this reports, per system: how many years fall in each kind,
the best intraday year's day count, and one example key for any "other" year so
the unknown layout can be read directly.

Usage (PowerShell, one line):
    python diagnose_rejected.py
"""

import argparse
from collections import Counter

import pandas as pd

from inventory_sites import list_site, summarise, DAILY_FILE, ANNUAL_FILE

MIN_PER_YEAR = 525_600.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=r"pvdaq\systems.csv")
    ap.add_argument("--max-interval", type=float, default=20.0)
    ap.add_argument("--min-years", type=float, default=3.0)
    ap.add_argument("--out", default=r"pvdaq\diagnose_rejected.csv")
    a = ap.parse_args()

    cat = pd.read_csv(a.catalogue)
    cat = cat.loc[:, ~cat.columns.str.startswith("Unnamed")]
    for c in ("years", "number_records", "available_sensor_channels", "latitude", "longitude"):
        cat[c] = pd.to_numeric(cat[c], errors="coerce")
    cat["est_interval_min"] = MIN_PER_YEAR / (
        cat["number_records"] / cat["years"] / cat["available_sensor_channels"])
    name = cat["system_public_name"].astype(str)
    cand = cat[~name.str.startswith("Pvoutput")
               & cat["qa_status"].astype(str).str.strip().str.lower().eq("pass")
               & (cat["years"] >= a.min_years)
               & (cat["est_interval_min"] <= a.max_interval)
               & cat["latitude"].notna() & cat["longitude"].notna()]
    print(f"examining {len(cand)} candidates ...")

    rows, other_examples = [], []
    for i, (_, r) in enumerate(cand.iterrows(), 1):
        sid = int(r["system_id"])
        try:
            objs = list_site(sid)
        except Exception as e:
            rows.append({"system_id": sid, "error": str(e)[:80]})
            continue
        summ = summarise(objs)
        kinds = Counter(x["kind"] for x in summ)
        intr = [x for x in summ if x["kind"] in ("intraday", "mixed")]
        best = max((x["days"] for x in intr), default=0)
        n330 = sum(1 for x in intr if x["days"] >= 330)
        # An example key from any object matching neither pattern.
        ex = next((k for _, k, _ in objs
                   if not DAILY_FILE.search(k) and not ANNUAL_FILE.search(k)), "")
        if ex:
            other_examples.append((sid, ex))
        rows.append({"system_id": sid, "name": str(r["system_public_name"])[:40],
                     "n_objects": len(objs), "years_listed": len(summ),
                     "intraday_years": kinds.get("intraday", 0) + kinds.get("mixed", 0),
                     "daily_years": kinds.get("daily", 0), "other_years": kinds.get("other", 0),
                     "years_ge_330_days": n330, "best_intraday_days": best})
        if i % 20 == 0:
            print(f"  {i}/{len(cand)}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print("\nwhat the candidates contain:")
    print(f"  no objects at all (empty prefix):          {int((df['n_objects'] == 0).sum())}")
    print(f"  intraday present, but never >= 330 d/yr:   "
          f"{int(((df['intraday_years'] > 0) & (df['years_ge_330_days'] == 0)).sum())}")
    print(f"  exactly one year >= 330 d (need two):       {int((df['years_ge_330_days'] == 1).sum())}")
    print(f"  two or more years >= 330 d:                 {int((df['years_ge_330_days'] >= 2).sum())}")
    print(f"  any 'other' (unrecognised) layout:          {int((df['other_years'] > 0).sum())}")
    print(f"  daily aggregates only:                      "
          f"{int(((df['daily_years'] > 0) & (df['intraday_years'] == 0)).sum())}")
    print("\ndistribution of the best intraday year's day count:")
    print(pd.cut(df["best_intraday_days"], [-1, 0, 100, 200, 300, 329, 366]).value_counts().sort_index().to_string())
    if other_examples:
        print("\nexample keys in an unrecognised layout (first 5):")
        for sid, k in other_examples[:5]:
            print(f"  {sid}: {k}")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
