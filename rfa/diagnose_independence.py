"""
diagnose_independence.py -- read-only. Why did C4 find no dependent units in the
release, when the earlier probe found five attack files sharing the Normal
solve before onset (agreement 1e-6 to 1e-13)?

For every event unit against the baseline unit it prints, over the pre-onset
segment, the largest difference under three normalizations:

    mad   what C4 uses: |a-b| / (1.4826 * MAD of the channel at the physical timescale)
    rms   |a-b| / RMS of the channel (signal magnitude)
    raw   |a-b| in the channel's own units

at both the raw resolution and the physical timescale, the channel that sets
the maximum, and whether C4's prefilter (first min_overlap samples) passes.
Nothing here changes a verdict.

    python rfa\\diagnose_independence.py --descriptor rfa\\descriptors\\microgrid_release.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rfa_core as C        # noqa: E402
import rfa_spec as S        # noqa: E402


def worst(diff, scale, ch):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.nanmax(np.abs(diff) / scale[None, :], axis=0)
    r[~np.isfinite(r)] = np.nan
    if np.all(np.isnan(r)):
        return float("nan"), "-"
    j = int(np.nanargmax(r))
    return float(r[j]), ch[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptor", required=True)
    a = ap.parse_args()
    desc = json.load(open(a.descriptor))
    ds = C.load_dataset(desc, verbose=False)
    ch = ds["channels"]
    base_name = desc.get("differential", {}).get("baseline_stratum", desc.get("null_stratum", ds["strata"][0]))
    base = [u for u in ds["units"] if u["stratum"] == base_name][0]
    sc_mad = C.channel_scale(ds, desc)
    Db, yb, _ = C.decimate(base, desc, ch)
    k0 = S.CHECKS["C4"]["min_overlap"]
    tol = S.CHECKS["C4"]["identity_tol"]
    print(f"baseline {base['id']}, identity_tol {tol:g}, prefilter on first {k0} physical samples\n")
    print(f"{'unit':<24s}{'pre n':>7s} | {'mad (phys)':>11s} {'ch':<8s} | {'rms (phys)':>11s} {'ch':<8s} | "
          f"{'rms (raw)':>11s} {'ch':<8s} | {'raw units':>10s} | prefilter")
    for u in ds["units"]:
        if u is base:
            continue
        D, y, _ = C.decimate(u, desc, ch)
        o = int(np.argmax(y > 0)) if y.max() > 0 else len(y)
        L = min(o, len(Db))
        diff = D[:L] - Db[:L]
        pooled = np.vstack([Db[:L], D[:L]])
        rms_phys = np.sqrt(np.nanmean(pooled ** 2, axis=0))
        m1 = worst(diff, sc_mad, ch)
        m2 = worst(diff, rms_phys, ch)
        # raw resolution, pre-onset
        orw = int(np.argmax(u["y"] > 0)) if u["y"].max() > 0 else len(u["y"])
        Lr = min(orw, len(base["X"]))
        Xa = u["X"][:Lr].astype("float64"); Xb = base["X"][:Lr].astype("float64")
        bad = C.corrupt_mask(u, desc, ch)[:Lr] | C.corrupt_mask(base, desc, ch)[:Lr]
        dr = np.where(bad, np.nan, Xa - Xb)
        rms_raw = np.sqrt(np.nanmean(np.where(bad, np.nan, Xb) ** 2, axis=0))
        m3 = worst(dr, rms_raw, ch)
        m4 = worst(dr, np.ones(len(ch)), ch)
        head = np.nanmax(np.abs(diff[:k0] / sc_mad[None, :])) if L >= k0 else float("nan")
        pf = "pass" if head < tol else f"reject ({head:.2e})"
        print(f"{u['id']:<24s}{L:>7d} | {m1[0]:>11.3e} {m1[1]:<8s} | {m2[0]:>11.3e} {m2[1]:<8s} | "
              f"{m3[0]:>11.3e} {m3[1]:<8s} | {m4[0]:>10.3e} | {pf}")

    # ---- channel groups: which channels carry the pre-onset mismatch?
    groups = {}
    for c in ch:
        g = "ac (v,i)" if c in set(C.canon(x) for x in desc.get("ac_channels", [])) else c.rstrip("0123456789")
        groups.setdefault(g, []).append(ch.index(c))
    print("\npre-onset max |a-b| / RMS(baseline), physical timescale, by channel group")
    print(f"{'unit':<24s}" + "".join(f"{g:>12s}" for g in groups))
    for u in ds["units"]:
        if u is base:
            continue
        D, y, _ = C.decimate(u, desc, ch)
        o = int(np.argmax(y > 0)) if y.max() > 0 else len(y)
        L = min(o, len(Db))
        rms_b = np.sqrt(np.nanmean(Db[:L] ** 2, axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.nanmax(np.abs(D[:L] - Db[:L]) / rms_b[None, :], axis=0)
        print(f"{u['id']:<24s}" + "".join(f"{np.nanmax(r[idx]):>12.2e}" for idx in groups.values()))

    # ---- C8 components: pre-onset and post-onset RMS separately
    cfg = desc.get("differential")
    if cfg:
        spec = S.CHECKS["C8"]
        cols = [ch.index(c) for c in C._match(ch, cfg["channels"])]
        sc = C.channel_scale(ds, desc)[cols]
        B = Db[:, cols] / sc
        print(f"\nC8 on {', '.join(cfg['channels'])}: RMS normalized difference (pre = {spec['pre_samples']} "
              f"samples before onset; post = first N samples after onset; spec uses N = {spec['post_samples']})")
        print(f"{'unit':<24s}{'pre':>11s}{'post 20':>11s}{'post 50':>11s}{'post 100':>11s}{'ratio(spec)':>13s}")
        for u in ds["units"]:
            if u is base:
                continue
            D, y, _ = C.decimate(u, desc, ch)
            if y.max() == 0:
                continue
            o = int(np.argmax(y > 0))
            A = D[:, cols] / sc
            p0 = max(0, o - spec["pre_samples"])
            pre = np.sqrt(np.nanmean((A[p0:o] - B[p0:o]) ** 2))
            post = {N: np.sqrt(np.nanmean((A[o:o + N] - B[o:o + N]) ** 2)) for N in (20, 50, 100)}
            ratio = post[spec["post_samples"]] / pre if pre > 0 else float("inf")
            print(f"{u['id']:<24s}{pre:>11.2e}{post[20]:>11.2e}{post[50]:>11.2e}{post[100]:>11.2e}{ratio:>13.3g}")
        print("(post 50 / post 100 are shown for diagnosis only; the verdict uses the frozen post window)")

    rep = os.path.join("results", desc["name"], "report.json")
    if os.path.exists(rep):
        r = json.load(open(rep))
        print("\nC8 gate ratios from the last report:")
        for k, v in r["checks"]["C8"]["detail"].get("gate_ratio", {}).items():
            print(f"  {k:<10s} {v}")


if __name__ == "__main__":
    main()
