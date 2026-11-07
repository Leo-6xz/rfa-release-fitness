"""
rfa_core.py -- loaders and the eight release-fitness checks.

A dataset is described by a JSON descriptor (see descriptors/). Two adapters:

  folder_per_stratum   <root>/<stratum>/<file>.csv, one file = one declared unit
                       (the IEEE DataPort microgrid release, the synthetic corpus,
                       and any labelled time-series release with that layout)
  pvdaq_catalogue      catalogue + bucket listings cached by pvdaq_funnel.py;
                       units are systems, independence is spatial

Every threshold comes from rfa_spec.py. Nothing here decides a verdict on its
own numbers.
"""

from __future__ import annotations

import copy
import glob
import json
import math
import os
import warnings

import numpy as np
import pandas as pd

import rfa_spec as S

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================================== loading

META = {"time", "t", "label", "attack_state", "y_bin", "y_multi", "phase",
        "scenario", "run_id", "source_file"}


def canon(c: str) -> str:
    k = str(c).strip().lower().replace(" ", "_")
    k = k.replace("pdg", "p_dg").replace("qdg", "q_dg").replace("fdg", "f_dg")
    for a in ("p", "q", "f"):
        k = k.replace(f"{a}__dg", f"{a}_dg")
    return k


def _match(channels, patterns):
    """Channels whose canonical name equals or starts with any pattern."""
    pats = [canon(p) for p in patterns]
    return [c for c in channels if any(c == p or c.startswith(p) for p in pats)]


def load_folder_dataset(desc: dict, verbose: bool = True) -> dict:
    root = desc["root"]
    tcol, lcol = canon(desc.get("time_column", "time")), canon(desc.get("label_column", "label"))
    strata = desc["strata"]
    declared = [canon(c) for c in desc["channels"]] if desc.get("channels") not in (None, "auto") else None
    units, channels = [], None
    for si, stratum in enumerate(strata):
        files = sorted(glob.glob(os.path.join(root, stratum, desc.get("file_glob", "*.csv"))))
        if desc.get("max_files_per_stratum"):
            files = files[: desc["max_files_per_stratum"]]
        if not files:
            raise FileNotFoundError(f"no files under {os.path.join(root, stratum)}")
        for path in files:
            df = pd.read_csv(path, low_memory=False)
            df.columns = [canon(c) for c in df.columns]
            if channels is None:
                avail = [c for c in df.columns if c not in META and c not in (tcol, lcol)]
                channels = [c for c in declared if c in df.columns] if declared else avail
            missing = [c for c in channels if c not in df.columns]
            num = df.reindex(columns=channels).apply(pd.to_numeric, errors="coerce").to_numpy("float64")
            maxabs = np.nanmax(np.where(np.isfinite(num), np.abs(num), np.nan), axis=0)
            with np.errstate(over="ignore"):
                X = num.astype("float32")          # beyond float32 range -> inf, i.e. still non-finite
            t = pd.to_numeric(df[tcol], errors="coerce").to_numpy("float64") if tcol in df.columns \
                else np.arange(len(df)) * float(desc["sample_interval_s"])
            y = (pd.to_numeric(df[lcol], errors="coerce").fillna(0).clip(0, 1).to_numpy("int8")
                 if lcol in df.columns else np.zeros(len(df), "int8"))
            units.append({"id": f"{stratum}/{os.path.basename(path)}", "stratum": stratum,
                          "stratum_idx": si, "X": X, "t": t, "y": y,
                          "maxabs": maxabs, "missing_channels": missing})
            if verbose:
                print(f"  loaded {units[-1]['id']:<40s} {len(df):>9,} rows")
            del df, num
    ds = {"kind": "timeseries", "units": units, "channels": channels, "strata": strata,
          "declared_channels": declared}
    return ds


def load_dataset(desc: dict, verbose: bool = True) -> dict:
    if desc["adapter"] == "folder_per_stratum":
        return load_folder_dataset(desc, verbose)
    if desc["adapter"] == "pvdaq_catalogue":
        return load_pvdaq_catalogue(desc, verbose)
    raise ValueError(f"unknown adapter {desc['adapter']}")


# =========================================================================== helpers

def bound_for(desc, ch):
    b = desc.get("bounds", {})
    return float(b.get(ch, b.get("default", S.CHECKS["C1"]["default_bound"])))


def corrupt_mask(unit, desc, channels):
    X = unit["X"]
    bounds = np.array([bound_for(desc, c) for c in channels], dtype="float64")
    return ~np.isfinite(X) | (np.abs(X) > bounds[None, :])


def factor_for(unit, desc):
    dt = observed_dt(unit)
    return max(1, int(round(float(desc["physical_dt_s"]) / dt))) if dt and dt > 0 else 1


def observed_dt(unit):
    t = unit["t"]
    if len(t) < 2:
        return None
    d = np.diff(t[np.isfinite(t)])
    d = d[d > 0]
    return float(np.median(d)) if len(d) else None


def decimate(unit, desc, channels):
    """Block-average to the physical timescale. Corrupt values are masked first,
    AC waveform channels are aggregated by RMS, labels by max."""
    key = ("_dec", float(desc["physical_dt_s"]))
    if key in unit:
        return unit[key]
    f = factor_for(unit, desc)
    X = unit["X"].astype("float64")
    X[corrupt_mask(unit, desc, channels)] = np.nan
    n = (len(X) // f) * f
    if n == 0:
        out = (np.empty((0, len(channels))), np.empty(0, "int8"), np.empty((0, len(channels))))
        unit[key] = out
        return out
    Xb = X[:n].reshape(n // f, f, len(channels))
    ac = set(canon(c) for c in desc.get("ac_channels", []))
    D = np.nanmean(Xb, axis=1)
    for j, c in enumerate(channels):
        if c in ac:
            D[:, j] = np.sqrt(np.nanmean(Xb[:, :, j] ** 2, axis=1))
    cm = corrupt_mask(unit, desc, channels)[:n].reshape(n // f, f, len(channels)).mean(axis=1)
    y = unit["y"][:n].reshape(n // f, f).max(axis=1)
    unit[key] = (D, y, cm)
    return unit[key]


def channel_scale(ds, desc):
    """Robust per-channel scale over all units at the physical timescale."""
    ch = ds["channels"]
    pooled = np.concatenate([decimate(u, desc, ch)[0] for u in ds["units"]], axis=0)
    med = np.nanmedian(pooled, axis=0)
    mad = np.nanmedian(np.abs(pooled - med), axis=0)
    sd = np.nanstd(pooled, axis=0)
    sc = np.where(mad > 0, 1.4826 * mad, sd)
    sc[~np.isfinite(sc) | (sc <= 0)] = 1.0
    return sc


def pre_event(D, y):
    k = int(np.argmax(y > 0)) if y.max() > 0 else len(y)
    return D[:k]


def _mis_exact(adj):
    """Exact maximum independent set size of ONE small connected graph."""
    n = len(adj)
    nbrs = [set(np.flatnonzero(adj[i])) for i in range(n)]
    best = [0]

    def rec(cands, size):
        if size + len(cands) <= best[0]:
            return
        if not cands:
            best[0] = max(best[0], size)
            return
        free = {v for v in cands if not (nbrs[v] & cands)}
        if free:
            rec(cands - free, size + len(free))
            return
        v = max(cands, key=lambda u: len(nbrs[u] & cands))
        rec(cands - {v} - nbrs[v], size + 1)
        rec(cands - {v}, size)

    rec(set(range(n)), 0)
    return best[0]


def max_independent(adj):
    """Exact maximum independent set size. The graph is split into connected
    components first and each is solved separately; without this, many small
    dependent pairs (e.g. duplicated units) make the search exponential in the
    number of pairs (v1.0.0 hung on 84 duplicates)."""
    adj = np.asarray(adj, bool)
    if adj.size == 0:
        return 0
    comp = components(adj)
    total = 0
    for c in np.unique(comp):
        idx = np.flatnonzero(comp == c)
        if len(idx) == 1:
            total += 1
        elif len(idx) == 2:
            total += 1                      # two units, dependent on each other
        else:
            total += _mis_exact(adj[np.ix_(idx, idx)])
    return total


def components(adj):
    n = len(adj)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(*np.nonzero(np.triu(adj, 1))):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots])


def haversine_km(lat1, lon1, lat2, lon2):
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# =========================================================================== checks

def c1_integrity(ds, desc, channels=None, cid="C1"):
    spec = S.CHECKS["C1"]
    channels = channels or ds["channels"]
    idx = [ds["channels"].index(c) for c in channels]
    rows = bad_rows = 0
    per_ch = np.zeros(len(channels))
    per_stratum = {}
    for u in ds["units"]:
        cm = corrupt_mask(u, desc, ds["channels"])[:, idx]
        r = cm.any(axis=1)
        rows += len(r); bad_rows += int(r.sum())
        per_ch += cm.sum(axis=0)
        a = per_stratum.setdefault(u["stratum"], [0, 0, 0])
        a[0] += len(r); a[1] += int(r.sum()); a[2] += int(r.any())
    frac = bad_rows / rows if rows else None
    maxabs = np.nanmax(np.vstack([u["maxabs"][idx] for u in ds["units"]]), axis=0)
    worst = sorted(((channels[j], int(per_ch[j]), float(maxabs[j])) for j in range(len(channels))
                    if per_ch[j] > 0), key=lambda x: -x[1])
    return {"id": cid, "status": S.band_max(frac, spec["pass_max"], spec["cond_max"]),
            "metric": frac, "metric_name": "fraction of rows corrupted",
            "detail": {"rows": rows, "corrupt_rows": bad_rows,
                       "units_affected": sum(v[2] for v in per_stratum.values()),
                       "units": len(ds["units"]),
                       "per_stratum_fraction": {k: v[1] / v[0] for k, v in per_stratum.items()},
                       "channels_affected": worst[:12],
                       "max_abs_value": float(np.nanmax(maxabs)) if np.isfinite(maxabs).any() else None}}


def c2_metadata(ds, desc):
    spec = S.CHECKS["C2"]
    declared = float(desc["sample_interval_s"])
    ratios = [observed_dt(u) / declared for u in ds["units"] if observed_dt(u)]
    bad_interval = [r for r in ratios if abs(r - 1.0) > spec["interval_tol"]]
    missing = sorted({c for u in ds["units"] for c in u["missing_channels"]})
    empty = [c for j, c in enumerate(ds["channels"])
             if all(np.isnan(u["X"][:, j]).all() for u in ds["units"])]
    ok = not bad_interval and not missing and not empty
    return {"id": "C2", "status": "PASS" if ok else "FAIL",
            "metric": float(np.median(ratios)) if ratios else None,
            "metric_name": "observed / declared sampling interval (median over units)",
            "detail": {"declared_interval_s": declared,
                       "units_outside_tolerance": len(bad_interval), "units": len(ratios),
                       "ratio_range": [float(min(ratios)), float(max(ratios))] if ratios else None,
                       "declared_channels_missing": missing, "channels_entirely_empty": empty}}


def dependence_graph(ds, desc):
    spec = S.CHECKS["C4"]
    ch = ds["channels"]
    sc = channel_scale(ds, desc)
    pres = []
    for u in ds["units"]:
        D, y, _ = decimate(u, desc, ch)
        pres.append(pre_event(D, y) / sc)
    n = len(pres)
    adj = np.zeros((n, n), bool)
    dmin = np.full((n, n), np.nan)
    k0 = spec["min_overlap"]
    heads = [p[:k0] for p in pres]
    for i in range(n):
        for j in range(i + 1, n):
            L = min(len(pres[i]), len(pres[j]))
            if L < k0:
                continue
            h = np.nanmax(np.abs(heads[i] - heads[j]))
            if not (h < spec["identity_tol"]):       # cheap prefilter on the first samples
                continue
            d = np.nanmax(np.abs(pres[i][:L] - pres[j][:L]))
            dmin[i, j] = dmin[j, i] = d
            if d < spec["identity_tol"]:
                adj[i, j] = adj[j, i] = True
    comparable = sum(1 for p in pres if len(p) >= k0)
    return adj, dmin, comparable, sc


def c3_c4_timeseries(ds, desc):
    s3, s4 = S.CHECKS["C3"], S.CHECKS["C4"]
    adj, dmin, comparable, _ = dependence_graph(ds, desc)
    n = len(ds["units"])
    dep_frac = float(adj.any(axis=1).mean()) if n else None
    pairs = [(ds["units"][i]["id"], ds["units"][j]["id"], float(dmin[i, j]))
             for i, j in zip(*np.nonzero(np.triu(adj, 1)))]
    strata = [u["stratum"] for u in ds["units"]]
    per = {}
    for s in ds["strata"]:
        idx = [i for i, x in enumerate(strata) if x == s]
        per[s] = {"declared_units": len(idx),
                  "independent_units": max_independent(adj[np.ix_(idx, idx)]) if idx else 0}
    labelled = any(u["y"].max() > 0 for u in ds["units"])
    m = min(v["independent_units"] for v in per.values()) if labelled else max_independent(adj)
    ds["_clusters"] = components(adj)
    c3 = {"id": "C3", "status": S.band_min(m, s3["pass_min"], s3["cond_min"]), "metric": m,
          "metric_name": "min independent units per stratum" if labelled else "independent units",
          "detail": {"per_stratum": per, "declared_units": n,
                     "independent_units_overall": max_independent(adj)}}
    c4 = {"id": "C4", "status": S.band_max(dep_frac, s4["pass_max"], s4["cond_max"]),
          "metric": dep_frac, "metric_name": "fraction of units with a dependent partner",
          "detail": {"dependent_pairs": len(pairs), "examples": pairs[:10],
                     "units_comparable": comparable, "identity_tol": s4["identity_tol"],
                     "test": "identical pre-event trajectory at the physical timescale"}}
    return c3, c4


# ---------------------------------------------------------------- windows and model

def windows(ds, desc):
    """Window features (signal) and artefact features, one row per window."""
    key = "_windows"
    if key in ds:
        return ds[key]
    ch = ds["channels"]
    W, Sd = int(desc.get("window", 20)), int(desc.get("stride", 5))
    null = desc.get("null_stratum", ds["strata"][0])
    sig, art, meta = [], [], []
    clusters = ds.get("_clusters")
    for ui, u in enumerate(ds["units"]):
        D, y, cm = decimate(u, desc, ch)
        n = len(D)
        if n < W:
            continue
        starts = np.arange(0, n - W + 1, Sd)
        V = np.lib.stride_tricks.sliding_window_view(D, W, axis=0)[starts]      # (nw, C, W)
        Vc = np.lib.stride_tricks.sliding_window_view(cm, W, axis=0)[starts]
        mean = np.nanmean(V, axis=2)
        std = np.nanstd(V, axis=2)
        x = np.arange(W) - (W - 1) / 2.0
        slope = np.nansum(np.nan_to_num(V - mean[:, :, None]) * x, axis=2) / (x @ x)
        rng = np.nanmax(V, axis=2) - np.nanmin(V, axis=2)
        sig.append(np.hstack([mean, std, slope, rng]))
        art.append(np.hstack([(starts / max(n - W, 1))[:, None],
                              np.full((len(starts), 1), n, float),
                              Vc.mean(axis=2)]))
        yw = np.lib.stride_tricks.sliding_window_view(y, W)[starts].max(axis=1)
        cls = 0 if u["stratum"] == null else ds["strata"].index(u["stratum"])
        meta.append(pd.DataFrame({"unit": ui, "stratum": u["stratum"], "start": starts,
                                  "y_bin": yw, "y": np.where(yw > 0, cls, 0),
                                  "cluster": clusters[ui] if clusters is not None else ui}))
    out = (np.vstack(sig), np.vstack(art), pd.concat(meta, ignore_index=True))
    ds[key] = out
    return out


def model(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = S.MODEL
    return HistGradientBoostingClassifier(max_iter=m["max_iter"], learning_rate=m["learning_rate"],
                                          max_leaf_nodes=m["max_leaf_nodes"],
                                          class_weight=m["class_weight"],
                                          early_stopping=m["early_stopping"], random_state=seed)


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    labels = np.unique(y_true)
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def split_random(meta, seed, frac):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(meta))
    y = meta["y"].to_numpy()
    strat = y if np.bincount(y).min() >= 2 else None
    tr, te = train_test_split(idx, test_size=frac, random_state=seed, stratify=strat)
    return tr, te


def split_grouped(meta, seed, frac, min_clusters):
    """Hold out whole independence clusters, chosen per stratum."""
    rng = np.random.default_rng(seed)
    per = meta.groupby("stratum")["cluster"].unique()
    if any(len(v) < min_clusters for v in per.values):
        return None
    test_cl = set()
    for s, cl in per.items():
        cl = [c for c in cl if c not in test_cl]
        k = max(1, int(math.ceil(frac * len(per[s]))))
        test_cl.update(rng.permutation(cl)[:k].tolist())
    te = meta.index[meta["cluster"].isin(test_cl)].to_numpy()
    tr = meta.index[~meta["cluster"].isin(test_cl)].to_numpy()
    if len(np.unique(meta.loc[tr, "y"])) < 2:
        return None
    return tr, te


def split_block(meta, seed, frac, W, Sd, n_blocks):
    """Contiguous purged blocks: every unit is cut into n_blocks contiguous blocks
    of windows, a random `frac` of the blocks is held out, and training windows
    that overlap a held-out block are purged."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    purge = int(math.ceil(W / Sd))
    for _, g in meta.groupby("unit"):
        idx = g.sort_values("start").index.to_numpy()
        b = np.minimum((np.arange(len(idx)) * n_blocks) // max(len(idx), 1), n_blocks - 1)
        k = max(1, int(round(frac * n_blocks)))
        test_b = set(rng.choice(n_blocks, size=k, replace=False).tolist())
        is_te = np.array([x in test_b for x in b])
        near = np.zeros(len(idx), bool)
        for p in np.flatnonzero(is_te):
            near[max(0, p - purge + 1):p] = True          # train windows overlapping from the left
            near[p + 1:p + purge] = True                   # and from the right
        te.extend(idx[is_te])
        tr.extend(idx[~is_te & ~near])
    return np.array(tr), np.array(te)


def fit_score(X, y, tr, te, seed):
    if len(np.unique(y[tr])) < 2 or len(te) == 0:
        return None
    m = model(seed).fit(X[tr], y[tr])
    return macro_f1(y[te], m.predict(X[te]))


def c5_split(ds, desc, repeats=None):
    spec = S.CHECKS["C5"]
    repeats = repeats or spec["repeats"]
    Xs, _, meta = windows(ds, desc)
    y = meta["y"].to_numpy()
    if len(np.unique(y)) < 2:
        na = {"id": "C5", "status": "NA", "metric": None, "metric_name": "no labels", "detail": {}}
        return na, {"id": "C5g", "status": "NA", "metric": None, "metric_name": "", "detail": {}}
    W, Sd = int(desc.get("window", 20)), int(desc.get("stride", 5))
    res = {"random": [], "grouped": [], "block": []}
    for r in range(repeats):
        tr, te = split_random(meta, r, spec["test_fraction"])
        res["random"].append(fit_score(Xs, y, tr, te, r))
        g = split_grouped(meta, r, spec["test_fraction"], spec["min_clusters_per_stratum"])
        if g is not None:
            res["grouped"].append(fit_score(Xs, y, *g, r))
        tr, te = split_block(meta, r, spec["test_fraction"], W, Sd, spec["block_count"])
        res["block"].append(fit_score(Xs, y, tr, te, r))
    summ = {k: (float(np.mean([v for v in vs if v is not None])),
                float(np.std([v for v in vs if v is not None])))
            for k, vs in res.items() if any(v is not None for v in vs)}
    ref = "grouped" if "grouped" in summ else "block"
    gap = summ["random"][0] - summ[ref][0] if ref in summ and "random" in summ else None
    c5 = {"id": "C5", "status": S.band_max(gap, spec["pass_max"], spec["cond_max"]),
          "metric": gap, "metric_name": f"macro-F1 random minus {ref}",
          "detail": {"macro_f1": summ, "reference_protocol": ref, "windows": int(len(meta)),
                     "repeats": repeats}}
    c5g = {"id": "C5g", "status": "PASS" if "grouped" in summ else "NOT_TESTABLE",
           "metric": int(meta.groupby("stratum")["cluster"].nunique().min()),
           "metric_name": "min independence clusters per stratum",
           "detail": {"needed": spec["min_clusters_per_stratum"]}}
    return c5, c5g


def c6_shortcut(ds, desc, repeats=None):
    spec = S.CHECKS["C6"]
    repeats = repeats or S.CHECKS["C5"]["repeats"]
    _, Xa, meta = windows(ds, desc)
    y = meta["y"].to_numpy()
    if len(np.unique(y)) < 2:
        return {"id": "C6", "status": "NA", "metric": None, "metric_name": "no labels", "detail": {}}
    from sklearn.dummy import DummyClassifier
    art, chance = [], []
    for r in range(repeats):
        tr, te = split_random(meta, r, S.CHECKS["C5"]["test_fraction"])
        art.append(fit_score(Xa, y, tr, te, r))
        for k in range(spec["dummy_seeds"]):
            d = DummyClassifier(strategy="stratified", random_state=1000 * r + k).fit(Xa[tr], y[tr])
            chance.append(macro_f1(y[te], d.predict(Xa[te])))
    a, c = float(np.mean(art)), float(np.mean(chance))
    return {"id": "C6", "status": S.band_max(a - c, spec["pass_max"], spec["cond_max"]),
            "metric": a - c, "metric_name": "artefact-only macro-F1 minus chance",
            "detail": {"artefact_only_macro_f1": a, "chance_macro_f1": c,
                       "features": S.MODEL["artefact_features"]}}


def c7_floor(ds, desc):
    spec = S.CHECKS["C7"]
    null = desc.get("null_stratum", ds["strata"][0])
    counts = {}
    for u in ds["units"]:
        if u["stratum"] == null:
            continue
        _, y, _ = decimate(u, desc, ds["channels"])
        counts[u["stratum"]] = counts.get(u["stratum"], 0) + int(y.sum())
    if not counts:
        return {"id": "C7", "status": "NA", "metric": None, "metric_name": "no event classes", "detail": {}}
    m = min(counts.values())
    return {"id": "C7", "status": S.band_min(m, spec["pass_min"], spec["cond_min"]), "metric": m,
            "metric_name": "min positive samples per class at the physical timescale",
            "detail": {"per_class": counts, "physical_dt_s": desc["physical_dt_s"]}}


def c8_differential(ds, desc):
    spec = S.CHECKS["C8"]
    cfg = desc.get("differential")
    if not cfg:
        return {"id": "C8", "status": "NA", "metric": None, "metric_name": "no baseline declared", "detail": {}}
    ch = ds["channels"]
    cols = [ch.index(c) for c in _match(ch, cfg["channels"])]
    sc = channel_scale(ds, desc)[cols]
    base = [decimate(u, desc, ch)[0][:, cols] / sc for u in ds["units"] if u["stratum"] == cfg["baseline_stratum"]]
    per = {}
    for u in ds["units"]:
        if u["stratum"] == cfg["baseline_stratum"]:
            continue
        D, y, _ = decimate(u, desc, ch)
        if y.max() == 0:
            continue
        o = int(np.argmax(y > 0))
        D = D[:, cols] / sc
        best = None
        for B in base:
            if len(B) < o + spec["post_samples"]:
                continue
            p0 = max(0, o - spec["pre_samples"])
            pre = np.sqrt(np.nanmean((D[p0:o] - B[p0:o]) ** 2)) if o > p0 else np.nan
            post = np.sqrt(np.nanmean((D[o:o + spec["post_samples"]] - B[o:o + spec["post_samples"]]) ** 2))
            if best is None or (np.isfinite(pre) and pre < best[0]):
                best = (pre, post)
        if best is None or not np.isfinite(best[0]):
            ratio = None
        else:
            ratio = float(best[1] / best[0]) if best[0] > 0 else float("inf")
        per.setdefault(u["stratum"], []).append(ratio)
    stat = {s: (float(np.median([r for r in v if r is not None])) if any(r is not None for r in v) else None)
            for s, v in per.items()}
    ok = [s for s, r in stat.items() if r is not None and r >= spec["pass_min"]]
    status = "PASS" if len(ok) == len(stat) and stat else ("CONDITIONAL" if ok else "FAIL")
    return {"id": "C8", "status": status, "metric": len(ok),
            "metric_name": "event classes passing the pre-onset gate",
            "detail": {"gate_ratio": stat, "admissible_classes": ok,
                       "excluded_classes": [s for s in stat if s not in ok],
                       "channels": [ch[j] for j in cols], "floor": spec["pass_min"]}}


# =========================================================================== PVDAQ catalogue

MIN_PER_YEAR = 525_600.0


def load_pvdaq_catalogue(desc, verbose=True):
    cat = pd.read_csv(desc["catalogue"])
    cat = cat.loc[:, ~cat.columns.str.startswith("Unnamed")]
    for c in ("years", "number_records", "available_sensor_channels", "latitude", "longitude"):
        cat[c] = pd.to_numeric(cat[c], errors="coerce")
    cache = json.load(open(desc["listing_cache"]))
    return {"kind": "catalogue", "cat": cat, "cache": cache}


def _usable_years(summary, min_days, n_years):
    good = {r["year"] for r in summary if r["kind"] in ("intraday", "mixed") and r["days"] >= min_days}
    for yv in sorted(good, reverse=True):
        if all(v in good for v in range(yv - n_years + 1, yv + 1)):
            return True
    return False


def pvdaq_checks(ds, desc):
    s2, s3, s4 = S.CHECKS["C2"], S.CHECKS["C3"], S.CHECKS["C4"]
    cat, cache = ds["cat"], ds["cache"]
    cat["est_interval_min"] = MIN_PER_YEAR / (cat["number_records"] / cat["years"] / cat["available_sensor_channels"])
    is_pvo = cat["system_public_name"].astype(str).str.startswith("Pvoutput")
    elig = cat[cat["qa_status"].astype(str).str.strip().str.lower().eq("pass")
               & (cat["years"] >= desc["min_years"]) & cat["latitude"].notna() & cat["longitude"].notna()
               & (cat["est_interval_min"] <= desc["max_interval_min"])]
    pvo = elig[is_pvo.loc[elig.index]]
    non = elig[~is_pvo.loc[elig.index]]

    def contradicts(sid):
        s = cache.get(str(int(sid)))
        if s is None:
            return None
        return (not s) or all(r["kind"] == "daily" for r in s)

    pv_list = [contradicts(s) for s in pvo["system_id"]]
    pv_seen = [v for v in pv_list if v is not None]
    non_list = [contradicts(s) for s in non["system_id"]]
    if any(v is None for v in non_list):
        raise RuntimeError("listing cache incomplete for non-PVOutput candidates; run pvdaq_funnel.py first")
    pv_rate = float(np.mean(pv_seen)) if pv_seen else None
    est = (pv_rate * len(pvo) + sum(non_list)) / len(elig) if pv_rate is not None else None
    c2 = {"id": "C2", "status": S.band_max(est, 0.0, s2["contradiction_cond_max"]), "metric": est,
          "metric_name": "estimated fraction of eligible systems whose content contradicts the catalogued interval",
          "detail": {"eligible": len(elig), "pvoutput": len(pvo), "pvoutput_listed": len(pv_seen),
                     "pvoutput_contradicting": int(sum(pv_seen)), "non_pvoutput": len(non),
                     "non_pvoutput_contradicting": int(sum(non_list)),
                     "declared": f"catalogue-implied interval <= {desc['max_interval_min']} min"}}

    units = non[[_usable_years(cache[str(int(s))], desc["min_days"], desc["n_years"])
                 for s in non["system_id"]]]
    la = np.radians(units["latitude"].to_numpy(float))
    lo = np.radians(units["longitude"].to_numpy(float))
    dist = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
    d0 = float(desc["min_distance_km"])
    adj = (dist < d0) & ~np.eye(len(units), dtype=bool)
    mis = max_independent(adj)
    sens = {f"{d:g} km": max_independent((dist < d) & ~np.eye(len(units), dtype=bool))
            for d in desc.get("sensitivity_km", [])}
    dep = float(adj.any(axis=1).mean()) if len(units) else None
    c3 = {"id": "C3", "status": S.band_min(mis, s3["pass_min"], s3["cond_min"]), "metric": mis,
          "metric_name": f"independent sites at >= {d0:g} km",
          "detail": {"candidate_units": len(units), "sensitivity": sens}}
    c4 = {"id": "C4", "status": S.band_max(dep, s4["pass_max"], s4["cond_max"]), "metric": dep,
          "metric_name": f"fraction of sites with another within {d0:g} km",
          "detail": {"dependent_pairs": int(np.triu(adj, 1).sum()),
                     "test": "great-circle distance below the declared minimum"}}
    return [c2, c3, c4]


# =========================================================================== driver

def run_checks(ds, desc, repeats=None, verbose=True):
    def log(m):
        if verbose:
            print(m, flush=True)

    if ds["kind"] == "catalogue":
        out = pvdaq_checks(ds, desc)
        for cid in ("C1", "C5", "C5g", "C6", "C7", "C8", "C1d"):
            out.append({"id": cid, "status": "NA", "metric": None,
                        "metric_name": "not applicable at catalogue level", "detail": {}})
        return {r["id"]: r for r in out}

    res = {}
    ds.pop("_windows", None)
    log("  C1 integrity"); res["C1"] = c1_integrity(ds, desc)
    if desc.get("differential"):
        chd = _match(ds["channels"], desc["differential"]["channels"])
        res["C1d"] = c1_integrity(ds, desc, chd, cid="C1d")
    else:
        res["C1d"] = {"id": "C1d", "status": "NA", "metric": None, "metric_name": "", "detail": {}}
    log("  C2 metadata"); res["C2"] = c2_metadata(ds, desc)
    log("  C3/C4 independence"); res["C3"], res["C4"] = c3_c4_timeseries(ds, desc)
    log("  C5 split protocol"); res["C5"], res["C5g"] = c5_split(ds, desc, repeats)
    log("  C6 shortcut probe"); res["C6"] = c6_shortcut(ds, desc, repeats)
    log("  C7 per-class floor"); res["C7"] = c7_floor(ds, desc)
    log("  C8 differential"); res["C8"] = c8_differential(ds, desc)
    return res


def verdicts(res, claims):
    st = {k: v["status"] for k, v in res.items()}
    out = {}
    for k in claims:
        v, why = S.claim_verdict(k, st)
        out[k] = {"claim": S.CLAIMS[k]["name"], "verdict": v, "because": why}
    return out


def clone(ds):
    """Shallow copy of the dataset with copied unit dicts (arrays shared until modified)."""
    d = {k: v for k, v in ds.items() if not k.startswith("_")}
    d["units"] = [{k: v for k, v in u.items() if not (isinstance(k, tuple) and k[0] == "_dec")}
                  for u in ds["units"]]
    return d
