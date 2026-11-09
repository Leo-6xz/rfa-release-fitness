"""
msu_file_holdout.py -- EXPLORATORY (outside the frozen specification).

Scores the spec's reference model on the MSU-ORNL binary task with whole data
files held out (5 of 15 files per repeat, 3 repeats), for comparison with the
random-row and held-out-scenario figures that fitness_report.py produces.

    python rfa\\msu_file_holdout.py --descriptor rfa\\descriptors\\msu_power_system.json
"""
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rfa_core as C   # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--descriptor", required=True); a = ap.parse_args()
desc = json.load(open(a.descriptor))
files = sorted(glob.glob(desc["files"]), key=lambda s: int(re.findall(r"\d+", os.path.basename(s))[0]))
df = pd.concat([pd.read_csv(f).assign(_file=i) for i, f in enumerate(files)], ignore_index=True)
feat = [c for c in df.columns if c not in (desc["unit_column"], "_file")]
X = df[feat].apply(pd.to_numeric, errors="coerce").to_numpy(float); X[~np.isfinite(X)] = np.nan
natural = set(desc["class_of_unit"]["Natural"])
y = (~df[desc["unit_column"]].isin(natural)).astype(int).to_numpy()
fid = df["_file"].to_numpy(); scores = []
for r in range(3):
    te_f = np.random.default_rng(r).choice(len(files), 5, replace=False)
    te = np.flatnonzero(np.isin(fid, te_f)); tr = np.flatnonzero(~np.isin(fid, te_f))
    scores.append(C.fit_score(X, y, tr, te, r))
print(f"file-held-out binary macro-F1: {np.mean(scores):.3f} +/- {np.std(scores):.4f}  {scores}")
