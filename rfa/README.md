# rfa: release-fitness assessment

Place this folder at C:\research\benchmark\rfa and run everything from C:\research\benchmark.
Requires numpy, pandas and scikit-learn (>= 1.2).

1. `python rfa\rfa_spec.py` prints the spec hash. Commit rfa\rfa_spec.py (or otherwise timestamp the hash) BEFORE step 2.
2. `python rfa\fitness_report.py --descriptor rfa\descriptors\microgrid_release.json`
3. `python rfa\fitness_report.py --descriptor rfa\descriptors\microgrid_synthetic.json`
4. `python rfa\inject_defects.py --descriptor rfa\descriptors\microgrid_synthetic.json`
5. `python rfa\fitness_report.py --descriptor rfa\descriptors\pvdaq.json`   (needs pvdaq\systems.csv and pvdaq\listing_cache.json from pvdaq_funnel.py)

Results go to results\<dataset name>\ (report.json, checks.csv, fitness_matrix.csv) and
results\microgrid_synthetic_defects\ (defect_matrix.csv, defect_summary.csv, defect_metrics.json).

`--repeats 1` exists for quick testing only; the paper runs use the spec's value.
make_test_corpus.py builds small stand-in corpora for testing the code; it is not used for the paper.
See SPEC.md for every check, threshold and verdict rule.
