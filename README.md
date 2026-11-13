# Release-Fitness Assessment Workflow for Power-System Machine-Learning Datasets

Reference implementation of the release-fitness assessment described in

> E. E. Ogar, O. C. Ogiesoba-Eguakun, S. Rath, and S. Chaitusaney, "Measuring the Dataset, Not the Model: A Release-Fitness Assessment Workflow for Power-System Machine-Learning Datasets," submitted to IEEE PES General Meeting 2027.

The workflow evaluates a dataset release, not a model trained on it. Eight checks (C1–C8), with thresholds fixed and hash-stamped before analysis, issue a PASS / CONDITIONAL / FAIL verdict for each claim a release is meant to support (K1 in-distribution accuracy, K2 generalization to unseen units, K3 per-class performance, K4 physical-model validation).

**Specification:** version 1.1.0, hash `60f433f1d2401c8d`, frozen 2026-11-13. The reports in `results/` were produced under version 1.0.0 (hash `8ffc444dbc915624`, frozen 2026-11-07), except `results/microgrid_release_v110`, a full rerun of the microgrid release under 1.1.0 that reproduces every check value. Version 1.1.0 changes only a verdict rule (C6 from hard to soft for the accuracy claims) and corrects two edge cases in C8 that affect no reported ratio; `python rfa/reverdict.py` recomputes every verdict from the stored check results under the current specification and confirms that none changes. See `rfa/SPEC.md`. `python rfa/rfa_spec.py` prints the hash of your copy.

## Contents

|**Path**|**What it is**|
|-|-|
|`rfa/rfa_spec.py`, `rfa/SPEC.md`|The frozen specification: checks, thresholds, claim dependencies|
|`rfa/rfa_core.py`|Dataset adapters and the eight checks|
|`rfa/fitness_report.py`|Runs the assessment on one dataset descriptor|
|`rfa/reverdict.py`|Recomputes claim verdicts from stored check results under the current specification|
|`tests/test_c8.py`|Checks the C8 gate on constructed records, including identical and exactly matching ones|
|`rfa/inject_defects.py`|Plants defects D1–D6 in a clean corpus and checks which check catches each|
|`rfa/diagnose_independence.py`, `rfa/msu_file_holdout.py`|Read-only diagnostics used in the paper's discussion; they change no verdict|
|`rfa/make_test_corpus.py`|Builds small stand-in corpora for testing the code; not used for the paper|
|`rfa/descriptors/`|One JSON descriptor per dataset: where the files are, the unit of independence, the intended claims|
|`rfa/survey/`|Coded survey of split protocols in 15 papers on the MSU–ORNL datasets, with verbatim quotes and PDF pages, and the coding method|
|`results/`|The reports behind every number in the paper|
|`pvdaq_funnel.py` and the other root scripts|NREL PVDAQ catalog screening (eligible systems, intraday records, site separation)|

## Requirements

Python 3.10 or later with numpy, pandas and scikit-learn (1.2 or later). The paper results were produced with Python 3.14.

```
pip install numpy pandas scikit-learn
```

## Data

No third-party data is redistributed here. Obtain each dataset from its source and set the path in its descriptor (`root`, `files` or `catalogue`). The descriptors in this repository hold the authors' local paths.

|Dataset|Source|Descriptor|
|-|-|-|
|Labeled cyber-physical microgrid dataset|IEEE DataPort, doi:10.21227/bmtr-fr76|`microgrid_release.json`|
|MSU–ORNL power-system attack datasets (multiclass, 15 files)|Mirror pinned at commit `0e07bddd072b96c53dece691270002db5cb689ad` of github.com/VictoKu1/IndustrialControlSystemCyberAttackDetectingCourse, `Class/multiclass/data1.csv` … `data15.csv`|`msu_power_system.json`|
|NREL PVDAQ public systems|Open Energy Data Initiative, https://registry.opendata.aws/oedi-data-lake/|`pvdaq.json`|
|Synthetic 280-run microgrid corpus|Zenodo, doi:[10.5281/zenodo.24082859](https://doi.org/10.5281/zenodo.24082859)|`microgrid_synthetic.json`|

For MSU–ORNL, place the 15 CSV files in `msu/multiclass/`; the 15 files hold 78,377 rows in total, and `data2.csv` has 5,069.

## Reproducing the paper

Run from the repository root:

```
python rfa/fitness_report.py --descriptor rfa/descriptors/microgrid_release.json
python rfa/fitness_report.py --descriptor rfa/descriptors/microgrid_synthetic.json
python rfa/inject_defects.py --descriptor rfa/descriptors/microgrid_synthetic.json
python rfa/fitness_report.py --descriptor rfa/descriptors/msu_power_system.json
python rfa/fitness_report.py --descriptor rfa/descriptors/pvdaq.json
```

Each run writes `report.json`, `checks.csv` and `fitness_matrix.csv` to `results/<dataset>/`; the defect run writes `defect_matrix.csv`, `defect_summary.csv` and `defect_metrics.json` to `results/microgrid_synthetic_defects/`. `--repeats` exists for quick testing only; the paper uses the specification's value.

Expected MSU–ORNL result: C1 FAIL (0.0804), C3 CONDITIONAL (9), C5 FAIL (0.2658), C6 PASS (−0.0203); K1–K3 FAIL.

## License

Code: Apache License 2.0 (see `LICENSE`). The survey table and the result files in `rfa/survey/` and `results/` are additionally offered under CC BY 4.0.

## Citation

Cite this software through its Zenodo concept DOI, which always resolves to the latest version:

> E. E. Ogar, O. C. Ogiesoba-Eguakun, S. Rath, and S. Chaitusaney, "Release-fitness assessment workflow for power-system machine-learning datasets," Zenodo, Nov. 2026. [Software]. doi: [10.5281/zenodo.24075893](https://doi.org/10.5281/zenodo.24075893).

`CITATION.cff` holds the same entry, and GitHub's "Cite this repository" button exports it.

## Related records

- The clean corpus used for the planted-defect validation:
  E. E. Ogar and S. Chaitusaney, "Synthetic Cyberattack Corpus for Distributed Secondary Control of a Ten-Generator Islanded Microgrid (280 Independent Runs)," Zenodo, Nov. 2026. [Data set]. doi: [10.5281/zenodo.24082859](https://doi.org/10.5281/zenodo.24082859).
- The simulator and controller that generated it:
  E. E. Ogar and S. Chaitusaney, "mgxrl: Trust-weighted resilient consensus control for microgrids (code, trained policies and results)," Zenodo, Nov. 2026. [Software]. doi: [10.5281/zenodo.24134204](https://doi.org/10.5281/zenodo.24134204).
