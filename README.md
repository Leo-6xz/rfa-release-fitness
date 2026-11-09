# Release-Fitness Assessment Workflow for Power-System Machine-Learning Datasets

Reference implementation of the release-fitness assessment described in

> E. E. Ogar, O. C. Ogiesoba-Eguakun, S. Rath, and S. Chaitusaney, "Measuring the Dataset, Not the Model: A Release-Fitness Assessment Workflow for Power-System Machine-Learning Datasets," submitted to IEEE PES General Meeting 2027.

The workflow evaluates a dataset release, not a model trained on it. Eight checks (C1–C8), with thresholds fixed and hash-stamped before analysis, issue a PASS / CONDITIONAL / FAIL verdict for each claim a release is meant to support (K1 in-distribution accuracy, K2 generalization to unseen units, K3 per-class performance, K4 physical-model validation).

**Frozen specification:** version 1.0.0, hash `8ffc444dbc915624`, frozen 2026-11-07. Every report in `results/` records the spec hash it was produced under. `python rfa/rfa\\\_spec.py` prints the hash of your copy.

## Contents

|**Path**|**What it is**|
|-|-|
|`rfa/rfa\\\_spec.py`, `rfa/SPEC.md`|The frozen specification: checks, thresholds, claim dependencies|
|`rfa/rfa\\\_core.py`|Dataset adapters and the eight checks|
|`rfa/fitness\\\_report.py`|Runs the assessment on one dataset descriptor|
|`rfa/inject\\\_defects.py`|Plants defects D1–D6 in a clean corpus and checks which check catches each|
|`rfa/diagnose\\\_independence.py`, `rfa/msu\\\_file\\\_holdout.py`|Read-only diagnostics used in the paper's discussion; they change no verdict|
|`rfa/make\\\_test\\\_corpus.py`|Builds small stand-in corpora for testing the code; not used for the paper|
|`rfa/descriptors/`|One JSON descriptor per dataset: where the files are, the unit of independence, the intended claims|
|`rfa/survey/`|Coded survey of split protocols in 15 papers on the MSU–ORNL datasets, with verbatim quotes and PDF pages, and the coding method|
|`results/`|The reports behind every number in the paper|
|`pvdaq\\\_funnel.py` and the other root scripts|NREL PVDAQ catalog screening (eligible systems, intraday records, site separation)|

## Requirements

Python 3.10 or later with numpy, pandas and scikit-learn (1.2 or later). The paper results were produced with Python 3.14.

```
pip install numpy pandas scikit-learn
```

## Data

No third-party data is redistributed here. Obtain each dataset from its source and set the path in its descriptor (`root`, `files` or `catalogue`). The descriptors in this repository hold the authors' local paths.

|Dataset|Source|Descriptor|
|-|-|-|
|Labeled cyber-physical microgrid dataset|IEEE DataPort, doi:10.21227/bmtr-fr76|`microgrid\\\_release.json`|
|MSU–ORNL power-system attack datasets (multiclass, 15 files)|Mirror pinned at commit `0e07bddd072b96c53dece691270002db5cb689ad` of github.com/VictoKu1/IndustrialControlSystemCyberAttackDetectingCourse, `Class/multiclass/data1.csv` … `data15.csv`|`msu\\\_power\\\_system.json`|
|NREL PVDAQ public systems|Open Energy Data Initiative, https://registry.opendata.aws/oedi-data-lake/|`pvdaq.json`|
|Synthetic 280-run microgrid corpus|Synthetic Cyberattack Corpus for Distributed Secondary Control of a Ten-Generator Islanded Microgrid (280 Independent Runs), Zenodo, 2026. \[Data set]. \[Online]. Available: https://doi.org/10.5281/zenodo.24082860|`microgrid\\\_synthetic.json`|

For MSU–ORNL, place the 15 CSV files in `msu/multiclass/`; the 15 files hold 78,377 rows in total, and `data2.csv` has 5,069.

## Reproducing the paper

Run from the repository root:

```
python rfa/fitness\\\_report.py --descriptor rfa/descriptors/microgrid\\\_release.json
python rfa/fitness\\\_report.py --descriptor rfa/descriptors/microgrid\\\_synthetic.json
python rfa/inject\\\_defects.py --descriptor rfa/descriptors/microgrid\\\_synthetic.json
python rfa/fitness\\\_report.py --descriptor rfa/descriptors/msu\\\_power\\\_system.json
python rfa/fitness\\\_report.py --descriptor rfa/descriptors/pvdaq.json
```

Each run writes `report.json`, `checks.csv` and `fitness\\\_matrix.csv` to `results/<dataset>/`; the defect run writes `defect\\\_matrix.csv`, `defect\\\_summary.csv` and `defect\\\_metrics.json` to `results/microgrid\\\_synthetic\\\_defects/`. `--repeats` exists for quick testing only; the paper uses the specification's value.

Expected MSU–ORNL result: C1 FAIL (0.0804), C3 CONDITIONAL (9), C5 FAIL (0.2658), C6 PASS (−0.0203); K1–K3 FAIL.

## License

Code: Apache License 2.0 (see `LICENSE`). The survey table and the result files in `rfa/survey/` and `results/` are additionally offered under CC BY 4.0.

## Citation

See `CITATION.cff`, or use the "Cite this repository" button on GitHub.

