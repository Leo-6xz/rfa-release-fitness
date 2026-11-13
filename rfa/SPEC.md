# Release-fitness specification v1.1.0

Frozen 2026-11-13. Hash `60f433f1d2401c8d` (printed by `python rfa_spec.py`; recorded in every report). The thresholds below were fixed before the final analysis was run. Changing any of them changes the hash.

**Change from v1.0.0** (frozen 2026-11-07, hash `8ffc444dbc915624`): after co-author review, C6 moved from hard to soft for K1–K3. A positive shortcut probe shows an artefact the reference model can exploit under the tested protocol, which qualifies an accuracy claim; a negative probe does not establish that no shortcut exists. Recomputing every reported verdict from the stored check results under v1.1.0 (`rfa/reverdict.py`) changes none of them: each public-release FAIL rests on a model-free check (C1, and C3 for K2). No threshold changed.

**Verdict meaning.** PASS means no defect was detected. For the model-free checks (C1–C4, C7, C8) the check measures the property directly. For the model-based checks (C5, C6) it is a negative result under the reference model, the tested features and the tested protocol, not proof of absence.

## Checks

| Check | Name | Metric | PASS | CONDITIONAL | FAIL |
|---|---|---|---|---|---|
| C1 | Integrity | fraction of samples (rows) containing a non-finite value or a value beyond the declared physical bound, over all declared channels | ≤ 0 | ≤ 0.001 | > 0.001 |
| C2 | Metadata fidelity | agreement between what the release declares and what it contains: sampling interval (observed/declared), declared channels present and non-empty; for catalogue-level releases, the estimated fraction of systems whose content contradicts the catalogued interval | interval within ±10%, channels present | catalogue: contradiction ≤ 0.1 | otherwise |
| C3 | Effective sample size | maximum number of mutually independent units (per class stratum for labelled data, overall otherwise), i.e. the maximum independent set of the dependence graph from C4 | ≥ 10 | ≥ 3 | < 3 |
| C4 | Unit of independence | fraction of declared units that have at least one dependent partner (identical pre-event trajectory within identity_tol, or closer than the declared minimum distance for spatial units) | ≤ 0 | ≤ 0.1 | > 0.1 |
| C5 | Split-protocol sensitivity | macro-F1 under a random split minus macro-F1 under a split that holds out whole independence clusters (grouped), or, when that is impossible, randomly chosen contiguous blocks (block_count per unit) with overlapping training windows purged | ≤ 0.02 | ≤ 0.1 | > 0.1 |
| C6 | Shortcut probe | macro-F1 of the reference model trained ONLY on acquisition artefacts (position in record, record length, corruption pattern), minus the macro-F1 of a stratified random guess, under a random split | ≤ 0.05 | ≤ 0.2 | > 0.2 |
| C7 | Per-class floor | minimum number of positive (event) samples in any class at the declared physical timescale | ≥ 100 | ≥ 30 | < 30 |
| C8 | Differential admissibility | RMS normalised difference between an event unit and its matched baseline unit in the post_samples after onset, divided by the same quantity in the pre_samples before onset | all classes ≥ 10 | some classes | no class |

C1d is C1 restricted to the channels a differential comparison uses. C5g records whether a grouped (whole-cluster) split was possible.

## Rationale

- **C1 Integrity.** Any corrupted sample is a defect; up to 0.1% of rows can be masked without materially changing class balance, beyond that the corruption is part of what a model learns.
- **C2 Metadata fidelity.** A user selects data from its description. A description that does not match the content silently selects the wrong population.
- **C3 Effective sample size.** Three independent units per class is the minimum for a held-out unit with two left to train on; ten allows 5-fold grouped cross-validation with two units per fold.
- **C4 Unit of independence.** Row counts and file counts overstate evidence when units share a solve or a weather system. Dependence is tested, not assumed.
- **C5 Split-protocol sensitivity.** If the protocol moves the score by more than run-to-run noise, a random-split number is not evidence of generalisation.
- **C6 Shortcut probe.** If the label can be recovered without any physical signal, a model can score well without learning the phenomenon.
- **C7 Per-class floor.** About 96 independent samples bound a per-class recall estimate to a 95% half-width of 0.10 (Wilson, p = 0.5); below 30 the interval spans most of [0, 1]. Samples within a trajectory are correlated, so this floor is necessary, not sufficient.
- **C8 Differential admissibility.** A differential (event minus baseline) comparison isolates the event only if the two records agree before it; the post-onset divergence must exceed the pre-onset mismatch by an order of magnitude. When the records agree exactly before onset, the event is admissible only if they then differ; identical records fail.

## Claims

A dataset is fit for a claim, not in the abstract. *Hard*: the check's verdict propagates (NOT_TESTABLE counts as FAIL). *Soft*: FAIL is downgraded to CONDITIONAL. *Testable*: only whether the check could run matters. NA checks are ignored; the claim takes the worst contribution.

| Claim | Hard | Soft | Testable |
|---|---|---|---|
| K1 In-distribution accuracy | C1 | C2, C6, C7 | – |
| K2 Generalization to unseen units | C1, C3 | C2, C4, C6 | C5g |
| K3 Per-class performance | C1, C7 | C2, C6 | – |
| K4 Physical-model validation (differential) | C1d, C8 | C2 | – |

## Reference model

sklearn.ensemble.HistGradientBoostingClassifier (max_iter 200, learning_rate 0.1, max_leaf_nodes 31, class_weight balanced, no early stopping). Signal features: per-channel window mean, std, slope, range over a window of the declared length at the declared physical timescale. Artefact features (C6 only): position_in_record, record_length, corrupt_fraction_per_channel. Score: macro-F1 over the classes present in the test set.

## Declared by the dataset descriptor, not by this specification

The physical timescale and window, the physical bounds per channel, the null (no-event) class, the baseline class and channels for C8, and the minimum distance for spatial independence. These are statements about the dataset and its intended use and must be fixed in the descriptor before the run, like the thresholds. PVDAQ declares 100 km; the report also gives C3 at 200, 50, 25 and 10 km so the dependence on that choice is visible.
