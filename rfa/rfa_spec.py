"""
rfa_spec.py -- the release-fitness specification, frozen before the final analysis.

Every threshold the workflow uses is defined here and nowhere else. The report
records SPEC_VERSION and spec_hash(), so a reader can confirm that the verdicts
in the paper were produced by this file and not by thresholds chosen after the
results were seen. Commit this file, and note the hash, BEFORE running the
final analysis. Changing any value changes the hash.

Verdict scale for a check:
    PASS          the property holds
    CONDITIONAL   the property holds only partly; a claim relying on it must
                  carry the stated restriction
    FAIL          the property does not hold
    NOT_TESTABLE  the check applies but the data cannot support it
                  (for example, a grouped split with one unit per class)
    NA            the check does not apply to this dataset (no labels, no
                  declared baseline, ...)

A dataset is not "fit" or "unfit" in the abstract. It is fit for a CLAIM.
CLAIMS below lists, for each claim type, which checks it relies on and how:
    hard      the check's verdict propagates to the claim unchanged
              (NOT_TESTABLE counts as FAIL)
    soft      a FAIL is downgraded to CONDITIONAL: the claim survives, but
              only with the restriction the check names
    testable  only whether the check could be run matters (NOT_TESTABLE ->
              FAIL, anything else -> PASS)
NA checks are ignored. The claim verdict is the worst of the contributions.
"""

import hashlib
import json

SPEC_VERSION = "1.0.0"
FROZEN_ON = "2026-11-07"

CHECKS = {
    "C1": {
        "name": "Integrity",
        "metric": "fraction of samples (rows) containing a non-finite value or a value "
                  "beyond the declared physical bound, over all declared channels",
        "pass_max": 0.0,
        "cond_max": 0.001,
        "default_bound": 1.0e7,
        "rationale": "Any corrupted sample is a defect; up to 0.1% of rows can be masked "
                     "without materially changing class balance, beyond that the corruption "
                     "is part of what a model learns.",
    },
    "C2": {
        "name": "Metadata fidelity",
        "metric": "agreement between what the release declares and what it contains: "
                  "sampling interval (observed/declared), declared channels present and "
                  "non-empty; for catalogue-level releases, the estimated fraction of "
                  "systems whose content contradicts the catalogued interval",
        "interval_tol": 0.10,
        "contradiction_cond_max": 0.10,
        "rationale": "A user selects data from its description. A description that does "
                     "not match the content silently selects the wrong population.",
    },
    "C3": {
        "name": "Effective sample size",
        "metric": "maximum number of mutually independent units (per class stratum for "
                  "labelled data, overall otherwise), i.e. the maximum independent set "
                  "of the dependence graph from C4",
        "pass_min": 10,
        "cond_min": 3,
        "rationale": "Three independent units per class is the minimum for a held-out "
                     "unit with two left to train on; ten allows 5-fold grouped "
                     "cross-validation with two units per fold.",
    },
    "C4": {
        "name": "Unit of independence",
        "metric": "fraction of declared units that have at least one dependent partner "
                  "(identical pre-event trajectory within identity_tol, or closer than "
                  "the declared minimum distance for spatial units)",
        "pass_max": 0.0,
        "cond_max": 0.10,
        "identity_tol": 1.0e-6,
        "min_overlap": 10,
        "rationale": "Row counts and file counts overstate evidence when units share a "
                     "solve or a weather system. Dependence is tested, not assumed.",
    },
    "C5": {
        "name": "Split-protocol sensitivity",
        "metric": "macro-F1 under a random split minus macro-F1 under a split that holds "
                  "out whole independence clusters (grouped), or, when that is "
                  "impossible, randomly chosen contiguous blocks (block_count per unit) "
                  "with overlapping training windows purged",
        "pass_max": 0.02,
        "cond_max": 0.10,
        "test_fraction": 0.30,
        "repeats": 3,
        "min_clusters_per_stratum": 3,
        "block_count": 10,
        "rationale": "If the protocol moves the score by more than run-to-run noise, a "
                     "random-split number is not evidence of generalisation.",
    },
    "C6": {
        "name": "Shortcut probe",
        "metric": "macro-F1 of the reference model trained ONLY on acquisition artefacts "
                  "(position in record, record length, corruption pattern), minus the "
                  "macro-F1 of a stratified random guess, under a random split",
        "pass_max": 0.05,
        "cond_max": 0.20,
        "dummy_seeds": 20,
        "rationale": "If the label can be recovered without any physical signal, a model "
                     "can score well without learning the phenomenon.",
    },
    "C7": {
        "name": "Per-class floor",
        "metric": "minimum number of positive (event) samples in any class at the "
                  "declared physical timescale",
        "pass_min": 100,
        "cond_min": 30,
        "rationale": "About 96 independent samples bound a per-class recall estimate to a "
                     "95% half-width of 0.10 (Wilson, p = 0.5); below 30 the interval "
                     "spans most of [0, 1]. Samples within a trajectory are correlated, "
                     "so this floor is necessary, not sufficient.",
    },
    "C8": {
        "name": "Differential admissibility",
        "metric": "RMS normalised difference between an event unit and its matched "
                  "baseline unit in the post_samples after onset, divided by the same "
                  "quantity in the pre_samples before onset",
        "pass_min": 10.0,
        "pre_samples": 20,
        "post_samples": 20,
        "rationale": "A differential (event minus baseline) comparison isolates the event "
                     "only if the two records agree before it; the post-onset divergence "
                     "must exceed the pre-onset mismatch by an order of magnitude.",
    },
}

MODEL = {
    "estimator": "sklearn.ensemble.HistGradientBoostingClassifier",
    "max_iter": 200,
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "class_weight": "balanced",
    "early_stopping": False,
    "window_stats": ["mean", "std", "slope", "range"],
    "artefact_features": ["position_in_record", "record_length", "corrupt_fraction_per_channel"],
}

CLAIMS = {
    "K1": {"name": "In-distribution accuracy",
           "hard": ["C1", "C6"], "soft": ["C2", "C7"], "testable": []},
    "K2": {"name": "Generalization to unseen units",
           "hard": ["C1", "C3", "C6"], "soft": ["C2", "C4"], "testable": ["C5g"]},
    "K3": {"name": "Per-class performance",
           "hard": ["C1", "C6", "C7"], "soft": ["C2"], "testable": []},
    "K4": {"name": "Physical-model validation (differential)",
           "hard": ["C1d", "C8"], "soft": ["C2"], "testable": []},
}
# C1d is C1 evaluated only on the channels a differential comparison uses.
# C5g records whether a grouped (whole-cluster) split was possible at all. C5
# itself measures the protocol gap and tells the author WHICH split to report;
# a large gap does not make a dataset unfit for K2, an impossible grouped split
# does.

ORDER = {"PASS": 0, "CONDITIONAL": 1, "FAIL": 2}


def spec_dict():
    return {"version": SPEC_VERSION, "frozen_on": FROZEN_ON,
            "checks": CHECKS, "model": MODEL, "claims": CLAIMS}


def spec_hash():
    blob = json.dumps(spec_dict(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def band_max(value, pass_max, cond_max):
    """Lower is better."""
    if value is None:
        return "NOT_TESTABLE"
    if value <= pass_max:
        return "PASS"
    if value <= cond_max:
        return "CONDITIONAL"
    return "FAIL"


def band_min(value, pass_min, cond_min):
    """Higher is better."""
    if value is None:
        return "NOT_TESTABLE"
    if value >= pass_min:
        return "PASS"
    if value >= cond_min:
        return "CONDITIONAL"
    return "FAIL"


def claim_verdict(claim, statuses):
    """statuses: {check_id: status}. Returns (verdict, reasons)."""
    rule = CLAIMS[claim]
    worst, reasons = "PASS", []

    def take(v, cid, how):
        nonlocal worst
        if ORDER[v] > ORDER[worst]:
            worst = v
        if v != "PASS":
            reasons.append(f"{cid} {statuses.get(cid)} ({how})")

    for cid in rule["hard"]:
        s = statuses.get(cid, "NA")
        if s == "NA":
            continue
        take("FAIL" if s == "NOT_TESTABLE" else s, cid, "hard")
    for cid in rule["soft"]:
        s = statuses.get(cid, "NA")
        if s == "NA":
            continue
        v = "FAIL" if s == "NOT_TESTABLE" else s
        take("CONDITIONAL" if v == "FAIL" else v, cid, "soft")
    for cid in rule["testable"]:
        s = statuses.get(cid, "NA")
        if s == "NA":
            continue
        take("FAIL" if s == "NOT_TESTABLE" else "PASS", cid, "testable")
    return worst, reasons


if __name__ == "__main__":
    print(f"release-fitness spec {SPEC_VERSION}, frozen {FROZEN_ON}, hash {spec_hash()}")
