"""Generate transparent, synthetic biomedical causal-inference scenarios.

All records are simulated. No patient data are used.
The causal truth is written under evaluation/, not in agent-visible metadata.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
OBSERVED_DIR = ROOT / "datasets" / "observed"


def generate(seed: int = 42, n: int = 3000) -> dict:
    if n < 200:
        raise ValueError("n must be at least 200 for the demo scenarios")
    rng = np.random.default_rng(seed)
    distractor_rng = np.random.default_rng(seed + 10000)
    OBSERVED_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "evaluation").mkdir(parents=True, exist_ok=True)

    # Scenario 1: biomarker and disease share a cause (age).
    age = np.clip(rng.normal(60, 12, n), 18, 94)
    biomarker = 10 + 0.8 * (age - 60) + rng.normal(0, 7, n)
    severity = 40 + 0.65 * (age - 60) + rng.normal(0, 7, n)
    pd.DataFrame({
        "age": age,
        "baseline_cholesterol": distractor_rng.normal(185, 30, n),
        "biomarker_x": biomarker,
        "progression_score": severity,
    }).to_csv(OBSERVED_DIR / "biomarker_confounding.csv", index=False)

    # Scenario 2: confounding by indication reverses an apparent treatment effect.
    age2 = np.clip(rng.normal(62, 12, n), 18, 94)
    baseline = rng.normal(0, 1, n)
    probability = expit(-0.35 + 1.45 * baseline + 0.02 * (age2 - 62))
    treatment = rng.binomial(1, probability)
    recovery = (70 - 11 * baseline - 0.2 * (age2 - 62)
                + 5.0 * treatment + rng.normal(0, 5, n))
    after_treatment_marker = 15 + 2.5 * baseline - 1.8 * treatment + rng.normal(0, 2, n)
    pd.DataFrame({
        "age": age2,
        "baseline_blood_pressure": distractor_rng.normal(125, 14, n),
        "baseline_severity": baseline,
        "drug_a": treatment,
        "post_treatment_marker": after_treatment_marker,
        "recovery_score": recovery,
    }).to_csv(OBSERVED_DIR / "treatment_confounding.csv", index=False)

    # Scenario 3: age is a confounder, but a genuine direct biomarker effect exists.
    age3 = np.clip(rng.normal(60, 12, n), 18, 94)
    biomarker3 = 10 + 0.8 * (age3 - 60) + rng.normal(0, 7, n)
    progression3 = (40 + 0.65 * (age3 - 60)
                    + 0.55 * (biomarker3 - 10) + rng.normal(0, 7, n))
    pd.DataFrame({
        "age": age3,
        "baseline_cholesterol": distractor_rng.normal(185, 30, n),
        "biomarker_x": biomarker3,
        "progression_score": progression3,
    }).to_csv(OBSERVED_DIR / "biomarker_true_effect.csv", index=False)

    # Scenario 4: bivariate cross-sectional data; causal direction is not identifiable.
    severity4 = rng.normal(0, 1, n)
    biomarker4 = 1.8 * severity4 + rng.normal(0, 1.5, n)
    pd.DataFrame({
        "biomarker_x": biomarker4,
        "progression_score": severity4,
    }).to_csv(OBSERVED_DIR / "reverse_causation.csv", index=False)

    catalog = {
        "biomarker_confounding": {
            "title": "Biomarker and progression: suspected direct effect",
            "claim": "Higher biomarker X causally increases disease progression.",
            "design": "synthetic observational cohort",
            "exposure": "biomarker_x", "outcome": "progression_score",
            "variables": {
                "age": {"role": "covariate", "timing": "pre_exposure", "description": "Age at baseline (years)"},
                "baseline_cholesterol": {"role": "covariate", "timing": "pre_exposure", "description": "Cholesterol concentration at baseline"},
                "biomarker_x": {"role": "exposure", "timing": "baseline", "description": "Continuous baseline biomarker level"},
                "progression_score": {"role": "outcome", "timing": "follow_up", "description": "Disease progression score; higher means worse"},
            },
        },
        "treatment_confounding": {
            "title": "Drug A and recovery: suspected harm",
            "claim": "Drug A causes worse recovery because treated patients have lower recovery scores.",
            "design": "synthetic observational treatment cohort",
            "exposure": "drug_a", "outcome": "recovery_score",
            "variables": {
                "age": {"role": "covariate", "timing": "pre_exposure", "description": "Age before treatment (years)"},
                "baseline_blood_pressure": {"role": "covariate", "timing": "pre_exposure", "description": "Blood pressure measured before treatment"},
                "baseline_severity": {"role": "covariate", "timing": "pre_exposure", "description": "Disease severity measured before treatment; higher is worse"},
                "drug_a": {"role": "exposure", "timing": "treatment", "description": "Whether Drug A was prescribed (0=no, 1=yes)"},
                "post_treatment_marker": {"role": "covariate", "timing": "post_exposure", "description": "Inflammation marker measured after treatment"},
                "recovery_score": {"role": "outcome", "timing": "follow_up", "description": "Recovery score; higher means better"},
            },
        },
        "biomarker_true_effect": {
            "title": "Biomarker and progression: true effect",
            "claim": "Higher biomarker X causally increases disease progression.",
            "design": "synthetic observational cohort",
            "exposure": "biomarker_x", "outcome": "progression_score",
            "variables": {
                "age": {"role": "covariate", "timing": "pre_exposure", "description": "Age at baseline (years)"},
                "baseline_cholesterol": {"role": "covariate", "timing": "pre_exposure", "description": "Cholesterol concentration at baseline"},
                "biomarker_x": {"role": "exposure", "timing": "baseline", "description": "Continuous baseline biomarker level"},
                "progression_score": {"role": "outcome", "timing": "follow_up", "description": "Disease progression score; higher means worse"},
            },
        },
        "reverse_causation": {
            "title": "Biomarker and progression: unknown direction",
            "claim": "Higher biomarker X causes greater disease progression.",
            "design": "synthetic cross-sectional observational sample",
            "exposure": "biomarker_x", "outcome": "progression_score",
            "variables": {
                "biomarker_x": {"role": "exposure", "timing": "simultaneous", "description": "Biomarker measured at the same visit as progression"},
                "progression_score": {"role": "outcome", "timing": "simultaneous", "description": "Disease progression at that visit"},
            },
        },
    }
    with (ROOT / "datasets" / "catalog.json").open("w") as f:
        json.dump(catalog, f, indent=2)

    # EVALUATOR ONLY. Do not provide this file to the LLM or the tool responses.
    truth = {
        "biomarker_confounding": {
            "causal_edges": [["age", "biomarker_x"], ["age", "progression_score"]],
            "true_direct_effect_per_unit": 0.0,
            "expected_counterhypothesis": "age_confounds_biomarker_and_progression",
        },
        "treatment_confounding": {
            "causal_edges": [["age", "drug_a"], ["baseline_severity", "drug_a"],
                             ["age", "recovery_score"], ["baseline_severity", "recovery_score"],
                             ["drug_a", "recovery_score"]],
            "true_average_treatment_effect": 5.0,
            "expected_counterhypothesis": "confounding_by_indication",
        },
        "biomarker_true_effect": {
            "causal_edges": [["age", "biomarker_x"], ["age", "progression_score"],
                             ["biomarker_x", "progression_score"]],
            "true_direct_effect_per_unit": 0.55,
            "expected_counterhypothesis": "confounding_considered_but_effect_persists",
        },
        "reverse_causation": {
            "causal_edges": [["progression_score", "biomarker_x"]],
            "expected_counterhypothesis": "reverse_causation",
            "observational_direction_identifiable": False,
        },
    }
    with (ROOT / "evaluation" / "ground_truth.json").open("w") as f:
        json.dump(truth, f, indent=2)
    return catalog


if __name__ == "__main__":
    catalog = generate()
    print(f"Generated {len(catalog)} synthetic datasets under {OBSERVED_DIR}")
