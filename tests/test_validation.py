"""Validation tests against the ground-truth generator.

These tests assert that the estimator recovers known treatment effects and
behaves sensibly under null, backdating, and noisy settings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attribution.decomposition import decompose
from attribution.estimators.selector import select_method
from attribution.estimators.synth import fit_synthetic_control
from attribution.generator import EffectShape, GeneratorConfig, generate_panel
from attribution.refutation import run_refutation


def _estimate_total_effect(panel: pd.DataFrame, treated_id: str, treatment_month: int) -> tuple[float, float]:
    """Return total effect and placebo pseudo p-value using synthetic control."""
    result = fit_synthetic_control(panel, treated_id, treatment_month)
    post = result.effect.index >= treatment_month
    total_effect = float(result.effect[post].sum())

    refutation = run_refutation(panel, treated_id, treatment_month, select_method(panel, treated_id, treatment_month))
    return total_effect, refutation.placebo_pseudo_pvalue, refutation


def test_recovery_step_effect() -> None:
    """Recovery: +10% step effect within ±30% of truth and placebo p <= 0.1."""
    successes = 0
    n_seeds = 5
    for seed in range(n_seeds):
        panel, truth = generate_panel(
            seed=seed,
            config=GeneratorConfig(
                n_companies=25,
                n_months=72,
                treatment_month=49,
                effect_shape=EffectShape.STEP,
                effect_size=0.10,
                noise_scale=1.0,
            ),
        )
        total_effect, pvalue, _ = _estimate_total_effect(panel, truth["treated_id"], truth["treatment_month"])

        treated = panel[panel["company_id"] == truth["treated_id"]].set_index("month").sort_index()
        pre_mean = treated.loc[: truth["treatment_month"] - 1, "ebitda"].mean()
        post_months = treated.index[treated.index >= truth["treatment_month"]]
        true_total = pre_mean * 0.10 * len(post_months)

        if abs(total_effect - true_total) / abs(true_total) <= 0.30 and pvalue <= 0.1:
            successes += 1

    assert successes >= 4, f"Recovery test passed {successes}/{n_seeds} seeds"


def test_null_calibration() -> None:
    """Null calibration: no true effect should rarely be flagged."""
    non_significant = 0
    n_seeds = 5
    effects = []
    for seed in range(n_seeds):
        panel, truth = generate_panel(
            seed=seed + 100,
            config=GeneratorConfig(
                n_companies=25,
                n_months=72,
                treatment_month=49,
                effect_shape=EffectShape.STEP,
                effect_size=0.0,
                noise_scale=1.0,
            ),
        )
        total_effect, pvalue, _ = _estimate_total_effect(panel, truth["treated_id"], truth["treatment_month"])
        effects.append(total_effect)
        if pvalue > 0.1:
            non_significant += 1

    assert non_significant >= 4, f"Only {non_significant}/5 null seeds were non-significant; effects: {effects}"


def test_backdating_sanity() -> None:
    """Backdating placebo should not find a significant effect at the fake date."""
    panel, truth = generate_panel(
        seed=200,
        config=GeneratorConfig(
            n_companies=25,
            n_months=72,
            treatment_month=49,
            effect_shape=EffectShape.STEP,
            effect_size=0.10,
        ),
    )
    method = select_method(panel, truth["treated_id"], truth["treatment_month"])
    refutation = run_refutation(panel, truth["treated_id"], truth["treatment_month"], method)

    assert refutation.backdating_pseudo_pvalue is not None
    assert refutation.backdating_pseudo_pvalue > 0.1, "Backdating should not be flagged significant"
    assert abs(refutation.backdating_effect) < 0.5 * 25, "Backdated effect should be small"  # heuristic upper bound


def test_selector_behaviour() -> None:
    """Selector chooses ITS with few donors and SC with many donors + long pre-period."""
    panel_few, truth_few = generate_panel(
        seed=300,
        config=GeneratorConfig(n_companies=6, n_months=60, treatment_month=37),
    )
    sel_few = select_method(panel_few, truth_few["treated_id"], truth_few["treatment_month"])
    assert sel_few.method == "its"

    panel_many, truth_many = generate_panel(
        seed=301,
        config=GeneratorConfig(n_companies=25, n_months=60, treatment_month=37),
    )
    sel_many = select_method(panel_many, truth_many["treated_id"], truth_many["treatment_month"])
    assert sel_many.method == "synthetic_control"


def test_poor_fit_honesty() -> None:
    """With extreme noise the report flags unreliability."""
    panel, truth = generate_panel(
        seed=400,
        config=GeneratorConfig(
            n_companies=25,
            n_months=72,
            treatment_month=49,
            effect_shape=EffectShape.STEP,
            effect_size=0.10,
            noise_scale=5.0,
        ),
    )
    refutation = run_refutation(
        panel,
        truth["treated_id"],
        truth["treatment_month"],
        select_method(panel, truth["treated_id"], truth["treatment_month"]),
    )
    assert refutation.reliability_flag in {"weak", "unreliable"}
