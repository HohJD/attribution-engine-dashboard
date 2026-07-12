"""Tests for the estimator, selector, and decomposition modules."""

from __future__ import annotations

import pandas as pd
import pytest

from attribution.decomposition import decompose
from attribution.estimators.its import fit_its
from attribution.estimators.selector import select_method
from attribution.estimators.synth import fit_synthetic_control
from attribution.generator import GeneratorConfig, generate_panel


def test_selector_chooses_synthetic_control_with_many_donors() -> None:
    panel, truth = generate_panel(seed=10, config=GeneratorConfig(n_companies=30, n_months=60, treatment_month=37))
    selection = select_method(panel, truth["treated_id"], truth["treatment_month"])
    assert selection.method == "synthetic_control"


def test_selector_chooses_its_with_few_donors() -> None:
    panel, truth = generate_panel(
        seed=11,
        config=GeneratorConfig(n_companies=6, n_months=60, treatment_month=37),
    )
    selection = select_method(panel, truth["treated_id"], truth["treatment_month"])
    assert selection.method == "its"


def test_synthetic_control_runs() -> None:
    panel, truth = generate_panel(seed=12, config=GeneratorConfig(n_companies=25, n_months=60, treatment_month=37))
    result = fit_synthetic_control(panel, truth["treated_id"], truth["treatment_month"])
    assert result.weights is not None
    assert result.pre_rmspe >= 0
    assert not result.effect.empty


def test_its_runs() -> None:
    panel, truth = generate_panel(seed=13, config=GeneratorConfig(n_companies=6, n_months=60, treatment_month=37))
    result = fit_its(panel, truth["treated_id"], truth["treatment_month"])
    assert result.effect is not None
    assert result.fit_r2 >= 0


def test_decomposition_runs() -> None:
    panel, truth = generate_panel(seed=14, config=GeneratorConfig(n_companies=20, n_months=60, treatment_month=37))
    result = fit_synthetic_control(panel, truth["treated_id"], truth["treatment_month"])
    decomp = decompose(panel, truth["treated_id"], truth["treatment_month"], result.effect)
    assert isinstance(decomp.total_growth, float)
    assert abs(decomp.total_growth - (decomp.market_component + decomp.sector_component + decomp.alpha_component + decomp.unexplained)) < 1e-6
