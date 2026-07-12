"""Tests for the synthetic panel generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attribution.generator import EffectShape, GeneratorConfig, generate_panel


def test_generator_basic() -> None:
    panel, truth = generate_panel(seed=42, config=GeneratorConfig())
    assert isinstance(panel, pd.DataFrame)
    assert set(panel.columns) == {"company_id", "sector", "month", "revenue", "ebitda", "margin"}
    assert truth["treated_id"] in panel["company_id"].unique()
    assert truth["treatment_month"] == 61


def test_generator_step_effect() -> None:
    config = GeneratorConfig(
        n_companies=20,
        n_months=60,
        treatment_month=37,
        effect_shape=EffectShape.STEP,
        effect_size=0.10,
    )
    panel, truth = generate_panel(seed=1, config=config)
    no_effect_config = GeneratorConfig(**config.__dict__)
    no_effect_config.effect_size = 0.0
    panel_no, _ = generate_panel(seed=1, config=no_effect_config)

    treated = panel[panel["company_id"] == truth["treated_id"]].set_index("month").sort_index()
    treated_no = panel_no[panel_no["company_id"] == truth["treated_id"]].set_index("month").sort_index()
    post = treated.loc[37:, "ebitda"]
    post_no = treated_no.loc[37:, "ebitda"]
    assert post.mean() > post_no.mean() * 1.05


def test_generator_ramp_effect() -> None:
    panel, truth = generate_panel(
        seed=2,
        config=GeneratorConfig(
            effect_shape=EffectShape.RAMP,
            effect_size=0.10,
            ramp_months=12,
        ),
    )
    true_effect = np.array(truth["true_effect"])
    assert true_effect[truth["treatment_month"] - 1] == pytest.approx(0.0, abs=1e-9)
    assert true_effect[-1] == pytest.approx(0.10, abs=1e-9)


def test_generator_growth_effect() -> None:
    panel, truth = generate_panel(
        seed=3,
        config=GeneratorConfig(effect_shape=EffectShape.GROWTH, effect_size=0.10),
    )
    true_effect = np.array(truth["true_effect"])
    assert true_effect[truth["treatment_month"] - 1] == pytest.approx(0.0, abs=1e-9)
    assert true_effect[-1] > 0.0


def test_generator_missing_data() -> None:
    panel, _ = generate_panel(
        seed=4,
        config=GeneratorConfig(missing_pct=0.1),
    )
    assert panel["ebitda"].isna().sum() > 0


def test_generator_confounded_donor() -> None:
    _, truth = generate_panel(
        seed=5,
        config=GeneratorConfig(confounded_donor=True),
    )
    assert truth["confounded_id"] is not None
    assert truth["confounded_id"] != truth["treated_id"]
