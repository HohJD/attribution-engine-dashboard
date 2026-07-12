"""Synthetic panel data generator for attribution-engine.

Generates a monthly panel of companies with a shared market factor, sector
factors, and idiosyncratic noise. One company is designated as the portco and
receives a known treatment effect from a specified month onward.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class EffectShape(str, Enum):
    """Supported treatment effect shapes."""

    STEP = "step"
    RAMP = "ramp"
    GROWTH = "growth"


@dataclass
class GeneratorConfig:
    """Configuration for the synthetic panel generator.

    Parameters
    ----------
    n_companies
        Number of companies in the panel (including the portco).
    n_months
        Total number of months in the panel.
    n_sectors
        Number of sectors.
    treatment_month
        1-based month index at which the treatment begins.
    effect_shape
        Shape of the injected effect.
    effect_size
        Magnitude parameter for the effect.
    ramp_months
        For ``EffectShape.RAMP``, number of months to reach full effect.
    noise_scale
        Scaling factor for the idiosyncratic noise.
    missing_pct
        Probability of missing EBITDA/revenue for a donor in a given month.
    confounded_donor
        If True, one randomly selected donor receives a spillover effect.
    market_persistence
        AR(1) coefficient for the market factor.
    sector_persistence
        AR(1) coefficient for sector factors.
    """

    n_companies: int = 40
    n_months: int = 84
    n_sectors: int = 4
    treatment_month: int = 61
    effect_shape: EffectShape = EffectShape.STEP
    effect_size: float = 0.10
    ramp_months: int = 12
    noise_scale: float = 1.0
    missing_pct: float = 0.0
    confounded_donor: bool = False
    market_persistence: float = 0.85
    sector_persistence: float = 0.75

    def __post_init__(self) -> None:
        if self.treatment_month < 12 or self.treatment_month >= self.n_months:
            raise ValueError("treatment_month must be in [12, n_months)")
        if self.effect_shape == EffectShape.RAMP and self.ramp_months <= 0:
            raise ValueError("ramp_months must be positive")


def _simulate_ar1(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate a stationary AR(1) series with zero mean."""
    series = np.empty(n)
    series[0] = rng.normal(0, sigma / np.sqrt(1 - rho**2))
    for t in range(1, n):
        series[t] = rho * series[t - 1] + rng.normal(0, sigma)
    return series


def generate_panel(
    seed: int | None = None,
    config: GeneratorConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a synthetic monthly panel and the ground-truth treatment effect.

    Parameters
    ----------
    seed
        Random seed for reproducibility.
    config
        Generator configuration. Uses defaults if None.

    Returns
    -------
    panel_df
        DataFrame with columns ``company_id``, ``sector``, ``month``,
        ``revenue_growth``, ``revenue``, ``ebitda``, and ``margin``.
    truth
        Dictionary containing the treated id, treatment month, effect shape,
        and the true monthly effect on EBITDA.
    """
    config = config or GeneratorConfig()
    rng = np.random.default_rng(seed)

    n = config.n_months
    k = config.n_sectors
    ids = [f"C{i:03d}" for i in range(config.n_companies)]
    sectors = [f"S{rng.integers(0, k):01d}" for i in range(config.n_companies)]
    sector_ids = [int(s[1]) for s in sectors]

    # Market factor
    market = _simulate_ar1(n, config.market_persistence, 0.02, rng)

    # Sector factors
    sector_factors = np.stack(
        [_simulate_ar1(n, config.sector_persistence, 0.015, rng) for _ in range(k)]
    )

    # Loadings
    market_betas = rng.normal(1.0, 0.25, config.n_companies)
    sector_loadings = rng.normal(1.0, 0.30, config.n_companies)
    idio_vol = rng.uniform(0.005, 0.025, config.n_companies) * config.noise_scale

    # Base margin depends on sector and a company-level random effect
    base_margins = rng.uniform(0.08, 0.35, config.n_companies)
    margin_means = base_margins.copy()
    margin_persistence = 0.90

    # Sector-level mean margin
    sector_margin_means = [0.15, 0.22, 0.28, 0.10][:k]
    for i, s in enumerate(sector_ids):
        margin_means[i] = 0.6 * margin_means[i] + 0.4 * sector_margin_means[s]

    # Designate portco: highest pre-treatment revenue scale
    treated_idx = int(rng.integers(0, config.n_companies))
    treated_id = ids[treated_idx]

    records: list[dict[str, Any]] = []
    revenue = np.ones((config.n_companies, n)) * 10.0
    margin = np.empty((config.n_companies, n))
    margin[:, 0] = margin_means
    ebitda = np.empty((config.n_companies, n))

    # True effect on the treated company (multiplicative on EBITDA)
    true_effect = np.zeros(n)
    post = np.arange(n) >= config.treatment_month - 1
    if config.effect_shape == EffectShape.STEP:
        true_effect[post] = config.effect_size
    elif config.effect_shape == EffectShape.RAMP:
        ramp_idx = np.arange(n) - (config.treatment_month - 1)
        ramp = np.clip(ramp_idx / config.ramp_months, 0, 1)
        true_effect[post] = config.effect_size * ramp[post]
    elif config.effect_shape == EffectShape.GROWTH:
        post_idx = np.arange(n) - (config.treatment_month - 1)
        true_effect[post] = config.effect_size * post_idx[post] / 100.0

    for i in range(config.n_companies):
        for t in range(n):
            if t == 0:
                rev_growth = (
                    market_betas[i] * market[t]
                    + sector_loadings[i] * sector_factors[sector_ids[i], t]
                    + rng.normal(0, idio_vol[i])
                )
                revenue[i, t] = 100.0 * np.exp(rev_growth)
            else:
                prev_margin = margin[i, t - 1]
                margin[i, t] = (
                    margin_means[i] * (1 - margin_persistence)
                    + margin_persistence * prev_margin
                    + rng.normal(0, 0.003)
                )
                margin[i, t] = max(margin[i, t], 0.02)
                rev_growth = (
                    market_betas[i] * market[t]
                    + sector_loadings[i] * sector_factors[sector_ids[i], t]
                    + rng.normal(0, idio_vol[i])
                )
                revenue[i, t] = revenue[i, t - 1] * np.exp(rev_growth)

            ebitda[i, t] = revenue[i, t] * margin[i, t]

    # Apply treatment to portco
    for t in range(n):
        if t >= config.treatment_month - 1:
            ebitda[treated_idx, t] = ebitda[treated_idx, t] * (1 + true_effect[t])

    # Optionally inject a spillover effect on one donor
    confounded_id: str | None = None
    if config.confounded_donor:
        donor_candidates = [i for i in range(config.n_companies) if i != treated_idx]
        confounded_idx = int(rng.choice(donor_candidates))
        confounded_id = ids[confounded_idx]
        for t in range(n):
            if t >= config.treatment_month - 1:
                ebitda[confounded_idx, t] = ebitda[confounded_idx, t] * (1 + 0.5 * true_effect[t])

    # Missing data injection
    for i in range(config.n_companies):
        if i == treated_idx:
            continue
        missing = rng.random(n) < config.missing_pct
        for t in np.where(missing)[0]:
            revenue[i, t] = np.nan
            ebitda[i, t] = np.nan
            margin[i, t] = np.nan

    for i, cid in enumerate(ids):
        for t in range(n):
            if i == treated_idx or not np.isnan(ebitda[i, t]):
                records.append(
                    {
                        "company_id": cid,
                        "sector": sectors[i],
                        "month": t + 1,
                        "revenue": revenue[i, t],
                        "ebitda": ebitda[i, t],
                        "margin": margin[i, t],
                    }
                )
            elif i != treated_idx and config.missing_pct > 0:
                records.append(
                    {
                        "company_id": cid,
                        "sector": sectors[i],
                        "month": t + 1,
                        "revenue": np.nan,
                        "ebitda": np.nan,
                        "margin": np.nan,
                    }
                )

    panel_df = pd.DataFrame(records)
    panel_df = panel_df.sort_values(["company_id", "month"]).reset_index(drop=True)

    truth = {
        "treated_id": treated_id,
        "treatment_month": config.treatment_month,
        "effect_shape": config.effect_shape.value,
        "effect_size": config.effect_size,
        "ramp_months": config.ramp_months,
        "true_effect": true_effect.tolist(),
        "confounded_id": confounded_id,
    }

    return panel_df, truth
