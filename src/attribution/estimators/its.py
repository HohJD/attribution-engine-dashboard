"""Interrupted time series estimator for attribution-engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper


@dataclass
class ITSResult:
    """Results from the interrupted time series estimator."""

    level_change: float
    level_change_se: float
    trend_change: float
    trend_change_se: float
    pre_trend: float
    pre_level: float
    effect: pd.Series
    counterfactual: pd.Series
    treated: pd.Series
    model: RegressionResultsWrapper
    fit_r2: float
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "level_change": float(self.level_change),
            "level_change_se": float(self.level_change_se),
            "trend_change": float(self.trend_change),
            "trend_change_se": float(self.trend_change_se),
            "fit_r2": float(self.fit_r2),
            "effect": self.effect.to_dict(),
            "counterfactual": self.counterfactual.to_dict(),
            "treated": self.treated.to_dict(),
        }


def fit_its(
    panel: pd.DataFrame,
    treated_id: str,
    treatment_month: int,
    value_col: str = "ebitda",
    use_covariates: bool = False,
    market_series: pd.Series | None = None,
    sector_series: pd.Series | None = None,
) -> ITSResult:
    """Fit a segmented regression on the log of the treated unit's outcome.

    Parameters
    ----------
    panel
        Long-format panel.
    treated_id
        Identifier of the treated company.
    treatment_month
        1-based month index where the post-period begins.
    value_col
        Outcome column (default ``ebitda``).
    use_covariates
        If True, include ``market_series`` and ``sector_series`` as controls.
    market_series
        Monthly market index aligned with the treated unit's months.
    sector_series
        Monthly sector index aligned with the treated unit's months.

    Returns
    -------
    ITSResult with the estimated treatment effects and fitted model.
    """
    df = panel[panel["company_id"] == treated_id].sort_values("month").copy()
    if value_col not in df.columns:
        raise ValueError(f"Column {value_col} not found in panel")

    y = np.log(df[value_col].replace(0, np.nan).dropna())
    months = df.loc[y.index, "month"].astype(int).values
    y = y.values

    post = (months >= treatment_month).astype(int)
    trend = months - months[0]
    post_trend = post * (months - treatment_month + 1)
    post_trend = np.clip(post_trend, 0, None)

    X = pd.DataFrame(
        {
            "const": np.ones(len(months)),
            "trend": trend,
            "post": post,
            "post_trend": post_trend,
        },
        index=months,
    )

    if use_covariates:
        if market_series is not None:
            X["market"] = market_series.reindex(months).values
        if sector_series is not None:
            X["sector"] = sector_series.reindex(months).values

    # Newey-West with automatic lag selection: rule of thumb 4 * (n/100)^(2/9)
    nobs = len(y)
    nw_lags = max(1, int(4 * (nobs / 100) ** (2 / 9)))

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

    pre_level = float(model.params["const"])
    pre_trend = float(model.params["trend"])
    level_change = float(model.params["post"])
    trend_change = float(model.params["post_trend"])
    level_change_se = float(model.bse["post"])
    trend_change_se = float(model.bse["post_trend"])

    counterfactual_log = (
        model.params["const"]
        + model.params["trend"] * trend
        + (model.params.get("market", 0.0) * X["market"].values if "market" in X.columns else 0)
        + (model.params.get("sector", 0.0) * X["sector"].values if "sector" in X.columns else 0)
    )
    counterfactual = pd.Series(np.exp(counterfactual_log), index=months)
    treated_series = pd.Series(np.exp(y), index=months)
    effect = treated_series - counterfactual

    return ITSResult(
        level_change=level_change,
        level_change_se=level_change_se,
        trend_change=trend_change,
        trend_change_se=trend_change_se,
        pre_trend=pre_trend,
        pre_level=pre_level,
        effect=effect,
        counterfactual=counterfactual,
        treated=treated_series,
        model=model,
        fit_r2=float(model.rsquared),
        method="ITS with Newey-West SEs",
    )
