"""Decompose post-treatment EBITDA growth into market, sector, and alpha components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class DecompositionResult:
    """Results of the post-treatment growth decomposition."""

    market_component: float
    sector_component: float
    alpha_component: float
    unexplained: float
    total_growth: float
    beta_market: float
    beta_sector: float
    market_index: pd.Series
    sector_index: pd.Series

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_component": float(self.market_component),
            "sector_component": float(self.sector_component),
            "alpha_component": float(self.alpha_component),
            "unexplained": float(self.unexplained),
            "total_growth": float(self.total_growth),
            "beta_market": float(self.beta_market),
            "beta_sector": float(self.beta_sector),
            "market_index": self.market_index.to_dict(),
            "sector_index": self.sector_index.to_dict(),
        }

    def to_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "component": [
                    "market",
                    "sector",
                    "alpha",
                    "unexplained",
                    "total_growth",
                ],
                "value": [
                    self.market_component,
                    self.sector_component,
                    self.alpha_component,
                    self.unexplained,
                    self.total_growth,
                ],
            }
        )


def _index_from_donors(panel: pd.DataFrame, treated_id: str, treatment_month: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Build market and sector indices from cross-donor averages.

    Returns the monthly market index and the treated unit's sector index.
    """
    df = panel.copy()
    donors = df[df["company_id"] != treated_id]
    market_index = (
        donors.groupby("month")["ebitda"].mean()
    )
    # normalize to start at 1
    market_index = market_index / market_index.iloc[0]

    treated_sector = df[df["company_id"] == treated_id]["sector"].iloc[0]
    sector_donors = donors[donors["sector"] == treated_sector]
    sector_index = (
        sector_donors.groupby("month")["ebitda"].mean()
    )
    sector_index = sector_index / sector_index.iloc[0]

    treated_df = df[df["company_id"] == treated_id].set_index("month").sort_index()
    treated_ebitda = treated_df["ebitda"]
    treated_revenue = treated_df["revenue"]

    return market_index, sector_index, treated_ebitda, treated_revenue


def decompose(
    panel: pd.DataFrame,
    treated_id: str,
    treatment_month: int,
    effect: pd.Series,
) -> DecompositionResult:
    """Decompose post-treatment EBITDA growth into market, sector, and alpha.

    The market beta is estimated by regressing the treated unit's pre-treatment
    EBITDA on the market index (cross-donor average). The sector beta is
    estimated by regressing the residual on the sector index (cross-donor
    same-sector average), net of the market.

    Components are expressed as the total *change* in EBITDA from the last
    pre-treatment month to the end of the post-period, in the same units as the
    EBITDA series.

    Parameters
    ----------
    panel
        Long-format panel.
    treated_id
        Identifier of the treated company.
    treatment_month
        1-based month index where the post-period begins.
    effect
        Estimated monthly treatment effect from the chosen estimator.

    Returns
    -------
    DecompositionResult
    """
    market_index, sector_index, treated_ebitda, treated_revenue = _index_from_donors(
        panel, treated_id, treatment_month
    )
    pre = treated_ebitda.index < treatment_month
    post = treated_ebitda.index >= treatment_month

    # Pre-treatment regression of log(EBITDA) on log(market index)
    y_pre = np.log(treated_ebitda[pre])
    mkt_pre = np.log(market_index.reindex(y_pre.index).dropna())
    common = y_pre.index.intersection(mkt_pre.index)
    y_arr = y_pre.loc[common].values
    X_arr = sm.add_constant(mkt_pre.loc[common].values)
    beta_mkt_model = sm.OLS(y_arr, X_arr).fit()
    beta_market = float(beta_mkt_model.params[1])

    # Sector beta: residual vs sector index, net of market
    resid = y_arr - beta_market * mkt_pre.loc[common].values
    sec_pre = np.log(sector_index.reindex(common).dropna())
    common2 = common.intersection(sec_pre.index)
    if len(common2) > 2:
        X_sec = sm.add_constant(sec_pre.loc[common2].values)
        sec_model = sm.OLS(resid[: len(common2)], X_sec).fit()
        beta_sector = float(sec_model.params[1])
    else:
        beta_sector = 0.0

    # Post-period growth decomposition
    last_pre = treated_ebitda[pre].iloc[-1]
    last_post = treated_ebitda[post].iloc[-1]
    total_growth = last_post - last_pre

    market_index_pre = market_index.reindex(treated_ebitda[pre].index)
    market_index_post = market_index.reindex(treated_ebitda[post].index)
    market_change = market_index_post.iloc[-1] - market_index_pre.iloc[-1]
    market_component = beta_market * market_change * last_pre

    sector_index_pre = sector_index.reindex(treated_ebitda[pre].index)
    sector_index_post = sector_index.reindex(treated_ebitda[post].index)
    sector_change = sector_index_post.iloc[-1] - sector_index_pre.iloc[-1]
    sector_component = beta_sector * sector_change * last_pre

    alpha_component = float(effect[post].sum())
    unexplained = total_growth - (market_component + sector_component + alpha_component)

    return DecompositionResult(
        market_component=market_component,
        sector_component=sector_component,
        alpha_component=alpha_component,
        unexplained=unexplained,
        total_growth=total_growth,
        beta_market=beta_market,
        beta_sector=beta_sector,
        market_index=market_index,
        sector_index=sector_index,
    )
