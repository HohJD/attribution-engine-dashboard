"""Synthetic control estimator for attribution-engine.

Implements the Abadie et al. synthetic control method. The implementation
tries to use ``pysyncon``; if it is unavailable or raises an error, it falls
back to a vanilla quadratic-programming formulation using ``scipy.optimize``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


try:
    import pysyncon

    _PYSYNCON_AVAILABLE = True
except Exception:  # pragma: no cover
    pysyncon = None
    _PYSYNCON_AVAILABLE = False


@dataclass
class SyntheticControlResult:
    """Results from the synthetic control estimator."""

    weights: pd.Series
    pre_rmspe: float
    effect: pd.Series
    counterfactual: pd.Series
    treated: pd.Series
    method: str
    predictors: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "pre_rmspe": float(self.pre_rmspe),
            "weights": self.weights.to_dict(),
            "effect": self.effect.to_dict(),
            "counterfactual": self.counterfactual.to_dict(),
            "treated": self.treated.to_dict(),
            "predictors": self.predictors.to_dict(),
        }


def _prepare_data(
    panel: pd.DataFrame,
    treated_id: str,
    treatment_month: int,
    value_col: str = "ebitda",
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Pivot the panel into wide matrices and compute predictors."""
    pivot = panel.pivot(index="month", columns="company_id", values=value_col)
    pivot = pivot.sort_index()
    pivot_rev = panel.pivot(index="month", columns="company_id", values="revenue")
    pivot_rev = pivot_rev.sort_index()
    pivot_margin = panel.pivot(index="month", columns="company_id", values="margin")
    pivot_margin = pivot_margin.sort_index()

    treated = pivot[treated_id].copy()
    donors = pivot.drop(columns=[treated_id])

    pre_months = pivot.index < treatment_month
    post_months = pivot.index >= treatment_month

    # Build predictors: pre-period means of level, growth, margin, revenue growth
    def _mean_window(series: pd.Series, window: int = 6) -> float:
        valid = series[pre_months].dropna()
        if len(valid) == 0:
            return 0.0
        return float(valid.iloc[-window:].mean())

    def _growth_mean(series: pd.Series, window: int = 6) -> float:
        valid = series[pre_months].dropna()
        if len(valid) <= 1:
            return 0.0
        growth = valid.pct_change().iloc[-window:].dropna()
        return float(growth.mean()) if len(growth) > 0 else 0.0

    predictors_treated: dict[str, float] = {}
    predictors_donors: dict[str, pd.Series] = {}
    for name, series in [
        ("ebitda_level", treated),
        ("ebitda_growth", treated),
        ("margin", pivot_margin[treated_id]),
        ("revenue_growth", pivot_rev[treated_id]),
    ]:
        if name in ("ebitda_growth", "revenue_growth"):
            predictors_treated[name] = _growth_mean(series)
            predictors_donors[name] = pivot.apply(
                lambda col: _growth_mean(col)
            )
        elif name == "margin":
            predictors_treated[name] = _mean_window(series)
            predictors_donors[name] = pivot_margin.apply(lambda col: _mean_window(col))
        else:
            predictors_treated[name] = _mean_window(series)
            predictors_donors[name] = pivot.apply(lambda col: _mean_window(col))

    Y_pre = treated[pre_months].dropna()
    Y_post = treated[post_months].dropna()
    Y_donors_pre = donors.loc[Y_pre.index]
    Y_donors_post = donors.loc[Y_post.index]

    # Drop donors with any missing pre-period observations
    valid_cols = Y_donors_pre.columns[~Y_donors_pre.isna().any()].tolist()
    if len(valid_cols) == 0:
        raise ValueError("No donors have complete pre-treatment data.")
    Y_donors_pre = Y_donors_pre[valid_cols]
    Y_donors_post = Y_donors_post[valid_cols]
    donors = donors[valid_cols]

    X1 = pd.DataFrame(predictors_treated, index=[treated_id]).T
    X0 = pd.DataFrame(predictors_donors).T
    X0 = X0.reindex(columns=valid_cols)

    # Normalize predictors to equalize scales
    stacked = pd.concat([X1, X0], axis=1)
    means = stacked.mean(axis=1).replace(0, np.nan)
    sds = stacked.std(axis=1).replace(0, np.nan)
    X1_norm = ((X1[treated_id] - means) / sds).fillna(0).to_frame(treated_id)
    X0_norm = X0.sub(means, axis=0).div(sds, axis=0).fillna(0)

    return Y_pre, Y_post, Y_donors_pre, Y_donors_post, X1_norm, X0_norm, treated, donors


def _solve_synthetic_control(
    X0: pd.DataFrame, x1: pd.Series, Y0_pre: pd.DataFrame, y1_pre: pd.Series
) -> tuple[np.ndarray, float]:
    """Solve for donor weights by minimizing pre-period prediction error.

    The objective combines the predictor match (Abadie-style) and the
    pre-period fit so the weights fit both the predictors and the actual
    pre-treatment series.
    """
    n_donors = X0.shape[1]
    if n_donors == 0:
        raise ValueError("No usable donors remain after filtering.")

    X0m = X0.values
    x1v = x1.values
    Y0m = Y0_pre.values
    y1v = y1_pre.values

    # Scale matrices to unit-less losses
    X0s = X0m / (np.linalg.norm(x1v) + 1e-12)
    x1s = x1v / (np.linalg.norm(x1v) + 1e-12)
    Y0s = Y0m / (np.linalg.norm(y1v) + 1e-12)
    y1s = y1v / (np.linalg.norm(y1v) + 1e-12)

    def objective(w: np.ndarray) -> float:
        pred_err = np.sum((x1s - X0s @ w) ** 2)
        fit_err = np.sum((y1s - Y0s @ w) ** 2)
        return float(pred_err + fit_err)

    def grad(w: np.ndarray) -> np.ndarray:
        return -2 * X0s.T @ (x1s - X0s @ w) - 2 * Y0s.T @ (y1s - Y0s @ w)

    w0 = np.full(n_donors, 1.0 / n_donors)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_donors

    result = minimize(
        objective,
        w0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        warnings.warn(f"Synthetic control optimizer did not converge: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    weights = weights / (weights.sum() + 1e-12)
    pre_rmspe = float(np.sqrt(np.mean((y1v - Y0m @ weights) ** 2)))
    return weights, pre_rmspe


def fit_synthetic_control(
    panel: pd.DataFrame,
    treated_id: str,
    treatment_month: int,
    value_col: str = "ebitda",
    try_pysyncon: bool = False,
) -> SyntheticControlResult:
    """Fit a synthetic control on ``panel`` for ``treated_id``.

    Parameters
    ----------
    panel
        Long-format panel with ``company_id``, ``month``, ``ebitda``,
        ``revenue``, and ``margin``.
    treated_id
        Identifier of the treated company.
    treatment_month
        1-based month index where the post-period begins.
    value_col
        Column to use as the outcome (default ``ebitda``).
    try_pysyncon
        If True and pysyncon is installed, use it; otherwise fall back.

    Returns
    -------
    SyntheticControlResult with weights, counterfactual, and effects.
    """
    (
        Y_pre,
        Y_post,
        Y_donors_pre,
        Y_donors_post,
        X1,
        X0,
        treated,
        donors,
    ) = _prepare_data(panel, treated_id, treatment_month, value_col)

    if try_pysyncon and _PYSYNCON_AVAILABLE and pysyncon is not None:
        try:
            return _fit_pysyncon(
                Y_pre,
                Y_post,
                Y_donors_pre,
                Y_donors_post,
                X1,
                X0,
                treated,
                donors,
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"pysyncon failed ({exc}); using vanilla synthetic control.")

    return _fit_vanilla(
        Y_pre,
        Y_post,
        Y_donors_pre,
        Y_donors_post,
        X1,
        X0,
        treated,
        donors,
    )


def _fit_vanilla(
    Y_pre: pd.Series,
    Y_post: pd.Series,
    Y_donors_pre: pd.DataFrame,
    Y_donors_post: pd.DataFrame,
    X1: pd.DataFrame,
    X0: pd.DataFrame,
    treated: pd.Series,
    donors: pd.DataFrame,
) -> SyntheticControlResult:
    x1 = X1[treated.name]
    weights, pre_rmspe = _solve_synthetic_control(X0, x1, Y_donors_pre, Y_pre)
    weights_series = pd.Series(weights, index=donors.columns, name="weight")
    weights_series = weights_series[weights_series > 1e-6].sort_values(ascending=False)

    counterfactual = donors @ weights
    counterfactual = counterfactual.reindex(treated.index)
    effect = treated - counterfactual

    return SyntheticControlResult(
        weights=weights_series,
        pre_rmspe=pre_rmspe,
        effect=effect,
        counterfactual=counterfactual,
        treated=treated,
        method="vanilla synthetic control (scipy)",
        predictors=X1,
    )


def _fit_pysyncon(
    Y_pre: pd.Series,
    Y_post: pd.Series,
    Y_donors_pre: pd.DataFrame,
    Y_donors_post: pd.DataFrame,
    X1: pd.DataFrame,
    X0: pd.DataFrame,
    treated: pd.Series,
    donors: pd.DataFrame,
) -> SyntheticControlResult:
    """Fit using pysyncon."""
    x1 = X1[treated.name]
    X0t = X0.T
    X1t = x1.to_frame().T
    X1t.index = [treated.name]
    X0t.index = donors.columns

    # pysyncon expects predictors in the shape where rows are units and columns are predictors
    X_pre = X1t.join(X0t.T, how="outer").T
    X_pre = X_pre.fillna(0)

    # Build the pysyncon Dataprep
    dataprep = pysyncon.Dataprep(
        foo=treated.to_frame(),
        predictors=X_pre,
        predictors_op="mean",
        special_predictors=[
            ("ebitda", Y_pre.index, "mean"),
        ],
        dependent="ebitda",
        unit_variable="company_id",
        time_variable="month",
        treatment_identifier=treated.name,
        controls_identifier=list(donors.columns),
        time_predictors_prior=Y_pre.index.min(),
        time_optimization=Y_pre.index.min(),
        time_post=int(Y_post.index.min()) if len(Y_post) > 0 else Y_pre.index.max() + 1,
    )
    synth = pysyncon.Synth()
    synth.fit(dataprep=dataprep)
    weights_series = pd.Series(synth.w, index=donors.columns, name="weight")
    weights_series = weights_series[weights_series > 1e-6].sort_values(ascending=False)

    counterfactual = donors @ weights_series.reindex(donors.columns).fillna(0).values
    counterfactual = counterfactual.reindex(treated.index)
    effect = treated - counterfactual
    pre_rmspe = float(np.sqrt(np.mean((Y_pre - counterfactual.loc[Y_pre.index]) ** 2)))

    return SyntheticControlResult(
        weights=weights_series,
        pre_rmspe=pre_rmspe,
        effect=effect,
        counterfactual=counterfactual,
        treated=treated,
        method="pysyncon",
        predictors=X1,
    )
