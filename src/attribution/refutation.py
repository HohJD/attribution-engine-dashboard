"""Refutation tests for attribution-engine."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from attribution.estimators.selector import MethodSelection
from attribution.estimators.synth import SyntheticControlResult, fit_synthetic_control


@dataclass
class RefutationResult:
    """Aggregated refutation test results."""

    placebo_pseudo_pvalue: float
    placebo_ratios: pd.Series
    backdating_effect: float | None
    backdating_pseudo_pvalue: float | None
    leave_one_out: pd.DataFrame
    pre_rmspe_ratio: float
    reliability_flag: str
    pass_summary: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "placebo_pseudo_pvalue": float(self.placebo_pseudo_pvalue),
            "placebo_ratios": self.placebo_ratios.to_dict(),
            "backdating_effect": self.backdating_effect,
            "backdating_pseudo_pvalue": self.backdating_pseudo_pvalue,
            "leave_one_out": self.leave_one_out.to_dict(),
            "pre_rmspe_ratio": float(self.pre_rmspe_ratio),
            "reliability_flag": self.reliability_flag,
            "pass_summary": self.pass_summary,
        }


def _rmspe_ratio(result: SyntheticControlResult, treatment_month: int, end_month: int | None = None) -> float:
    pre = result.effect.index < treatment_month
    post = result.effect.index >= treatment_month
    if end_month is not None:
        post = post & (result.effect.index < end_month)
    pre_mspe = float(np.mean(result.effect[pre] ** 2))
    post_mspe = float(np.mean(result.effect[post] ** 2))
    if pre_mspe == 0:
        return np.inf
    return float(np.sqrt(post_mspe / pre_mspe))


def in_space_placebo(
    panel: pd.DataFrame,
    treated_id: str,
    treatment_month: int,
    estimator: Callable[..., SyntheticControlResult] = fit_synthetic_control,
    end_month: int | None = None,
) -> tuple[float, pd.Series]:
    """Run synthetic control on each donor as if it were treated.

    Returns the pseudo p-value based on the rank of the treated unit's
    post/pre RMSPE ratio against the placebo distribution.
    """
    donors = [c for c in panel["company_id"].unique() if c != treated_id]
    treated_result = estimator(panel, treated_id, treatment_month)
    treated_ratio = _rmspe_ratio(treated_result, treatment_month, end_month=end_month)

    ratios: dict[str, float] = {}
    for donor in donors:
        try:
            donor_result = estimator(panel, donor, treatment_month)
            ratios[donor] = _rmspe_ratio(donor_result, treatment_month, end_month=end_month)
        except Exception as exc:
            warnings.warn(f"Placebo run for {donor} failed: {exc}")

    ratios_series = pd.Series(ratios)
    all_ratios = pd.concat([ratios_series, pd.Series({treated_id: treated_ratio})])
    all_ratios = all_ratios.replace([np.inf, -np.inf], np.nan).dropna()
    rank = (all_ratios >= treated_ratio).sum()
    n = len(all_ratios)
    pvalue = rank / n if n > 0 else 1.0
    return float(pvalue), ratios_series


def in_time_placebo(
    panel: pd.DataFrame,
    treated_id: str,
    treatment_month: int,
    shift: int = 12,
    estimator: Callable[..., SyntheticControlResult] = fit_synthetic_control,
    end_month: int | None = None,
) -> tuple[float | None, float | None]:
    """Shift treatment date earlier and estimate the effect at the fake date.

    The estimated effect is restricted to the window between the fake date and
    the true treatment month (or ``end_month`` if supplied). A real effect should
    not appear before it actually begins.
    """
    fake_month = max(1, treatment_month - shift)
    if fake_month < 12:
        return None, None
    if end_month is None:
        end_month = treatment_month
    try:
        result = estimator(panel, treated_id, fake_month)
        post = (result.effect.index >= fake_month) & (result.effect.index < end_month)
        post_effect = float(result.effect[post].sum())
        pvalue, _ = in_space_placebo(panel, treated_id, fake_month, estimator=estimator, end_month=end_month)
        return post_effect, pvalue
    except Exception as exc:
        warnings.warn(f"In-time placebo failed: {exc}")
        return None, None


def leave_one_out(
    panel: pd.DataFrame,
    treated_id: str,
    treatment_month: int,
    top_n: int = 5,
    estimator: Callable[..., SyntheticControlResult] = fit_synthetic_control,
) -> pd.DataFrame:
    """Drop each of the top-N weighted donors and re-estimate the effect.

    Returns a DataFrame with the effect estimate and donor weight for each run.
    """
    base = estimator(panel, treated_id, treatment_month)
    top_donors = base.weights.head(top_n).index.tolist()

    rows: list[dict[str, Any]] = []
    post = base.effect.index >= treatment_month
    base_effect = float(base.effect[post].sum())
    rows.append({"dropped": "none", "weight": 1.0, "total_effect": base_effect})

    for donor in top_donors:
        reduced = panel[panel["company_id"] != donor]
        try:
            result = estimator(reduced, treated_id, treatment_month)
            post = result.effect.index >= treatment_month
            effect = float(result.effect[post].sum())
            rows.append({"dropped": donor, "weight": base.weights.get(donor, 0.0), "total_effect": effect})
        except Exception as exc:
            warnings.warn(f"Leave-one-out failed for {donor}: {exc}")
            rows.append({"dropped": donor, "weight": base.weights.get(donor, 0.0), "total_effect": np.nan})

    return pd.DataFrame(rows)


def run_refutation(
    panel: pd.DataFrame,
    treated_id: str,
    treatment_month: int,
    method: MethodSelection,
    estimator: Callable[..., SyntheticControlResult] = fit_synthetic_control,
    rmspe_threshold: float = 0.20,
) -> RefutationResult:
    """Run all refutation tests and return an aggregated result."""
    base_result = estimator(panel, treated_id, treatment_month)
    pre = base_result.effect.index < treatment_month
    pre_rmspe = float(np.sqrt(np.mean(base_result.effect[pre] ** 2)))
    treated_scale = float(base_result.treated[pre].mean())
    pre_rmspe_ratio = pre_rmspe / treated_scale if treated_scale > 0 else np.inf

    reliability_flag = "ok"
    if pre_rmspe_ratio > rmspe_threshold:
        reliability_flag = "unreliable"
    elif pre_rmspe_ratio > 0.5 * rmspe_threshold:
        reliability_flag = "weak"

    placebo_p, placebo_ratios = in_space_placebo(panel, treated_id, treatment_month, estimator=estimator)
    back_effect, back_p = in_time_placebo(panel, treated_id, treatment_month, estimator=estimator)
    loo = leave_one_out(panel, treated_id, treatment_month, estimator=estimator)

    pass_summary: dict[str, str] = {
        "in_space_placebo": "pass" if placebo_p <= 0.1 else "warn",
        "in_time_placebo": "pass" if back_p is not None and back_p > 0.1 else "warn",
        "leave_one_out": "pass" if len(loo) > 1 else "warn",
        "pre_fit": "pass" if reliability_flag == "ok" else "fail",
    }

    return RefutationResult(
        placebo_pseudo_pvalue=placebo_p,
        placebo_ratios=placebo_ratios,
        backdating_effect=back_effect,
        backdating_pseudo_pvalue=back_p,
        leave_one_out=loo,
        pre_rmspe_ratio=pre_rmspe_ratio,
        reliability_flag=reliability_flag,
        pass_summary=pass_summary,
    )
