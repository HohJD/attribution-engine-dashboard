"""Lightweight PE/finance calculation utilities for the Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import newton


def compute_irr(cash_flows: list[float]) -> float | None:
    """Compute IRR for an arbitrary cash-flow stream.

    Uses Newton's method. Returns None if no solution is found.
    """
    if not cash_flows or cash_flows[0] >= 0:
        return None
    n = len(cash_flows)

    def npv(r: float) -> float:
        return sum(cf / (1 + r) ** i for i, cf in enumerate(cash_flows))

    def npv_prime(r: float) -> float:
        return sum(-i * cf / (1 + r) ** (i + 1) for i, cf in enumerate(cash_flows) if i > 0)

    try:
        return float(newton(npv, 0.10, fprime=npv_prime, maxiter=100))
    except (RuntimeError, ValueError):
        try:
            return float(newton(npv, 0.10, maxiter=100))
        except (RuntimeError, ValueError):
            return None


def compute_moic(distributions: float, invested: float) -> float:
    """Multiple on invested capital (MOIC)."""
    if invested == 0:
        return 0.0
    return distributions / invested


def compute_dpi(distributions: float, paid_in: float) -> float:
    """Distributions to paid-in capital."""
    if paid_in == 0:
        return 0.0
    return distributions / paid_in


def compute_tvpi(distributions: float, nav: float, paid_in: float) -> float:
    """Total value to paid-in capital."""
    if paid_in == 0:
        return 0.0
    return (distributions + nav) / paid_in


def lbo_model(
    entry_ebitda: float,
    entry_multiple: float,
    equity_pct: float,
    debt_pct: float,
    ebitda_cagr: float,
    exit_multiple: float,
    years: int,
    cash_sweep_pct: float,
    interest_rate: float,
) -> dict[str, float]:
    """Run a simple LBO model.

    Assumes all debt is paid down proportionally from free cash flow sweep.
    Returns MOIC, IRR, and a simple value creation bridge.
    """
    entry_ev = entry_ebitda * entry_multiple
    equity = entry_ev * equity_pct
    debt = entry_ev * debt_pct
    if equity + debt == 0:
        return {}

    # Simple yearly cash flow: EBITDA grows, cash sweep pays down debt
    remaining_debt = debt
    for _ in range(years):
        # Not a full 3-statement model; we approximate free cash flow as a
        # percentage of EBITDA used for debt paydown.
        ebitda = entry_ebitda * (1 + ebitda_cagr) ** years
        fcf = ebitda * cash_sweep_pct
        interest = remaining_debt * interest_rate
        remaining_debt = max(0.0, remaining_debt - (fcf - interest))

    exit_ebitda = entry_ebitda * (1 + ebitda_cagr) ** years
    exit_ev = exit_ebitda * exit_multiple
    exit_equity = exit_ev - remaining_debt
    moic = exit_equity / equity if equity > 0 else 0.0
    cash_flows = [-equity] + [0.0] * (years - 1) + [exit_equity]
    irr = compute_irr(cash_flows)

    return {
        "entry_ev": entry_ev,
        "equity": equity,
        "debt": debt,
        "exit_ebitda": exit_ebitda,
        "exit_ev": exit_ev,
        "remaining_debt": remaining_debt,
        "exit_equity": exit_equity,
        "moic": moic,
        "irr": irr if irr is not None else 0.0,
        "ebitda_growth_value": exit_ebitda - entry_ebitda,
        "multiple_expansion_value": exit_ebitda * (exit_multiple - entry_multiple),
    }


def dcf_valuation(
    revenue: float,
    growth_rates: list[float],
    ebitda_margin: float,
    tax_rate: float,
    d_and_a_pct: float,
    capex_pct: float,
    nwc_pct: float,
    wacc: float,
    terminal_growth: float,
) -> dict[str, float | pd.Series]:
    """Project a simple unlevered DCF and return enterprise value.
    """
    years = len(growth_rates)
    revs = [revenue]
    for g in growth_rates:
        revs.append(revs[-1] * (1 + g))
    revs = revs[1:]  # exclude starting point
    revenues = pd.Series(revs, index=range(1, years + 1))
    ebitda = revenues * ebitda_margin
    d_and_a = revenues * d_and_a_pct
    capex = revenues * capex_pct
    nwc_change = revenues.diff().fillna(revenues.iloc[0]) * nwc_pct
    tax = ebitda * tax_rate
    fcf = ebitda - tax + d_and_a - capex - nwc_change
    pv = fcf / ((1 + wacc) ** fcf.index)
    terminal_value = fcf.iloc[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** years)
    enterprise_value = pv.sum() + pv_terminal

    return {
        "revenues": revenues,
        "ebitda": ebitda,
        "free_cash_flow": fcf,
        "pv_fcf": pv,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
    }


def compute_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EV/Revenue, EV/EBITDA, and P/E for a comps table.

    Expects columns: company, ev, revenue, ebitda, net_income.
    """
    required = {"company", "ev", "revenue", "ebitda", "net_income"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")
    out = df.copy()
    out["ev_revenue"] = out["ev"] / out["revenue"].replace(0, np.nan)
    out["ev_ebitda"] = out["ev"] / out["ebitda"].replace(0, np.nan)
    out["pe"] = out["ev"] / out["net_income"].replace(0, np.nan)
    return out


def wacc(
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float,
    cost_of_debt: float,
    tax_rate: float,
    debt_pct: float,
    equity_pct: float,
) -> dict[str, float]:
    """Compute cost of equity and WACC."""
    cost_of_equity = risk_free_rate + beta * equity_risk_premium
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
    wacc_value = equity_pct * cost_of_equity + debt_pct * after_tax_cost_of_debt
    return {
        "cost_of_equity": cost_of_equity,
        "after_tax_cost_of_debt": after_tax_cost_of_debt,
        "wacc": wacc_value,
    }


def value_creation_bridge(
    entry_revenue: float,
    exit_revenue: float,
    entry_margin: float,
    exit_margin: float,
    entry_multiple: float,
    exit_multiple: float,
) -> dict[str, float]:
    """Attribute EV change to revenue growth, margin expansion, and multiple expansion."""
    entry_ebitda = entry_revenue * entry_margin
    exit_ebitda = exit_revenue * exit_margin
    entry_ev = entry_ebitda * entry_multiple
    exit_ev = exit_ebitda * exit_multiple

    # Bridge: start with entry EV, then add each component.
    revenue_component = (exit_revenue - entry_revenue) * entry_margin * entry_multiple
    margin_component = exit_revenue * (exit_margin - entry_margin) * entry_multiple
    multiple_component = exit_ebitda * (exit_multiple - entry_multiple)

    return {
        "entry_ev": entry_ev,
        "exit_ev": exit_ev,
        "entry_ebitda": entry_ebitda,
        "exit_ebitda": exit_ebitda,
        "revenue_growth_value": revenue_component,
        "margin_expansion_value": margin_component,
        "multiple_expansion_value": multiple_component,
        "total_value_change": exit_ev - entry_ev,
    }
