"""Cost of capital and WACC calculator."""

from __future__ import annotations

import streamlit as st

from attribution.finance_tools import wacc

st.set_page_config(page_title="WACC", layout="wide")

st.title("WACC & Cost of Capital")
st.markdown(
    "Calculate the weighted average cost of capital using the CAPM framework for equity."
)

with st.form("wacc_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_free = st.number_input("Risk-free rate", value=0.04, step=0.01, format="%.2f")
        beta = st.number_input("Beta", value=1.10, step=0.05, format="%.2f")
        erp = st.number_input("Equity risk premium", value=0.05, step=0.01, format="%.2f")
    with col2:
        cost_of_debt = st.number_input("Cost of debt", value=0.06, step=0.01, format="%.2f")
        tax_rate = st.number_input("Tax rate", value=0.25, step=0.01, format="%.2f")
    with col3:
        equity_pct = st.slider("Equity %", 0.0, 1.0, 0.60, 0.05)
        debt_pct = 1.0 - equity_pct
        st.metric("Debt %", f"{debt_pct:.0%}")
    submitted = st.form_submit_button("Calculate WACC")

if submitted:
    result = wacc(
        risk_free_rate=risk_free,
        beta=beta,
        equity_risk_premium=erp,
        cost_of_debt=cost_of_debt,
        tax_rate=tax_rate,
        debt_pct=debt_pct,
        equity_pct=equity_pct,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Cost of equity", f"{result['cost_of_equity']:.1%}")
    c2.metric("After-tax cost of debt", f"{result['after_tax_cost_of_debt']:.1%}")
    c3.metric("WACC", f"{result['wacc']:.1%}")

with st.expander("CAPM & WACC formulas"):
    st.markdown(
        """
        - **Cost of equity** = Risk-free rate + Beta × Equity risk premium
        - **After-tax cost of debt** = Cost of debt × (1 - Tax rate)
        - **WACC** = (E / V) × Cost of equity + (D / V) × After-tax cost of debt
        - **E / V** = Equity percentage; **D / V** = Debt percentage
        """
    )
