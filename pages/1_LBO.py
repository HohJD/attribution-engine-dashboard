"""LBO returns calculator."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from attribution.finance_tools import lbo_model

st.set_page_config(page_title="LBO Model", layout="wide")

st.title("LBO Model")
st.markdown(
    "A simple leveraged buyout model. Estimate returns based on entry/exit multiples, "
    "EBITDA growth, debt paydown, and the equity contribution."
)

with st.form("lbo_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        entry_ebitda = st.number_input("Entry EBITDA", value=100.0, min_value=0.0, step=10.0)
        entry_multiple = st.number_input("Entry EV/EBITDA", value=10.0, min_value=0.1, step=0.5)
        exit_multiple = st.number_input("Exit EV/EBITDA", value=12.0, min_value=0.1, step=0.5)
    with col2:
        equity_pct = st.slider("Equity %", 0.0, 1.0, 0.35, 0.05)
        debt_pct = 1.0 - equity_pct
        st.metric("Debt %", f"{debt_pct:.0%}")
        ebitda_cagr = st.number_input("EBITDA CAGR", value=0.08, step=0.01, format="%.2f")
        years = st.number_input("Holding years", value=5, min_value=1, max_value=30, step=1)
    with col3:
        cash_sweep_pct = st.slider("FCF sweep to debt", 0.0, 1.0, 0.5, 0.05)
        interest_rate = st.number_input("Interest rate", value=0.07, step=0.01, format="%.2f")
    submitted = st.form_submit_button("Calculate")

if submitted:
    result = lbo_model(
        entry_ebitda=entry_ebitda,
        entry_multiple=entry_multiple,
        equity_pct=equity_pct,
        debt_pct=debt_pct,
        ebitda_cagr=ebitda_cagr,
        exit_multiple=exit_multiple,
        years=int(years),
        cash_sweep_pct=cash_sweep_pct,
        interest_rate=interest_rate,
    )

    if not result:
        st.error("Invalid inputs.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MOIC", f"{result['moic']:.2f}x")
        c2.metric("IRR", f"{result['irr']:.1%}")
        c3.metric("Exit Equity", f"{result['exit_equity']:,.0f}")
        c4.metric("Remaining Debt", f"{result['remaining_debt']:,.0f}")

        st.subheader("Transaction summary")
        summary = pd.DataFrame(
            {
                "Item": [
                    "Entry EV",
                    "Equity",
                    "Debt",
                    "Exit EBITDA",
                    "Exit EV",
                    "Exit Equity",
                ],
                "Value": [
                    result["entry_ev"],
                    result["equity"],
                    result["debt"],
                    result["exit_ebitda"],
                    result["exit_ev"],
                    result["exit_equity"],
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.subheader("Value creation bridge")
        bridge = pd.DataFrame(
            {
                "Driver": [
                    "EBITDA growth",
                    "Multiple expansion",
                    "Total EV change",
                ],
                "Value": [
                    result["ebitda_growth_value"],
                    result["multiple_expansion_value"],
                    result["ebitda_growth_value"] + result["multiple_expansion_value"],
                ],
            }
        )
        st.bar_chart(bridge.set_index("Driver"), use_container_width=True)

with st.expander("What is MOIC / IRR?"):
    st.markdown(
        """
        - **MOIC** = Exit equity / Invested equity. A 2.5x means you get 2.5 dollars back for each dollar invested.
        - **IRR** = The annualized rate of return that makes the net present value of all cash flows equal to zero.
        - **Multiple expansion** = Selling the same EBITDA at a higher EV/EBITDA multiple.
        - **EBITDA growth** = Growing the company's operating profit over the holding period.
        """
    )
