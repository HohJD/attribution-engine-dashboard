"""DCF valuation model."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from attribution.finance_tools import dcf_valuation

st.set_page_config(page_title="DCF Valuation", layout="wide")

st.title("DCF Valuation")
st.markdown(
    "Unlevered free cash flow valuation. Input revenue growth, margins, and WACC to get enterprise value."
)

with st.form("dcf_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        revenue = st.number_input("Year 0 revenue", value=1000.0, min_value=0.0, step=100.0)
        years = st.number_input("Projection years", value=5, min_value=1, max_value=30, step=1)
        growth = st.number_input("Annual revenue growth", value=0.08, step=0.01, format="%.2f")
    with col2:
        ebitda_margin = st.number_input("EBITDA margin", value=0.20, step=0.01, format="%.2f")
        tax_rate = st.number_input("Tax rate", value=0.25, step=0.01, format="%.2f")
        d_and_a_pct = st.number_input("D&A / revenue", value=0.03, step=0.01, format="%.2f")
    with col3:
        capex_pct = st.number_input("Capex / revenue", value=0.04, step=0.01, format="%.2f")
        nwc_pct = st.number_input("ΔNWC / revenue growth", value=0.10, step=0.01, format="%.2f")
        wacc = st.number_input("WACC", value=0.10, step=0.01, format="%.2f")
        terminal_growth = st.number_input("Terminal growth", value=0.02, step=0.01, format="%.2f")
    submitted = st.form_submit_button("Value")

if submitted:
    growth_rates = [growth] * int(years)
    result = dcf_valuation(
        revenue=revenue,
        growth_rates=growth_rates,
        ebitda_margin=ebitda_margin,
        tax_rate=tax_rate,
        d_and_a_pct=d_and_a_pct,
        capex_pct=capex_pct,
        nwc_pct=nwc_pct,
        wacc=wacc,
        terminal_growth=terminal_growth,
    )

    c1, c2 = st.columns(2)
    c1.metric("Enterprise value", f"{result['enterprise_value']:,.0f}")
    c2.metric("PV of terminal value", f"{result['pv_terminal']:,.0f}")

    st.subheader("Projected financials")
    df = pd.DataFrame(
        {
            "Revenue": result["revenues"],
            "EBITDA": result["ebitda"],
            "FCF": result["free_cash_flow"],
            "PV of FCF": result["pv_fcf"],
        }
    )
    st.dataframe(df, use_container_width=True)
    st.line_chart(df[["Revenue", "FCF"]], use_container_width=True)

with st.expander("DCF formula"):
    st.markdown(
        """
        - **FCF** = EBITDA × (1 - tax rate) + D&A - Capex - ΔNWC
        - **PV of FCF** = FCF / (1 + WACC) ^ year
        - **Terminal value** = FCF_n × (1 + g) / (WACC - g)
        - **Enterprise value** = Σ PV of FCF + PV of terminal value
        """
    )
