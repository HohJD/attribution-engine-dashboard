"""Comparable companies multiples analysis."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from attribution.finance_tools import compute_multiples

st.set_page_config(page_title="Comps", layout="wide")

st.title("Comparable Companies")
st.markdown(
    "Enter a peer set with EV, revenue, EBITDA, and net income. The app computes median multiples "
    "and lets you value a target company."
)

st.subheader("Peer set")
with st.expander("Edit peers"):
    default_data = pd.DataFrame(
        [
            {"company": "Peer A", "ev": 1500, "revenue": 300, "ebitda": 60, "net_income": 30},
            {"company": "Peer B", "ev": 2200, "revenue": 400, "ebitda": 90, "net_income": 45},
            {"company": "Peer C", "ev": 800, "revenue": 200, "ebitda": 40, "net_income": 15},
        ]
    )
    peers = st.data_editor(default_data, use_container_width=True, num_rows="dynamic")

if not peers.empty:
    comps = compute_multiples(peers)
    st.subheader("Multiples")
    st.dataframe(comps[["company", "ev_revenue", "ev_ebitda", "pe"]], use_container_width=True)

    median = comps[["ev_revenue", "ev_ebitda", "pe"]].median()
    mean = comps[["ev_revenue", "ev_ebitda", "pe"]].mean()
    stats = pd.DataFrame({"Median": median, "Mean": mean})
    st.dataframe(stats, use_container_width=True)

    st.subheader("Value target company")
    col1, col2, col3 = st.columns(3)
    with col1:
        target_revenue = st.number_input("Target revenue", value=250.0, min_value=0.0)
    with col2:
        target_ebitda = st.number_input("Target EBITDA", value=50.0, min_value=0.0)
    with col3:
        target_net_income = st.number_input("Target net income", value=20.0, min_value=0.0)

    if target_revenue and target_ebitda and target_net_income:
        ev_from_rev = target_revenue * median["ev_revenue"]
        ev_from_ebitda = target_ebitda * median["ev_ebitda"]
        ev_from_pe = target_net_income * median["pe"]
        valuation = pd.DataFrame(
            {
                "Method": ["EV/Revenue", "EV/EBITDA", "P/E"],
                "Implied EV": [ev_from_rev, ev_from_ebitda, ev_from_pe],
            }
        )
        st.dataframe(valuation, use_container_width=True, hide_index=True)
        avg_ev = valuation["Implied EV"].mean()
        st.metric("Average implied EV", f"{avg_ev:,.0f}")

with st.expander("How to read multiples"):
    st.markdown(
        """
        - **EV/Revenue** is useful for high-growth or low-margin companies.
        - **EV/EBITDA** is the standard PE valuation multiple; it ignores leverage and capital structure.
        - **P/E** is equity value / net income; it is affected by debt and tax differences.
        """
    )
