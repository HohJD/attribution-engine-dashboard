"""PE finance formula reference."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Learning Hub", layout="wide")

st.title("PE Finance Learning Hub")
st.markdown("Quick reference for the core concepts built into this dashboard.")

sections = {
    "Attribution": {
        "text": (
            "The attribution engine splits post-acquisition EBITDA growth into market, sector, and "
            "company-specific alpha using synthetic control or interrupted time-series methods."
        ),
        "formulas": [
            "Total growth = Market + Sector + Alpha + Unexplained",
            "Effect = Treated EBITDA - Synthetic counterfactual",
        ],
    },
    "LBO": {
        "text": (
            "A leveraged buyout uses debt to finance the acquisition. Returns come from EBITDA growth, "
            "multiple expansion, and debt paydown."
        ),
        "formulas": [
            "Entry EV = EBITDA × Entry multiple",
            "Exit Equity = Exit EV - Remaining debt",
            "MOIC = Exit Equity / Invested Equity",
        ],
    },
    "DCF": {
        "text": (
            "Discounted cash flow values a company by projecting future free cash flow and discounting "
            "it back to today."
        ),
        "formulas": [
            "FCF = EBITDA × (1 - Tax) + D&A - Capex - ΔNWC",
            "PV = FCF / (1 + WACC) ^ year",
            "Terminal value = FCF_n × (1 + g) / (WACC - g)",
        ],
    },
    "Multiples": {
        "text": (
            "Comparable company analysis values a target by applying median peer multiples to its "
            "financial metrics."
        ),
        "formulas": [
            "EV = EBITDA × EV/EBITDA multiple",
            "EV = Revenue × EV/Revenue multiple",
            "Equity value = Net income × P/E multiple",
        ],
    },
    "Returns": {
        "text": (
            "Private equity funds report returns using MOIC and IRR. DPI and TVPI split realized and "
            "unrealized value."
        ),
        "formulas": [
            "MOIC = (Distributions + NAV) / Paid-in",
            "DPI = Distributions / Paid-in",
            "TVPI = (Distributions + NAV) / Paid-in",
        ],
    },
    "WACC": {
        "text": (
            "WACC is the blended cost of debt and equity used to discount future cash flows."
        ),
        "formulas": [
            "Cost of equity = Rf + Beta × ERP",
            "After-tax cost of debt = Cost of debt × (1 - Tax rate)",
            "WACC = (E/V) × Re + (D/V) × Rd × (1 - T)",
        ],
    },
}

for title, content in sections.items():
    with st.expander(title):
        st.markdown(content["text"])
        st.markdown("**Key formulas**")
        for formula in content["formulas"]:
            st.markdown(f"- `{formula}`")
