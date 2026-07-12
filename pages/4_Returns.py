"""Private equity returns calculator."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from attribution.finance_tools import compute_dpi, compute_irr, compute_moic, compute_tvpi

st.set_page_config(page_title="PE Returns", layout="wide")

st.title("PE Returns Calculator")
st.markdown(
    "Compute MOIC, IRR, DPI, and TVPI from a cash-flow schedule. Negative numbers are contributions (outflows)."
)

st.subheader("Cash flows")
with st.expander("Edit cash flows"):
    default_flows = pd.DataFrame(
        [
            {"year": 0, "contribution": -100, "distribution": 0, "nav": 0},
            {"year": 1, "contribution": -50, "distribution": 0, "nav": 0},
            {"year": 2, "contribution": 0, "distribution": 30, "nav": 80},
            {"year": 3, "contribution": 0, "distribution": 200, "nav": 0},
        ]
    )
    flows = st.data_editor(default_flows, use_container_width=True, num_rows="dynamic")

if not flows.empty:
    flows = flows.sort_values("year")
    net_cash = (flows["distribution"] - flows["contribution"]).tolist()
    paid_in = flows["contribution"].sum()
    distributions = flows["distribution"].sum()
    final_nav = flows["nav"].iloc[-1]

    moic = compute_moic(distributions + final_nav, abs(paid_in))
    dpi = compute_dpi(distributions, abs(paid_in))
    tvpi = compute_tvpi(distributions, final_nav, abs(paid_in))
    irr = compute_irr(net_cash)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MOIC", f"{moic:.2f}x")
    c2.metric("IRR", f"{irr:.1%}" if irr is not None else "N/A")
    c3.metric("DPI", f"{dpi:.2f}x")
    c4.metric("TVPI", f"{tvpi:.2f}x")

    st.bar_chart(flows.set_index("year")[["contribution", "distribution", "nav"]], use_container_width=True)

with st.expander("Metric definitions"):
    st.markdown(
        """
        - **MOIC** = (Distributions + residual NAV) / Paid-in capital
        - **IRR** = Internal rate of return of the net cash-flow stream
        - **DPI** = Cumulative distributions / paid-in capital (realized return)
        - **TVPI** = (Distributions + NAV) / paid-in capital (realized + unrealized)
        """
    )
