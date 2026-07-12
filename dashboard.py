"""Interactive Streamlit dashboard for attribution-engine."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULT_JSON = OUTPUT_DIR / "result.json"
PANEL_CSV = PROJECT_ROOT / "panel.csv"


def _run_cmd(cmd: list[str]) -> str:
    """Run a CLI command in the project root and return stdout, or raise a clear error."""
    try:
        completed = subprocess.run(  # noqa: S603
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        msg = f"Command failed:\n{' '.join(cmd)}\n\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        raise RuntimeError(msg) from exc
    return completed.stdout


def run_pipeline(panel_path: Path, treated_id: str, treatment_month: int) -> None:
    """Run the attribution CLI for the selected panel and treated unit."""
    cmd = [
        sys.executable,
        "-m",
        "attribution.cli",
        "run",
        str(panel_path),
        "--treated",
        treated_id,
        "--treatment-month",
        str(treatment_month),
    ]
    _run_cmd(cmd)


def generate_sample_panel() -> None:
    """Generate the default sample panel."""
    cmd = [
        sys.executable,
        "-m",
        "attribution.cli",
        "generate",
        "--seed",
        "42",
        "--out",
        str(PANEL_CSV),
    ]
    _run_cmd(cmd)


def ensure_result() -> None:
    """Generate sample data and run pipeline if no result exists."""
    if not PANEL_CSV.exists():
        generate_sample_panel()
    if not RESULT_JSON.exists():
        run_pipeline(PANEL_CSV, "C000", 61)


def load_result() -> dict:
    """Load the JSON result from the output directory."""
    return json.loads(RESULT_JSON.read_text())


def plot_treated_vs_synthetic(result: dict) -> None:
    """Render the treated vs synthetic counterfactual chart."""
    estimator = result["estimator_result"]
    treated = pd.Series(estimator["treated"], name="treated")
    counterfactual = pd.Series(estimator["counterfactual"], name="counterfactual")
    chart_data = pd.DataFrame({"Treated": treated, "Counterfactual": counterfactual})
    st.line_chart(chart_data, use_container_width=True)


def plot_effect(result: dict) -> None:
    """Render the monthly treatment effect."""
    effect = pd.Series(result["estimator_result"]["effect"], name="effect")
    st.bar_chart(effect, use_container_width=True)


def plot_decomposition(result: dict) -> None:
    """Render the decomposition bar chart."""
    decomp = result["decomposition"]
    chart_data = pd.DataFrame(
        {
            "component": ["Market", "Sector", "Alpha", "Unexplained"],
            "value": [
                decomp["market_component"],
                decomp["sector_component"],
                decomp["alpha_component"],
                decomp["unexplained"],
            ],
        }
    )
    chart_data = chart_data.set_index("component")
    st.bar_chart(chart_data, use_container_width=True)


def show_refutation(result: dict) -> None:
    """Show refutation test results and leave-one-out table."""
    ref = result["refutation"]
    pass_summary = ref["pass_summary"]
    cols = st.columns(4)
    for i, (test, status) in enumerate(pass_summary.items()):
        color = {"pass": "green", "warn": "orange", "fail": "red"}.get(status, "gray")
        cols[i].metric(test.replace("_", " ").title(), status, delta_color="off")
        # small color hint via markdown since metric delta_color is limited
        cols[i].markdown(f"<span style='color:{color}'>●</span>", unsafe_allow_html=True)

    st.write("**Leave-one-out robustness**")
    loo = ref["leave_one_out"]
    loo_df = pd.DataFrame.from_dict(loo, orient="index").reset_index()
    if not loo_df.empty:
        loo_df = loo_df.rename(columns={"index": "dropped"})
        st.dataframe(loo_df, use_container_width=True)


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="Attribution Engine Dashboard", layout="wide")
    st.title("Attribution Engine Dashboard")
    st.markdown("Synthetic control / interrupted time-series attribution for PE value creation.")

    try:
        ensure_result()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    result = load_result()

    st.sidebar.header("Run")
    if st.sidebar.button("Regenerate sample panel"):
        try:
            generate_sample_panel()
            run_pipeline(PANEL_CSV, "C000", 61)
            st.rerun()
        except RuntimeError as exc:
            st.sidebar.error(str(exc))

    uploaded = st.sidebar.file_uploader("Upload panel CSV", type="csv")
    if uploaded is not None:
        uploaded_path = PROJECT_ROOT / "uploaded_panel.csv"
        uploaded_path.write_bytes(uploaded.getvalue())
        df = pd.read_csv(uploaded_path)
        treated = st.sidebar.selectbox("Treated company", df["company_id"].unique())
        treatment_month = st.sidebar.number_input(
            "Treatment month", min_value=1, max_value=int(df["month"].max()), value=61
        )
        if st.sidebar.button("Run attribution"):
            try:
                run_pipeline(uploaded_path, treated, int(treatment_month))
                st.rerun()
            except RuntimeError as exc:
                st.sidebar.error(str(exc))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Method")
    st.sidebar.write(f"**{result['method']['method']}**")
    st.sidebar.write(result["method"]["reason"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total effect", f"{result['total_effect']:,.2f}")
    col2.metric("Estimator", result["estimator"])
    col3.metric("Reliability", result["refutation"]["reliability_flag"])
    col4.metric("Placebo p-value", f"{result['refutation']['placebo_pseudo_pvalue']:.3f}")

    tab_overview, tab_effect, tab_decomp, tab_refutation = st.tabs(
        ["Overview", "Effect", "Decomposition", "Refutation"]
    )

    with tab_overview:
        st.subheader("Treated vs Synthetic Counterfactual")
        plot_treated_vs_synthetic(result)

    with tab_effect:
        st.subheader("Monthly Treatment Effect")
        plot_effect(result)

    with tab_decomp:
        st.subheader("EBITDA Growth Decomposition")
        plot_decomposition(result)
        st.write(result["decomposition"])

    with tab_refutation:
        st.subheader("Refutation Tests")
        show_refutation(result)


if __name__ == "__main__":
    main()
