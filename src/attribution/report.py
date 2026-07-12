"""Plain-text, JSON, and diagnostic plotting outputs for attribution-engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table

from attribution.decomposition import DecompositionResult
from attribution.estimators.its import ITSResult
from attribution.estimators.selector import MethodSelection
from attribution.estimators.synth import SyntheticControlResult
from attribution.refutation import RefutationResult


def save_json(
    output_path: Path,
    method: MethodSelection,
    effect: pd.Series,
    total_effect: float,
    decomposition: DecompositionResult,
    refutation: RefutationResult,
    estimator_name: str,
    estimator_dict: dict[str, Any],
) -> None:
    """Write the full result to a JSON file."""
    result = {
        "method": method.to_dict(),
        "estimator": estimator_name,
        "estimator_result": estimator_dict,
        "total_effect": float(total_effect),
        "effect_path": effect.to_dict(),
        "decomposition": decomposition.to_dict(),
        "refutation": refutation.to_dict(),
    }
    output_path.write_text(json.dumps(result, indent=2, default=str))


def print_console_report(
    console: Console,
    method: MethodSelection,
    effect: pd.Series,
    total_effect: float,
    decomposition: DecompositionResult,
    refutation: RefutationResult,
    estimator_name: str,
) -> None:
    """Print a human-readable report to the console."""
    console.rule("Attribution Engine — Phase 1 Report")
    console.print(f"[bold]Method:[/bold] {estimator_name} ({method.method})")
    console.print(f"[bold]Reason:[/bold] {method.reason}")
    console.print(f"[bold]Pre-fit reliability:[/bold] {refutation.reliability_flag}")

    console.rule("Effect estimate")
    console.print(f"Total post-treatment effect: {total_effect:,.2f}")
    console.print(f"Mean monthly post-treatment effect: {effect[effect.index >= max(effect.index) * 0.5].mean():,.2f}")

    console.rule("Decomposition")
    table = Table("Component", "Value")
    table.add_row("Market", f"{decomposition.market_component:,.2f}")
    table.add_row("Sector", f"{decomposition.sector_component:,.2f}")
    table.add_row("Alpha", f"{decomposition.alpha_component:,.2f}")
    table.add_row("Unexplained", f"{decomposition.unexplained:,.2f}")
    table.add_row("Total growth", f"{decomposition.total_growth:,.2f}")
    console.print(table)

    console.rule("Refutation tests")
    ref_table = Table("Test", "Status", "Value")
    ref_table.add_row("In-space placebo pseudo-p", refutation.pass_summary["in_space_placebo"], f"{refutation.placebo_pseudo_pvalue:.3f}")
    ref_table.add_row("In-time placebo", refutation.pass_summary["in_time_placebo"], f"{refutation.backdating_pseudo_pvalue}")
    ref_table.add_row("Leave-one-out", refutation.pass_summary["leave_one_out"], f"{len(refutation.leave_one_out)} runs")
    ref_table.add_row("Pre-fit", refutation.pass_summary["pre_fit"], f"RMSPE ratio {refutation.pre_rmspe_ratio:.3f}")
    console.print(ref_table)

    if refutation.reliability_flag == "unreliable":
        console.print(
            "[bold red]WARNING: pre-treatment fit is poor. The effect estimate is unreliable.[/bold red]"
        )


def _plot_treated_vs_synthetic(
    output_dir: Path,
    effect: pd.Series,
    treated: pd.Series,
    counterfactual: pd.Series,
    treatment_month: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(treated.index, treated, label="Treated", linewidth=2)
    ax.plot(counterfactual.index, counterfactual, label="Synthetic counterfactual", linestyle="--")
    ax.axvline(treatment_month, color="red", linestyle=":", label="Treatment month")
    ax.set_title("Treated vs Synthetic Counterfactual")
    ax.set_xlabel("Month")
    ax.set_ylabel("EBITDA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "treated_vs_synthetic.png", dpi=150)
    plt.close(fig)


def _plot_placebo_spaghetti(
    output_dir: Path,
    panel: pd.DataFrame,
    treated_id: str,
    treatment_month: int,
    placebo_ratios: pd.Series,
    treated_ratio: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for company_id in panel["company_id"].unique():
        if company_id == treated_id:
            continue
        sub = panel[panel["company_id"] == company_id].set_index("month").sort_index()
        ax.plot(sub.index, sub["ebitda"], color="gray", alpha=0.3, linewidth=0.5)
    treated = panel[panel["company_id"] == treated_id].set_index("month").sort_index()
    ax.plot(treated.index, treated["ebitda"], color="red", linewidth=2, label="Treated")
    ax.axvline(treatment_month, color="black", linestyle=":", label="Treatment month")
    ax.set_title("Placebo Spaghetti Plot")
    ax.set_xlabel("Month")
    ax.set_ylabel("EBITDA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "placebo_spaghetti.png", dpi=150)
    plt.close(fig)


def _plot_decomposition(
    output_dir: Path,
    decomposition: DecompositionResult,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    components = ["market", "sector", "alpha", "unexplained"]
    values = [
        decomposition.market_component,
        decomposition.sector_component,
        decomposition.alpha_component,
        decomposition.unexplained,
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax.bar(components, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Post-Treatment EBITDA Growth Decomposition")
    ax.set_ylabel("EBITDA")
    fig.tight_layout()
    fig.savefig(output_dir / "decomposition.png", dpi=150)
    plt.close(fig)


def save_diagnostic_plots(
    output_dir: Path,
    panel: pd.DataFrame,
    treated_id: str,
    treatment_month: int,
    effect: pd.Series,
    treated: pd.Series,
    counterfactual: pd.Series,
    decomposition: DecompositionResult,
    refutation: RefutationResult,
) -> None:
    """Save all diagnostic plots to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_treated_vs_synthetic(output_dir, effect, treated, counterfactual, treatment_month)
    _plot_placebo_spaghetti(output_dir, panel, treated_id, treatment_month, refutation.placebo_ratios, 0.0)
    _plot_decomposition(output_dir, decomposition)
