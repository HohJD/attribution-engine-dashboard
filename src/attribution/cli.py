"""Command-line interface for attribution-engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console

from attribution.decomposition import DecompositionResult, decompose
from attribution.estimators.its import ITSResult, fit_its
from attribution.estimators.selector import MethodSelection, select_method
from attribution.estimators.synth import SyntheticControlResult, fit_synthetic_control
from attribution.generator import EffectShape, GeneratorConfig, generate_panel
from attribution.refutation import RefutationResult, run_refutation
from attribution.report import print_console_report, save_diagnostic_plots, save_json

app = typer.Typer(help="attribution-engine: PE value-creation attribution toolkit")
console = Console()


def _build_market_sector_indices(panel: pd.DataFrame, treated_id: str) -> tuple[pd.Series, pd.Series]:
    """Build market and sector index series from the donor pool."""
    donors = panel[panel["company_id"] != treated_id]
    market_index = donors.groupby("month")["ebitda"].mean()
    market_index = market_index / market_index.iloc[0]

    treated_sector = panel[panel["company_id"] == treated_id]["sector"].iloc[0]
    sector_donors = donors[donors["sector"] == treated_sector]
    sector_index = sector_donors.groupby("month")["ebitda"].mean()
    sector_index = sector_index / sector_index.iloc[0]
    return market_index, sector_index


def _run_attribution(
    panel: pd.DataFrame,
    treated_id: str,
    treatment_month: int,
    output_dir: Path = Path("./output"),
    rmspe_threshold: float = 0.20,
) -> dict[str, Any]:
    """Run the full attribution pipeline and return results."""
    method = select_method(panel, treated_id, treatment_month)

    market_index, sector_index = _build_market_sector_indices(panel, treated_id)

    if method.method == "synthetic_control":
        estimator_result = fit_synthetic_control(panel, treated_id, treatment_month)
        effect = estimator_result.effect
        counterfactual = estimator_result.counterfactual
        treated = estimator_result.treated
        estimator_name = estimator_result.method
        estimator_dict = estimator_result.to_dict()
    else:
        estimator_result = fit_its(
            panel,
            treated_id,
            treatment_month,
            use_covariates=True,
            market_series=market_index,
            sector_series=sector_index,
        )
        effect = estimator_result.effect
        counterfactual = estimator_result.counterfactual
        treated = estimator_result.treated
        estimator_name = estimator_result.method
        estimator_dict = estimator_result.to_dict()

    post = effect.index >= treatment_month
    total_effect = float(effect[post].sum())

    decomposition = decompose(panel, treated_id, treatment_month, effect)

    refutation = run_refutation(
        panel,
        treated_id,
        treatment_month,
        method,
        rmspe_threshold=rmspe_threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_diagnostic_plots(
        output_dir,
        panel,
        treated_id,
        treatment_month,
        effect,
        treated,
        counterfactual,
        decomposition,
        refutation,
    )

    json_path = output_dir / "result.json"
    save_json(
        json_path,
        method,
        effect,
        total_effect,
        decomposition,
        refutation,
        estimator_name,
        estimator_dict,
    )

    print_console_report(
        console,
        method,
        effect,
        total_effect,
        decomposition,
        refutation,
        estimator_name,
    )

    return {
        "method": method,
        "effect": effect,
        "total_effect": total_effect,
        "decomposition": decomposition,
        "refutation": refutation,
        "estimator_name": estimator_name,
        "estimator_result": estimator_result,
        "json_path": json_path,
    }


@app.command()
def generate(
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    out: Path = typer.Option("panel.csv", help="Output CSV path"),
    n_companies: int = typer.Option(40, help="Number of companies"),
    n_months: int = typer.Option(84, help="Number of months"),
    n_sectors: int = typer.Option(4, help="Number of sectors"),
    treatment_month: int = typer.Option(61, help="1-based treatment month"),
    effect_shape: EffectShape = typer.Option(EffectShape.STEP, help="Effect shape"),
    effect_size: float = typer.Option(0.10, help="Effect size"),
    noise_scale: float = typer.Option(1.0, help="Noise scale"),
    missing_pct: float = typer.Option(0.0, help="Missing data percentage"),
    confounded_donor: bool = typer.Option(False, help="Inject a confounded donor"),
) -> None:
    """Generate a synthetic panel and save it to CSV."""
    config = GeneratorConfig(
        n_companies=n_companies,
        n_months=n_months,
        n_sectors=n_sectors,
        treatment_month=treatment_month,
        effect_shape=EffectShape(effect_shape),
        effect_size=effect_size,
        noise_scale=noise_scale,
        missing_pct=missing_pct,
        confounded_donor=confounded_donor,
    )
    panel, truth = generate_panel(seed=seed, config=config)
    panel.to_csv(out, index=False)
    truth_path = out.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(truth, indent=2, default=str))
    console.print(f"Generated panel with {len(panel)} rows → {out}")
    console.print(f"Ground truth written to {truth_path}")


@app.command()
def run(
    panel: Path = typer.Argument(..., help="Path to panel CSV"),
    treated: str = typer.Option(..., help="Treated unit ID"),
    treatment_month: int = typer.Option(..., help="1-based treatment month"),
    output_dir: Path = typer.Option("./output", help="Directory for outputs"),
    rmspe_threshold: float = typer.Option(0.20, help="Pre-treatment RMSPE threshold"),
) -> None:
    """Run the attribution pipeline on a panel."""
    df = pd.read_csv(panel)
    _run_attribution(df, treated, treatment_month, output_dir, rmspe_threshold)


if __name__ == "__main__":
    app()
