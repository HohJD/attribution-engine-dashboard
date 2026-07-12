# attribution-engine

A Python library and CLI for private-equity portfolio value-creation attribution.

Given the monthly financial history of a PE-backed company (the "treated" portco),
a pool of comparable public or private companies (the "donors"), and an
intervention date (e.g. acquisition or a specific operational initiative), the
tool decomposes post-intervention EBITDA growth into:

1. **Market beta** — what would have happened because of broad market moves.
2. **Sector effect** — the component explained by the treated company's sector.
3. **Company-specific alpha** — the estimated treatment effect attributable to the
   intervention.

The core methods are **synthetic control** (primary) and **interrupted time
series** (fallback when the donor pool is thin).

> **Phase 1 scope:** synthetic data generator + estimation core + refutation
tests. No front end, no real data ingestion, no deployment.

## Quickstart

Install the package and development dependencies.

```bash
# With uv
uv venv
uv pip install -e ".[dev]"

# Or with pip in a virtual environment
pip install -e ".[dev]"
```

Generate a synthetic panel and run the full pipeline.

```bash
attribution generate --seed 42 --out panel.csv
attribution run panel.csv --treated C000 --treatment-month 61
```

Outputs are written to `./output/`:

- `result.json` — full JSON result.
- `treated_vs_synthetic.png` — treated vs synthetic counterfactual.
- `placebo_spaghetti.png` — donor spaghetti plot.
- `decomposition.png` — market / sector / alpha / unexplained bar chart.

## Worked example

```python
from attribution.generator import GeneratorConfig, generate_panel
from attribution.estimators.selector import select_method
from attribution.estimators.synth import fit_synthetic_control
from attribution.decomposition import decompose
from attribution.refutation import run_refutation

# Generate a synthetic panel with a known +10% step effect at month 61
panel, truth = generate_panel(
    seed=42,
    config=GeneratorConfig(
        n_companies=30,
        n_months=72,
        treatment_month=49,
        effect_shape="step",
        effect_size=0.10,
    ),
)

method = select_method(panel, truth["treated_id"], truth["treatment_month"])
print(method.method, method.reason)

# Run the chosen estimator
if method.method == "synthetic_control":
    result = fit_synthetic_control(panel, truth["treated_id"], truth["treatment_month"])
else:
    from attribution.estimators.its import fit_its
    result = fit_its(panel, truth["treated_id"], truth["treatment_month"])

effect = result.effect
post = effect.index >= truth["treatment_month"]
print("Total effect:", effect[post].sum())

# Decompose growth
decomp = decompose(panel, truth["treated_id"], truth["treatment_month"], effect)
print(decomp.to_table())

# Refutation tests
refutation = run_refutation(panel, truth["treated_id"], truth["treatment_month"], method)
print("Pseudo p-value:", refutation.placebo_pseudo_pvalue)
```

## Methodology

### Synthetic control

Synthetic control constructs a weighted combination of donor companies that best
matches the treated company *before* the intervention. The weights are
non-negative and sum to one, so the counterfactual is an interpretable
"synthetic twin" of the portco. The treatment effect in each post-intervention
month is the gap between the treated company and its synthetic twin.

This method is preferred when the donor pool is deep (at least 8 usable donors)
and the pre-treatment period is long (at least 24 months). With enough donors,
we can match the portco's pre-trend, margin, and revenue growth, which makes the
counterfactual more credible.

### Interrupted time series (ITS)

When the donor pool is too thin for synthetic control, we fall back to an
interrupted time series model. ITS models the treated company's own history as a
trend with a potential level shift and trend change at the intervention date. We
use Newey-West standard errors to account for autocorrelation. Optionally, a
market or sector index can be added as a covariate.

### Method choice

The selector is rule-based:

- **Synthetic control** when donors >= 8 and pre-intervention months >= 24.
- **ITS** otherwise.

The chosen method and the reason are always reported in the output.

## Development

Run the tests with:

```bash
pytest
```

Lint with:

```bash
ruff check src tests
```

## Project status

Phase 1 is focused on synthetic data and the estimation core. Real market data
ingestion, a front end, staggered adoption, and deployment are out of scope.
