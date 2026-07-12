"""Method selection logic for attribution-engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class MethodSelection:
    """Method selection result."""

    method: str
    reason: str
    n_donors: int
    n_pre_months: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "reason": self.reason,
            "n_donors": self.n_donors,
            "n_pre_months": self.n_pre_months,
        }


def select_method(
    panel: pd.DataFrame,
    treated_id: str,
    treatment_month: int,
    min_donors: int = 8,
    min_pre_months: int = 24,
) -> MethodSelection:
    """Choose between synthetic control and interrupted time series.

    Selection rule:
    - Use synthetic control if the donor pool has at least ``min_donors``
      usable donors and the pre-treatment period has at least ``min_pre_months``
      months.
    - Otherwise use ITS with covariate adjustment.

    Parameters
    ----------
    panel
        Long-format panel.
    treated_id
        Identifier of the treated company.
    treatment_month
        1-based month index where the post-period begins.
    min_donors
        Minimum number of usable donors for synthetic control.
    min_pre_months
        Minimum number of pre-treatment months for synthetic control.

    Returns
    -------
    MethodSelection
    """
    pre_months = panel[panel["month"] < treatment_month]
    donor_counts = (
        pre_months[pre_months["company_id"] != treated_id]
        .groupby("company_id")["month"]
        .count()
    )
    usable_donors = int((donor_counts >= min_pre_months).sum())
    n_pre_months = int((pre_months["company_id"] == treated_id).sum())

    if usable_donors >= min_donors and n_pre_months >= min_pre_months:
        return MethodSelection(
            method="synthetic_control",
            reason=(
                f"{usable_donors} usable donors >= {min_donors} and "
                f"{n_pre_months} pre-treatment months >= {min_pre_months}"
            ),
            n_donors=usable_donors,
            n_pre_months=n_pre_months,
        )
    return MethodSelection(
        method="its",
        reason=(
            f"Donor pool too thin ({usable_donors} usable donors < {min_donors} "
            f"or {n_pre_months} pre-months < {min_pre_months}); defaulting to ITS"
        ),
        n_donors=usable_donors,
        n_pre_months=n_pre_months,
    )
