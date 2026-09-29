"""Loading an observed monthly series for one commodity group (Phase 8a).

Phases 4-5 hard-coded OVERALL in both ``evaluate_sarima`` and
``generate_forecast``, each with its own private copy of the loader and the
contiguity check. Phase 8a forecasts all six FFPI series, so those two helpers
move here and are shared: the *only* thing that varies per series is which
``commodity_group`` is loaded. The walk-forward harness (``baseline.py``) and the
model (``sarima.py``) are reused untouched, which is what keeps the six results
comparable to each other and to the Phase 3/4 numbers.

The contiguity assertion is deliberately per-series and fatal. FAO publishes all
six indices on the same monthly cadence, so a gap would mean a broken ingest, not
an irregular series — and a gap would silently corrupt both the walk-forward fold
structure and SARIMA's seasonal lags. Better to fail loudly on that one series.
"""

from __future__ import annotations

import pandas as pd
from django.core.management.base import CommandError

from apps.catalog.models import CommodityGroup
from apps.prices.models import PriceIndexMonthly

# The OVERALL FFPI plus its five sub-indices (SPEC §5).
GROUP_CODES = ("OVERALL", "CEREALS", "OILS", "DAIRY", "MEAT", "SUGAR")

DEFAULT_GROUP = "OVERALL"


def get_group(code: str) -> CommodityGroup:
    """Resolve a commodity group code, or fail with a usable message."""
    try:
        return CommodityGroup.objects.get(code=code)
    except CommodityGroup.DoesNotExist:
        known = ", ".join(
            CommodityGroup.objects.order_by("code").values_list("code", flat=True)
        )
        raise CommandError(
            f"CommodityGroup {code!r} not found (known: {known or 'none'}) — "
            "run seed_catalog first."
        )


def load_group_series(group: CommodityGroup) -> pd.Series:
    """Observed PriceIndexMonthly for one group as a period-indexed float series."""
    rows = list(
        PriceIndexMonthly.objects.filter(commodity_group=group)
        .order_by("period")
        .values_list("period", "value_nominal")
    )
    if not rows:
        raise CommandError(
            f"No {group.code} rows in PriceIndexMonthly — run ingest_ffpi first."
        )
    index = pd.DatetimeIndex([pd.Timestamp(p) for p, _ in rows])
    values = [float(v) for _, v in rows]
    return pd.Series(values, index=index, name=group.code)


def assert_contiguous_monthly(series: pd.Series) -> None:
    """Fail loudly if this series has any missing month between first and last.

    Raised per series, naming the series and the missing months, so a broken
    sub-index is diagnosable without re-running the other five.
    """
    expected = pd.date_range(series.index.min(), series.index.max(), freq="MS")
    if len(series) != len(expected) or not series.index.equals(expected):
        missing = expected.difference(series.index)
        raise CommandError(
            f"{series.name} series is not contiguous monthly "
            f"({len(series)} rows, expected {len(expected)}); missing "
            f"{[f'{m:%Y-%m}' for m in missing]}"
        )
