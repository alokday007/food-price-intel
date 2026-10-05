"""seed_from_csv (Phase 8b): all-groups CSV, idempotent, loud on unknown groups."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.catalog.models import CommodityGroup, DataSource
from apps.prices.models import PriceIndexMonthly

HEADER = "commodity_group,period,value_nominal,value_real\n"


@pytest.fixture
def catalog(db):
    DataSource.objects.create(name="FAO FFPI")
    for code in ("OVERALL", "SUGAR"):
        CommodityGroup.objects.create(code=code, name=code)


def _csv(tmp_path, body: str):
    path = tmp_path / "seed.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def _seed(path) -> str:
    out = StringIO()
    call_command("seed_from_csv", "--file", str(path), stdout=out)
    return out.getvalue()


def test_second_run_creates_zero_rows(catalog, tmp_path):
    path = _csv(
        tmp_path,
        "OVERALL,2026-01-01,120.00,110.00\n"
        "OVERALL,2026-02-01,121.00,\n"
        "SUGAR,2026-01-01,90.00,80.00\n",
    )
    first = _seed(path)
    assert "created : 3" in first
    assert PriceIndexMonthly.objects.count() == 3

    second = _seed(path)
    assert "created : 0" in second
    assert "updated : 3" in second
    assert PriceIndexMonthly.objects.count() == 3

    row = PriceIndexMonthly.objects.get(commodity_group__code="OVERALL", period=date(2026, 1, 1))
    assert (row.value_nominal, row.value_real) == (Decimal("120.00"), Decimal("110.00"))
    blank_real = PriceIndexMonthly.objects.get(commodity_group__code="OVERALL", period=date(2026, 2, 1))
    assert blank_real.value_real is None


def test_revision_updates_in_place(catalog, tmp_path):
    _seed(_csv(tmp_path, "SUGAR,2026-01-01,90.00,80.00\n"))
    _seed(_csv(tmp_path, "SUGAR,2026-01-01,91.50,81.00\n"))
    row = PriceIndexMonthly.objects.get()
    assert row.value_nominal == Decimal("91.50")


def test_unknown_group_fails_loudly_and_loads_nothing(catalog, tmp_path):
    path = _csv(
        tmp_path,
        "OVERALL,2026-01-01,120.00,110.00\n"
        "COFFEE,2026-01-01,150.00,140.00\n",
    )
    with pytest.raises(CommandError, match="unknown commodity_group 'COFFEE'"):
        _seed(path)
    assert PriceIndexMonthly.objects.count() == 0  # whole run rolled back


def test_old_single_series_header_is_rejected(catalog, tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("period,value_nominal\n2026-01-01,120.00\n", encoding="utf-8")
    with pytest.raises(CommandError, match="Unexpected CSV header"):
        _seed(path)
