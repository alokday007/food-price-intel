"""/api/forecast/<code>/ (Phase 8b) on small hand-built fixtures, not the real dataset.

Fixture shape, chosen to catch the subtle bugs:
- OVERALL has two SARIMA runs; the API must serve only the newer one.
- SUGAR has observations AFTER its run's train_end, so the naive_1 baseline must
  anchor to train_end, not to the latest observed month.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

import pytest

from apps.catalog.models import CommodityGroup, DataSource
from apps.forecasting import views
from apps.forecasting.models import ForecastPoint, ForecastRun
from apps.prices.models import PriceIndexMonthly

EXPECTED_KEYS = {"commodity_group", "label", "history", "forecast", "naive_forecast", "scoreboard"}


def _observe(group, source, values: dict[str, str]) -> None:
    for period, value in values.items():
        y, m = (int(x) for x in period.split("-"))
        PriceIndexMonthly.objects.create(
            commodity_group=group, period=date(y, m, 1), value_nominal=Decimal(value), source=source
        )


def _run(group, train_end: date, preds: list[tuple[date, str]]) -> ForecastRun:
    run = ForecastRun.objects.create(
        model_name="sarima", commodity_group=group, train_end=train_end, horizon=len(preds)
    )
    for step, (period, value) in enumerate(preds, start=1):
        ForecastPoint.objects.create(run=run, period=period, horizon_step=step, value_pred=Decimal(value))
    return run


@pytest.fixture
def scoreboard_file(tmp_path, monkeypatch):
    path = tmp_path / "scoreboard.json"
    path.write_text(
        json.dumps(
            {
                "eval_end": "2026-03",
                "tie_threshold": 0.02,
                "baseline": "naive_1",
                "groups": {
                    "SUGAR": {
                        "1": {"sarima_mae": 5.2, "naive_1_mae": 4.9, "rel_delta": 0.061,
                              "verdict": "LOSES", "folds": 10},
                        "3": {"sarima_mae": 10.2, "naive_1_mae": 9.5, "rel_delta": 0.074,
                              "verdict": "LOSES", "folds": 8},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(views, "SCOREBOARD_PATH", path)
    return path


@pytest.fixture
def series(db):
    source = DataSource.objects.create(name="FAO FFPI")
    overall = CommodityGroup.objects.create(code="OVERALL", name="FAO Food Price Index")
    sugar = CommodityGroup.objects.create(code="SUGAR", name="Sugar Price Index")

    _observe(overall, source, {"2026-01": "120.00", "2026-02": "121.00", "2026-03": "122.00"})
    _run(overall, date(2026, 2, 1), [(date(2026, 3, 1), "99.00")])  # older run
    _run(overall, date(2026, 3, 1), [(date(2026, 4, 1), "123.00"), (date(2026, 5, 1), "124.00")])

    # SUGAR: observed through 2026-04, but its run was fitted through 2026-02.
    _observe(sugar, source, {"2026-01": "90.00", "2026-02": "95.50", "2026-03": "107.00", "2026-04": "110.00"})
    _run(sugar, date(2026, 2, 1), [(date(2026, 3, 1), "97.00"), (date(2026, 4, 1), "98.00"),
                                    (date(2026, 5, 1), "99.00")])


def test_overall_legacy_route_keeps_original_fields(client, series, scoreboard_file):
    resp = client.get("/api/forecast/overall/")
    assert resp.status_code == 200
    data = resp.json()
    assert EXPECTED_KEYS <= data.keys()
    assert data["commodity_group"] == "OVERALL"
    assert data["history"][-1] == {"period": "2026-03", "value": 122.0}
    # Original forecast shape, served from the LATEST run only.
    assert set(data["forecast"]) == {"model", "run_id", "train_end", "points"}
    assert data["forecast"]["train_end"] == "2026-03"
    assert [p["period"] for p in data["forecast"]["points"]] == ["2026-04", "2026-05"]
    assert data["forecast"]["points"][0] == {"period": "2026-04", "horizon_step": 1, "value": 123.0}
    # OVERALL is absent from this scoreboard file -> null, not a crash.
    assert data["scoreboard"] is None


def test_sugar_case_insensitive_with_scoreboard(client, series, scoreboard_file):
    resp = client.get("/api/forecast/SUGAR/")
    assert resp.status_code == 200
    data = resp.json()
    assert EXPECTED_KEYS <= data.keys()
    assert data["commodity_group"] == "SUGAR"
    assert data["label"] == "Sugar"
    assert data["scoreboard"]["eval_end"] == "2026-03"
    assert data["scoreboard"]["horizons"]["1"]["verdict"] == "LOSES"


def test_naive_forecast_anchored_to_train_end(client, series, scoreboard_file):
    data = client.get("/api/forecast/sugar/").json()
    forecast, naive = data["forecast"], data["naive_forecast"]

    train_end_value = next(p["value"] for p in data["history"] if p["period"] == forecast["train_end"])
    assert train_end_value == 95.5
    # Anchored to train_end (2026-02), NOT the latest observation (2026-04 = 110).
    assert naive["train_end"] == forecast["train_end"] == "2026-02"
    assert naive["anchor_value"] == train_end_value
    assert all(p["value"] == train_end_value for p in naive["points"])
    assert [p["period"] for p in naive["points"]] == [p["period"] for p in forecast["points"]]
    assert [p["horizon_step"] for p in naive["points"]] == [1, 2, 3]


def test_naive_forecast_is_not_persisted(client, series, scoreboard_file):
    runs, points = ForecastRun.objects.count(), ForecastPoint.objects.count()
    client.get("/api/forecast/sugar/")
    assert (ForecastRun.objects.count(), ForecastPoint.objects.count()) == (runs, points)


def test_unknown_series_is_json_404(client, series, scoreboard_file):
    resp = client.get("/api/forecast/bogus/")
    assert resp.status_code == 404
    assert resp["Content-Type"] == "application/json"
    assert "error" in resp.json()


def test_missing_scoreboard_file_is_null(client, series, tmp_path, monkeypatch):
    monkeypatch.setattr(views, "SCOREBOARD_PATH", tmp_path / "does-not-exist.json")
    resp = client.get("/api/forecast/sugar/")
    assert resp.status_code == 200
    assert resp.json()["scoreboard"] is None


def test_known_group_without_data_or_runs(client, db, scoreboard_file):
    CommodityGroup.objects.create(code="DAIRY", name="Dairy Price Index")
    data = client.get("/api/forecast/dairy/").json()
    assert data["history"] == []
    assert data["forecast"] is None
    assert data["naive_forecast"] is None
