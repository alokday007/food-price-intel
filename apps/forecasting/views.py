"""Read-only views: any FFPI series' history + latest forecast as JSON, and a chart page.

Phase 6 served OVERALL only; Phase 8b serves all six series through one view,
``/api/forecast/<code>/``. ``/api/forecast/overall/`` is just that route with
code=overall, and its original fields (commodity_group, history, forecast) are
unchanged — Phase 8b only *adds* fields.

This layer only *reads*. History comes exclusively from PriceIndexMonthly
(observed FAO data); the SARIMA forecast comes exclusively from ForecastPoint
(model output). The two are never crossed — an observation is not a prediction.
The naive_1 baseline is derived on the fly from observations and never persisted.
"""

from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

from apps.forecasting.models import ForecastRun
from apps.forecasting.series import GROUP_CODES
from apps.prices.models import PriceIndexMonthly

MODEL_NAME = "sarima"

# Display labels for the chart selector, in selector order (codes = GROUP_CODES).
SERIES_LABELS = {
    "OVERALL": "Food Price Index",
    "CEREALS": "Cereals",
    "OILS": "Vegetable Oils",
    "DAIRY": "Dairy",
    "MEAT": "Meat",
    "SUGAR": "Sugar",
}

# Walk-forward results are too slow to compute on the production dyno, so they
# ship as a committed artifact written by ``evaluate_sarima --summarize --out``.
SCOREBOARD_PATH = Path(settings.BASE_DIR) / "data" / "seed" / "scoreboard.json"


def _history(code: str) -> list[dict]:
    return [
        {"period": period.strftime("%Y-%m"), "value": float(value)}
        for period, value in (
            PriceIndexMonthly.objects.filter(commodity_group__code=code)
            .order_by("period")
            .values_list("period", "value_nominal")
        )
    ]


def _latest_forecast(code: str) -> tuple[ForecastRun | None, dict | None]:
    """The single most recent SARIMA run for the group, and its points.

    LATEST-RUN RULE: there can be many ForecastRun rows (each generate_forecast
    appends a new immutable run). We serve exactly ONE run — the most recent by
    created_at for (model=sarima, group) — and only that run's points, ordered by
    period. Points from different runs are never merged. No run -> None.
    """
    run = (
        ForecastRun.objects.filter(model_name=MODEL_NAME, commodity_group__code=code)
        .order_by("-created_at")
        .first()
    )
    if run is None:
        return None, None
    points = [
        {
            "period": period.strftime("%Y-%m"),
            "horizon_step": step,
            "value": float(value),
        }
        for period, step, value in (
            run.points.order_by("period").values_list("period", "horizon_step", "value_pred")
        )
    ]
    return run, {
        "model": run.model_name,
        "run_id": run.pk,
        "train_end": run.train_end.strftime("%Y-%m"),
        "points": points,
    }


def _naive_forecast(run: ForecastRun | None, forecast: dict | None, history: list[dict]) -> dict | None:
    """naive_1 ("no change") baseline for the same periods as the SARIMA run.

    Anchored to the run's train_end — NOT the latest observed month — so both
    forecast lines leave from the same observed point even when newer months have
    been ingested since the run. Computed per request; never stored.
    """
    if run is None:
        return None
    anchor_period = forecast["train_end"]
    anchor = next((p for p in history if p["period"] == anchor_period), None)
    if anchor is None:
        return None  # train_end no longer observed — nothing honest to anchor to
    return {
        "model": "naive_1",
        "train_end": anchor_period,
        "anchor_value": anchor["value"],
        "points": [
            {"period": p["period"], "horizon_step": p["horizon_step"], "value": anchor["value"]}
            for p in forecast["points"]
        ],
    }


def _scoreboard(code: str) -> dict | None:
    """This group's walk-forward scoreboard entries, or None if unavailable.

    A missing, unreadable or incomplete artifact degrades to null rather than
    breaking the chart.
    """
    try:
        payload = json.loads(SCOREBOARD_PATH.read_text(encoding="utf-8"))
        entries = payload["groups"][code]
        return {
            "eval_end": payload["eval_end"],
            "tie_threshold": payload["tie_threshold"],
            "baseline": payload.get("baseline", "naive_1"),
            "horizons": entries,
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def forecast_api(request, code: str):
    """GET /api/forecast/<code>/ — observed history, SARIMA, naive_1 and scoreboard.

    ``code`` is case-insensitive; an unknown code is a JSON 404, never a 500.
    """
    code = code.upper()
    if code not in GROUP_CODES:
        return JsonResponse(
            {"error": f"Unknown series {code.lower()!r}.", "known": [c.lower() for c in GROUP_CODES]},
            status=404,
        )

    history = _history(code)
    run, forecast = _latest_forecast(code)
    return JsonResponse(
        {
            "commodity_group": code,
            "label": SERIES_LABELS[code],
            "history": history,
            "forecast": forecast,
            "naive_forecast": _naive_forecast(run, forecast, history),
            "scoreboard": _scoreboard(code),
        }
    )


def forecast_page(request):
    """GET /forecast/ — a single chart page that fetches the JSON and plots it."""
    series = [{"code": code.lower(), "label": SERIES_LABELS[code]} for code in GROUP_CODES]
    return render(request, "forecasting/forecast.html", {"series": series})
