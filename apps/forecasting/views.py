"""Phase 6 read-only views: OVERALL history + latest forecast as JSON, and a chart page.

This phase only *reads*. History comes exclusively from PriceIndexMonthly
(observed FAO data); forecast comes exclusively from ForecastPoint (model output).
The two are never crossed — an observation is not a prediction.
"""

from __future__ import annotations

from django.http import JsonResponse
from django.shortcuts import render

from apps.forecasting.models import ForecastRun
from apps.prices.models import PriceIndexMonthly

OVERALL_CODE = "OVERALL"
MODEL_NAME = "sarima"


def forecast_overall_api(request):
    """GET /api/forecast/overall/ — observed OVERALL history + the latest forecast.

    LATEST-RUN RULE: there can be many ForecastRun rows (each generate_forecast
    appends a new immutable run). We serve exactly ONE run — the most recent by
    created_at for (model=sarima, group=OVERALL) — and only that run's points,
    ordered by period. Points from different runs are never merged. If no run
    exists yet, ``forecast`` is null (never a 500).
    """
    history = [
        {"period": period.strftime("%Y-%m"), "value": float(value)}
        for period, value in (
            PriceIndexMonthly.objects.filter(commodity_group__code=OVERALL_CODE)
            .order_by("period")
            .values_list("period", "value_nominal")
        )
    ]

    latest_run = (
        ForecastRun.objects.filter(
            model_name=MODEL_NAME, commodity_group__code=OVERALL_CODE
        )
        .order_by("-created_at")
        .first()
    )

    forecast = None
    if latest_run is not None:
        points = [
            {
                "period": period.strftime("%Y-%m"),
                "horizon_step": step,
                "value": float(value),
            }
            for period, step, value in (
                latest_run.points.order_by("period").values_list(
                    "period", "horizon_step", "value_pred"
                )
            )
        ]
        forecast = {
            "model": latest_run.model_name,
            "run_id": latest_run.pk,
            "train_end": latest_run.train_end.strftime("%Y-%m"),
            "points": points,
        }

    return JsonResponse(
        {
            "commodity_group": OVERALL_CODE,
            "history": history,
            "forecast": forecast,
        }
    )


def forecast_page(request):
    """GET /forecast/ — a single chart page that fetches the JSON and plots it."""
    return render(request, "forecasting/forecast.html")
