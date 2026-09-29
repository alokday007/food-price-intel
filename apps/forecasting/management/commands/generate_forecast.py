"""Fit SARIMA on a full FFPI series and persist a forward forecast (Phase 5, generalised in 8a).

This is the persistence step, not an evaluation: unlike ``evaluate_sarima`` (which
walk-forwards over history), here we fit once on ALL observed data through the
latest real month and predict the next H months that do NOT yet exist in the data.
Those predictions are written to ``ForecastPoint`` — never to
``prices.PriceIndexMonthly``, which holds observed FAO data only.

Idempotency strategy: **each run is a new immutable record** (option (a)). A run
captures "what the model predicted at this moment, fitting through train_end";
re-running is a fresh forecast event, so it creates a new ForecastRun rather than
mutating history. This keeps an audit trail of successive forecasts and is the
natural fit for the model (append-only historical facts). It cannot create
duplicate *points* within a run — unique(run, period) guarantees that.

Phase 8a adds ``--group`` (default OVERALL, so existing invocations are
unchanged). Each group gets its own run carrying its own commodity_group FK, so
the six series' forecasts stay isolated from one another as well as from the
observations table.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.forecasting.models import ForecastPoint, ForecastRun
from apps.forecasting.sarima import (
    DEFAULT_ORDER,
    DEFAULT_SEASONAL_ORDER,
    SarimaForecaster,
)
from apps.forecasting.series import (
    DEFAULT_GROUP,
    GROUP_CODES,
    assert_contiguous_monthly,
    get_group,
    load_group_series,
)
from apps.catalog.models import DataSource
from apps.prices.models import PriceIndexMonthly

FAO_SOURCE_NAME = "FAO FFPI"


class Command(BaseCommand):
    help = "Fit SARIMA on one full FFPI series and persist a forward forecast."

    def add_arguments(self, parser):
        parser.add_argument(
            "--group",
            default=DEFAULT_GROUP,
            help=(
                f"Commodity group code to forecast. Default {DEFAULT_GROUP} "
                f"(backward compatible). Known: {', '.join(GROUP_CODES)}."
            ),
        )
        parser.add_argument(
            "--horizon",
            type=int,
            default=3,
            help="Number of months to forecast after the last observed month (default 3).",
        )

    def handle(self, *args, **options):
        horizon = options["horizon"]
        if horizon < 1:
            raise CommandError("--horizon must be >= 1.")

        group = get_group(options["group"].strip().upper())
        series = load_group_series(group)
        assert_contiguous_monthly(series)

        train_end = series.index[-1]
        forecast_periods = [
            (train_end + pd.DateOffset(months=step)).normalize()
            for step in range(1, horizon + 1)
        ]

        # Fit ONCE on the full observed series and read the H-step forecast.
        values = series.to_numpy(dtype=float)
        cutoff_idx = len(values) - 1  # last observed month is the training cutoff
        forecaster = SarimaForecaster(
            order=DEFAULT_ORDER,
            seasonal_order=DEFAULT_SEASONAL_ORDER,
            max_horizon=horizon,
        )
        preds = [forecaster(values, cutoff_idx, step) for step in range(1, horizon + 1)]

        source = DataSource.objects.filter(name=FAO_SOURCE_NAME).first()
        obs_before = PriceIndexMonthly.objects.count()

        with transaction.atomic():
            run = ForecastRun.objects.create(
                model_name="sarima",
                model_params={
                    "order": list(DEFAULT_ORDER),
                    "seasonal_order": list(DEFAULT_SEASONAL_ORDER),
                },
                commodity_group=group,
                train_end=train_end.date(),
                horizon=horizon,
                source=source,
            )
            points = [
                ForecastPoint(
                    run=run,
                    period=period.date(),
                    horizon_step=step,
                    value_pred=Decimal(str(value)).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    ),
                )
                for step, (period, value) in enumerate(
                    zip(forecast_periods, preds), start=1
                )
            ]
            ForecastPoint.objects.bulk_create(points)

        obs_after = PriceIndexMonthly.objects.count()
        self._report(series, run, points, obs_before, obs_after)

    # -- reporting ----------------------------------------------------------

    def _report(self, series, run, points, obs_before, obs_after) -> None:
        self.stdout.write(self.style.SUCCESS(f"Created ForecastRun #{run.pk}"))
        self.stdout.write(f"  model      : {run.model_name} {run.model_params}")
        self.stdout.write(f"  group      : {run.commodity_group.code}")
        self.stdout.write(f"  train_end  : {run.train_end:%Y-%m}  (last OBSERVED month)")
        self.stdout.write(f"  horizon    : {run.horizon}")
        self.stdout.write("")

        self.stdout.write(self.style.MIGRATE_HEADING("Persisted ForecastPoint rows"))
        header = f"{'period':<10}{'step':>6}{'value_pred':>14}"
        self.stdout.write(header)
        self.stdout.write("-" * len(header))
        for p in points:
            self.stdout.write(f"{p.period:%Y-%m}    {p.horizon_step:>4}{p.value_pred:>14}")
        self.stdout.write("")

        # PERIOD CORRECTNESS + PLAUSIBILITY: last 3 observed beside the forecast.
        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"Continuity for {run.commodity_group.code}: last 3 OBSERVED vs PREDICTED"
            )
        )
        for period, val in series.tail(3).items():
            self.stdout.write(f"  OBSERVED  {period:%Y-%m} = {val:.2f}")
        for p in points:
            self.stdout.write(f"  PREDICTED {p.period:%Y-%m} = {p.value_pred}")
        self.stdout.write("")

        # SEPARATION: predictions must not touch the observations table.
        marker = "UNCHANGED" if obs_before == obs_after else "CHANGED (!!)"
        self.stdout.write(
            self.style.MIGRATE_HEADING("Separation from observations")
        )
        self.stdout.write(
            f"  PriceIndexMonthly.count(): before={obs_before} after={obs_after} -> {marker}"
        )
        self.stdout.write(
            f"  ForecastRun.count()={ForecastRun.objects.count()}  "
            f"ForecastPoint.count()={ForecastPoint.objects.count()}"
        )
