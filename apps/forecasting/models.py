"""Forecast persistence tables (SPEC §4, Phase 5).

These hold *predictions*, and predictions only. Observed FAO data lives in
``prices.PriceIndexMonthly`` and must never be mixed with model output — a
forecast is not an observation. So a forecast run and its points are stored here,
entirely separate from the observations table.

A run is an immutable historical record: "at this moment, fitting on data through
train_end, the model predicted these H months." We never overwrite a past run;
re-forecasting creates a new run (see ``generate_forecast``).
"""

from django.db import models

from apps.catalog.models import CommodityGroup, DataSource


class ForecastRun(models.Model):
    """One fit-and-forecast event: model + params + training cutoff + horizon."""

    model_name = models.CharField(max_length=40, help_text="e.g. 'sarima'.")
    # SARIMA order + seasonal_order live here as JSON so a run fully documents the
    # model it came from without a schema change per model family.
    model_params = models.JSONField(default=dict)
    # PROTECT on the dimension: a commodity group must not vanish out from under a
    # run that references it. (Contrast the CASCADE from run -> point below.)
    commodity_group = models.ForeignKey(
        CommodityGroup, on_delete=models.PROTECT, related_name="forecast_runs"
    )
    train_end = models.DateField(help_text="Last observed period used for fitting.")
    horizon = models.IntegerField(help_text="Number of months forecast forward.")
    created_at = models.DateTimeField(auto_now_add=True)
    # Nullable provenance FK; also PROTECT — a source is a dimension row.
    source = models.ForeignKey(
        DataSource,
        on_delete=models.PROTECT,
        related_name="forecast_runs",
        null=True,
        blank=True,
    )

    class Meta:
        db_table = "forecasting_forecast_run"
        ordering = ["-created_at"]
        indexes = [
            models.Index(
                fields=["commodity_group", "created_at"],
                name="ix_forecast_run_group_created",
            )
        ]

    def __str__(self) -> str:
        return (
            f"{self.model_name} {self.commodity_group.code} "
            f"train_end={self.train_end:%Y-%m} H={self.horizon} (#{self.pk})"
        )


class ForecastPoint(models.Model):
    """A single predicted month belonging to a run."""

    # CASCADE from run -> point: a point is meaningless without its run, so
    # deleting a run deletes its points. (Contrast PROTECT on the dimensions
    # above, which must never disappear beneath a run.)
    run = models.ForeignKey(
        ForecastRun, on_delete=models.CASCADE, related_name="points"
    )
    period = models.DateField(help_text="First day of the forecast month.")
    horizon_step = models.IntegerField(help_text="1..H, months ahead of train_end.")
    value_pred = models.DecimalField(max_digits=8, decimal_places=2)

    class Meta:
        db_table = "forecasting_forecast_point"
        ordering = ["run", "period"]
        constraints = [
            models.UniqueConstraint(
                fields=["run", "period"], name="uq_forecast_point_run_period"
            )
        ]
        indexes = [models.Index(fields=["period"], name="ix_forecast_point_period")]

    def __str__(self) -> str:
        return f"run#{self.run_id} {self.period:%Y-%m} (+{self.horizon_step}): {self.value_pred}"
