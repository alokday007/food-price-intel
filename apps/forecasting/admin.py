"""Django admin registrations for the forecast persistence tables."""

from django.contrib import admin

from .models import ForecastPoint, ForecastRun


@admin.register(ForecastRun)
class ForecastRunAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "model_name",
        "commodity_group",
        "train_end",
        "horizon",
        "created_at",
        "source",
    )
    list_filter = ("model_name", "commodity_group", "train_end")
    date_hierarchy = "created_at"
    raw_id_fields = ("commodity_group", "source")


@admin.register(ForecastPoint)
class ForecastPointAdmin(admin.ModelAdmin):
    list_display = ("id", "run", "period", "horizon_step", "value_pred")
    list_filter = ("horizon_step", "period")
    date_hierarchy = "period"
    raw_id_fields = ("run",)
