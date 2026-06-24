"""Django admin registrations for the price fact tables."""

from django.contrib import admin

from .models import CountryFoodCpiMonthly, PriceIndexMonthly


@admin.register(PriceIndexMonthly)
class PriceIndexMonthlyAdmin(admin.ModelAdmin):
    list_display = (
        "commodity_group",
        "period",
        "value_nominal",
        "value_real",
        "source",
        "ingested_at",
    )
    list_filter = ("commodity_group", "source", "period")
    date_hierarchy = "period"
    raw_id_fields = ("commodity_group", "source")


@admin.register(CountryFoodCpiMonthly)
class CountryFoodCpiMonthlyAdmin(admin.ModelAdmin):
    list_display = ("country", "period", "food_cpi", "general_cpi", "source")
    list_filter = ("source", "period")
    date_hierarchy = "period"
    raw_id_fields = ("country", "source")
    search_fields = ("country__iso3", "country__name")
