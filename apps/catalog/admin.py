"""Django admin registrations for the catalog dimension tables."""

from django.contrib import admin

from .models import CommodityGroup, Country, DataSource


@admin.register(CommodityGroup)
class CommodityGroupAdmin(admin.ModelAdmin):
    list_display = ("code", "name")
    list_filter = ("code",)
    search_fields = ("code", "name")


@admin.register(Country)
class CountryAdmin(admin.ModelAdmin):
    list_display = ("iso3", "m49_code", "name", "region")
    list_filter = ("region",)
    search_fields = ("iso3", "name")


@admin.register(DataSource)
class DataSourceAdmin(admin.ModelAdmin):
    list_display = ("name", "url", "licence", "last_ingested_at")
    list_filter = ("licence",)
    search_fields = ("name",)
