"""Dimension tables for the Food Price Intelligence star-ish schema (SPEC §4).

These are the slowly-changing reference entities that the fact tables in the
``prices`` app point at: the commodity groups (FFPI + sub-indices), the country
list, and the upstream data sources (for provenance).
"""

from django.db import models


class CommodityGroup(models.Model):
    """A FFPI series: the overall index or one of its five sub-indices."""

    code = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100)

    class Meta:
        db_table = "catalog_commodity_group"
        ordering = ["code"]

    def __str__(self) -> str:
        return f"{self.code} — {self.name}"


class Country(models.Model):
    """A country in the FAOSTAT CPI universe, keyed by ISO3 / UN M49."""

    iso3 = models.CharField(max_length=3, unique=True)
    m49_code = models.IntegerField()
    name = models.CharField(max_length=120)
    region = models.CharField(max_length=80, null=True, blank=True)

    class Meta:
        db_table = "catalog_country"
        ordering = ["name"]
        verbose_name_plural = "countries"

    def __str__(self) -> str:
        return f"{self.iso3} — {self.name}"


class DataSource(models.Model):
    """An upstream dataset, with provenance for the last ingest."""

    name = models.CharField(max_length=120)
    url = models.CharField(max_length=300)
    licence = models.CharField(max_length=80)
    last_ingested_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "catalog_data_source"
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name
