"""Fact tables for the Food Price Intelligence schema (SPEC §4).

These hold the monthly observations. Dimension FKs use ``on_delete=PROTECT`` so
a fact row can never be silently orphaned by deleting a commodity group,
country, or data source. Each table has a unique constraint on its natural key
to enable idempotent upserts during ingestion (SPEC §5).
"""

from django.db import models

from apps.catalog.models import CommodityGroup, Country, DataSource


class PriceIndexMonthly(models.Model):
    """Monthly FFPI / sub-index value (base 2014–2016 = 100)."""

    commodity_group = models.ForeignKey(
        CommodityGroup, on_delete=models.PROTECT, related_name="price_points"
    )
    period = models.DateField(help_text="First day of the month.")
    value_nominal = models.DecimalField(max_digits=8, decimal_places=2)
    value_real = models.DecimalField(
        max_digits=8, decimal_places=2, null=True, blank=True
    )
    source = models.ForeignKey(
        DataSource, on_delete=models.PROTECT, related_name="price_points"
    )
    ingested_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "prices_price_index_monthly"
        ordering = ["commodity_group", "period"]
        constraints = [
            models.UniqueConstraint(
                fields=["commodity_group", "period"],
                name="uq_price_index_group_period",
            )
        ]
        indexes = [models.Index(fields=["period"], name="ix_price_index_period")]

    def __str__(self) -> str:
        return f"{self.commodity_group.code} {self.period:%Y-%m}: {self.value_nominal}"


class CountryFoodCpiMonthly(models.Model):
    """Monthly FAOSTAT food / general CPI for a country."""

    country = models.ForeignKey(
        Country, on_delete=models.PROTECT, related_name="cpi_points"
    )
    period = models.DateField(help_text="First day of the month.")
    food_cpi = models.DecimalField(
        max_digits=10, decimal_places=3, null=True, blank=True
    )
    general_cpi = models.DecimalField(
        max_digits=10, decimal_places=3, null=True, blank=True
    )
    source = models.ForeignKey(
        DataSource, on_delete=models.PROTECT, related_name="cpi_points"
    )

    class Meta:
        db_table = "prices_country_food_cpi_monthly"
        ordering = ["country", "period"]
        constraints = [
            models.UniqueConstraint(
                fields=["country", "period"],
                name="uq_country_cpi_country_period",
            )
        ]
        indexes = [
            models.Index(
                fields=["country", "period"], name="ix_country_cpi_country_period"
            )
        ]

    def __str__(self) -> str:
        return f"{self.country.iso3} {self.period:%Y-%m}"
