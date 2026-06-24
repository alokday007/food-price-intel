"""Seed the catalog dimension tables (SPEC §5).

Idempotent: every row goes through ``update_or_create`` keyed on its natural
key, so running this command twice leaves the database in an identical state
and reports zero newly-created rows on the second run.

Seeds:
- 6 CommodityGroup rows (overall FFPI + the 5 sub-indices).
- The country list from ``pycountry`` (ISO3 + UN M49 numeric code + name).
- 2 DataSource rows (FAO FFPI, FAOSTAT CPI).
"""

import pycountry
from django.core.management.base import BaseCommand
from django.db import transaction

from apps.catalog.models import CommodityGroup, Country, DataSource

# (code, display name) — OVERALL plus the five FFPI sub-indices.
COMMODITY_GROUPS = [
    ("OVERALL", "FAO Food Price Index"),
    ("CEREALS", "Cereals Price Index"),
    ("OILS", "Vegetable Oils Price Index"),
    ("DAIRY", "Dairy Price Index"),
    ("MEAT", "Meat Price Index"),
    ("SUGAR", "Sugar Price Index"),
]

# (name, url, licence)
DATA_SOURCES = [
    (
        "FAO FFPI",
        "https://www.fao.org/worldfoodsituation/foodpricesindex/en/",
        "CC BY 4.0",
    ),
    (
        "FAOSTAT CPI",
        "https://www.fao.org/faostat/en/#data/CP",
        "CC BY 4.0",
    ),
]


class Command(BaseCommand):
    help = "Idempotently seed commodity groups, countries, and data sources."

    @transaction.atomic
    def handle(self, *args, **options):
        groups_created = self._seed_commodity_groups()
        countries_created, countries_total = self._seed_countries()
        sources_created = self._seed_data_sources()

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Catalog seed complete:"))
        self.stdout.write(
            f"  CommodityGroup: {groups_created} created "
            f"({CommodityGroup.objects.count()} total)"
        )
        self.stdout.write(
            f"  Country:        {countries_created} created "
            f"({countries_total} total)"
        )
        self.stdout.write(
            f"  DataSource:     {sources_created} created "
            f"({DataSource.objects.count()} total)"
        )

    def _seed_commodity_groups(self) -> int:
        created = 0
        for code, name in COMMODITY_GROUPS:
            _, was_created = CommodityGroup.objects.update_or_create(
                code=code, defaults={"name": name}
            )
            created += was_created
        return created

    def _seed_countries(self) -> tuple[int, int]:
        created = 0
        total = 0
        for country in pycountry.countries:
            m49 = getattr(country, "numeric", None)
            if m49 is None:
                # No UN M49 numeric code → skip (shouldn't happen for real countries).
                continue
            total += 1
            _, was_created = Country.objects.update_or_create(
                iso3=country.alpha_3,
                defaults={
                    "m49_code": int(m49),
                    "name": country.name,
                },
            )
            created += was_created
        return created, total

    def _seed_data_sources(self) -> int:
        created = 0
        for name, url, licence in DATA_SOURCES:
            _, was_created = DataSource.objects.update_or_create(
                name=name, defaults={"url": url, "licence": licence}
            )
            created += was_created
        return created
