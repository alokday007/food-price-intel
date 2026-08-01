"""Idempotently seed OVERALL PriceIndexMonthly from the committed seed CSV (Phase 7a).

This is the DEPLOY-TIME data path: on the live DB (after migrate + seed_catalog),
run this to load the observed OVERALL series from ``data/seed/ffpi_overall.csv``
without needing the raw xlsx workbook. It uses the same revision-safe upsert as
``ingest_ffpi`` — ``update_or_create`` keyed on (commodity_group, period) — so
re-running corrects revised figures instead of duplicating rows. A second run
therefore creates 0 rows.

``ingest_ffpi`` (the xlsx path) is unchanged and remains the source of the CSV via
``export_observed_csv``; this command is the lightweight loader for deploys.
"""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from apps.catalog.models import CommodityGroup, DataSource
from apps.prices.models import PriceIndexMonthly

OVERALL_CODE = "OVERALL"
SOURCE_NAME = "FAO FFPI"
DEFAULT_CSV = Path(settings.BASE_DIR) / "data" / "seed" / "ffpi_overall.csv"

# Same hard sanity bounds as ingest_ffpi: clearly-corrupt values abort the run.
HARD_MIN, HARD_MAX = Decimal("0"), Decimal("1000")


class Command(BaseCommand):
    help = "Idempotently upsert OVERALL PriceIndexMonthly from data/seed/ffpi_overall.csv."

    def add_arguments(self, parser):
        parser.add_argument(
            "--file",
            default=str(DEFAULT_CSV),
            help=f"Path to the seed CSV (default {DEFAULT_CSV}).",
        )

    def handle(self, *args, **options):
        path = Path(options["file"])
        if not path.exists():
            raise CommandError(f"Seed CSV not found: {path}")

        try:
            group = CommodityGroup.objects.get(code=OVERALL_CODE)
        except CommodityGroup.DoesNotExist:
            raise CommandError(
                f"CommodityGroup '{OVERALL_CODE}' not found — run seed_catalog first."
            )
        try:
            source = DataSource.objects.get(name=SOURCE_NAME)
        except DataSource.DoesNotExist:
            raise CommandError(
                f"DataSource '{SOURCE_NAME}' not found — run seed_catalog first."
            )

        created = updated = 0
        with path.open(newline="", encoding="utf-8") as fh, transaction.atomic():
            reader = csv.DictReader(fh)
            if reader.fieldnames != ["period", "value_nominal"]:
                raise CommandError(
                    f"Unexpected CSV header {reader.fieldnames}; "
                    "expected ['period', 'value_nominal']."
                )
            for line_no, row in enumerate(reader, start=2):
                period = self._parse_period(row["period"], line_no)
                value = self._parse_value(row["value_nominal"], line_no)
                _, was_created = PriceIndexMonthly.objects.update_or_create(
                    commodity_group=group,
                    period=period,
                    defaults={"value_nominal": value, "source": source},
                )
                created += was_created
                updated += not was_created

        source.last_ingested_at = timezone.now()
        source.save(update_fields=["last_ingested_at"])

        self.stdout.write(self.style.SUCCESS("seed_from_csv complete:"))
        self.stdout.write(f"  created : {created}")
        self.stdout.write(f"  updated : {updated}")

    def _parse_period(self, text: str, line_no: int) -> date:
        try:
            y, m, d = (int(part) for part in text.split("-"))
            return date(y, m, d)
        except (ValueError, TypeError):
            raise CommandError(f"Line {line_no}: unparseable period {text!r}.")

    def _parse_value(self, text: str, line_no: int) -> Decimal:
        try:
            value = Decimal(text)
        except InvalidOperation:
            raise CommandError(f"Line {line_no}: unparseable value {text!r}.")
        if not (HARD_MIN < value < HARD_MAX):
            raise CommandError(
                f"Line {line_no}: value {value} outside ({HARD_MIN}, {HARD_MAX})."
            )
        return value
