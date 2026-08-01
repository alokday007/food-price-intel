"""Export the observed OVERALL FFPI series to a committed seed CSV (Phase 7a).

This dumps the current ``PriceIndexMonthly`` OVERALL rows (period, value_nominal)
to ``data/seed/ffpi_overall.csv``. That file IS committed: it is small, public
CC-BY FAO data, and it makes the deploy reproducible without shipping the raw
xlsx workbook. ``data/seed/`` is deliberately NOT gitignored (only ``data/raw/``
is). The deploy-time counterpart that loads this file is ``seed_from_csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.prices.models import PriceIndexMonthly

OVERALL_CODE = "OVERALL"
SEED_PATH = Path(settings.BASE_DIR) / "data" / "seed" / "ffpi_overall.csv"


class Command(BaseCommand):
    help = "Export observed OVERALL PriceIndexMonthly rows to data/seed/ffpi_overall.csv."

    def handle(self, *args, **options):
        rows = list(
            PriceIndexMonthly.objects.filter(commodity_group__code=OVERALL_CODE)
            .order_by("period")
            .values_list("period", "value_nominal")
        )
        if not rows:
            raise CommandError(
                f"No {OVERALL_CODE} rows in PriceIndexMonthly — nothing to export."
            )

        SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SEED_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["period", "value_nominal"])
            for period, value in rows:
                writer.writerow([period.strftime("%Y-%m-%d"), value])

        self.stdout.write(
            self.style.SUCCESS(
                f"Wrote {len(rows)} {OVERALL_CODE} rows to {SEED_PATH}"
            )
        )
        self.stdout.write(
            f"  first: {rows[0][0]:%Y-%m-%d},{rows[0][1]}  "
            f"last: {rows[-1][0]:%Y-%m-%d},{rows[-1][1]}"
        )
