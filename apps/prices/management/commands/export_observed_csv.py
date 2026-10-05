"""Export all six observed FFPI series to a committed seed CSV (Phase 7a, all groups in 8b).

This dumps every ``PriceIndexMonthly`` row for the six FFPI groups
(commodity_group, period, value_nominal, value_real) to
``data/seed/ffpi_observed.csv``. That file IS committed: it is small, public
CC-BY FAO data, and it makes the deploy reproducible without shipping the raw
xlsx workbook. ``data/seed/`` is deliberately NOT gitignored (only ``data/raw/``
is). The deploy-time counterpart that loads this file is ``seed_from_csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.forecasting.series import GROUP_CODES
from apps.prices.models import PriceIndexMonthly

SEED_PATH = Path(settings.BASE_DIR) / "data" / "seed" / "ffpi_observed.csv"
HEADER = ["commodity_group", "period", "value_nominal", "value_real"]


class Command(BaseCommand):
    help = "Export observed PriceIndexMonthly rows for all 6 FFPI groups to data/seed/ffpi_observed.csv."

    def handle(self, *args, **options):
        rows = list(
            PriceIndexMonthly.objects.filter(commodity_group__code__in=GROUP_CODES)
            .order_by("commodity_group__code", "period")
            .values_list("commodity_group__code", "period", "value_nominal", "value_real")
        )
        present = {code for code, *_ in rows}
        missing = [code for code in GROUP_CODES if code not in present]
        if missing:
            raise CommandError(
                f"No PriceIndexMonthly rows for {missing} — refusing to write a partial seed."
            )

        SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SEED_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(HEADER)
            for code, period, nominal, real in rows:
                writer.writerow(
                    [code, period.strftime("%Y-%m-%d"), nominal, "" if real is None else real]
                )

        self.stdout.write(self.style.SUCCESS(f"Wrote {len(rows)} rows to {SEED_PATH}"))
        for code in GROUP_CODES:
            periods = [p for c, p, *_ in rows if c == code]
            self.stdout.write(
                f"  {code:<8} {len(periods)} rows  {periods[0]:%Y-%m}..{periods[-1]:%Y-%m}"
            )
