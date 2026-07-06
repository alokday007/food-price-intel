"""Evaluate the forecasting BASELINE on the OVERALL FFPI series (SPEC §6).

Pulls the global FFPI nominal series from the database, runs expanding-window
walk-forward validation of the seasonal-naive forecaster (and a naive-1 forecaster
for comparison) at horizons H=1 and H=3, and prints MAE / RMSE / MAPE.

This command trains nothing and stores nothing. Its whole job is to establish the
bar that every real model in Phase 4 must beat. If a fancy model can't beat these
numbers, it adds no value — and that has to be reported honestly (SPEC §6).
"""

from __future__ import annotations

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from apps.forecasting.baseline import (
    FORECASTERS,
    count_folds,
    metrics_by_horizon,
    walk_forward,
)
from apps.prices.models import PriceIndexMonthly

OVERALL_CODE = "OVERALL"
HORIZONS = (1, 3)
DEFAULT_INITIAL_TRAIN_END = "2015-12"


class Command(BaseCommand):
    help = "Walk-forward evaluation of the seasonal-naive FFPI baseline (OVERALL)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--initial-train-end",
            default=DEFAULT_INITIAL_TRAIN_END,
            help=(
                "Last month (YYYY-MM) of the initial training window; the first "
                f"walk-forward cutoff. Default {DEFAULT_INITIAL_TRAIN_END}."
            ),
        )

    def handle(self, *args, **options):
        series = self._load_overall_series()
        self._assert_contiguous_monthly(series)

        try:
            initial_train_end = pd.Timestamp(options["initial_train_end"] + "-01")
        except ValueError:
            raise CommandError(
                f"--initial-train-end must be YYYY-MM, got {options['initial_train_end']!r}."
            )
        if initial_train_end not in series.index:
            raise CommandError(
                f"--initial-train-end {initial_train_end:%Y-%m} is outside the series "
                f"({series.index.min():%Y-%m}..{series.index.max():%Y-%m})."
            )

        self.stdout.write(
            f"OVERALL FFPI series: {len(series)} months "
            f"{series.index.min():%Y-%m}..{series.index.max():%Y-%m} (contiguous, no gaps)."
        )
        self.stdout.write(
            f"Initial train window ends {initial_train_end:%Y-%m}; "
            f"expanding-window walk-forward, horizons {list(HORIZONS)}.\n"
        )

        results = {}
        for name, forecaster in FORECASTERS.items():
            steps = walk_forward(series, forecaster, HORIZONS, initial_train_end)
            results[name] = {
                "metrics": metrics_by_horizon(steps),
                "folds": count_folds(steps),
                "steps": len(steps),
            }

        self._print_table(results)
        self._print_comparison(results)

    # -- data loading -------------------------------------------------------

    def _load_overall_series(self) -> pd.Series:
        rows = list(
            PriceIndexMonthly.objects.filter(commodity_group__code=OVERALL_CODE)
            .order_by("period")
            .values_list("period", "value_nominal")
        )
        if not rows:
            raise CommandError(
                f"No {OVERALL_CODE} rows in PriceIndexMonthly — run ingest_ffpi first."
            )
        index = pd.DatetimeIndex([pd.Timestamp(p) for p, _ in rows])
        values = [float(v) for _, v in rows]
        return pd.Series(values, index=index, name=OVERALL_CODE)

    def _assert_contiguous_monthly(self, series: pd.Series) -> None:
        """Fail loudly if any month is missing (SPEC §5: no gaps silently dropped).

        A gap would make horizon arithmetic (t-12, cutoff+h) point at the wrong
        calendar month, silently corrupting every metric — so we refuse to run.
        """
        expected = pd.date_range(series.index.min(), series.index.max(), freq="MS")
        if len(series) != len(expected) or not series.index.equals(expected):
            missing = expected.difference(series.index)
            dupes = series.index[series.index.duplicated()]
            raise CommandError(
                "OVERALL series is not contiguous monthly. "
                f"expected {len(expected)} months, got {len(series)}. "
                f"missing={[f'{m:%Y-%m}' for m in missing]} "
                f"duplicated={[f'{d:%Y-%m}' for d in dupes]}"
            )

    # -- reporting ----------------------------------------------------------

    def _print_table(self, results: dict) -> None:
        self.stdout.write(self.style.MIGRATE_HEADING("Walk-forward metrics"))
        header = f"{'model':<16}{'H':>3}{'n':>6}{'MAE':>10}{'RMSE':>10}{'MAPE %':>10}"
        self.stdout.write(header)
        self.stdout.write("-" * len(header))
        for name, res in results.items():
            for h in HORIZONS:
                m = res["metrics"].get(h)
                if m is None:
                    continue
                self.stdout.write(
                    f"{name:<16}{h:>3}{m['n']:>6}"
                    f"{m['mae']:>10.3f}{m['rmse']:>10.3f}{m['mape']:>10.2f}"
                )
        self.stdout.write("")

    def _print_comparison(self, results: dict) -> None:
        sn = results["seasonal_naive"]
        n1 = results["naive_1"]
        self.stdout.write(
            f"Folds (distinct cutoffs): seasonal_naive={sn['folds']} "
            f"(steps={sn['steps']}), naive_1={n1['folds']} (steps={n1['steps']})."
        )
        self.stdout.write(self.style.MIGRATE_HEADING("\nseasonal_naive vs naive_1 (MAE)"))
        for h in HORIZONS:
            sn_mae = sn["metrics"][h]["mae"]
            n1_mae = n1["metrics"][h]["mae"]
            harder = "naive_1" if n1_mae < sn_mae else "seasonal_naive"
            self.stdout.write(
                f"  H={h}: seasonal_naive MAE={sn_mae:.3f}  naive_1 MAE={n1_mae:.3f}  "
                f"-> harder bar to beat: {harder}"
            )
        self.stdout.write(
            self.style.SUCCESS(
                "\nBaseline established. Phase 4 models must beat the lower MAE above."
            )
        )
