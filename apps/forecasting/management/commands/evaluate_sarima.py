"""Evaluate SARIMA on the OVERALL FFPI series and judge it against the bar (Phase 4).

Reuses the exact Phase 3 walk-forward harness (``walk_forward`` /
``metrics_by_horizon`` from ``baseline.py``) so the fold structure, cutoffs and
horizons are byte-for-byte identical to the baseline run — the only thing that
changes is the forecaster. Anything else would invalidate the comparison.

The bar to beat is naive_1 (random walk), not seasonal-naive: Phase 3 measured
naive_1 MAE=1.85 at H=1. This command prints SARIMA and naive_1 side by side and
states, honestly, whether SARIMA beats / ties / loses — no tuning-until-it-wins.
"""

from __future__ import annotations

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from apps.forecasting.baseline import (
    metrics_by_horizon,
    naive_last_predict,
    count_folds,
    walk_forward,
)
from apps.forecasting.sarima import (
    DEFAULT_ORDER,
    DEFAULT_SEASONAL_ORDER,
    SarimaForecaster,
)
from apps.prices.models import PriceIndexMonthly

OVERALL_CODE = "OVERALL"
HORIZONS = (1, 3)
DEFAULT_INITIAL_TRAIN_END = "2015-12"

# The number to beat, carried over from Phase 3's naive_1 backtest.
NAIVE1_BAR = {1: 1.85, 3: 4.359}

# Expected fold counts from Phase 3 — a guard that the harness is truly identical.
EXPECTED_FOLDS = {1: 125, 3: 123}


class Command(BaseCommand):
    help = "Walk-forward evaluation of SARIMA on OVERALL FFPI, judged vs naive_1."

    def add_arguments(self, parser):
        parser.add_argument(
            "--initial-train-end",
            default=DEFAULT_INITIAL_TRAIN_END,
            help=f"Last month (YYYY-MM) of the initial train window. Default {DEFAULT_INITIAL_TRAIN_END}.",
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
                f"--initial-train-end {initial_train_end:%Y-%m} is outside the series."
            )

        self.stdout.write(
            f"OVERALL FFPI series: {len(series)} months "
            f"{series.index.min():%Y-%m}..{series.index.max():%Y-%m} (contiguous)."
        )
        self.stdout.write(
            f"SARIMA order={DEFAULT_ORDER} seasonal_order={DEFAULT_SEASONAL_ORDER}; "
            f"expanding-window walk-forward from {initial_train_end:%Y-%m}, "
            f"re-fit every fold, horizons {list(HORIZONS)}.\n"
        )
        self.stdout.write("Fitting SARIMA per fold (this is the slow part)...")

        sarima = SarimaForecaster(
            order=DEFAULT_ORDER,
            seasonal_order=DEFAULT_SEASONAL_ORDER,
            max_horizon=max(HORIZONS),
        )
        sarima_steps = walk_forward(series, sarima, HORIZONS, initial_train_end)
        naive_steps = walk_forward(series, naive_last_predict, HORIZONS, initial_train_end)

        sarima_metrics = metrics_by_horizon(sarima_steps)
        naive_metrics = metrics_by_horizon(naive_steps)

        self._assert_same_folds(sarima_steps, naive_steps)
        self._print_table(sarima_metrics, naive_metrics)
        self._print_verdict(sarima_metrics)

    # -- data loading (mirrors evaluate_baseline) ---------------------------

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
        expected = pd.date_range(series.index.min(), series.index.max(), freq="MS")
        if len(series) != len(expected) or not series.index.equals(expected):
            missing = expected.difference(series.index)
            raise CommandError(
                f"OVERALL series is not contiguous monthly; missing "
                f"{[f'{m:%Y-%m}' for m in missing]}"
            )

    def _assert_same_folds(self, sarima_steps, naive_steps) -> None:
        """The comparison is only valid if both forecasters ran the identical folds."""
        sarima_keys = {(s.cutoff, s.horizon) for s in sarima_steps}
        naive_keys = {(s.cutoff, s.horizon) for s in naive_steps}
        if sarima_keys != naive_keys:
            raise CommandError(
                "Fold mismatch between SARIMA and naive_1 — comparison invalid."
            )
        for h in HORIZONS:
            folds = count_folds([s for s in sarima_steps if s.horizon == h])
            if folds != EXPECTED_FOLDS[h]:
                raise CommandError(
                    f"H={h} fold count {folds} != Phase 3's {EXPECTED_FOLDS[h]} — "
                    "harness drifted."
                )

    # -- reporting ----------------------------------------------------------

    def _print_table(self, sarima_metrics, naive_metrics) -> None:
        self.stdout.write(self.style.MIGRATE_HEADING("\nWalk-forward metrics (same folds as Phase 3)"))
        header = f"{'model':<12}{'H':>3}{'n':>6}{'MAE':>10}{'RMSE':>10}{'MAPE %':>10}"
        self.stdout.write(header)
        self.stdout.write("-" * len(header))
        for h in HORIZONS:
            for name, m in (("sarima", sarima_metrics[h]), ("naive_1", naive_metrics[h])):
                self.stdout.write(
                    f"{name:<12}{h:>3}{m['n']:>6}"
                    f"{m['mae']:>10.3f}{m['rmse']:>10.3f}{m['mape']:>10.2f}"
                )
        self.stdout.write("")

    def _print_verdict(self, sarima_metrics) -> None:
        self.stdout.write(self.style.MIGRATE_HEADING("VERDICT (vs naive_1 — the Phase 3 bar)"))
        for h in HORIZONS:
            sarima_mae = sarima_metrics[h]["mae"]
            bar = NAIVE1_BAR[h]
            # A small tolerance so a photo-finish is called a tie, not a spurious win.
            tol = 0.02 * bar
            if sarima_mae < bar - tol:
                outcome = "BEATS"
            elif sarima_mae > bar + tol:
                outcome = "LOSES"
            else:
                outcome = "TIES"
            self.stdout.write(
                f"  SARIMA H={h} MAE = {sarima_mae:.2f} vs naive_1 {bar} -> {outcome}"
            )
        self.stdout.write(
            "\nReported straight from the documented order — no tuning-to-win. "
            "A tie or loss to the random walk is an honest, valid Phase 4 result."
        )
