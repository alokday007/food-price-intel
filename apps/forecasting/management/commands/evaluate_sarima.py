"""Evaluate SARIMA on any FFPI series and judge it against naive_1 (Phase 4, generalised in 8a).

Reuses the exact Phase 3 walk-forward harness (``walk_forward`` /
``metrics_by_horizon`` from ``baseline.py``) so the fold structure, cutoffs and
horizons are byte-for-byte identical to the baseline run — the only thing that
changes is the forecaster, and (as of Phase 8a) which commodity group's series is
loaded. Anything else would invalidate the comparison.

The bar to beat is naive_1 (random walk), not seasonal-naive. Phase 3 measured
naive_1 MAE=1.85 at H=1 on OVERALL; the sub-indices are more volatile and each
has its OWN naive_1 bar, so the bar is now *measured on the same folds* in this
same run rather than hard-coded. For OVERALL we additionally assert the measured
bar still matches the recorded Phase 3 number — a regression guard proving the
generalisation did not move the goalposts.

This command prints SARIMA and naive_1 side by side and states, honestly, whether
SARIMA beats / ties / loses. No tuning-until-it-wins: the order is the documented
default for every series, and a loss on a volatile series is a real finding.
"""

from __future__ import annotations

import json

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
from apps.forecasting.series import (
    DEFAULT_GROUP,
    GROUP_CODES,
    assert_contiguous_monthly,
    get_group,
    load_group_series,
)

HORIZONS = (1, 3)
DEFAULT_INITIAL_TRAIN_END = "2015-12"

# Phase 3's recorded naive_1 result on OVERALL. Not used as the bar any more (the
# bar is measured per series below) — kept as a regression check that the shared
# harness still reproduces the historical number for OVERALL.
PHASE3_OVERALL_NAIVE1 = {1: 1.85, 3: 4.359}

# Expected fold counts from Phase 3 — a guard that the harness is truly identical.
# All six FFPI series share one publication cadence and range (1990-01..), so the
# fold structure is identical across groups; a mismatch means real drift.
EXPECTED_FOLDS = {1: 125, 3: 123}

# Photo-finishes are called a tie rather than a spurious win, as a fraction of the
# bar so it scales with each series' own error level.
TIE_TOLERANCE = 0.02


def evaluate_group(code: str, initial_train_end: pd.Timestamp) -> dict:
    """Walk-forward SARIMA vs naive_1 for one group; returns a plain-dict result.

    Split out from ``handle`` so the same evaluation can be driven per series
    without shelling out and re-parsing printed text.
    """
    group = get_group(code)
    series = load_group_series(group)
    assert_contiguous_monthly(series)

    if initial_train_end not in series.index:
        raise CommandError(
            f"--initial-train-end {initial_train_end:%Y-%m} is outside the "
            f"{code} series ({series.index.min():%Y-%m}..{series.index.max():%Y-%m})."
        )

    sarima = SarimaForecaster(
        order=DEFAULT_ORDER,
        seasonal_order=DEFAULT_SEASONAL_ORDER,
        max_horizon=max(HORIZONS),
    )
    sarima_steps = walk_forward(series, sarima, HORIZONS, initial_train_end)
    naive_steps = walk_forward(series, naive_last_predict, HORIZONS, initial_train_end)

    # The comparison is only valid if both forecasters ran the identical folds.
    if {(s.cutoff, s.horizon) for s in sarima_steps} != {
        (s.cutoff, s.horizon) for s in naive_steps
    }:
        raise CommandError(f"{code}: fold mismatch SARIMA vs naive_1 — comparison invalid.")
    for h in HORIZONS:
        folds = count_folds([s for s in sarima_steps if s.horizon == h])
        if folds != EXPECTED_FOLDS[h]:
            raise CommandError(
                f"{code}: H={h} fold count {folds} != Phase 3's {EXPECTED_FOLDS[h]} — "
                "harness drifted."
            )

    sarima_metrics = metrics_by_horizon(sarima_steps)
    naive_metrics = metrics_by_horizon(naive_steps)

    # Regression guard: OVERALL must still reproduce the Phase 3 naive_1 bar.
    if code == "OVERALL":
        for h, recorded in PHASE3_OVERALL_NAIVE1.items():
            measured = naive_metrics[h]["mae"]
            if abs(measured - recorded) > 0.01:
                raise CommandError(
                    f"OVERALL naive_1 H={h} MAE {measured:.3f} != Phase 3's "
                    f"{recorded} — the harness or data changed."
                )

    horizons = {}
    for h in HORIZONS:
        sarima_mae = sarima_metrics[h]["mae"]
        bar = naive_metrics[h]["mae"]
        tol = TIE_TOLERANCE * bar
        if sarima_mae < bar - tol:
            verdict = "BEATS"
        elif sarima_mae > bar + tol:
            verdict = "LOSES"
        else:
            verdict = "TIES"
        horizons[h] = {
            "sarima": sarima_metrics[h],
            "naive_1": naive_metrics[h],
            "verdict": verdict,
        }

    return {
        "group": code,
        "n_months": len(series),
        "first": f"{series.index.min():%Y-%m}",
        "last": f"{series.index.max():%Y-%m}",
        "series_mean": float(series.mean()),
        "horizons": horizons,
    }


class Command(BaseCommand):
    help = "Walk-forward evaluation of SARIMA on one FFPI series, judged vs naive_1."

    def add_arguments(self, parser):
        parser.add_argument(
            "--group",
            default=DEFAULT_GROUP,
            help=(
                f"Commodity group code to evaluate. Default {DEFAULT_GROUP} "
                f"(backward compatible). Known: {', '.join(GROUP_CODES)}."
            ),
        )
        parser.add_argument(
            "--initial-train-end",
            default=DEFAULT_INITIAL_TRAIN_END,
            help=f"Last month (YYYY-MM) of the initial train window. Default {DEFAULT_INITIAL_TRAIN_END}.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit the result as one JSON line (for cross-series summaries).",
        )

    def handle(self, *args, **options):
        code = options["group"].strip().upper()
        try:
            initial_train_end = pd.Timestamp(options["initial_train_end"] + "-01")
        except ValueError:
            raise CommandError(
                f"--initial-train-end must be YYYY-MM, got {options['initial_train_end']!r}."
            )

        if not options["json"]:
            self.stdout.write(
                f"SARIMA order={DEFAULT_ORDER} seasonal_order={DEFAULT_SEASONAL_ORDER}; "
                f"expanding-window walk-forward from {initial_train_end:%Y-%m}, "
                f"re-fit every fold, horizons {list(HORIZONS)}."
            )
            self.stdout.write(f"Fitting SARIMA per fold on {code} (this is the slow part)...")

        result = evaluate_group(code, initial_train_end)

        if options["json"]:
            self.stdout.write(json.dumps(result))
            return

        self.stdout.write(
            f"\n{result['group']} series: {result['n_months']} months "
            f"{result['first']}..{result['last']} (contiguous), "
            f"mean level {result['series_mean']:.1f}."
        )
        self._print_table(result)
        self._print_verdict(result)

    # -- reporting ----------------------------------------------------------

    def _print_table(self, result) -> None:
        self.stdout.write(
            self.style.MIGRATE_HEADING("\nWalk-forward metrics (same folds as Phase 3)")
        )
        header = f"{'model':<12}{'H':>3}{'n':>6}{'MAE':>10}{'RMSE':>10}{'MAPE %':>10}"
        self.stdout.write(header)
        self.stdout.write("-" * len(header))
        for h in HORIZONS:
            for name in ("sarima", "naive_1"):
                m = result["horizons"][h][name]
                self.stdout.write(
                    f"{name:<12}{h:>3}{m['n']:>6}"
                    f"{m['mae']:>10.3f}{m['rmse']:>10.3f}{m['mape']:>10.2f}"
                )
        self.stdout.write("")

    def _print_verdict(self, result) -> None:
        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"VERDICT for {result['group']} (vs naive_1, measured on the same folds)"
            )
        )
        for h in HORIZONS:
            row = result["horizons"][h]
            self.stdout.write(
                f"  SARIMA H={h} MAE = {row['sarima']['mae']:.2f} vs naive_1 "
                f"{row['naive_1']['mae']:.2f} -> {row['verdict']}"
            )
        self.stdout.write(
            "\nReported straight from the documented order — no tuning-to-win. "
            "A tie or loss to the random walk is an honest, valid result."
        )
