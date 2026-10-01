"""Evaluate SARIMA on any FFPI series and judge it against naive_1 (Phase 4, generalised in 8a).

Reuses the exact Phase 3 walk-forward harness (``walk_forward`` /
``metrics_by_horizon`` from ``baseline.py``) so the fold structure, cutoffs and
horizons are byte-for-byte identical to the baseline run — the only thing that
changes is the forecaster, and (as of Phase 8a) which commodity group's series is
loaded. Anything else would invalidate the comparison.

The bar to beat is naive_1 (random walk), not seasonal-naive. Phase 3 measured
naive_1 MAE=1.85 at H=1 on OVERALL; the sub-indices are more volatile and each
has its OWN naive_1 bar, so the bar is now *measured on the same folds* in this
same run rather than hard-coded. For OVERALL on the Phase 3 window
(``--eval-end 2026-05``) we additionally assert the measured bar still matches
the recorded Phase 3 number — a regression guard proving the generalisation did
not move the goalposts. On any other window (e.g. after a data refresh) that
assertion is skipped, because the old number is simply not the right answer there.

This command prints SARIMA and naive_1 side by side and states, honestly, whether
SARIMA beats / ties / loses. No tuning-until-it-wins: the order is the documented
default for every series, and a loss on a volatile series is a real finding.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

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

# Phase 3's recorded result on OVERALL, measured on data ending 2026-05 with the
# default initial train end. Not the bar any more (the bar is measured per series
# below) — kept as a regression check that the shared harness still reproduces the
# historical numbers. Only meaningful on that exact window, so it fires only for
# ``--group OVERALL --eval-end 2026-05`` (see ``phase3_regression_applies``).
PHASE3_EVAL_END = "2026-05"
PHASE3_OVERALL_NAIVE1 = {1: 1.85, 3: 4.359}
PHASE3_EXPECTED_FOLDS = {1: 125, 3: 123}

# Verdict rule, defined once in ``verdict()``: SARIMA TIES naive_1 when their MAEs
# differ by less than this FRACTION of the naive_1 MAE (i.e. < 2%), otherwise it
# BEATS or LOSES by sign. Relative, so the photo-finish band scales with each
# series' own error level instead of being a fixed index-point gap.
TIE_REL_THRESHOLD = 0.02


def relative_delta(sarima_mae: float, naive_mae: float) -> float:
    """(SARIMA - naive_1) / naive_1; negative means SARIMA has lower error."""
    return (sarima_mae - naive_mae) / naive_mae


def verdict(sarima_mae: float, naive_mae: float) -> str:
    """The single definition of BEATS / TIES / LOSES (see TIE_REL_THRESHOLD)."""
    delta = relative_delta(sarima_mae, naive_mae)
    if abs(delta) < TIE_REL_THRESHOLD:
        return "TIES"
    return "BEATS" if delta < 0 else "LOSES"


def expected_folds(periods: pd.DatetimeIndex, initial_train_end: pd.Timestamp, h: int) -> int:
    """Folds an expanding walk-forward must produce for horizon ``h``.

    Cutoffs run from ``initial_train_end`` to the last month that still has a
    target ``h`` months later inside the series.
    """
    return len(periods) - periods.get_loc(initial_train_end) - h


def phase3_regression_applies(code: str, eval_end: pd.Timestamp, initial_train_end: pd.Timestamp) -> bool:
    """The Phase 3 numbers only hold on OVERALL, the 2026-05 window and default start."""
    return (
        code == "OVERALL"
        and f"{eval_end:%Y-%m}" == PHASE3_EVAL_END
        and f"{initial_train_end:%Y-%m}" == DEFAULT_INITIAL_TRAIN_END
    )


def _parse_month(value: str, flag: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value + "-01")
    except ValueError:
        raise CommandError(f"{flag} must be YYYY-MM, got {value!r}.")


def evaluate_group(
    code: str,
    initial_train_end: pd.Timestamp,
    eval_end: pd.Timestamp | None = None,
) -> dict:
    """Walk-forward SARIMA vs naive_1 for one group; returns a plain-dict result.

    ``eval_end`` truncates the series (inclusive) before walk-forward, so a past
    evaluation window can be reproduced after newer months are ingested. None
    means the last observed month.

    Split out from ``handle`` so the same evaluation can be driven per series
    without shelling out and re-parsing printed text.
    """
    group = get_group(code)
    series = load_group_series(group)
    assert_contiguous_monthly(series)

    if eval_end is None:
        eval_end = series.index.max()
    elif eval_end not in series.index:
        raise CommandError(
            f"--eval-end {eval_end:%Y-%m} is outside the {code} series "
            f"({series.index.min():%Y-%m}..{series.index.max():%Y-%m})."
        )
    series = series.loc[:eval_end]

    if initial_train_end not in series.index:
        raise CommandError(
            f"--initial-train-end {initial_train_end:%Y-%m} is outside the "
            f"{code} series ({series.index.min():%Y-%m}..{series.index.max():%Y-%m})."
        )
    check_phase3 = phase3_regression_applies(code, eval_end, initial_train_end)

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
        expected = expected_folds(series.index, initial_train_end, h)
        if folds != expected:
            raise CommandError(
                f"{code}: H={h} fold count {folds} != expected {expected} — harness drifted."
            )
        if check_phase3 and folds != PHASE3_EXPECTED_FOLDS[h]:
            raise CommandError(
                f"{code}: H={h} fold count {folds} != Phase 3's "
                f"{PHASE3_EXPECTED_FOLDS[h]} — harness drifted."
            )

    sarima_metrics = metrics_by_horizon(sarima_steps)
    naive_metrics = metrics_by_horizon(naive_steps)

    # Regression guard: on the Phase 3 window, OVERALL must still reproduce the
    # recorded naive_1 bar.
    if check_phase3:
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
        naive_mae = naive_metrics[h]["mae"]
        horizons[h] = {
            "sarima": sarima_metrics[h],
            "naive_1": naive_metrics[h],
            "rel_delta": relative_delta(sarima_mae, naive_mae),
            "verdict": verdict(sarima_mae, naive_mae),
        }

    return {
        "group": code,
        "phase3_regression_checked": check_phase3,
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
            "--eval-end",
            default=None,
            help=(
                "Last month (YYYY-MM) of the series to evaluate on; later months are "
                "dropped before walk-forward. Default: last observed month. The Phase 3 "
                f"regression check runs only for --group OVERALL --eval-end {PHASE3_EVAL_END}."
            ),
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit the result as one JSON line (for cross-series summaries).",
        )
        parser.add_argument(
            "--summarize",
            action="store_true",
            help=(
                "Fit nothing: read --json result lines from stdin and print the "
                "cross-series scoreboard with verdict totals counted from them."
            ),
        )

    def handle(self, *args, **options):
        if options["summarize"]:
            self._summarize(sys.stdin)
            return

        code = options["group"].strip().upper()
        initial_train_end = _parse_month(options["initial_train_end"], "--initial-train-end")
        eval_end = (
            _parse_month(options["eval_end"], "--eval-end") if options["eval_end"] else None
        )

        if not options["json"]:
            self.stdout.write(
                f"SARIMA order={DEFAULT_ORDER} seasonal_order={DEFAULT_SEASONAL_ORDER}; "
                f"expanding-window walk-forward from {initial_train_end:%Y-%m}, "
                f"re-fit every fold, horizons {list(HORIZONS)}."
            )
            self.stdout.write(f"Fitting SARIMA per fold on {code} (this is the slow part)...")

        result = evaluate_group(code, initial_train_end, eval_end)

        if options["json"]:
            self.stdout.write(json.dumps(result))
            return

        self.stdout.write(
            f"\n{result['group']} series: {result['n_months']} months "
            f"{result['first']}..{result['last']} (contiguous), "
            f"mean level {result['series_mean']:.1f}."
        )
        if result["phase3_regression_checked"]:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Phase 3 regression check PASSED: naive_1 MAE matches "
                    f"{PHASE3_OVERALL_NAIVE1} and folds match {PHASE3_EXPECTED_FOLDS}."
                )
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
                f"  SARIMA H={h} MAE = {row['sarima']['mae']:.3f} vs naive_1 "
                f"{row['naive_1']['mae']:.3f} ({row['rel_delta']:+.1%}) -> {row['verdict']}"
            )
        self.stdout.write(
            f"\nTIES = |delta| < {TIE_REL_THRESHOLD:.0%} of naive_1 MAE. "
            "Reported straight from the documented order — no tuning-to-win. "
            "A tie or loss to the random walk is an honest, valid result."
        )

    def _summarize(self, stream) -> None:
        """Cross-series scoreboard; totals are counted from verdict(), never typed."""
        results = [json.loads(line) for line in stream if line.strip()]
        if not results:
            raise CommandError("--summarize: no JSON result lines on stdin.")

        header = (
            f"{'group':<9}{'H':>3}{'last':>9}{'SARIMA MAE':>12}{'naive_1 MAE':>13}"
            f"{'delta %':>9}  verdict"
        )
        self.stdout.write(header)
        self.stdout.write("-" * len(header))
        tally: Counter[str] = Counter()
        for result in results:
            for h in HORIZONS:
                # JSON turns the int horizon keys into strings.
                row = result["horizons"][str(h)]
                sarima_mae = row["sarima"]["mae"]
                naive_mae = row["naive_1"]["mae"]
                call = verdict(sarima_mae, naive_mae)
                tally[call] += 1
                self.stdout.write(
                    f"{result['group']:<9}{h:>3}{result['last']:>9}{sarima_mae:>12.3f}"
                    f"{naive_mae:>13.3f}{relative_delta(sarima_mae, naive_mae):>+9.1%}  {call}"
                )
        self.stdout.write("-" * len(header))
        self.stdout.write(
            f"TOTAL ({sum(tally.values())} group-horizons): "
            + " / ".join(f"{v} {tally[v]}" for v in ("BEATS", "TIES", "LOSES"))
            + f"   [TIES = |delta| < {TIE_REL_THRESHOLD:.0%} of naive_1 MAE]"
        )
