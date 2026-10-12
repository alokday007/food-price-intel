"""Phase 9a: evaluate LightGBM (per-series and pooled) against naive_1 and SARIMA.

Evaluation only — nothing is persisted to the database, nothing is served.

Fairness guarantees, each enforced by an assertion rather than a promise:

* Same harness: ``walk_forward`` / ``metrics_by_horizon`` from ``baseline.py``,
  same initial train end and same ``eval_end`` as the committed scoreboard.
* Same folds: naive_1 is recomputed in this run and must equal the naive_1 MAE
  recorded in ``data/seed/scoreboard.json`` for every group and horizon (to the
  4 dp the scoreboard stores). Identical naive_1 errors over 129/127 folds is
  only possible on identical folds.
* SARIMA is not re-fit: its MAEs, rel_deltas and verdicts are read from the
  scoreboard (deltas are never recomputed from its 4-dp MAEs), after asserting
  the scoreboard's ``eval_end``, ``initial_train_end`` and per-row fold counts
  match this run's.

The scoreboard file is read, never written. Results go to
``data/eval/lgbm_9a.json`` (generated here; do not hand-edit).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from apps.forecasting.baseline import (
    count_folds,
    metrics_by_horizon,
    naive_last_predict,
    walk_forward,
)
from apps.forecasting.display import format_delta_pct
from apps.forecasting.lgbm import (
    FEATURES,
    LGBM_PARAMS,
    LgbmPooledForecaster,
    LgbmSeriesForecaster,
)
from apps.forecasting.management.commands.evaluate_sarima import (
    HORIZONS,
    TIE_REL_THRESHOLD,
    expected_folds,
    relative_delta,
    verdict,
)
from apps.forecasting.series import (
    GROUP_CODES,
    assert_contiguous_monthly,
    get_group,
    load_group_series,
)

SCOREBOARD = Path("data/seed/scoreboard.json")
DEFAULT_OUT = Path("data/eval/lgbm_9a.json")

MODELS = ("sarima", "lgbm_series", "lgbm_pooled")  # each judged vs naive_1
ALL_MODELS = ("naive_1",) + MODELS

# MAE / naive_1 MAE below this is not a plausible win on monthly commodity
# indices — it is the signature of look-ahead leakage. Flagged, never celebrated.
LEAK_RATIO = 0.5


def load_panel(eval_end: pd.Timestamp) -> dict[str, pd.Series]:
    """All six series, each contiguous and truncated at ``eval_end``."""
    panel = {}
    for code in GROUP_CODES:
        series = load_group_series(get_group(code))
        assert_contiguous_monthly(series)
        if eval_end not in series.index:
            raise CommandError(f"{code}: eval_end {eval_end:%Y-%m} not in series.")
        panel[code] = series.loc[:eval_end]
    return panel


def _step_keys(steps):
    return {(s.cutoff, s.horizon) for s in steps}


class Command(BaseCommand):
    help = "Phase 9a: walk-forward LightGBM (series + pooled) vs naive_1 and SARIMA."

    def add_arguments(self, parser):
        parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"Default {DEFAULT_OUT}.")

    def handle(self, *args, **options):
        board = json.loads(SCOREBOARD.read_text(encoding="utf-8"))
        eval_end = pd.Timestamp(board["eval_end"] + "-01")
        initial_train_end = pd.Timestamp(board["initial_train_end"] + "-01")
        if board["eval_end"] != "2026-09":
            raise CommandError(f"scoreboard eval_end is {board['eval_end']}, expected 2026-09.")
        if [int(h) for h in board["horizons"]] != list(HORIZONS):
            raise CommandError(f"scoreboard horizons {board['horizons']} != {list(HORIZONS)}.")

        panel = load_panel(eval_end)
        self.stdout.write(
            f"lightgbm {lightgbm.__version__}; panel {list(panel)}; eval_end {eval_end:%Y-%m}; "
            f"initial_train_end {initial_train_end:%Y-%m}; horizons {list(HORIZONS)}"
        )

        pooled = LgbmPooledForecaster(panel)
        runtime = {"naive_1": 0.0, "lgbm_series": 0.0, "lgbm_pooled": 0.0}
        n_fits_series = 0
        groups = {}

        for code in GROUP_CODES:
            series = panel[code]
            steps = {}
            t0 = time.perf_counter()
            steps["naive_1"] = walk_forward(series, naive_last_predict, HORIZONS, initial_train_end)
            runtime["naive_1"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            steps["lgbm_series"] = walk_forward(
                series, LgbmSeriesForecaster(series.index), HORIZONS, initial_train_end
            )
            runtime["lgbm_series"] += time.perf_counter() - t0
            n_fits_series += len(steps["lgbm_series"])

            t0 = time.perf_counter()
            steps["lgbm_pooled"] = walk_forward(
                series, pooled.for_series(code), HORIZONS, initial_train_end
            )
            runtime["lgbm_pooled"] += time.perf_counter() - t0

            # Identical folds across every model run here.
            ref = _step_keys(steps["naive_1"])
            for name in ("lgbm_series", "lgbm_pooled"):
                if _step_keys(steps[name]) != ref:
                    raise CommandError(f"{code}: {name} folds differ from naive_1.")

            metrics = {name: metrics_by_horizon(s) for name, s in steps.items()}
            entry = {}
            for h in HORIZONS:
                folds = count_folds([s for s in steps["naive_1"] if s.horizon == h])
                if folds != expected_folds(series.index, initial_train_end, h):
                    raise CommandError(f"{code} H={h}: {folds} folds, harness drifted.")
                recorded = board["groups"][code][str(h)]
                naive_mae = metrics["naive_1"][h]["mae"]
                # Proof of identical folds: naive_1 reproduces the scoreboard.
                if round(naive_mae, 4) != recorded["naive_1_mae"]:
                    raise CommandError(
                        f"{code} H={h}: naive_1 MAE {naive_mae:.6f} != scoreboard "
                        f"{recorded['naive_1_mae']} — folds are not identical."
                    )
                if recorded["folds"] != folds:
                    raise CommandError(
                        f"{code} H={h}: scoreboard SARIMA folds {recorded['folds']} != {folds}."
                    )
                maes = {
                    "naive_1": naive_mae,
                    "sarima": recorded["sarima_mae"],  # 4 dp — display/ranking only
                    "lgbm_series": metrics["lgbm_series"][h]["mae"],
                    "lgbm_pooled": metrics["lgbm_pooled"][h]["mae"],
                }
                # Deltas and verdicts come from FULL-precision MAEs, as in the
                # scoreboard. The scoreboard keeps SARIMA's MAE only to 4 dp, so
                # recomputing its delta from that rounded MAE drifts (OILS H=1:
                # -0.022495 vs the scoreboard's -0.02250). Instead take SARIMA's
                # rel_delta and verdict as recorded — both were computed there from
                # the unrounded MAEs — and LightGBM's from this run's unrounded MAEs.
                rel = {
                    m: relative_delta(maes[m], naive_mae) for m in ("lgbm_series", "lgbm_pooled")
                }
                rel["sarima"] = recorded["rel_delta"]
                calls = {m: verdict(maes[m], naive_mae) for m in ("lgbm_series", "lgbm_pooled")}
                calls["sarima"] = recorded["verdict"]
                if verdict(naive_mae * (1 + rel["sarima"]), naive_mae) != calls["sarima"]:
                    raise CommandError(
                        f"{code} H={h}: scoreboard SARIMA verdict {calls['sarima']} disagrees "
                        f"with its rel_delta {rel['sarima']}."
                    )
                ratio = {m: 1 + rel[m] for m in MODELS}  # MAE / naive_1 MAE
                entry[str(h)] = {
                    "folds": folds,
                    "mae": {m: round(v, 4) for m, v in maes.items()},
                    "rel_delta_vs_naive_1": {m: round(rel[m], 5) for m in MODELS},
                    "verdict_vs_naive_1": {m: calls[m] for m in MODELS},
                    "ratio_to_naive_1": {m: round(ratio[m], 4) for m in MODELS},
                    "leak_flags": [m for m in MODELS if ratio[m] < LEAK_RATIO],
                    "best": min(ALL_MODELS, key=lambda m: maes[m]),
                }
            groups[code] = entry
            self.stdout.write(f"  {code} done")

        totals = {m: Counter() for m in MODELS}
        best_counts = Counter()
        for entry in groups.values():
            for row in entry.values():
                for m in MODELS:
                    totals[m][row["verdict_vs_naive_1"][m]] += 1
                best_counts[row["best"]] += 1

        names = pooled.feature_names()
        importance = {}
        for h in HORIZONS:
            gain = pooled.gain[h]
            order = np.argsort(-gain, kind="stable")
            importance[str(h)] = [
                {"feature": names[i], "gain_share": round(float(gain[i] / gain.sum()), 4)}
                for i in order
            ]

        payload = {
            "generated_by": "manage.py evaluate_lgbm (Phase 9a; do not hand-edit)",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "lightgbm_version": lightgbm.__version__,
            "eval_end": board["eval_end"],
            "initial_train_end": board["initial_train_end"],
            "horizons": list(HORIZONS),
            "tie_threshold": TIE_REL_THRESHOLD,
            "baseline": "naive_1",
            "sarima_source": f"{SCOREBOARD.as_posix()} (generated_at {board['generated_at']}; not re-fit)",
            "checks": {
                "naive_1_matches_scoreboard": True,
                "sarima_eval_end_and_folds_match": True,
                "all_models_identical_folds": True,
            },
            "design": {
                "target": "y[t+h] - y[t]; forecast = y[t] + predicted change",
                "strategy": "direct, one model per horizon, refit every fold",
                "features": FEATURES,
                "pooled_extra_feature": "series (categorical)",
                "params": LGBM_PARAMS,
            },
            "leak_ratio_flag_below": LEAK_RATIO,
            "totals": {m: {v: totals[m][v] for v in ("BEATS", "TIES", "LOSES")} for m in MODELS},
            "best_counts": {m: best_counts[m] for m in ALL_MODELS},
            "groups": groups,
            "pooled_importance_gain": importance,
            "runtime_seconds": {k: round(v, 2) for k, v in runtime.items()},
            "fits": {"lgbm_series": n_fits_series, "lgbm_pooled": pooled.n_fits},
        }
        out = Path(options["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        self._report(payload)
        self.stdout.write(self.style.SUCCESS(f"\nWrote {out}"))

    # -- reporting ----------------------------------------------------------

    def _report(self, p) -> None:
        self.stdout.write(
            self.style.SUCCESS(
                "\nnaive_1 reproduces scoreboard.json on all 12 rows (identical folds); "
                "SARIMA eval_end/folds match."
            )
        )
        short = {"sarima": "SARIMA", "lgbm_series": "LGBM-ser", "lgbm_pooled": "LGBM-pool"}
        header = (
            f"{'group':<8}{'H':>2}{'folds':>6}{'naive_1':>9}{'SARIMA':>9}{'LGBM-ser':>10}{'LGBM-pool':>11}"
            f"   {'SARIMA Δ':<15}{'LGBM-ser Δ':<15}{'LGBM-pool Δ':<15}best"
        )
        self.stdout.write(self.style.MIGRATE_HEADING("\nScoreboard (MAE; Δ vs naive_1 + verdict)"))
        self.stdout.write(header)
        self.stdout.write("-" * len(header))
        for code, entry in p["groups"].items():
            for h, row in entry.items():
                mae = row["mae"]
                cells = "".join(
                    f"{format_delta_pct(row['rel_delta_vs_naive_1'][m]):>6}{row['verdict_vs_naive_1'][m]:<8}"
                    for m in MODELS
                )
                self.stdout.write(
                    f"{code:<8}{h:>2}{row['folds']:>6}{mae['naive_1']:>9.3f}{mae['sarima']:>9.3f}"
                    f"{mae['lgbm_series']:>10.3f}{mae['lgbm_pooled']:>11.3f}   {cells}{row['best']}"
                )
        self.stdout.write("-" * len(header))
        for m in MODELS:
            t = p["totals"][m]
            self.stdout.write(
                f"TOTAL {short[m]:<10} BEATS {t['BEATS']}  TIES {t['TIES']}  LOSES {t['LOSES']}"
            )
        self.stdout.write(
            "Best (lowest MAE) per row: "
            + ", ".join(f"{m} {c}" for m, c in p["best_counts"].items())
            + f"   [TIES = |Δ| < {p['tie_threshold']:.0%} of naive_1 MAE]"
        )

        self.stdout.write(self.style.MIGRATE_HEADING("\nSanity: MAE / naive_1 MAE"))
        self.stdout.write(f"{'group':<8}{'H':>2}" + "".join(f"{short[m]:>11}" for m in MODELS) + "  flags")
        flagged = 0
        for code, entry in p["groups"].items():
            for h, row in entry.items():
                flags = row["leak_flags"]
                flagged += len(flags)
                self.stdout.write(
                    f"{code:<8}{h:>2}"
                    + "".join(f"{row['ratio_to_naive_1'][m]:>11.3f}" for m in MODELS)
                    + ("  PROBABLE LEAK: " + ",".join(flags) if flags else "")
                )
        self.stdout.write(
            f"{flagged} ratio(s) below {p['leak_ratio_flag_below']} (probable leak)."
        )

        self.stdout.write(
            self.style.MIGRATE_HEADING("\nLGBM-pooled top-10 feature importance (gain share, summed over all fold models)")
        )
        for h, rows in p["pooled_importance_gain"].items():
            self.stdout.write(f"  H={h}: " + ", ".join(
                f"{r['feature']} {r['gain_share']:.1%}" for r in rows[:10]
            ))

        self.stdout.write(self.style.MIGRATE_HEADING("\nRuntime (seconds, all 6 series)"))
        for k, v in p["runtime_seconds"].items():
            fits = p["fits"].get(k)
            self.stdout.write(f"  {k:<12} {v:>8.2f}" + (f"  ({fits} model fits)" if fits else ""))
