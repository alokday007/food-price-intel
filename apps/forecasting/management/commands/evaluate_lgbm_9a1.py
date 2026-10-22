"""Phase 9a.1: pooled LightGBM with a scale-free target, judged by a pre-registered rule.

Pre-registration: ``docs/experiments/9a1_prereg.md`` (committed before this
code existed). Evaluation only — nothing persisted to the DB, nothing served.

Only the new variant (``LgbmPooledPctForecaster``) and naive_1 are run here.
naive_1 is recomputed and must equal, to the 4 dp both files store, the
naive_1 MAE in ``data/eval/lgbm_9a.json`` and — while its eval_end is still
2026-09 — in ``data/seed/scoreboard.json``. Identical naive_1 errors over
129/127 folds is only possible on identical folds. SARIMA and both 9a LightGBM
MAEs are read from ``lgbm_9a.json``, never re-run.

Both input files are read, never written. Results go to
``data/eval/lgbm_9a1.json`` (generated here; do not hand-edit).
"""

from __future__ import annotations

import json
import time
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
from apps.forecasting.lgbm import FEATURES, LGBM_PARAMS, LgbmPooledPctForecaster
from apps.forecasting.management.commands.evaluate_lgbm import load_panel
from apps.forecasting.management.commands.evaluate_sarima import (
    HORIZONS,
    TIE_REL_THRESHOLD,
    expected_folds,
    verdict,
)
from apps.forecasting.series import GROUP_CODES

SCOREBOARD = Path("data/seed/scoreboard.json")
LGBM_9A = Path("data/eval/lgbm_9a.json")
DEFAULT_OUT = Path("data/eval/lgbm_9a1.json")
PREREG = "docs/experiments/9a1_prereg.md"

# Fixed by the pre-registration; not arguments that can drift.
EVAL_END = "2026-09"
EXPECTED_FOLDS = {1: 129, 3: 127}
BEST_MARGIN = 0.02  # "best" = MAE more than 2% below min(SARIMA, naive_1)
MIN_BEST_ROWS = 2
SARIMA_LOSSES_PREREG = 4  # SARIMA's LOSES count in lgbm_9a.json when pre-registered

NEW = "lgbm_pooled_pct"
TABLE_MODELS = ("naive_1", "sarima", "lgbm_pooled", NEW)


def decide(rows: list[dict], sarima_losses: int) -> dict:
    """Apply the pre-registered decision rule to the 12 rows, mechanically.

    Each row needs ``naive_1``, ``sarima`` and ``new`` MAEs (index points).
    ``d = (new - ref) / ref`` with ``ref = min(sarima, naive_1)``;
    best rows ``B = #{d < -BEST_MARGIN}``; losses ``L = #{verdict(new, naive_1) == LOSES}``.
    SUCCESS iff ``B >= MIN_BEST_ROWS`` and ``L <= sarima_losses``.
    """
    best_rows, loss_rows = [], []
    for r in rows:
        ref = min(r["sarima"], r["naive_1"])
        d = (r["new"] - ref) / ref
        if d < -BEST_MARGIN:
            best_rows.append(r["row"])
        if verdict(r["new"], r["naive_1"]) == "LOSES":
            loss_rows.append(r["row"])
    success = len(best_rows) >= MIN_BEST_ROWS and len(loss_rows) <= sarima_losses
    return {
        "best_rows": best_rows,
        "B": len(best_rows),
        "loss_rows": loss_rows,
        "L": len(loss_rows),
        "sarima_losses": sarima_losses,
        "rule": (
            f"SUCCESS iff B >= {MIN_BEST_ROWS} (d < -{BEST_MARGIN:.0%} vs min(SARIMA, naive_1)) "
            f"and L <= {sarima_losses} (LOSES vs naive_1)"
        ),
        "verdict": "SUCCESS" if success else "FAILURE",
    }


class Command(BaseCommand):
    help = "Phase 9a.1: pooled LightGBM, log-change target, vs naive_1 / SARIMA / 9a pooled."

    def add_arguments(self, parser):
        parser.add_argument("--eval-end", default=EVAL_END, help=f"Must be {EVAL_END} (pre-registered).")
        parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"Default {DEFAULT_OUT}.")

    def handle(self, *args, **options):
        if options["eval_end"] != EVAL_END:
            raise CommandError(f"--eval-end is fixed at {EVAL_END} by {PREREG}.")
        prior = json.loads(LGBM_9A.read_text(encoding="utf-8"))
        board = json.loads(SCOREBOARD.read_text(encoding="utf-8"))
        if prior["eval_end"] != EVAL_END:
            raise CommandError(f"{LGBM_9A} eval_end {prior['eval_end']} != {EVAL_END}.")
        if prior["horizons"] != list(HORIZONS):
            raise CommandError(f"{LGBM_9A} horizons {prior['horizons']} != {list(HORIZONS)}.")
        if prior["design"]["params"] != LGBM_PARAMS:
            raise CommandError("LGBM_PARAMS differ from those 9a was run with.")
        if prior["design"]["features"] != FEATURES:
            raise CommandError("FEATURES differ from those 9a was run with.")
        sarima_losses = prior["totals"]["sarima"]["LOSES"]
        if sarima_losses != SARIMA_LOSSES_PREREG:
            raise CommandError(
                f"SARIMA LOSES in {LGBM_9A} is {sarima_losses}, pre-registered {SARIMA_LOSSES_PREREG}."
            )
        check_board = board["eval_end"] == EVAL_END

        eval_end = pd.Timestamp(EVAL_END + "-01")
        initial_train_end = pd.Timestamp(prior["initial_train_end"] + "-01")
        panel = load_panel(eval_end)
        self.stdout.write(
            f"lightgbm {lightgbm.__version__}; panel {list(panel)}; eval_end {EVAL_END}; "
            f"initial_train_end {prior['initial_train_end']}; horizons {list(HORIZONS)}"
        )

        model = LgbmPooledPctForecaster(panel)
        runtime = 0.0
        groups, decision_rows = {}, []
        for code in GROUP_CODES:
            series = panel[code]
            naive_steps = walk_forward(series, naive_last_predict, HORIZONS, initial_train_end)
            t0 = time.perf_counter()
            new_steps = walk_forward(series, model.for_series(code), HORIZONS, initial_train_end)
            runtime += time.perf_counter() - t0
            if {(s.cutoff, s.horizon) for s in new_steps} != {(s.cutoff, s.horizon) for s in naive_steps}:
                raise CommandError(f"{code}: {NEW} folds differ from naive_1.")
            naive_m, new_m = metrics_by_horizon(naive_steps), metrics_by_horizon(new_steps)

            entry = {}
            for h in HORIZONS:
                folds = count_folds([s for s in naive_steps if s.horizon == h])
                if folds != EXPECTED_FOLDS[h] or folds != expected_folds(series.index, initial_train_end, h):
                    raise CommandError(f"{code} H={h}: {folds} folds, expected {EXPECTED_FOLDS[h]}.")
                rec = prior["groups"][code][str(h)]
                if rec["folds"] != folds:
                    raise CommandError(f"{code} H={h}: 9a folds {rec['folds']} != {folds}.")
                naive_mae, new_mae = naive_m[h]["mae"], new_m[h]["mae"]
                # Proof of identical folds, against both recorded runs.
                if round(naive_mae, 4) != rec["mae"]["naive_1"]:
                    raise CommandError(
                        f"{code} H={h}: naive_1 {naive_mae:.6f} != lgbm_9a.json {rec['mae']['naive_1']}."
                    )
                if check_board and round(naive_mae, 4) != board["groups"][code][str(h)]["naive_1_mae"]:
                    raise CommandError(
                        f"{code} H={h}: naive_1 {naive_mae:.6f} != scoreboard "
                        f"{board['groups'][code][str(h)]['naive_1_mae']}."
                    )

                maes = {
                    "naive_1": naive_mae,
                    "sarima": rec["mae"]["sarima"],
                    "lgbm_series": rec["mae"]["lgbm_series"],
                    "lgbm_pooled": rec["mae"]["lgbm_pooled"],
                    NEW: new_mae,
                }
                ref_model = "sarima" if maes["sarima"] < naive_mae else "naive_1"
                ref = maes[ref_model]
                delta_vs_ref = (new_mae - ref) / ref
                rel_vs_naive = (new_mae - naive_mae) / naive_mae
                entry[str(h)] = {
                    "folds": folds,
                    "mae": {m: round(v, 4) for m, v in maes.items()},
                    "ref_model": ref_model,
                    "delta_vs_ref": round(delta_vs_ref, 5),
                    "rel_delta_vs_naive_1": round(rel_vs_naive, 5),
                    "verdict_vs_naive_1": verdict(new_mae, naive_mae),
                    "vs_9a_pooled": "better" if round(new_mae, 4) < maes["lgbm_pooled"]
                    else "worse" if round(new_mae, 4) > maes["lgbm_pooled"] else "equal",
                    "best": min(TABLE_MODELS, key=lambda m: maes[m]),
                }
                decision_rows.append(
                    {"row": f"{code} H={h}", "naive_1": naive_mae, "sarima": maes["sarima"], "new": new_mae}
                )
            groups[code] = entry
            self.stdout.write(f"  {code} done")

        decision = decide(decision_rows, sarima_losses)
        cmp = [row["vs_9a_pooled"] for e in groups.values() for row in e.values()]
        vs_9a = {k: cmp.count(k) for k in ("better", "worse", "equal")}
        totals = {v: sum(row["verdict_vs_naive_1"] == v for e in groups.values() for row in e.values())
                  for v in ("BEATS", "TIES", "LOSES")}

        names = model.feature_names()
        importance = {}
        for h in HORIZONS:
            gain = model.gain[h]
            order = np.argsort(-gain, kind="stable")
            importance[str(h)] = [
                {"feature": names[i], "gain_share": round(float(gain[i] / gain.sum()), 4)} for i in order
            ]

        payload = {
            "generated_by": "manage.py evaluate_lgbm_9a1 (Phase 9a.1; do not hand-edit)",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "preregistration": PREREG,
            "lightgbm_version": lightgbm.__version__,
            "eval_end": EVAL_END,
            "initial_train_end": prior["initial_train_end"],
            "horizons": list(HORIZONS),
            "tie_threshold": TIE_REL_THRESHOLD,
            "comparators_source": f"{LGBM_9A.as_posix()} (generated_at {prior['generated_at']}; not re-run)",
            "checks": {
                "naive_1_matches_lgbm_9a": True,
                "naive_1_matches_scoreboard": True if check_board else "skipped (scoreboard eval_end moved)",
                "folds_129_127": True,
                "params_and_features_match_9a": True,
            },
            "design": {
                "target": "log(y[t+h] / y[t]); forecast = y[t] * exp(predicted log change)",
                "change_features": "chg_* built on monthly log changes; month and level_rel_12m as 9a",
                "strategy": "direct, one model per horizon, refit every fold; series categorical",
                "features": FEATURES,
                "params": LGBM_PARAMS,
            },
            "decision": decision,
            "totals_vs_naive_1": totals,
            "vs_9a_pooled": vs_9a,
            "groups": groups,
            "importance_gain": importance,
            "runtime_seconds": round(runtime, 2),
            "fits": model.n_fits,
        }
        out = Path(options["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._report(payload)
        self.stdout.write(self.style.SUCCESS(f"\nWrote {out}"))

    def _report(self, p) -> None:
        short = {"naive_1": "naive_1", "sarima": "SARIMA", "lgbm_pooled": "pool-9a", NEW: "pool-pct"}
        header = (
            f"{'group':<8}{'H':>2}" + "".join(f"{short[m]:>10}" for m in TABLE_MODELS)
            + f"  {'best':<16}{'Δ vs ref':>9}  {'ref':<8}{'vs naive_1':<11}vs 9a"
        )
        self.stdout.write(self.style.MIGRATE_HEADING("\n9a.1 table (MAE, index points)"))
        self.stdout.write(header)
        self.stdout.write("-" * len(header))
        for code, entry in p["groups"].items():
            for h, row in entry.items():
                self.stdout.write(
                    f"{code:<8}{h:>2}" + "".join(f"{row['mae'][m]:>10.3f}" for m in TABLE_MODELS)
                    + f"  {row['best']:<16}{format_delta_pct(row['delta_vs_ref']):>9}  {row['ref_model']:<8}"
                    f"{row['verdict_vs_naive_1']:<11}{row['vs_9a_pooled']}"
                )
        d = p["decision"]
        self.stdout.write(self.style.MIGRATE_HEADING("\nPre-registered decision rule"))
        self.stdout.write(f"  {d['rule']}")
        self.stdout.write(f"  B = {d['B']}  best rows: {d['best_rows'] or '-'}")
        self.stdout.write(f"  L = {d['L']}  (SARIMA: {d['sarima_losses']})  loss rows: {d['loss_rows'] or '-'}")
        self.stdout.write(f"  VERDICT: {d['verdict']}")
        t, v = p["totals_vs_naive_1"], p["vs_9a_pooled"]
        self.stdout.write(f"\n{NEW} vs naive_1: BEATS {t['BEATS']}  TIES {t['TIES']}  LOSES {t['LOSES']}")
        self.stdout.write(f"vs 9a pooled: better {v['better']}  worse {v['worse']}  equal {v['equal']}")
        self.stdout.write(self.style.MIGRATE_HEADING("\nTop-10 feature importance (gain share, all fold models)"))
        for h, rows in p["importance_gain"].items():
            self.stdout.write(f"  H={h}: " + ", ".join(f"{r['feature']} {r['gain_share']:.1%}" for r in rows[:10]))
        self.stdout.write(f"\nRuntime {p['runtime_seconds']:.1f}s, {p['fits']} model fits")
