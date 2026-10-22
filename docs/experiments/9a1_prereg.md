# Phase 9a.1 — Pre-registration: pooled LightGBM with a scale-free target

Written and committed **before any 9a.1 model code exists or runs.** Everything
below is fixed. Results that come in later do not change the design, the
evaluation, or the decision rule.

## Hypothesis

In 9a, pooled LightGBM predicted the change in **index points**, `y[t+h] - y[t]`.
Its L2 training loss was therefore dominated by the most volatile series (Oils
moves about 4× more per month than Meat), so the shared trees mostly fit Oils and
Sugar. A **scale-free target** puts all six series on a common scale. Pooled
LightGBM can then learn from all six series equally, and this should turn the
pooling into better forecasts.

## The single change vs 9a pooled

| | 9a pooled (`LgbmPooledForecaster`) | 9a.1 pooled-pct (new) |
|---|---|---|
| Target | `y[t+h] - y[t]` | `log(y[t+h] / y[t])` |
| Base change series | `chg[t] = y[t] - y[t-1]` | `chg[t] = log(y[t]) - log(y[t-1])` |
| `chg_lag1..12` | lags of point change | lags of log change |
| `chg_mean{3,6,12}`, `chg_std{3,6,12}` | rolling over point change | rolling over log change |
| Forecast | `y[t] + ŷ` | `y[t] * exp(ŷ)` |

**Identical to 9a pooled. These are not to be touched:**

- `LGBM_PARAMS`, exactly as in `apps/forecasting/lgbm.py` (L2 objective,
  300 trees, lr 0.05, 15 leaves, min_child_samples 20, subsample 0.8/freq 1,
  colsample 0.8, seed 42, deterministic, row-wise, 1 thread). No tuning and no
  early stopping.
- `month` feature (calendar month of `t`).
- `level_rel_12m = y[t] / mean(y[t-11..t]) - 1`. It is already scale-free, so it
  stays unchanged.
- Series code as a categorical feature. One model is trained on all six series.
  There are no cross-series *input* features.
- Direct multi-step: a separate model per horizon (H=1, H=3). The model is refit
  at every fold.
- The same no-leakage contract applies. Every series is sliced at the fold's
  cutoff date before any feature or label is built. A training row is used only
  if its label month `t + h` is `<= cutoff`.
- Same walk-forward harness, same cutoffs, same MAE.
- The two existing 9a forecasters are left unchanged.

**Guard:** before any log is taken, the code asserts that every value it uses is
strictly positive.

## Evaluation (fixed)

- `--eval-end 2026-09` is fixed, even if newer data arrives before the run.
- `initial_train_end` 2015-12 → **129 folds at H=1, 127 folds at H=3**. The run
  asserts these fold counts.
- **12 rows**: {OVERALL, CEREALS, OILS, DAIRY, MEAT, SUGAR} × {H=1, H=3}.
- **Metric: MAE in index points**, the same metric used for every other model.
  The log-scale target only affects training. Forecasts go back to points before
  they are scored.
- **Comparators:**
  - naive_1 is recomputed in the run. The run asserts that it equals
    `data/seed/scoreboard.json` (if that file's eval_end is still 2026-09) and
    `data/eval/lgbm_9a.json`.
  - SARIMA, LGBM-series (9a) and LGBM-pooled (9a) MAEs are read from
    `lgbm_9a.json` and are not re-run.

## Decision rule (fixed now, applied mechanically in code)

For each row `r`, let `ref(r) = min(MAE_SARIMA(r), MAE_naive_1(r))` and
`d(r) = (MAE_new(r) - ref(r)) / ref(r)`.

1. **Best rows:** `B = #{r : d(r) < -0.02}`, meaning the new model is strictly
   more than 2% below the better of SARIMA and naive_1.
2. **Loss rows:** `L = #{r : verdict(MAE_new(r), MAE_naive_1(r)) == "LOSES"}`.
   This uses the existing `verdict()` with `TIE_REL_THRESHOLD = 0.02`.
   SARIMA's count on the same rows is **4** (BEATS 6 / TIES 2 / LOSES 4, per
   `lgbm_9a.json`).

**SUCCESS** if and only if `B >= 2` **and** `L <= 4`.
**FAILURE** otherwise.

If the result is FAILURE, the LightGBM line of work **closes**. There will be no
further variants: no new features, params, targets, objectives or ensembles.

Note: this "best" uses a 2% margin, so it is stricter than the plain
lowest-MAE `best` field in `lgbm_9a.json`. The verdict uses the definition above
only.

## What happens either way

- Results are written by code to `data/eval/lgbm_9a1.json`. That file holds the
  12-row MAEs, `d(r)`, the verdicts vs naive_1, `B`, `L`, the computed verdict,
  the 9a-pooled vs 9a.1 row comparison, and the top feature importances at H=1
  and H=3.
- `data/seed/` and `data/eval/lgbm_9a.json` are not modified.
- Leakage tests for the new variant (planted-future, cross-series poisoning,
  features use only data `<= t`, determinism) must pass to the same standard
  as 9a.
- The README is not changed in 9a.1. Updating it is a separate follow-up,
  whatever the outcome.
