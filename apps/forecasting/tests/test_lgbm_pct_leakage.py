"""No-look-ahead tests for the Phase 9a.1 scale-free pooled LightGBM (SPEC §6, §15).

Same standard as ``test_lgbm_leakage.py``: backward-only features, planted
future in the target, cross-series poisoning (with a leaky subclass proving the
test bites), and determinism. Plus the positivity guard on logs, the forecast
transform, and the pre-registered decision rule. Perturbed runs always use a
FRESH forecaster instance. Pure numpy/pandas/lightgbm — no DB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apps.forecasting.baseline import metrics_by_horizon, walk_forward
from apps.forecasting.lgbm import (
    FEATURES,
    MIN_HISTORY,
    LgbmPooledPctForecaster,
    build_log_features,
)
from apps.forecasting.management.commands.evaluate_lgbm_9a1 import decide
from apps.forecasting.tests.test_lgbm_leakage import (
    HORIZONS,
    INITIAL_TRAIN_END,
    PROBE_CUTOFFS,
    _panel,
    _poison_other,
    _poisoned,
)

# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def test_every_log_feature_row_uses_only_data_up_to_t():
    series = _panel()["AAA"]
    values, periods = series.to_numpy(), series.index
    full = build_log_features(values, periods)
    assert list(full.columns) == FEATURES

    for t in range(MIN_HISTORY, len(values)):
        truncated = build_log_features(values[: t + 1], periods[: t + 1]).iloc[t]
        poisoned = build_log_features(_poisoned(values, t), periods).iloc[t]
        pd.testing.assert_series_equal(full.iloc[t], truncated, check_exact=True, check_names=False)
        pd.testing.assert_series_equal(full.iloc[t], poisoned, check_exact=True, check_names=False)
        assert not full.iloc[t].isna().any(), f"row {t} should be fully populated"


def test_log_feature_definitions():
    """chg_* are on LOG changes; month and level_rel_12m are exactly as in 9a."""
    series = _panel()["BBB"]
    y = series.to_numpy()
    t = 40
    row = build_log_features(y, series.index).iloc[t]
    lchg = np.diff(np.log(y))  # lchg[k] = log change ending at k+1
    assert row["chg_lag1"] == pytest.approx(np.log(y[t] / y[t - 1]))
    assert row["chg_lag12"] == pytest.approx(np.log(y[t - 11] / y[t - 12]))
    assert row["chg_mean3"] == pytest.approx(lchg[t - 3 : t].mean())
    assert row["chg_std12"] == pytest.approx(lchg[t - 12 : t].std(ddof=1))
    assert row["level_rel_12m"] == pytest.approx(y[t] / y[t - 11 : t + 1].mean() - 1)
    assert row["month"] == series.index[t].month


def test_log_features_are_scale_free():
    """Multiplying a series by a constant leaves every feature unchanged."""
    series = _panel()["CCC"]
    a = build_log_features(series.to_numpy(), series.index)
    b = build_log_features(series.to_numpy() * 4.0, series.index)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_non_positive_values_are_rejected(bad):
    series = _panel()["AAA"]
    values = series.to_numpy().copy()
    values[50] = bad
    with pytest.raises(ValueError, match="strictly positive"):
        build_log_features(values, series.index)
    with pytest.raises(ValueError, match="strictly positive"):
        LgbmPooledPctForecaster({"AAA": pd.Series(values, index=series.index)}).for_series("AAA")(
            values, 100, 1
        )


# ---------------------------------------------------------------------------
# Planted future + cross-series poisoning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cutoff_idx", PROBE_CUTOFFS)
@pytest.mark.parametrize("h", HORIZONS)
def test_pct_variant_ignores_planted_future_in_target(cutoff_idx, h):
    panel = _panel()
    values = panel["AAA"].to_numpy()
    base = LgbmPooledPctForecaster(panel).for_series("AAA")(values, cutoff_idx, h)

    poisoned_panel = dict(panel)
    poisoned_panel["AAA"] = pd.Series(_poisoned(values, cutoff_idx), index=panel["AAA"].index)
    after = LgbmPooledPctForecaster(poisoned_panel).for_series("AAA")(
        _poisoned(values, cutoff_idx), cutoff_idx, h
    )
    assert np.isfinite(base)
    assert base == after


@pytest.mark.parametrize("cutoff_idx", PROBE_CUTOFFS)
@pytest.mark.parametrize("h", HORIZONS)
@pytest.mark.parametrize("other", ["BBB", "CCC"])
def test_pct_variant_ignores_other_series_future(cutoff_idx, h, other):
    panel = _panel()
    values = panel["AAA"].to_numpy()
    base = LgbmPooledPctForecaster(panel).for_series("AAA")(values, cutoff_idx, h)
    after = LgbmPooledPctForecaster(_poison_other(panel, other, cutoff_idx)).for_series("AAA")(
        values, cutoff_idx, h
    )
    assert base == after


class _LeakyPct(LgbmPooledPctForecaster):
    """Deliberately slices the other series ONE month past the cutoff."""

    def _other_series_upto(self, code, cutoff):
        return self.panel[code].loc[: cutoff + pd.DateOffset(months=1)]


def test_pct_cross_series_poisoning_test_catches_a_planted_leak():
    panel = _panel()
    values = panel["AAA"].to_numpy()
    cutoff_idx, h = PROBE_CUTOFFS[0], 1
    base = _LeakyPct(panel).for_series("AAA")(values, cutoff_idx, h)
    after = _LeakyPct(_poison_other(panel, "BBB", cutoff_idx)).for_series("AAA")(values, cutoff_idx, h)
    assert base != after


# ---------------------------------------------------------------------------
# Forecast transform, determinism
# ---------------------------------------------------------------------------


def test_forecast_is_last_value_times_exp_of_predicted_log_change():
    panel = _panel()
    values = panel["AAA"].to_numpy()
    cutoff_idx, h = PROBE_CUTOFFS[1], 3
    f = LgbmPooledPctForecaster(panel)
    got = f.for_series("AAA")(values, cutoff_idx, h)
    slices = f._slices("AAA", values, cutoff_idx)
    x_now = build_log_features(*slices["AAA"]).iloc[[-1]]
    pred = f._model(slices, h).predict(f._with_series(x_now, "AAA"))[0]
    assert got == pytest.approx(values[cutoff_idx] * np.exp(pred), rel=1e-12)
    assert f.n_fits == 1  # second _model call hit the cache


def _run(panel):
    pooled = LgbmPooledPctForecaster(panel)
    out = {}
    for code, series in panel.items():
        steps = walk_forward(series, pooled.for_series(code), HORIZONS, INITIAL_TRAIN_END)
        out[code] = {h: m["mae"] for h, m in metrics_by_horizon(steps).items()}
    return out


def test_two_identical_runs_give_identical_maes():
    first, second = _run(_panel()), _run(_panel())
    assert first == second
    for maes in first.values():
        assert set(maes) == set(HORIZONS)
        assert all(np.isfinite(v) for v in maes.values())


# ---------------------------------------------------------------------------
# Pre-registered decision rule
# ---------------------------------------------------------------------------


def _row(name, naive, sarima, new):
    return {"row": name, "naive_1": naive, "sarima": sarima, "new": new}


def test_decision_rule_success_needs_two_clear_bests_and_few_losses():
    rows = [
        _row("a", 10.0, 9.0, 8.7),  # d = -3.3% vs SARIMA -> best
        _row("b", 10.0, 11.0, 9.7),  # d = -3.0% vs naive -> best
        _row("c", 10.0, 9.0, 8.85),  # d = -1.7% -> not best (margin)
    ] + [_row(f"x{i}", 10.0, 10.0, 10.5) for i in range(4)]  # 4 LOSES (+5%)
    d = decide(rows, sarima_losses=4)
    assert d["best_rows"] == ["a", "b"] and d["L"] == 4
    assert d["verdict"] == "SUCCESS"


def test_decision_rule_fails_on_one_best_or_too_many_losses():
    one_best = [_row("a", 10.0, 9.0, 8.7), _row("b", 10.0, 9.0, 8.9)]
    assert decide(one_best, sarima_losses=4)["verdict"] == "FAILURE"

    many_losses = [_row("a", 10.0, 9.0, 8.0), _row("b", 10.0, 9.0, 8.0)] + [
        _row(f"x{i}", 10.0, 10.0, 10.5) for i in range(5)
    ]
    d = decide(many_losses, sarima_losses=4)
    assert d["B"] == 2 and d["L"] == 5 and d["verdict"] == "FAILURE"


def test_decision_rule_margin_is_strict():
    """Exactly 2% below the reference is NOT best."""
    rows = [_row("a", 100.0, 200.0, 98.0), _row("b", 100.0, 200.0, 98.0)]
    assert decide(rows, sarima_losses=4)["B"] == 0
