"""No-look-ahead tests for the Phase 9a LightGBM forecasters (SPEC §6, §15).

LightGBM adds two leakage surfaces the baselines did not have, and each gets
its own mechanical test:

1. Features and labels are *built* from the series. A feature using a centred
   window, or a training label whose target month lies after the cutoff, would
   leak while the code runs fine. -> feature test + planted-future test.
2. The pooled model reads the *other five* series. If any of them were sliced
   later than the target's cutoff, a different index's future would leak in.
   -> cross-series poisoning test, plus a deliberately leaky subclass proving
   that test actually bites.

Perturbed runs always use a FRESH forecaster instance, so no cache can return
the unperturbed answer and mask a leak. Pure numpy/pandas/lightgbm — no DB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apps.forecasting.baseline import metrics_by_horizon, walk_forward
from apps.forecasting.lgbm import (
    FEATURES,
    MIN_HISTORY,
    LgbmPooledForecaster,
    LgbmSeriesForecaster,
    build_features,
)

HORIZONS = (1, 3)
N_MONTHS = 150
# Late initial train end keeps the suite fast: a handful of folds, each a real fit.
INITIAL_TRAIN_END = pd.Timestamp("2010-01-01") + pd.DateOffset(months=N_MONTHS - 6)
PROBE_CUTOFFS = (110, 130, N_MONTHS - 4)
POISON = 1.0e6


def _panel(n_months: int = N_MONTHS) -> dict[str, pd.Series]:
    """Three synthetic trend + seasonal + noise series sharing one monthly index."""
    idx = pd.date_range("2010-01-01", periods=n_months, freq="MS")
    t = np.arange(n_months)
    rng = np.random.default_rng(7)
    out = {}
    for i, code in enumerate(("AAA", "BBB", "CCC")):
        level = 100 + 20 * i
        values = (
            level
            + (0.1 + 0.05 * i) * t
            + (5 + i) * np.sin(2 * np.pi * (t + 2 * i) / 12)
            + np.cumsum(rng.normal(0, 1.0, n_months))
        )
        out[code] = pd.Series(values, index=idx, name=code)
    return out


def _poisoned(values: np.ndarray, cutoff_idx: int) -> np.ndarray:
    out = np.array(values, dtype=float)
    out[cutoff_idx + 1 :] += POISON
    return out


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def test_every_feature_row_uses_only_data_up_to_t():
    """Row t is identical when built from values[:t+1], and when the future is poisoned."""
    series = _panel()["AAA"]
    values, periods = series.to_numpy(), series.index
    full = build_features(values, periods)
    assert list(full.columns) == FEATURES

    for t in range(MIN_HISTORY, len(values)):
        truncated = build_features(values[: t + 1], periods[: t + 1]).iloc[t]
        poisoned = build_features(_poisoned(values, t), periods).iloc[t]
        pd.testing.assert_series_equal(full.iloc[t], truncated, check_exact=True, check_names=False)
        pd.testing.assert_series_equal(full.iloc[t], poisoned, check_exact=True, check_names=False)
        assert not full.iloc[t].isna().any(), f"row {t} should be fully populated"


def test_feature_definitions_are_backward_looking():
    """Spot-check the documented definitions at one row."""
    series = _panel()["BBB"]
    y = series.to_numpy()
    t = 40
    row = build_features(y, series.index).iloc[t]
    chg = np.diff(y)  # chg[k] = y[k+1] - y[k], i.e. the change ending at k+1
    assert row["chg_lag1"] == pytest.approx(y[t] - y[t - 1])
    assert row["chg_lag12"] == pytest.approx(y[t - 11] - y[t - 12])
    assert row["chg_mean3"] == pytest.approx(chg[t - 3 : t].mean())
    assert row["chg_std12"] == pytest.approx(chg[t - 12 : t].std(ddof=1))
    assert row["level_rel_12m"] == pytest.approx(y[t] / y[t - 11 : t + 1].mean() - 1)
    assert row["month"] == series.index[t].month


# ---------------------------------------------------------------------------
# Planted-future tests (both variants)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cutoff_idx", PROBE_CUTOFFS)
@pytest.mark.parametrize("h", HORIZONS)
def test_series_variant_ignores_planted_future(cutoff_idx, h):
    series = _panel()["AAA"]
    values = series.to_numpy()
    base = LgbmSeriesForecaster(series.index)(values, cutoff_idx, h)
    after = LgbmSeriesForecaster(series.index)(_poisoned(values, cutoff_idx), cutoff_idx, h)
    assert np.isfinite(base)
    assert base == after


@pytest.mark.parametrize("cutoff_idx", PROBE_CUTOFFS)
@pytest.mark.parametrize("h", HORIZONS)
def test_pooled_variant_ignores_planted_future_in_target(cutoff_idx, h):
    """Poison the target series after the cutoff — in the harness array AND in the panel."""
    panel = _panel()
    values = panel["AAA"].to_numpy()
    base = LgbmPooledForecaster(panel).for_series("AAA")(values, cutoff_idx, h)

    poisoned_panel = dict(panel)
    poisoned_panel["AAA"] = pd.Series(_poisoned(values, cutoff_idx), index=panel["AAA"].index)
    after = LgbmPooledForecaster(poisoned_panel).for_series("AAA")(
        _poisoned(values, cutoff_idx), cutoff_idx, h
    )
    assert np.isfinite(base)
    assert base == after


# ---------------------------------------------------------------------------
# Cross-series poisoning (pooled only)
# ---------------------------------------------------------------------------


def _poison_other(panel, other: str, cutoff_idx: int) -> dict[str, pd.Series]:
    out = dict(panel)
    s = panel[other]
    out[other] = pd.Series(_poisoned(s.to_numpy(), cutoff_idx), index=s.index, name=other)
    return out


@pytest.mark.parametrize("cutoff_idx", PROBE_CUTOFFS)
@pytest.mark.parametrize("h", HORIZONS)
@pytest.mark.parametrize("other", ["BBB", "CCC"])
def test_pooled_variant_ignores_other_series_future(cutoff_idx, h, other):
    panel = _panel()
    values = panel["AAA"].to_numpy()
    base = LgbmPooledForecaster(panel).for_series("AAA")(values, cutoff_idx, h)
    after = LgbmPooledForecaster(_poison_other(panel, other, cutoff_idx)).for_series("AAA")(
        values, cutoff_idx, h
    )
    assert base == after


class _LeakyPooled(LgbmPooledForecaster):
    """Deliberately slices the other series ONE month past the cutoff."""

    def _other_series_upto(self, code, cutoff):
        return self.panel[code].loc[: cutoff + pd.DateOffset(months=1)]


def test_cross_series_poisoning_test_catches_a_planted_leak():
    """The poisoning check must fail on a forecaster that peeks one month ahead."""
    panel = _panel()
    values = panel["AAA"].to_numpy()
    cutoff_idx, h = PROBE_CUTOFFS[0], 1
    base = _LeakyPooled(panel).for_series("AAA")(values, cutoff_idx, h)
    after = _LeakyPooled(_poison_other(panel, "BBB", cutoff_idx)).for_series("AAA")(
        values, cutoff_idx, h
    )
    assert base != after


# ---------------------------------------------------------------------------
# Harness integration + determinism
# ---------------------------------------------------------------------------


def _run(panel):
    """Walk-forward MAEs for both variants on every series, fresh forecasters."""
    pooled = LgbmPooledForecaster(panel)
    out = {}
    for code, series in panel.items():
        for name, f in (
            ("series", LgbmSeriesForecaster(series.index)),
            ("pooled", pooled.for_series(code)),
        ):
            steps = walk_forward(series, f, HORIZONS, INITIAL_TRAIN_END)
            out[(code, name)] = {h: m["mae"] for h, m in metrics_by_horizon(steps).items()}
    return out


def test_two_identical_runs_give_identical_maes():
    first, second = _run(_panel()), _run(_panel())
    assert first == second
    for maes in first.values():
        assert set(maes) == set(HORIZONS)
        assert all(np.isfinite(v) for v in maes.values())


def test_pooled_model_is_shared_across_targets_within_a_cutoff():
    """One fit per (cutoff, h), reused for every target series — not one per target."""
    panel = _panel()
    pooled = LgbmPooledForecaster(panel)
    for code, series in panel.items():
        walk_forward(series, pooled.for_series(code), HORIZONS, INITIAL_TRAIN_END)
    n_cutoffs = N_MONTHS - panel["AAA"].index.get_loc(INITIAL_TRAIN_END)
    expected = sum(n_cutoffs - h for h in HORIZONS)
    assert pooled.n_fits == expected
