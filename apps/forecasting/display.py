"""The one place a stored rel_delta becomes a displayed percentage.

The live site renders a verdict's percentage in the browser as
``(Math.abs(rel_delta) * 100).toFixed(1)``. ``toFixed`` rounds the exact
binary value of its argument to 1 decimal and breaks an exact tie UPWARD, so
``0.0225 -> 2.25 -> "2.3"``. Python's ``f"{x:.1%}"`` breaks ties to EVEN and
prints ``"2.2"`` for the same stored number — which is how the README once
said -2.2% while the site said 2.3% for Vegetable Oils at H=1.

Every Python surface that prints a delta (the README scoreboard via
``render_scoreboard``, the Phase 9a report via ``evaluate_lgbm``) goes through
``format_delta_pct`` so it reads exactly as the site does.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def format_delta_pct(rel_delta: float) -> str:
    """Signed 1-dp percent, magnitude rounded as the site's ``toFixed(1)`` does.

    The magnitude is computed exactly as the site does — ``abs(rel_delta) * 100``
    in double precision — and that double's exact value (``Decimal(float)`` is
    exact) is rounded half-up to 1 dp. Sign: ``-`` when the model's MAE is lower
    than the baseline's, ``+`` otherwise. E.g. ``-0.0225 -> "-2.3%"``.
    """
    magnitude = Decimal(abs(rel_delta) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{'-' if rel_delta < 0 else '+'}{magnitude}%"
