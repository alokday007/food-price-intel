"""No models in Phase 3.

The forecasting app is introduced here to host the baseline forecaster and the
walk-forward validation harness only (SPEC §6). The persisted ML output tables
(``forecast_run`` / ``forecast_point`` from SPEC §4) are deliberately NOT created
yet — nothing is trained or stored in this phase. They arrive in Phase 4 alongside
the first real model that has to beat the baseline established here.
"""
