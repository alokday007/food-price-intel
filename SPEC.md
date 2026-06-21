# Food Price Intelligence — Project Specification

**Author:** Alok (alokday007)
**Status:** v1 spec, ready to build
**Target complexity:** ~8/10
**One-line:** Ingest FAO food-price data into PostgreSQL, forecast the next 1–3 months with ML, flag anomalous price spikes, and auto-generate a plain-English monthly briefing with Claude — served through a Django web app.

---

## 0. How to use this spec

Hand this file to Claude Code one **phase** at a time (see §11 Build Order), not all at once. Each phase ends with a concrete "Definition of Done" you can verify before moving on, so you always have a working app. Build in `D:\Projects\Food Price Intel` — **not** inside OneDrive (that broke your venv last time). Run PostgreSQL and Redis in Docker so you avoid native-Windows install pain.

---

## 1. Goal & scope

Build a dashboard that answers three questions every month:
1. **What happened?** — the latest FAO Food Price Index (FFPI) and its five commodity sub-indices, plus country-level food inflation.
2. **What's next?** — a 1–3 month forecast of the FFPI with confidence intervals.
3. **What's unusual?** — months where prices moved far more than expected (early-warning anomaly flags).
4. **So what?** — a Claude-generated plain-English briefing tying it together.

Out of scope for v1: user accounts, multi-tenant, payments, mobile app.

---

## 2. Data sources

| Source | What | Cadence | Access | Use in app |
|---|---|---|---|---|
| **FAO Food Price Index (FFPI)** | Global index + 5 sub-indices (Cereals, Vegetable Oils, Dairy, Meat, Sugar), nominal & real, 1990→present | Monthly | Excel/CSV download from fao.org/worldfoodsituation/foodpricesindex | Forecasting target + headline charts |
| **FAOSTAT Consumer Price Indices** | Country-level Food CPI & general CPI, monthly from Jan 2000, 245+ countries | Monthly/periodic | FAOSTAT bulk download + API | Per-country views (incl. India) |
| **UN/FAO country list** | ISO3, M49 codes, region names | Static | FAOSTAT definitions | `country` dimension seed |

**Licence/attribution:** FAOSTAT is largely CC BY 4.0; cite FAO as the source and link FFPI + FAOSTAT. Add a "Data & methodology" section to the README (same discipline as Cric Metrics' Cricsheet attribution). Not affiliated with the FAO.

---

## 3. Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │            Django + DRF (web)                │
   Browser ◀────▶│  templates + HTMX + charts │ REST API (/api) │
                 └───────┬───────────────────────────┬─────────┘
                         │ ORM                        │
                 ┌───────▼────────┐          ┌────────▼─────────┐
                 │  PostgreSQL    │◀────────▶│  Celery workers   │
                 │ (relational DB)│  results │  + Celery beat    │
                 └───────▲────────┘          └────────┬─────────┘
                         │                            │ broker/cache
            ┌────────────┴───────────┐        ┌───────▼────────┐
            │ Ingestion mgmt commands│        │     Redis      │
            │ (FFPI, FAOSTAT CPI)    │        └────────────────┘
            └────────────┬───────────┘
                         │
   ML pipeline ──────────┴──────────▶ joblib model artifacts (/models)
   (features → SARIMA + boosting → forecast → anomalies)

   AI layer ── Claude (anthropic SDK) ──▶ monthly briefing (stored, cached)
```

---

## 4. Relational database schema (the core requirement)

Designed as a normalized **star-ish schema** so it genuinely exercises foreign keys, unique constraints, and indexes — the database earns its place, it isn't decorative. Use the Django ORM; understand the SQL it generates.

### Dimension tables

**`catalog_commodity_group`**
| column | type | notes |
|---|---|---|
| id | PK | |
| code | varchar(20) unique | e.g. `OVERALL`, `CEREALS`, `OILS`, `DAIRY`, `MEAT`, `SUGAR` |
| name | varchar(100) | display name |

**`catalog_country`**
| column | type | notes |
|---|---|---|
| id | PK | |
| iso3 | char(3) unique | e.g. `IND` |
| m49_code | int | UN M49 |
| name | varchar(120) | |
| region | varchar(80) | nullable |

**`catalog_data_source`**
| column | type | notes |
|---|---|---|
| id | PK | |
| name | varchar(120) | `FAO FFPI`, `FAOSTAT CPI` |
| url | varchar(300) | |
| licence | varchar(80) | `CC BY 4.0` |
| last_ingested_at | timestamptz | nullable — provenance |

### Fact tables

**`prices_price_index_monthly`** — FFPI + sub-indices
| column | type | notes |
|---|---|---|
| id | PK | |
| commodity_group_id | FK → commodity_group | |
| period | date | first day of month |
| value_nominal | numeric(8,2) | base 2014–2016 = 100 |
| value_real | numeric(8,2) | nullable |
| source_id | FK → data_source | |
| ingested_at | timestamptz | |
| | **unique(commodity_group_id, period)** | enables idempotent upsert |
| | **index(period)** | |

**`prices_country_food_cpi_monthly`** — FAOSTAT
| column | type | notes |
|---|---|---|
| id | PK | |
| country_id | FK → country | |
| period | date | |
| food_cpi | numeric(10,3) | nullable |
| general_cpi | numeric(10,3) | nullable |
| source_id | FK → data_source | |
| | **unique(country_id, period)** | |
| | **index(country_id, period)** | |

### ML / AI output tables

**`forecasting_forecast_run`**
| column | type | notes |
|---|---|---|
| id | PK | |
| model_name | varchar(40) | `seasonal_naive` \| `sarima` \| `lgbm` |
| target_code | varchar(20) | which commodity group |
| horizon_months | int | 1–3 |
| trained_at | timestamptz | |
| metrics_json | jsonb | backtest MAE/RMSE/MAPE |
| params_json | jsonb | model hyperparams |
| artifact_path | varchar(300) | joblib file |

**`forecasting_forecast_point`**
| column | type | notes |
|---|---|---|
| id | PK | |
| run_id | FK → forecast_run | |
| period | date | future month |
| yhat | numeric(8,2) | |
| yhat_lower | numeric(8,2) | CI lower |
| yhat_upper | numeric(8,2) | CI upper |
| | **unique(run_id, period)** | |

**`anomalies_anomaly`**
| column | type | notes |
|---|---|---|
| id | PK | |
| commodity_group_id | FK | |
| period | date | |
| observed_value | numeric(8,2) | |
| expected_value | numeric(8,2) | |
| residual | numeric(8,2) | |
| z_score | numeric(6,2) | |
| severity | varchar(10) | `low`/`med`/`high` |
| method | varchar(30) | e.g. `residual_zscore` |
| detected_at | timestamptz | |

**`briefings_briefing`**
| column | type | notes |
|---|---|---|
| id | PK | |
| period | date | the month briefed |
| claude_model | varchar(40) | |
| content_md | text | the briefing |
| data_hash | char(64) | sha256 of inputs → cache key |
| forecast_run_id | FK → forecast_run | nullable |
| generated_at | timestamptz | |
| | **unique(period, data_hash)** | don't regenerate/recharge |

---

## 5. Data ingestion (ETL)

Implement as **Django management commands** (idiomatic, testable, schedulable):

- `python manage.py seed_catalog` — load commodity groups + country list.
- `python manage.py ingest_ffpi` — download FFPI Excel/CSV, parse with pandas, **upsert** into `price_index_monthly`.
- `python manage.py ingest_faostat_cpi` — download FAOSTAT Food CPI bulk, parse, upsert into `country_food_cpi_monthly`.

**Rules:**
- **Upsert, never append.** FAO *revises previous months*. Use `update_or_create` / a bulk upsert on the unique key so re-runs correct revised figures instead of duplicating rows. (This is the same class of trap as the Cric Metrics super-over field — the data shifts under you.)
- Idempotent: running twice produces identical DB state.
- Record provenance: update `data_source.last_ingested_at`.
- Validate on load: month is parseable, value within sane bounds (e.g. 20–300), no gaps silently dropped.

Libraries: `pandas`, `requests` (or `httpx`), `openpyxl` for Excel.

---

## 6. ML design

**Target:** global FFPI nominal, monthly. (Optionally repeat per sub-index.)

**Models — build all three and compare:**
1. **`seasonal_naive`** (baseline) — forecast = value 12 months ago. *This is the bar.* If the fancy model can't beat it, it adds no value — report that honestly.
2. **`sarima`** (statsmodels) — captures trend + 12-month seasonality, gives native confidence intervals.
3. **`lgbm`** (LightGBM, or XGBoost) — gradient boosting on engineered features.

**Feature engineering (for the boosting model):** lags (1, 2, 3, 6, 12), rolling mean & std (3, 6, 12), month-of-year (seasonality), linear time index (trend). Stretch: add an exogenous oil-price series (FAO notes energy linkage drives oil/sugar).

**Validation — non-negotiable:**
- **Walk-forward / expanding-window** cross-validation only. Never random K-fold on time series.
- **No look-ahead leakage:** compute every lag/rolling feature using only past data; never fit scalers or feature statistics on the full series. Write a test for this (see §9) — it's the subtle bug that silently invalidates results, analogous to your phase-denominator bug.
- Metrics: **MAE, RMSE, MAPE**. Store per run in `metrics_json`.
- Confidence intervals: SARIMA native; for LightGBM use quantile objective (q=0.05/0.95) or residual bootstrap → `yhat_lower`/`yhat_upper`.

**Persistence:** `joblib.dump` to `/models/{target}_{model}.pkl`; path in `forecast_run.artifact_path`.

---

## 7. Anomaly detection

Transparent and explainable (better than a black box for a portfolio piece you must defend):
- Fit the model, take residuals (observed − expected). Compute rolling z-score of residuals (or of month-over-month % change).
- Flag `|z| > 2.5` → write to `anomaly` with severity by magnitude.
- **Sanity check:** the March 2022 spike must be flagged `high`. If it isn't, the detector is wrong.

---

## 8. AI layer (Claude briefing)

Monthly, after a forecast run:
- Assemble exact figures: latest FFPI, MoM & YoY change per group, the forecast + interval, any anomalies.
- Call Claude (`anthropic` SDK) with a prompt that says: *use only the numbers provided; do not invent figures; output markdown.*
- Store in `briefing`; **cache by `data_hash`** so you don't re-call (and re-pay) for unchanged inputs.
- **Verify:** spot-check that the numbers in the rendered briefing match the DB — an LLM that fabricates a statistic is the food-price equivalent of a wrong NRR. Don't ship it unverified.

Cost note: this is pay-as-you-go USD API billing, but a single short monthly briefing is cents.

---

## 9. API (Django REST Framework)

| Method | Path | Returns |
|---|---|---|
| GET | `/api/index/?group=OVERALL&from=&to=` | monthly index series |
| GET | `/api/index/groups/` | list of commodity groups |
| GET | `/api/country-cpi/?iso3=IND&from=&to=` | country food CPI series |
| GET | `/api/forecast/?target=OVERALL&horizon=3` | latest forecast points + intervals + backtest metrics |
| GET | `/api/anomalies/?from=&to=` | flagged anomalies |
| GET | `/api/briefing/?period=latest` | latest briefing markdown |

Refresh is triggered by Celery beat, **not** a public endpoint.

---

## 10. Frontend

Django templates + **HTMX** (new to you; pairs cleanly with Django — dynamic updates without a JS framework) + a charting library (reuse **Plotly.js**, or learn **Apache ECharts**).

Dashboard layout: headline FFPI line chart with forecast band and anomaly markers; small-multiples of the five sub-indices; a country selector (HTMX swaps just the chart fragment on change); a briefing panel rendering `content_md`.

---

## 11. Build order (phased — always shippable)

| Phase | Deliverable | Definition of Done | Cumulative complexity |
|---|---|---|---|
| **0. Env** | Docker compose (web + Postgres + Redis), Django skeleton, `.env` | `docker compose up` serves Django, connects to Postgres | ~3 |
| **1. Schema** | All models + migrations + `seed_catalog` | Migrate clean; Django admin shows tables; groups + countries seeded | ~4.5 |
| **2. Ingestion** | `ingest_ffpi`, `ingest_faostat_cpi`, idempotent upsert | DB populated; **re-running doesn't duplicate** (revision handling verified); row counts sane | ~5.5 |
| **3. API + basic chart** | DRF endpoints + FFPI line chart | Chart renders real data; **latest value eyeballed against fao.org** (external validation, Cric-Metrics style) | ~6.5 |
| **4. ML** | naive + SARIMA + LightGBM, walk-forward backtest, forecast endpoint + band | Metrics table stored; ML **beats seasonal-naive on MAE** — or you report honestly why it doesn't | ~7.5 |
| **5. Anomalies** | residual z-score detection + chart markers | March 2022 spike flagged `high` | ~8 |
| **6. AI briefing** | Claude integration + panel + caching | Briefing renders; **its numbers match the DB** | ~8.5 |
| **7. Orchestration** | Celery beat monthly pipeline + HTMX country selector | Manual trigger runs end-to-end; scheduled task registered | ~8.5–9 |
| **8. Deploy** | Render web + managed Postgres + Redis | Live URL works; migrations + initial ingest run in prod | ~9 |

**To land exactly at 8/10:** Phases 0–6 get you there. Phase 7 (Celery) is the biggest single jump and the most resume-valuable; if it stalls you, ship with a manually-run `refresh_all` command (~7.5) and add Celery later.

---

## 12. Testing (pytest + pytest-django)

- **Ingestion idempotency:** ingest twice → identical row count; a revised value updates in place.
- **No leakage** (the critical ML test): feature matrix for month *t* contains no information from *t* or later.
- **Serializers:** API shapes and date filtering.
- **Anomaly threshold:** synthetic spike is flagged; flat series is not.
- **Briefing integrity:** numbers in output exist in the input payload.

---

## 13. Project structure

```
foodpriceintel/
├── manage.py
├── docker-compose.yml          # web, postgres, redis
├── Dockerfile
├── requirements.txt
├── .env.example
├── config/                     # Django project
│   ├── settings.py
│   ├── urls.py
│   ├── celery.py
│   └── wsgi.py / asgi.py
├── apps/
│   ├── catalog/                # commodity_group, country, data_source
│   ├── prices/                 # fact models + DRF
│   │   └── management/commands/
│   │       ├── seed_catalog.py
│   │       ├── ingest_ffpi.py
│   │       └── ingest_faostat_cpi.py
│   ├── forecasting/            # features.py, train.py, predict.py, tasks.py
│   ├── anomalies/              # detect.py
│   ├── briefings/              # claude_client.py, tasks.py
│   └── dashboard/              # templates/ + HTMX views
├── models/                     # joblib artifacts (gitignored)
└── tests/
```

---

## 14. Dependencies (`requirements.txt`)

```
Django
djangorestframework
psycopg[binary]
django-htmx
pandas
numpy
requests
openpyxl
statsmodels
lightgbm           # or xgboost
scikit-learn
joblib
celery
redis
anthropic
python-dotenv
gunicorn
pytest
pytest-django
```

New to you vs reused: **new** — Django, DRF, PostgreSQL, Django ORM, HTMX, statsmodels, LightGBM, joblib, Celery, Redis, Docker, pytest-django, anthropic SDK. **Reused as scaffolding** — pandas, Plotly.js (optional), Render, REST principles. Enough new to learn a lot; enough familiar that you're not learning everything cold.

---

## 15. Risks & gotchas (read before building)

- **FAO revises prior months** → upsert, don't append (§5).
- **Look-ahead leakage** → walk-forward only, no fitting on full series (§6, §12).
- **Beat the dumb baseline** → seasonal-naive is the bar, not SARIMA (§6).
- **Nominal vs real & base period** (2014–2016 = 100) → never mix the two series.
- **FAOSTAT bulk files are large** → ingest incrementally; don't load everything into memory at once.
- **Claude must not invent numbers** → pass exact figures, instruct "use only provided data," then verify (§8).
- **Windows/Docker** → run Postgres + Redis in Docker; keep the repo out of OneDrive.
- **API cost** → cache briefings by data hash.

---

## 16. Stretch goals (beyond 8/10)

- Exogenous oil-price feature; multi-series forecasting across all sub-indices.
- Backtest-comparison view: your model vs seasonal-naive vs SARIMA, side by side.
- Country food-inflation choropleth.
- Scenario panel: "if oil +20%, projected FFPI path."
```
