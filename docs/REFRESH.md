# Monthly refresh runbook

FAO publishes a new FFPI release early each month (one new month, plus revisions to
recent months). This runbook takes a new release from download to the live site.
Commands are **PowerShell** from the repo root, with the local stack running
(`docker compose up -d`). Do the steps **in order** — later steps assume earlier
checks passed. If an expected number doesn't match, stop and find out why.

## Row-count arithmetic (know these before you start)

Every month adds exactly **one row per series**, and there are **6 series**
(OVERALL, CEREALS, OILS, DAIRY, MEAT, SUGAR):

- **+1 month = +6 rows** in `PriceIndexMonthly` (and in `ffpi_observed.csv`).
- Rows per series = months from 1990-01 through the latest month
  = `(year − 1990) × 12 + month`. Total = 6 × that.

| data through | rows per series | total rows |
|---|---|---|
| 2026-09 | 441 | 2646 |
| 2026-10 | 442 | 2652 |
| 2026-11 | 443 | 2658 |

FAO's revisions to *existing* months change values, never the row count.

---

## 1. Download the FAO workbook

Download the Food Price Index data file from the
[FAO Food Price Index page](https://www.fao.org/worldfoodsituation/foodpricesindex/en/)
into `data/raw/ffpi/`, named by release month, e.g. `ffpi-data-2026-11.xlsx`.
`data/raw/` is gitignored — the workbook is never committed. Keep older workbooks.

## 2. Ingest

```powershell
docker compose exec -T web python manage.py ingest_ffpi --file data/raw/ffpi/ffpi-data-2026-11.xlsx
```

Check the output: `records parsed` = the new total (e.g. 2652), `created` = **6**
for one new month, `updated` = the previous total. `created` ≠ 6 × (new months) → stop.

## 3. Scoreboard: evaluate each series, then summarize

One series per command — all six in one command takes ~10+ minutes and can hit
tool timeouts. Each takes ~2–3 minutes. The JSON is written *inside* the container
so PowerShell never re-encodes it.

```powershell
$groups = 'OVERALL','CEREALS','OILS','DAIRY','MEAT','SUGAR'
docker compose exec -T web sh -c 'mkdir -p /tmp/eval && rm -f /tmp/eval/*.json'
foreach ($g in $groups) {
  docker compose exec -T web sh -c "python manage.py evaluate_sarima --group $g --json > /tmp/eval/$g.json"
}
docker compose exec -T web sh -c 'cat /tmp/eval/OVERALL.json /tmp/eval/CEREALS.json /tmp/eval/OILS.json /tmp/eval/DAIRY.json /tmp/eval/MEAT.json /tmp/eval/SUGAR.json | python manage.py evaluate_sarima --summarize --out data/seed/scoreboard.json'
```

The summary prints the table and `TOTAL … BEATS n / TIES n / LOSES n` — counted
in code. `--summarize` refuses mixed eval-end months or duplicate series. Every
series should show the new month as `last` and one more fold than last time.

Then regenerate the README table from the file (never edit it by hand):

```powershell
docker compose exec -T web python manage.py render_scoreboard
```

The README's "Models evaluated" section is intentionally **not** refreshed: it renders from the one-off `data/eval/lgbm_9a.json` and `data/eval/lgbm_9a1.json` (both eval_end 2026-09), so the command leaves it unchanged — don't re-run `evaluate_lgbm` or `evaluate_lgbm_9a1`.

Optional harness sanity check — reproduces the recorded Phase 3 baseline on the
old window, and asserts it:

```powershell
docker compose exec -T web python manage.py evaluate_sarima --group OVERALL --eval-end 2026-05
```

If this fails after an ingest, FAO revised a month at or before 2026-05: find
which month moved; don't loosen the assertion.

## 4. Generate forecasts locally

```powershell
foreach ($g in $groups) {
  docker compose exec -T web python manage.py generate_forecast --group $g --horizon 3
}
```

Each run prints `train_end` = the new month and 3 points for the following
months, and `PriceIndexMonthly.count(): … UNCHANGED`. The local `ForecastRun`
count goes up by exactly **6** (one per series).

## 5. Export the deploy seed CSV

```powershell
docker compose exec -T web python manage.py export_observed_csv
git diff --stat data/seed/ffpi_observed.csv
```

The committed CSV is your before-snapshot, so the diff *is* the ingest report:
**6 added lines per new month**, plus a changed line for every row FAO revised
(nominal or real value). Read the revision lines — a large revision deep in
history is worth a second look. The file has total rows + 1 header line.

## 6. Run the tests

```powershell
docker compose exec -T web python -m pytest -q
```

All must pass. `test_readme_scoreboard` fails if you skipped `render_scoreboard`.

## 7. Commit and push

```powershell
git add data/seed/ffpi_observed.csv data/seed/scoreboard.json README.md
git status            # no .xlsx, no .env
git commit -m "Refresh FFPI through 2026-11"
git push
```

Render redeploys on push. On boot, `start.sh` runs migrate → seed_catalog →
**seed_from_csv** (upserts the new CSV into Neon) → collectstatic → gunicorn.
Wait for the deploy to go live, then check the live data is new:

- https://food-price-intel.onrender.com/api/forecast/overall/ — last `history`
  period is the new month; `scoreboard.eval_end` is the new month.

The forecast lines on the live site are still last month's until step 8.

## 8. Generate forecasts on Neon

Copy the Neon connection string to the clipboard (Neon console → Connection
details), then:

```powershell
$env:NEON_URL = (Get-Clipboard -Raw).Trim()
# Guard: an empty DATABASE_URL silently falls back to the LOCAL database.
if (-not $env:NEON_URL -or $env:NEON_URL -notmatch 'neon\.tech') { throw "NEON_URL is not a Neon URL" }
```

**Check you're on Neon by run count** — before writing anything:

```powershell
docker compose exec -T -e "DATABASE_URL=$env:NEON_URL" web python manage.py shell -c "from django.db import connection; from apps.prices.models import PriceIndexMonthly as P; from apps.forecasting.models import ForecastRun as R; print(connection.settings_dict['HOST'], P.objects.count(), R.objects.count())"
```

- The host must be the Neon host (`…neon.tech`), not `db`/`localhost`.
- `PriceIndexMonthly` must equal the local total from step 2 (the deploy seeded it).
- Note the `ForecastRun` count: it is Neon's own history and won't match the
  local count. **If it equals your local count, assume you are on the local DB
  and stop.**

Then forecast every series against Neon:

```powershell
foreach ($g in $groups) {
  docker compose exec -T -e "DATABASE_URL=$env:NEON_URL" web python manage.py generate_forecast --group $g --horizon 3
}
```

Verify: re-run the count command above — Neon's `ForecastRun` count is exactly
**+6** from what you noted, and `PriceIndexMonthly` is unchanged. The predicted
values should match your local step-4 runs exactly (same data, same model).

Clear the secret from the session:

```powershell
Remove-Item Env:NEON_URL
```

## 9. Final live check

Open https://food-price-intel.onrender.com/forecast/ (and `?series=sugar` etc.):
the dashed SARIMA and dotted no-change lines start from the new month, and the
verdict text says "evaluated through" the new month.
