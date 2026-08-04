#!/bin/sh
# Production container entrypoint (Phase 7a-fix).
#
# Render's FREE tier has no Pre-Deploy Command, so the release steps that would
# normally run before the container starts have to run at boot instead. All three
# are safe to repeat on every restart:
#   - migrate       : no-op once the schema is current
#   - seed_catalog  : update_or_create on natural keys (dimension rows)
#   - seed_from_csv : update_or_create on (commodity_group, period) — 0 rows created
#                     on a second run, and it *requires* seed_catalog to have run
#                     (it looks up CommodityGroup 'OVERALL' + DataSource 'FAO FFPI')
#   - collectstatic : gathers assets into STATIC_ROOT for WhiteNoise (DEBUG=False)
#
# Local docker-compose is unaffected: the web service overrides the image command
# with the Django dev server (see its `command:` in docker-compose.yml).
set -e

echo "==> migrate"
python manage.py migrate --noinput

echo "==> seed_catalog"
python manage.py seed_catalog

echo "==> seed_from_csv"
python manage.py seed_from_csv

echo "==> collectstatic"
python manage.py collectstatic --noinput

# exec so gunicorn replaces this shell as PID 1 and receives SIGTERM/SIGINT
# directly (clean shutdown on Render redeploys). $PORT is injected by Render;
# default to 8000 so the script is runnable locally too.
echo "==> gunicorn"
exec gunicorn config.wsgi:application --bind "0.0.0.0:${PORT:-8000}"
