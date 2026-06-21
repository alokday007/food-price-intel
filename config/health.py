"""Health-check endpoint for Phase 0.

Actually exercises the two backing services: runs a trivial query against the
database and round-trips a value through the cache. Returns 200 only when both
succeed; otherwise 503 with the failing component named.
"""

from django.core.cache import cache
from django.db import connection
from django.http import JsonResponse


def _check_db() -> bool:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        return cursor.fetchone() == (1,)


def _check_cache() -> bool:
    cache.set("healthz", "ok", timeout=5)
    return cache.get("healthz") == "ok"


def healthz(request):
    """GET /healthz/ — report DB and cache liveness."""
    db_ok = False
    cache_ok = False

    try:
        db_ok = _check_db()
    except Exception:
        db_ok = False

    try:
        cache_ok = _check_cache()
    except Exception:
        cache_ok = False

    payload = {
        "status": "ok" if (db_ok and cache_ok) else "error",
        "db": "ok" if db_ok else "error",
        "cache": "ok" if cache_ok else "error",
    }
    status_code = 200 if (db_ok and cache_ok) else 503
    return JsonResponse(payload, status=status_code)
