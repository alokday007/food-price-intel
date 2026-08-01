"""
Django settings for the Food Price Intelligence project (config).

Everything that varies between environments is driven from environment variables
(loaded from a local .env via python-dotenv in dev; injected directly in prod).
No secrets are committed — see .env.example for the full variable list.

Production target (Phase 7a): Render web service building from the Dockerfile,
external Neon Postgres via DATABASE_URL, no Redis. The defaults here are
production-safe (DEBUG off, SECRET_KEY required) while the local .env keeps
docker-compose dev working exactly as before.
"""

from pathlib import Path

import dj_database_url
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# Load a local .env if present (no-op in containers that inject env directly).
load_dotenv(BASE_DIR / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean-ish environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# --- Core security / debug ---
# DEBUG defaults to False (production-safe); the local .env sets DEBUG=True.
DEBUG = env_bool("DEBUG", False)

# SECRET_KEY: a dev default is acceptable only when DEBUG is on. In production
# (DEBUG=False) an unset key is a hard error — fail loudly rather than ship a
# guessable key.
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    if DEBUG:
        SECRET_KEY = "django-insecure-dev-only-change-me"
    else:
        raise RuntimeError(
            "SECRET_KEY environment variable is required when DEBUG=False."
        )

ALLOWED_HOSTS = [
    h.strip()
    for h in os.getenv(
        "ALLOWED_HOSTS", "localhost,127.0.0.1,.onrender.com"
    ).split(",")
    if h.strip()
]

# --- Applications ---
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Project apps (live under the apps/ package — see each AppConfig.name).
    "apps.catalog",
    "apps.prices",
    "apps.forecasting",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # WhiteNoise must sit immediately after SecurityMiddleware so it can serve
    # static files (incl. admin CSS) with DEBUG=False, before other middleware.
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# --- Database ---
# In production Render/Neon inject a single DATABASE_URL; use it when present.
# Otherwise fall back to the discrete POSTGRES_* vars so docker-compose local dev
# is unchanged. dj_database_url parses either into Django's DATABASES dict.
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    DATABASES = {
        "default": dj_database_url.parse(DATABASE_URL, conn_max_age=600)
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("POSTGRES_DB", "foodpriceintel"),
            "USER": os.getenv("POSTGRES_USER", "foodpriceintel"),
            "PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
            "HOST": os.getenv("POSTGRES_HOST", "localhost"),
            "PORT": os.getenv("POSTGRES_PORT", "5432"),
        }
    }

# --- Cache ---
# Redis is dropped from production (nothing uses it yet), so the cache config is
# guarded: use RedisCache only when REDIS_URL is set (local docker-compose), and
# fall back to in-process LocMemCache otherwise. This keeps startup — and the
# cache round-trip in /healthz/ — working with no Redis present.
REDIS_URL = os.getenv("REDIS_URL")
if REDIS_URL:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": REDIS_URL,
        }
    }
else:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        }
    }

# --- Password validation ---
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# --- Internationalization ---
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# --- Static files (WhiteNoise) ---
# STATIC_ROOT is where collectstatic gathers files for WhiteNoise to serve in
# production. The compressed+manifest storage lets WhiteNoise serve hashed,
# far-future-cacheable assets with DEBUG=False.
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
