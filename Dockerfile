# Food Price Intelligence — Phase 0 image.
FROM python:3.13-slim

# Keep Python lean and unbuffered for predictable container logging.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY . .

# The executable bit is carried in git (update-index --chmod=+x); re-assert it
# here so the image works even if the build context loses file modes.
RUN chmod +x /app/start.sh

EXPOSE 8000

# Production launch (default image CMD, used by Render). Render's free tier has
# no Pre-Deploy Command, so start.sh runs the release steps (migrate, seed,
# collectstatic) at boot and then execs gunicorn on $PORT — see start.sh.
# Local docker-compose overrides this with the dev server (see the web service's
# `command:` in docker-compose.yml), so local dev is unchanged. To run plain
# gunicorn without the release steps, override the command:
#   docker run <image> sh -c 'gunicorn config.wsgi:application --bind 0.0.0.0:${PORT:-8000}'
CMD ["/app/start.sh"]
