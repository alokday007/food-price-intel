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

EXPOSE 8000

# Production launch (default image CMD, used by Render). Bind to $PORT, which
# Render injects; default to 8000 when unset. migrate / collectstatic / seed are
# run as separate release steps, not baked into the CMD.
# Local docker-compose overrides this with the dev server (see the web service's
# `command:` in docker-compose.yml), so local dev is unchanged.
CMD ["sh", "-c", "gunicorn config.wsgi:application --bind 0.0.0.0:${PORT:-8000}"]
