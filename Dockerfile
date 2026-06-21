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

# Dev command (used by docker-compose). For production, run gunicorn instead, e.g.:
#   gunicorn config.wsgi:application --bind 0.0.0.0:8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
