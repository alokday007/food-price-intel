"""Root URL configuration for the Food Price Intelligence project."""

from django.contrib import admin
from django.urls import path

from config.health import healthz

urlpatterns = [
    path("admin/", admin.site.urls),
    path("healthz/", healthz, name="healthz"),
]
