"""Root URL configuration for the Food Price Intelligence project."""

from django.contrib import admin
from django.urls import path

from apps.forecasting.views import forecast_overall_api, forecast_page
from config.health import healthz

urlpatterns = [
    path("admin/", admin.site.urls),
    path("healthz/", healthz, name="healthz"),
    path("api/forecast/overall/", forecast_overall_api, name="forecast_overall_api"),
    path("forecast/", forecast_page, name="forecast_page"),
]
