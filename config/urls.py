"""Root URL configuration for the Food Price Intelligence project."""

from django.contrib import admin
from django.urls import path
from django.views.generic import RedirectView

from apps.forecasting.views import forecast_api, forecast_page
from config.health import healthz

urlpatterns = [
    # The chart is the site; send the bare domain there (302, not permanent).
    path("", RedirectView.as_view(pattern_name="forecast_page", permanent=False), name="root"),
    path("admin/", admin.site.urls),
    path("healthz/", healthz, name="healthz"),
    # The original Phase 6 endpoint, kept as a named route onto the generic view.
    path(
        "api/forecast/overall/",
        forecast_api,
        {"code": "overall"},
        name="forecast_overall_api",
    ),
    path("api/forecast/<str:code>/", forecast_api, name="forecast_api"),
    path("forecast/", forecast_page, name="forecast_page"),
]
