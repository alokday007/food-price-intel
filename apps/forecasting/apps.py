from django.apps import AppConfig


class ForecastingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    # Full dotted path: the app lives in the apps/ package, so Django needs the
    # qualified name to discover its management commands.
    name = "apps.forecasting"
