from django.apps import AppConfig


class PricesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    # Full dotted path: the app lives in the apps/ package, so Django needs
    # the qualified name to find its models and migrations.
    name = "apps.prices"
