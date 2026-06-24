from django.apps import AppConfig


class CatalogConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    # Must be the full dotted path because the app lives in the apps/ package;
    # otherwise Django won't detect its migrations.
    name = "apps.catalog"
