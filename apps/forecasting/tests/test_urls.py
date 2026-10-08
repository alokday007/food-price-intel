"""Site routing: the bare domain lands on the chart."""

from __future__ import annotations


def test_root_redirects_to_forecast_page(client):
    resp = client.get("/")
    assert resp.status_code == 302  # temporary, so browsers don't cache it forever
    assert resp["Location"] == "/forecast/"


def test_root_follows_through_to_chart(client):
    """Following the redirect reaches the chart page."""
    resp = client.get("/", follow=True)
    assert resp.status_code == 200
    assert resp.redirect_chain == [("/forecast/", 302)]
