"""Static-frontend tests: the dashboard HTML loads at / and contains
the React-mount root and the brand title from the design."""

import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("POLEVISION_DB", str(tmp_path / "polevision.db"))
    monkeypatch.setenv("POLEVISION_WORKSPACE", str(tmp_path))
    import importlib
    import app.main as main
    importlib.reload(main)
    return TestClient(main.app)


def test_root_returns_dashboard_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    # Sanity-check: the dashboard's title and React mount marker.
    assert "PoleVision" in body, "Brand name missing from dashboard HTML"
    assert 'id="root"' in body or "root" in body, (
        "Expected a React root element in the dashboard HTML"
    )


def test_static_assets_served_at_static_prefix(client):
    """If we add JS/CSS files later they live under /static/. Hitting
    a known existing static asset should return 200; hitting a missing
    one should return 404."""
    r = client.get("/static/index.html")
    assert r.status_code == 200
    assert "PoleVision" in r.text

    r404 = client.get("/static/does-not-exist.css")
    assert r404.status_code == 404


def test_threed_tab_includes_three_js_cdn(client):
    """The 3D tab is rendered with three.js loaded via CDN. Confirm
    the script tag is present in the HTML so we don't accidentally
    drop the import while editing the file."""
    r = client.get("/")
    assert "three@0.146" in r.text or "three@" in r.text, (
        "Expected three.js CDN script tag in dashboard HTML"
    )
    assert "PLYLoader" in r.text, "Expected PLYLoader script tag"
    assert "OrbitControls" in r.text, "Expected OrbitControls script tag"
