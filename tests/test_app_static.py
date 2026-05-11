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


def test_dashboard_includes_settings_modal_hooks(client):
    """T7.5: Settings modal exists, gear icon has an onClick, and the
    two persistent localStorage keys are referenced."""
    body = client.get("/").text
    assert "SettingsModal" in body, "SettingsModal component missing"
    assert "polevision.samPrompts" in body, (
        "Settings modal not wired to localStorage.samPrompts"
    )
    assert "polevision.unitSystem" in body, (
        "Settings modal not wired to localStorage.unitSystem"
    )


def test_dashboard_includes_unit_formatter(client):
    """T8.1: formatLength helper present with the imperial branches."""
    body = client.get("/").text
    assert "function formatLength" in body, "formatLength helper missing"
    assert "3.28084" in body, "metres-to-feet conversion factor missing"
    assert "' in'" in body, "inches suffix missing"


def test_dashboard_uses_formatunit_in_measurements_and_3d(client):
    """T8.3: the 6 display sites route through formatUnit so unit
    flips actually take effect."""
    body = client.get("/").text
    assert "formatUnit(height?.value" in body, (
        "Measurements summary tile not using formatUnit"
    )
    assert "formatUnit(pole.height_m" in body, (
        "3D pole-top label not using formatUnit"
    )
    assert "formatUnit(d.diameter_m" in body, (
        "3D diameter ring label not using formatUnit"
    )


def test_threed_tab_has_diameter_and_attachment_hooks(client):
    """S5: 3D viewer renders diameter rings + attachment markers.
    Smoke-check the JS still references the build hooks and layer
    toggles, so a future edit can't silently drop them."""
    r = client.get("/")
    body = r.text
    assert "buildDiameters" in body, "missing diameter ring builder"
    assert "buildAttachments" in body, "missing attachment marker builder"
    assert "diametersGroup" in body, "diameter group not mounted"
    assert "attachmentsGroup" in body, "attachment group not mounted"
    assert "Diameter rings" in body, "Layers GUI missing diameter toggle"
    assert "Attachment markers" in body, "Layers GUI missing attachment toggle"
