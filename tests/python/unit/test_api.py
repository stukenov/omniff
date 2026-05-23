import pytest


def test_create_app():
    try:
        from omniff.api import create_app

        app = create_app()
        assert app.title == "OmniFF API"
    except (ImportError, RuntimeError):
        pytest.skip("FastAPI or dependencies not installed")


def test_health_endpoint():
    try:
        from fastapi.testclient import TestClient

        from omniff.api import create_app

        client = TestClient(create_app())
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
    except (ImportError, RuntimeError):
        pytest.skip("FastAPI or dependencies not installed")


def test_routes_endpoint():
    try:
        from fastapi.testclient import TestClient

        from omniff.api import create_app

        client = TestClient(create_app())
        resp = client.get("/routes")
        assert resp.status_code == 200
        assert "TEXT_SIMPLE" in resp.json()["routes"]
    except (ImportError, RuntimeError):
        pytest.skip("FastAPI or dependencies not installed")
