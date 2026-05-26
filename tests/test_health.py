"""Test für den /health-Endpoint mit Redis-Status (#16)."""

from fastapi.testclient import TestClient

from src import __version__
import src.main as main


def test_health_reports_version_and_redis_status():
    client = TestClient(main.app)
    resp = client.get("/health")
    # 200 wenn Redis erreichbar, sonst 503 — beides valide
    assert resp.status_code in (200, 503)
    body = resp.json()
    assert body["version"] == __version__
    assert "redis" in body
    assert body["status"] in ("ok", "degraded")
