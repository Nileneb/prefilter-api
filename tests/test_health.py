"""Tests für /health (Liveness) und /healthz (Readiness, Redis) (#16)."""

from fastapi.testclient import TestClient

import src.main as main
from src import __version__


def test_health_liveness_always_200():
    client = TestClient(main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["version"] == __version__


def test_healthz_reports_redis_status():
    client = TestClient(main.app)
    resp = client.get("/healthz")
    # 200 wenn Redis erreichbar, sonst 503 — beides valide
    assert resp.status_code in (200, 503)
    body = resp.json()
    assert body["version"] == __version__
    assert "redis" in body
    assert body["status"] in ("ok", "degraded")
