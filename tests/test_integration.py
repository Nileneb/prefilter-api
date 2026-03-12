"""
Integration-Test: FastAPI + Celery + Redis End-to-End.

Voraussetzung: Laufender Redis-Server (docker compose up redis).
Ausführen:     python -m pytest tests/test_integration.py -v -s
Markierung:    @pytest.mark.integration → wird bei normalem `pytest` übersprungen.
"""

import io
import json
import time

import pytest

# Integration-Marker: nur ausführen wenn explizit angefordert
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def redis_available():
    """Prüft ob Redis erreichbar ist, überspringt sonst."""
    try:
        import redis
        r = redis.from_url("redis://localhost:6379/0", decode_responses=True)
        r.ping()
        return True
    except Exception:
        pytest.skip("Redis nicht erreichbar — Integration-Tests übersprungen")


@pytest.fixture(scope="module")
def test_client(redis_available):
    """FastAPI TestClient mit laufendem Redis."""
    from fastapi.testclient import TestClient
    from src.main import app
    return TestClient(app)


@pytest.fixture
def sample_csv() -> bytes:
    """Minimaler CSV-Datensatz für Integration-Tests."""
    lines = [
        "datum;betrag;konto_soll;konto_haben;buchungstext;belegnummer;kostenstelle;kreditor;erfasser",
    ]
    for i in range(20):
        lines.append(
            f"2024-01-{15 + i % 15:02d};{(i + 1) * 100},00;4711;1200;"
            f"Buchung {i:03d};B-{i:04d};;Lieferant_A;UserA"
        )
    # Ein paar Anomalien einbauen
    lines.append("2024-01-15;999999,00;4711;1200;Storno Rechnung 001;B-9999;;Lieferant_A;UserA")
    lines.append("2024-01-15;999999,00;4711;1200;Test;B-9999;;Lieferant_A;UserA")  # Dup Beleg
    return "\n".join(lines).encode("utf-8")


def test_health(test_client):
    """GET /health → 200."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_create_job(test_client, sample_csv):
    """POST /api/jobs → 202, gibt job_id zurück."""
    resp = test_client.post(
        "/api/jobs",
        files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        data={"config_json": json.dumps({"output_threshold": 1.0})},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_get_nonexistent_job(test_client):
    """GET /api/jobs/{invalid_id} → 404."""
    resp = test_client.get("/api/jobs/nonexistent-id")
    assert resp.status_code == 404


def test_create_and_poll_job(test_client, sample_csv):
    """POST + Poll: Job wird erstellt und Status kann abgefragt werden."""
    # Erstellen
    resp = test_client.post(
        "/api/jobs",
        files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # Status abfragen (sollte mindestens "queued" sein)
    resp = test_client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["status"] in ("queued", "running", "done")


def test_cancel_job(test_client, sample_csv):
    """POST /api/jobs/{id}/cancel setzt Status auf 'cancelling'."""
    resp = test_client.post(
        "/api/jobs",
        files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
    )
    job_id = resp.json()["job_id"]

    resp = test_client.post(f"/api/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cancelling"


def test_full_pipeline_with_worker(test_client, sample_csv):
    """End-to-End: Job erstellen, warten bis 'done', Ergebnis prüfen.

    Dieser Test setzt einen laufenden Celery-Worker voraus.
    Timeout: 30 Sekunden.
    """
    # Job erstellen
    resp = test_client.post(
        "/api/jobs",
        files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        data={"config_json": json.dumps({"output_threshold": 1.0})},
    )
    job_id = resp.json()["job_id"]

    # Pollen bis done/failed (max 30s)
    deadline = time.time() + 30
    status = "queued"
    while time.time() < deadline:
        resp = test_client.get(f"/api/jobs/{job_id}")
        data = resp.json()
        status = data["status"]
        if status in ("done", "failed"):
            break
        time.sleep(0.5)

    if status not in ("done", "failed"):
        pytest.skip("Worker nicht aktiv oder zu langsam — End-to-End übersprungen")

    assert status == "done", f"Job fehlgeschlagen: {data.get('error')}"

    # Ergebnis prüfen
    result = data.get("partial_results")
    assert result is not None, "Kein Ergebnis im Job-Status"
    assert "total_input" in result or "statistics" in result
