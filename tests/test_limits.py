"""Tests für Upload- und Webhook-Größenlimits (#2, #8)."""

import importlib

import pytest


def test_webhook_trims_rows_and_logs(monkeypatch):
    import src.webhook as webhook
    monkeypatch.setattr(webhook, "WEBHOOK_MAX_ROWS", 100)
    monkeypatch.setattr(webhook, "WEBHOOK_MAX_LOG_LINES", 50)

    payload = {
        "verdaechtige_buchungen": [{"belegnummer": str(i)} for i in range(250)],
        "logs": [f"line {i}" for i in range(200)],
        "message": "x",
    }
    trimmed = webhook._trim_payload(payload)
    assert len(trimmed["verdaechtige_buchungen"]) == 100
    assert trimmed["verdaechtige_buchungen_truncated"] == {"sent": 100, "total": 250}
    assert len(trimmed["logs"]) == 50
    assert trimmed["logs"][0] == "line 150"  # letzte 50
    # Original unverändert
    assert len(payload["verdaechtige_buchungen"]) == 250


def test_webhook_no_trim_when_small(monkeypatch):
    import src.webhook as webhook
    payload = {"verdaechtige_buchungen": [{"x": 1}], "logs": ["a"]}
    trimmed = webhook._trim_payload(payload)
    assert "verdaechtige_buchungen_truncated" not in trimmed
    assert len(trimmed["verdaechtige_buchungen"]) == 1


def test_upload_too_large_returns_413(monkeypatch):
    from fastapi.testclient import TestClient
    import src.main as main
    # Limit künstlich auf 1 KB senken
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1024)
    client = TestClient(main.app)

    big = b"x" * 4096
    resp = client.post(
        "/api/jobs",
        files={"file": ("big.csv", big, "text/csv")},
    )
    assert resp.status_code == 413
    assert "zu groß" in resp.json()["error"]
