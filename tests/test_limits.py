"""Tests für das Webhook-Payload-Limit (#8). Upload hat bewusst KEIN Limit."""


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
