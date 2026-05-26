"""
Buchungs-Anomalie Pre-Filter — Webhook Client with retry.
"""

import json
import os
import time

import httpx

from src.logging_config import get_logger

logger = get_logger("prefilter.webhook")

MAX_RETRIES = 3
RETRY_DELAYS = [1, 3, 5]  # seconds between retries

# WHY(#8): 1000 Buchungen × ~500 B + Logs können >1 MB werden → Langdock-Limit /
# Timeout. Payload vor dem Senden kappen (konfigurierbar via ENV).
WEBHOOK_MAX_ROWS = int(os.environ.get("WEBHOOK_MAX_ROWS", "100"))
WEBHOOK_MAX_LOG_LINES = int(os.environ.get("WEBHOOK_MAX_LOG_LINES", "50"))


def _trim_payload(payload: dict) -> dict:
    """Kürzt Buchungen + Logs auf konfigurierte Limits (non-destruktiv: Kopie)."""
    trimmed = dict(payload)
    rows = payload.get("verdaechtige_buchungen")
    if isinstance(rows, list) and WEBHOOK_MAX_ROWS > 0 and len(rows) > WEBHOOK_MAX_ROWS:
        trimmed["verdaechtige_buchungen"] = rows[:WEBHOOK_MAX_ROWS]
        trimmed["verdaechtige_buchungen_truncated"] = {
            "sent": WEBHOOK_MAX_ROWS, "total": len(rows),
        }
    logs = payload.get("logs")
    if isinstance(logs, list) and WEBHOOK_MAX_LOG_LINES > 0 and len(logs) > WEBHOOK_MAX_LOG_LINES:
        trimmed["logs"] = logs[-WEBHOOK_MAX_LOG_LINES:]
    return trimmed


def push_to_langdock(payload: dict, webhook_url: str) -> dict:
    """POST the prefilter result JSON to a Langdock webhook.

    Retries up to MAX_RETRIES times on transient errors (5xx, timeout).
    Logs response body on errors for debugging. Payload wird vorher gekürzt (#8).
    """
    if not webhook_url:
        return {"error": "Keine Webhook-URL konfiguriert"}

    payload = _trim_payload(payload)
    payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    logger.info("Webhook-Payload", size_kb=round(payload_bytes / 1024, 1),
                rows=len(payload.get("verdaechtige_buchungen", [])))

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = httpx.post(
                webhook_url,
                json=payload,
                timeout=30.0,
                headers={"Content-Type": "application/json"},
            )
            if r.status_code < 500:
                if r.status_code >= 400:
                    logger.warning(
                        f"Webhook HTTP {r.status_code} (attempt {attempt}): "
                        f"{r.text[:500]}"
                    )
                else:
                    logger.info(f"Webhook OK {r.status_code} (attempt {attempt})")
                return {"status": r.status_code, "response": r.text[:500]}

            last_error = f"HTTP {r.status_code}: {r.text[:300]}"
            logger.warning(
                f"Webhook 5xx (attempt {attempt}/{MAX_RETRIES}): {last_error}"
            )
        except httpx.TimeoutException as e:
            last_error = f"Timeout: {e}"
            logger.warning(
                f"Webhook timeout (attempt {attempt}/{MAX_RETRIES}): {last_error}"
            )
        except Exception as e:
            last_error = str(e)
            logger.error(
                f"Webhook error (attempt {attempt}/{MAX_RETRIES}): {last_error}"
            )

        if attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[attempt - 1]
            logger.info(f"Retry in {delay}s …")
            time.sleep(delay)

    return {"error": f"Fehlgeschlagen nach {MAX_RETRIES} Versuchen: {last_error}"}
