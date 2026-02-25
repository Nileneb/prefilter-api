"""
Buchungs-Anomalie Pre-Filter — Webhook Client with retry.
"""

import time
import logging

import httpx

logger = logging.getLogger("prefilter")

MAX_RETRIES = 3
RETRY_DELAYS = [1, 3, 5]  # seconds between retries


def push_to_langdock(payload: dict, webhook_url: str) -> dict:
    """POST the prefilter result JSON to a Langdock webhook.

    Retries up to MAX_RETRIES times on transient errors (5xx, timeout).
    Logs response body on errors for debugging.
    """
    if not webhook_url:
        return {"error": "Keine Webhook-URL konfiguriert"}

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
                # Success or client error (4xx) — don't retry
                if r.status_code >= 400:
                    logger.warning(
                        f"Webhook HTTP {r.status_code} (attempt {attempt}): "
                        f"{r.text[:500]}"
                    )
                else:
                    logger.info(f"Webhook OK {r.status_code} (attempt {attempt})")
                return {"status": r.status_code, "response": r.text[:500]}

            # 5xx — retry
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

        # Wait before retry (except last attempt)
        if attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[attempt - 1]
            logger.info(f"Retry in {delay}s …")
            time.sleep(delay)

    return {"error": f"Fehlgeschlagen nach {MAX_RETRIES} Versuchen: {last_error}"}
