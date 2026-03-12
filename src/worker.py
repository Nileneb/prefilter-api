"""
Buchungs-Anomalie Pre-Filter — Celery Worker v4.0

Task: analyze_task(job_id, filepath, config_dict)
  1. Liest Buchungsdatei ein
  2. Führt AnomalyEngine aus
  3. Publiziert Log-Events via Redis Pub/Sub → WebSocket
  4. Prüft nach jedem Test auf Abbruch-Signal
  5. Speichert Ergebnis in Redis Hash

Redis-Keys:
    job:{id}              → Hash (status, progress_pct, current_test, result, error)
    job:{id}:cancelled    → String "1" (Abbruch-Signal)
    job:{id}:logs         → Pub/Sub Channel (JSON-Events)
"""

from __future__ import annotations

import json
import os
import time

import redis as redis_sync
from celery import Celery

from src.config import AnalysisConfig
from src.parser import read_upload, map_columns
from src.engine import AnomalyEngine, NUM_TESTS

REDIS_URL         = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_BACKEND_URL = REDIS_URL.replace("/0", "/1").replace("/2", "/1")

celery_app = Celery(
    "prefilter",
    broker=REDIS_URL,
    backend=REDIS_BACKEND_URL,
)
celery_app.conf.update(
    task_serializer   = "json",
    result_serializer = "json",
    accept_content    = ["json"],
    task_track_started = True,
    worker_prefetch_multiplier = 1,
)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYZE TASK
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(bind=True, name="prefilter.analyze", max_retries=0)
def analyze_task(self, job_id: str, filepath: str, config_dict: dict) -> str:
    """Analyse-Task: läuft im Celery-Worker, völlig asynchron zum API-Server."""
    r = redis_sync.from_url(REDIS_URL, decode_responses=True)
    start_time  = time.time()
    test_counter = [0]

    # ── Helper-Funktionen ─────────────────────────────────────────────────────

    def _is_cancelled() -> bool:
        return bool(r.get(f"job:{job_id}:cancelled"))

    def _publish(msg: str, test: str = "", pct: float = 0.0, done: bool = False) -> None:
        event = json.dumps({
            "test":         test,
            "msg":          msg,
            "progress_pct": round(pct, 1),
            "done":         done,
        })
        r.publish(f"job:{job_id}:logs", event)

    def _set_status(status: str, pct: float = 0.0, test: str = "") -> None:
        r.hset(f"job:{job_id}", mapping={
            "status":       status,
            "progress_pct": str(round(pct, 1)),
            "current_test": test,
        })

    # ── Hauptlogik ────────────────────────────────────────────────────────────

    try:
        _set_status("running", 0.0)
        _publish("Lese Datei...", pct=0)

        # 1) Datei einlesen
        df = read_upload(filepath)
        df = map_columns(df)
        _publish(f"Datei gelesen: {len(df)} Buchungen", pct=2)

        # 2) Config validieren
        try:
            config = AnalysisConfig.model_validate(config_dict)
        except Exception:
            config = AnalysisConfig()

        # 3) Engine mit instrumentiertem Logging
        engine = AnomalyEngine(df, config=config, cancel_check=_is_cancelled)

        original_log = engine._log

        def instrumented_log(msg: str) -> None:
            original_log(msg)
            # Progress-Updates bei [NN/14] Test-Nachrichten
            if msg.startswith("[") and "/" in msg and "] " in msg:
                test_counter[0] += 1
                pct          = 5 + (test_counter[0] / NUM_TESTS) * 90
                current_test = msg.split("] ")[1].split(":")[0]
                _set_status("running", pct, current_test)
                _publish(msg, test=current_test, pct=pct)
            else:
                _publish(msg, pct=5)

        engine._log = instrumented_log

        # 4) Analyse ausführen
        result = engine.run()

        # 5) Ergebnis speichern
        elapsed = round(time.time() - start_time, 1)
        r.hset(f"job:{job_id}", mapping={
            "status":       "done",
            "progress_pct": "100",
            "current_test": "",
            "elapsed_s":    str(elapsed),
            "result":       json.dumps(result),
        })
        _publish(
            f"Analyse abgeschlossen in {elapsed}s — "
            f"{result['statistics']['total_output']} verdächtige Buchungen",
            pct=100,
            done=True,
        )
        return job_id

    except Exception as exc:
        r.hset(f"job:{job_id}", mapping={
            "status": "failed",
            "error":  str(exc),
        })
        _publish(f"Fehler: {exc}", done=True)
        raise

    finally:
        # Temporäre Datei löschen
        try:
            os.unlink(filepath)
        except OSError:
            pass
        r.close()
