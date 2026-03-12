"""
Buchungs-Anomalie Pre-Filter — Celery Worker v4.2

Tasks:
  prefilter.analyze       — Hauptaufgabe: sequentiell ODER parallel (chord)
  prefilter.run_test      — Phase 2: Einzelnen Test auf vollem DataFrame ausführen
  prefilter.merge         — Phase 3: Flags zusammenführen, Scores berechnen, Export

Parallele Pipeline (>= PARALLEL_THRESHOLD Zeilen):
  analyze_task (Phase 1 inline) → chord(14× run_test_task) → merge_task

Redis-Keys:
    job:{id}              → Hash (status, progress_pct, current_test, result, error)
    job:{id}:cancelled    → String "1" (Abbruch-Signal)
    job:{id}:logs         → Pub/Sub Channel (JSON-Events)
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import redis as redis_sync
from celery import Celery, chord, group

from src.config import AnalysisConfig
from src.logging_config import setup_logging, get_logger
from src.parser import read_upload, map_columns
from src.engine import AnomalyEngine, NUM_TESTS, _ALL_TESTS, _TEST_BY_NAME
from src.tests.base import EngineStats

setup_logging()
logger = get_logger("prefilter.worker")

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
    # Große Datensätze: Task erst nach Abschluss aus Queue entfernen
    # → bei Worker-Crash wird der Task erneut verteilt
    task_acks_late    = True,
    task_reject_on_worker_lost = True,
    # Soft/Hard Time-Limits: 10min soft, 15min hard (für 500k+ Zeilen)
    task_soft_time_limit = 600,
    task_time_limit      = 900,
)

PARALLEL_THRESHOLD = int(os.environ.get("PARALLEL_THRESHOLD", "100000"))


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _redis_client() -> redis_sync.Redis:
    return redis_sync.from_url(REDIS_URL, decode_responses=True)

def _publish(r, job_id: str, msg: str, test: str = "", pct: float = 0.0, done: bool = False) -> None:
    event = json.dumps({
        "test":         test,
        "msg":          msg,
        "progress_pct": round(pct, 1),
        "done":         done,
    })
    r.publish(f"job:{job_id}:logs", event)

def _set_status(r, job_id: str, status: str, pct: float = 0.0, test: str = "") -> None:
    r.hset(f"job:{job_id}", mapping={
        "status":       status,
        "progress_pct": str(round(pct, 1)),
        "current_test": test,
    })

def _is_cancelled(r, job_id: str) -> bool:
    return bool(r.get(f"job:{job_id}:cancelled"))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: RUN SINGLE TEST (parallel pipeline — one per test)
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(bind=True, name="prefilter.run_test", max_retries=0)
def run_test_task(self, prepare_result: dict, test_name: str) -> dict:
    """Phase 2: Einen einzelnen Test auf dem vollen DataFrame ausführen."""
    r = _redis_client()
    try:
        job_id = prepare_result["job_id"]

        if _is_cancelled(r, job_id):
            return {"test_name": test_name, "flagged": [], "count": 0}

        df = pd.read_parquet(prepare_result["parquet_path"])
        stats = EngineStats.from_dict(prepare_result["stats_dict"])

        try:
            config = AnalysisConfig.model_validate(prepare_result["config_dict"])
        except Exception:
            config = AnalysisConfig()

        engine = AnomalyEngine.__new__(AnomalyEngine)
        engine.df = df
        engine.config = config
        engine.cancel_check = None
        engine.logs = []
        engine.flag_counts = {}
        engine.stammdaten_report = {"fuzzy_kreditor_matches": []}

        flagged = engine.run_single_test(test_name, stats)
        count = engine.flag_counts.get(test_name, 0)

        # Progress-Update: jeder abgeschlossene Test = 80/14 ≈ 5.7% innerhalb Phase 2
        _publish(r, job_id, f"Test {test_name}: {count} Treffer", test=test_name)

        return {"test_name": test_name, "flagged": flagged, "count": count}
    finally:
        r.close()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: MERGE TASK (parallel pipeline — callback)
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(bind=True, name="prefilter.merge", max_retries=0)
def merge_task(self, test_results: list[dict], prepare_result: dict) -> str:
    """Phase 3: Flags zusammenführen, Scores berechnen, Export."""
    r = _redis_client()
    job_id = prepare_result["job_id"]
    start_time = float(r.hget(f"job:{job_id}", "started_at") or time.time())

    try:
        _set_status(r, job_id, "running", 90.0, "merge")
        _publish(r, job_id, "Ergebnisse zusammenführen...", pct=90)

        df = pd.read_parquet(prepare_result["parquet_path"])

        try:
            config = AnalysisConfig.model_validate(prepare_result["config_dict"])
        except Exception:
            config = AnalysisConfig()

        engine = AnomalyEngine.__new__(AnomalyEngine)
        engine.df = df
        engine.config = config
        engine.cancel_check = None
        engine.logs = []
        engine.flag_counts = {}
        engine.stammdaten_report = {"fuzzy_kreditor_matches": []}

        result = engine.apply_flags_and_export(test_results)

        elapsed = round(time.time() - start_time, 1)
        r.hset(f"job:{job_id}", mapping={
            "status":       "done",
            "progress_pct": "100",
            "current_test": "",
            "elapsed_s":    str(elapsed),
            "result":       json.dumps(result),
        })
        _publish(
            r, job_id,
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
        _publish(r, job_id, f"Fehler: {exc}", done=True)
        raise

    finally:
        # Cleanup: Parquet + Original-Datei löschen
        for path in (prepare_result.get("parquet_path"), prepare_result.get("filepath")):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        r.close()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYZE TASK (sequential — original flow, also entry point for parallel)
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(bind=True, name="prefilter.analyze", max_retries=0)
def analyze_task(self, job_id: str, filepath: str, config_dict: dict) -> str:
    """Analyse-Task: entscheidet ob sequentiell oder parallel."""
    r = _redis_client()
    start_time = time.time()
    n_rows = 0  # Default: sequentieller Pfad (Datei löschen in finally)

    try:
        _set_status(r, job_id, "running", 0.0)
        _publish(r, job_id, "Lese Datei...", pct=0)

        # 1) Datei einlesen um Zeilenzahl zu prüfen
        df = read_upload(filepath)
        df = map_columns(df)
        n_rows = len(df)
        _publish(r, job_id, f"Datei gelesen: {n_rows} Buchungen", pct=2)

        # 2) Entscheidung: parallel oder sequentiell
        if n_rows >= PARALLEL_THRESHOLD:
            _publish(r, job_id, f"Parallele Pipeline (>={PARALLEL_THRESHOLD} Zeilen)", pct=3)
            r.hset(f"job:{job_id}", "started_at", str(start_time))

            # Phase 1 inline: vorbereiten + Parquet speichern
            try:
                config = AnalysisConfig.model_validate(config_dict)
            except Exception:
                config = AnalysisConfig()

            engine = AnomalyEngine(df, config=config)
            stats = engine.compute_stats()
            stats_dict = stats.to_dict()

            parquet_path = os.path.join(
                os.path.dirname(filepath) or "/data/uploads",
                f"job_{job_id}_prepared.parquet",
            )
            engine.df.to_parquet(parquet_path, index=True)

            _set_status(r, job_id, "running", 10.0, "prepare_done")
            _publish(r, job_id, "Vorbereitung abgeschlossen, starte Tests...", pct=10)

            prepare_result = {
                "parquet_path": parquet_path,
                "config_dict":  config_dict,
                "stats_dict":   stats_dict,
                "n_rows":       n_rows,
                "job_id":       job_id,
                "filepath":     filepath,
            }

            r.close()

            # chord: group(14 tests) → merge
            test_group = group(
                run_test_task.s(prepare_result, t.name) for t in _ALL_TESTS
            )
            callback = merge_task.s(prepare_result)
            chord(test_group)(callback)
            return job_id

        # 3) Sequentieller Pfad (kleine Dateien)
        _run_sequential(r, job_id, df, config_dict, start_time)
        return job_id

    except Exception as exc:
        r.hset(f"job:{job_id}", mapping={
            "status": "failed",
            "error":  str(exc),
        })
        _publish(r, job_id, f"Fehler: {exc}", done=True)
        raise

    finally:
        # Quelldatei nur im sequentiellen Pfad löschen; parallel macht merge_task
        if n_rows < PARALLEL_THRESHOLD:
            try:
                os.unlink(filepath)
            except OSError:
                pass
        r.close()


def _run_sequential(r, job_id: str, df, config_dict: dict, start_time: float) -> None:
    """Sequentieller Analyse-Pfad (< PARALLEL_THRESHOLD Zeilen)."""
    test_counter = [0]

    try:
        config = AnalysisConfig.model_validate(config_dict)
    except Exception:
        config = AnalysisConfig()

    engine = AnomalyEngine(df, config=config, cancel_check=lambda: _is_cancelled(r, job_id))

    original_log = engine._log

    def instrumented_log(msg: str) -> None:
        original_log(msg)
        if msg.startswith("[") and "/" in msg and "] " in msg:
            test_counter[0] += 1
            pct          = 5 + (test_counter[0] / NUM_TESTS) * 90
            current_test = msg.split("] ")[1].split(":")[0]
            _set_status(r, job_id, "running", pct, current_test)
            _publish(r, job_id, msg, test=current_test, pct=pct)
        else:
            _publish(r, job_id, msg, pct=5)

    engine._log = instrumented_log

    result = engine.run()

    elapsed = round(time.time() - start_time, 1)
    r.hset(f"job:{job_id}", mapping={
        "status":       "done",
        "progress_pct": "100",
        "current_test": "",
        "elapsed_s":    str(elapsed),
        "result":       json.dumps(result),
    })
    _publish(
        r, job_id,
        f"Analyse abgeschlossen in {elapsed}s — "
        f"{result['statistics']['total_output']} verdächtige Buchungen",
        pct=100,
        done=True,
    )
