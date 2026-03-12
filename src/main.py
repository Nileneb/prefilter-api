"""
Buchungs-Anomalie Pre-Filter — FastAPI Backend v4.0

Endpoints:
    POST   /api/jobs                  — Job anlegen, Datei hochladen
    GET    /api/jobs/{id}             — Job-Status abfragen
    POST   /api/jobs/{id}/cancel      — Job abbrechen (Stop-Button)
    WS     /ws/jobs/{id}             — Realtime Log-Stream via WebSocket

Gradio-UI wird unter /ui gemountet (Übergangsphase).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import redis.asyncio as aioredis

from src.models import JobResponse, JobStatusResponse

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
JOB_TTL   = int(os.environ.get("JOB_TTL_SECONDS", "3600"))   # 1 Stunde

app = FastAPI(
    title="Buchungs-Anomalie Pre-Filter API",
    version="4.0.0",
    description="14 statistische Anomalie-Tests für Buchungsdaten",
)


# ── Helper ────────────────────────────────────────────────────────────────────
def _redis() -> aioredis.Redis:
    return aioredis.from_url(REDIS_URL, decode_responses=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Liveness check."""
    return {"status": "ok", "version": "4.0.0"}


@app.post("/api/jobs", response_model=JobResponse, status_code=202)
async def create_job(
    file: UploadFile = File(..., description="CSV, XLS oder XLSX Buchungsdatei"),
    config_json: str = Form(default="{}", description="JSON-kodiertes AnalysisConfig"),
):
    """Neuen Analyse-Job anlegen. Gibt Job-ID zurück; Analyse läuft asynchron."""
    job_id = str(uuid.uuid4())

    # Datei in temporäres Verzeichnis schreiben
    suffix = os.path.splitext(file.filename or "upload")[1].lower() or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        content = await file.read()
        f.write(content)
        filepath = f.name

    # Config parsen (default: leeres Dict → AnalysisConfig-Defaults)
    try:
        config_dict = json.loads(config_json)
    except Exception:
        config_dict = {}

    # Job-Status in Redis anlegen
    r = _redis()
    await r.hset(f"job:{job_id}", mapping={
        "status":       "queued",
        "progress_pct": "0",
        "current_test": "",
        "started_at":   str(time.time()),
        "filename":     file.filename or "",
    })
    await r.expire(f"job:{job_id}", JOB_TTL)

    # Celery-Task enqueuen (lazy import verhindert Import-Fehler ohne Redis)
    from src.worker import analyze_task
    analyze_task.delay(job_id, filepath, config_dict)

    return JobResponse(job_id=job_id, status="queued")


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    """Job-Status abfragen."""
    r = _redis()
    data = await r.hgetall(f"job:{job_id}")
    if not data:
        return JSONResponse(
            status_code=404,
            content={"error": f"Job {job_id!r} nicht gefunden"},
        )

    started_at = float(data.get("started_at", "0"))
    elapsed_s  = round(time.time() - started_at, 1) if started_at else 0.0

    result_raw = data.get("result")
    partial    = json.loads(result_raw) if result_raw else None

    return JobStatusResponse(
        job_id       = job_id,
        status       = data.get("status", "unknown"),
        progress_pct = float(data.get("progress_pct", "0")),
        current_test = data.get("current_test", ""),
        elapsed_s    = elapsed_s,
        partial_results = partial,
        error        = data.get("error") or None,
    )


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Job abbrechen. Setzt Redis-Flag; Worker prüft dieses zwischen den Tests."""
    r = _redis()
    await r.set(f"job:{job_id}:cancelled", "1", ex=JOB_TTL)
    await r.hset(f"job:{job_id}", "status", "cancelling")
    return {"job_id": job_id, "status": "cancelling"}


@app.websocket("/ws/jobs/{job_id}")
async def job_logs_ws(websocket: WebSocket, job_id: str):
    """WebSocket-Endpoint für Realtime-Logs.

    Sendet JSON-Events:
        {"test": "BETRAG_ZSCORE", "msg": "...", "progress_pct": 14.0}

    Verbindung bleibt offen bis der Job beendet ist oder der Client trennt.
    """
    await websocket.accept()
    r = _redis()
    pubsub = r.pubsub()
    await pubsub.subscribe(f"job:{job_id}:logs")

    try:
        async for message in pubsub.listen():
            if message.get("type") == "message":
                await websocket.send_text(message["data"])
                # Job-Ende erkennen: "done" oder "failed" in der Nachricht
                try:
                    evt = json.loads(message["data"])
                    if evt.get("progress_pct", 0) >= 100 or evt.get("done"):
                        break
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(f"job:{job_id}:logs")
        await r.aclose()
