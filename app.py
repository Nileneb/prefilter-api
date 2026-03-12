#!/usr/bin/env python3
"""
Buchungs-Anomalie Pre-Filter v4.0
Gradio UI → Celery Worker-Pool → Ergebnis-Anzeige + Langdock Webhook

Die UI erstellt Celery-Tasks direkt über Redis (kein FastAPI-Middleman).
Workers skalieren mit: docker compose up --scale worker=N
"""

import json
import os
import shutil
import time
import uuid
import logging

import redis as redis_lib
import pandas as pd
import gradio as gr
from celery import Celery

from src.webhook import push_to_langdock
from src.config import AnalysisConfig
from src.logging_config import setup_logging, get_logger

# ── Logging ──────────────────────────────────────────────────
setup_logging()
logger = get_logger("prefilter.ui")

# ── Config ───────────────────────────────────────────────────
REDIS_URL            = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
UPLOAD_DIR           = os.environ.get("UPLOAD_DIR", "/tmp/prefilter")
LANGDOCK_WEBHOOK_URL = os.environ.get("LANGDOCK_WEBHOOK_URL", "")
GRADIO_USERNAME      = os.environ.get("GRADIO_USERNAME", "")
GRADIO_PASSWORD      = os.environ.get("GRADIO_PASSWORD", "")
ROOT_PATH            = os.environ.get("ROOT_PATH", "")
JOB_TTL              = int(os.environ.get("JOB_TTL_SECONDS", "3600"))

# ── Redis + Celery (lazy connect, Fallback wenn nicht verfügbar) ────
_LOCAL_MODE = False
try:
    _r = redis_lib.from_url(REDIS_URL, decode_responses=True)
    _r.ping()
    _celery = Celery("prefilter", broker=REDIS_URL)
    logger.info("Redis erreichbar → Worker-Modus aktiv")
except Exception:
    _r = None
    _celery = None
    _LOCAL_MODE = True
    logger.warning("Redis nicht erreichbar → Lokaler Fallback-Modus (direkte Analyse)")

# Upload-Verzeichnis sicherstellen
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Modul-Level Job-ID für Cancel-Button (single-user, 2 Nutzer reichen)
_current_job_id: str | None = None


# ══════════════════════════════════════════════════════════════
# RESULT HELPER
# ══════════════════════════════════════════════════════════════
def _format_result(result: dict, webhook_url: str) -> tuple:
    """Formatiert das engine.run()-Ergebnis-Dict für die Gradio-Ausgabe."""
    stats    = result["statistics"]
    flag_str = "\n".join(
        f"  {k}: {v}" for k, v in
        sorted(stats["flag_counts"].items(), key=lambda x: -x[1]) if v > 0
    )

    summary = (
        f"Analyse abgeschlossen\n\n"
        f"Gesamt: {stats['total_input']} Buchungen\n"
        f"Verdaechtig: {stats['total_suspicious']} ({stats['filter_ratio']})\n"
        f"Ausgegeben: {stats['total_output']} (Top nach Score)\n"
        f"Avg Score: {stats['avg_score']}\n\n"
        f"Buchungs-Flags:\n{flag_str}\n"
    )

    top3 = result["verdaechtige_buchungen"][:3]
    if top3:
        summary += "\nTop-3 verdaechtige Buchungen:\n"
        for i, r in enumerate(top3, 1):
            summary += (
                f"  {i}. {r['belegnummer']}  Score={r['anomaly_score']}  "
                f"Betrag={r['betrag']:.2f}EUR\n"
                f"     Flags: {r['anomaly_flags']}\n"
            )

    # Webhook push
    url = webhook_url.strip() if webhook_url else LANGDOCK_WEBHOOK_URL
    if url:
        wh_result = push_to_langdock(result, url)
        if "error" in wh_result:
            webhook_status = f"Webhook-Fehler: {wh_result['error']}"
        else:
            webhook_status = f"Webhook gesendet -> Status {wh_result['status']}"
    else:
        webhook_status = "Keine Webhook-URL -> Ergebnisse nur lokal angezeigt"

    summary += f"\n\nWebhook: {webhook_status}"

    if result["verdaechtige_buchungen"]:
        display_df = pd.DataFrame(result["verdaechtige_buchungen"])
        display_df = display_df.sort_values("anomaly_score", ascending=False)
    else:
        display_df = pd.DataFrame()

    return summary, "\n".join(result["logs"]), display_df


# ══════════════════════════════════════════════════════════════
# GRADIO HANDLER
# ══════════════════════════════════════════════════════════════
def analyze_file(
    file,
    webhook_url: str,
    zscore_threshold: float,
    iqr_factor: float,
    near_duplicate_days: int,
    output_threshold: float,
    progress=gr.Progress(),
):
    """Erstellt einen Celery-Worker-Task und pollt Redis fuer das Ergebnis.
    Fallback: direkte Analyse ohne Redis/Celery wenn nicht verfügbar."""
    global _current_job_id
    _current_job_id = None

    if file is None:
        return "Bitte eine Datei hochladen.", "", None

    filepath = file if isinstance(file, str) else str(file)
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in {".csv", ".xls", ".xlsx"}:
        return f"Nicht unterstuetzt: {ext} -- nur CSV, XLS, XLSX", "", None

    config_dict = {
        "zscore_threshold":    zscore_threshold,
        "iqr_factor":          iqr_factor,
        "near_duplicate_days": int(near_duplicate_days),
        "output_threshold":    output_threshold,
    }

    # ── Lokaler Fallback-Modus (ohne Redis/Celery) ────────────
    if _LOCAL_MODE:
        progress(0.0, desc="Lokaler Modus: Analyse wird direkt ausgeführt...")
        from src.parser import read_upload, map_columns
        from src.engine import AnomalyEngine

        try:
            config = AnalysisConfig.model_validate(config_dict)
        except Exception:
            config = AnalysisConfig()

        df = read_upload(filepath)
        df = map_columns(df)
        progress(0.1, desc="Datei geladen, starte Analyse...")

        engine = AnomalyEngine(df, config=config)
        result = engine.run()
        progress(1.0, desc="Analyse abgeschlossen")
        return _format_result(result, webhook_url)

    # ── Worker-Modus (Redis + Celery) ─────────────────────────
    job_id   = str(uuid.uuid4())
    dest     = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    shutil.copy2(filepath, dest)

    # 2) Job in Redis anlegen
    _r.hset(f"job:{job_id}", mapping={
        "status":       "queued",
        "progress_pct": "0",
        "current_test": "",
        "started_at":   str(time.time()),
        "filename":     os.path.basename(filepath),
    })
    _r.expire(f"job:{job_id}", JOB_TTL)

    # 3) Celery-Task erstellen -- Worker holt sich den Job aus der Queue
    _celery.send_task("prefilter.analyze", args=[job_id, dest, config_dict])
    _current_job_id = job_id
    logger.info("Job erstellt: %s -> Worker-Pool", job_id)

    # 4) Redis pollen bis done / failed / cancelled
    progress(0.0, desc="Job eingereicht, warte auf Worker ...")
    while True:
        data = _r.hgetall(f"job:{job_id}")
        if not data:
            raise gr.Error("Job in Redis nicht gefunden")

        pct       = float(data.get("progress_pct", "0")) / 100.0
        test_name = data.get("current_test", "")
        status    = data.get("status", "queued")

        progress(pct, desc=test_name or f"Status: {status}")

        if status in ("done", "failed", "cancelled"):
            break

        time.sleep(0.5)

    _current_job_id = None

    # 5) Ergebnis aufbereiten
    if status == "cancelled":
        return "Analyse abgebrochen.", "", None

    if status == "failed":
        err = data.get("error", "Unbekannter Fehler")
        return f"Analyse fehlgeschlagen: {err}", "", None

    result_raw = data.get("result")
    if not result_raw:
        return "Kein Ergebnis vom Worker erhalten.", "", None

    result = json.loads(result_raw)
    return _format_result(result, webhook_url)


def cancel_analysis():
    """Setzt Abbruch-Signal direkt in Redis -- Worker stoppt nach dem laufenden Test."""
    if _LOCAL_MODE:
        return "Lokaler Modus: Abbruch nicht unterstützt (Analyse läuft synchron)."
    if _current_job_id:
        _r.set(f"job:{_current_job_id}:cancelled", "1", ex=JOB_TTL)
        _r.hset(f"job:{_current_job_id}", "status", "cancelling")
    return "Abbruch-Signal gesendet -- wird nach dem laufenden Test wirksam."


# ── Build UI ─────────────────────────────────────────────────
with gr.Blocks(
    title="Buchungs-Anomalie Pre-Filter",
) as demo:

    gr.Markdown("# Buchungs-Anomalie Pre-Filter", elem_classes="main-title")
    gr.Markdown(
        "Buchungsdaten hochladen (CSV / XLS / XLSX) -> 14 statistische Tests -> "
        "verdaechtige Buchungen an Langdock Agent senden",
        elem_classes="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="Buchungsdatei hochladen",
                file_types=[".csv", ".xls", ".xlsx"],
                type="filepath",
            )
        with gr.Column(scale=2):
            webhook_input = gr.Textbox(
                label="Langdock Webhook-URL",
                placeholder="https://api.langdock.com/webhook/...",
                value=LANGDOCK_WEBHOOK_URL,
                info="Leer lassen = nur lokale Analyse, kein Push",
            )

    # ── Konfigurierbare Schwellenwerte ────────────────────────
    with gr.Accordion("Erweiterte Einstellungen", open=False):
        with gr.Row():
            zscore_slider = gr.Slider(
                minimum=1.0, maximum=5.0, value=2.5, step=0.1,
                label="Z-Score Schwelle (BETRAG_ZSCORE)",
                info="Hoeher = weniger sensitiv (Standard: 2.5)",
            )
            iqr_slider = gr.Slider(
                minimum=0.5, maximum=5.0, value=1.5, step=0.1,
                label="IQR-Faktor (BETRAG_IQR)",
                info="Q3 + Faktor x IQR = Fence (Standard: 1.5)",
            )
        with gr.Row():
            near_dup_slider = gr.Slider(
                minimum=1, maximum=30, value=3, step=1,
                label="Near-Duplicate Tage (NEAR_DUPLICATE)",
                info="Zeitfenster in Tagen (Standard: 3)",
            )
            threshold_slider = gr.Slider(
                minimum=0.5, maximum=5.0, value=2.0, step=0.5,
                label="Output-Schwellenwert",
                info="Min. Score fuer Ausgabe (Standard: 2.0)",
            )

    with gr.Row():
        analyze_btn = gr.Button("Analyse starten", variant="primary", size="lg")
        cancel_btn  = gr.Button("Abbrechen", variant="stop", size="lg")

    with gr.Tabs():
        with gr.Tab("Ergebnis"):
            summary_output = gr.Textbox(
                label="Zusammenfassung", lines=20, interactive=False,
            )
        with gr.Tab("Verdaechtige Buchungen"):
            table_output = gr.Dataframe(
                label="Verdaechtige Buchungen (sortiert nach Score)",
                interactive=False,
                wrap=True,
            )
        with gr.Tab("Logs"):
            logs_output = gr.Textbox(
                label="Engine-Logs", lines=25, interactive=False,
            )

    analyze_btn.click(
        fn=analyze_file,
        inputs=[
            file_input, webhook_input,
            zscore_slider, iqr_slider, near_dup_slider, threshold_slider,
        ],
        outputs=[summary_output, logs_output, table_output],
    )

    cancel_btn.click(
        fn=cancel_analysis,
        inputs=[],
        outputs=[summary_output],
    )

    gr.Markdown(
        "---\n"
        "**14 Tests:** Z-Score | IQR | Near-Duplicate | Doppelte Belegnummer | "
        "Beleg-Kreditor-Duplikat | Storno | Neuer Kreditor + hoher Betrag | "
        "Konto-Betrags-Anomalie | Leerer Buchungstext | Velocity-Anomalie | "
        "Rechnungsdatum-Periode | Buchungstext-Periode | "
        "Monats-Entwicklung | Fehlende Monatsbuchung"
    )


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    auth = (GRADIO_USERNAME, GRADIO_PASSWORD) if GRADIO_USERNAME and GRADIO_PASSWORD else None
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7864,
        "share":       False,
        "auth":        auth,
        "theme":       gr.themes.Soft(),
        "css":         ".main-title { text-align: center; margin-bottom: 0.5em; } "
                       ".subtitle   { text-align: center; color: #666; margin-bottom: 1.5em; }",
    }
    if ROOT_PATH:
        launch_kwargs["root_path"] = ROOT_PATH
    demo.launch(**launch_kwargs)
