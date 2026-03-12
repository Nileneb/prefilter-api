#!/usr/bin/env python3
"""
Buchungs-Anomalie Pre-Filter v5.0
Gradio UI → Celery Worker-Pool → Ergebnis-Anzeige + Langdock Webhook

Features v5:
  - Live-Log Streaming (Generator-basiert)
  - Daten-Validierung mit Spalten-Check
  - Test-Toggles (14 Checkboxen, auto-disable bei fehlenden Spalten)
"""

import json
import os
import shutil
import time
import uuid
from datetime import datetime

import redis as redis_lib
import pandas as pd
import gradio as gr
from celery import Celery

from src.webhook import push_to_langdock
from src.config import AnalysisConfig
from src.logging_config import setup_logging, get_logger
from src.validator import (
    ALL_TEST_NAMES, TEST_CATEGORIES,
    validate_columns, format_validation_report, ValidationResult,
)

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

os.makedirs(UPLOAD_DIR, exist_ok=True)

_current_job_id: str | None = None


# ══════════════════════════════════════════════════════════════
# LIVE-LOG HELPER
# ══════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════
# RESULT HELPER
# ══════════════════════════════════════════════════════════════

def _format_result(result: dict, webhook_url: str) -> tuple[str, pd.DataFrame]:
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

    return summary, display_df


# ══════════════════════════════════════════════════════════════
# VALIDATION HANDLER
# ══════════════════════════════════════════════════════════════

def validate_file(file):
    """Nach Upload: Datei parsen, Spalten prüfen, Checkboxen updaten."""
    if file is None:
        # Alles zurücksetzen
        updates = [gr.update(value="")] + [gr.update(value=True) for _ in ALL_TEST_NAMES]
        return updates

    filepath = file if isinstance(file, str) else str(file)
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in {".csv", ".xls", ".xlsx"}:
        updates = [gr.update(value=f"❌ Nicht unterstützt: {ext}")]
        updates += [gr.update(value=True) for _ in ALL_TEST_NAMES]
        return updates

    from src.parser import read_upload, map_columns
    try:
        df = read_upload(filepath)
        df = map_columns(df)
    except Exception as e:
        updates = [gr.update(value=f"❌ Fehler beim Lesen: {e}")]
        updates += [gr.update(value=True) for _ in ALL_TEST_NAMES]
        return updates

    v = validate_columns(df)
    report = format_validation_report(v)

    # Checkbox-States: blockierte Tests ausschalten
    checkbox_updates = []
    for test_name in ALL_TEST_NAMES:
        if test_name in v.tests_blocked:
            checkbox_updates.append(gr.update(value=False))
        else:
            checkbox_updates.append(gr.update(value=True))

    return [gr.update(value=report)] + checkbox_updates


# ══════════════════════════════════════════════════════════════
# GRADIO HANDLER (Generator für Live-Log)
# ══════════════════════════════════════════════════════════════

def analyze_file(
    file,
    webhook_url: str,
    zscore_threshold: float,
    iqr_factor: float,
    near_duplicate_days: int,
    output_threshold: float,
    *test_toggles,
):
    """Generator: yielded (summary, logs, table) bei jedem Schritt."""
    global _current_job_id
    _current_job_id = None
    live_log_lines: list[str] = []

    def log(msg: str) -> None:
        live_log_lines.append(f"[{_ts()}] {msg}")

    def current_state(summary="", table=None):
        return summary, "\n".join(live_log_lines), table

    if file is None:
        yield current_state("Bitte eine Datei hochladen.")
        return

    filepath = file if isinstance(file, str) else str(file)
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in {".csv", ".xls", ".xlsx"}:
        yield current_state(f"Nicht unterstuetzt: {ext} -- nur CSV, XLS, XLSX")
        return

    # Enabled tests aus Checkboxen
    enabled_tests: set[str] = set()
    for i, test_name in enumerate(ALL_TEST_NAMES):
        if i < len(test_toggles) and test_toggles[i]:
            enabled_tests.add(test_name)

    disabled = set(ALL_TEST_NAMES) - enabled_tests
    if disabled:
        log(f"⏭️ Deaktivierte Tests: {', '.join(sorted(disabled))}")

    config_dict = {
        "zscore_threshold":    zscore_threshold,
        "iqr_factor":          iqr_factor,
        "near_duplicate_days": int(near_duplicate_days),
        "output_threshold":    output_threshold,
    }

    # ── Lokaler Fallback-Modus ────────────────────────────────
    if _LOCAL_MODE:
        log("📂 Lokaler Modus: Datei wird geladen...")
        yield current_state("Analyse läuft...")

        from src.parser import read_upload, map_columns
        from src.engine import AnomalyEngine

        try:
            config = AnalysisConfig.model_validate(config_dict)
        except Exception:
            config = AnalysisConfig()

        df = read_upload(filepath)
        df = map_columns(df)
        log(f"📂 Datei geladen: {os.path.basename(filepath)} ({len(df):,} Zeilen)".replace(",", "."))
        yield current_state("Analyse läuft...")

        engine = AnomalyEngine(df, config=config)

        # Instrumented run mit Live-Logging
        from src.engine import _ALL_TESTS, NUM_TESTS
        stats = engine._compute_stats()
        log(f"📊 Statistiken berechnet")
        yield current_state("Analyse läuft...")

        for i, test in enumerate(_ALL_TESTS, start=1):
            if engine._is_cancelled():
                log("⛔ Analyse abgebrochen")
                break
            if test.name not in enabled_tests:
                log(f"⏭️ Test {i}/{NUM_TESTS}: {test.name} übersprungen (deaktiviert)")
                engine.flag_counts[test.name] = 0
                yield current_state("Analyse läuft...")
                continue
            count = test.run_with_logging(df, stats, config) if hasattr(test, "run_with_logging") else test.run(df, stats, config)
            engine.flag_counts[test.name] = count
            engine._log(f"[{i:02d}/{NUM_TESTS}] {test.name}: {count}")
            emoji = "🔴" if count > 1000 else "🟡" if count > 0 else "✅"
            log(f"{emoji} Test {i}/{NUM_TESTS}: {test.name} → {count:,} Flags".replace(",", "."))
            yield current_state("Analyse läuft...")

        engine._compute_scores()
        result = engine._export()
        log(f"✅ Fertig! {result['statistics']['total_suspicious']:,} verdächtige Buchungen".replace(",", "."))

        summary, display_df = _format_result(result, webhook_url)
        yield current_state(summary, display_df)
        return

    # ── Worker-Modus (Redis + Celery) ─────────────────────────
    job_id = str(uuid.uuid4())
    dest   = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    shutil.copy2(filepath, dest)

    _r.hset(f"job:{job_id}", mapping={
        "status":       "queued",
        "progress_pct": "0",
        "current_test": "",
        "started_at":   str(time.time()),
        "filename":     os.path.basename(filepath),
    })
    _r.expire(f"job:{job_id}", JOB_TTL)

    task_args = [job_id, dest, config_dict]
    if enabled_tests != set(ALL_TEST_NAMES):
        task_args.append(sorted(enabled_tests))

    _celery.send_task("prefilter.analyze", args=task_args)
    _current_job_id = job_id
    logger.info("Job erstellt: %s -> Worker-Pool", job_id)
    log(f"📤 Job {job_id[:8]}... eingereicht")
    yield current_state("Warte auf Worker...")

    prev_test = ""
    while True:
        data = _r.hgetall(f"job:{job_id}")
        if not data:
            yield current_state("Job in Redis nicht gefunden.")
            return

        status    = data.get("status", "queued")
        test_name = data.get("current_test", "")

        if test_name and test_name != prev_test:
            log(f"🔬 {test_name}...")
            prev_test = test_name
            yield current_state("Analyse läuft...")

        if status in ("done", "failed", "cancelled"):
            break

        time.sleep(0.5)

    _current_job_id = None

    if status == "cancelled":
        log("⛔ Analyse abgebrochen")
        yield current_state("Analyse abgebrochen.")
        return

    if status == "failed":
        err = data.get("error", "Unbekannter Fehler")
        log(f"❌ Fehler: {err}")
        yield current_state(f"Analyse fehlgeschlagen: {err}")
        return

    result_raw = data.get("result")
    if not result_raw:
        yield current_state("Kein Ergebnis vom Worker erhalten.")
        return

    result = json.loads(result_raw)

    # Engine-Logs in Live-Log übernehmen
    for engine_log in result.get("logs", []):
        if engine_log.startswith("["):
            # Test-Ergebnis-Zeilen
            log(f"📋 {engine_log}")

    elapsed = data.get("elapsed_s", "?")
    log(f"✅ Fertig in {elapsed}s — {result['statistics']['total_suspicious']:,} verdächtige Buchungen".replace(",", "."))

    summary, display_df = _format_result(result, webhook_url)
    yield current_state(summary, display_df)


def cancel_analysis():
    if _LOCAL_MODE:
        return "Lokaler Modus: Abbruch nicht unterstützt (Analyse läuft synchron)."
    if _current_job_id:
        _r.set(f"job:{_current_job_id}:cancelled", "1", ex=JOB_TTL)
        _r.hset(f"job:{_current_job_id}", "status", "cancelling")
    return "Abbruch-Signal gesendet -- wird nach dem laufenden Test wirksam."


# ═══════════════════════════════════════════════════════════════
# BUILD UI
# ═══════════════════════════════════════════════════════════════

with gr.Blocks(
    title="Buchungs-Anomalie Pre-Filter",
) as demo:

    gr.Markdown("# Buchungs-Anomalie Pre-Filter", elem_classes="main-title")
    gr.Markdown(
        "Buchungsdaten hochladen (CSV / XLS / XLSX) → 14 statistische Tests → "
        "verdächtige Buchungen an Langdock Agent senden",
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

    # ── Datenqualitäts-Check ──────────────────────────────────
    validation_output = gr.Textbox(
        label="📋 Datenqualitäts-Check",
        lines=8,
        interactive=False,
        visible=True,
    )

    # ── Konfigurierbare Schwellenwerte ────────────────────────
    with gr.Accordion("⚙️ Erweiterte Einstellungen", open=False):
        with gr.Row():
            zscore_slider = gr.Slider(
                minimum=1.0, maximum=5.0, value=2.5, step=0.1,
                label="Z-Score Schwelle (BETRAG_ZSCORE)",
                info="Höher = weniger sensitiv (Standard: 2.5)",
            )
            iqr_slider = gr.Slider(
                minimum=0.5, maximum=5.0, value=1.5, step=0.1,
                label="IQR-Faktor (BETRAG_IQR)",
                info="Q3 + Faktor × IQR = Fence (Standard: 1.5)",
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
                info="Min. Score für Ausgabe (Standard: 2.0)",
            )

    # ── Test-Konfiguration (14 Checkboxen) ────────────────────
    test_checkboxes: list[gr.Checkbox] = []
    with gr.Accordion("🔧 Test-Konfiguration", open=False):
        gr.Markdown("Tests an-/abschalten. Blockierte Tests werden nach dem Datei-Upload automatisch deaktiviert.")
        for category, test_names in TEST_CATEGORIES.items():
            with gr.Row():
                for test_name in test_names:
                    cb = gr.Checkbox(label=test_name, value=True)
                    test_checkboxes.append(cb)

    with gr.Row():
        analyze_btn = gr.Button("▶️ Analyse starten", variant="primary", size="lg")
        cancel_btn  = gr.Button("⛔ Abbrechen", variant="stop", size="lg")

    with gr.Tabs():
        with gr.Tab("Ergebnis"):
            summary_output = gr.Textbox(
                label="Zusammenfassung", lines=20, interactive=False,
            )
        with gr.Tab("Verdächtige Buchungen"):
            table_output = gr.Dataframe(
                label="Verdächtige Buchungen (sortiert nach Score)",
                interactive=False,
                wrap=True,
            )
        with gr.Tab("📜 Live-Log"):
            logs_output = gr.Textbox(
                label="Live-Log", lines=25, interactive=False,
            )

    # ── Event: File-Upload → Validierung ──────────────────────
    file_input.change(
        fn=validate_file,
        inputs=[file_input],
        outputs=[validation_output] + test_checkboxes,
    )

    # ── Event: Analyse starten ────────────────────────────────
    analyze_btn.click(
        fn=analyze_file,
        inputs=[
            file_input, webhook_input,
            zscore_slider, iqr_slider, near_dup_slider, threshold_slider,
            *test_checkboxes,
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
        "**14 Tests:** Z-Score | IQR | Konto-Betrag | Near-Duplicate | "
        "Doppelte Belegnummer | Beleg-Kreditor-Duplikat | Storno | "
        "Leerer Buchungstext | Rechnungsdatum-Periode | Buchungstext-Periode | "
        "Neuer Kreditor | Velocity-Anomalie | "
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
