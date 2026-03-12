#!/usr/bin/env python3
"""
Buchungs-Anomalie Pre-Filter v4.0
Gradio Upload UI → FastAPI + Celery Worker-Pool → Langdock Webhook Push
"""

import os
import json
import time
import logging

import httpx
import pandas as pd
import gradio as gr

from src.webhook import push_to_langdock

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("prefilter")

# ── Config ───────────────────────────────────────────────────
LANGDOCK_WEBHOOK_URL = os.environ.get("LANGDOCK_WEBHOOK_URL", "")
GRADIO_USERNAME      = os.environ.get("GRADIO_USERNAME", "")
GRADIO_PASSWORD      = os.environ.get("GRADIO_PASSWORD", "")
ROOT_PATH            = os.environ.get("ROOT_PATH", "")
API_URL              = os.environ.get("API_URL", "http://localhost:8000")

# Modul-Level Job-ID für Cancel-Button (single-user, ausreichend für 2 Nutzer)
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
        f"✅ Analyse abgeschlossen\n\n"
        f"📊 Gesamt: {stats['total_input']} Buchungen\n"
        f"🔍 Verdächtig: {stats['total_output']} ({stats['filter_ratio']})\n"
        f"📈 Ø Score: {stats['avg_score']}\n\n"
        f"Buchungs-Flags:\n{flag_str}\n"
    )

    top3 = result["verdaechtige_buchungen"][:3]
    if top3:
        summary += "\n🏆 Top-3 verdächtige Buchungen:\n"
        for i, r in enumerate(top3, 1):
            summary += (
                f"  {i}. {r['belegnummer']}  Score={r['anomaly_score']}  "
                f"Betrag={r['betrag']:.2f}€\n"
                f"     Flags: {r['anomaly_flags']}\n"
            )

    # Webhook push
    url = webhook_url.strip() if webhook_url else LANGDOCK_WEBHOOK_URL
    if url:
        wh_result = push_to_langdock(result, url)
        if "error" in wh_result:
            webhook_status = f"❌ Webhook-Fehler: {wh_result['error']}"
        else:
            webhook_status = f"✅ Webhook gesendet → Status {wh_result['status']}"
    else:
        webhook_status = "ℹ️ Keine Webhook-URL → Ergebnisse nur lokal angezeigt"

    summary += f"\n\n📡 {webhook_status}"

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
    """Main Gradio handler: Job an FastAPI senden, auf Worker warten, Ergebnis anzeigen."""
    global _current_job_id
    _current_job_id = None

    if file is None:
        return "⚠️ Bitte eine Datei hochladen.", "", None

    filepath = file if isinstance(file, str) else str(file)
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in {".csv", ".xls", ".xlsx"}:
        return f"⚠️ Nicht unterstützt: {ext} — nur CSV, XLS, XLSX", "", None

    config = {
        "zscore_threshold":    zscore_threshold,
        "iqr_factor":          iqr_factor,
        "near_duplicate_days": int(near_duplicate_days),
        "output_threshold":    output_threshold,
    }

    # 1) Job beim FastAPI-Backend einreichen
    progress(0.0, desc="Job wird eingereicht …")
    try:
        with open(filepath, "rb") as fh:
            resp = httpx.post(
                f"{API_URL}/api/jobs",
                files={"file": (os.path.basename(filepath), fh)},
                data={"config_json": json.dumps(config)},
                timeout=30.0,
            )
        resp.raise_for_status()
    except httpx.ConnectError:
        raise gr.Error(f"API nicht erreichbar ({API_URL}) — läuft docker compose up?")
    except httpx.HTTPStatusError as e:
        raise gr.Error(f"API-Fehler: {e.response.status_code} — {e.response.text[:200]}")

    job_id = resp.json()["job_id"]
    _current_job_id = job_id
    logger.info("Job eingereicht: %s", job_id)

    # 2) Polling-Loop — Worker läuft, UI zeigt Fortschritt
    while True:
        try:
            status_resp = httpx.get(f"{API_URL}/api/jobs/{job_id}", timeout=10.0)
            status_resp.raise_for_status()
        except httpx.HTTPError as e:
            raise gr.Error(f"Statusabfrage fehlgeschlagen: {e}")

        status     = status_resp.json()
        pct        = float(status.get("progress_pct", 0)) / 100.0
        test_name  = status.get("current_test", "")
        progress(pct, desc=test_name or "Analyse läuft …")

        job_status = status["status"]
        if job_status in ("done", "failed", "cancelled"):
            break

        time.sleep(0.5)

    _current_job_id = None

    # 3) Ergebnis aufbereiten
    if job_status == "cancelled":
        return "⏹ Analyse abgebrochen.", "", None

    if job_status == "failed":
        err = status.get("error", "Unbekannter Fehler")
        return f"❌ Analyse fehlgeschlagen: {err}", "", None

    result = status.get("partial_results") or {}
    return _format_result(result, webhook_url)


def cancel_analysis():
    """Sendet Abbruch-Signal an den laufenden Worker-Job."""
    if _current_job_id:
        try:
            httpx.post(f"{API_URL}/api/jobs/{_current_job_id}/cancel", timeout=5.0)
        except httpx.HTTPError:
            pass
    return "⏹ Abbruch-Signal gesendet — wird nach dem laufenden Test wirksam."


# ── Build UI ─────────────────────────────────────────────────
with gr.Blocks(
    title="Buchungs-Anomalie Pre-Filter",
    theme=gr.themes.Soft(),
    css="""
    .main-title { text-align: center; margin-bottom: 0.5em; }
    .subtitle   { text-align: center; color: #666; margin-bottom: 1.5em; }
    """,
) as app:

    gr.Markdown("# 🔍 Buchungs-Anomalie Pre-Filter", elem_classes="main-title")
    gr.Markdown(
        "Buchungsdaten hochladen (CSV / XLS / XLSX) → 14 statistische Tests → "
        "verdächtige Buchungen an Langdock Agent senden",
        elem_classes="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="📁 Buchungsdatei hochladen",
                file_types=[".csv", ".xls", ".xlsx"],
                type="filepath",
            )
        with gr.Column(scale=2):
            webhook_input = gr.Textbox(
                label="🔗 Langdock Webhook-URL",
                placeholder="https://api.langdock.com/webhook/...",
                value=LANGDOCK_WEBHOOK_URL,
                info="Leer lassen = nur lokale Analyse, kein Push",
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

    with gr.Row():
        analyze_btn = gr.Button("🚀 Analyse starten", variant="primary", size="lg")
        cancel_btn  = gr.Button("⏹ Abbrechen", variant="stop", size="lg")

    with gr.Tabs():
        with gr.Tab("📋 Ergebnis"):
            summary_output = gr.Textbox(
                label="Zusammenfassung", lines=20, interactive=False,
            )
        with gr.Tab("📊 Verdächtige Buchungen"):
            table_output = gr.Dataframe(
                label="Verdächtige Buchungen (sortiert nach Score)",
                interactive=False,
                wrap=True,
            )
        with gr.Tab("📝 Logs"):
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
        "**14 Tests:** Z-Score · IQR · Near-Duplicate · Doppelte Belegnummer · "
        "Beleg-Kreditor-Duplikat · Storno · Neuer Kreditor + hoher Betrag · "
        "Konto-Betrags-Anomalie · Leerer Buchungstext · Velocity-Anomalie · "
        "Rechnungsdatum-Periode · Buchungstext-Periode · "
        "Monats-Entwicklung · Fehlende Monatsbuchung\n\n"
        "**API:** `POST /api/jobs` · `GET /api/jobs/{id}` · "
        "`POST /api/jobs/{id}/cancel` · `WS /ws/jobs/{id}` · "
        "Swagger: `/docs`"
    )


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    auth = (GRADIO_USERNAME, GRADIO_PASSWORD) if GRADIO_USERNAME and GRADIO_PASSWORD else None
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7864,
        "share":       False,
        "auth":        auth,
    }
    if ROOT_PATH:
        launch_kwargs["root_path"] = ROOT_PATH
    app.launch(**launch_kwargs)
