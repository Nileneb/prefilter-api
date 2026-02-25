#!/usr/bin/env python3
"""
Buchungs-Anomalie Pre-Filter v2.0
Gradio Upload UI + 21 Anomaly Tests + Langdock Webhook Push
"""

import os
import logging

import pandas as pd
import gradio as gr

from modules.parser import read_upload, map_columns
from modules.engine import AnomalyEngine
from modules.webhook import push_to_langdock

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("prefilter")

# ── Config ───────────────────────────────────────────────────
LANGDOCK_WEBHOOK_URL = os.environ.get("LANGDOCK_WEBHOOK_URL", "")
GRADIO_USERNAME = os.environ.get("GRADIO_USERNAME", "")
GRADIO_PASSWORD = os.environ.get("GRADIO_PASSWORD", "")
ROOT_PATH = os.environ.get("ROOT_PATH", "")


# ══════════════════════════════════════════════════════════════
# GRADIO HANDLER
# ══════════════════════════════════════════════════════════════
def analyze_file(file, webhook_url: str):
    """Main Gradio handler: file in → analysis + webhook push."""
    if file is None:
        return "⚠️ Bitte eine Datei hochladen.", "", None

    filepath = file.name if hasattr(file, "name") else str(file)
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in {".csv", ".xls", ".xlsx"}:
        return f"⚠️ Nicht unterstützt: {ext} — nur CSV, XLS, XLSX", "", None

    # 1) Einlesen
    try:
        df = read_upload(filepath)
    except Exception as e:
        return f"❌ Datei-Fehler: {e}", "", None

    # 2) Spalten mappen
    df = map_columns(df)

    # 3) Anomalie-Engine
    engine = AnomalyEngine(df)
    result = engine.run()

    # 4) Zusammenfassung für UI
    stats = result["statistics"]
    flag_str = "\n".join(
        f"  {k}: {v}" for k, v in
        sorted(stats["flag_counts"].items(), key=lambda x: -x[1]) if v > 0
    )

    summary = (
        f"✅ Analyse abgeschlossen\n\n"
        f"📊 Gesamt: {stats['total_input']} Buchungen\n"
        f"🔍 Verdächtig: {stats['total_output']} ({stats['filter_ratio']})\n"
        f"📈 Ø Score: {stats['avg_score']}\n\n"
        f"Flags:\n{flag_str}\n"
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

    # 5) Tabelle für Gradio
    if result["verdaechtige_buchungen"]:
        display_df = pd.DataFrame(result["verdaechtige_buchungen"])
        display_df = display_df.sort_values("anomaly_score", ascending=False)
    else:
        display_df = pd.DataFrame()

    # 6) Webhook push
    webhook_status = ""
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

    return summary, "\n".join(result["logs"]), display_df


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
        "Buchungsdaten hochladen (CSV / XLS / XLSX) → 21 statistische Tests → "
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

    analyze_btn = gr.Button("🚀 Analyse starten", variant="primary", size="lg")

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
        inputs=[file_input, webhook_input],
        outputs=[summary_output, logs_output, table_output],
    )

    gr.Markdown(
        "---\n"
        "**21 Tests:** Z-Score · IQR · Seltene Kontierung · "
        "Wochenende/Monats-/Quartalsende · Außerhalb Geschäftszeit · "
        "Near-Duplicate · Benford 1-Ziffer · Benford 2-Ziffern · "
        "Runde Beträge · Erfasser-Anomalie · Split-Verdacht · "
        "Schwellenwert-Cluster · Belegnummer-Lücken · "
        "Doppelte Belegnummern · Beleg-Kreditor-Duplikat · "
        "Storno · Neuer Kreditor + hoher Betrag · Soll=Haben · "
        "Konto-Betrags-Anomalie · Text-Kreditor-Mismatch · "
        "Fuzzy Kreditor · Leerer Buchungstext · Velocity-Anomalie"
    )


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    auth = (GRADIO_USERNAME, GRADIO_PASSWORD) if GRADIO_USERNAME and GRADIO_PASSWORD else None
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7864,
        "share": False,
        "auth": auth,
    }
    if ROOT_PATH:
        launch_kwargs["root_path"] = ROOT_PATH
    app.launch(**launch_kwargs)
