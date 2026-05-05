#!/usr/bin/env python3
"""
Buchungs-Anomalie Pre-Filter v5.0
Gradio UI → Celery Worker-Pool → Ergebnis-Anzeige + Langdock Webhook

Features v5:
  - Live-Log Streaming (Generator-basiert)
  - Daten-Validierung mit Spalten-Check
  - Test-Toggles (13 Checkboxen, auto-disable bei fehlenden Spalten)
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
from src.file_store import store_upload, store_result, list_uploads
from src.logging_config import setup_logging, get_logger
from src.validator import (
    ALL_TEST_NAMES, TEST_CATEGORIES,
    validate_columns, format_validation_report, ValidationResult,
)
from src.charts import ChartBuilder, DynamicChartBuilder, DYNAMIC_CHART_TYPES, classify_columns, check_column_quality, _empty_figure
from src.feedback import FeedbackStore, FeedbackLabel, MIN_LABELS_FOR_STATS
from src.feedback_stats import format_feedback_report

import plotly.graph_objects as go

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
_last_engine_df: pd.DataFrame | None = None   # für on-demand Charts
_last_result: dict | None = None
_last_file_path: str | None = None            # für Chart-Rebuild im Worker-Modus
_last_flags_parquet: str | None = None        # Flags-Parquet vom Worker für Chart-Rebuild
_last_mandant_id: str = "unknown"             # für Feedback-Speicherung
_last_analysis_ts: str = ""                   # Zeitpunkt der letzten Analyse
_feedback_store = FeedbackStore()


# ══════════════════════════════════════════════════════════════
# LIVE-LOG HELPER
# ══════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════
# RESULT HELPER
# ══════════════════════════════════════════════════════════════

def _save_csv(df: pd.DataFrame) -> str | None:
    """Speichert DataFrame als CSV in /tmp und gibt den Pfad zurück."""
    if df is None or df.empty:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("/tmp", f"verdaechtige_buchungen_{ts}.csv")
    df.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
    return path


def _detect_mandant(df: pd.DataFrame, filepath: str) -> str:
    """Erkennt Mandanten-ID aus Daten oder Dateiname."""
    import re as _re
    if "mandant" in df.columns:
        mandant_vals = df["mandant"].astype(str).str.strip()
        mandant_vals = mandant_vals[mandant_vals != ""]
        if not mandant_vals.empty:
            return mandant_vals.mode().iloc[0]
    match = _re.search(r"(\d{2,6})", os.path.basename(filepath))
    if match:
        return match.group(1)
    return "unknown"


def _build_charts(df: pd.DataFrame, result: dict) -> dict:
    """Baut alle Plotly-Charts aus dem Engine-DataFrame."""
    try:
        builder = ChartBuilder(df, result)
        return builder.all_charts()
    except Exception:
        return {}


def _format_result(result: dict, webhook_url: str) -> tuple[str, pd.DataFrame]:
    """Formatiert das engine.run()-Ergebnis-Dict für die Gradio-Ausgabe."""
    stats    = result["statistics"]
    flag_str = "\n".join(
        f"  {k}: {v}" for k, v in
        sorted(stats["flag_counts"].items(), key=lambda x: -x[1]) if v > 0
    )

    max_score = stats.get('max_possible_score', '?')
    summary = (
        f"Analyse abgeschlossen\n\n"
        f"Gesamt: {stats['total_input']} Buchungen\n"
        f"Verdaechtig: {stats['total_suspicious']} ({stats['filter_ratio']})\n"
        f"Ausgegeben: {stats['total_output']}"
        f"{' (limitiert auf Top ' + str(stats['total_output']) + ' nach Score)' if stats['total_output'] < stats['total_suspicious'] else ''}\n"
        f"Avg Score: {stats['avg_score']}/{max_score}\n\n"
        f"Buchungs-Flags:\n{flag_str}\n"
    )

    top3 = result["verdaechtige_buchungen"][:3]
    if top3:
        summary += "\nTop-3 verdaechtige Buchungen:\n"
        for i, r in enumerate(top3, 1):
            summary += (
                f"  {i}. Beleg {r['belegnummer']}  Score={r['anomaly_score']}/{max_score}  "
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
        display_df.insert(0, "bewertung", "")
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
    prefix_ignore: str,
    text_konto_threshold: float,
    text_konto_konto_min: int,
    text_konto_konto_max: int,
    *test_toggles,
):
    """Generator: yielded (summary, logs, table, csv, charts...) bei jedem Schritt."""
    global _current_job_id
    global _last_engine_df, _last_result, _last_file_path, _last_flags_parquet
    global _last_mandant_id, _last_analysis_ts
    _current_job_id = None
    _last_engine_df = None
    _last_result = None
    _last_file_path = None
    _last_flags_parquet = None
    _last_analysis_ts = datetime.now().isoformat()
    live_log_lines: list[str] = []

    def log(msg: str) -> None:
        live_log_lines.append(f"[{_ts()}] {msg}")

    def current_state(summary="", table=None, csv_path=None):
        return (summary, "\n".join(live_log_lines), table, gr.update(value=csv_path, visible=csv_path is not None))

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
        "doppelte_beleg_prefix_ignore": prefix_ignore.strip() if prefix_ignore else "",
        "text_konto_threshold": text_konto_threshold,
        "text_konto_konto_min": int(text_konto_konto_min),
        "text_konto_konto_max": int(text_konto_konto_max),
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
        csv_path = _save_csv(display_df)

        # Upload + Ergebnis persistent speichern
        try:
            mandant_id = _detect_mandant(df, filepath)
            _last_mandant_id = mandant_id
            store_upload(mandant_id, filepath, os.path.basename(filepath))
            store_result(mandant_id, os.path.basename(filepath), result, display_df)
            log(f"📁 Upload + Ergebnis gespeichert unter data/uploads/{mandant_id}/")
        except Exception as e:
            log(f"⚠️ File-Store fehlgeschlagen: {e}")

        # Engine-Daten für on-demand Charts speichern
        _last_engine_df = engine.df
        _last_result = result
        log("📊 Charts können jetzt einzeln im Tab 'Visualisierungen' generiert werden.")

        yield current_state(summary, display_df, csv_path)
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
    _r.expire(f"job:{job_id}:log", JOB_TTL)

    task_args = [job_id, dest, config_dict]
    if enabled_tests != set(ALL_TEST_NAMES):
        task_args.append(sorted(enabled_tests))

    _celery.send_task("prefilter.analyze", args=task_args)
    _current_job_id = job_id
    logger.info("Job erstellt: %s -> Worker-Pool", job_id)
    log(f"📤 Job {job_id[:8]}... eingereicht")
    yield current_state("Warte auf Worker...")

    prev_test = ""
    log_cursor = 0  # Cursor für Redis-Log-Liste
    while True:
        data = _r.hgetall(f"job:{job_id}")
        if not data:
            yield current_state("Job in Redis nicht gefunden.")
            return

        status    = data.get("status", "queued")
        test_name = data.get("current_test", "")

        # Live-Log aus Redis-Liste lesen
        new_entries = _r.lrange(f"job:{job_id}:log", log_cursor, -1)
        if new_entries:
            for entry in new_entries:
                live_log_lines.append(entry)
            log_cursor += len(new_entries)
            yield current_state("Analyse läuft...")

        if test_name and test_name != prev_test:
            prev_test = test_name

        if status in ("done", "failed", "cancelled"):
            # Letzte Log-Einträge abholen
            final_entries = _r.lrange(f"job:{job_id}:log", log_cursor, -1)
            if final_entries:
                for entry in final_entries:
                    live_log_lines.append(entry)
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

    elapsed = data.get("elapsed_s", "?")
    log(f"✅ Fertig in {elapsed}s — {result['statistics']['total_suspicious']:,} verdächtige Buchungen".replace(",", "."))

    summary, display_df = _format_result(result, webhook_url)
    csv_path = _save_csv(display_df)
    # Worker-Modus: Result + Dateipfad speichern für Chart-Rebuild
    _last_result = result
    _last_file_path = dest
    _last_flags_parquet = data.get("flags_parquet")
    _last_engine_df = None  # wird on-demand beim ersten Chart-Klick gebaut
    # Mandant für Feedback ableiten
    try:
        from src.parser import read_upload, map_columns
        _tmp_df = read_upload(dest)
        _tmp_df = map_columns(_tmp_df)
        _last_mandant_id = _detect_mandant(_tmp_df, dest)
    except Exception:
        _last_mandant_id = _detect_mandant(pd.DataFrame(), dest)
    log("📊 Charts können jetzt im Tab 'Visualisierungen' generiert werden.")
    yield current_state(summary, display_df, csv_path)


def cancel_analysis():
    if _LOCAL_MODE:
        return "Lokaler Modus: Abbruch nicht unterstützt (Analyse läuft synchron)."
    if _current_job_id:
        _r.set(f"job:{_current_job_id}:cancelled", "1", ex=JOB_TTL)
        _r.hset(f"job:{_current_job_id}", "status", "cancelling")
    return "Abbruch-Signal gesendet -- wird nach dem laufenden Test wirksam."


# ══════════════════════════════════════════════════════════════
# FEEDBACK HANDLER
# ══════════════════════════════════════════════════════════════

def save_feedback(table_data: pd.DataFrame, pruefer: str) -> str:
    """Iteriert über die Ergebnis-Tabelle und speichert alle Zeilen mit Bewertung."""
    if table_data is None or table_data.empty:
        return "⚠️ Keine Daten vorhanden."
    if not pruefer or not pruefer.strip():
        return "⚠️ Bitte Prüfer-Kürzel eingeben."
    if "bewertung" not in table_data.columns:
        return "⚠️ Spalte 'bewertung' fehlt in Tabelle."

    pruefer = pruefer.strip()
    saved = 0
    valid_labels = {"tp", "fp", "unsure"}

    for idx, row in table_data.iterrows():
        raw_label = str(row.get("bewertung", "")).strip().lower()
        if not raw_label or raw_label not in valid_labels:
            continue
        label = FeedbackLabel(
            mandant_id=_last_mandant_id,
            analysis_timestamp=_last_analysis_ts,
            row_index=int(idx),
            belegnummer=str(row.get("belegnummer", "")),
            anomaly_score=float(row.get("anomaly_score", 0)),
            anomaly_flags=str(row.get("anomaly_flags", "")),
            label=raw_label,
            pruefer=pruefer,
        )
        _feedback_store.save(label)
        saved += 1

    total = _feedback_store.count(_last_mandant_id)
    return f"✅ {saved} Labels gespeichert. Gesamt für Mandant {_last_mandant_id}: {total}"


# ══════════════════════════════════════════════════════════════
# ON-DEMAND CHART FUNKTIONEN
# ══════════════════════════════════════════════════════════════

def _rebuild_df_for_charts(filepath: str, result: dict) -> pd.DataFrame | None:
    """Baut den Engine-DataFrame aus der Originaldatei für Charts nach.

    Liest die Datei, bereitet Spalten vor (_prepare), setzt dann Flags+Scores
    aus dem gespeicherten Flags-Parquet oder initialisiert leer.
    """
    from src.parser import read_upload, map_columns
    from src.engine import AnomalyEngine
    try:
        df = read_upload(filepath)
        df = map_columns(df)
        engine = AnomalyEngine(df)  # _prepare() wird in __init__ aufgerufen

        # Flags aus Parquet laden (vom Worker gespeichert)
        if _last_flags_parquet and os.path.exists(_last_flags_parquet):
            flags_df = pd.read_parquet(_last_flags_parquet)
            for col in flags_df.columns:
                engine.df[col] = flags_df[col]
        else:
            # Fallback: Flag-Spalten + Score leer initialisieren
            for flag_name in result.get("statistics", {}).get("flag_counts", {}).keys():
                col = f"flag_{flag_name}"
                if col not in engine.df.columns:
                    engine.df[col] = False
            if "_score" not in engine.df.columns:
                engine.df["_score"] = 0.0
        return engine.df
    except Exception as exc:
        logger.error("Chart-Rebuild fehlgeschlagen: %s", exc, exc_info=True)
        return None

def _get_chart_builder() -> ChartBuilder | None:
    """Gibt einen ChartBuilder zurück wenn Analysedaten vorhanden, sonst None."""
    global _last_engine_df
    if _last_result is None:
        return None
    # Lazy-Rebuild: Im Worker-Modus wird der DataFrame beim ersten Chart-Klick gebaut
    if _last_engine_df is None and _last_file_path is not None:
        _last_engine_df = _rebuild_df_for_charts(_last_file_path, _last_result)
    if _last_engine_df is None:
        return None
    return ChartBuilder(_last_engine_df, _last_result)


def generate_score_distribution():
    b = _get_chart_builder()
    if b is None:
        return gr.update(value=None)
    return b.score_distribution()


def generate_flag_frequency():
    b = _get_chart_builder()
    if b is None:
        return gr.update(value=None)
    return b.flag_frequency()


def generate_monthly_pnl():
    b = _get_chart_builder()
    if b is None:
        return gr.update(value=None)
    return b.monthly_pnl()


def generate_top_accounts():
    b = _get_chart_builder()
    if b is None:
        return gr.update(value=None)
    return b.top_accounts()


def generate_ertrag_aufwand():
    b = _get_chart_builder()
    if b is None:
        return gr.update(value=None)
    return b.ertrag_aufwand_monthly()


def generate_heatmap():
    b = _get_chart_builder()
    if b is None:
        return gr.update(value=None)
    return b.volume_heatmap()


def generate_betrag_vs_score():
    b = _get_chart_builder()
    if b is None:
        return gr.update(value=None)
    return b.betrag_vs_score()


def generate_treemap():
    b = _get_chart_builder()
    if b is None:
        return gr.update(value=None)
    return b.kreditor_treemap()


def generate_zeitreihe():
    b = _get_chart_builder()
    if b is None:
        return gr.update(value=None)
    return b.zeitreihe_konto()


def generate_sh_balance():
    b = _get_chart_builder()
    if b is None:
        return gr.update(value=None)
    return b.soll_haben_balance()


def generate_all_charts():
    """Alle 10 Charts auf einmal generieren."""
    b = _get_chart_builder()
    if b is None:
        return [gr.update(value=None)] * 10
    charts = b.all_charts()
    keys = [
        "score_distribution", "flag_frequency", "monthly_pnl", "top_accounts",
        "ertrag_aufwand_monthly", "volume_heatmap", "betrag_vs_score",
        "kreditor_treemap", "zeitreihe_konto", "soll_haben_balance",
    ]
    return [charts.get(k) for k in keys]


# ══════════════════════════════════════════════════════════════
# DYNAMISCHER CHART-BUILDER (Event-Handler)
# ══════════════════════════════════════════════════════════════

_last_dynamic_fig: go.Figure | None = None


def _populate_dynamic_dropdowns():
    """Befüllt Dropdowns mit Spalten aus dem letzten Engine-DataFrame."""
    global _last_engine_df
    # Lazy-Rebuild triggern (Worker-Modus)
    if _last_engine_df is None and _last_file_path is not None and _last_result is not None:
        _last_engine_df = _rebuild_df_for_charts(_last_file_path, _last_result)
    if _last_engine_df is None:
        empty = gr.update(choices=[], value=None)
        return [empty] * 5

    cols = classify_columns(_last_engine_df)
    num_choices = cols["numeric"]
    all_choices = cols["all"]
    cat_choices = ["(keine)"] + cols["categorical"]
    size_choices = ["(keine)"] + cols["numeric"]

    return [
        gr.update(choices=all_choices, value=all_choices[0] if all_choices else None),
        gr.update(choices=num_choices, value=num_choices[0] if num_choices else None),
        gr.update(choices=num_choices, value=num_choices[1] if len(num_choices) > 1 else None),
        gr.update(choices=cat_choices, value="(keine)"),
        gr.update(choices=size_choices, value="(keine)"),
    ]


def _toggle_z_axis(chart_type):
    """Zeigt Z-Achse nur bei 3D-Charts."""
    return gr.update(visible="3D" in chart_type)


def _build_dynamic_chart(chart_type, x, y, z, color, size):
    """Baut den dynamischen Chart."""
    global _last_dynamic_fig, _last_engine_df
    # Lazy-Rebuild triggern (Worker-Modus)
    if _last_engine_df is None and _last_file_path is not None and _last_result is not None:
        _last_engine_df = _rebuild_df_for_charts(_last_file_path, _last_result)
    if _last_engine_df is None:
        _last_dynamic_fig = None
        return _empty_figure("Erst eine Analyse durchführen"), ""
    builder = DynamicChartBuilder(_last_engine_df)
    warnings = []
    for col in [x, y, z]:
        if col and col != "(keine)":
            w = check_column_quality(_last_engine_df, col)
            if w:
                warnings.append(w)
    fig = builder.build(chart_type, x, y, z, color, size)
    _last_dynamic_fig = fig
    return fig, "\n".join(warnings)


def _export_chart_html():
    """Exportiert den letzten dynamischen Chart als HTML."""
    import tempfile
    if _last_dynamic_fig is None:
        return gr.update(value=None, visible=False)
    path = os.path.join(tempfile.gettempdir(), f"chart_{int(time.time())}.html")
    _last_dynamic_fig.write_html(path, include_plotlyjs="cdn")
    return gr.update(value=path, visible=True)


def _generate_3d_landscape():
    """Generiert den vordefinierten 3D-Scatter."""
    b = _get_chart_builder()
    if b is None:
        return gr.update(value=None)
    return b.anomaly_landscape_3d()


# ═══════════════════════════════════════════════════════════════
# BUILD UI
# ═══════════════════════════════════════════════════════════════

with gr.Blocks(
    title="Buchungs-Anomalie Pre-Filter",
) as demo:

    gr.Markdown("# Buchungs-Anomalie Pre-Filter", elem_classes="main-title")
    gr.Markdown(
        "Buchungsdaten hochladen (CSV / XLS / XLSX) → 13 statistische Tests → "
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
        with gr.Row():
            prefix_ignore_input = gr.Textbox(
                label="Belegnummer-Präfixe ignorieren (kommagetrennt)",
                placeholder="z.B. RW, SB",
                value="",
                info="Belegnummern mit diesen Präfixen werden bei DOPPELTE_BELEGNUMMER ignoriert",
            )
        with gr.Row():
            text_konto_slider = gr.Slider(
                minimum=0.05, maximum=0.95, value=0.3, step=0.05,
                label="TEXT_KONTO_MATCH Threshold (Cosine-Similarity)",
                info="Buchungstext ↔ Kontobezeichnung: unter diesem Wert → Anomalie (Standard: 0.30)",
            )
        with gr.Row():
            text_konto_min_input = gr.Number(
                value=40000, minimum=0, maximum=99999, precision=0,
                label="Sachkonto Min (inkl.)",
                info="Untergrenze konto_soll für TEXT_KONTO_MATCH (Standard: 40000)",
            )
            text_konto_max_input = gr.Number(
                value=80000, minimum=0, maximum=99999, precision=0,
                label="Sachkonto Max (exkl.)",
                info="Obergrenze konto_soll, exklusiv (Standard: 80000 → prüft bis 79999)",
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
                label="Verdächtige Buchungen (sortiert nach Score) — Spalte 'bewertung' mit tp/fp/unsure ausfüllen",
                interactive=True,
                wrap=True,
            )
            export_file = gr.File(label="📥 CSV-Download", visible=False)
            gr.Markdown("### 🏷️ Prüfer-Feedback")
            gr.Markdown(
                "Bewerte verdächtige Buchungen: Trage in der Spalte **bewertung** ein:\n"
                "- **tp** = True Positive (echte Anomalie)\n"
                "- **fp** = False Positive (Fehlalarm)\n"
                "- **unsure** = Unklar"
            )
            with gr.Row():
                feedback_pruefer = gr.Textbox(
                    label="Prüfer-Kürzel", placeholder="z.B. MM, JS",
                    scale=1,
                )
                feedback_save_btn = gr.Button("💾 Feedback speichern", variant="secondary", scale=1)
                feedback_status = gr.Textbox(label="Status", interactive=False, scale=2)
        with gr.Tab("📜 Live-Log"):
            logs_output = gr.Textbox(
                label="Live-Log", lines=25, interactive=False,
            )
        with gr.Tab("📊 Visualisierungen"):
            gr.Markdown(
                "### Charts on-demand generieren\n"
                "Erst die Analyse abschließen, dann einzelne Charts per Klick generieren. "
                "Bei großen Datensätzen wird automatisch eine Stichprobe verwendet."
            )
            all_charts_btn = gr.Button("📊 Alle Charts generieren", variant="primary")
            gr.Markdown("### Überblicks-Dashboards")
            with gr.Row():
                with gr.Column():
                    btn_score_dist = gr.Button("▶ Score-Verteilung", size="sm")
                    chart_score_dist = gr.Plot(label="Score-Verteilung")
                with gr.Column():
                    btn_flag_freq = gr.Button("▶ Flag-Häufigkeit", size="sm")
                    chart_flag_freq = gr.Plot(label="Flag-Häufigkeit")
            with gr.Row():
                with gr.Column():
                    btn_monthly_pnl = gr.Button("▶ Monatliche Betrags-Entwicklung", size="sm")
                    chart_monthly_pnl = gr.Plot(label="Monatliche Betrags-Entwicklung")
                with gr.Column():
                    btn_top_acc = gr.Button("▶ Top-Konten nach Score", size="sm")
                    chart_top_acc = gr.Plot(label="Top-Konten nach Score")
            with gr.Row():
                with gr.Column():
                    btn_ertrag_aufw = gr.Button("▶ Ertrag vs. Aufwand", size="sm")
                    chart_ertrag_aufw = gr.Plot(label="Ertrag vs. Aufwand")
                with gr.Column():
                    btn_heatmap = gr.Button("▶ Buchungsvolumen-Heatmap", size="sm")
                    chart_heatmap = gr.Plot(label="Buchungsvolumen-Heatmap")
            gr.Markdown("### Anomalie-Details")
            with gr.Row():
                with gr.Column():
                    btn_scatter = gr.Button("▶ Betrag vs. Score", size="sm")
                    chart_scatter = gr.Plot(label="Betrag vs. Score")
                with gr.Column():
                    btn_treemap = gr.Button("▶ Kreditor-Treemap", size="sm")
                    chart_treemap = gr.Plot(label="Kreditor-Treemap")
            with gr.Row():
                with gr.Column():
                    btn_zeitreihe = gr.Button("▶ Zeitreihe (Top-Konto)", size="sm")
                    chart_zeitreihe = gr.Plot(label="Zeitreihe (Top-Konto)")
                with gr.Column():
                    btn_sh_balance = gr.Button("▶ Soll/Haben-Balance", size="sm")
                    chart_sh_balance = gr.Plot(label="Soll/Haben-Balance")

            gr.Markdown("### 3D-Ansichten")
            with gr.Row():
                with gr.Column():
                    btn_3d_landscape = gr.Button("▶ 3D Anomalie-Landschaft", size="sm")
                    chart_3d_landscape = gr.Plot(label="3D Anomalie-Landschaft")

        with gr.Tab("🔬 Eigene Visualisierung"):
            gr.Markdown(
                "### Dynamischer Chart-Builder\n"
                "Wähle Diagrammtyp und Achsen aus den verfügbaren Spalten. "
                "Erst die Analyse abschließen, dann hier eigene Charts erstellen."
            )
            dyn_load_btn = gr.Button("🔄 Spalten laden", size="sm")
            with gr.Row():
                with gr.Column(scale=1):
                    dyn_chart_type = gr.Dropdown(
                        label="Diagrammtyp",
                        choices=list(DYNAMIC_CHART_TYPES.keys()),
                        value="Scatter",
                    )
                with gr.Column(scale=1):
                    dyn_x = gr.Dropdown(label="X-Achse", choices=[], interactive=True)
                with gr.Column(scale=1):
                    dyn_y = gr.Dropdown(label="Y-Achse", choices=[], interactive=True)

            with gr.Row():
                with gr.Column(scale=1):
                    dyn_z = gr.Dropdown(
                        label="Z-Achse (nur 3D)",
                        choices=[], interactive=True, visible=False,
                    )
                with gr.Column(scale=1):
                    dyn_color = gr.Dropdown(
                        label="Farbe (optional)",
                        choices=[], interactive=True,
                    )
                with gr.Column(scale=1):
                    dyn_size = gr.Dropdown(
                        label="Größe (optional, nur Scatter)",
                        choices=[], interactive=True,
                    )

            dyn_quality_warning = gr.Textbox(
                label="Datenqualitäts-Hinweise", lines=2, interactive=False, visible=True,
            )
            with gr.Row():
                dyn_build_btn = gr.Button("📊 Chart erstellen", variant="primary")
                dyn_export_btn = gr.Button("📥 Als HTML exportieren", size="sm")
            dyn_chart_output = gr.Plot(label="Dynamischer Chart")
            dyn_export_file = gr.File(label="Chart-Download", visible=False)

        with gr.Tab("📁 History"):
            gr.Markdown("### Analyse-Verlauf")
            history_mandant = gr.Textbox(label="Mandant-ID", value="150")
            history_btn = gr.Button("🔄 Laden")
            history_table = gr.Dataframe(label="Gespeicherte Analysen", interactive=False)

        with gr.Tab("📈 Feedback-Stats"):
            gr.Markdown(
                "### Feedback-Statistiken\n"
                f"Zeigt Precision und FP-Rate pro Flag. Mindestens **{MIN_LABELS_FOR_STATS}** Labels erforderlich."
            )
            stats_mandant = gr.Textbox(label="Mandant-ID (leer = alle)", value="")
            stats_btn = gr.Button("📈 Statistiken laden")
            stats_output = gr.Textbox(label="Bericht", lines=20, interactive=False)

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
            prefix_ignore_input,
            text_konto_slider,
            text_konto_min_input,
            text_konto_max_input,
            *test_checkboxes,
        ],
        outputs=[
            summary_output, logs_output, table_output, export_file,
        ],
    )

    cancel_btn.click(
        fn=cancel_analysis,
        inputs=[],
        outputs=[summary_output],
    )

    # ── Event: Feedback speichern ─────────────────────────────
    feedback_save_btn.click(
        fn=save_feedback,
        inputs=[table_output, feedback_pruefer],
        outputs=[feedback_status],
    )

    # ── Event: History laden ──────────────────────────────────
    def load_history(mandant_id):
        uploads = list_uploads(mandant_id.strip())
        if not uploads:
            return pd.DataFrame({"Hinweis": ["Keine gespeicherten Analysen gefunden"]})
        return pd.DataFrame(uploads)

    history_btn.click(
        fn=load_history,
        inputs=[history_mandant],
        outputs=[history_table],
    )

    # ── Event: Feedback-Statistiken laden ─────────────────────
    def load_feedback_stats(mandant_id: str) -> str:
        mid = mandant_id.strip() or None
        return format_feedback_report(_feedback_store, mid)

    stats_btn.click(
        fn=load_feedback_stats,
        inputs=[stats_mandant],
        outputs=[stats_output],
    )

    # ── Events: Einzelne Charts on-demand ─────────────────────
    all_chart_outputs = [
        chart_score_dist, chart_flag_freq, chart_monthly_pnl, chart_top_acc,
        chart_ertrag_aufw, chart_heatmap, chart_scatter, chart_treemap,
        chart_zeitreihe, chart_sh_balance,
    ]
    all_charts_btn.click(fn=generate_all_charts, inputs=[], outputs=all_chart_outputs)

    btn_score_dist.click(fn=generate_score_distribution, inputs=[], outputs=[chart_score_dist])
    btn_flag_freq.click(fn=generate_flag_frequency, inputs=[], outputs=[chart_flag_freq])
    btn_monthly_pnl.click(fn=generate_monthly_pnl, inputs=[], outputs=[chart_monthly_pnl])
    btn_top_acc.click(fn=generate_top_accounts, inputs=[], outputs=[chart_top_acc])
    btn_ertrag_aufw.click(fn=generate_ertrag_aufwand, inputs=[], outputs=[chart_ertrag_aufw])
    btn_heatmap.click(fn=generate_heatmap, inputs=[], outputs=[chart_heatmap])
    btn_scatter.click(fn=generate_betrag_vs_score, inputs=[], outputs=[chart_scatter])
    btn_treemap.click(fn=generate_treemap, inputs=[], outputs=[chart_treemap])
    btn_zeitreihe.click(fn=generate_zeitreihe, inputs=[], outputs=[chart_zeitreihe])
    btn_sh_balance.click(fn=generate_sh_balance, inputs=[], outputs=[chart_sh_balance])

    # ── Events: 3D-Preset ────────────────────────────────────
    btn_3d_landscape.click(fn=_generate_3d_landscape, inputs=[], outputs=[chart_3d_landscape])

    # ── Events: Dynamischer Chart-Builder ─────────────────────
    dyn_load_btn.click(
        fn=_populate_dynamic_dropdowns,
        inputs=[],
        outputs=[dyn_x, dyn_y, dyn_z, dyn_color, dyn_size],
    )
    dyn_chart_type.change(
        fn=_toggle_z_axis,
        inputs=[dyn_chart_type],
        outputs=[dyn_z],
    )
    dyn_build_btn.click(
        fn=_build_dynamic_chart,
        inputs=[dyn_chart_type, dyn_x, dyn_y, dyn_z, dyn_color, dyn_size],
        outputs=[dyn_chart_output, dyn_quality_warning],
    )
    dyn_export_btn.click(
        fn=_export_chart_html,
        inputs=[],
        outputs=[dyn_export_file],
    )

    gr.Markdown(
        "---\n"
        "**14 Tests:** Z-Score | IQR | Konto-Betrag | Near-Duplicate | "
        "Doppelte Belegnummer | Beleg-Kreditor-Duplikat | Storno | "
        "Leerer Buchungstext | Rechnungsdatum-Periode | Buchungstext-Periode | "
        "Neuer Kreditor | "
        "Monats-Entwicklung | Fehlende Monatsbuchung | Isolation-Anomalie"
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
