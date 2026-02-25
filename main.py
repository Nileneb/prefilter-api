#!/usr/bin/env python3
"""
Buchungs-Anomalie Pre-Filter v1.0
Gradio Upload UI + Anomaly Engine + Langdock Webhook Push
"""

import io
import os
import re
import math
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import gradio as gr

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("prefilter")

# ── Config ───────────────────────────────────────────────────
LANGDOCK_WEBHOOK_URL = os.environ.get("LANGDOCK_WEBHOOK_URL", "")
GRADIO_USERNAME = os.environ.get("GRADIO_USERNAME", "")
GRADIO_PASSWORD = os.environ.get("GRADIO_PASSWORD", "")

WEIGHTS = {
    "BETRAG_ZSCORE":       2.0,
    "BETRAG_IQR":          1.5,
    "SELTENE_KONTIERUNG":  1.5,
    "WOCHENENDE":          1.0,
    "MONATSENDE":          0.5,
    "QUARTALSENDE":        0.5,
    "NEAR_DUPLICATE":      2.0,
    "BENFORD":             1.0,
    "RUNDER_BETRAG":       1.0,
    "ERFASSER_ANOMALIE":   1.5,
    "SPLIT_VERDACHT":      2.0,
    "BELEG_LUECKE":        1.0,
    "STORNO":              1.5,
    "NEUER_KREDITOR_HOCH": 2.5,
}

CRITICAL_FLAGS = {
    "BETRAG_ZSCORE", "NEAR_DUPLICATE", "SPLIT_VERDACHT",
    "NEUER_KREDITOR_HOCH", "STORNO",
}

OUTPUT_THRESHOLD = 1.0
MAX_OUTPUT_ROWS  = 1000

# ── Column normalisation ────────────────────────────────────
COLUMN_ALIASES = {
    "datum":        ["datum", "date", "buchungsdatum", "belegdatum"],
    "betrag":       ["betrag", "amount", "summe", "wert"],
    "konto_soll":   ["konto_soll", "kontosoll", "soll", "sollkonto", "debit"],
    "konto_haben":  ["konto_haben", "kontohaben", "haben", "habenkonto", "credit"],
    "buchungstext": ["buchungstext", "text", "beschreibung", "verwendungszweck"],
    "belegnummer":  ["belegnummer", "beleg", "belegnr", "beleg_nr", "voucher"],
    "kostenstelle": ["kostenstelle", "kst", "cost_center"],
    "kreditor":     ["kreditor", "lieferant", "vendor", "supplier", "creditor"],
    "erfasser":     ["erfasser", "user", "benutzer", "ersteller", "created_by"],
}


def _norm(name: str) -> str:
    s = name.lower().strip()
    for old, new in [("ä","ae"),("ö","oe"),("ü","ue"),("ß","ss")]:
        s = s.replace(old, new)
    s = re.sub(r"[^a-z0-9]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    normed = {_norm(c): c for c in df.columns}
    rename = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normed:
                rename[normed[alias]] = canon
                break
        else:
            for alias in aliases:
                for n, orig in normed.items():
                    if alias in n and orig not in rename:
                        rename[orig] = canon
                        break
                if canon in rename.values():
                    break
    return df.rename(columns=rename)


# ── Parsing helpers ──────────────────────────────────────────
def _parse_german_number(val) -> float:
    if pd.isna(val):
        return 0.0
    s = str(val).strip().replace("€", "").replace("$", "").replace("£", "").replace(" ", "")
    if not s:
        return 0.0
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_date(val) -> Optional[datetime]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        try: return datetime(int(m[1]), int(m[2]), int(m[3]))
        except: pass
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})", s)
    if m:
        try: return datetime(int(m[3]), int(m[2]), int(m[1]))
        except: pass
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{2})$", s)
    if m:
        try: return datetime(2000 + int(m[3]), int(m[2]), int(m[1]))
        except: pass
    try:
        return pd.to_datetime(s).to_pydatetime()
    except:
        return None


# ── File ingestion ───────────────────────────────────────────
def read_upload(filepath: str) -> pd.DataFrame:
    name = filepath.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(filepath, engine="openpyxl", dtype=str)
    if name.endswith(".xls"):
        return pd.read_excel(filepath, engine="xlrd", dtype=str)
    # CSV — auto-detect separator
    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
            if len(df.columns) >= 3:
                return df
        except:
            continue
    raise ValueError("CSV konnte nicht geparst werden — mindestens 3 Spalten erwartet.")


# ══════════════════════════════════════════════════════════════
# ANOMALY ENGINE
# ══════════════════════════════════════════════════════════════
class AnomalyEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logs: list[str] = []
        self.flag_counts: dict[str, int] = {}
        self._prepare()

    def _log(self, msg: str):
        self.logs.append(msg)
        logger.info(msg)

    def _flag(self, idx, name: str):
        w = WEIGHTS.get(name, 1.0)
        flags: list = self.df.at[idx, "_flags"]
        if name not in flags:
            flags.append(name)
            self.df.at[idx, "_score"] += w

    def _prepare(self):
        df = self.df
        for col in COLUMN_ALIASES:
            if col not in df.columns:
                df[col] = ""
        df.fillna("", inplace=True)
        df["_betrag"] = df["betrag"].apply(_parse_german_number)
        df["_abs"]    = df["_betrag"].abs()
        df["_datum"]  = df["datum"].apply(_parse_date)
        df["_score"]  = 0.0
        df["_flags"]  = [[] for _ in range(len(df))]
        self._log(f"Geladen: {len(df)} Buchungen")
        self._log(f"Spalten: {', '.join(c for c in df.columns if not c.startswith('_'))}")

    def _stats(self):
        vals = self.df.loc[self.df["_abs"] > 0, "_abs"]
        self.b_mean  = vals.mean() if len(vals) else 0
        self.b_std   = vals.std()  if len(vals) > 1 else 0
        q1 = vals.quantile(0.25) if len(vals) else 0
        q3 = vals.quantile(0.75) if len(vals) else 0
        self.b_iqr   = q3 - q1
        self.b_fence = q3 + 1.5 * self.b_iqr
        self._log(f"Beträge: n={len(vals)}, μ={self.b_mean:.0f}, σ={self.b_std:.0f}, "
                  f"Q1={q1:.0f}, Q3={q3:.0f}, IQR={self.b_iqr:.0f}, Fence={self.b_fence:.0f}")

    # ── 12 Tests ─────────────────────────────────────────────
    def _t01_zscore(self):
        c = 0
        if self.b_std > 0:
            for i, row in self.df.iterrows():
                if row["_abs"] > 0:
                    z = (row["_abs"] - self.b_mean) / self.b_std
                    if z > 2.5:
                        self._flag(i, "BETRAG_ZSCORE"); c += 1
        self.flag_counts["BETRAG_ZSCORE"] = c
        self._log(f"[01/12] BETRAG_ZSCORE: {c}")

    def _t02_iqr(self):
        c = 0
        if self.b_iqr > 0:
            for i, row in self.df.iterrows():
                if row["_abs"] > self.b_fence:
                    self._flag(i, "BETRAG_IQR"); c += 1
        self.flag_counts["BETRAG_IQR"] = c
        self._log(f"[02/12] BETRAG_IQR: {c}")

    def _t03_seltene_kontierung(self):
        df = self.df
        df["_konto_pair"] = df["konto_soll"].astype(str) + "→" + df["konto_haben"].astype(str)
        counts = df["_konto_pair"].value_counts()
        threshold = max(2, math.ceil(len(df) * 0.01))
        c = 0
        for i, row in df.iterrows():
            if counts.get(row["_konto_pair"], 0) <= threshold:
                self._flag(i, "SELTENE_KONTIERUNG"); c += 1
        self.flag_counts["SELTENE_KONTIERUNG"] = c
        self._log(f"[03/12] SELTENE_KONTIERUNG: {c}  (Schwelle: ≤{threshold})")

    def _t04_zeitlich(self):
        cWE = cME = cQE = 0
        for i, row in self.df.iterrows():
            d = row["_datum"]
            if d is None:
                continue
            if d.weekday() >= 5:
                self._flag(i, "WOCHENENDE"); cWE += 1
            last_day = (d.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            if d.day >= last_day.day - 2:
                self._flag(i, "MONATSENDE"); cME += 1
                if d.month in (3, 6, 9, 12):
                    self._flag(i, "QUARTALSENDE"); cQE += 1
        self.flag_counts["WOCHENENDE"]   = cWE
        self.flag_counts["MONATSENDE"]   = cME
        self.flag_counts["QUARTALSENDE"] = cQE
        self._log(f"[04/12] WOCHENENDE: {cWE}, MONATSENDE: {cME}, QUARTALSENDE: {cQE}")

    def _t05_near_duplicate(self):
        df = self.df
        c = 0
        groups: dict[str, list] = {}
        for i, row in df.iterrows():
            key = f"{row['_abs']}|{row['konto_soll']}|{row['konto_haben']}"
            groups.setdefault(key, []).append(i)
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    ia, ib = idxs[a], idxs[b]
                    da, db = df.at[ia, "_datum"], df.at[ib, "_datum"]
                    is_dup = False
                    if da and db:
                        if abs((da - db).days) <= 3:
                            is_dup = True
                    else:
                        is_dup = True
                    if is_dup:
                        self._flag(ia, "NEAR_DUPLICATE")
                        self._flag(ib, "NEAR_DUPLICATE")
                        c += 1
        self.flag_counts["NEAR_DUPLICATE"] = c
        self._log(f"[05/12] NEAR_DUPLICATE: {c}")

    def _t06_benford(self):
        expected = {1:.301, 2:.176, 3:.125, 4:.097, 5:.079, 6:.067, 7:.058, 8:.051, 9:.046}
        fd_counts = {d: 0 for d in range(1, 10)}
        total = 0
        for val in self.df["_abs"]:
            if val > 0:
                s = str(val).lstrip("0").lstrip(".").lstrip("0")
                if s and s[0].isdigit():
                    first = int(s[0])
                    if 1 <= first <= 9:
                        fd_counts[first] += 1; total += 1
        c = 0
        if total > 50:
            deviant = set()
            for d in range(1, 10):
                if abs(fd_counts[d]/total - expected[d]) > 0.08:
                    deviant.add(d)
            if deviant:
                self._log(f"    Benford-Abweichungen: Ziffern {deviant}")
                for i, row in self.df.iterrows():
                    if row["_abs"] > 0:
                        s = str(row["_abs"]).lstrip("0").lstrip(".").lstrip("0")
                        if s and s[0].isdigit() and int(s[0]) in deviant:
                            self._flag(i, "BENFORD"); c += 1
        self.flag_counts["BENFORD"] = c
        self._log(f"[06/12] BENFORD: {c}")

    def _t07_runder_betrag(self):
        c = 0
        for i, row in self.df.iterrows():
            a = row["_abs"]
            if (a >= 5000 and a % 1000 == 0) or (a >= 1000 and a % 500 == 0):
                self._flag(i, "RUNDER_BETRAG"); c += 1
        self.flag_counts["RUNDER_BETRAG"] = c
        self._log(f"[07/12] RUNDER_BETRAG: {c}")

    def _t08_erfasser(self):
        df = self.df
        erf_counts = df.loc[df["erfasser"] != "", "erfasser"].value_counts()
        erf_total = erf_counts.sum()
        threshold = max(3, math.ceil(erf_total * 0.03))
        c = 0
        for i, row in df.iterrows():
            e = row["erfasser"]
            if e and erf_counts.get(e, 0) <= threshold:
                self._flag(i, "ERFASSER_ANOMALIE"); c += 1
        self.flag_counts["ERFASSER_ANOMALIE"] = c
        self._log(f"[08/12] ERFASSER_ANOMALIE: {c}  (Schwelle: ≤{threshold})")

    def _t09_split(self):
        df = self.df
        c = 0
        groups: dict[str, list] = {}
        for i, row in df.iterrows():
            d = row["_datum"]
            dk = d.strftime("%Y-%m-%d") if d else "nodate"
            actor = row["kreditor"] or row["erfasser"] or "unknown"
            key = f"{dk}|{actor}|{row['konto_soll']}"
            groups.setdefault(key, []).append(i)
        for idxs in groups.values():
            if len(idxs) >= 3:
                for i in idxs:
                    self._flag(i, "SPLIT_VERDACHT"); c += 1
        self.flag_counts["SPLIT_VERDACHT"] = c
        self._log(f"[09/12] SPLIT_VERDACHT: {c}")

    def _t10_beleg_luecke(self):
        entries = []
        for i, row in self.df.iterrows():
            m = re.search(r"(\d+)", str(row["belegnummer"]))
            if m:
                entries.append((int(m.group(1)), i))
        entries.sort()
        c = 0
        if len(entries) > 10:
            for j in range(1, len(entries)):
                gap = entries[j][0] - entries[j-1][0]
                if gap > 5:
                    self._flag(entries[j-1][1], "BELEG_LUECKE")
                    self._flag(entries[j][1],   "BELEG_LUECKE")
                    c += 1
        self.flag_counts["BELEG_LUECKE"] = c
        self._log(f"[10/12] BELEG_LUECKE: {c}")

    def _t11_storno(self):
        c = 0
        patterns = ["storno", "korrektur", "rückbuchung", "rueckbuchung", "gutschrift"]
        for i, row in self.df.iterrows():
            txt = str(row["buchungstext"]).lower()
            if any(p in txt for p in patterns) or row["_betrag"] < 0:
                self._flag(i, "STORNO"); c += 1
        self.flag_counts["STORNO"] = c
        self._log(f"[11/12] STORNO: {c}")

    def _t12_neuer_kreditor_hoch(self):
        df = self.df
        kred_counts = df.loc[df["kreditor"] != "", "kreditor"].value_counts()
        schwelle = self.b_mean + 1.5 * self.b_std
        self._log(f"    Neuer-Kreditor Schwelle: ≤2 Buchungen + Betrag > {schwelle:.0f}")
        c = 0
        for i, row in df.iterrows():
            k = row["kreditor"]
            if k and kred_counts.get(k, 0) <= 2 and row["_abs"] > schwelle:
                self._flag(i, "NEUER_KREDITOR_HOCH"); c += 1
        self.flag_counts["NEUER_KREDITOR_HOCH"] = c
        self._log(f"[12/12] NEUER_KREDITOR_HOCH: {c}")

    # ── run all ──
    def run(self) -> dict:
        self._stats()
        self._t01_zscore()
        self._t02_iqr()
        self._t03_seltene_kontierung()
        self._t04_zeitlich()
        self._t05_near_duplicate()
        self._t06_benford()
        self._t07_runder_betrag()
        self._t08_erfasser()
        self._t09_split()
        self._t10_beleg_luecke()
        self._t11_storno()
        self._t12_neuer_kreditor_hoch()
        return self._export()

    def _export(self) -> dict:
        df = self.df
        mask = df["_score"] >= OUTPUT_THRESHOLD
        for i, row in df.iterrows():
            if any(f in CRITICAL_FLAGS for f in row["_flags"]):
                mask.at[i] = True

        verdaechtig = df[mask].sort_values("_score", ascending=False).head(MAX_OUTPUT_ROWS)

        out_cols = ["datum", "konto_soll", "konto_haben", "betrag",
                    "buchungstext", "belegnummer", "kostenstelle",
                    "kreditor", "erfasser"]
        rows = []
        for i, row in verdaechtig.iterrows():
            r = {c: str(row.get(c, "")) for c in out_cols}
            r["betrag"] = row["_betrag"]
            r["anomaly_score"] = round(row["_score"], 2)
            r["anomaly_flags"] = "|".join(row["_flags"])
            if row["_datum"]:
                r["datum"] = row["_datum"].strftime("%Y-%m-%d")
            rows.append(r)

        total = len(df)
        n_verd = int(mask.sum())
        avg_score = round(float(df["_score"].mean()), 2)
        total_flags = int(df["_flags"].apply(len).sum())

        top_flags = sorted(
            ((k, v) for k, v in self.flag_counts.items() if v > 0),
            key=lambda x: -x[1]
        )

        summary_lines = [
            f"ERGEBNIS: {n_verd} von {total} verdächtig ({n_verd/total*100:.1f}%)",
            f"Flags gesamt: {total_flags}, Ø Score: {avg_score}",
            f"Top-Flags: {', '.join(f'{k}:{v}' for k,v in top_flags) or 'keine'}",
        ]
        if rows:
            summary_lines.append(f"Höchster Score: {rows[0]['anomaly_score']} ({rows[0]['belegnummer']})")

        for line in summary_lines:
            self._log(line)

        return {
            "message": f"{n_verd} verdächtige Buchungen ({n_verd/total*100:.1f}%)",
            "statistics": {
                "total_input":  total,
                "total_output": len(rows),
                "filter_ratio": f"{n_verd/total*100:.1f}%",
                "avg_score":    avg_score,
                "flag_counts":  self.flag_counts,
            },
            "verdaechtige_buchungen": rows,
            "logs": self.logs,
        }


# ══════════════════════════════════════════════════════════════
# WEBHOOK PUSH
# ══════════════════════════════════════════════════════════════
def push_to_langdock(payload: dict, webhook_url: str) -> dict:
    """POST the prefilter result JSON to a Langdock webhook."""
    if not webhook_url:
        return {"error": "Keine Webhook-URL konfiguriert"}
    try:
        r = httpx.post(
            webhook_url,
            json=payload,
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        )
        return {"status": r.status_code, "response": r.text[:500]}
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════
# GRADIO UI
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
    df = _map_columns(df)

    # 3) Anomalie-Engine
    engine = AnomalyEngine(df)
    result = engine.run()

    # 4) Zusammenfassung für UI
    stats = result["statistics"]
    flag_str = "\n".join(f"  {k}: {v}" for k, v in
                         sorted(stats["flag_counts"].items(), key=lambda x: -x[1]) if v > 0)

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
            summary += (f"  {i}. {r['belegnummer']}  Score={r['anomaly_score']}  "
                        f"Betrag={r['betrag']:.2f}€\n"
                        f"     Flags: {r['anomaly_flags']}\n")

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
    """
) as app:

    gr.Markdown("# 🔍 Buchungs-Anomalie Pre-Filter", elem_classes="main-title")
    gr.Markdown(
        "Buchungsdaten hochladen (CSV / XLS / XLSX) → 12 statistische Tests → "
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
            summary_output = gr.Textbox(label="Zusammenfassung", lines=20, interactive=False)
        with gr.Tab("📊 Verdächtige Buchungen"):
            table_output = gr.Dataframe(
                label="Verdächtige Buchungen (sortiert nach Score)",
                interactive=False,
                wrap=True,
            )
        with gr.Tab("📝 Logs"):
            logs_output = gr.Textbox(label="Engine-Logs", lines=25, interactive=False)

    analyze_btn.click(
        fn=analyze_file,
        inputs=[file_input, webhook_input],
        outputs=[summary_output, logs_output, table_output],
    )

    gr.Markdown(
        "---\n"
        "**12 Tests:** Z-Score · IQR · Seltene Kontierung · Wochenende/Monats-/Quartalsende · "
        "Near-Duplicate · Benford · Runde Beträge · Erfasser-Anomalie · Split-Verdacht · "
        "Belegnummer-Lücken · Storno · Neuer Kreditor + hoher Betrag"
    )


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    auth = (GRADIO_USERNAME, GRADIO_PASSWORD) if GRADIO_USERNAME and GRADIO_PASSWORD else None
    app.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=False,
        auth=auth,
        root_path="https://ws.linn.games",
    )
