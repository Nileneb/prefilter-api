"""
Buchungs-Anomalie Pre-Filter — Plotly-Visualisierungen

Erzeugt Überblicks- und Detail-Charts aus dem Engine-DataFrame.

Public API:
    ChartBuilder(df, result) → .all_charts() → dict[str, go.Figure]
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.accounting import kontoklasse

_CHART_MAX_ROWS = 500_000


def _downsample_for_charts(df: pd.DataFrame, max_rows: int = _CHART_MAX_ROWS) -> pd.DataFrame:
    """Sampelt den DataFrame herunter wenn er zu groß ist.

    Strategie: Alle geflaggten Zeilen behalten + zufällige Stichprobe
    des Rests bis max_rows erreicht sind.
    """
    if len(df) <= max_rows:
        return df
    flagged = df[df["_score"] > 0]
    unflagged = df[df["_score"] <= 0]
    remaining = max(0, max_rows - len(flagged))
    if remaining > 0 and len(unflagged) > remaining:
        unflagged = unflagged.sample(n=remaining, random_state=42)
    return pd.concat([flagged, unflagged], ignore_index=True)


class ChartBuilder:
    """Baut Plotly-Charts aus dem DataFrame nach dem Engine-Run."""

    def __init__(self, df: pd.DataFrame, result: dict):
        self.df = _downsample_for_charts(df)
        self.result = result

    # ── Phase 1: Überblicks-Dashboards ────────────────────────

    def score_distribution(self) -> go.Figure:
        """Histogramm der Anomaly-Scores aller Buchungen."""
        scores = self.df["_score"]
        threshold = self.result["statistics"].get("avg_score", 2.0)
        fig = px.histogram(
            scores, nbins=50,
            labels={"value": "Anomaly Score", "count": "Anzahl"},
            title="Anomaly-Score-Verteilung",
        )
        fig.update_layout(showlegend=False, bargap=0.05)
        fig.add_vline(
            x=threshold, line_dash="dash", line_color="red",
            annotation_text=f"Ø {threshold:.2f}",
        )
        return fig

    def flag_frequency(self) -> go.Figure:
        """Horizontales Balkendiagramm der Flag-Häufigkeiten."""
        counts = self.result["statistics"].get("flag_counts", {})
        # Nur Flags mit > 0 Treffern anzeigen
        filtered = {k: v for k, v in counts.items() if v > 0}
        if not filtered:
            return _empty_figure("Keine Flags ausgelöst")
        names = list(filtered.keys())
        values = list(filtered.values())
        # Sortiert nach Häufigkeit
        pairs = sorted(zip(names, values), key=lambda x: x[1])
        names, values = zip(*pairs)
        fig = px.bar(
            x=list(values), y=list(names), orientation="h",
            labels={"x": "Anzahl", "y": "Test"},
            title="Flag-Häufigkeit",
        )
        fig.update_layout(showlegend=False)
        return fig

    def monthly_pnl(self) -> go.Figure:
        """Monatliche Betrags-Entwicklung nach Kontoklasse (Linienchart)."""
        df = self.df.copy()
        if "_datum" not in df.columns:
            return _empty_figure("Keine Datums-/Betragsdaten")
        df = df[df["_datum"].notna()].copy()
        if df.empty:
            return _empty_figure("Keine gültigen Datumswerte")

        # _betrag_signed nutzen wenn vorhanden, sonst Fallback auf _betrag
        betrag_col = "_betrag_signed" if "_betrag_signed" in df.columns else "_betrag"
        use_abs_fallback = betrag_col == "_betrag"

        df["_monat"] = df["_datum"].dt.to_period("M").dt.to_timestamp()
        df["_kontoklasse"] = kontoklasse(df["konto_soll"]) if "_kontoklasse" not in df.columns else df["_kontoklasse"]

        agg = (
            df.groupby(["_monat", "_kontoklasse"], observed=True)[betrag_col]
            .sum()
            .reset_index()
        )
        fig = px.line(
            agg, x="_monat", y=betrag_col, color="_kontoklasse",
            labels={"_monat": "Monat", betrag_col: "Summe Betrag", "_kontoklasse": "Kontoklasse"},
            title="Monatliche Betrags-Entwicklung nach Kontoklasse",
        )
        if use_abs_fallback:
            fig.add_annotation(
                text="⚠️ Soll/Haben fehlt — nur Absolutbeträge",
                xref="paper", yref="paper", x=0.5, y=1.05,
                showarrow=False, font=dict(size=11, color="orange"),
            )
        return fig

    def top_accounts(self, n: int = 10) -> go.Figure:
        """Top-N Konten nach durchschnittlichem Anomaly-Score."""
        df = self.df
        if "konto_soll" not in df.columns:
            return _empty_figure("Keine Kontodaten")

        konto_scores = (
            df[df["_score"] > 0]
            .groupby("konto_soll", observed=True)["_score"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_score", "count": "n_flagged"})
            .sort_values("avg_score", ascending=False)
            .head(n)
            .reset_index()
        )
        if konto_scores.empty:
            return _empty_figure("Keine Konten mit Anomalie-Score")

        konto_scores = konto_scores.sort_values("avg_score")
        fig = px.bar(
            konto_scores, x="avg_score", y="konto_soll", orientation="h",
            text="n_flagged",
            labels={"avg_score": "Ø Anomaly Score", "konto_soll": "Konto", "n_flagged": "Flagged"},
            title=f"Top-{n} Konten nach Anomalie-Score",
        )
        fig.update_traces(texttemplate="%{text} Buchungen", textposition="outside")
        fig.update_layout(showlegend=False)
        return fig

    def ertrag_aufwand_monthly(self) -> go.Figure:
        """Ertrag vs. Aufwand pro Monat (gruppiertes Balkendiagramm)."""
        df = self.df.copy()
        if "_datum" not in df.columns:
            return _empty_figure("Keine Datums-/Betragsdaten")
        df = df[df["_datum"].notna()].copy()
        if df.empty:
            return _empty_figure("Keine gültigen Datumswerte")

        betrag_col = "_betrag_signed" if "_betrag_signed" in df.columns else "_betrag"

        df["_monat"] = df["_datum"].dt.to_period("M").dt.to_timestamp()
        df["_kontoklasse"] = kontoklasse(df["konto_soll"]) if "_kontoklasse" not in df.columns else df["_kontoklasse"]

        ea = df[df["_kontoklasse"].isin(["Ertrag", "Aufwand"])]
        if ea.empty:
            return _empty_figure("Keine Ertrags-/Aufwandskonten")

        agg = (
            ea.groupby(["_monat", "_kontoklasse"], observed=True)[betrag_col]
            .sum()
            .reset_index()
        )
        fig = px.bar(
            agg, x="_monat", y=betrag_col, color="_kontoklasse",
            barmode="group",
            labels={"_monat": "Monat", betrag_col: "Summe Betrag", "_kontoklasse": "Kontoklasse"},
            title="Ertrag vs. Aufwand pro Monat",
        )
        return fig

    def volume_heatmap(self) -> go.Figure:
        """Buchungsvolumen-Heatmap (Wochentag × Monat)."""
        df = self.df.copy()
        if "_datum" not in df.columns:
            return _empty_figure("Keine Datumsdaten")
        df = df[df["_datum"].notna()].copy()
        if df.empty:
            return _empty_figure("Keine gültigen Datumswerte")

        df["_wochentag"] = df["_datum"].dt.day_name()
        df["_monat"] = df["_datum"].dt.to_period("M").astype(str)

        pivot = df.groupby(["_wochentag", "_monat"]).size().reset_index(name="count")

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_labels = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
        pivot["_wochentag"] = pd.Categorical(pivot["_wochentag"], categories=day_order, ordered=True)
        pivot = pivot.sort_values("_wochentag")

        fig = px.density_heatmap(
            pivot, x="_monat", y="_wochentag", z="count",
            labels={"_monat": "Monat", "_wochentag": "Wochentag", "count": "Buchungen"},
            title="Buchungsvolumen (Wochentag × Monat)",
            category_orders={"_wochentag": day_order},
        )
        fig.update_yaxes(ticktext=day_labels, tickvals=day_order)
        return fig

    # ── Phase 2: Anomalie-Detail-Charts ───────────────────────

    def betrag_vs_score(self) -> go.Figure:
        """Scatter: Betrag vs. Anomaly-Score, Farbe = höchstes Flag."""
        df = self.df
        flagged = df[df["_score"] > 0].copy()
        if flagged.empty:
            return _empty_figure("Keine geflaggten Buchungen")

        # Bestimme das gewichtigste Flag pro Buchung
        flag_cols = [c for c in df.columns if c.startswith("flag_")]
        if flag_cols:
            flagged["_top_flag"] = flagged[flag_cols].idxmax(axis=1).str.replace("flag_", "", regex=False)
        else:
            flagged["_top_flag"] = "unbekannt"

        fig = px.scatter(
            flagged, x="_abs", y="_score", color="_top_flag",
            labels={"_abs": "Betrag (abs.)", "_score": "Anomaly Score", "_top_flag": "Top Flag"},
            title="Betrag vs. Anomaly Score",
            opacity=0.6,
        )
        fig.update_layout(legend_title_text="Flag")
        return fig

    def kreditor_treemap(self) -> go.Figure:
        """Treemap: Volumen pro Kreditor, Farbe = Anomaly-Score."""
        df = self.df
        if "kreditor" not in df.columns:
            return _empty_figure("Keine Kreditordaten")

        kred = (
            df[df["kreditor"].astype(str).str.strip() != ""]
            .groupby("kreditor", observed=True)
            .agg(summe=("_abs", "sum"), avg_score=("_score", "mean"), count=("_abs", "count"))
            .reset_index()
        )
        kred = kred[kred["count"] >= 2].nlargest(30, "summe")
        if kred.empty:
            return _empty_figure("Keine Kreditoren mit >= 2 Buchungen")

        fig = px.treemap(
            kred, path=["kreditor"], values="summe", color="avg_score",
            color_continuous_scale="RdYlGn_r",
            labels={"summe": "Gesamtbetrag", "avg_score": "Ø Score"},
            title="Top-30 Kreditoren (Volumen & Anomalie-Score)",
        )
        return fig

    def zeitreihe_konto(self, konto: str | None = None) -> go.Figure:
        """Zeitreihe für ein einzelnes Konto mit Anomalie-Markierungen."""
        df = self.df
        if konto is None:
            # Wähle das Konto mit dem höchsten avg Score
            top = (
                df[df["_score"] > 0]
                .groupby("konto_soll", observed=True)["_score"]
                .mean()
                .idxmax()
            )
            konto = str(top) if pd.notna(top) else None
        if konto is None:
            return _empty_figure("Kein Konto mit Anomalien")

        sub = df[df["konto_soll"].astype(str) == str(konto)].copy()
        if sub.empty or "_datum" not in sub.columns:
            return _empty_figure(f"Keine Daten für Konto {konto}")

        sub = sub[sub["_datum"].notna()].sort_values("_datum")
        fig = px.line(
            sub, x="_datum", y="_betrag",
            title=f"Zeitreihe Konto {konto}",
            labels={"_datum": "Datum", "_betrag": "Betrag"},
        )
        anomalies = sub[sub["_score"] > 0]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies["_datum"], y=anomalies["_betrag"],
                mode="markers", marker=dict(color="red", size=8, symbol="x"),
                name="Anomalie",
            ))
        return fig

    def soll_haben_balance(self) -> go.Figure:
        """Soll/Haben-Balance pro Top-Konto (divergierendes Balkendiagramm)."""
        df = self.df
        if "soll_haben" not in df.columns or "konto_soll" not in df.columns:
            return _empty_figure("Soll/Haben-Spalte nicht verfügbar")

        sh = df["soll_haben"].astype(str).str.strip().str.upper()
        has_sh = sh.isin(["S", "SOLL", "H", "HABEN"])
        if not has_sh.any():
            return _empty_figure("Keine gültigen Soll/Haben-Werte")

        work = df[has_sh].copy()
        work["_sh"] = sh[has_sh]
        work["_is_soll"] = work["_sh"].isin(["S", "SOLL"])
        betrag_col = "_betrag_signed" if "_betrag_signed" in work.columns else "_abs"

        soll = (
            work[work["_is_soll"]]
            .groupby("konto_soll", observed=True)[betrag_col]
            .sum()
            .rename("Soll")
        )
        haben = (
            work[~work["_is_soll"]]
            .groupby("konto_soll", observed=True)[betrag_col]
            .sum()
            .rename("Haben")
        )

        balance = pd.concat([soll, haben], axis=1).fillna(0)
        balance["diff"] = (balance["Soll"] - balance["Haben"]).abs()
        top = balance.nlargest(15, "diff").reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top["konto_soll"].astype(str), x=-top["Haben"],
            name="Haben", orientation="h", marker_color="#2ca02c",
        ))
        fig.add_trace(go.Bar(
            y=top["konto_soll"].astype(str), x=top["Soll"],
            name="Soll", orientation="h", marker_color="#d62728",
        ))
        fig.update_layout(
            title="Soll/Haben-Balance (Top-15 Konten nach Differenz)",
            barmode="relative",
            xaxis_title="Betrag",
            yaxis_title="Konto",
        )
        return fig

    # ── Aggregator ────────────────────────────────────────────

    def all_charts(self) -> dict[str, go.Figure]:
        """Gibt alle Charts als Dict zurück. 3D-Charts NICHT enthalten (Performance)."""
        charts = {}
        methods = [
            ("score_distribution", self.score_distribution),
            ("flag_frequency", self.flag_frequency),
            ("monthly_pnl", self.monthly_pnl),
            ("top_accounts", self.top_accounts),
            ("ertrag_aufwand_monthly", self.ertrag_aufwand_monthly),
            ("volume_heatmap", self.volume_heatmap),
            ("betrag_vs_score", self.betrag_vs_score),
            ("kreditor_treemap", self.kreditor_treemap),
            ("zeitreihe_konto", self.zeitreihe_konto),
            ("soll_haben_balance", self.soll_haben_balance),
        ]
        for name, method in methods:
            try:
                charts[name] = method()
            except Exception:
                charts[name] = _empty_figure(f"Fehler bei {name}")
        return charts

    # ── 3D-Preset-Charts (on-demand, NICHT in all_charts) ────

    def anomaly_landscape_3d(self) -> go.Figure:
        """3D-Scatter: Anomalie-Landschaft (Betrag × Konto × Zeit)."""
        df = self.df.copy()
        if "_datum" not in df.columns or "_abs" not in df.columns:
            return _empty_figure("Keine Betrags-/Datumsdaten")

        df = df[df["_datum"].notna() & df["_abs"].notna()].copy()
        if df.empty:
            return _empty_figure("Keine gültigen Daten")

        df["_datum_num"] = (df["_datum"] - df["_datum"].min()).dt.days
        df["_konto_num"] = pd.to_numeric(df["konto_soll"].astype(str), errors="coerce")
        df = df[df["_konto_num"].notna()]
        if df.empty:
            return _empty_figure("Keine numerischen Kontonummern")

        fig = px.scatter_3d(
            df,
            x="_abs",
            y="_konto_num",
            z="_datum_num",
            color="_score",
            color_continuous_scale="RdYlGn_r",
            size="_abs",
            size_max=15,
            hover_data=["buchungstext", "belegnummer", "kreditor", "_score"],
            opacity=0.6,
            title="Anomalie-Landschaft 3D (Betrag × Konto × Zeit)",
            labels={
                "_abs": "Betrag (€)",
                "_konto_num": "Kontonummer",
                "_datum_num": "Tage seit Beginn",
                "_score": "Anomaly Score",
            },
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="Betrag (€)",
                yaxis_title="Kontonummer",
                zaxis_title="Zeit (Tage)",
            ),
        )
        return fig


def _empty_figure(message: str) -> go.Figure:
    """Erzeugt eine leere Plotly-Figure mit Hinweistext."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="white",
    )
    return fig


# ══════════════════════════════════════════════════════════════
# DYNAMISCHER CHART-BUILDER (Phase 1–3)
# ══════════════════════════════════════════════════════════════

# Interne Spalten die im Dynamic Builder sichtbar sein sollen
_SHOW_INTERNAL = {"_abs", "_betrag", "_betrag_signed", "_score", "_datum", "_kontoklasse"}

DYNAMIC_CHART_TYPES: dict[str, Any] = {
    "Scatter": px.scatter,
    "Bar": px.bar,
    "Histogram": px.histogram,
    "Box": px.box,
    "Violin": px.violin,
    "Heatmap (Dichte)": px.density_heatmap,
    "Linie / Zeitreihe": px.line,
    "Scatter 3D": px.scatter_3d,
}


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Kategorisiert DataFrame-Spalten für den dynamischen Chart-Builder."""
    numeric = df.select_dtypes("number").columns.tolist()
    categorical = df.select_dtypes(["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes("datetime").columns.tolist()
    boolean = df.select_dtypes("bool").columns.tolist()

    def _filter(cols: list[str]) -> list[str]:
        return [c for c in cols if not c.startswith("_") or c in _SHOW_INTERNAL]

    return {
        "numeric": _filter(numeric),
        "categorical": _filter(categorical) + _filter(boolean),
        "datetime": _filter(datetime_cols),
        "all": _filter(numeric + categorical + datetime_cols + boolean),
    }


def check_column_quality(df: pd.DataFrame, col: str) -> str:
    """Gibt eine Warnung zurück wenn die Spalte >50% NaN hat."""
    if col not in df.columns:
        return f"⚠️ Spalte '{col}' nicht vorhanden"
    pct_null = df[col].isna().mean() * 100
    if pct_null > 50:
        return f"⚠️ {col}: {pct_null:.0f}% fehlende Werte — Chart möglicherweise nicht aussagekräftig"
    return ""


def _enrich_for_dynamic(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt abgeleitete Spalten hinzu die für dynamische Charts nützlich sind."""
    df = df.copy()
    if "_datum" in df.columns and df["_datum"].notna().any():
        df["Wochentag"] = df["_datum"].dt.day_name()
        df["Monat"] = df["_datum"].dt.to_period("M").astype(str)
        df["Quartal"] = df["_datum"].dt.to_period("Q").astype(str)
        df["Kalenderwoche"] = df["_datum"].dt.isocalendar().week.astype(int)

    if "_score" in df.columns:
        df["Risiko-Kategorie"] = pd.cut(
            df["_score"],
            bins=[-0.01, 0, 2, 4, float("inf")],
            labels=["Unauffällig", "Niedrig", "Mittel", "Hoch"],
        )

    if "_abs" in df.columns:
        df["Betrags-Klasse"] = pd.cut(
            df["_abs"],
            bins=[0, 100, 1000, 10000, 100000, float("inf")],
            labels=["<100€", "100-1k€", "1k-10k€", "10k-100k€", ">100k€"],
        )

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    if flag_cols:
        df["Anzahl_Flags"] = df[flag_cols].sum(axis=1)

    return df


class DynamicChartBuilder:
    """Erstellt Charts dynamisch basierend auf User-Auswahl."""

    def __init__(self, df: pd.DataFrame):
        self.df = _enrich_for_dynamic(_downsample_for_charts(df))
        self.cols = classify_columns(self.df)

    def build(
        self,
        chart_type: str,
        x: str,
        y: str | None = None,
        z: str | None = None,
        color: str | None = None,
        size: str | None = None,
    ) -> go.Figure:
        if chart_type not in DYNAMIC_CHART_TYPES:
            return _empty_figure(f"Unbekannter Diagrammtyp: {chart_type}")

        if not x or x not in self.df.columns:
            return _empty_figure(f"X-Achse '{x}' nicht verfügbar")

        chart_fn = DYNAMIC_CHART_TYPES[chart_type]
        kwargs: dict[str, Any] = {"data_frame": self.df, "x": x}

        if y and chart_type != "Histogram":
            if y not in self.df.columns:
                return _empty_figure(f"Y-Achse '{y}' nicht verfügbar")
            kwargs["y"] = y

        if z and "3D" in chart_type:
            if z not in self.df.columns:
                return _empty_figure(f"Z-Achse '{z}' nicht verfügbar")
            kwargs["z"] = z

        if color and color != "(keine)" and color in self.df.columns:
            kwargs["color"] = color

        if size and size != "(keine)" and "Scatter" in chart_type and size in self.df.columns:
            kwargs["size"] = size
            kwargs["size_max"] = 20

        hover_cols = [c for c in ["buchungstext", "belegnummer", "kreditor", "_score"]
                      if c in self.df.columns and c not in (x, y, z, color, size)]
        if hover_cols:
            kwargs["hover_data"] = hover_cols[:4]

        if "Scatter" in chart_type:
            kwargs["opacity"] = 0.7

        try:
            fig = chart_fn(**kwargs)
            title = f"{chart_type}: {x}"
            if y and chart_type != "Histogram":
                title += f" × {y}"
            if z and "3D" in chart_type:
                title += f" × {z}"
            fig.update_layout(title=title, template="plotly_white")
            return fig
        except Exception as e:
            return _empty_figure(f"Fehler: {e}")
