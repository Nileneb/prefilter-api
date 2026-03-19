"""
Buchungs-Anomalie Pre-Filter — Betrags-Tests

Tests:
    BETRAG_ZSCORE          — Betrag > Z-Score-Schwelle (NUR Ertrags- + Aufwandskonten)
    BETRAG_IQR             — Betrag > IQR-Fence (NUR Ertrags- + Aufwandskonten)
    KONTO_BETRAG_ANOMALIE  — Betrag weicht > X% vom Konto-Durchschnitt ab (5% Ertrag / 20% Aufwand)

NUR Ertrags- (40000–59999) und Aufwandskonten (60000–79999) werden analysiert.
Bestandskonten (0–39999) und Kostenrechnungskonten (≥80000) sind ausgeschlossen.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.accounting import kontoklasse
from src.config import AnalysisConfig
from src.tests.base import AnomalyTest, EngineStats

# Nur Ertrags- und Aufwandskonten werden analysiert
_GUV_KLASSEN = {"Ertrag", "Aufwand"}


class BetragZscore(AnomalyTest):
    name = "BETRAG_ZSCORE"
    weight = 2.0
    critical = True
    required_columns = ["_abs", "_betrag", "konto_soll"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        # Stornos aus Berechnung ausschließen
        is_storno = df.get("_is_storno", pd.Series(False, index=df.index))
        has_val = (df["_abs"] > 0) & (~is_storno)
        if not has_val.any():
            return self._flag(df, pd.Series(False, index=df.index))

        klasse = df["_kontoklasse"] if "_kontoklasse" in df.columns else kontoklasse(df["konto_soll"])
        mask = pd.Series(False, index=df.index)

        # NUR Ertrag + Aufwand (Bestand + Kostenrechnung ausgeschlossen)
        for kl in klasse[has_val].unique():
            if kl not in _GUV_KLASSEN:
                continue
            sel = has_val & (klasse == kl)
            vals = df.loc[sel, "_abs"]
            if len(vals) < 2:
                continue
            mean = vals.mean()
            std = vals.std()
            if std > 0:
                z = (df["_abs"] - mean) / std
                mask = mask | (sel & (z > config.zscore_threshold))

        return self._flag(df, mask)


class BetragIqr(AnomalyTest):
    name = "BETRAG_IQR"
    weight = 1.5
    critical = False
    required_columns = ["_abs", "_betrag", "konto_soll"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        # Stornos aus Berechnung ausschließen
        is_storno = df.get("_is_storno", pd.Series(False, index=df.index))
        has_val = (df["_abs"] > 0) & (~is_storno)
        if not has_val.any():
            return self._flag(df, pd.Series(False, index=df.index))

        klasse = df["_kontoklasse"] if "_kontoklasse" in df.columns else kontoklasse(df["konto_soll"])
        mask = pd.Series(False, index=df.index)

        # NUR Ertrag + Aufwand (Bestand + Kostenrechnung ausgeschlossen)
        for kl in klasse[has_val].unique():
            if kl not in _GUV_KLASSEN:
                continue
            sel = has_val & (klasse == kl)
            vals = df.loc[sel, "_abs"]
            if len(vals) < 4:
                continue
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                fence = q3 + config.iqr_factor * iqr
                mask = mask | (sel & (df["_abs"] > fence) & (df["_abs"] > config.iqr_min_betrag))

        return self._flag(df, mask)


class KontoBetragAnomalie(AnomalyTest):
    name = "KONTO_BETRAG_ANOMALIE"
    weight = 2.0
    critical = True
    required_columns = ["_abs", "_betrag", "konto_soll"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        # Stornos aus Berechnung ausschließen
        is_storno = df.get("_is_storno", pd.Series(False, index=df.index))
        has_konto = df["konto_soll"].astype(str).str.strip() != ""

        # NUR Ertrag + Aufwand
        kl = df["_kontoklasse"] if "_kontoklasse" in df.columns else kontoklasse(df["konto_soll"])
        pnl_mask = kl.isin(_GUV_KLASSEN)
        work_mask = has_konto & (df["_abs"] > 0) & (~is_storno) & pnl_mask

        if not work_mask.any():
            return 0

        work = df.loc[work_mask]
        konto_stats = (
            work.groupby("konto_soll", observed=True)["_abs"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "konto_mean", "count": "konto_count"})
        )
        konto_stats = konto_stats[
            konto_stats["konto_count"] >= config.konto_min_buchungen
        ].copy()

        if konto_stats.empty:
            return 0

        # Kontoklasse pro Konto
        konto_stats["_kl"] = kl[work_mask].groupby(
            work["konto_soll"], observed=True
        ).first()

        # Differenzierte %-Abweichungsgrenzen (5% Ertrag / 20% Aufwand)
        konto_stats["max_pct"] = konto_stats["_kl"].map({
            "Ertrag": config.ertrag_abweichung_pct,
            "Aufwand": config.aufwand_abweichung_pct,
        })
        konto_stats["thresh_upper"] = konto_stats["konto_mean"] * (1 + konto_stats["max_pct"])
        konto_stats["thresh_lower"] = konto_stats["konto_mean"] * (1 - konto_stats["max_pct"])

        df_tmp = df.join(
            konto_stats[["thresh_upper", "thresh_lower"]], on="konto_soll"
        )
        mask = (
            work_mask
            & (
                (df["_abs"] > df_tmp["thresh_upper"].fillna(float("inf")))
                | (df["_abs"] < df_tmp["thresh_lower"].fillna(0))
            )
        )
        return self._flag(df, mask)


def get_tests() -> list[AnomalyTest]:
    return [BetragZscore(), BetragIqr(), KontoBetragAnomalie()]
