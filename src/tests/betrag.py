"""
Buchungs-Anomalie Pre-Filter — Betrags-Tests

Tests:
    BETRAG_ZSCORE          — Betrag > Z-Score-Schwelle (NUR Ertrags- + Aufwandskonten)
    BETRAG_IQR             — Betrag > IQR-Fence (NUR Ertrags- + Aufwandskonten)
    KONTO_BETRAG_ANOMALIE  — Betrag weicht > konto_betrag_sigma Standardabweichungen vom Konto-Durchschnitt ab

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
    required_columns = ["_abs", "_betrag", "konto_soll", "_is_storno", "_kontoklasse"]

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
    required_columns = ["_abs", "_betrag", "konto_soll", "_is_storno", "_kontoklasse"]

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
    required_columns = ["_abs", "_betrag", "konto_soll", "_is_storno", "_kontoklasse"]

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

        # Z-Score pro Konto (statt %-Abweichung vom Mittelwert)
        konto_stats = (
            work.groupby("konto_soll", observed=True)["_abs"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "konto_mean", "std": "konto_std", "count": "konto_count"})
        )
        konto_stats = konto_stats[
            konto_stats["konto_count"] >= config.konto_min_buchungen
        ].copy()

        if konto_stats.empty:
            return 0

        # std=NaN (nur 1 Buchung) → 0 → kein Outlier möglich
        konto_stats["konto_std"] = konto_stats["konto_std"].fillna(0)
        konto_stats["thresh_upper"] = (
            konto_stats["konto_mean"] + config.konto_betrag_sigma * konto_stats["konto_std"]
        )
        konto_stats["thresh_lower"] = (
            (konto_stats["konto_mean"] - config.konto_betrag_sigma * konto_stats["konto_std"])
            .clip(lower=0)
        )

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
