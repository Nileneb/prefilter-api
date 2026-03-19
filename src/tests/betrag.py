"""
Buchungs-Anomalie Pre-Filter — Betrags-Tests

Tests:
    BETRAG_ZSCORE          — Betrag > Z-Score-Schwelle (je Kontoklasse)
    BETRAG_IQR             — Betrag > IQR-Fence (je Kontoklasse)
    KONTO_BETRAG_ANOMALIE  — Betrag ist Ausreißer auf Kontoebene
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import AnalysisConfig
from src.tests.base import AnomalyTest, EngineStats


def _kontoklasse(konto_soll: pd.Series) -> pd.Series:
    """Bestimmt Kontoklasse aus der Kontonummer (erste Ziffer(n)).

    Ertrag: 40000–59999, Aufwand: 60000–79999, Bestand: 0–39999
    """
    num = pd.to_numeric(
        konto_soll.astype(str).str.strip().str.replace(r"\D", "", regex=True),
        errors="coerce",
    )
    klasse = pd.Series("bestand", index=konto_soll.index)
    klasse = klasse.where(~((num >= 40000) & (num < 60000)), "ertrag")
    klasse = klasse.where(~((num >= 60000) & (num < 80000)), "aufwand")
    return klasse


class BetragZscore(AnomalyTest):
    name = "BETRAG_ZSCORE"
    weight = 2.0
    critical = True
    required_columns = ["_abs", "_betrag", "konto_soll"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        has_val = df["_abs"] > 0
        if not has_val.any():
            return self._flag(df, pd.Series(False, index=df.index))

        klasse = _kontoklasse(df["konto_soll"])
        mask = pd.Series(False, index=df.index)

        for kl in klasse[has_val].unique():
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
        has_val = df["_abs"] > 0
        if not has_val.any():
            return self._flag(df, pd.Series(False, index=df.index))

        klasse = _kontoklasse(df["konto_soll"])
        mask = pd.Series(False, index=df.index)

        for kl in klasse[has_val].unique():
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
        has_konto = df["konto_soll"].astype(str).str.strip() != ""

        konto_stats = (
            df.loc[has_konto & (df["_abs"] > 0)]
            .groupby("konto_soll", observed=True)["_abs"]
            .agg(konto_mean="mean", konto_std="std", konto_count="count")
        )
        konto_stats = konto_stats[
            (konto_stats["konto_count"] >= config.konto_min_buchungen)
            & (konto_stats["konto_std"] > 0)
        ].copy()
        konto_stats["threshold"] = (
            konto_stats["konto_mean"] + config.konto_betrag_sigma * konto_stats["konto_std"]
        )

        if konto_stats.empty:
            return 0

        df_tmp = df.join(
            konto_stats["threshold"].rename("_konto_thresh"), on="konto_soll"
        )
        mask = (
            has_konto
            & (df["_abs"] > 0)
            & (df["_abs"] > df_tmp["_konto_thresh"].fillna(float("inf")))
        )
        return self._flag(df, mask)


def get_tests() -> list[AnomalyTest]:
    return [BetragZscore(), BetragIqr(), KontoBetragAnomalie()]
