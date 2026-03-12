"""
Buchungs-Anomalie Pre-Filter — Betrags-Tests

Tests:
    BETRAG_ZSCORE          — Betrag > Z-Score-Schwelle
    BETRAG_IQR             — Betrag > IQR-Fence
    KONTO_BETRAG_ANOMALIE  — Betrag ist Ausreißer auf Kontoebene
"""

from __future__ import annotations

import pandas as pd

from src.config import AnalysisConfig
from src.tests.base import AnomalyTest, EngineStats


class BetragZscore(AnomalyTest):
    name = "BETRAG_ZSCORE"
    weight = 2.0
    critical = True

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        if stats.b_std > 0:
            z    = (df["_abs"] - stats.b_mean) / stats.b_std
            mask = (df["_abs"] > 0) & (z > config.zscore_threshold)
        else:
            mask = pd.Series(False, index=df.index)
        return self._flag(df, mask)


class BetragIqr(AnomalyTest):
    name = "BETRAG_IQR"
    weight = 1.5
    critical = False

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        if stats.b_iqr > 0:
            mask = df["_abs"] > stats.b_fence
        else:
            mask = pd.Series(False, index=df.index)
        return self._flag(df, mask)


class KontoBetragAnomalie(AnomalyTest):
    name = "KONTO_BETRAG_ANOMALIE"
    weight = 2.0
    critical = True

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        has_konto = df["konto_soll"].astype(str).str.strip() != ""

        konto_stats = (
            df.loc[has_konto & (df["_abs"] > 0)]
            .groupby("konto_soll")["_abs"]
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
