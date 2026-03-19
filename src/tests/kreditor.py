"""
Buchungs-Anomalie Pre-Filter — Kreditor-Tests

Tests:
    NEUER_KREDITOR_HOCH — Neuer Kreditor (≤N Buchungen) mit hohem Betrag

"""

from __future__ import annotations

import pandas as pd

from src.config import AnalysisConfig
from src.tests.base import AnomalyTest, EngineStats


class NeuerKreditorHoch(AnomalyTest):
    name = "NEUER_KREDITOR_HOCH"
    weight = 2.5
    critical = True
    required_columns = ["_abs", "kreditor", "_kreditor_canonical", "_is_storno"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        # Stornos ausschließen
        is_storno = df.get("_is_storno", pd.Series(False, index=df.index))
        # Kanonischer Kreditorname (via Clustering) wenn vorhanden
        kred_col = "_kreditor_canonical" if "_kreditor_canonical" in df.columns else "kreditor"
        kred_vals = df[kred_col].astype(str).str.strip()
        has_kred = (kred_vals != "") & (~is_storno)
        kred_cnt = kred_vals.loc[has_kred].value_counts()
        schwelle = (
            stats.b_mean + config.new_kreditor_amount_sigma * stats.b_std
            if stats.b_std > 0 else float("inf")
        )
        mapped = kred_vals.map(kred_cnt).astype(float).fillna(0).astype(int)
        mask   = has_kred & (mapped <= config.new_kreditor_max_bookings) & (df["_abs"] > schwelle)
        return self._flag(df, mask)


def get_tests() -> list[AnomalyTest]:
    return [NeuerKreditorHoch()]
