"""
Buchungs-Anomalie Pre-Filter — Kreditor-Tests

Tests:
    NEUER_KREDITOR_HOCH — Neuer Kreditor (≤N Buchungen) mit hohem Betrag
    VELOCITY_ANOMALIE   — Ungewöhnlich viele Buchungen eines Kreditors pro Monat
"""

from __future__ import annotations

import pandas as pd

from src.config import AnalysisConfig
from src.tests.base import AnomalyTest, EngineStats


class NeuerKreditorHoch(AnomalyTest):
    name = "NEUER_KREDITOR_HOCH"
    weight = 2.5
    critical = True
    required_columns = ["_abs", "kreditor"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        has_kred = df["kreditor"].astype(str).str.strip() != ""
        kred_cnt = df.loc[has_kred, "kreditor"].value_counts()
        schwelle = (
            stats.b_mean + config.new_kreditor_amount_sigma * stats.b_std
            if stats.b_std > 0 else float("inf")
        )
        mapped = df["kreditor"].map(kred_cnt).astype(float).fillna(0).astype(int)
        mask   = has_kred & (mapped <= config.new_kreditor_max_bookings) & (df["_abs"] > schwelle)
        return self._flag(df, mask)


class VelocityAnomalie(AnomalyTest):
    name = "VELOCITY_ANOMALIE"
    weight = 1.5
    critical = False
    required_columns = ["_datum", "kreditor", "erfasser"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        # Guard: Ohne echte Erfasser-Daten ist der Test sinnlos
        erfasser = df["erfasser"].astype(str).str.strip()
        if (erfasser == "").all() or erfasser.nunique() < 2:
            return 0

        has_kred_date = (
            df["kreditor"].astype(str).str.strip() != ""
        ) & df["_datum"].notna()
        subset = df.loc[has_kred_date].copy()

        if subset.empty or len(subset) < 10:
            return 0

        subset["_ym"] = subset["_datum"].dt.to_period("M").astype(str)
        kred_month = (
            subset.groupby(["kreditor", "_ym"], observed=True).size().reset_index(name="cnt")
        )
        kred_stats = (
            kred_month.groupby("kreditor", observed=True)["cnt"]
            .agg(km_mean="mean", km_std="std", n_months="count")
            .reset_index()
        )
        kred_stats = kred_stats[
            kred_stats["n_months"] >= config.velocity_min_months
        ].copy()
        kred_stats["km_std"] = kred_stats["km_std"].fillna(kred_stats["km_mean"])
        kred_stats["threshold"] = (kred_stats["km_mean"] * 3).combine(
            kred_stats["km_mean"] + 2 * kred_stats["km_std"], max
        )

        spikes = (
            kred_month
            .merge(kred_stats[["kreditor", "threshold"]], on="kreditor", how="inner")
            .query("cnt >= threshold")
        )
        if not spikes.empty:
            spike_set      = set(zip(spikes["kreditor"], spikes["_ym"]))
            subset["_key"] = list(zip(subset["kreditor"], subset["_ym"]))
            flagged_sub    = subset[subset["_key"].isin(spike_set)]
            df.loc[flagged_sub.index, f"flag_{self.name}"] = True

        return int(df[f"flag_{self.name}"].sum())


def get_tests() -> list[AnomalyTest]:
    return [NeuerKreditorHoch(), VelocityAnomalie()]
