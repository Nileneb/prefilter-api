"""
Buchungs-Anomalie Pre-Filter — Zeitreihen-Tests

Tests:
    MONATS_ENTWICKLUNG      — Z-Score auf monatl. GuV-Konten-Summen (pro Konto)
    FEHLENDE_MONATSBUCHUNG  — Konto fehlt Buchung in Monat, wo sonst aktiv
"""

from __future__ import annotations

import pandas as pd

from src.accounting import kontoklasse
from src.config import AnalysisConfig
from src.tests.base import AnomalyTest, EngineStats


class MonatsEntwicklung(AnomalyTest):
    name = "MONATS_ENTWICKLUNG"
    weight = 1.5
    critical = False
    required_columns = ["_abs", "_datum", "konto_soll"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        # Stornos ausschließen
        is_storno = df.get("_is_storno", pd.Series(False, index=df.index))
        has_konto_date = df["_datum"].notna() & (df["_abs"] > 0) & (~is_storno)
        subset         = df.loc[has_konto_date].copy()

        if len(subset) < 10:
            return 0

        # GuV-Konten (Ertrag + Aufwand)
        kl = df.loc[has_konto_date, "_kontoklasse"] if "_kontoklasse" in df.columns else kontoklasse(subset["konto_soll"])
        pnl_mask = kl.isin(["Ertrag", "Aufwand"])
        pnl      = subset.loc[pnl_mask].copy()

        if len(pnl) < 10:
            return 0

        pnl["_ym"] = pnl["_datum"].dt.to_period("M").astype(str)

        monthly = (
            pnl.groupby(["konto_soll", "_ym"], observed=True)["_abs"]
            .sum()
            .reset_index(name="monatssumme")
        )
        konto_stats = (
            monthly.groupby("konto_soll", observed=True)["monatssumme"]
            .agg(ks_mean="mean", ks_std="std", ks_count="count")
            .reset_index()
        )
        konto_stats = konto_stats[
            konto_stats["ks_count"] >= config.monats_entwicklung_min_monate
        ]

        if konto_stats.empty:
            return 0

        # std=NaN (nur 1 Monat) → 0 → kein Outlier möglich
        konto_stats["ks_std"] = konto_stats["ks_std"].fillna(0)

        # Z-Score pro Konto-Monat (statt %-Abweichung)
        with_stats = monthly.merge(
            konto_stats[["konto_soll", "ks_mean", "ks_std"]], on="konto_soll"
        )
        # std=0 → alle Monate identisch → kein Outlier möglich
        with_stats["z"] = (
            (with_stats["monatssumme"] - with_stats["ks_mean"]).abs()
            / with_stats["ks_std"].replace(0, float("inf"))
        )
        outliers = with_stats[with_stats["z"] > config.monats_entwicklung_zscore]
        spike_set = set(zip(outliers["konto_soll"], outliers["_ym"]))

        if not spike_set:
            return 0

        pnl["_key"]  = list(zip(pnl["konto_soll"], pnl["_ym"]))
        flagged_sub  = pnl[pnl["_key"].isin(spike_set)]
        df.loc[flagged_sub.index, f"flag_{self.name}"] = True
        return int(df[f"flag_{self.name}"].sum())


class FehlendeMonatsbuchung(AnomalyTest):
    name = "FEHLENDE_MONATSBUCHUNG"
    weight = 1.0
    critical = False
    required_columns = ["_datum", "konto_soll"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        # Stornos ausschließen
        is_storno = df.get("_is_storno", pd.Series(False, index=df.index))
        has_konto_date = (
            df["_datum"].notna()
            & (df["konto_soll"].astype(str).str.strip() != "")
            & (~is_storno)
        )
        subset = df.loc[has_konto_date].copy()
        self.log("Subset", rows_with_konto_date=len(subset), total=len(df))

        if len(subset) < 10:
            self.log("Abbruch: zu wenig Daten", rows=len(subset))
            return 0

        subset["_ym"] = subset["_datum"].dt.to_period("M")
        all_periods   = sorted(subset["_ym"].unique())
        if len(all_periods) < 3:
            self.log("Abbruch: weniger als 3 Monate", months=len(all_periods))
            return 0

        # Volle Zeitspanne — nur so werden echte Lücken erkannt
        full_range = list(pd.period_range(all_periods[0], all_periods[-1], freq="M"))
        self.log("Zeitspanne", von=str(all_periods[0]), bis=str(all_periods[-1]),
                 monate_gesamt=len(full_range))

        # Mindest-Zeitfenster: < 6 Monate → zu kurz für aussagekräftige Lücken
        if len(full_range) < 6:
            self.log("Abbruch: Zeitspanne zu kurz", months=len(full_range))
            return 0

        konto_month_cnt = subset.groupby("konto_soll", observed=True)["_ym"].nunique()
        # min_active basiert auf voller Zeitspanne, nicht nur auf vorhandenen Monaten
        min_active      = max(3, int(len(full_range) * config.fehlende_buchung_min_quote))
        regular         = konto_month_cnt[konto_month_cnt >= min_active].index

        self.log("Reguläre Konten",
                 total_konten=len(konto_month_cnt),
                 min_active_months=min_active,
                 quote=config.fehlende_buchung_min_quote,
                 regular_konten=len(regular))

        if regular.empty:
            self.log("Keine regulären Konten gefunden")
            # Debug: Top-5 Konten mit meisten Monaten
            top5 = konto_month_cnt.nlargest(5)
            for konto, n in top5.items():
                self.log(f"Top-Konto", konto=str(konto), active_months=int(n),
                         needed=min_active)
            return 0

        prev_of    = {p: full_range[i - 1] for i, p in enumerate(full_range) if i > 0}
        next_of    = {p: full_range[i + 1] for i, p in enumerate(full_range[:-1])}

        flagged: set[int] = set()
        konten_mit_luecken = 0
        for konto in regular:
            konto_data = subset[subset["konto_soll"] == konto]
            booked     = set(konto_data["_ym"])
            idx_by_ym  = konto_data.groupby("_ym").groups
            gaps = [p for p in full_range if p not in booked]

            if gaps:
                konten_mit_luecken += 1

            for period in gaps:
                for adj in (prev_of.get(period), next_of.get(period)):
                    if adj and adj in idx_by_ym:
                        flagged.update(idx_by_ym[adj])

        self.log("Ergebnis",
                 konten_mit_luecken=konten_mit_luecken,
                 total_regular=len(regular),
                 flagged=len(flagged))

        if flagged:
            df.loc[list(flagged), f"flag_{self.name}"] = True
        return len(flagged)


def get_tests() -> list[AnomalyTest]:
    return [MonatsEntwicklung(), FehlendeMonatsbuchung()]
