"""
Buchungs-Anomalie Pre-Filter — Buchungslogik-Tests

Tests:
    STORNO                  — Storno/Korrektur-Buchungen
    LEERER_BUCHUNGSTEXT     — Fehlender oder generischer Buchungstext
    RECHNUNGSDATUM_PERIODE  — Rechnungsmonat ≠ Buchungsmonat
    BUCHUNGSTEXT_PERIODE    — Periodenangabe im Text ≠ Buchungsdatum
"""

from __future__ import annotations

import pandas as pd

from src.config import AnalysisConfig
from src.parser import parse_date_series
from src.tests.base import AnomalyTest, EngineStats


class Storno(AnomalyTest):
    name = "STORNO"
    weight = 1.5
    critical = True

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        txt = df["buchungstext"].astype(str).str.lower()
        gutschrift_schwelle = (
            stats.b_mean + stats.b_std if stats.b_std > 0 else float("inf")
        )
        hard_mask       = txt.str.contains(
            r"storno|korrektur|r[uü]ckbuchung", regex=True, na=False
        )
        neg_mask        = df["_betrag"] < 0
        gutschrift_mask = (
            txt.str.contains("gutschrift", na=False)
            & (df["_abs"] > gutschrift_schwelle)
        )
        # Generalumgekehrt-Kennzeichen aus Diamant-Export
        gu = df["generalumgekehrt"].astype(str).str.strip().str.lower()
        gu_mask = gu.isin({"1", "true", "j", "ja", "yes", "x"})

        mask = hard_mask | neg_mask | gutschrift_mask | gu_mask
        return self._flag(df, mask)


class LeererBuchungstext(AnomalyTest):
    name = "LEERER_BUCHUNGSTEXT"
    weight = 1.0
    critical = False

    _GENERIC = frozenset({
        "diverse", "verschiedenes", "sonstiges", "test",
        "korrektur", "umbuchung", "xxx", "---", "...", "k.a.",
        "keine angabe", "n/a", "na", "tbd", "todo",
    })

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        txt  = df["buchungstext"].astype(str).str.strip()
        mask = (txt == "") | (txt.str.lower().isin(self._GENERIC)) | (txt.str.len() <= 2)
        return self._flag(df, mask)


class RechnungsdatumPeriode(AnomalyTest):
    name = "RECHNUNGSDATUM_PERIODE"
    weight = 1.5
    critical = False

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        rdatum = parse_date_series(df["rechnungsdatum"])
        has_both = df["_datum"].notna() & rdatum.notna()

        # Fallback: Buchungsperiode nutzen wenn kein Rechnungsdatum vorhanden
        if not has_both.any() and "buchungsperiode" in df.columns:
            rdatum = parse_date_series(df["buchungsperiode"])
            has_both = df["_datum"].notna() & rdatum.notna()

        if not has_both.any():
            return 0
        buch_period = df.loc[has_both, "_datum"].dt.to_period("M")
        rech_period = rdatum.loc[has_both].dt.to_period("M")
        mask_inner  = rech_period != buch_period
        flagged_idx = buch_period.index[mask_inner]
        df.loc[flagged_idx, f"flag_{self.name}"] = True
        return len(flagged_idx)


class BuchungstextPeriode(AnomalyTest):
    name = "BUCHUNGSTEXT_PERIODE"
    weight = 1.0
    critical = False

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        has_date = df["_datum"].notna()
        if not has_date.any():
            return 0
        txt       = df["buchungstext"].astype(str)
        extracted = txt.str.extract(r'\b(\d{1,2})[/.](\d{4})\b', expand=True)
        ext_month = pd.to_numeric(extracted[0], errors="coerce")
        ext_year  = pd.to_numeric(extracted[1], errors="coerce")
        has_period = ext_month.notna() & ext_year.notna() & ext_month.between(1, 12)
        active = has_date & has_period
        if not active.any():
            return 0
        buch_month = df["_datum"].dt.month
        buch_year  = df["_datum"].dt.year
        mismatch   = (ext_month != buch_month) | (ext_year != buch_year)
        mask       = active & mismatch
        return self._flag(df, mask)


def get_tests() -> list[AnomalyTest]:
    return [Storno(), LeererBuchungstext(), RechnungsdatumPeriode(), BuchungstextPeriode()]
