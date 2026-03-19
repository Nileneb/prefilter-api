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

# Alle Werte die in generalumgekehrt als "kein Storno" gelten.
# Verwendet in engine._prepare() (Codier-Regel #15).
_GU_FALSY = frozenset({"", "0", "0.0", "0.00", "nan", "null", "none", "na", "n/a", "false", "nein", "no"})


class Storno(AnomalyTest):
    name = "STORNO"
    weight = 1.5
    critical = False
    required_columns = ["_abs", "_betrag", "_is_storno", "buchungstext", "generalumgekehrt"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        # System-Stornos (mit Generalumgekehrt-Referenz) sind NORMAL und
        # werden NICHT geflaggt. _is_storno identifiziert sie weiterhin
        # fuer den Ausschluss aus anderen Tests.
        # Nur VERDAECHTIGE Storno-Muster werden hier geflaggt:
        #   1. Text-basierte Stornos OHNE GU-Referenz (manuell/verdaechtig)
        #   2. Gutschrift mit hohem Betrag (ohne Storno-Text)

        gu = df.get("generalumgekehrt", pd.Series("", index=df.index))
        gu_str = gu.astype(str).str.strip().str.rstrip(";")
        has_gu = ~gu_str.str.lower().isin(_GU_FALSY)

        is_storno = df["_is_storno"].fillna(False)
        text_only_storno = is_storno & ~has_gu

        # Gutschrift mit hohem Betrag (nicht-Storno-Gutschrift)
        gutschrift_schwelle = (
            stats.b_mean + stats.b_std if stats.b_std > 0 else float("inf")
        )
        txt = df["buchungstext"].astype(str).str.lower()
        gutschrift_mask = (
            txt.str.contains("gutschrift", na=False)
            & ~txt.str.contains("storn", na=False)
            & (df["_abs"] > gutschrift_schwelle)
        )

        mask = text_only_storno | gutschrift_mask
        return self._flag(df, mask)


class LeererBuchungstext(AnomalyTest):
    name = "LEERER_BUCHUNGSTEXT"
    weight = 1.0
    critical = False
    required_columns = ["buchungstext", "_kontoklasse"]

    _GENERIC = frozenset({
        "diverse", "verschiedenes", "sonstiges", "test",
        "xxx", "---", "...", "k.a.",
        "keine angabe", "n/a", "na", "tbd", "todo",
    })

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        txt  = df["buchungstext"].astype(str).str.strip()
        mask = (txt == "") | (txt.str.lower().isin(self._GENERIC)) | (txt.str.len() <= 2)
        # Nur PnL-Konten (Ertrag/Aufwand) flaggen — Bestand/Kostenrechnung ignorieren
        if "_kontoklasse" in df.columns:
            pnl_mask = df["_kontoklasse"].isin({"Ertrag", "Aufwand"})
            mask = mask & pnl_mask
        return self._flag(df, mask)


class RechnungsdatumPeriode(AnomalyTest):
    name = "RECHNUNGSDATUM_PERIODE"
    weight = 1.5
    critical = False
    required_columns = ["_datum", "erfassungsdatum", "buchungsperiode", "_is_storno"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        # _datum = Belegdatum = Rechnungsdatum (Synonyme)
        # Vergleich: Belegdatum vs Erfassungsdatum (wann gebucht?)
        erfdat = None
        source = None

        # Priorität 1: erfassungsdatum (ErfassungAm)
        if "erfassungsdatum" in df.columns:
            erfdat = parse_date_series(df["erfassungsdatum"])
            has_both = df["_datum"].notna() & erfdat.notna()
            if has_both.any():
                source = "erfassungsdatum"

        # Priorität 2: buchungsperiode
        if source is None and "buchungsperiode" in df.columns:
            erfdat = parse_date_series(df["buchungsperiode"])
            has_both = df["_datum"].notna() & erfdat.notna()
            if has_both.any():
                source = "buchungsperiode"

        self.log("Datumsquelle", source=source or "keine",
                 rows_with_both=int(has_both.sum()) if source else 0)

        if source is None:
            self.log("Keine vergleichbaren Datumspaare gefunden")
            return 0
        buch_period = df.loc[has_both, "_datum"].dt.to_period("M")
        rech_period = erfdat.loc[has_both].dt.to_period("M")

        # Nur flaggen wenn Differenz > 2 Monate (normale Abgrenzung ignorieren)
        diff_months = (
            (rech_period.dt.year - buch_period.dt.year) * 12
            + (rech_period.dt.month - buch_period.dt.month)
        ).abs()
        mask_inner = diff_months > 2

        # Stornos ausschließen — Periodenabweichung bei Stornos ist normal
        is_storno = df.get("_is_storno", pd.Series(False, index=df.index))
        mask_inner = mask_inner & (~is_storno.loc[has_both])

        self.log(
            "Monats-Differenzen",
            mean_diff=round(float(diff_months[mask_inner].mean()), 1) if mask_inner.any() else 0,
            max_diff=int(diff_months[mask_inner].max()) if mask_inner.any() else 0,
            count_gt_6m=int((diff_months > 6).sum()),
            count_gt_2m=int(mask_inner.sum()),
            total_pairs=int(len(diff_months)),
        )

        flagged_idx = buch_period.index[mask_inner]
        df.loc[flagged_idx, f"flag_{self.name}"] = True
        return len(flagged_idx)


class BuchungstextPeriode(AnomalyTest):
    name = "BUCHUNGSTEXT_PERIODE"
    weight = 1.0
    critical = False
    required_columns = ["_datum", "buchungstext", "_is_storno"]

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
        # Stornos ausschließen — Periodenabweichung bei Stornos ist normal
        is_storno = df.get("_is_storno", pd.Series(False, index=df.index))
        mask       = active & mismatch & (~is_storno)
        return self._flag(df, mask)


def get_tests() -> list[AnomalyTest]:
    return [Storno(), LeererBuchungstext(), RechnungsdatumPeriode(), BuchungstextPeriode()]
