"""
Buchungs-Anomalie Pre-Filter — Duplikat-Tests

Tests:
    NEAR_DUPLICATE          — Gleicher Betrag + Konten innerhalb N Tagen
    DOPPELTE_BELEGNUMMER    — Gleiche Belegnummer mehrfach
    BELEG_KREDITOR_DUPLIKAT — Gleiche Belegnr.+Kreditor oder Kreditor+Betrag+Datum
"""

from __future__ import annotations

import pandas as pd

from src.config import AnalysisConfig
from src.tests.base import AnomalyTest, EngineStats


class NearDuplicate(AnomalyTest):
    name = "NEAR_DUPLICATE"
    weight = 2.0
    critical = True

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        flagged: set[int] = set()
        window = config.near_duplicate_days

        # Gruppierung: Betrag + Konto + Kreditor (statt Konto+Gegenkonto)
        # Das unterscheidet z.B. 100 Gehälter an verschiedene Personen
        kred = df["kreditor"].astype(str).str.strip()
        has_kred = kred != ""

        # Mit Kreditor: Betrag + Konto + Kreditor
        if has_kred.any():
            for _, grp in df[has_kred].groupby(
                ["_abs", "konto_soll", "kreditor"], sort=False, observed=True
            ):
                if len(grp) < 2:
                    continue
                self._flag_near_window(grp, window, flagged)

        # Ohne Kreditor: Betrag + Konto + Buchungstext (Fallback)
        no_kred = ~has_kred
        if no_kred.any():
            for _, grp in df[no_kred].groupby(
                ["_abs", "konto_soll", "buchungstext"], sort=False, observed=True
            ):
                if len(grp) < 2:
                    continue
                self._flag_near_window(grp, window, flagged)

        if flagged:
            df.loc[list(flagged), f"flag_{self.name}"] = True
        return len(flagged)

    @staticmethod
    def _flag_near_window(grp: pd.DataFrame, window: int, flagged: set) -> None:
        undated = grp[grp["_datum"].isna()]
        if len(undated) >= 2:
            flagged.update(undated.index)

        dated = grp[grp["_datum"].notna()].sort_values("_datum")
        if len(dated) < 2:
            return

        diffs = dated["_datum"].diff().dt.days
        close_mask = diffs.fillna(window + 1) <= window
        prev_mask = close_mask.shift(-1, fill_value=False)
        flagged.update(dated.index[close_mask | prev_mask])


class DoppelteBelegnummer(AnomalyTest):
    name = "DOPPELTE_BELEGNUMMER"
    weight = 2.0
    critical = True

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        beleg = df["belegnummer"].astype(str).str.strip()
        # Echtes Duplikat: gleiche Belegnr + gleiches Konto + gleicher Betrag
        key = beleg + "|" + df["konto_soll"].astype(str) + "|" + df["_betrag"].astype(str)
        mask = (beleg != "") & key.duplicated(keep=False)
        return self._flag(df, mask)


class BelegKreditorDuplikat(AnomalyTest):
    name = "BELEG_KREDITOR_DUPLIKAT"
    weight = 2.5
    critical = True

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        flagged: set[int] = set()
        beleg   = df["belegnummer"].astype(str).str.strip()
        kred    = df["kreditor"].astype(str).str.strip()
        window  = config.beleg_kreditor_days

        # Level 1: gleiche Belegnr + gleicher Kreditor + GLEICHER BETRAG
        # Verschiedene Beträge auf gleicher Belegnr = normale Aufschlüsselung!
        has_all = (beleg != "") & (kred != "")
        if has_all.any():
            bkb = beleg + "|" + kred + "|" + df["_betrag"].astype(str)
            bkb_dup_mask = has_all & bkb.duplicated(keep=False)
            flagged.update(df.index[bkb_dup_mask])

        # Level 2: gleicher Kreditor + gleicher Betrag + Datum ≤ window Tage
        # ABER: unterschiedliche Belegnummern (sonst schon oben gefangen)
        has_ka = (kred != "") & (df["_abs"] > 0) & (beleg != "")
        for _, grp in df.loc[has_ka].groupby(["kreditor", "_betrag"], sort=False, observed=True):
            if len(grp) < 2:
                continue
            # Nur wenn VERSCHIEDENE Belegnummern
            if grp["belegnummer"].nunique() < 2:
                continue
            dated = grp[grp["_datum"].notna()].sort_values("_datum")
            if len(dated) < 2:
                continue
            idxs  = dated.index.tolist()
            dvals = dated["_datum"].tolist()
            for i in range(len(dvals)):
                for j in range(i + 1, len(dvals)):
                    if (dvals[j] - dvals[i]).days > window:
                        break
                    # Nur flaggen wenn unterschiedliche Belegnummer
                    if dated.loc[idxs[i], "belegnummer"] != dated.loc[idxs[j], "belegnummer"]:
                        flagged.add(idxs[i])
                        flagged.add(idxs[j])

        if flagged:
            df.loc[list(flagged), f"flag_{self.name}"] = True
        return len(flagged)


def get_tests() -> list[AnomalyTest]:
    return [NearDuplicate(), DoppelteBelegnummer(), BelegKreditorDuplikat()]
