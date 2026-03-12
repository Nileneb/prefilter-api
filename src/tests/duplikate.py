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

        for _, grp in df.groupby(["_abs", "konto_soll", "konto_haben"], sort=False):
            if len(grp) < 2:
                continue

            undated = grp[grp["_datum"].isna()]
            if len(undated) >= 2:
                flagged.update(undated.index)

            dated = grp[grp["_datum"].notna()].sort_values("_datum")
            if len(dated) < 2:
                continue
            idxs  = dated.index.tolist()
            dvals = dated["_datum"].tolist()
            for i in range(len(dvals)):
                for j in range(i + 1, len(dvals)):
                    if (dvals[j] - dvals[i]).days > window:
                        break
                    flagged.add(idxs[i])
                    flagged.add(idxs[j])

        if flagged:
            df.loc[list(flagged), f"flag_{self.name}"] = True
        return len(flagged)


class DoppelteBelegnummer(AnomalyTest):
    name = "DOPPELTE_BELEGNUMMER"
    weight = 2.0
    critical = True

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        beleg = df["belegnummer"].astype(str).str.strip()
        mask  = (beleg != "") & beleg.duplicated(keep=False)
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

        # Level 1: gleiche Belegnr. + gleicher Kreditor (vektorisiert)
        has_both = (beleg != "") & (kred != "")
        if has_both.any():
            bk          = beleg + "|" + kred
            bk_dup_mask = has_both & bk.duplicated(keep=False)
            flagged.update(df.index[bk_dup_mask])

        # Level 2: gleicher Kreditor + gleicher Betrag + Datum ≤ window Tage
        has_ka = (kred != "") & (df["_abs"] > 0)
        for _, grp in df.loc[has_ka].groupby(["kreditor", "_abs"], sort=False):
            if len(grp) < 2:
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
                    flagged.add(idxs[i])
                    flagged.add(idxs[j])

        if flagged:
            df.loc[list(flagged), f"flag_{self.name}"] = True
        return len(flagged)


def get_tests() -> list[AnomalyTest]:
    return [NearDuplicate(), DoppelteBelegnummer(), BelegKreditorDuplikat()]
