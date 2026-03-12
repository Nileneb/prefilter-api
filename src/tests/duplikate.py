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
    required_columns = ["_abs", "_datum", "konto_soll", "kreditor", "buchungstext"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        flagged: set[int] = set()
        window = config.near_duplicate_days
        max_grp = config.near_duplicate_max_group_size
        reg_months = config.near_duplicate_regular_months

        # Reguläre Zahlungsmuster identifizieren:
        # Kreditor+Konto+Betrag in ≥ reg_months verschiedenen Monaten → regulär
        regular_keys: set[tuple] = set()
        has_datum = df["_datum"].notna()
        if has_datum.any() and reg_months > 0:
            dated = df.loc[has_datum]
            ym = dated["_datum"].dt.to_period("M")
            for key, grp in dated.groupby(
                ["_abs", "konto_soll", "kreditor"], sort=False, observed=True
            ):
                if grp.empty:
                    continue
                n_months = ym.loc[grp.index].nunique()
                if n_months >= reg_months:
                    regular_keys.add(key)

        # Gruppierung: Betrag + Konto + Kreditor
        kred = df["kreditor"].astype(str).str.strip()
        has_kred = kred != ""

        # Mit Kreditor: Betrag + Konto + Kreditor
        if has_kred.any():
            for key, grp in df[has_kred].groupby(
                ["_abs", "konto_soll", "kreditor"], sort=False, observed=True
            ):
                if len(grp) < 2 or len(grp) > max_grp:
                    continue
                # Reguläre Zahlungsmuster überspringen
                if key in regular_keys:
                    continue
                self._flag_near_window(grp, window, flagged, max_grp)

        # Ohne Kreditor: Betrag + Konto + Buchungstext (Fallback)
        no_kred = ~has_kred
        if no_kred.any():
            for _, grp in df[no_kred].groupby(
                ["_abs", "konto_soll", "buchungstext"], sort=False, observed=True
            ):
                if len(grp) < 2 or len(grp) > max_grp:
                    continue
                self._flag_near_window(grp, window, flagged, max_grp)

        if flagged:
            df.loc[list(flagged), f"flag_{self.name}"] = True
        return len(flagged)

    @staticmethod
    def _flag_near_window(grp: pd.DataFrame, window: int, flagged: set, max_grp: int) -> None:
        undated = grp[grp["_datum"].isna()]
        if 2 <= len(undated) <= max_grp:
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
    required_columns = ["_betrag", "belegnummer", "konto_soll", "konto_haben"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        # Echtes Duplikat: gleiche Belegnr + gleiches Soll-/Habenkonto + gleicher Betrag
        # Soll/Haben-Paare mit gleicher Belegnr aber verschiedenen Konten → normal
        # Erst ab 3+ identischen Keys verdächtig (2 = Soll/Haben-Paar)
        has_beleg = df["belegnummer"].astype(str).str.strip() != ""
        if not has_beleg.any():
            return 0
        sub = df.loc[has_beleg]
        grp_size = sub.groupby(
            ["belegnummer", "konto_soll", "konto_haben", "_betrag"],
            sort=False, observed=True,
        ).transform("size")
        mask = pd.Series(False, index=df.index)
        mask.loc[grp_size.index[grp_size >= 3]] = True
        return self._flag(df, mask)


class BelegKreditorDuplikat(AnomalyTest):
    name = "BELEG_KREDITOR_DUPLIKAT"
    weight = 2.5
    critical = True
    required_columns = ["_abs", "_betrag", "_datum", "belegnummer", "kreditor"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        flagged: set[int] = set()
        beleg   = df["belegnummer"].astype(str).str.strip()
        kred    = df["kreditor"].astype(str).str.strip()
        window  = config.beleg_kreditor_days

        # Level 1: gleiche Belegnr + gleicher Kreditor + GLEICHER BETRAG
        # Verwende Groupby statt String-Konkatenation (spart RAM bei 788k Zeilen)
        has_all = (beleg != "") & (kred != "")
        if has_all.any():
            sub = df.loc[has_all]
            grp_size = sub.groupby(
                ["belegnummer", "kreditor", "_betrag"], sort=False, observed=True
            ).transform("size")
            dup_idx = grp_size.index[grp_size > 1]
            flagged.update(dup_idx)

        # Level 2: gleicher Kreditor + gleicher Betrag + Datum ≤ window Tage
        # ABER: unterschiedliche Belegnummern (sonst schon oben gefangen)
        # Gruppen > max_group_size überspringen (Dauerschuldverhältnisse)
        max_grp = config.beleg_kreditor_max_group_size
        has_ka = (kred != "") & (df["_abs"] > 0) & (beleg != "")
        for _, grp in df.loc[has_ka].groupby(["kreditor", "_betrag"], sort=False, observed=True):
            if len(grp) < 2 or len(grp) > max_grp:
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
