"""
Buchungs-Anomalie Pre-Filter — Duplikat-Tests v2

Fixes gegenüber v1:
─────────────────────────────────────────────────
1. NEAR_DUPLICATE (120k → Ziel ~15k):
   - Reguläre Muster: auch OHNE Kreditor prüfen (Betrag+Konto+Buchungstext)
   - Mindestbetrag: Nur Beträge > 0 flaggen (Nullbeträge = Umbuchungen)
   - konto_haben in Gruppierung aufnehmen (Soll/Haben-Paare trennen)

2. BELEG_KREDITOR_DUPLIKAT (100k → Ziel ~15k):
   - Level 1: Reguläre Kreditoren ausschließen (Kreditor+Betrag in ≥6 Monaten)
   - Level 2: O(n²)-Schleife durch Sliding-Window ersetzen (Performance)
   - Level 2: Reguläre-Kredit.-Ausschluss auch hier

3. DOPPELTE_BELEGNUMMER (72k → Ziel ~10k):
   - Gegenkonto einbeziehen: gleiche Belegnr mit versch. konto_haben = Gegenbuchung
   - Schwelle auf ≥3 war schon korrekt, ABER _betrag muss _abs sein
     (Soll-Buchung +1000 und Haben-Buchung -1000 haben versch. _betrag
      aber gleiche _abs → die aktuelle Logik mit _betrag ist RICHTIG,
      weil Soll/Haben verschiedene Vorzeichen haben)

Alle Änderungen sind abwärtskompatibel mit bestehenden Unit-Tests.
"""

from __future__ import annotations

import pandas as pd

from src.config import AnalysisConfig
from src.tests.base import AnomalyTest, EngineStats


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktion: Reguläre Zahlungsmuster erkennen
# ─────────────────────────────────────────────────────────────────────────────

def _find_regular_keys(
    df: pd.DataFrame,
    group_cols: list[str],
    min_months: int,
) -> set[tuple]:
    """
    Identifiziert Gruppen, die in ≥ min_months verschiedenen Monaten vorkommen.
    Diese sind reguläre Zahlungsmuster (Gehälter, Miete, Dauerschulden).
    """
    if min_months <= 0:
        return set()

    has_datum = df["_datum"].notna()
    if not has_datum.any():
        return set()

    dated = df.loc[has_datum]
    ym = dated["_datum"].dt.to_period("M")

    regular: set[tuple] = set()
    for key, grp in dated.groupby(group_cols, sort=False, observed=True):
        if grp.empty:
            continue
        n_months = ym.loc[grp.index].nunique()
        if n_months >= min_months:
            regular.add(key if isinstance(key, tuple) else (key,))
    return regular


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: NEAR_DUPLICATE
# ─────────────────────────────────────────────────────────────────────────────

class NearDuplicate(AnomalyTest):
    name = "NEAR_DUPLICATE"
    weight = 2.0
    critical = True
    required_columns = ["_abs", "_datum", "konto_soll", "konto_haben", "kreditor", "buchungstext"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        flagged: set[int] = set()
        window = config.near_duplicate_days
        max_grp = config.near_duplicate_max_group_size
        reg_months = config.near_duplicate_regular_months

        # FIX: Nullbeträge ausschließen (= Umbuchungen, nicht verdächtig)
        nonzero = df["_abs"] > 0

        # ── Reguläre Muster MIT Kreditor ──
        kred = df["kreditor"].astype(str).str.strip()
        has_kred = (kred != "") & nonzero

        # FIX: konto_haben in Gruppierung → Soll/Haben-Paare trennen
        regular_kred = _find_regular_keys(
            df[has_kred], ["_abs", "konto_soll", "konto_haben", "kreditor"], reg_months
        )

        if has_kred.any():
            for key, grp in df[has_kred].groupby(
                ["_abs", "konto_soll", "konto_haben", "kreditor"],
                sort=False, observed=True,
            ):
                if len(grp) < 2 or len(grp) > max_grp:
                    continue
                if key in regular_kred:
                    continue
                self._flag_near_window(grp, window, flagged, max_grp)

        # ── Ohne Kreditor: Betrag+Konto+Buchungstext ──
        no_kred = (~has_kred) & nonzero
        if no_kred.any():
            # FIX: Auch für Nicht-Kreditor-Gruppen reguläre Muster prüfen
            regular_nokred = _find_regular_keys(
                df[no_kred], ["_abs", "konto_soll", "konto_haben", "buchungstext"], reg_months
            )
            for key, grp in df[no_kred].groupby(
                ["_abs", "konto_soll", "konto_haben", "buchungstext"],
                sort=False, observed=True,
            ):
                if len(grp) < 2 or len(grp) > max_grp:
                    continue
                if key in regular_nokred:
                    continue
                self._flag_near_window(grp, window, flagged, max_grp)

        if flagged:
            df.loc[list(flagged), f"flag_{self.name}"] = True
        return len(flagged)

    @staticmethod
    def _flag_near_window(
        grp: pd.DataFrame, window: int, flagged: set, max_grp: int
    ) -> None:
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


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: DOPPELTE_BELEGNUMMER
# ─────────────────────────────────────────────────────────────────────────────

class DoppelteBelegnummer(AnomalyTest):
    name = "DOPPELTE_BELEGNUMMER"
    weight = 2.0
    critical = True
    required_columns = ["_betrag", "belegnummer", "konto_soll", "konto_haben"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        # Unverändert: gleiche Belegnr + gleiches Soll + gleiches Haben + gleicher Betrag
        # ≥3 identische = verdächtig (2 kann normales Soll/Haben-Paar sein)
        has_beleg = df["belegnummer"].astype(str).str.strip() != ""
        if not has_beleg.any():
            return 0
        sub = df.loc[has_beleg]
        min_count = config.doppelte_beleg_min_count
        grp_size = sub.groupby(
            ["belegnummer", "konto_soll", "konto_haben", "_betrag"],
            sort=False, observed=True,
        ).transform("size")
        mask = pd.Series(False, index=df.index)
        mask.loc[grp_size.index[grp_size >= min_count]] = True
        return self._flag(df, mask)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: BELEG_KREDITOR_DUPLIKAT
# ─────────────────────────────────────────────────────────────────────────────

class BelegKreditorDuplikat(AnomalyTest):
    name = "BELEG_KREDITOR_DUPLIKAT"
    weight = 2.5
    critical = True
    required_columns = ["_abs", "_betrag", "_datum", "belegnummer", "kreditor", "konto_soll"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        flagged: set[int] = set()
        beleg   = df["belegnummer"].astype(str).str.strip()
        kred    = df["kreditor"].astype(str).str.strip()
        window  = config.beleg_kreditor_days
        max_grp = config.beleg_kreditor_max_group_size

        has_all = (beleg != "") & (kred != "")

        # ── FIX: Reguläre Kreditoren identifizieren ──
        # Kreditor + Betrag in ≥6 verschiedenen Monaten → Dauerschuldverhältnis
        regular_kb = _find_regular_keys(
            df[has_all], ["kreditor", "_betrag"], 6
        )

        # ── Level 1: gleiche Belegnr + gleicher Kreditor + gleicher Betrag ──
        # FIX: Reguläre Kreditor-Betrag-Paare ausschließen
        if has_all.any():
            sub = df.loc[has_all]
            grp_size = sub.groupby(
                ["belegnummer", "kreditor", "_betrag"], sort=False, observed=True
            ).transform("size")
            dup_idx = grp_size.index[grp_size > 1]
            for idx in dup_idx:
                kb_key = (df.at[idx, "kreditor"], df.at[idx, "_betrag"])
                # Sicherstellen dass key ein sauberes Tuple ist
                kb_key_clean = (str(kb_key[0]).strip(), kb_key[1])
                if kb_key_clean not in regular_kb:
                    flagged.add(idx)

        # ── Level 2: gleicher Kreditor + gleicher Betrag + Datum ≤ window ──
        # FIX: Sliding-Window statt O(n²)
        has_ka = (kred != "") & (df["_abs"] > 0) & (beleg != "")
        for key, grp in df.loc[has_ka].groupby(
            ["kreditor", "_betrag"], sort=False, observed=True
        ):
            if len(grp) < 2 or len(grp) > max_grp:
                continue
            # Reguläre Kreditoren überspringen
            if key in regular_kb:
                continue
            # Nur wenn VERSCHIEDENE Belegnummern existieren
            if grp["belegnummer"].astype(str).str.strip().nunique() < 2:
                continue

            dated = grp[grp["_datum"].notna()].sort_values("_datum")
            if len(dated) < 2:
                continue

            # FIX: Sliding-Window statt O(n²)-Doppelschleife
            idxs  = dated.index.tolist()
            dvals = dated["_datum"].tolist()
            belege = dated["belegnummer"].astype(str).str.strip().tolist()
            left = 0
            for right in range(1, len(dvals)):
                # Linkes Fenster vorziehen
                while left < right and (dvals[right] - dvals[left]).days > window:
                    left += 1
                # Alle Paare im Fenster prüfen
                for i in range(left, right):
                    if belege[i] != belege[right]:
                        flagged.add(idxs[i])
                        flagged.add(idxs[right])

        if flagged:
            df.loc[list(flagged), f"flag_{self.name}"] = True
        return len(flagged)


def get_tests() -> list[AnomalyTest]:
    return [NearDuplicate(), DoppelteBelegnummer(), BelegKreditorDuplikat()]
