"""
Buchungs-Anomalie Pre-Filter — Duplikat-Tests v3 (mit Logging)

Tests:
    NEAR_DUPLICATE          — Gleicher Betrag + Konten innerhalb N Tagen
    DOPPELTE_BELEGNUMMER    — Gleiche Belegnummer mehrfach
    BELEG_KREDITOR_DUPLIKAT — Gleiche Belegnr.+Kreditor oder Kreditor+Betrag+Datum
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
    logger=None,
    label: str = "",
) -> set[tuple]:
    """
    Identifiziert Gruppen, die in ≥ min_months verschiedenen Monaten vorkommen.
    """
    if min_months <= 0:
        return set()
    has_datum = df["_datum"].notna()
    if not has_datum.any():
        if logger:
            logger(f"reguläre Muster ({label}): kein Datum vorhanden → 0 reguläre",
                   label=label, regular_count=0)
        return set()

    dated = df.loc[has_datum]
    ym = dated["_datum"].dt.to_period("M")
    n_months_total = ym.nunique()

    regular: set[tuple] = set()
    total_groups = 0
    for key, grp in dated.groupby(group_cols, sort=False, observed=True):
        total_groups += 1
        if grp.empty:
            continue
        n_months = ym.loc[grp.index].nunique()
        if n_months >= min_months:
            regular.add(key if isinstance(key, tuple) else (key,))

    regular_rows = 0
    if regular:
        for key, grp in dated.groupby(group_cols, sort=False, observed=True):
            k = key if isinstance(key, tuple) else (key,)
            if k in regular:
                regular_rows += len(grp)

    if logger:
        logger(
            f"reguläre Muster ({label}): {len(regular)} von {total_groups} Gruppen "
            f"in ≥{min_months}/{n_months_total} Monaten, {regular_rows} Zeilen betroffen",
            label=label,
            regular_groups=len(regular),
            total_groups=total_groups,
            regular_rows=regular_rows,
            min_months=min_months,
            months_in_data=n_months_total,
        )
    return regular


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

        self.log("Config", window=window, max_grp=max_grp, reg_months=reg_months)

        # Stornos und Nullbeträge ausschließen
        is_storno = df.get("_is_storno", pd.Series(False, index=df.index))
        nonzero = (df["_abs"] > 0) & (~is_storno)
        n_zero = int((df["_abs"] <= 0).sum())
        n_storno = int(is_storno.sum())
        self.log("Ausgeschlossen", n_zero=n_zero, n_storno=n_storno)
        self.metric("n_zero_amounts", n_zero)
        self.metric("n_storno_excluded", n_storno)

        kred = df["kreditor"].astype(str).str.strip()
        has_kred = (kred != "") & nonzero
        no_kred = (~(kred != "")) & nonzero

        self.log(
            "Aufteilung",
            mit_kreditor=int(has_kred.sum()),
            ohne_kreditor=int(no_kred.sum()),
            nullbetrag=n_zero,
        )

        # ── MIT Kreditor ──
        kred_group_cols = ["_abs", "konto_soll", "kreditor"]
        regular_kred = _find_regular_keys(
            df[has_kred], kred_group_cols,
            reg_months, logger=self.log, label="kred",
        )

        n_groups_kred = 0
        n_skipped_regular = 0
        n_skipped_size = 0
        n_flagged_kred = 0

        if has_kred.any():
            for key, grp in df[has_kred].groupby(
                kred_group_cols,
                sort=False, observed=True,
            ):
                n_groups_kred += 1
                if key in regular_kred:
                    n_skipped_regular += 1
                    continue
                if len(grp) < 2 or len(grp) > max_grp:
                    n_skipped_size += 1
                    continue
                before = len(flagged)
                self._flag_near_window(grp, window, flagged, max_grp)
                n_flagged_kred += len(flagged) - before

        self.log(
            "MIT Kreditor fertig",
            total_groups=n_groups_kred,
            skipped_size=n_skipped_size,
            skipped_regular=n_skipped_regular,
            flagged=n_flagged_kred,
        )
        self.metric("kred_groups", n_groups_kred)
        self.metric("kred_skipped_regular", n_skipped_regular)
        self.metric("kred_flagged", n_flagged_kred)

        # ── OHNE Kreditor ──
        nokred_group_cols = ["_abs", "konto_soll", "buchungstext"]
        regular_nokred = _find_regular_keys(
            df[no_kred], nokred_group_cols,
            reg_months, logger=self.log, label="no_kred",
        )

        n_groups_nokred = 0
        n_skipped_regular_nk = 0
        n_skipped_size_nk = 0
        n_flagged_nokred = 0

        if no_kred.any():
            for key, grp in df[no_kred].groupby(
                nokred_group_cols,
                sort=False, observed=True,
            ):
                n_groups_nokred += 1
                if key in regular_nokred:
                    n_skipped_regular_nk += 1
                    continue
                if len(grp) < 2 or len(grp) > max_grp:
                    n_skipped_size_nk += 1
                    continue
                before = len(flagged)
                self._flag_near_window(grp, window, flagged, max_grp)
                n_flagged_nokred += len(flagged) - before

        self.log(
            "OHNE Kreditor fertig",
            total_groups=n_groups_nokred,
            skipped_size=n_skipped_size_nk,
            skipped_regular=n_skipped_regular_nk,
            flagged=n_flagged_nokred,
        )
        self.metric("nokred_groups", n_groups_nokred)
        self.metric("nokred_flagged", n_flagged_nokred)

        if flagged:
            df.loc[list(flagged), f"flag_{self.name}"] = True

        self.log("TOTAL", total_flagged=len(flagged))
        return len(flagged)

    @staticmethod
    def _flag_near_window(
        grp: pd.DataFrame, window: int, flagged: set, max_grp: int
    ) -> None:
        # Beleg-interne Zeilen (gleiche _beleg_id) nicht als Duplikat werten
        beleg_ids = grp["_beleg_id"].astype(str) if "_beleg_id" in grp.columns else None

        undated = grp[grp["_datum"].isna()]
        if 2 <= len(undated) <= max_grp:
            if beleg_ids is not None:
                # Nur flaggen wenn verschiedene _beleg_ids
                if undated.loc[undated.index, "_beleg_id"].astype(str).nunique() > 1:
                    flagged.update(undated.index)
            else:
                flagged.update(undated.index)

        dated = grp[grp["_datum"].notna()].sort_values("_datum")
        if len(dated) < 2:
            return

        diffs = dated["_datum"].diff().dt.days
        close_mask = diffs.fillna(window + 1) <= window

        # Beleg-intern: Paare mit gleicher _beleg_id überspringen
        if beleg_ids is not None:
            dated_beleg = dated["_beleg_id"].astype(str)
            prev_beleg = dated_beleg.shift(1)
            same_beleg = dated_beleg == prev_beleg
            close_mask = close_mask & (~same_beleg)

        prev_mask = close_mask.shift(-1, fill_value=False)
        flagged.update(dated.index[close_mask | prev_mask])


class DoppelteBelegnummer(AnomalyTest):
    name = "DOPPELTE_BELEGNUMMER"
    weight = 2.0
    critical = True
    required_columns = ["_betrag", "belegnummer", "konto_soll"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        min_count = getattr(config, "doppelte_beleg_min_count", 3)
        prefix_ignore_raw = getattr(config, "doppelte_beleg_prefix_ignore", "")
        prefixes = [p.strip().upper() for p in prefix_ignore_raw.split(",") if p.strip()] if prefix_ignore_raw else []
        self.log("Config", min_count=min_count, prefix_ignore=prefixes)

        # Stornos ausschließen
        is_storno = df.get("_is_storno", pd.Series(False, index=df.index))

        beleg_str = df["belegnummer"].astype(str).str.strip()
        has_beleg = (beleg_str != "") & (~is_storno)

        # Präfix-Whitelist anwenden
        if prefixes:
            beleg_upper = beleg_str.str.upper()
            prefix_mask = pd.Series(False, index=df.index)
            for prefix in prefixes:
                prefix_mask = prefix_mask | beleg_upper.str.startswith(prefix, na=False)
            n_excluded = int((has_beleg & prefix_mask).sum())
            self.log("Präfix-Whitelist", excluded=n_excluded, prefixes=prefixes)
            self.metric("prefix_excluded", n_excluded)
            has_beleg = has_beleg & ~prefix_mask

        n_with_beleg = int(has_beleg.sum())
        self.log("Buchungen mit Belegnummer", count=n_with_beleg)
        self.metric("rows_with_beleg", n_with_beleg)

        if not has_beleg.any():
            return 0

        sub = df.loc[has_beleg]
        group_cols = ["belegnummer", "konto_soll", "_betrag"]

        # Reguläre Zahlungsmuster ausschließen (Nummernkreise, Buchungsläufe)
        reg_months = getattr(config, "doppelte_beleg_regular_months", 6)
        regular_beleg = _find_regular_keys(
            sub, group_cols, reg_months,
            logger=self.log, label="doppelte_beleg",
        )

        # Beleg-intern: Nur verschiedene _beleg_id's zählen
        has_beleg_id = "_beleg_id" in sub.columns
        if has_beleg_id:
            # Zähle distinkte _beleg_id pro Gruppe
            distinct_belege = (
                sub.groupby(group_cols, sort=False, observed=True)["_beleg_id"]
                .transform("nunique")
            )
            grp_size = distinct_belege
        else:
            grp_size = sub.groupby(group_cols, sort=False, observed=True).transform("size").iloc[:, 0] if isinstance(sub.groupby(group_cols, sort=False, observed=True).transform("size"), pd.DataFrame) else sub.groupby(group_cols, sort=False, observed=True).transform("size")

        # Reguläre Muster: grp_size auf 0 setzen
        if regular_beleg:
            for key, grp in sub.groupby(group_cols, sort=False, observed=True):
                k = key if isinstance(key, tuple) else (key,)
                if k in regular_beleg:
                    grp_size.loc[grp.index] = 0

        # Soll/Haben-Paare ausschließen: S+H mit gleicher Belegnr. = Paar
        sh = df["soll_haben"].astype(str).str.strip().str.upper()
        has_sh = sh.isin(["S", "SOLL", "H", "HABEN"])
        if has_sh.any():
            is_soll = sh.isin(["S", "SOLL"])
            for beleg_val, grp in sub.groupby("belegnummer", sort=False, observed=True):
                if len(grp) == 2:
                    idx = grp.index
                    if has_sh.loc[idx].all():
                        soll_count = is_soll.loc[idx].sum()
                        if soll_count == 1:  # Genau 1 Soll + 1 Haben = Paar
                            grp_size.loc[idx] = 0  # Nicht als Duplikat zählen

        # Verteilung loggen
        for threshold in [2, 3, 4, 5, 10]:
            cnt = int((grp_size >= threshold).sum())
            self.log(f"Gruppen ≥{threshold}", threshold=threshold, rows=cnt)
            self.metric(f"rows_ge_{threshold}", cnt)

        # Top-5 häufigste Belegnummern
        beleg_counts = sub["belegnummer"].value_counts().head(5)
        for bn, cnt in beleg_counts.items():
            self.log(f"Top-Belegnr", belegnummer=str(bn), count=int(cnt))

        mask = pd.Series(False, index=df.index)
        mask.loc[grp_size.index[grp_size >= min_count]] = True
        count = self._flag(df, mask)

        self.log("TOTAL", total_flagged=count, min_count=min_count)
        return count


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

        self.log("Config", window=window, max_grp=max_grp)

        # Stornos ausschließen
        is_storno = df.get("_is_storno", pd.Series(False, index=df.index))

        has_all = (beleg != "") & (kred != "") & (~is_storno)
        self.log("Buchungen mit Beleg+Kreditor", count=int(has_all.sum()))
        self.metric("rows_with_beleg_kred", int(has_all.sum()))

        # ── Dynamisches reg_months basierend auf Datenmonate ──
        has_datum = df["_datum"].notna()
        if has_datum.any():
            n_months_total = df.loc[has_datum, "_datum"].dt.to_period("M").nunique()
        else:
            n_months_total = 0
        reg_months = max(6, int(n_months_total * config.beleg_kreditor_regular_pct))
        self.log("Reguläre Muster", n_months_total=n_months_total, reg_months=reg_months,
                 regular_pct=config.beleg_kreditor_regular_pct)

        # ── Reguläre Kreditoren ──
        regular_kb = _find_regular_keys(
            df[has_all], ["kreditor", "_abs"], reg_months,
            logger=self.log, label="beleg_kred_regular",
        )

        # ── Level 1: gleiche Belegnr + Kreditor + Betrag ──
        level1_count = 0
        level1_regular = 0
        if has_all.any():
            sub = df.loc[has_all]
            grp_counts = sub.groupby(
                ["belegnummer", "kreditor", "_abs"], sort=False, observed=True
            )
            # Beleg-intern: Nur verschiedene _beleg_id's zählen
            has_beleg_id = "_beleg_id" in sub.columns
            if has_beleg_id:
                grp_size = grp_counts["_beleg_id"].transform("nunique")
            else:
                grp_size = grp_counts.transform("size")
                grp_size = grp_size.iloc[:, 0] if isinstance(grp_size, pd.DataFrame) else grp_size
            dup_idx = grp_size.index[grp_size >= 4]
            level1_raw = len(dup_idx)

            for idx in dup_idx:
                kb = (str(df.at[idx, "kreditor"]).strip(), df.at[idx, "_abs"])
                if kb in regular_kb:
                    level1_regular += 1
                else:
                    flagged.add(idx)
                    level1_count += 1

        self.log(
            "Level 1 fertig",
            raw=level1_raw if has_all.any() else 0,
            regular_excluded=level1_regular,
            flagged=level1_count,
        )
        self.metric("level1_raw", level1_raw if has_all.any() else 0)
        self.metric("level1_regular_excluded", level1_regular)
        self.metric("level1_flagged", level1_count)

        # ── Level 2: gleicher Kreditor + Betrag + Datum ≤ window ──
        has_ka = (kred != "") & (df["_abs"] > 0) & (beleg != "") & (~is_storno)
        level2_count = 0
        level2_groups = 0
        level2_skipped_size = 0
        level2_skipped_same_beleg = 0
        level2_skipped_regular = 0
        level2_skipped_no_dates = 0

        for key, grp in df.loc[has_ka].groupby(
            ["kreditor", "_abs"], sort=False, observed=True
        ):
            level2_groups += 1
            if len(grp) < 2 or len(grp) > max_grp:
                level2_skipped_size += 1
                continue
            if key in regular_kb:
                level2_skipped_regular += 1
                continue
            if grp["belegnummer"].astype(str).str.strip().nunique() < 3:
                level2_skipped_same_beleg += 1
                continue

            dated = grp[grp["_datum"].notna()].sort_values("_datum")
            if len(dated) < 2:
                level2_skipped_no_dates += 1
                continue

            idxs  = dated.index.tolist()
            dvals = dated["_datum"].tolist()
            belege = dated["belegnummer"].astype(str).str.strip().tolist()
            beleg_ids = dated["_beleg_id"].astype(str).tolist() if "_beleg_id" in dated.columns else None
            left = 0
            for right in range(1, len(dvals)):
                while left < right and (dvals[right] - dvals[left]).days > window:
                    left += 1
                for i in range(left, right):
                    if belege[i] != belege[right]:
                        # Beleg-intern: gleiche _beleg_id = kein Duplikat
                        if beleg_ids is not None and beleg_ids[i] == beleg_ids[right]:
                            continue
                        before = len(flagged)
                        flagged.add(idxs[i])
                        flagged.add(idxs[right])
                        level2_count += len(flagged) - before

        self.log(
            "Level 2 fertig",
            total_groups=level2_groups,
            skipped_size=level2_skipped_size,
            skipped_regular=level2_skipped_regular,
            skipped_same_beleg=level2_skipped_same_beleg,
            skipped_no_dates=level2_skipped_no_dates,
            flagged=level2_count,
        )
        self.metric("level2_groups", level2_groups)
        self.metric("level2_flagged", level2_count)

        if flagged:
            df.loc[list(flagged), f"flag_{self.name}"] = True

        self.log("TOTAL", total_flagged=len(flagged))
        return len(flagged)


def get_tests() -> list[AnomalyTest]:
    return [NearDuplicate(), DoppelteBelegnummer(), BelegKreditorDuplikat()]
