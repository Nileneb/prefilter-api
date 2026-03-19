"""
Buchungs-Anomalie Pre-Filter — Duplikat-Tests v3 (mit Logging)

Tests:
    NEAR_DUPLICATE          — Gleicher Betrag + Konten innerhalb N Tagen
    DOPPELTE_BELEGNUMMER    — Gleiche Belegnummer mehrfach
    BELEG_KREDITOR_DUPLIKAT — Gleiche Belegnr.+Kreditor oder Kreditor+Betrag+Datum
"""

from __future__ import annotations

import numpy as np
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
    required_columns = ["_abs", "_datum", "konto_soll", "kreditor", "buchungstext",
                        "_is_storno", "_beleg_id", "_kreditor_canonical"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        flagged: set[int] = set()
        window = config.near_duplicate_days
        max_grp = config.near_duplicate_max_group_size
        reg_months = config.near_duplicate_regular_months
        text_sim_threshold = getattr(config, "near_duplicate_text_similarity", 0.0)

        # Text-Embeddings aus EngineStats (von engine._compute_stats() gesetzt)
        text_embeddings = getattr(stats, "text_embeddings", None)
        has_embeddings = text_embeddings is not None and text_sim_threshold > 0

        self.log("Config", window=window, max_grp=max_grp, reg_months=reg_months,
                 text_sim_threshold=text_sim_threshold, has_embeddings=has_embeddings)

        # Stornos und Nullbeträge ausschließen
        is_storno = df.get("_is_storno", pd.Series(False, index=df.index))
        nonzero = (df["_abs"] > 0) & (~is_storno)
        n_zero = int((df["_abs"] <= 0).sum())
        n_storno = int(is_storno.sum())
        self.log("Ausgeschlossen", n_zero=n_zero, n_storno=n_storno)
        self.metric("n_zero_amounts", n_zero)
        self.metric("n_storno_excluded", n_storno)

        # Kreditor-Spalte: kanonisch wenn vorhanden, sonst Original
        kred_col = "_kreditor_canonical" if "_kreditor_canonical" in df.columns else "kreditor"
        kred = df[kred_col].astype(str).str.strip()
        has_kred = (kred != "") & nonzero
        no_kred = (~(kred != "")) & nonzero

        self.log(
            "Aufteilung",
            mit_kreditor=int(has_kred.sum()),
            ohne_kreditor=int(no_kred.sum()),
            nullbetrag=n_zero,
            kreditor_col=kred_col,
        )

        # ── MIT Kreditor ──
        kred_group_cols = ["_abs", "konto_soll", kred_col]
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
                self._flag_near_window(grp, window, flagged, max_grp,
                                       text_embeddings, text_sim_threshold)
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
        # Mit Embeddings: nur nach Betrag+Konto gruppieren, Ähnlichkeit per Embedding
        # Ohne Embeddings: Fallback auf exakten Buchungstext-Match
        if has_embeddings:
            nokred_group_cols = ["_abs", "konto_soll"]
        else:
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
                self._flag_near_window(grp, window, flagged, max_grp,
                                       text_embeddings, text_sim_threshold)
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
        grp: pd.DataFrame,
        window: int,
        flagged: set,
        max_grp: int,
        text_embeddings: np.ndarray | None = None,
        text_sim_threshold: float = 0.0,
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
        # Same-day (0 Tage) ausschließen: Soll/Haben-Gegenbuchungen oder
        # Korrektur+Neubuchung am selben Tag sind kein echtes Duplikat
        close_mask = diffs.fillna(window + 1).between(1, window)

        # Beleg-intern: Paare mit gleicher _beleg_id überspringen
        if beleg_ids is not None:
            dated_beleg = dated["_beleg_id"].astype(str)
            prev_beleg = dated_beleg.shift(1)
            same_beleg = dated_beleg == prev_beleg
            close_mask = close_mask & (~same_beleg)

        # Text-Embedding-Similarity: nur flaggen wenn Buchungstexte ähnlich genug
        if text_embeddings is not None and text_sim_threshold > 0 and close_mask.any():
            # Die DataFrame-Positionen auf Embedding-Array-Positionen mappen
            # (Index im DF = Position im Embedding-Array, da Engine sequenziell
            # nummeriert und Embeddings in gleicher Reihenfolge erstellt)
            idx_arr = dated.index.to_numpy()
            curr_pos = np.arange(len(idx_arr))
            prev_pos = curr_pos - 1
            # Nur für close_mask == True die Similarity prüfen
            close_locs = np.where(close_mask.values)[0]
            if len(close_locs) > 0:
                curr_df_idx = idx_arr[close_locs]
                prev_df_idx = idx_arr[close_locs - 1]
                # Safety: Indices müssen gültig sein
                max_emb_idx = len(text_embeddings) - 1
                valid = (curr_df_idx <= max_emb_idx) & (prev_df_idx <= max_emb_idx) & (prev_df_idx >= 0)
                if valid.all():
                    sims = np.einsum(
                        "ij,ij->i",
                        text_embeddings[curr_df_idx],
                        text_embeddings[prev_df_idx],
                    )
                    # Mask out pairs below similarity threshold
                    low_sim = sims < text_sim_threshold
                    if low_sim.any():
                        close_vals = close_mask.values.copy()
                        close_vals[close_locs[low_sim]] = False
                        close_mask = pd.Series(close_vals, index=close_mask.index)

        prev_mask = close_mask.shift(-1, fill_value=False)
        flagged.update(dated.index[close_mask | prev_mask])


class DoppelteBelegnummer(AnomalyTest):
    name = "DOPPELTE_BELEGNUMMER"
    weight = 2.0
    critical = True
    required_columns = ["_betrag", "belegnummer", "konto_soll",
                        "_is_storno", "_beleg_id", "soll_haben"]

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
            grp_size = (
                sub.groupby(group_cols, sort=False, observed=True)["_beleg_id"]
                .transform("nunique")
            )
        else:
            grp_size = sub.groupby(group_cols, sort=False, observed=True).transform("size")
            grp_size = grp_size.iloc[:, 0] if isinstance(grp_size, pd.DataFrame) else grp_size

        # Reguläre Muster: grp_size auf 0 setzen
        if regular_beleg:
            for key, grp in sub.groupby(group_cols, sort=False, observed=True):
                k = key if isinstance(key, tuple) else (key,)
                if k in regular_beleg:
                    grp_size.loc[grp.index] = 0

        # Soll/Haben-Paare vektorisiert ausschließen: S+H mit gleicher Belegnr. = Paar
        sh = sub.get("soll_haben", pd.Series("", index=sub.index)).astype(str).str.strip().str.upper()
        has_sh_row = sh.isin(["S", "SOLL", "H", "HABEN"])
        if has_sh_row.any():
            is_soll = sh.isin(["S", "SOLL"])
            beleg_col = sub["belegnummer"]
            beleg_grp_size = beleg_col.groupby(beleg_col, observed=True).transform("size")
            sh_count = has_sh_row.astype(int).groupby(beleg_col, observed=True).transform("sum")
            soll_count = is_soll.astype(int).groupby(beleg_col, observed=True).transform("sum")
            is_pair = (beleg_grp_size == 2) & (sh_count == 2) & (soll_count == 1)
            grp_size.loc[is_pair] = 0

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
    required_columns = ["_abs", "_betrag", "_datum", "belegnummer", "kreditor",
                        "konto_soll", "_is_storno", "_beleg_id", "_kreditor_canonical"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        flagged: set[int] = set()
        beleg   = df["belegnummer"].astype(str).str.strip()
        # Kanonischer Kreditorname wenn vorhanden
        kred_col = "_kreditor_canonical" if "_kreditor_canonical" in df.columns else "kreditor"
        kred    = df[kred_col].astype(str).str.strip()
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
            df[has_all], [kred_col, "_abs"], reg_months,
            logger=self.log, label="beleg_kred_regular",
        )

        # ── Level 1: gleiche Belegnr + Kreditor + Betrag ──
        level1_count = 0
        level1_regular = 0
        if has_all.any():
            sub = df.loc[has_all]
            grp_counts = sub.groupby(
                ["belegnummer", kred_col, "_abs"], sort=False, observed=True
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
                kb = (str(df.at[idx, kred_col]).strip(), df.at[idx, "_abs"])
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

        # ── Level 2: gleicher Kreditor + Betrag + Datum ≤ window (vektorisiert) ──
        has_ka = (kred != "") & (df["_abs"] > 0) & (beleg != "") & (~is_storno)
        level2_count = 0
        grp_cols = [kred_col, "_abs"]

        sub2 = df.loc[has_ka & has_datum].copy()
        if len(sub2) >= 2:
            # Vektorisierte Gruppen-Statistiken für Pre-Filter
            sub2 = sub2.sort_values(grp_cols + ["_datum"])
            g = sub2.groupby(grp_cols, sort=False, observed=True)
            sub2["_grp_size"] = g[kred_col].transform("size")
            sub2["_n_beleg"] = g["belegnummer"].transform("nunique")
            sub2["_dom"] = sub2["_datum"].dt.day
            sub2["_n_dom"] = g["_dom"].transform("nunique")

            # Pre-Filter: size ∈ [2, max_grp], ≥ 3 versch. Belegnummern, > 1 Kalendertag
            keep = (
                (sub2["_grp_size"] >= 2)
                & (sub2["_grp_size"] <= max_grp)
                & (sub2["_n_beleg"] >= 3)
                & (sub2["_n_dom"] > 1)
            )

            # Reguläre Muster ausschließen
            if regular_kb:
                key_idx = pd.MultiIndex.from_frame(sub2[grp_cols])
                reg_idx = pd.MultiIndex.from_tuples(list(regular_kb), names=grp_cols)
                keep = keep & ~key_idx.isin(reg_idx)

            qs = sub2.loc[keep]

            if len(qs) >= 2:
                # Consecutive-pair-Erkennung innerhalb Gruppen
                prev_kred = qs[kred_col].shift(1)
                prev_abs = qs["_abs"].shift(1)
                same_group = (qs[kred_col] == prev_kred) & (qs["_abs"] == prev_abs)

                date_diff = qs["_datum"].diff().dt.days
                close = same_group & date_diff.between(0, window)

                # Verschiedene Belegnummer
                beleg_s = qs["belegnummer"].astype(str).str.strip()
                diff_beleg = beleg_s != beleg_s.shift(1)

                # Verschiedene _beleg_id
                if "_beleg_id" in qs.columns:
                    bid = qs["_beleg_id"].astype(str)
                    diff_bid = bid != bid.shift(1)
                else:
                    diff_bid = pd.Series(True, index=qs.index)

                close_mask = close & diff_beleg & diff_bid
                # Rückwärts-Propagation: auch den Partner flaggen
                prev_mask = close_mask.shift(-1, fill_value=False)
                level2_flagged = set(qs.index[close_mask | prev_mask])
                flagged.update(level2_flagged)
                level2_count = len(level2_flagged)

        self.log("Level 2 fertig", flagged=level2_count)
        self.metric("level2_flagged", level2_count)

        if flagged:
            df.loc[list(flagged), f"flag_{self.name}"] = True

        self.log("TOTAL", total_flagged=len(flagged))
        return len(flagged)


def get_tests() -> list[AnomalyTest]:
    return [NearDuplicate(), DoppelteBelegnummer(), BelegKreditorDuplikat()]
