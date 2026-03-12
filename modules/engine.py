"""
Buchungs-Anomalie Pre-Filter — Anomaly Engine v4.0

14 statistische Tests, vollständig vektorisiert (kein iterrows, kein O(n²)).
Boolean-Spalten statt Python-Listen in DataFrame-Zellen.

Tests (lt. Anomalie_Tests.xlsx + Phase 2):
    BETRAG_ZSCORE, BETRAG_IQR, NEAR_DUPLICATE, DOPPELTE_BELEGNUMMER,
    BELEG_KREDITOR_DUPLIKAT, STORNO, NEUER_KREDITOR_HOCH,
    KONTO_BETRAG_ANOMALIE, LEERER_BUCHUNGSTEXT, VELOCITY_ANOMALIE,
    RECHNUNGSDATUM_PERIODE, BUCHUNGSTEXT_PERIODE,
    MONATS_ENTWICKLUNG, FEHLENDE_MONATSBUCHUNG
"""

import logging
import re

import pandas as pd

from modules.parser import COLUMN_ALIASES, parse_german_number_series, parse_date_series

logger = logging.getLogger("prefilter")

# ── Weights & Config ─────────────────────────────────────────────────────────
WEIGHTS: dict[str, float] = {
    "BETRAG_ZSCORE":           2.0,
    "BETRAG_IQR":              1.5,
    "NEAR_DUPLICATE":          2.0,
    "DOPPELTE_BELEGNUMMER":    2.0,
    "BELEG_KREDITOR_DUPLIKAT": 2.5,
    "STORNO":                  1.5,
    "NEUER_KREDITOR_HOCH":     2.5,
    "KONTO_BETRAG_ANOMALIE":   2.0,
    "LEERER_BUCHUNGSTEXT":     1.0,
    "VELOCITY_ANOMALIE":       1.5,
    # Phase 2 — neue Tests
    "RECHNUNGSDATUM_PERIODE":  1.5,
    "BUCHUNGSTEXT_PERIODE":    1.0,
    "MONATS_ENTWICKLUNG":      1.5,
    "FEHLENDE_MONATSBUCHUNG":  1.0,
}

CRITICAL_FLAGS: set[str] = {
    "BETRAG_ZSCORE", "NEAR_DUPLICATE", "NEUER_KREDITOR_HOCH",
    "STORNO", "DOPPELTE_BELEGNUMMER", "BELEG_KREDITOR_DUPLIKAT",
}

OUTPUT_THRESHOLD = 2.0
MAX_OUTPUT_ROWS  = 1000
NUM_TESTS        = 14

_FLAG_NAMES = list(WEIGHTS.keys())


# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logs: list[str] = []
        self.flag_counts: dict[str, int] = {}
        self.stammdaten_report: dict[str, list] = {"fuzzy_kreditor_matches": []}
        self._prepare()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self.logs.append(msg)
        logger.info(msg)

    def _prepare(self) -> None:
        df = self.df
        for col in COLUMN_ALIASES:
            if col not in df.columns:
                df[col] = ""
        df.fillna("", inplace=True)

        # Vektorisiertes Parsing (kein .apply() per Zeile)
        df["_betrag"] = parse_german_number_series(df["betrag"]).fillna(0.0)
        df["_abs"]    = df["_betrag"].abs()
        df["_datum"]  = parse_date_series(df["datum"])
        df["_score"]  = 0.0

        # Boolean-Spalten statt Python-Listen in Zellen — das ist der Kern-Fix
        # für 24h+ Laufzeit. _flag_mask() / _flag() entfallen komplett.
        for name in _FLAG_NAMES:
            df[f"flag_{name}"] = False

        visible_cols = [c for c in df.columns if not c.startswith("_") and not c.startswith("flag_")]
        self._log(f"Geladen: {len(df)} Buchungen")
        self._log(f"Spalten: {', '.join(visible_cols)}")

    def _compute_scores(self) -> None:
        """Score = Σ(flag_X * weight_X) — vollständig vektorisiert."""
        score = pd.Series(0.0, index=self.df.index)
        for name, weight in WEIGHTS.items():
            score += self.df[f"flag_{name}"].astype(float) * weight
        self.df["_score"] = score

    def _stats(self) -> None:
        vals = self.df.loc[self.df["_abs"] > 0, "_abs"]
        self.b_mean  = float(vals.mean()) if len(vals) else 0.0
        self.b_std   = float(vals.std())  if len(vals) > 1 else 0.0
        q1 = float(vals.quantile(0.25)) if len(vals) else 0.0
        q3 = float(vals.quantile(0.75)) if len(vals) else 0.0
        self.b_iqr   = q3 - q1
        self.b_fence = q3 + 1.5 * self.b_iqr
        self._log(
            f"Beträge: n={len(vals)}, μ={self.b_mean:.0f}, σ={self.b_std:.0f}, "
            f"Q1={q1:.0f}, Q3={q3:.0f}, IQR={self.b_iqr:.0f}, Fence={self.b_fence:.0f}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 10 Tests — alle ohne iterrows(), kein O(n²)
    # ══════════════════════════════════════════════════════════════════════════

    def _t01_zscore(self) -> None:
        """Betrag Z-Score > 2.5 — vollständig vektorisiert."""
        df = self.df
        if self.b_std > 0:
            z    = (df["_abs"] - self.b_mean) / self.b_std
            mask = (df["_abs"] > 0) & (z > 2.5)
        else:
            mask = pd.Series(False, index=df.index)
        df.loc[mask, "flag_BETRAG_ZSCORE"] = True
        c = int(mask.sum())
        self.flag_counts["BETRAG_ZSCORE"] = c
        self._log(f"[01/{NUM_TESTS}] BETRAG_ZSCORE: {c}")

    def _t02_iqr(self) -> None:
        """Betrag > IQR-Fence — vollständig vektorisiert."""
        df = self.df
        if self.b_iqr > 0:
            mask = df["_abs"] > self.b_fence
        else:
            mask = pd.Series(False, index=df.index)
        df.loc[mask, "flag_BETRAG_IQR"] = True
        c = int(mask.sum())
        self.flag_counts["BETRAG_IQR"] = c
        self._log(f"[02/{NUM_TESTS}] BETRAG_IQR: {c}")

    def _t06_near_duplicate(self) -> None:
        """Gleicher Betrag + Konten innerhalb 3 Tagen.

        Sortiertes Sliding-Window pro Gruppe → O(n log n) statt O(n²).
        Fehlende Daten: nur mit anderen fehlenden Daten gematcht."""
        df      = self.df
        flagged: set[int] = set()

        for _, grp in df.groupby(["_abs", "konto_soll", "konto_haben"], sort=False):
            if len(grp) < 2:
                continue

            # Zeilen ohne Datum → gegenseitig flaggen wenn ≥2
            undated = grp[grp["_datum"].isna()]
            if len(undated) >= 2:
                flagged.update(undated.index)

            # Zeilen mit Datum → sortiertes Sliding-Window (3-Tage-Fenster)
            dated = grp[grp["_datum"].notna()].sort_values("_datum")
            if len(dated) < 2:
                continue
            idxs  = dated.index.tolist()
            dvals = dated["_datum"].tolist()
            for i in range(len(dvals)):
                for j in range(i + 1, len(dvals)):
                    if (dvals[j] - dvals[i]).days > 3:
                        break   # sortiert → restliche j ebenfalls > 3 Tage
                    flagged.add(idxs[i])
                    flagged.add(idxs[j])

        if flagged:
            df.loc[list(flagged), "flag_NEAR_DUPLICATE"] = True
        c = len(flagged)
        self.flag_counts["NEAR_DUPLICATE"] = c
        self._log(f"[03/{NUM_TESTS}] NEAR_DUPLICATE: {c}")

    def _t13_doppelte_belegnummer(self) -> None:
        """Doppelte Belegnummern — vollständig vektorisiert."""
        df    = self.df
        beleg = df["belegnummer"].astype(str).str.strip()
        mask  = (beleg != "") & beleg.duplicated(keep=False)
        df.loc[mask, "flag_DOPPELTE_BELEGNUMMER"] = True
        c = int(mask.sum())
        self.flag_counts["DOPPELTE_BELEGNUMMER"] = c
        self._log(f"[04/{NUM_TESTS}] DOPPELTE_BELEGNUMMER: {c}")

    def _t14_beleg_kreditor_duplikat(self) -> None:
        """Gleiche Belegnr.+Kreditor ODER gleicher Kreditor+Betrag innerhalb 7 Tagen.

        Level 1: vollständig vektorisiert.
        Level 2: sortiertes Sliding-Window pro (Kreditor, Betrag)-Gruppe → O(n log n)."""
        df      = self.df
        flagged: set[int] = set()
        beleg   = df["belegnummer"].astype(str).str.strip()
        kred    = df["kreditor"].astype(str).str.strip()

        # Level 1: gleiche Belegnr. + gleicher Kreditor (vektorisiert)
        has_both = (beleg != "") & (kred != "")
        if has_both.any():
            bk          = beleg + "|" + kred
            bk_dup_mask = has_both & bk.duplicated(keep=False)
            flagged.update(df.index[bk_dup_mask])

        # Level 2: gleicher Kreditor + gleicher Betrag + Datum ≤7 Tage
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
                    if (dvals[j] - dvals[i]).days > 7:
                        break
                    flagged.add(idxs[i])
                    flagged.add(idxs[j])

        if flagged:
            df.loc[list(flagged), "flag_BELEG_KREDITOR_DUPLIKAT"] = True
        c = len(flagged)
        self.flag_counts["BELEG_KREDITOR_DUPLIKAT"] = c
        self._log(f"[05/{NUM_TESTS}] BELEG_KREDITOR_DUPLIKAT: {c}")

    def _t15_storno(self) -> None:
        """Storno/Umkehrbuchungen — vollständig vektorisiert.
        'Gutschrift' nur flaggen wenn Betrag > μ+σ."""
        df                  = self.df
        txt                 = df["buchungstext"].astype(str).str.lower()
        gutschrift_schwelle = self.b_mean + self.b_std if self.b_std > 0 else float("inf")

        hard_mask       = txt.str.contains(
            r"storno|korrektur|r[uü]ckbuchung", regex=True, na=False
        )
        neg_mask        = df["_betrag"] < 0
        gutschrift_mask = (
            txt.str.contains("gutschrift", na=False)
            & (df["_abs"] > gutschrift_schwelle)
        )
        mask = hard_mask | neg_mask | gutschrift_mask
        df.loc[mask, "flag_STORNO"] = True
        c = int(mask.sum())
        self.flag_counts["STORNO"] = c
        self._log(f"[06/{NUM_TESTS}] STORNO: {c}  (Gutschrift-Schwelle: >{gutschrift_schwelle:.0f})")

    def _t16_neuer_kreditor_hoch(self) -> None:
        """Neuer Kreditor (≤2 Buchungen) mit hohem Betrag — vollständig vektorisiert."""
        df        = self.df
        has_kred  = df["kreditor"].astype(str).str.strip() != ""
        kred_cnt  = df.loc[has_kred, "kreditor"].value_counts()
        schwelle  = self.b_mean + 1.5 * self.b_std if self.b_std > 0 else float("inf")
        self._log(f"    Neuer-Kreditor Schwelle: ≤2 Buchungen + Betrag > {schwelle:.0f}")
        mapped   = df["kreditor"].map(kred_cnt).fillna(0)
        mask     = has_kred & (mapped <= 2) & (df["_abs"] > schwelle)
        df.loc[mask, "flag_NEUER_KREDITOR_HOCH"] = True
        c = int(mask.sum())
        self.flag_counts["NEUER_KREDITOR_HOCH"] = c
        self._log(f"[07/{NUM_TESTS}] NEUER_KREDITOR_HOCH: {c}")

    def _t18_konto_betrag_anomalie(self) -> None:
        """Betrag >3σ über Konto-Durchschnitt — vectorised Join."""
        df        = self.df
        has_konto = df["konto_soll"].astype(str).str.strip() != ""

        konto_stats = (
            df.loc[has_konto & (df["_abs"] > 0)]
            .groupby("konto_soll")["_abs"]
            .agg(konto_mean="mean", konto_std="std", konto_count="count")
        )
        konto_stats = konto_stats[
            (konto_stats["konto_count"] >= 5) & (konto_stats["konto_std"] > 0)
        ].copy()
        konto_stats["threshold"] = konto_stats["konto_mean"] + 3 * konto_stats["konto_std"]

        c = 0
        if not konto_stats.empty:
            df_tmp = df.join(
                konto_stats["threshold"].rename("_konto_thresh"), on="konto_soll"
            )
            mask = (
                has_konto
                & (df["_abs"] > 0)
                & (df["_abs"] > df_tmp["_konto_thresh"].fillna(float("inf")))
            )
            df.loc[mask, "flag_KONTO_BETRAG_ANOMALIE"] = True
            c = int(mask.sum())
        self.flag_counts["KONTO_BETRAG_ANOMALIE"] = c
        self._log(f"[08/{NUM_TESTS}] KONTO_BETRAG_ANOMALIE: {c}")

    def _t22_velocity_anomalie(self) -> None:
        """Kreditor mit >3× durchschnittlichen Monatsbuchungen — vektorisierter Join."""
        df            = self.df
        has_kred_date = (
            df["kreditor"].astype(str).str.strip() != ""
        ) & df["_datum"].notna()
        subset = df.loc[has_kred_date].copy()

        if subset.empty or len(subset) < 10:
            self.flag_counts["VELOCITY_ANOMALIE"] = 0
            self._log(f"[09/{NUM_TESTS}] VELOCITY_ANOMALIE: 0  (zu wenig Daten)")
            return

        subset["_ym"] = subset["_datum"].dt.to_period("M").astype(str)

        kred_month = (
            subset.groupby(["kreditor", "_ym"]).size().reset_index(name="cnt")
        )
        kred_stats = (
            kred_month.groupby("kreditor")["cnt"]
            .agg(km_mean="mean", km_std="std", n_months="count")
            .reset_index()
        )
        kred_stats = kred_stats[kred_stats["n_months"] >= 3].copy()
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
            spike_set = set(zip(spikes["kreditor"], spikes["_ym"]))
            self._log(f"    Velocity-Spikes: {spike_set}")
            subset["_key"]  = list(zip(subset["kreditor"], subset["_ym"]))
            flagged_sub     = subset[subset["_key"].isin(spike_set)]
            df.loc[flagged_sub.index, "flag_VELOCITY_ANOMALIE"] = True

        c = int(df["flag_VELOCITY_ANOMALIE"].sum())
        self.flag_counts["VELOCITY_ANOMALIE"] = c
        self._log(f"[09/{NUM_TESTS}] VELOCITY_ANOMALIE: {c}")

    def _t21_leerer_buchungstext(self) -> None:
        """Leerer oder generischer Buchungstext — vollständig vektorisiert."""
        df = self.df
        generic_patterns = {
            "diverse", "verschiedenes", "sonstiges", "test",
            "korrektur", "umbuchung", "xxx", "---", "...", "k.a.",
            "keine angabe", "n/a", "na", "tbd", "todo",
        }
        txt  = df["buchungstext"].astype(str).str.strip()
        mask = (txt == "") | (txt.str.lower().isin(generic_patterns)) | (txt.str.len() <= 2)
        df.loc[mask, "flag_LEERER_BUCHUNGSTEXT"] = True
        c = int(mask.sum())
        self.flag_counts["LEERER_BUCHUNGSTEXT"] = c
        self._log(f"[10/{NUM_TESTS}] LEERER_BUCHUNGSTEXT: {c}")

    # ── Phase-2-Tests ─────────────────────────────────────────────────────────

    def _t23_rechnungsdatum_periode(self) -> None:
        """Rechnungsdatum in anderer Periode als Buchungsdatum — vektorisiert.

        Nur aktiv wenn die Spalte 'rechnungsdatum' befüllt ist."""
        df = self.df
        rdatum = parse_date_series(df["rechnungsdatum"])
        has_both = df["_datum"].notna() & rdatum.notna()
        c = 0
        if has_both.any():
            buch_period = df.loc[has_both, "_datum"].dt.to_period("M")
            rech_period = rdatum.loc[has_both].dt.to_period("M")
            mask_inner  = rech_period != buch_period
            flagged_idx = buch_period.index[mask_inner]
            df.loc[flagged_idx, "flag_RECHNUNGSDATUM_PERIODE"] = True
            c = len(flagged_idx)
        self.flag_counts["RECHNUNGSDATUM_PERIODE"] = c
        self._log(f"[11/{NUM_TESTS}] RECHNUNGSDATUM_PERIODE: {c}")

    def _t24_buchungstext_periode(self) -> None:
        """Periodenangabe im Buchungstext stimmt nicht mit Buchungsdatum überein.

        Erkennt Muster wie '01/2024', '1.2024' im Text — vektorisiert."""
        df       = self.df
        has_date = df["_datum"].notna()
        c        = 0
        if has_date.any():
            txt = df["buchungstext"].astype(str)
            # Extrahiere MM/YYYY oder MM.YYYY aus Text (z.B. "Rechnung 01/2024")
            extracted = txt.str.extract(r'\b(\d{1,2})[/.](\d{4})\b', expand=True)
            ext_month = pd.to_numeric(extracted[0], errors="coerce")
            ext_year  = pd.to_numeric(extracted[1], errors="coerce")
            has_period = ext_month.notna() & ext_year.notna() & ext_month.between(1, 12)

            active = has_date & has_period
            if active.any():
                buch_month = df["_datum"].dt.month
                buch_year  = df["_datum"].dt.year
                mismatch   = (ext_month != buch_month) | (ext_year != buch_year)
                mask       = active & mismatch
                df.loc[mask, "flag_BUCHUNGSTEXT_PERIODE"] = True
                c = int(mask.sum())
        self.flag_counts["BUCHUNGSTEXT_PERIODE"] = c
        self._log(f"[12/{NUM_TESTS}] BUCHUNGSTEXT_PERIODE: {c}")

    def _t25_monats_entwicklung(self) -> None:
        """Ungewöhnliche Monatsveränderung auf Aufwands-/Ertragskonten.

        Filtert GuV-Konten (40000–79999), berechnet monatliche Summen je Konto
        und flaggt Buchungen in Ausreißer-Monaten (|Z| > 2.5) — vektorisiert."""
        df            = self.df
        has_konto_date = df["_datum"].notna() & (df["_abs"] > 0)
        subset        = df.loc[has_konto_date].copy()
        c             = 0

        if len(subset) < 10:
            self.flag_counts["MONATS_ENTWICKLUNG"] = 0
            self._log(f"[13/{NUM_TESTS}] MONATS_ENTWICKLUNG: 0  (zu wenig Daten)")
            return

        # GuV-Konten aus konto_soll filtern (40000–79999)
        konto_num = pd.to_numeric(
            subset["konto_soll"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
        pnl_mask = (konto_num >= 40000) & (konto_num <= 79999)
        pnl      = subset.loc[pnl_mask].copy()

        if len(pnl) < 10:
            self.flag_counts["MONATS_ENTWICKLUNG"] = 0
            self._log(f"[13/{NUM_TESTS}] MONATS_ENTWICKLUNG: 0  (zu wenig GuV-Buchungen)")
            return

        pnl["_ym"] = pnl["_datum"].dt.to_period("M").astype(str)

        # Monatssummen je Konto
        monthly = (
            pnl.groupby(["konto_soll", "_ym"])["_abs"]
            .sum()
            .reset_index(name="monatssumme")
        )
        konto_stats = (
            monthly.groupby("konto_soll")["monatssumme"]
            .agg(ks_mean="mean", ks_std="std", ks_count="count")
            .reset_index()
        )
        konto_stats = konto_stats[
            (konto_stats["ks_count"] >= 3) & (konto_stats["ks_std"] > 0)
        ]

        if konto_stats.empty:
            self.flag_counts["MONATS_ENTWICKLUNG"] = 0
            self._log(f"[13/{NUM_TESTS}] MONATS_ENTWICKLUNG: 0")
            return

        with_stats = monthly.merge(konto_stats[["konto_soll", "ks_mean", "ks_std"]], on="konto_soll")
        with_stats["zscore"] = (
            (with_stats["monatssumme"] - with_stats["ks_mean"]) / with_stats["ks_std"]
        )
        outliers   = with_stats[with_stats["zscore"].abs() > 2.5]
        spike_set  = set(zip(outliers["konto_soll"], outliers["_ym"]))

        if spike_set:
            self._log(f"    Monats-Ausreißer: {len(spike_set)} (Konto, Monat)-Paare")
            pnl["_key"]    = list(zip(pnl["konto_soll"], pnl["_ym"]))
            flagged_sub    = pnl[pnl["_key"].isin(spike_set)]
            df.loc[flagged_sub.index, "flag_MONATS_ENTWICKLUNG"] = True
            c = int(df["flag_MONATS_ENTWICKLUNG"].sum())

        self.flag_counts["MONATS_ENTWICKLUNG"] = c
        self._log(f"[13/{NUM_TESTS}] MONATS_ENTWICKLUNG: {c}")

    def _t26_fehlende_monatsbuchung(self) -> None:
        """Regelmäßige Konten mit fehlendem Monat — Nachbarschafts-Flag.

        Für Konten die in ≥60% aller Monate buchen:
        Fehlt ein Monat → Buchungen des Vor- / Nachmonats werden flaggt."""
        df            = self.df
        has_konto_date = (
            df["_datum"].notna()
            & (df["konto_soll"].astype(str).str.strip() != "")
        )
        subset = df.loc[has_konto_date].copy()
        c      = 0

        if len(subset) < 10:
            self.flag_counts["FEHLENDE_MONATSBUCHUNG"] = 0
            self._log(f"[14/{NUM_TESTS}] FEHLENDE_MONATSBUCHUNG: 0  (zu wenig Daten)")
            return

        subset["_ym"] = subset["_datum"].dt.to_period("M")
        all_periods   = sorted(subset["_ym"].unique())
        if len(all_periods) < 3:
            self.flag_counts["FEHLENDE_MONATSBUCHUNG"] = 0
            self._log(f"[14/{NUM_TESTS}] FEHLENDE_MONATSBUCHUNG: 0  (zu wenig Monate)")
            return

        # Konten die in ≥60% aller Monate buchen
        konto_month_cnt = subset.groupby("konto_soll")["_ym"].nunique()
        min_active      = max(3, len(all_periods) * 0.6)
        regular         = konto_month_cnt[konto_month_cnt >= min_active].index

        if regular.empty:
            self.flag_counts["FEHLENDE_MONATSBUCHUNG"] = 0
            self._log(f"[14/{NUM_TESTS}] FEHLENDE_MONATSBUCHUNG: 0  (keine regulären Konten)")
            return

        # Nachbarschafts-Index über die VOLLE Zeitspanne — nur so werden echte Lücken erkannt
        full_range = list(pd.period_range(all_periods[0], all_periods[-1], freq="M"))
        prev_of = {p: full_range[i - 1] for i, p in enumerate(full_range) if i > 0}
        next_of = {p: full_range[i + 1] for i, p in enumerate(full_range[:-1])}

        flagged: set[int] = set()
        for konto in regular:
            konto_data  = subset[subset["konto_soll"] == konto]
            booked      = set(konto_data["_ym"])
            idx_by_ym   = konto_data.groupby("_ym").groups  # {period: [idx, ...]}

            for period in full_range:
                if period not in booked:
                    for adj in (prev_of.get(period), next_of.get(period)):
                        if adj and adj in idx_by_ym:
                            flagged.update(idx_by_ym[adj])

        if flagged:
            df.loc[list(flagged), "flag_FEHLENDE_MONATSBUCHUNG"] = True
            c = len(flagged)

        self.flag_counts["FEHLENDE_MONATSBUCHUNG"] = c
        self._log(f"[14/{NUM_TESTS}] FEHLENDE_MONATSBUCHUNG: {c}")

    # ── run all ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        self._stats()
        self._t01_zscore()
        self._t02_iqr()
        self._t06_near_duplicate()
        self._t13_doppelte_belegnummer()
        self._t14_beleg_kreditor_duplikat()
        self._t15_storno()
        self._t16_neuer_kreditor_hoch()
        self._t18_konto_betrag_anomalie()
        self._t22_velocity_anomalie()
        self._t21_leerer_buchungstext()
        self._t23_rechnungsdatum_periode()
        self._t24_buchungstext_periode()
        self._t25_monats_entwicklung()
        self._t26_fehlende_monatsbuchung()
        self._compute_scores()
        return self._export()

    def _export(self) -> dict:
        df = self.df

        # Kritische-Flag-Maske vollständig vektorisiert
        crit_cols    = [f"flag_{n}" for n in CRITICAL_FLAGS if f"flag_{n}" in df.columns]
        critical_mask = (
            df[crit_cols].any(axis=1) if crit_cols
            else pd.Series(False, index=df.index)
        )
        output_mask  = (df["_score"] >= OUTPUT_THRESHOLD) | critical_mask
        verdaechtig  = df[output_mask].sort_values("_score", ascending=False).head(MAX_OUTPUT_ROWS)

        # Flag-Strings vektorisiert aufbauen (nur über gefilterte Teilmenge)
        flag_str_series = verdaechtig.apply(
            lambda row: "|".join(n for n in _FLAG_NAMES if row.get(f"flag_{n}", False)),
            axis=1,
        )

        out_cols = [
            "datum", "konto_soll", "konto_haben", "betrag",
            "buchungstext", "belegnummer", "kostenstelle", "kreditor", "erfasser",
        ]
        rows: list[dict] = []
        for i, row in verdaechtig.iterrows():
            r = {c: str(row.get(c, "")) for c in out_cols}
            r["betrag"]        = row["_betrag"]
            r["anomaly_score"] = round(row["_score"], 2)
            r["anomaly_flags"] = flag_str_series.at[i]
            if pd.notna(row["_datum"]):
                r["datum"] = row["_datum"].strftime("%Y-%m-%d")
            rows.append(r)

        total       = len(df)
        n_verd      = int(output_mask.sum())
        avg_score   = round(float(df["_score"].mean()), 2) if total > 0 else 0.0
        total_flags = int(sum(int(df[f"flag_{n}"].sum()) for n in _FLAG_NAMES))
        pct         = n_verd / total * 100 if total > 0 else 0.0

        top_flags = sorted(
            ((k, v) for k, v in self.flag_counts.items() if v > 0),
            key=lambda x: -x[1],
        )
        summary_lines = [
            f"ERGEBNIS: {n_verd} von {total} verdächtig ({pct:.1f}%)",
            f"Flags gesamt: {total_flags}, Ø Score: {avg_score}",
            f"Top-Flags: {', '.join(f'{k}:{v}' for k, v in top_flags) or 'keine'}",
        ]
        if rows:
            summary_lines.append(
                f"Höchster Score: {rows[0]['anomaly_score']} ({rows[0]['belegnummer']})"
            )
        for line in summary_lines:
            self._log(line)

        return {
            "message": f"{n_verd} verdächtige Buchungen ({pct:.1f}%)",
            "statistics": {
                "total_input":  total,
                "total_output": len(rows),
                "filter_ratio": f"{pct:.1f}%",
                "avg_score":    avg_score,
                "flag_counts":  self.flag_counts,
            },
            "verdaechtige_buchungen": rows,
            "stammdaten_report":      self.stammdaten_report,
            "logs":                   self.logs,
        }
