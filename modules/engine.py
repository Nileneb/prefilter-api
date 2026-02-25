"""
Buchungs-Anomalie Pre-Filter — Anomaly Engine v3.0

21 statistische Tests, vektorisiert wo möglich.
Upgrades: RapidFuzz, Benford 2-Ziffern + MAD, erweiterte Belegnummern,
          Schwellenwert-Cluster, Velocity Check, leere Buchungstexte.
"""

import re
import math
import logging
from datetime import timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process as rf_process
from rapidfuzz.distance import JaroWinkler

from modules.parser import COLUMN_ALIASES, parse_german_number, parse_date

logger = logging.getLogger("prefilter")

# ── Weights & Config ────────────────────────────────────────
# Note: FUZZY_KREDITOR is a Stammdaten-level finding (reported
# once per name pair), NOT a per-booking flag.  It therefore has
# no weight here and does not inflate individual scores.
WEIGHTS = {
    "BETRAG_ZSCORE":             2.0,
    "BETRAG_IQR":                1.5,
    "SELTENE_KONTIERUNG":        1.5,
    "WOCHENENDE":                1.0,
    "AUSSERHALB_GESCHAEFTSZEIT": 1.5,
    "MONATSENDE":                0.25,
    "QUARTALSENDE":              0.5,
    "NEAR_DUPLICATE":            2.0,
    "BENFORD_1ZIFFER":           1.0,
    "BENFORD_2ZIFFERN":          1.5,
    "RUNDER_BETRAG":             1.0,
    "ERFASSER_ANOMALIE":         1.5,
    "SPLIT_VERDACHT":            2.0,
    "SCHWELLENWERT_CLUSTER":     2.0,
    "BELEG_LUECKE":              1.0,
    "DOPPELTE_BELEGNUMMER":      2.0,
    "BELEG_KREDITOR_DUPLIKAT":   2.5,
    "STORNO":                    1.5,
    "NEUER_KREDITOR_HOCH":       2.5,
    "SOLL_GLEICH_HABEN":         2.0,
    "KONTO_BETRAG_ANOMALIE":     2.0,
    "TEXT_KREDITOR_MISMATCH":     1.5,
    "LEERER_BUCHUNGSTEXT":       1.0,
    "VELOCITY_ANOMALIE":         1.5,
}

CRITICAL_FLAGS = {
    "BETRAG_ZSCORE", "NEAR_DUPLICATE", "SPLIT_VERDACHT",
    "NEUER_KREDITOR_HOCH", "STORNO", "DOPPELTE_BELEGNUMMER",
    "BELEG_KREDITOR_DUPLIKAT", "SOLL_GLEICH_HABEN",
    "SCHWELLENWERT_CLUSTER",
}

# Configurable approval thresholds for split-detection
APPROVAL_THRESHOLDS = [5000, 10000, 25000, 50000]

# Rechtsformen to strip for fuzzy vendor matching
RECHTSFORMEN = [
    "gmbh", "gmbh & co. kg", "gmbh & co kg", "gmbh&co.kg",
    "ag", "e.v.", "ev", "e.v", "kg", "ohg", "gbr", "ug",
    "mbh", "co.", "inc.", "ltd.", "ltd", "se", "sa",
    "& co.", "&co.", "co", "corp.", "corp",
]

OUTPUT_THRESHOLD = 2.0
MAX_OUTPUT_ROWS  = 1000

NUM_TESTS = 21


def _norm_vendor(name: str) -> str:
    """Normalise vendor name for fuzzy comparison."""
    s = name.lower().strip()
    # Umlaute
    for old, new in [("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")]:
        s = s.replace(old, new)
    # Remove Rechtsformen
    for rf in sorted(RECHTSFORMEN, key=len, reverse=True):
        s = s.replace(rf, " ")
    # Collapse whitespace and punctuation
    s = re.sub(r"[^a-z0-9]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ══════════════════════════════════════════════════════════════
# ANOMALY ENGINE
# ══════════════════════════════════════════════════════════════
# Stopwords for TEXT_KREDITOR_MISMATCH — generic booking texts that
# naturally appear across many creditors and should not trigger alerts.
TEXT_STOPWORDS = {
    "rechnung", "miete", "wartung", "zahlung", "gutschrift",
    "überweisung", "ueberweisung", "dauerauftrag", "abbuchung",
    "lastschrift", "gehalt", "lohn", "provision", "honorar",
    "abrechnung", "ratenzahlung", "vorauszahlung", "anzahlung",
    "kaution", "beitrag", "mitgliedsbeitrag", "spende",
    "erstattung", "rückerstattung", "rueckerstattung",
    "porto", "versand", "fracht", "nebenkosten",
    "strom", "gas", "wasser", "telefon", "internet",
    "reinigung", "entsorgung", "reparatur", "service",
    "beratung", "schulung", "lizenz", "miete büro",
    "büromaterial", "bueromaterial", "verbrauchsmaterial",
}

# Minimum text length for TEXT_KREDITOR_MISMATCH analysis
TEXT_MISMATCH_MIN_LEN = 15


class AnomalyEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logs: list[str] = []
        self.flag_counts: dict[str, int] = {}
        self.stammdaten_report: dict[str, list] = {
            "fuzzy_kreditor_matches": [],
        }
        self._prepare()

    # ── helpers ──────────────────────────────────────────────
    def _log(self, msg: str):
        self.logs.append(msg)
        logger.info(msg)

    def _flag(self, idx, name: str):
        w = WEIGHTS.get(name, 1.0)
        flags: list = self.df.at[idx, "_flags"]
        if name not in flags:
            flags.append(name)
            self.df.at[idx, "_score"] += w

    def _flag_mask(self, mask: pd.Series, name: str):
        """Vectorised flag: apply flag to all rows where mask is True."""
        w = WEIGHTS.get(name, 1.0)
        for idx in self.df.index[mask]:
            flags: list = self.df.at[idx, "_flags"]
            if name not in flags:
                flags.append(name)
                self.df.at[idx, "_score"] += w

    def _prepare(self):
        df = self.df
        for col in COLUMN_ALIASES:
            if col not in df.columns:
                df[col] = ""
        df.fillna("", inplace=True)
        df["_betrag"] = df["betrag"].apply(parse_german_number)
        df["_abs"]    = df["_betrag"].abs()
        df["_datum"]  = df["datum"].apply(parse_date)
        df["_score"]  = 0.0
        df["_flags"]  = [[] for _ in range(len(df))]
        self._log(f"Geladen: {len(df)} Buchungen")
        self._log(f"Spalten: {', '.join(c for c in df.columns if not c.startswith('_'))}")

    def _stats(self):
        vals = self.df.loc[self.df["_abs"] > 0, "_abs"]
        self.b_mean  = vals.mean() if len(vals) else 0
        self.b_std   = vals.std()  if len(vals) > 1 else 0
        q1 = vals.quantile(0.25) if len(vals) else 0
        q3 = vals.quantile(0.75) if len(vals) else 0
        self.b_iqr   = q3 - q1
        self.b_fence = q3 + 1.5 * self.b_iqr
        self._log(f"Beträge: n={len(vals)}, μ={self.b_mean:.0f}, σ={self.b_std:.0f}, "
                  f"Q1={q1:.0f}, Q3={q3:.0f}, IQR={self.b_iqr:.0f}, Fence={self.b_fence:.0f}")

    # ══════════════════════════════════════════════════════════
    # 21 Tests
    # ══════════════════════════════════════════════════════════

    def _t01_zscore(self):
        """Betrag Z-Score > 2.5 — vectorised."""
        df = self.df
        if self.b_std > 0:
            z = (df["_abs"] - self.b_mean) / self.b_std
            mask = (df["_abs"] > 0) & (z > 2.5)
        else:
            mask = pd.Series(False, index=df.index)
        self._flag_mask(mask, "BETRAG_ZSCORE")
        c = int(mask.sum())
        self.flag_counts["BETRAG_ZSCORE"] = c
        self._log(f"[01/{NUM_TESTS}] BETRAG_ZSCORE: {c}")

    def _t02_iqr(self):
        """Betrag > IQR fence — vectorised."""
        df = self.df
        if self.b_iqr > 0:
            mask = df["_abs"] > self.b_fence
        else:
            mask = pd.Series(False, index=df.index)
        self._flag_mask(mask, "BETRAG_IQR")
        c = int(mask.sum())
        self.flag_counts["BETRAG_IQR"] = c
        self._log(f"[02/{NUM_TESTS}] BETRAG_IQR: {c}")

    def _t03_seltene_kontierung(self):
        """Rare account pair — vectorised."""
        df = self.df
        df["_konto_pair"] = df["konto_soll"].astype(str) + "→" + df["konto_haben"].astype(str)
        counts = df["_konto_pair"].value_counts()
        threshold = max(2, math.ceil(len(df) * 0.01))
        mapped = df["_konto_pair"].map(counts)
        mask = mapped <= threshold
        self._flag_mask(mask, "SELTENE_KONTIERUNG")
        c = int(mask.sum())
        self.flag_counts["SELTENE_KONTIERUNG"] = c
        self._log(f"[03/{NUM_TESTS}] SELTENE_KONTIERUNG: {c}  (Schwelle: ≤{threshold})")

    def _t04_zeitlich(self):
        """Weekend, month-end, quarter-end — vectorised."""
        df = self.df
        has_date = df["_datum"].notna()
        weekdays = df.loc[has_date, "_datum"].apply(lambda d: d.weekday())

        we_mask = pd.Series(False, index=df.index)
        we_mask.loc[has_date] = weekdays >= 5
        self._flag_mask(we_mask, "WOCHENENDE")

        def _is_month_end(d):
            last = (d.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            return d.day >= last.day - 2

        me_mask = pd.Series(False, index=df.index, dtype=object)
        me_vals = df.loc[has_date, "_datum"].apply(_is_month_end)
        for idx in me_vals.index:
            me_mask.at[idx] = bool(me_vals.at[idx])
        me_mask = me_mask.astype(bool)
        self._flag_mask(me_mask, "MONATSENDE")

        qe_mask = me_mask & df["_datum"].apply(
            lambda d: d.month in (3, 6, 9, 12) if d else False
        )
        self._flag_mask(qe_mask, "QUARTALSENDE")

        self.flag_counts["WOCHENENDE"]   = int(we_mask.sum())
        self.flag_counts["MONATSENDE"]   = int(me_mask.sum())
        self.flag_counts["QUARTALSENDE"] = int(qe_mask.sum())
        self._log(f"[04/{NUM_TESTS}] WOCHENENDE: {self.flag_counts['WOCHENENDE']}, "
                  f"MONATSENDE: {self.flag_counts['MONATSENDE']}, "
                  f"QUARTALSENDE: {self.flag_counts['QUARTALSENDE']}")

    def _t05_ausserhalb_geschaeftszeit(self):
        """Buchungen außerhalb 6:00-22:00."""
        df = self.df
        c = 0
        for i, row in df.iterrows():
            d = row["_datum"]
            if d is not None and (d.hour < 6 or d.hour >= 22) and \
               (d.hour != 0 or d.minute != 0 or d.second != 0):
                self._flag(i, "AUSSERHALB_GESCHAEFTSZEIT")
                c += 1
        self.flag_counts["AUSSERHALB_GESCHAEFTSZEIT"] = c
        self._log(f"[05/{NUM_TESTS}] AUSSERHALB_GESCHAEFTSZEIT: {c}")

    def _t06_near_duplicate(self):
        """Near-duplicate: same amount+accounts within 3 days.
        Counts unique flagged bookings. Missing date only matches if both missing."""
        df = self.df
        flagged = set()
        groups: dict[str, list] = {}
        for i, row in df.iterrows():
            key = f"{row['_abs']}|{row['konto_soll']}|{row['konto_haben']}"
            groups.setdefault(key, []).append(i)

        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    ia, ib = idxs[a], idxs[b]
                    da, db = df.at[ia, "_datum"], df.at[ib, "_datum"]
                    is_dup = False
                    if da is not None and db is not None:
                        if abs((da - db).days) <= 3:
                            is_dup = True
                    elif da is None and db is None:
                        is_dup = True
                    if is_dup:
                        self._flag(ia, "NEAR_DUPLICATE")
                        self._flag(ib, "NEAR_DUPLICATE")
                        flagged.add(ia)
                        flagged.add(ib)

        c = len(flagged)
        self.flag_counts["NEAR_DUPLICATE"] = c
        self._log(f"[06/{NUM_TESTS}] NEAR_DUPLICATE: {c}")

    def _t07_benford(self):
        """Benford's law — 1-digit + 2-digit test with MAD (Nigrini).

        Amounts >10 only. MAD thresholds per Nigrini:
        1-digit: >0.015 = nonconformity
        2-digit: >0.012 = nonconformity"""
        df = self.df
        benford_vals = df.loc[df["_abs"] > 10, "_abs"]

        # ── 1-digit test ──
        expected_1 = {d: math.log10(1 + 1/d) for d in range(1, 10)}
        fd1_counts = {d: 0 for d in range(1, 10)}
        total_1 = 0
        for val in benford_vals:
            first = int(str(int(val))[0])
            if 1 <= first <= 9:
                fd1_counts[first] += 1
                total_1 += 1

        c1 = 0
        deviant_1 = set()
        mad_1 = 0.0
        if total_1 > 50:
            deviations = []
            for d in range(1, 10):
                dev = abs(fd1_counts[d] / total_1 - expected_1[d])
                deviations.append(dev)
                if dev > 0.08:
                    deviant_1.add(d)
            mad_1 = sum(deviations) / len(deviations)

            if mad_1 > 0.015 and deviant_1:
                self._log(f"    Benford 1-Ziffer MAD={mad_1:.4f}, Abweichungen: {deviant_1}")
                for i, row in df.iterrows():
                    if row["_abs"] > 10:
                        first = int(str(int(row["_abs"]))[0])
                        if first in deviant_1:
                            self._flag(i, "BENFORD_1ZIFFER")
                            c1 += 1
        self.flag_counts["BENFORD_1ZIFFER"] = c1
        self._log(f"[07/{NUM_TESTS}] BENFORD_1ZIFFER: {c1}  (MAD={mad_1:.4f}, n={total_1})")

        # ── 2-digit test ──
        expected_2 = {}
        for d in range(10, 100):
            expected_2[d] = math.log10(1 + 1/d)
        fd2_counts = {d: 0 for d in range(10, 100)}
        total_2 = 0
        for val in benford_vals:
            s = str(int(val))
            if len(s) >= 2:
                two = int(s[:2])
                if 10 <= two <= 99:
                    fd2_counts[two] += 1
                    total_2 += 1

        c2 = 0
        deviant_2 = set()
        mad_2 = 0.0
        if total_2 > 100:
            deviations = []
            for d in range(10, 100):
                dev = abs(fd2_counts[d] / total_2 - expected_2[d])
                deviations.append(dev)
                if dev > 0.03 and fd2_counts[d] / total_2 > 2 * expected_2[d]:
                    deviant_2.add(d)
            mad_2 = sum(deviations) / len(deviations)

            if mad_2 > 0.012 and deviant_2:
                self._log(f"    Benford 2-Ziffern MAD={mad_2:.4f}, Abweichungen: "
                          f"{sorted(deviant_2)}")
                for i, row in df.iterrows():
                    if row["_abs"] > 10:
                        s = str(int(row["_abs"]))
                        if len(s) >= 2:
                            two = int(s[:2])
                            if two in deviant_2:
                                self._flag(i, "BENFORD_2ZIFFERN")
                                c2 += 1
        self.flag_counts["BENFORD_2ZIFFERN"] = c2
        self._log(f"[08/{NUM_TESTS}] BENFORD_2ZIFFERN: {c2}  (MAD={mad_2:.4f}, n={total_2})")

    def _t08_runder_betrag(self):
        """Round amounts — vectorised."""
        df = self.df
        mask = ((df["_abs"] >= 5000) & (df["_abs"] % 1000 == 0)) | \
               ((df["_abs"] >= 1000) & (df["_abs"] % 500 == 0))
        self._flag_mask(mask, "RUNDER_BETRAG")
        c = int(mask.sum())
        self.flag_counts["RUNDER_BETRAG"] = c
        self._log(f"[09/{NUM_TESTS}] RUNDER_BETRAG: {c}")

    def _t09_erfasser(self):
        """Rare user (≤3% of bookings) — vectorised."""
        df = self.df
        erf_counts = df.loc[df["erfasser"] != "", "erfasser"].value_counts()
        erf_total = erf_counts.sum()
        threshold = max(3, math.ceil(erf_total * 0.03))
        has_erf = df["erfasser"] != ""
        mapped = df["erfasser"].map(erf_counts).fillna(0)
        mask = has_erf & (mapped <= threshold)
        self._flag_mask(mask, "ERFASSER_ANOMALIE")
        c = int(mask.sum())
        self.flag_counts["ERFASSER_ANOMALIE"] = c
        self._log(f"[10/{NUM_TESTS}] ERFASSER_ANOMALIE: {c}  (Schwelle: ≤{threshold})")

    def _t10_split(self):
        """Split booking suspicion (≥3 bookings same day/actor/account)."""
        df = self.df
        c = 0
        groups: dict[str, list] = {}
        for i, row in df.iterrows():
            d = row["_datum"]
            dk = d.strftime("%Y-%m-%d") if d else "nodate"
            actor = row["kreditor"] or row["erfasser"] or "unknown"
            key = f"{dk}|{actor}|{row['konto_soll']}"
            groups.setdefault(key, []).append(i)
        for idxs in groups.values():
            if len(idxs) >= 3:
                for i in idxs:
                    self._flag(i, "SPLIT_VERDACHT")
                    c += 1
        self.flag_counts["SPLIT_VERDACHT"] = c
        self._log(f"[11/{NUM_TESTS}] SPLIT_VERDACHT: {c}")

    def _t11_schwellenwert_cluster(self):
        """Cluster of amounts just below approval thresholds.

        Flags when >5% of bookings fall in the 90-100% band below
        a configurable threshold (e.g. 4500-5000 for a 5000 limit)."""
        df = self.df
        total_with_amount = int((df["_abs"] > 0).sum())
        if total_with_amount < 20:
            self.flag_counts["SCHWELLENWERT_CLUSTER"] = 0
            self._log(f"[12/{NUM_TESTS}] SCHWELLENWERT_CLUSTER: 0  (zu wenig Daten)")
            return

        c = 0
        flagged_thresholds = []
        for threshold in APPROVAL_THRESHOLDS:
            lower = threshold * 0.90
            band_mask = (df["_abs"] >= lower) & (df["_abs"] < threshold)
            band_count = int(band_mask.sum())
            band_pct = band_count / total_with_amount * 100

            if band_count >= 3 and band_pct > 5:
                flagged_thresholds.append(f"{threshold} ({band_count}x, {band_pct:.1f}%)")
                self._flag_mask(band_mask, "SCHWELLENWERT_CLUSTER")
                c += band_count

        self.flag_counts["SCHWELLENWERT_CLUSTER"] = c
        if flagged_thresholds:
            self._log(f"    Schwellenwerte: {', '.join(flagged_thresholds)}")
        self._log(f"[12/{NUM_TESTS}] SCHWELLENWERT_CLUSTER: {c}")

    def _t12_beleg_luecke(self):
        """Gaps >5 in sequential document numbers."""
        entries = []
        for i, row in self.df.iterrows():
            m = re.search(r"(\d+)", str(row["belegnummer"]))
            if m:
                entries.append((int(m.group(1)), i))
        entries.sort()
        c = 0
        if len(entries) > 10:
            for j in range(1, len(entries)):
                gap = entries[j][0] - entries[j - 1][0]
                if gap > 5:
                    self._flag(entries[j - 1][1], "BELEG_LUECKE")
                    self._flag(entries[j][1], "BELEG_LUECKE")
                    c += 1
        self.flag_counts["BELEG_LUECKE"] = c
        self._log(f"[13/{NUM_TESTS}] BELEG_LUECKE: {c}")

    def _t13_doppelte_belegnummer(self):
        """Duplicate document numbers — exact duplicates."""
        df = self.df
        beleg = df["belegnummer"].astype(str).str.strip()
        has_beleg = beleg != ""
        dups = beleg[has_beleg & beleg.duplicated(keep=False)]
        c = 0
        for idx in dups.index:
            self._flag(idx, "DOPPELTE_BELEGNUMMER")
            c += 1
        self.flag_counts["DOPPELTE_BELEGNUMMER"] = c
        self._log(f"[14/{NUM_TESTS}] DOPPELTE_BELEGNUMMER: {c}")

    def _t14_beleg_kreditor_duplikat(self):
        """Same doc# + same creditor, OR same creditor + same amount + date ≤7d.

        Two-level check for likely double payments."""
        df = self.df
        c = 0
        flagged = set()

        # Level 1: same belegnummer + same kreditor
        has_both = (df["belegnummer"].astype(str).str.strip() != "") & \
                   (df["kreditor"].astype(str).str.strip() != "")
        subset = df.loc[has_both].copy()
        if not subset.empty:
            subset["_bk"] = subset["belegnummer"].astype(str).str.strip() + "|" + \
                            subset["kreditor"].astype(str).str.strip()
            bk_dups = subset["_bk"][subset["_bk"].duplicated(keep=False)]
            for idx in bk_dups.index:
                self._flag(idx, "BELEG_KREDITOR_DUPLIKAT")
                flagged.add(idx)

        # Level 2: same kreditor + same amount + date within 7 days
        has_kred_amt = (df["kreditor"].astype(str).str.strip() != "") & (df["_abs"] > 0)
        kred_groups: dict[str, list] = {}
        for i, row in df.loc[has_kred_amt].iterrows():
            key = f"{row['kreditor']}|{row['_abs']}"
            kred_groups.setdefault(key, []).append(i)

        for idxs in kred_groups.values():
            if len(idxs) < 2:
                continue
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    ia, ib = idxs[a], idxs[b]
                    da, db = df.at[ia, "_datum"], df.at[ib, "_datum"]
                    if da is not None and db is not None and abs((da - db).days) <= 7:
                        if ia not in flagged:
                            self._flag(ia, "BELEG_KREDITOR_DUPLIKAT")
                            flagged.add(ia)
                        if ib not in flagged:
                            self._flag(ib, "BELEG_KREDITOR_DUPLIKAT")
                            flagged.add(ib)

        c = len(flagged)
        self.flag_counts["BELEG_KREDITOR_DUPLIKAT"] = c
        self._log(f"[15/{NUM_TESTS}] BELEG_KREDITOR_DUPLIKAT: {c}")

    def _t15_storno(self):
        """Storno/reversal — 'gutschrift' only flagged if amount unusual."""
        df = self.df
        hard_patterns = ["storno", "korrektur", "rückbuchung", "rueckbuchung"]
        c = 0
        gutschrift_schwelle = self.b_mean + self.b_std if self.b_std > 0 else float("inf")

        for i, row in df.iterrows():
            txt = str(row["buchungstext"]).lower()
            is_storno = False
            if any(p in txt for p in hard_patterns) or row["_betrag"] < 0:
                is_storno = True
            elif "gutschrift" in txt and row["_abs"] > gutschrift_schwelle:
                is_storno = True
            if is_storno:
                self._flag(i, "STORNO")
                c += 1
        self.flag_counts["STORNO"] = c
        self._log(f"[16/{NUM_TESTS}] STORNO: {c}  (Gutschrift-Schwelle: >{gutschrift_schwelle:.0f})")

    def _t16_neuer_kreditor_hoch(self):
        """New creditor (≤2 bookings) with high amount."""
        df = self.df
        kred_counts = df.loc[df["kreditor"] != "", "kreditor"].value_counts()
        schwelle = self.b_mean + 1.5 * self.b_std if self.b_std > 0 else float("inf")
        self._log(f"    Neuer-Kreditor Schwelle: ≤2 Buchungen + Betrag > {schwelle:.0f}")
        c = 0
        for i, row in df.iterrows():
            k = row["kreditor"]
            if k and kred_counts.get(k, 0) <= 2 and row["_abs"] > schwelle:
                self._flag(i, "NEUER_KREDITOR_HOCH")
                c += 1
        self.flag_counts["NEUER_KREDITOR_HOCH"] = c
        self._log(f"[17/{NUM_TESTS}] NEUER_KREDITOR_HOCH: {c}")

    def _t17_soll_gleich_haben(self):
        """Debit account = Credit account."""
        df = self.df
        soll = df["konto_soll"].astype(str).str.strip()
        haben = df["konto_haben"].astype(str).str.strip()
        has_both = (soll != "") & (haben != "")
        mask = has_both & (soll == haben)
        self._flag_mask(mask, "SOLL_GLEICH_HABEN")
        c = int(mask.sum())
        self.flag_counts["SOLL_GLEICH_HABEN"] = c
        self._log(f"[18/{NUM_TESTS}] SOLL_GLEICH_HABEN: {c}")

    def _t18_konto_betrag_anomalie(self):
        """Per-account amount plausibility (>3σ above account mean)."""
        df = self.df
        has_konto = df["konto_soll"].astype(str).str.strip() != ""
        konto_groups = df.loc[has_konto & (df["_abs"] > 0)].groupby("konto_soll")["_abs"]
        konto_stats = konto_groups.agg(["mean", "std", "count"])
        konto_stats = konto_stats[konto_stats["count"] >= 5]

        c = 0
        for konto, stats in konto_stats.iterrows():
            if stats["std"] > 0:
                threshold = stats["mean"] + 3 * stats["std"]
                konto_mask = (df["konto_soll"] == konto) & (df["_abs"] > threshold)
                for idx in df.index[konto_mask]:
                    self._flag(idx, "KONTO_BETRAG_ANOMALIE")
                    c += 1
        self.flag_counts["KONTO_BETRAG_ANOMALIE"] = c
        self._log(f"[19/{NUM_TESTS}] KONTO_BETRAG_ANOMALIE: {c}")

    def _t19_text_kreditor_mismatch(self):
        """Same booking text used with ≥3 different creditors.

        Filters:
        - Texts shorter than TEXT_MISMATCH_MIN_LEN are ignored
        - Known stopwords (generic terms like 'Rechnung', 'Miete')
          are excluded"""
        df = self.df
        has_both = (df["buchungstext"].astype(str).str.strip() != "") & \
                   (df["kreditor"].astype(str).str.strip() != "")
        subset = df.loc[has_both, ["buchungstext", "kreditor"]].copy()
        if subset.empty:
            self.flag_counts["TEXT_KREDITOR_MISMATCH"] = 0
            self._log(f"[20/{NUM_TESTS}] TEXT_KREDITOR_MISMATCH: 0")
            return
        subset["_text_norm"] = subset["buchungstext"].astype(str).str.lower().str.strip()

        # Filter: min length + stopwords
        len_ok = subset["_text_norm"].str.len() >= TEXT_MISMATCH_MIN_LEN
        not_stop = ~subset["_text_norm"].isin(TEXT_STOPWORDS)
        subset = subset.loc[len_ok & not_stop]

        skipped = int((~len_ok).sum()) + int((~not_stop).sum())

        text_kreditors = subset.groupby("_text_norm")["kreditor"].nunique()
        suspicious_texts = set(text_kreditors[text_kreditors >= 3].index)

        c = 0
        if suspicious_texts:
            for i, row in df.loc[has_both].iterrows():
                txt = str(row["buchungstext"]).lower().strip()
                if txt in suspicious_texts:
                    self._flag(i, "TEXT_KREDITOR_MISMATCH")
                    c += 1
        self.flag_counts["TEXT_KREDITOR_MISMATCH"] = c
        self._log(f"[20/{NUM_TESTS}] TEXT_KREDITOR_MISMATCH: {c}  "
                  f"(Stopwords/kurze Texte übersprungen: {skipped})")

    def _t20_fuzzy_kreditor(self):
        """Fuzzy matching on creditor names with RapidFuzz.

        STAMMDATEN-LEVEL: Results are reported as a separate table
        (name_a, name_b, similarity, match_type) — NOT as per-booking
        flags, to avoid inflating the suspicious-bookings count.

        Strategy:
        1. Normalise (lowercase, Umlaute→ASCII, strip Rechtsformen)
        2. Group exact normalisation duplicates
        3. Jaro-Winkler (≥0.85) + Token Sort Ratio (≥85) on remaining"""
        df = self.df
        unique_kreds = df.loc[df["kreditor"].astype(str).str.strip() != "", "kreditor"].unique()
        matches_out: list[dict] = []

        if len(unique_kreds) < 2:
            self.stammdaten_report["fuzzy_kreditor_matches"] = []
            self._log(f"[21/{NUM_TESTS}] FUZZY_KREDITOR (Stammdaten): 0  (zu wenig Kreditoren)")
            return

        # Build normalised lookup
        normed_map = {}
        for k in unique_kreds:
            normed_map[k] = _norm_vendor(k)

        # Group by normalised form
        norm_to_originals: dict[str, list] = {}
        for orig, norm in normed_map.items():
            norm_to_originals.setdefault(norm, []).append(orig)

        # Exact normalisation matches → Stammdaten report
        for norm, origs in norm_to_originals.items():
            if len(origs) >= 2:
                for a_idx in range(len(origs)):
                    for b_idx in range(a_idx + 1, len(origs)):
                        matches_out.append({
                            "name_a": origs[a_idx],
                            "name_b": origs[b_idx],
                            "similarity": 100.0,
                            "match_type": "Norm-Duplikat",
                        })
                self._log(f"    Norm-Duplikat: {origs}")

        # Fuzzy on unique normalised names
        unique_normed = list(norm_to_originals.keys())
        if len(unique_normed) >= 2 and len(unique_normed) <= 10000:
            seen: list[str] = []
            for norm in unique_normed:
                if seen:
                    # Jaro-Winkler similarity (0-1 scale, cutoff 0.85)
                    jw_match = rf_process.extractOne(
                        norm, seen,
                        scorer=JaroWinkler.similarity,
                        score_cutoff=0.85,
                    )
                    # Token Sort Ratio (handles word order)
                    ts_match = rf_process.extractOne(
                        norm, seen,
                        scorer=fuzz.token_sort_ratio,
                        score_cutoff=85,
                    )

                    found: dict[str, tuple[float, str]] = {}
                    if jw_match:
                        found[jw_match[0]] = (jw_match[1] * 100, "Jaro-Winkler")
                    if ts_match:
                        matched_norm_ts = ts_match[0]
                        if matched_norm_ts in found:
                            found[matched_norm_ts] = (
                                max(found[matched_norm_ts][0], ts_match[1]),
                                "Jaro-Winkler+TokenSort",
                            )
                        else:
                            found[matched_norm_ts] = (ts_match[1], "TokenSort")

                    for matched_norm, (sim, mtype) in found.items():
                        if matched_norm != norm:
                            group_a = norm_to_originals.get(norm, [])
                            group_b = norm_to_originals.get(matched_norm, [])
                            for na in group_a:
                                for nb in group_b:
                                    matches_out.append({
                                        "name_a": na,
                                        "name_b": nb,
                                        "similarity": round(sim, 1),
                                        "match_type": mtype,
                                    })
                            self._log(f"    Fuzzy ({mtype}, {sim:.1f}%): {group_a} ↔ {group_b}")

                seen.append(norm)

        self.stammdaten_report["fuzzy_kreditor_matches"] = matches_out
        self._log(f"[21/{NUM_TESTS}] FUZZY_KREDITOR (Stammdaten): {len(matches_out)} Paare")

    def _t21_leerer_buchungstext(self):
        """Empty or generic booking text (ISA 240 compliance)."""
        df = self.df
        generic_patterns = {
            "diverse", "verschiedenes", "sonstiges", "test",
            "korrektur", "umbuchung", "xxx", "---", "...", "k.a.",
            "keine angabe", "n/a", "na", "tbd", "todo",
        }
        c = 0
        for i, row in df.iterrows():
            txt = str(row["buchungstext"]).strip()
            if not txt:
                self._flag(i, "LEERER_BUCHUNGSTEXT")
                c += 1
            elif txt.lower() in generic_patterns or len(txt) <= 2:
                self._flag(i, "LEERER_BUCHUNGSTEXT")
                c += 1
        self.flag_counts["LEERER_BUCHUNGSTEXT"] = c
        self._log(f"[22/{NUM_TESTS}] LEERER_BUCHUNGSTEXT: {c}")

    def _t22_velocity_anomalie(self):
        """Velocity check: sudden spike in bookings per creditor per month.

        Flags all bookings in months where a creditor has >3× their average."""
        df = self.df
        has_kred_date = (df["kreditor"].astype(str).str.strip() != "") & df["_datum"].notna()
        subset = df.loc[has_kred_date].copy()
        if subset.empty or len(subset) < 10:
            self.flag_counts["VELOCITY_ANOMALIE"] = 0
            self._log(f"[23/{NUM_TESTS}] VELOCITY_ANOMALIE: 0  (zu wenig Daten)")
            return

        subset["_yearmonth"] = subset["_datum"].apply(
            lambda d: f"{d.year}-{d.month:02d}" if d else None
        )

        kred_month = subset.groupby(["kreditor", "_yearmonth"]).size().reset_index(name="count")
        kred_avg = kred_month.groupby("kreditor")["count"].agg(["mean", "std", "count"]).rename(
            columns={"count": "n_months"}
        )
        kred_avg = kred_avg[kred_avg["n_months"] >= 3]

        flagged_indices = set()
        for kred, stats in kred_avg.iterrows():
            avg = stats["mean"]
            std = stats["std"] if stats["std"] > 0 else avg
            threshold = max(avg * 3, avg + 2 * std)
            kred_data = kred_month[kred_month["kreditor"] == kred]
            spike_months = kred_data.loc[kred_data["count"] >= threshold, "_yearmonth"].tolist()
            if spike_months:
                self._log(f"    Velocity {kred}: Spikes in {spike_months} "
                          f"(Ø={avg:.1f}, Schwelle={threshold:.1f})")
                for ym in spike_months:
                    mask = has_kred_date & (df["kreditor"] == kred)
                    for idx in df.index[mask]:
                        d = df.at[idx, "_datum"]
                        if d and f"{d.year}-{d.month:02d}" == ym:
                            self._flag(idx, "VELOCITY_ANOMALIE")
                            flagged_indices.add(idx)

        c = len(flagged_indices)
        self.flag_counts["VELOCITY_ANOMALIE"] = c
        self._log(f"[23/{NUM_TESTS}] VELOCITY_ANOMALIE: {c}")

    # ── run all ──────────────────────────────────────────────
    def run(self) -> dict:
        self._stats()
        self._t01_zscore()
        self._t02_iqr()
        self._t03_seltene_kontierung()
        self._t04_zeitlich()
        self._t05_ausserhalb_geschaeftszeit()
        self._t06_near_duplicate()
        self._t07_benford()
        self._t08_runder_betrag()
        self._t09_erfasser()
        self._t10_split()
        self._t11_schwellenwert_cluster()
        self._t12_beleg_luecke()
        self._t13_doppelte_belegnummer()
        self._t14_beleg_kreditor_duplikat()
        self._t15_storno()
        self._t16_neuer_kreditor_hoch()
        self._t17_soll_gleich_haben()
        self._t18_konto_betrag_anomalie()
        self._t19_text_kreditor_mismatch()
        self._t20_fuzzy_kreditor()
        self._t21_leerer_buchungstext()
        self._t22_velocity_anomalie()
        return self._export()

    def _export(self) -> dict:
        df = self.df
        mask = df["_score"] >= OUTPUT_THRESHOLD
        for i, row in df.iterrows():
            if any(f in CRITICAL_FLAGS for f in row["_flags"]):
                mask.at[i] = True

        verdaechtig = df[mask].sort_values("_score", ascending=False).head(MAX_OUTPUT_ROWS)

        out_cols = ["datum", "konto_soll", "konto_haben", "betrag",
                    "buchungstext", "belegnummer", "kostenstelle",
                    "kreditor", "erfasser"]
        rows = []
        for i, row in verdaechtig.iterrows():
            r = {c: str(row.get(c, "")) for c in out_cols}
            r["betrag"] = row["_betrag"]
            r["anomaly_score"] = round(row["_score"], 2)
            r["anomaly_flags"] = "|".join(row["_flags"])
            if row["_datum"]:
                r["datum"] = row["_datum"].strftime("%Y-%m-%d")
            rows.append(r)

        total = len(df)
        n_verd = int(mask.sum())
        avg_score = round(float(df["_score"].mean()), 2) if total > 0 else 0.0
        total_flags = int(df["_flags"].apply(len).sum()) if total > 0 else 0
        pct = n_verd / total * 100 if total > 0 else 0.0

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
            "stammdaten_report": self.stammdaten_report,
            "logs": self.logs,
        }
