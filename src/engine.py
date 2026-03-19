"""
Buchungs-Anomalie Pre-Filter — Anomaly Engine v4.2

Orchestriert alle 13 Test-Module aus src/tests/*.
Vollständig vektorisiert, kein iterrows(), kein O(n²).

Public API:
  - AnomalyEngine(df, config, cancel_check)
  - engine.run() → dict
  - engine.run_single_test(test_name, stats) → flagged indices
  - engine.compute_stats() → EngineStats
  - engine.apply_flags_and_export(flag_results) → dict
  - _ALL_TESTS, _TEST_BY_NAME, NUM_TESTS, WEIGHTS, CRITICAL_FLAGS
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from src.accounting import kontoklasse, compute_signed_betrag
from src.config import AnalysisConfig
from src.logging_config import get_logger
from src.parser import COLUMN_ALIASES, parse_german_number_series, parse_date_series
from src.tests.base import EngineStats
from src.tests.betrag import get_tests as get_betrag_tests
from src.tests.duplikate import get_tests as get_duplikate_tests
from src.tests.buchungslogik import get_tests as get_buchungslogik_tests
from src.tests.kreditor import get_tests as get_kreditor_tests
from src.tests.zeitreihe import get_tests as get_zeitreihe_tests

logger = get_logger("prefilter.engine")

# ── Test-Registry ────────────────────────────────────────────────────────────
# Reihenfolge ist bewusst: Betrags-Tests zuerst (für Stats), dann der Rest
_ALL_TESTS = (
    get_betrag_tests()
    + get_duplikate_tests()
    + get_buchungslogik_tests()
    + get_kreditor_tests()
    + get_zeitreihe_tests()
)

# Abgeleitete Lookup-Strukturen
WEIGHTS: dict[str, float]    = {t.name: t.weight for t in _ALL_TESTS}
CRITICAL_FLAGS: set[str]     = {t.name for t in _ALL_TESTS if t.critical}
_FLAG_NAMES: list[str]       = [t.name for t in _ALL_TESTS]
NUM_TESTS: int               = len(_ALL_TESTS)
MAX_POSSIBLE_SCORE: float    = sum(WEIGHTS.values())  # Summe aller Test-Gewichte
OUTPUT_THRESHOLD: float      = 2.0   # Default — überschreibbar per config
MAX_OUTPUT_ROWS: int         = 1000  # Default


# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        config: AnalysisConfig | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        self.df           = df
        self.config       = config or AnalysisConfig()
        self.cancel_check = cancel_check
        self.logs:         list[str]       = []
        self.flag_counts:  dict[str, int]  = {}
        self.stammdaten_report: dict[str, list] = {"fuzzy_kreditor_matches": []}
        self._prepare()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self.logs.append(msg)
        logger.info(msg)

    def _is_cancelled(self) -> bool:
        if self.cancel_check is not None:
            return self.cancel_check()
        return False

    def _prepare(self) -> None:
        df = self.df
        for col in COLUMN_ALIASES:
            if col not in df.columns:
                df[col] = ""
        df.fillna("", inplace=True)

        df["_betrag"] = parse_german_number_series(df["betrag"]).fillna(0.0).astype("float32")
        df["_abs"]    = df["_betrag"].abs()
        df["_datum"]  = parse_date_series(df["datum"])
        df["_kontoklasse"] = kontoklasse(df["konto_soll"])
        df["_betrag_signed"] = compute_signed_betrag(df).astype("float32")
        df["_score"]  = 0.0

        # Kategorische Spalten — beschleunigt GroupBy und spart RAM
        for col in ("konto_soll", "konto_haben", "kreditor", "belegnummer", "buchungstext"):
            if col in df.columns:
                df[col] = df[col].astype("category")

        # Boolean-Spalten — eine pro Flag (kein Listen-Anti-Pattern)
        for name in _FLAG_NAMES:
            df[f"flag_{name}"] = False

        visible_cols = [
            c for c in df.columns
            if not c.startswith("_") and not c.startswith("flag_")
        ]
        self._log(f"Geladen: {len(df)} Buchungen")
        self._log(f"Spalten: {', '.join(visible_cols)}")

    def _compute_stats(self) -> EngineStats:
        vals = self.df.loc[self.df["_abs"] > 0, "_abs"]
        if len(vals) == 0:
            self._log("Beträge: keine Nicht-Null-Werte")
            return EngineStats()
        q1 = float(vals.quantile(0.25))
        q3 = float(vals.quantile(0.75))
        iqr = q3 - q1
        stats = EngineStats(
            b_mean  = float(vals.mean()),
            b_std   = float(vals.std()) if len(vals) > 1 else 0.0,
            b_iqr   = iqr,
            b_fence = q3 + self.config.iqr_factor * iqr,
            n_vals  = len(vals),
        )
        self._log(
            f"Beträge: n={stats.n_vals}, μ={stats.b_mean:.0f}, σ={stats.b_std:.0f}, "
            f"Q1={q1:.0f}, Q3={q3:.0f}, IQR={stats.b_iqr:.0f}, Fence={stats.b_fence:.0f}"
        )
        return stats

    def compute_stats(self) -> EngineStats:
        """Public: berechnet und cached EngineStats."""
        self._stats_cache = self._compute_stats()
        return self._stats_cache

    def compute_stats_dict(self) -> dict:
        """Berechnet Stats und gibt sie als serialisierbares dict zurück."""
        stats = self.compute_stats()
        return stats.to_dict()

    def run_single_test(self, test_name: str, stats: EngineStats) -> list[int]:
        """Führt einen einzelnen Test aus, gibt geflaggte Indizes zurück."""
        test = _TEST_BY_NAME[test_name]
        # Flag-Spalte initialisieren (nötig im Parallel-Pfad ohne _prepare())
        flag_col = f"flag_{test_name}"
        if flag_col not in self.df.columns:
            self.df[flag_col] = False
        count = test.run_with_logging(self.df, stats, self.config) if hasattr(test, "run_with_logging") else test.run(self.df, stats, self.config)
        self.flag_counts[test_name] = count
        # NaN-Safety: _flag() setzt nur True, Rest kann NaN sein
        flags = self.df[flag_col].fillna(False)
        flagged = self.df.index[flags].tolist()
        return flagged

    def apply_flags_and_export(self, flag_results: list[dict]) -> dict:
        """Setzt Flags aus externen Ergebnissen, berechnet Scores, exportiert.

        flag_results: Liste von {test_name: str, flagged: list[int], count: int}
        """
        for r in flag_results:
            name = r["test_name"]
            flagged = r["flagged"]
            self.flag_counts[name] = r["count"]
            if flagged:
                self.df.loc[flagged, f"flag_{name}"] = True
        self._compute_scores()
        return self._export()

    def _compute_scores(self) -> None:
        score = pd.Series(0.0, index=self.df.index)
        for name, weight in WEIGHTS.items():
            score += self.df[f"flag_{name}"].astype(float) * weight
        self.df["_score"] = score

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, enabled_tests: set[str] | None = None) -> dict:
        stats = self._compute_stats()

        for i, test in enumerate(_ALL_TESTS, start=1):
            if self._is_cancelled():
                self._log("Analyse abgebrochen (Stop-Signal)")
                break
            if enabled_tests is not None and test.name not in enabled_tests:
                self._log(f"[{i:02d}/{NUM_TESTS}] {test.name}: übersprungen (deaktiviert)")
                self.flag_counts[test.name] = 0
                continue
            count = test.run_with_logging(self.df, stats, self.config) if hasattr(test, "run_with_logging") else test.run(self.df, stats, self.config)
            self.flag_counts[test.name] = count
            self._log(f"[{i:02d}/{NUM_TESTS}] {test.name}: {count}")

        self._compute_scores()
        return self._export()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self) -> dict:
        df = self.df
        threshold = self.config.output_threshold

        crit_cols     = [f"flag_{n}" for n in CRITICAL_FLAGS if f"flag_{n}" in df.columns]
        critical_mask = (
            df[crit_cols].any(axis=1) if crit_cols
            else pd.Series(False, index=df.index)
        )
        output_mask = (df["_score"] >= threshold) | critical_mask
        verdaechtig = (
            df[output_mask]
            .sort_values("_score", ascending=False)
            .head(self.config.max_output_rows)
            .copy()
        )

        # Flag-Strings vektorisiert via boolean matrix dot-product
        flag_cols = [f"flag_{n}" for n in _FLAG_NAMES]
        flag_matrix = verdaechtig[flag_cols].values
        flag_names = _FLAG_NAMES
        verdaechtig["anomaly_flags"] = [
            "|".join(n for n, v in zip(flag_names, row) if v)
            for row in flag_matrix
        ]

        out_cols = [
            "datum", "konto_soll", "konto_haben", "betrag",
            "buchungstext", "belegnummer", "kostenstelle", "kreditor", "soll_haben",
        ]

        # Vektorisiert: datum formatieren wo vorhanden
        has_datum = verdaechtig["_datum"].notna()
        verdaechtig.loc[has_datum, "datum"] = (
            verdaechtig.loc[has_datum, "_datum"].dt.strftime("%Y-%m-%d")
        )

        # Betrag durch geparsten Wert ersetzen, Score runden
        verdaechtig["betrag"] = verdaechtig["_betrag"]
        verdaechtig["anomaly_score"] = verdaechtig["_score"].round(2)

        # Alle Spalten als String (außer betrag + anomaly_score)
        for c in out_cols:
            if c != "betrag":
                verdaechtig[c] = verdaechtig[c].astype(str)

        # Dict-Output ohne iterrows
        rows = verdaechtig[out_cols + ["anomaly_score", "anomaly_flags"]].to_dict("records")

        total     = len(df)
        n_verd    = int(output_mask.sum())
        n_output  = len(rows)
        avg_score = round(float(df["_score"].mean()), 2) if total > 0 else 0.0
        pct       = n_verd / total * 100 if total > 0 else 0.0

        top_flags = sorted(
            ((k, v) for k, v in self.flag_counts.items() if v > 0),
            key=lambda x: -x[1],
        )
        max_score = MAX_POSSIBLE_SCORE
        summary_lines = [
            f"ERGEBNIS: {n_verd} von {total} verdächtig ({pct:.1f}%)",
            f"Ausgegeben: {n_output} (Top {n_output} nach Score)" if n_output < n_verd else f"Ausgegeben: {n_output}",
            f"Flags gesamt: {sum(self.flag_counts.values())}, Ø Score: {avg_score}/{max_score}",
            f"Top-Flags: {', '.join(f'{k}:{v}' for k, v in top_flags) or 'keine'}",
        ]
        if rows:
            summary_lines.append(
                f"Höchster Score: {rows[0]['anomaly_score']}/{max_score} (Beleg: {rows[0]['belegnummer']})"
            )
        for line in summary_lines:
            self._log(line)

        return {
            "message": f"{n_verd} verdächtige Buchungen ({pct:.1f}%)",
            "statistics": {
                "total_input":      total,
                "total_suspicious": n_verd,
                "total_output":     n_output,
                "filter_ratio":     f"{pct:.1f}%",
                "avg_score":        avg_score,
                "max_possible_score": MAX_POSSIBLE_SCORE,
                "flag_counts":      self.flag_counts,
            },
            "verdaechtige_buchungen": rows,
            "stammdaten_report":      self.stammdaten_report,
            "logs":                   self.logs,
        }

    # ── Backward-compatibility shims (for tests/test_engine.py) ──────────────
    # Tests call engine._stats() + engine._t*() directly. These delegates keep
    # them working without any test changes.

    _TEST_BY_NAME: dict = {}   # populated lazily below class

    def _stats(self) -> None:
        """Compat: compute and cache stats."""
        self._stats_cache = self._compute_stats()

    def _run_named_test(self, name: str) -> None:
        """Run a single test by flag name, using cached stats."""
        test  = _TEST_BY_NAME[name]
        stats = getattr(self, "_stats_cache", EngineStats())
        count = test.run_with_logging(self.df, stats, self.config) if hasattr(test, "run_with_logging") else test.run(self.df, stats, self.config)
        self.flag_counts[name] = count

    def _t01_zscore(self)                -> None: self._run_named_test("BETRAG_ZSCORE")
    def _t02_iqr(self)                   -> None: self._run_named_test("BETRAG_IQR")
    def _t06_near_duplicate(self)        -> None: self._run_named_test("NEAR_DUPLICATE")
    def _t13_doppelte_belegnummer(self)  -> None: self._run_named_test("DOPPELTE_BELEGNUMMER")
    def _t14_beleg_kreditor_duplikat(self) -> None: self._run_named_test("BELEG_KREDITOR_DUPLIKAT")
    def _t15_storno(self)                -> None: self._run_named_test("STORNO")
    def _t16_neuer_kreditor_hoch(self)   -> None: self._run_named_test("NEUER_KREDITOR_HOCH")
    def _t18_konto_betrag_anomalie(self) -> None: self._run_named_test("KONTO_BETRAG_ANOMALIE")
    def _t21_leerer_buchungstext(self)   -> None: self._run_named_test("LEERER_BUCHUNGSTEXT")
    def _t23_rechnungsdatum_periode(self)   -> None: self._run_named_test("RECHNUNGSDATUM_PERIODE")
    def _t24_buchungstext_periode(self)     -> None: self._run_named_test("BUCHUNGSTEXT_PERIODE")
    def _t25_monats_entwicklung(self)       -> None: self._run_named_test("MONATS_ENTWICKLUNG")
    def _t26_fehlende_monatsbuchung(self)   -> None: self._run_named_test("FEHLENDE_MONATSBUCHUNG")


# Populate the class-level lookup after _ALL_TESTS is defined
AnomalyEngine._TEST_BY_NAME = _TEST_BY_NAME = {t.name: t for t in _ALL_TESTS}
