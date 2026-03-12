"""
Buchungs-Anomalie Pre-Filter — Test-Base mit Logging v2

Ersetzt: src/tests/base.py

Änderungen:
─────────────────────────────────────────────────
1. AnomalyTest bekommt einen structlog-Logger
2. Jeder Test loggt automatisch:
   - Welche required_columns vorhanden/leer sind
   - Zwischenergebnisse (Gruppengrößen, Ausschlüsse, etc.)
   - Laufzeit
   - Finale Flag-Anzahl
3. Neues `self.log` Interface für Sub-Tests
4. Neues `self.stats_detail` dict für maschinenlesbare Metriken

Kompatibel mit allen bestehenden Tests — kein Test muss umgeschrieben
werden, bekommt aber automatisch Start/End/Column-Logging.
Tests können OPTIONAL self.log() nutzen für Zwischenschritte.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from src.logging_config import get_logger


_logger = get_logger("prefilter.tests")


@dataclass
class EngineStats:
    """Globale Statistiken über alle Beträge — von der Engine berechnet."""
    b_mean:  float = 0.0
    b_std:   float = 0.0
    b_iqr:   float = 0.0
    b_fence: float = 0.0
    n_vals:  int   = 0

    def to_dict(self) -> dict:
        return {
            "b_mean": self.b_mean,
            "b_std": self.b_std,
            "b_iqr": self.b_iqr,
            "b_fence": self.b_fence,
            "n_vals": self.n_vals,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EngineStats:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class AnomalyTest:
    """Basisklasse für alle Anomalie-Tests.

    Jeder Test MUSS definieren:
        name:             str           — Flag-Name (z.B. "NEAR_DUPLICATE")
        weight:           float         — Score-Gewicht
        critical:         bool          — Kritisches Flag?
        required_columns: list[str]     — Benötigte DataFrame-Spalten
        run(df, stats, config) -> int   — Hauptlogik, gibt Anzahl Flags zurück

    Jeder Test KANN nutzen:
        self.log(msg, **kw)             — Strukturiertes Zwischen-Logging
        self.metric(key, value)         — Maschinenlesbare Metrik speichern
        self._flag(df, mask) -> int     — Setzt Flag-Spalte aus Boolean-Maske
    """

    name: str = ""
    weight: float = 1.0
    critical: bool = False
    required_columns: list[str] = []

    def __init__(self):
        self._detail: dict[str, object] = {}
        self._start_time: float = 0.0

    # ── Logging-Interface für Sub-Tests ──────────────────────────────────────

    def log(self, msg: str, **kwargs) -> None:
        """Strukturiertes Log mit Test-Kontext."""
        _logger.info(msg, test=self.name, **kwargs)

    def metric(self, key: str, value: object) -> None:
        """Speichert eine maschinenlesbare Metrik für diesen Test-Lauf."""
        self._detail[key] = value

    # ── Column-Prüfung ──────────────────────────────────────────────────────

    def _check_columns(self, df: pd.DataFrame) -> dict[str, dict]:
        """Prüft required_columns und loggt Füllstände."""
        report = {}
        for col in self.required_columns:
            if col not in df.columns:
                report[col] = {"status": "MISSING", "filled": 0, "total": len(df)}
                self.log(f"Spalte fehlt: {col}", column=col, status="MISSING")
            else:
                if col.startswith("_"):
                    # Numerische Hilfsspalte
                    if df[col].dtype in ("float32", "float64", "int64"):
                        filled = int((df[col] != 0).sum())
                    else:
                        filled = int(df[col].notna().sum())
                else:
                    # String-Spalte
                    vals = df[col].astype(str).str.strip()
                    filled = int((vals != "").sum())
                pct = round(filled / len(df) * 100, 1) if len(df) > 0 else 0
                report[col] = {
                    "status": "OK" if filled > 0 else "EMPTY",
                    "filled": filled,
                    "total": len(df),
                    "pct": pct,
                }
                if filled == 0:
                    self.log(
                        f"Spalte leer: {col} (0/{len(df)})",
                        column=col, status="EMPTY",
                    )
                elif pct < 50:
                    self.log(
                        f"Spalte dünn besetzt: {col} ({filled}/{len(df)} = {pct}%)",
                        column=col, status="SPARSE", filled=filled, pct=pct,
                    )
        return report

    # ── Flag-Helper ──────────────────────────────────────────────────────────

    def _flag(self, df: pd.DataFrame, mask: pd.Series) -> int:
        """Setzt Flag-Spalte aus Boolean-Maske, gibt Anzahl zurück."""
        count = int(mask.sum())
        df.loc[mask, f"flag_{self.name}"] = True
        return count

    # ── Wrapper für run() mit automatischem Logging ──────────────────────────

    def run_with_logging(self, df: pd.DataFrame, stats: EngineStats, config) -> int:
        """Wrapper um run() — loggt Start, Spalten, Laufzeit, Ergebnis."""
        self._detail = {}
        self._start_time = time.perf_counter()

        self.log(
            "START",
            rows=len(df),
            weight=self.weight,
            critical=self.critical,
        )

        # Spalten prüfen
        col_report = self._check_columns(df)
        self._detail["columns"] = col_report

        # Eigentlichen Test ausführen
        count = self.run(df, stats, config)

        elapsed = time.perf_counter() - self._start_time
        self._detail["count"] = count
        self._detail["elapsed_s"] = round(elapsed, 3)

        pct = round(count / len(df) * 100, 2) if len(df) > 0 else 0

        self.log(
            "DONE",
            count=count,
            pct=f"{pct}%",
            elapsed_s=round(elapsed, 3),
        )

        return count

    # ── Abstrakte Methode ────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, stats: EngineStats, config) -> int:
        raise NotImplementedError
