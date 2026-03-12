"""
Buchungs-Anomalie Pre-Filter — Basis-Protokoll für Anomalie-Tests

Alle Test-Module in src/tests/ implementieren AnomalyTest.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import pandas as pd

from src.config import AnalysisConfig


@dataclass
class EngineStats:
    """Globale Betrags-Statistiken, einmal pro Engine-Lauf berechnet."""
    b_mean: float = 0.0
    b_std: float = 0.0
    b_iqr: float = 0.0
    b_fence: float = 0.0
    n_vals: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EngineStats":
        return cls(**d)


class AnomalyTest:
    """Basisklasse für Anomalie-Tests.

    Jede Unterklasse:
      - definiert `name` (Flag-Name, z.B. "BETRAG_ZSCORE")
      - definiert `weight` (Gewicht im Scoring)
      - definiert `critical` (immer ausgeben wenn True)
      - implementiert `run(df, stats, config)` → setzt df.loc[mask, f"flag_{self.name}"] = True,
        gibt Anzahl der Treffer zurück.
    """
    name: str = ""
    weight: float = 1.0
    critical: bool = False

    def run(
        self,
        df: pd.DataFrame,
        stats: EngineStats,
        config: AnalysisConfig,
    ) -> int:
        """Führt den Test aus. Setzt flag-Spalte in df in-place. Gibt Trefferzahl zurück."""
        raise NotImplementedError(f"{self.__class__.__name__}.run() nicht implementiert")

    def _flag(self, df: pd.DataFrame, mask: pd.Series) -> int:
        """Hilfsmethode: Flag-Spalte setzen + Trefferzahl zurückgeben."""
        df.loc[mask, f"flag_{self.name}"] = True
        return int(mask.sum())
