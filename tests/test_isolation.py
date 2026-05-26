"""Tests für den ISOLATION_ANOMALIE Mindest-Buchungs-Guard (#12)."""

import pandas as pd
import pytest

from src.config import AnalysisConfig
from src.engine import AnomalyEngine
from src.tests.isolation_anomaly import HAS_SKLEARN


def _df(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "datum": ["2024-01-15"] * n,
        "betrag": [str(100 + i) for i in range(n)],
        "konto_soll": ["4711"] * n,
        "buchungstext": [f"Buchung {i}" for i in range(n)],
        "belegnummer": [f"{i:05d}" for i in range(n)],
        "kreditor": ["Lieferant A"] * n,
    })


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn nicht verfügbar")
def test_isolation_zero_below_min_bookings():
    # WHY(#12): aktiviert, aber unter Mindestmenge → 0 + Warnung (kein FP-Rauschen)
    config = AnalysisConfig(isolation_enabled=True, isolation_min_bookings=5000)
    engine = AnomalyEngine(_df(100), config=config)
    result = engine.run()
    assert result["statistics"]["flag_counts"]["ISOLATION_ANOMALIE"] == 0


def test_isolation_disabled_by_default():
    engine = AnomalyEngine(_df(100))  # isolation_enabled=False default
    result = engine.run()
    assert result["statistics"]["flag_counts"]["ISOLATION_ANOMALIE"] == 0
