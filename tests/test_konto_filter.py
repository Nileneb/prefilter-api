"""Tests für den globalen Konto-Bereichsfilter (GuV), der für ALLE Tests gilt."""

import pandas as pd

from src.config import AnalysisConfig
from src.engine import AnomalyEngine


def _df(konto: str, n: int = 5) -> pd.DataFrame:
    # Leerer Buchungstext → LEERER_BUCHUNGSTEXT flaggt jede In-Scope-Zeile
    return pd.DataFrame({
        "datum": ["2024-01-15"] * n,
        "betrag": ["100"] * n,
        "konto_soll": [konto] * n,
        "buchungstext": [""] * n,
        "belegnummer": [str(i) for i in range(n)],
        "kreditor": ["X"] * n,
    })


def _leer_count(df, **cfg) -> int:
    result = AnomalyEngine(df, config=AnalysisConfig(**cfg)).run()
    return result["statistics"]["flag_counts"]["LEERER_BUCHUNGSTEXT"]


def test_default_guv_excludes_bestandskonto():
    # WHY: 1200 < 40000 = Bestandskonto → out of scope bei Default-GuV-Filter
    assert _leer_count(_df("1200")) == 0


def test_default_guv_includes_ertragskonto():
    assert _leer_count(_df("42000")) == 5  # 40000–59999 = Ertrag


def test_filter_disabled_includes_all_konten():
    assert _leer_count(_df("1200"), konto_filter_enabled=False) == 5


def test_widened_range_includes_bestandskonto():
    assert _leer_count(_df("1200"), konto_filter_min=0, konto_filter_max=99999) == 5


def test_filter_applies_to_amount_tests_too():
    # Betrag-Test (BETRAG_ZSCORE) flaggt Bestandskonto-Ausreißer nur ohne Filter
    n = 30
    betrag = ["100"] * (n - 1) + ["50000"]  # ein klarer Ausreißer
    df = pd.DataFrame({
        "datum": ["2024-01-15"] * n,
        "betrag": betrag,
        "konto_soll": ["1200"] * n,        # Bestandskonto
        "buchungstext": [f"Text {i}" for i in range(n)],
        "belegnummer": [str(i) for i in range(n)],
        "kreditor": ["X"] * n,
    })
    with_filter = AnomalyEngine(df.copy(), config=AnalysisConfig()).run()
    without = AnomalyEngine(df.copy(), config=AnalysisConfig(konto_filter_enabled=False)).run()
    assert with_filter["statistics"]["flag_counts"]["BETRAG_ZSCORE"] == 0
    assert without["statistics"]["flag_counts"]["BETRAG_ZSCORE"] >= 1
