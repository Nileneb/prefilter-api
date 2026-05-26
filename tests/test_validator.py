"""Tests für Spalten-Validierung, speziell RECHNUNGSDATUM_PERIODE-Block (#15)."""

import pandas as pd

from src.validator import validate_columns


def _base_df():
    return pd.DataFrame({
        "datum": ["2024-01-15", "2024-02-10"],
        "betrag": ["100,00", "200,00"],
        "buchungstext": ["Miete", "Strom"],
        "belegnummer": ["0001", "0002"],
        "kreditor": ["A", "B"],
        "konto_soll": ["4000", "5000"],
    })


def test_rechnungsdatum_blocked_without_compare_source():
    # WHY(#15): Diamant ohne erfassungsdatum/buchungsperiode → blockiert statt leer
    df = _base_df()
    v = validate_columns(df)
    assert "RECHNUNGSDATUM_PERIODE" in v.tests_blocked
    assert "RECHNUNGSDATUM_PERIODE" not in v.tests_ok


def test_rechnungsdatum_ok_with_erfassungsdatum():
    df = _base_df()
    df["erfassungsdatum"] = ["2024-03-01", "2024-03-02"]
    v = validate_columns(df)
    assert "RECHNUNGSDATUM_PERIODE" not in v.tests_blocked
    assert "RECHNUNGSDATUM_PERIODE" in v.tests_ok


def test_rechnungsdatum_ok_with_buchungsperiode():
    df = _base_df()
    df["buchungsperiode"] = ["2024-03", "2024-03"]
    v = validate_columns(df)
    assert "RECHNUNGSDATUM_PERIODE" not in v.tests_blocked
