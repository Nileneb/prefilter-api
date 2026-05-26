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


# ── Diamant-Modell: konto_haben/soll_haben-Abwesenheit ist KEIN Defekt ──────

def test_missing_konto_haben_is_info_not_warning():
    # WHY: Diamant hat kein Gegenkonto je Zeile → neutrale Info, keine ⚠️-Warnung
    df = _base_df()  # kein konto_haben
    v = validate_columns(df)
    assert any("Gegenkonto je Zeile" in i for i in v.infos)
    assert not any("konto_haben" in w for w in v.warnings)


def test_missing_soll_haben_with_signed_betrag_is_info():
    df = _base_df()
    df["betrag"] = ["-100,00", "200,00"]  # signiert → Vorzeichen aus Betrag
    v = validate_columns(df)
    assert any("signierten Betrag" in i for i in v.infos)
    assert not any("soll_haben fehlt" in w for w in v.warnings)


def test_missing_soll_haben_unsigned_betrag_warns():
    df = _base_df()  # betrag "100,00"/"200,00" → unsigniert
    v = validate_columns(df)
    assert any("Soll/Haben-Richtung unbekannt" in w for w in v.warnings)


def test_near_duplicate_not_degraded_by_absent_model_optional():
    # WHY: soll_haben-Abwesenheit (Modell) darf NEAR_DUPLICATE nicht "eingeschränkt" machen
    df = _base_df()
    df["betrag"] = ["-100,00", "200,00"]  # signiert
    v = validate_columns(df)
    reason = v.tests_degraded.get("NEAR_DUPLICATE", "")
    assert "soll_haben" not in reason
