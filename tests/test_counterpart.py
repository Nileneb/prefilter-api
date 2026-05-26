"""Tests für die vektorisierte find_counterpart_rows (#9)."""

import pandas as pd

from src.engine import find_counterpart_rows


def _df(rows):
    return pd.DataFrame(rows)


def test_matching_pair_fills_both_sides():
    df = _df([
        {"_beleg_id": "B1", "_abs": 100.0, "konto_soll": "4000", "konto_haben": ""},
        {"_beleg_id": "B1", "_abs": 100.0, "konto_soll": "1200", "konto_haben": ""},
    ])
    result, inferred = find_counterpart_rows(df)
    assert result.iloc[0] == "1200" and inferred.iloc[0]
    assert result.iloc[1] == "4000" and inferred.iloc[1]


def test_amount_mismatch_no_fill():
    df = _df([
        {"_beleg_id": "B1", "_abs": 100.0, "konto_soll": "4000", "konto_haben": ""},
        {"_beleg_id": "B1", "_abs": 200.0, "konto_soll": "1200", "konto_haben": ""},
    ])
    result, inferred = find_counterpart_rows(df)
    assert not inferred.any()


def test_existing_konto_haben_preserved():
    df = _df([
        {"_beleg_id": "B1", "_abs": 100.0, "konto_soll": "4000", "konto_haben": "9999"},
        {"_beleg_id": "B1", "_abs": 100.0, "konto_soll": "1200", "konto_haben": ""},
    ])
    result, inferred = find_counterpart_rows(df)
    assert result.iloc[0] == "9999" and not inferred.iloc[0]  # nicht überschrieben
    assert result.iloc[1] == "4000" and inferred.iloc[1]


def test_three_row_beleg_ignored():
    df = _df([
        {"_beleg_id": "B1", "_abs": 100.0, "konto_soll": "4000", "konto_haben": ""},
        {"_beleg_id": "B1", "_abs": 50.0, "konto_soll": "1200", "konto_haben": ""},
        {"_beleg_id": "B1", "_abs": 50.0, "konto_soll": "1300", "konto_haben": ""},
    ])
    result, inferred = find_counterpart_rows(df)
    assert not inferred.any()


def test_zero_amount_no_fill():
    df = _df([
        {"_beleg_id": "B1", "_abs": 0.0, "konto_soll": "4000", "konto_haben": ""},
        {"_beleg_id": "B1", "_abs": 0.0, "konto_soll": "1200", "konto_haben": ""},
    ])
    result, inferred = find_counterpart_rows(df)
    assert not inferred.any()
