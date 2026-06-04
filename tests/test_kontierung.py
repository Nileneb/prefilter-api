"""Tests für src/kontierung.py — IST-Report + Präzedenz-Suggester (ohne Embeddings)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.kontierung import (
    KontoIndex,
    KontoSuggester,
    _norm_kreditor,
    build_index,
    ist_report,
)


@pytest.fixture
def df() -> pd.DataFrame:
    # Lieferant A immer auf 4711, Lieferant B gespalten (60000/60001) → uneinheitlich
    rows = (
        [{"konto_soll": "4711", "kreditor": "Lieferant A", "buchungstext": "Büromaterial", "bezeichnung": "Wareneingang"}] * 5
        + [{"konto_soll": "60000", "kreditor": "Lieferant B", "buchungstext": "Reinigung", "bezeichnung": "Fremdleistung"}] * 3
        + [{"konto_soll": "60001", "kreditor": "Lieferant B", "buchungstext": "Reinigung Büro", "bezeichnung": "Fremdleistung"}] * 2
    )
    return pd.DataFrame(rows)


def test_norm_kreditor():
    assert _norm_kreditor("  Lieferant   A ") == "lieferant a"
    assert _norm_kreditor("LIEFERANT A") == "lieferant a"


def test_build_index_kreditor_konto(df):
    idx = build_index(df, embedder=None, gt={})
    assert idx.kreditor_konto["lieferant a"] == {"4711": 5}
    assert idx.kreditor_konto["lieferant b"] == {"60000": 3, "60001": 2}
    assert idx.emb is None


def test_suggest_known_kreditor_dominant_first(df):
    sug = KontoSuggester(build_index(df, embedder=None, gt={})).suggest("Lieferant B")
    assert sug[0].konto == "60000"  # 3× > 2×
    assert sug[1].konto == "60001"
    assert "Lieferant B" in sug[0].reason


def test_suggest_unknown_kreditor_no_embedder_empty(df):
    assert KontoSuggester(build_index(df, embedder=None, gt={})).suggest("Unbekannt GmbH", "x") == []


def test_suggest_uses_gt_bezeichnung(df):
    idx = build_index(df, embedder=None, gt={"4711": "Büromaterial GT"})
    sug = KontoSuggester(idx).suggest("Lieferant A")
    assert sug[0].bezeichnung == "Büromaterial GT"


def test_ist_report_flags_inconsistent_kreditor(df):
    rep = ist_report(build_index(df, embedder=None, gt={}))
    assert rep["n_inkonsistente_kreditoren"] == 1
    # uneinheitlicher Kreditor steht oben
    assert rep["kreditoren"][0]["kreditor"] == "lieferant b"
    assert rep["kreditoren"][0]["konsistent"] is False
    assert rep["kreditoren"][0]["n_konten"] == 2


def test_ist_report_name_drift(df):
    # GT-Name weicht von DIAMANT-Bezeichnung "Wareneingang" ab → Drift
    rep = ist_report(build_index(df, embedder=None, gt={"4711": "Sonstiges"}))
    drift_konten = {r["konto"]: r["namens_drift"] for r in rep["konten"]}
    assert drift_konten["4711"] is True
    assert drift_konten["60000"] is False  # kein GT-Eintrag → kein Drift


def test_save_load_roundtrip(tmp_path, df):
    idx = build_index(df, embedder=None, gt={"4711": "X"})
    idx.save(tmp_path / "idx")
    loaded = KontoIndex.load(tmp_path / "idx")
    assert loaded.kreditor_konto == idx.kreditor_konto
    assert loaded.gt == idx.gt
    assert loaded.emb is None
    assert KontoSuggester(loaded).suggest("Lieferant A")[0].konto == "4711"


def test_build_index_requires_konto():
    with pytest.raises(ValueError, match="konto_soll"):
        build_index(pd.DataFrame([{"kreditor": "A", "buchungstext": "x"}]), embedder=None, gt={})


def test_klasse_filter_only_sachkonto_lines():
    # Ein Beleg: K-Zeile (Personenkonto 700177) + S-Zeile (Sachkonto 60046).
    # Kreditor steht in beiden Zeilen; nur das Sachkonto darf in den Index.
    df = pd.DataFrame(
        [
            {"klasse": "K", "konto_soll": "700177", "kreditor": "GEMA", "buchungstext": "GEMA"},
            {"klasse": "S", "konto_soll": "60046", "kreditor": "GEMA", "buchungstext": "GEMA-Gebühren"},
        ]
    )
    idx = build_index(df, embedder=None, gt={})
    assert idx.kreditor_konto["gema"] == {"60046": 1}  # 700177 (Personenkonto) NICHT enthalten


def test_no_klasse_column_counts_all_rows():
    # Ohne klasse-Spalte (einfache Test-CSV) Fallback auf alle Zeilen.
    df = pd.DataFrame([{"konto_soll": "60046", "kreditor": "GEMA", "buchungstext": "x"}])
    assert build_index(df, embedder=None, gt={}).kreditor_konto["gema"] == {"60046": 1}
