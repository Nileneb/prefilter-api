"""Tests für POST /api/suggest (Sachkonto-Vorschlag aus dem Kontierungs-Index)."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from src import main
from src.kontierung import build_index


def _build_tmp_index(tmp_path):
    df = pd.DataFrame(
        [{"konto_soll": "60000", "kreditor": "Lieferant A", "buchungstext": "Reinigung"}] * 5
        + [{"konto_soll": "60001", "kreditor": "Lieferant A", "buchungstext": "x"}] * 1
    )
    build_index(df, embedder=None, gt={}).save(tmp_path / "idx")
    return tmp_path / "idx"


def test_suggest_known_kreditor(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "KONTO_INDEX_PATH", str(_build_tmp_index(tmp_path)))
    monkeypatch.setattr(main, "_konto_index", None)
    r = TestClient(main.app).post("/api/suggest", json={"kreditor": "Lieferant A"})
    assert r.status_code == 200
    sugs = r.json()["suggestions"]
    assert sugs[0]["konto"] == "60000"  # 5× > 1×
    assert "Lieferant A" in sugs[0]["reason"]


def test_suggest_unknown_kreditor_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "KONTO_INDEX_PATH", str(_build_tmp_index(tmp_path)))
    monkeypatch.setattr(main, "_konto_index", None)
    r = TestClient(main.app).post("/api/suggest", json={"kreditor": "Unbekannt"})
    assert r.status_code == 200
    assert r.json()["suggestions"] == []


def test_suggest_no_index_503(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "KONTO_INDEX_PATH", str(tmp_path / "fehlt"))
    monkeypatch.setattr(main, "_konto_index", None)
    r = TestClient(main.app).post("/api/suggest", json={"kreditor": "X"})
    assert r.status_code == 503
    assert "Index" in r.json()["warning"]
