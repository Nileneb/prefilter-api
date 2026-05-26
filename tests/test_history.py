"""Tests für die Run-History, speziell load_last_run beim ersten Lauf (#14)."""

import json

import pandas as pd

import src.history as history


def _write_run(base, mandant, ts, flag_counts):
    d = base / mandant
    d.mkdir(parents=True, exist_ok=True)
    (d / f"run_{ts}.json").write_text(
        json.dumps({"run_date": ts, "flag_counts": flag_counts, "monthly_stats": {}}),
        encoding="utf-8",
    )


def test_no_history_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path)
    assert history.load_last_run("m1") is None


def test_single_run_returns_none_no_self_compare(tmp_path, monkeypatch):
    # WHY(#14): ein einziger (gerade gespeicherter) Lauf → kein Vergleich
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path)
    _write_run(tmp_path, "m1", "2026-01-01T00-00-00", {"STORNO": 1})
    assert history.load_last_run("m1") is None


def test_two_runs_returns_previous(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path)
    _write_run(tmp_path, "m1", "2026-01-01T00-00-00", {"STORNO": 1})  # previous
    _write_run(tmp_path, "m1", "2026-02-01T00-00-00", {"STORNO": 5})  # current
    prev = history.load_last_run("m1")
    assert prev is not None and prev["flag_counts"]["STORNO"] == 1


def test_three_runs_returns_second_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path)
    _write_run(tmp_path, "m1", "2026-01-01T00-00-00", {"STORNO": 1})
    _write_run(tmp_path, "m1", "2026-02-01T00-00-00", {"STORNO": 3})  # previous
    _write_run(tmp_path, "m1", "2026-03-01T00-00-00", {"STORNO": 9})  # current
    prev = history.load_last_run("m1")
    assert prev is not None and prev["flag_counts"]["STORNO"] == 3


def test_monthly_stats_vectorized_matches_iterrows(tmp_path):
    # Sichert die Vektorisierung in C12 ab (Ergebnis-Form bleibt gleich)
    df = pd.DataFrame({
        "konto_soll": ["4000", "4000", "5000"],
        "_datum": pd.to_datetime(["2024-01-15", "2024-01-20", "2024-02-01"]),
        "_abs": [100.0, 50.0, 200.0],
    })
    stats = history._monthly_stats(df)
    assert stats["4000"]["2024-01"] == 150.0
    assert stats["5000"]["2024-02"] == 200.0
