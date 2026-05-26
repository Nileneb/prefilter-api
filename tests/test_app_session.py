"""End-to-End-Test des gr.State-Session-Refactors (#10, #17).

Verifiziert im lokalen Modus, dass analyze_file den Session-Dict befüllt und
dass ein nachfolgender Chart-Aufruf denselben Session-Dict liest (kein
Modul-Global mehr → kein Datenleck zwischen Sessions).
"""

import pandas as pd
import pytest


@pytest.fixture
def app_local(monkeypatch):
    import app as app_module
    # Lokalen Modus erzwingen (kein Redis/Celery nötig)
    monkeypatch.setattr(app_module, "_LOCAL_MODE", True)
    monkeypatch.setattr(app_module, "_r", None)
    monkeypatch.setattr(app_module, "_celery", None)
    return app_module


def _make_csv(tmp_path) -> str:
    df = pd.DataFrame({
        "datum": ["2024-01-15"] * 12,
        "betrag": [f"{i * 100},00" for i in range(1, 13)],
        "konto_soll": ["4711"] * 12,
        "buchungstext": [f"Buchung {i}" for i in range(12)],
        "belegnummer": [f"{i:04d}" for i in range(12)],
        "kreditor": ["Lieferant A"] * 12,
    })
    path = tmp_path / "buchungen.csv"
    df.to_csv(path, index=False, sep=";", encoding="utf-8")
    return str(path)


def test_session_isolated_across_two_runs(app_local, tmp_path):
    csv = _make_csv(tmp_path)
    toggles = [True] * len(app_local.ALL_TEST_NAMES)
    defaults = ("", 2.5, 1.5, 3, 2.0, "", 0.3, 40000, 80000)

    # Zwei unabhängige Sessions
    s1 = app_local._new_session()
    s2 = app_local._new_session()

    last1 = None
    for out in app_local.analyze_file(s1, csv, *defaults, *toggles):
        last1 = out
    # letztes yield: (summary, logs, table, csv_update, session)
    assert last1[-1] is s1
    assert s1["result"] is not None
    assert s1["engine_df"] is not None
    assert s2["result"] is None  # zweite Session unberührt

    # Chart-Handler liest dieselbe Session
    fig = app_local.generate_score_distribution(s1)
    assert fig is not None

    # Leere Session → kein Chart, aber kein Crash
    empty = app_local.generate_score_distribution(s2)
    assert empty is not None  # gr.update(value=None)


def test_dynamic_chart_persists_fig_in_session(app_local, tmp_path):
    csv = _make_csv(tmp_path)
    toggles = [True] * len(app_local.ALL_TEST_NAMES)
    defaults = ("", 2.5, 1.5, 3, 2.0, "", 0.3, 40000, 80000)
    s = app_local._new_session()
    for _ in app_local.analyze_file(s, csv, *defaults, *toggles):
        pass

    fig, _warn, sess = app_local._build_dynamic_chart(s, "Scatter", "betrag", "_score", None, "(keine)", "(keine)")
    assert sess is s
    assert s["dynamic_fig"] is not None
