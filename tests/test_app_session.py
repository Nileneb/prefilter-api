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
    n = len(app_local.ALL_TEST_NAMES)
    # webhook, zscore, iqr, near_dup, output, prefix, text_konto_thr,
    # konto_filter_all=True (alle Konten → Test-df nutzt Bestandskonto), min, max
    defaults = ("", 2.5, 1.5, 3, 2.0, "", 0.3, True, 0, 99999999)
    controls = [True] * n + [2.0] * n  # Enables + Gewicht-Slider

    # Zwei unabhängige Sessions
    s1 = app_local._new_session()
    s2 = app_local._new_session()

    last1 = None
    for out in app_local.analyze_file(s1, csv, *defaults, *controls):
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
    n = len(app_local.ALL_TEST_NAMES)
    defaults = ("", 2.5, 1.5, 3, 2.0, "", 0.3, True, 0, 99999999)
    controls = [True] * n + [2.0] * n
    s = app_local._new_session()
    for _ in app_local.analyze_file(s, csv, *defaults, *controls):
        pass

    fig, _warn, sess = app_local._build_dynamic_chart(s, "Scatter", "betrag", "_score", None, "(keine)", "(keine)")
    assert sess is s
    assert s["dynamic_fig"] is not None


def test_isolation_checkbox_is_the_only_switch(app_local, tmp_path):
    """Ankreuzen von ISOLATION_ANOMALIE startet den Test — kein versteckter
    zweiter isolation_enabled-Gate, der die Checkbox aushebelt."""
    from src.tests.isolation_anomaly import HAS_SKLEARN
    if not HAS_SKLEARN:
        pytest.skip("scikit-learn nicht verfügbar")

    n = 1100  # ≥ isolation_min_bookings (1000)
    betrag = ["100"] * (n - 20) + ["50000"] * 20  # klare Ausreißer
    df = pd.DataFrame({
        "datum": ["2024-01-15"] * n,
        "betrag": betrag,
        "konto_soll": ["42000"] * n,                 # GuV → im Default-Scope
        "buchungstext": ["Buchung"] * n,
        "belegnummer": [str(i) for i in range(n)],
        "kreditor": ["X"] * n,
    })
    csv = tmp_path / "big.csv"
    df.to_csv(csv, index=False, sep=";", encoding="utf-8")

    order = app_local._UI_TEST_ORDER
    n_tests = len(order)
    # Default-Filter (GuV), nur ISOLATION ankreuzen (single switch)
    defaults = ("", 2.5, 1.5, 3, 2.0, "", 0.3, False, 40000, 80000)
    enables = [name == "ISOLATION_ANOMALIE" for name in order]
    controls = enables + [2.0] * n_tests

    s = app_local._new_session()
    for _ in app_local.analyze_file(s, str(csv), *defaults, *controls):
        pass
    fc = s["result"]["statistics"]["flag_counts"]
    # Checkbox an → Test lief und flaggte (gated wäre 0)
    assert fc["ISOLATION_ANOMALIE"] > 0
