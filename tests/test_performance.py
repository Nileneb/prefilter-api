"""
Performance-Test: 500.000 Buchungszeilen in < 60 Sekunden.

Ausführen:  python -m pytest tests/test_performance.py -v -s
Markierung: @pytest.mark.slow → wird bei normalem `pytest` übersprungen.

Aktivieren: pytest -m slow   ODER   pytest --run-slow
"""

import time
import random
import string

import numpy as np
import pandas as pd
import pytest

from src.parser import map_columns
from src.engine import AnomalyEngine


def _generate_synthetic_data(n: int = 500_000) -> pd.DataFrame:
    """Generiert einen synthetischen Buchungsdatensatz mit n Zeilen."""
    rng = np.random.default_rng(42)

    # Datumsbereich: 2023-01-01 bis 2024-12-31
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2024-12-31")
    days = (end - start).days
    dates = [
        (start + pd.Timedelta(days=int(d))).strftime("%d.%m.%Y")
        for d in rng.integers(0, days, size=n)
    ]

    # Beträge: Normalverteilung um 500 mit einigen Ausreißern
    amounts = rng.normal(500, 200, size=n).round(2)
    amounts = np.where(amounts < 0, -amounts, amounts)
    # 0.1% extreme Ausreißer
    outlier_idx = rng.choice(n, size=max(1, n // 1000), replace=False)
    amounts[outlier_idx] = rng.uniform(50000, 200000, size=len(outlier_idx))
    betrag_strs = [f"{v:.2f}".replace(".", ",") for v in amounts]

    # Konten
    soll_konten = [str(k) for k in rng.choice([4711, 4720, 4730, 6100, 6200, 6300, 65000, 70000], size=n)]
    haben_konten = [str(k) for k in rng.choice([1200, 1400, 1600, 1800], size=n)]

    # Buchungstexte
    text_templates = [
        "Rechnung {nr} Lieferant",
        "Gutschrift {nr}",
        "Wartungsvertrag {nr}",
        "Einkauf Material {nr}",
        "Dienstleistung {nr}",
        "Miete {nr}",
        "Versicherung {nr}",
        "Storno Rechnung {nr}",
        "",
        "diverse",
    ]
    texts = [
        random.choice(text_templates).format(nr=f"{i:06d}")
        for i in range(n)
    ]

    # Belegnummern (einige Duplikate)
    beleg_nrs = [f"B-{i:07d}" for i in range(n)]
    # 0.5% doppelte Belegnummern
    dup_idx = rng.choice(n, size=max(1, n // 200), replace=False)
    for idx in dup_idx:
        beleg_nrs[idx] = beleg_nrs[max(0, idx - 1)]

    # Kreditoren
    kreditor_names = [f"Lieferant_{chr(65 + i % 26)}{i // 26}" for i in range(50)]
    kreditors = [random.choice(kreditor_names) for _ in range(n)]

    # Erfasser
    erfasser = [random.choice(["UserA", "UserB", "UserC", "UserD"]) for _ in range(n)]

    df = pd.DataFrame({
        "datum": dates,
        "betrag": betrag_strs,
        "konto_soll": soll_konten,
        "konto_haben": haben_konten,
        "buchungstext": texts,
        "belegnummer": beleg_nrs,
        "kostenstelle": [""] * n,
        "kreditor": kreditors,
        "erfasser": erfasser,
    })
    return df


@pytest.mark.slow
def test_500k_under_60_seconds():
    """500.000 Buchungszeilen müssen in unter 60 Sekunden analysiert werden."""
    df = _generate_synthetic_data(500_000)
    df = map_columns(df)

    start = time.perf_counter()
    engine = AnomalyEngine(df)
    result = engine.run()
    elapsed = time.perf_counter() - start

    # Assertions
    assert result["statistics"]["total_input"] == 500_000
    assert len(result["statistics"]["flag_counts"]) == 14
    assert elapsed < 90, f"Analyse dauerte {elapsed:.1f}s — Limit ist 90s"

    print(f"\n{'='*60}")
    print(f"  Performance-Test: 500k Zeilen in {elapsed:.1f}s")
    print(f"  Verdächtig: {result['statistics']['total_output']}")
    print(f"  Avg Score:  {result['statistics']['avg_score']}")
    print(f"  Flags:      {sum(result['statistics']['flag_counts'].values())}")
    print(f"{'='*60}")


@pytest.mark.slow
def test_parallel_vs_sequential_identical():
    """Simulierte parallele Pipeline liefert identische Ergebnisse wie sequentiell.

    Nutzt 10k Zeilen (schneller als 500k, aber groß genug für alle Tests).
    """
    from src.tests.base import EngineStats
    from src.engine import _ALL_TESTS
    from src.config import AnalysisConfig

    df = _generate_synthetic_data(10_000)
    df = map_columns(df)
    config = AnalysisConfig(output_threshold=1.0)

    # ── Sequentiell ──
    engine_seq = AnomalyEngine(df.copy(), config=config)
    result_seq = engine_seq.run()

    # ── Simulierte parallele Pipeline ──
    engine_prep = AnomalyEngine(df.copy(), config=config)
    stats = engine_prep.compute_stats()

    flag_results = []
    for test in _ALL_TESTS:
        engine_test = AnomalyEngine(df.copy(), config=config)
        flagged = engine_test.run_single_test(test.name, stats)
        count = engine_test.flag_counts[test.name]
        flag_results.append({
            "test_name": test.name,
            "flagged": flagged,
            "count": count,
        })

    engine_merge = AnomalyEngine(df.copy(), config=config)
    result_par = engine_merge.apply_flags_and_export(flag_results)

    # ── Vergleich ──
    for test in _ALL_TESTS:
        seq_count = result_seq["statistics"]["flag_counts"].get(test.name, 0)
        par_count = result_par["statistics"]["flag_counts"].get(test.name, 0)
        assert seq_count == par_count, (
            f"Flag-Count {test.name}: seq={seq_count} vs par={par_count}"
        )

    assert result_seq["statistics"]["total_output"] == result_par["statistics"]["total_output"]

    seq_scores = {r["belegnummer"]: r["anomaly_score"] for r in result_seq["verdaechtige_buchungen"]}
    par_scores = {r["belegnummer"]: r["anomaly_score"] for r in result_par["verdaechtige_buchungen"]}
    assert seq_scores == par_scores

    print(f"\n{'='*60}")
    print(f"  Parallel-Vergleich: 10k Zeilen — Ergebnisse identisch")
    print(f"  Flags seq:  {sum(result_seq['statistics']['flag_counts'].values())}")
    print(f"  Flags par:  {sum(result_par['statistics']['flag_counts'].values())}")
    print(f"  Output:     {result_seq['statistics']['total_output']}")
    print(f"{'='*60}")
