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
    assert elapsed < 60, f"Analyse dauerte {elapsed:.1f}s — Limit ist 60s"

    print(f"\n{'='*60}")
    print(f"  Performance-Test: 500k Zeilen in {elapsed:.1f}s")
    print(f"  Verdächtig: {result['statistics']['total_output']}")
    print(f"  Avg Score:  {result['statistics']['avg_score']}")
    print(f"  Flags:      {sum(result['statistics']['flag_counts'].values())}")
    print(f"{'='*60}")
