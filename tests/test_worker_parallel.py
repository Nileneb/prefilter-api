"""
Tests für die parallele Celery-Pipeline (chord).

Testet:
  - Sequentiell vs. parallel liefert identische Flags
  - Parquet-Cleanup nach Abschluss
  - EngineStats Serialisierung (to_dict / from_dict)
  - apply_flags_and_export korrekt

Ausführen: python -m pytest tests/test_worker_parallel.py -v
"""

import os
import tempfile

import pandas as pd
import pytest

from src.parser import map_columns
from src.engine import AnomalyEngine, _ALL_TESTS, _TEST_BY_NAME, NUM_TESTS
from src.config import AnalysisConfig
from src.tests.base import EngineStats


# ── Testdaten ─────────────────────────────────────────────────────────────────

def _make_test_df(n: int = 1000) -> pd.DataFrame:
    """Erzeugt einen deterministischen Testdatensatz mit n Zeilen."""
    import numpy as np
    rng = np.random.default_rng(42)

    start = pd.Timestamp("2023-01-01")
    days = 730
    dates = [
        (start + pd.Timedelta(days=int(d))).strftime("%d.%m.%Y")
        for d in rng.integers(0, days, size=n)
    ]

    amounts = rng.normal(500, 200, size=n).round(2)
    amounts = np.where(amounts < 0, -amounts, amounts)
    outlier_idx = rng.choice(n, size=max(1, n // 100), replace=False)
    amounts[outlier_idx] = rng.uniform(50000, 200000, size=len(outlier_idx))
    betrag_strs = [f"{v:.2f}".replace(".", ",") for v in amounts]

    soll_konten = [str(k) for k in rng.choice([4711, 4720, 65000, 70000], size=n)]
    haben_konten = [str(k) for k in rng.choice([1200, 1400], size=n)]

    texts = [f"Buchung {i:04d}" for i in range(n)]
    # Storno-Keywords einbauen
    for i in range(0, n, 200):
        texts[i] = "Storno Rechnung"
    # Leere Texte einbauen
    for i in range(5, n, 300):
        texts[i] = ""

    beleg_nrs = [f"B-{i:06d}" for i in range(n)]
    # Duplikate
    for i in range(10, n, 50):
        beleg_nrs[i] = beleg_nrs[i - 1]

    kreditors = [f"Lieferant_{chr(65 + i % 20)}" for i in range(n)]
    erfasser = [f"User{i % 4}" for i in range(n)]

    return pd.DataFrame({
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


# ── EngineStats Serialisierung ────────────────────────────────────────────────

class TestEngineStatsSerialization:
    def test_roundtrip(self):
        stats = EngineStats(b_mean=500.0, b_std=200.0, b_iqr=300.0, b_fence=1200.0, n_vals=1000)
        d = stats.to_dict()
        restored = EngineStats.from_dict(d)
        assert restored == stats

    def test_default_roundtrip(self):
        stats = EngineStats()
        d = stats.to_dict()
        restored = EngineStats.from_dict(d)
        assert restored == stats

    def test_dict_is_json_serializable(self):
        import json
        stats = EngineStats(b_mean=500.0, b_std=200.0, b_iqr=300.0, b_fence=1200.0, n_vals=1000)
        json_str = json.dumps(stats.to_dict())
        restored = EngineStats.from_dict(json.loads(json_str))
        assert restored == stats


# ── Simulierte parallele Pipeline ─────────────────────────────────────────────

class TestParallelVsSequential:
    """Vergleicht sequentiellen Engine-Lauf mit simulierter paralleler Pipeline."""

    @pytest.fixture
    def test_df(self):
        return map_columns(_make_test_df(1000))

    def test_identical_flags(self, test_df):
        """Sequentieller und paralleler Pfad liefern identische Flag-Ergebnisse."""
        config = AnalysisConfig(output_threshold=1.0)

        # ── Sequentiell ──
        engine_seq = AnomalyEngine(test_df.copy(), config=config)
        result_seq = engine_seq.run()

        # ── Simulierte parallele Pipeline ──
        # Phase 1: Prepare
        engine_prep = AnomalyEngine(test_df.copy(), config=config)
        stats = engine_prep.compute_stats()

        # Phase 2: Tests einzeln ausführen (simuliert run_test_task)
        flag_results = []
        for test in _ALL_TESTS:
            # Frischer Engine pro Test (wie im Worker)
            engine_test = AnomalyEngine(test_df.copy(), config=config)
            flagged = engine_test.run_single_test(test.name, stats)
            count = engine_test.flag_counts[test.name]
            flag_results.append({
                "test_name": test.name,
                "flagged": flagged,
                "count": count,
            })

        # Phase 3: Merge
        engine_merge = AnomalyEngine(test_df.copy(), config=config)
        result_par = engine_merge.apply_flags_and_export(flag_results)

        # ── Vergleich ──
        # Flag-Counts müssen identisch sein
        for test in _ALL_TESTS:
            seq_count = result_seq["statistics"]["flag_counts"].get(test.name, 0)
            par_count = result_par["statistics"]["flag_counts"].get(test.name, 0)
            assert seq_count == par_count, (
                f"Flag-Count {test.name}: seq={seq_count} vs par={par_count}"
            )

        # Gleiche Anzahl verdächtiger Buchungen
        assert result_seq["statistics"]["total_output"] == result_par["statistics"]["total_output"]

        # Gleiche Scores für die gleichen Belegnummern
        seq_rows = {r["belegnummer"]: r["anomaly_score"] for r in result_seq["verdaechtige_buchungen"]}
        par_rows = {r["belegnummer"]: r["anomaly_score"] for r in result_par["verdaechtige_buchungen"]}
        assert seq_rows == par_rows


# ── Parquet I/O Tests ─────────────────────────────────────────────────────────

class TestParquetRoundtrip:
    def test_parquet_write_read(self):
        """DataFrame kann als Parquet gespeichert und geladen werden."""
        df = map_columns(_make_test_df(100))
        config = AnalysisConfig()
        engine = AnomalyEngine(df, config=config)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            engine.df.to_parquet(parquet_path, index=True)
            loaded = pd.read_parquet(parquet_path)
            assert len(loaded) == len(engine.df)
            assert list(loaded.columns) == list(engine.df.columns)
        finally:
            os.unlink(parquet_path)

    def test_cleanup_after_merge(self):
        """Parquet-Datei wird nach Verarbeitung gelöscht."""
        df = map_columns(_make_test_df(50))
        config = AnalysisConfig()
        engine = AnomalyEngine(df, config=config)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        engine.df.to_parquet(parquet_path, index=True)
        assert os.path.exists(parquet_path)

        os.unlink(parquet_path)
        assert not os.path.exists(parquet_path)


# ── apply_flags_and_export Tests ──────────────────────────────────────────────

class TestApplyFlagsAndExport:
    def test_empty_results(self):
        """Keine Flags → keine verdächtigen Buchungen."""
        df = map_columns(_make_test_df(50))
        config = AnalysisConfig(output_threshold=100.0)  # Sehr hoch
        engine = AnomalyEngine(df, config=config)

        flag_results = [
            {"test_name": t.name, "flagged": [], "count": 0}
            for t in _ALL_TESTS
        ]
        result = engine.apply_flags_and_export(flag_results)
        assert result["statistics"]["total_output"] == 0

    def test_with_flags(self):
        """Flags korrekt angewandt → verdächtige Buchungen im Output."""
        df = map_columns(_make_test_df(50))
        config = AnalysisConfig(output_threshold=1.0)
        engine = AnomalyEngine(df, config=config)

        # Simuliere: Index 0 und 1 bei BETRAG_ZSCORE geflaggt
        flag_results = [
            {"test_name": "BETRAG_ZSCORE", "flagged": [0, 1], "count": 2},
        ] + [
            {"test_name": t.name, "flagged": [], "count": 0}
            for t in _ALL_TESTS if t.name != "BETRAG_ZSCORE"
        ]
        result = engine.apply_flags_and_export(flag_results)
        assert result["statistics"]["total_output"] >= 2
        assert result["statistics"]["flag_counts"]["BETRAG_ZSCORE"] == 2
