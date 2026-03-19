"""
Tests für das Feedback-System (FeedbackStore, FeedbackStats, ScoreReweighter).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import pandas as pd

from src.feedback import FeedbackStore, FeedbackLabel, MIN_LABELS_FOR_TRAINING, MIN_LABELS_FOR_STATS
from src.feedback_stats import compute_feedback_stats, format_feedback_report, FlagStats
from src.trainer import ScoreReweighter, TrainingLocked
from src.engine import AnomalyEngine, WEIGHTS
from src.config import AnalysisConfig


# ══════════════════════════════════════════════════════════════
# FeedbackStore
# ══════════════════════════════════════════════════════════════

class TestFeedbackStore:

    def _make_store(self, tmp_path: Path) -> FeedbackStore:
        return FeedbackStore(str(tmp_path / "feedback"))

    def _make_label(self, mandant: str = "150", label: str = "tp", flags: str = "BETRAG_ZSCORE") -> FeedbackLabel:
        return FeedbackLabel(
            mandant_id=mandant,
            analysis_timestamp="2026-03-19T10:00:00",
            row_index=0,
            belegnummer="R001",
            anomaly_score=3.5,
            anomaly_flags=flags,
            label=label,
            pruefer="MM",
        )

    def test_save_and_load(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        lbl = self._make_label()
        store.save(lbl)
        loaded = store.load_all("150")
        assert len(loaded) == 1
        assert loaded[0].label == "tp"
        assert loaded[0].pruefer == "MM"

    def test_count(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        for i in range(5):
            store.save(self._make_label())
        assert store.count("150") == 5
        assert store.count("999") == 0

    def test_total_count_across_mandanten(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        store.save(self._make_label("150"))
        store.save(self._make_label("110"))
        assert store.total_count() == 2

    def test_can_show_stats_threshold(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        assert not store.can_show_stats("150")
        for _ in range(MIN_LABELS_FOR_STATS):
            store.save(self._make_label())
        assert store.can_show_stats("150")

    def test_can_train_threshold(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        assert not store.can_train("150")

    def test_path_traversal_sanitized(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        lbl = self._make_label("../../etc/passwd")
        store.save(lbl)
        # Should NOT create a file outside the feedback dir
        path = store._path("../../etc/passwd")
        assert "etc" not in str(path.parent)

    def test_empty_mandant_fallback(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        path = store._path("")
        assert path.name == "unknown_feedback.jsonl"


# ══════════════════════════════════════════════════════════════
# FeedbackStats
# ══════════════════════════════════════════════════════════════

class TestFeedbackStats:

    def test_compute_stats_basic(self) -> None:
        labels = [
            FeedbackLabel("150", "t", 0, "R1", 3.0, "BETRAG_ZSCORE|STORNO", "tp", "MM"),
            FeedbackLabel("150", "t", 1, "R2", 2.0, "BETRAG_ZSCORE", "fp", "MM"),
            FeedbackLabel("150", "t", 2, "R3", 4.0, "STORNO", "unsure", "MM"),
        ]
        stats = compute_feedback_stats(labels)
        assert "BETRAG_ZSCORE" in stats
        assert stats["BETRAG_ZSCORE"].tp == 1
        assert stats["BETRAG_ZSCORE"].fp == 1
        assert stats["BETRAG_ZSCORE"].precision == 0.5
        assert stats["STORNO"].tp == 1
        assert stats["STORNO"].unsure == 1

    def test_empty_labels(self) -> None:
        stats = compute_feedback_stats([])
        assert stats == {}

    def test_format_report_insufficient_labels(self, tmp_path: Path) -> None:
        store = FeedbackStore(str(tmp_path / "fb"))
        report = format_feedback_report(store, "150")
        assert "Zu wenig Labels" in report

    def test_flag_stats_properties(self) -> None:
        fs = FlagStats(flag="TEST", tp=8, fp=2, unsure=1)
        assert fs.total == 11
        assert fs.precision == pytest.approx(0.8)
        assert fs.fp_rate == pytest.approx(0.2)

    def test_flag_stats_zero_division(self) -> None:
        fs = FlagStats(flag="TEST")
        assert fs.precision == 0.0
        assert fs.fp_rate == 0.0


# ══════════════════════════════════════════════════════════════
# ScoreReweighter
# ══════════════════════════════════════════════════════════════

class TestScoreReweighter:

    def _fill_store(self, store: FeedbackStore, n: int, mandant: str = "150") -> None:
        """Füllt den Store mit n Labels (abwechselnd tp/fp)."""
        for i in range(n):
            lbl = FeedbackLabel(
                mandant_id=mandant,
                analysis_timestamp="2026-03-19T10:00:00",
                row_index=i,
                belegnummer=f"R{i:04d}",
                anomaly_score=3.0,
                anomaly_flags="BETRAG_ZSCORE|NEAR_DUPLICATE",
                label="tp" if i % 2 == 0 else "fp",
                pruefer="MM",
            )
            store.save(lbl)

    def test_training_locked_insufficient_labels(self, tmp_path: Path) -> None:
        store = FeedbackStore(str(tmp_path / "fb"))
        self._fill_store(store, 10)
        reweighter = ScoreReweighter(store)
        with pytest.raises(TrainingLocked, match="Training gesperrt"):
            reweighter.train("150")

    def test_training_succeeds_with_enough_labels(self, tmp_path: Path) -> None:
        store = FeedbackStore(str(tmp_path / "fb"))
        self._fill_store(store, MIN_LABELS_FOR_TRAINING)
        reweighter = ScoreReweighter(store)
        weights = reweighter.train("150")
        assert isinstance(weights, dict)
        # Jedes Gewicht muss im erlaubten Bereich liegen
        for w in weights.values():
            assert 0.1 <= w <= 5.0

    def test_weight_range_clamped(self, tmp_path: Path) -> None:
        store = FeedbackStore(str(tmp_path / "fb"))
        # Alle tp → tp_rate=1.0 → weight * 1.5
        for i in range(MIN_LABELS_FOR_TRAINING):
            lbl = FeedbackLabel(
                mandant_id="150",
                analysis_timestamp="t",
                row_index=i,
                belegnummer=f"R{i}",
                anomaly_score=3.0,
                anomaly_flags="BETRAG_ZSCORE",
                label="tp",
                pruefer="MM",
            )
            store.save(lbl)
        weights = ScoreReweighter(store).train("150")
        if "BETRAG_ZSCORE" in weights:
            assert weights["BETRAG_ZSCORE"] <= 5.0

    def test_all_fp_halves_weight(self, tmp_path: Path) -> None:
        store = FeedbackStore(str(tmp_path / "fb"))
        for i in range(MIN_LABELS_FOR_TRAINING):
            lbl = FeedbackLabel(
                mandant_id="150",
                analysis_timestamp="t",
                row_index=i,
                belegnummer=f"R{i}",
                anomaly_score=3.0,
                anomaly_flags="BETRAG_ZSCORE",
                label="fp",
                pruefer="MM",
            )
            store.save(lbl)
        weights = ScoreReweighter(store).train("150")
        if "BETRAG_ZSCORE" in weights:
            default_w = WEIGHTS["BETRAG_ZSCORE"]
            # 100% FP → weight * 0.5
            assert weights["BETRAG_ZSCORE"] == pytest.approx(default_w * 0.5, abs=0.01)


# ══════════════════════════════════════════════════════════════
# custom_weights in Engine._compute_scores
# ══════════════════════════════════════════════════════════════

class TestCustomWeights:

    def test_custom_weights_applied(self) -> None:
        """custom_weights soll Default-Gewichte überschreiben."""
        df = pd.DataFrame({
            "datum":       ["01.01.2024"] * 3,
            "betrag":      ["100,00", "200,00", "300,00"],
            "konto_soll":  ["70000", "70000", "70000"],
            "buchungstext": ["Test"] * 3,
            "belegnummer":  ["B1", "B2", "B3"],
        })
        # Ohne custom_weights
        engine1 = AnomalyEngine(df.copy(), config=AnalysisConfig())
        engine1.run()
        scores1 = engine1.df["_score"].copy()

        # Mit custom_weights (alles auf 0.1 → Scores müssen kleiner sein)
        custom = {name: 0.1 for name in WEIGHTS}
        engine2 = AnomalyEngine(df.copy(), config=AnalysisConfig(custom_weights=custom))
        engine2.run()
        scores2 = engine2.df["_score"].copy()

        # Wenn es überhaupt Flags gibt, müssen die Scores kleiner sein
        if scores1.sum() > 0:
            assert scores2.sum() <= scores1.sum()

    def test_custom_weights_none_uses_defaults(self) -> None:
        """Ohne custom_weights → Default WEIGHTS."""
        df = pd.DataFrame({
            "datum":       ["01.01.2024"],
            "betrag":      ["100,00"],
            "konto_soll":  ["70000"],
            "buchungstext": ["Test"],
            "belegnummer":  ["B1"],
        })
        config = AnalysisConfig(custom_weights=None)
        engine = AnomalyEngine(df, config=config)
        engine.run()
        # Sollte normal durchlaufen
        assert "_score" in engine.df.columns


# ══════════════════════════════════════════════════════════════
# find_counterpart_rows
# ══════════════════════════════════════════════════════════════

class TestFindCounterpartRows:

    def test_basic_two_row_beleg(self) -> None:
        """Beleg mit genau 2 Zeilen → konto_haben wird aus Gegenseite gefüllt."""
        from src.engine import find_counterpart_rows
        df = pd.DataFrame({
            "_beleg_id":  ["B1", "B1"],
            "konto_soll": ["70000", "12000"],
            "konto_haben": ["", ""],
            "_abs":       [1000.0, 1000.0],
            "_betrag":    [1000.0, -1000.0],
        })
        filled, mask = find_counterpart_rows(df)
        assert filled.iloc[0] == "12000"
        assert filled.iloc[1] == "70000"
        assert mask.all()

    def test_no_fill_when_already_set(self) -> None:
        """Wenn konto_haben schon gesetzt → nicht überschreiben."""
        from src.engine import find_counterpart_rows
        df = pd.DataFrame({
            "_beleg_id":  ["B1", "B1"],
            "konto_soll": ["70000", "12000"],
            "konto_haben": ["99999", ""],
            "_abs":       [1000.0, 1000.0],
            "_betrag":    [1000.0, -1000.0],
        })
        filled, mask = find_counterpart_rows(df)
        assert filled.iloc[0] == "99999"  # Nicht überschrieben
        assert filled.iloc[1] == "70000"  # Gefüllt

    def test_skip_when_amounts_differ(self) -> None:
        """Beträge unterschiedlich → nicht füllen (kein Gegenbuchungs-Paar)."""
        from src.engine import find_counterpart_rows
        df = pd.DataFrame({
            "_beleg_id":  ["B1", "B1"],
            "konto_soll": ["70000", "12000"],
            "konto_haben": ["", ""],
            "_abs":       [1000.0, 500.0],
            "_betrag":    [1000.0, -500.0],
        })
        filled, mask = find_counterpart_rows(df)
        assert filled.iloc[0] == ""
        assert filled.iloc[1] == ""

    def test_skip_groups_with_more_than_2_rows(self) -> None:
        """Belege mit >2 Zeilen (Splitbuchung) → nicht füllen."""
        from src.engine import find_counterpart_rows
        df = pd.DataFrame({
            "_beleg_id":  ["B1", "B1", "B1"],
            "konto_soll": ["70000", "12000", "13000"],
            "konto_haben": ["", "", ""],
            "_abs":       [1000.0, 1000.0, 500.0],
            "_betrag":    [1000.0, -1000.0, 500.0],
        })
        filled, mask = find_counterpart_rows(df)
        assert not mask.any()
