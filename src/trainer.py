"""
Buchungs-Anomalie Pre-Filter — Score-Reweighter

Berechnet optimierte Flag-Gewichte aus Prüfer-Feedback.

⛔ Training ist HARD-LOCKED bis mindestens MIN_LABELS_FOR_TRAINING Labels.
   Kein Bypass möglich — die Methode wirft TrainingLocked.
"""

from __future__ import annotations

from src.feedback import FeedbackStore, FeedbackLabel, MIN_LABELS_FOR_TRAINING
from src.engine import WEIGHTS


class TrainingLocked(Exception):
    """Wird geworfen wenn Training versucht wird ohne genug Labels."""


class ScoreReweighter:
    """Berechnet neue Flag-Gewichte aus TP/FP-Feedback.

    Algorithmus:
      1. Für jedes Flag: tp_rate = tp / (tp + fp), wobei unsure ignoriert wird.
      2. Neues Gewicht = default_weight × (0.5 + tp_rate).
         → Bei 100% TP: Gewicht × 1.5 (hochgestuft)
         → Bei 50/50:   Gewicht × 1.0 (unverändert)
         → Bei 100% FP: Gewicht × 0.5 (halbiert)
      3. Gewichte werden auf [0.1, 5.0] geklammert.
    """

    MIN_FLAG_SAMPLES = 10  # Mindest-Labels pro Flag um Gewicht anzupassen

    def __init__(self, store: FeedbackStore) -> None:
        self.store = store

    def train(self, mandant_id: str | None = None) -> dict[str, float]:
        """Berechnet neue Gewichte. Wirft TrainingLocked wenn zu wenig Labels.

        Returns:
            Dict flag_name → neues Gewicht (nur geänderte Flags).
        """
        if not self.store.can_train(mandant_id):
            n = self.store.count(mandant_id) if mandant_id else self.store.total_count()
            raise TrainingLocked(
                f"Training gesperrt: {n}/{MIN_LABELS_FOR_TRAINING} Labels vorhanden."
            )

        if mandant_id:
            labels = self.store.load_all(mandant_id)
        else:
            labels = []
            for path in self.store.dir.glob("*_feedback.jsonl"):
                mid = path.stem.replace("_feedback", "")
                labels.extend(self.store.load_all(mid))

        # TP/FP pro Flag zählen
        flag_tp: dict[str, int] = {}
        flag_fp: dict[str, int] = {}
        for lbl in labels:
            flags = [f.strip() for f in lbl.anomaly_flags.split("|") if f.strip()]
            for flag in flags:
                if lbl.label == "tp":
                    flag_tp[flag] = flag_tp.get(flag, 0) + 1
                elif lbl.label == "fp":
                    flag_fp[flag] = flag_fp.get(flag, 0) + 1

        # Neue Gewichte berechnen
        new_weights: dict[str, float] = {}
        for flag_name, default_weight in WEIGHTS.items():
            tp = flag_tp.get(flag_name, 0)
            fp = flag_fp.get(flag_name, 0)
            total = tp + fp
            if total < self.MIN_FLAG_SAMPLES:
                continue  # Nicht genug Daten → Default beibehalten
            tp_rate = tp / total
            w = default_weight * (0.5 + tp_rate)
            new_weights[flag_name] = max(0.1, min(5.0, round(w, 2)))

        return new_weights
