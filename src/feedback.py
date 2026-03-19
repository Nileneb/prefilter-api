"""
Buchungs-Anomalie Pre-Filter — Prüfer-Feedback

Speichert Labels persistent (JSON-Lines), ermöglicht späteres Training.
Jeder Mandant hat eine eigene Feedback-Datei.

⛔ Training ist GESPERRT bis mindestens MIN_LABELS_FOR_TRAINING Labels vorhanden.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

FEEDBACK_DIR = os.environ.get("FEEDBACK_DIR", "data/feedback")
MIN_LABELS_FOR_STATS = 200
MIN_LABELS_FOR_TRAINING = 500


@dataclass
class FeedbackLabel:
    """Ein einzelnes Prüfer-Label für eine verdächtige Buchung."""

    mandant_id: str
    analysis_timestamp: str  # Wann die Analyse lief
    row_index: int  # Index im DataFrame
    belegnummer: str
    anomaly_score: float
    anomaly_flags: str  # pipe-getrennt
    label: str  # "tp" (true positive) | "fp" (false positive) | "unsure"
    pruefer: str  # Wer hat gelabelt
    kommentar: str = ""  # Optionaler Freitext
    labeled_at: str = ""  # Zeitstempel des Labels

    def __post_init__(self) -> None:
        if not self.labeled_at:
            self.labeled_at = datetime.now().isoformat()


class FeedbackStore:
    """Persistente Speicherung von Prüfer-Feedback (JSON-Lines)."""

    def __init__(self, feedback_dir: str = FEEDBACK_DIR) -> None:
        self.dir = Path(feedback_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, mandant_id: str) -> Path:
        # Sanitize mandant_id to prevent path traversal
        safe_id = "".join(c for c in mandant_id if c.isalnum() or c in "-_")
        if not safe_id:
            safe_id = "unknown"
        return self.dir / f"{safe_id}_feedback.jsonl"

    def save(self, label: FeedbackLabel) -> None:
        """Speichert ein Label (append)."""
        with open(self._path(label.mandant_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")

    def load_all(self, mandant_id: str) -> list[FeedbackLabel]:
        """Lädt alle Labels eines Mandanten."""
        path = self._path(mandant_id)
        if not path.exists():
            return []
        labels: list[FeedbackLabel] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(FeedbackLabel(**json.loads(line)))
        return labels

    def count(self, mandant_id: str) -> int:
        """Anzahl Labels eines Mandanten."""
        return len(self.load_all(mandant_id))

    def total_count(self) -> int:
        """Gesamtzahl Labels über alle Mandanten."""
        total = 0
        for path in self.dir.glob("*_feedback.jsonl"):
            with open(path, encoding="utf-8") as f:
                total += sum(1 for line in f if line.strip())
        return total

    def can_show_stats(self, mandant_id: str | None = None) -> bool:
        """True wenn genug Labels für Statistiken vorhanden."""
        n = self.count(mandant_id) if mandant_id else self.total_count()
        return n >= MIN_LABELS_FOR_STATS

    def can_train(self, mandant_id: str | None = None) -> bool:
        """True wenn genug Labels für Training vorhanden."""
        n = self.count(mandant_id) if mandant_id else self.total_count()
        return n >= MIN_LABELS_FOR_TRAINING
