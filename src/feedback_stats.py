"""
Buchungs-Anomalie Pre-Filter — Feedback-Statistiken

Berechnet Precision, Recall, FP-Rate pro Flag aus gesammelten Labels.
Nur verfügbar ab MIN_LABELS_FOR_STATS Labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from src.feedback import FeedbackStore, FeedbackLabel, MIN_LABELS_FOR_STATS


@dataclass
class FlagStats:
    """Statistiken für ein einzelnes Flag."""

    flag: str
    tp: int = 0
    fp: int = 0
    unsure: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.unsure

    @property
    def precision(self) -> float:
        """Anteil True Positives an gelabelten (ohne unsure)."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def fp_rate(self) -> float:
        """Anteil False Positives an gelabelten (ohne unsure)."""
        denom = self.tp + self.fp
        return self.fp / denom if denom > 0 else 0.0


def compute_feedback_stats(
    labels: list[FeedbackLabel],
) -> dict[str, FlagStats]:
    """Berechnet pro Flag die TP/FP/Unsure-Verteilung.

    Gibt ein Dict flag_name → FlagStats zurück.
    """
    stats: dict[str, FlagStats] = {}

    for lbl in labels:
        flags = [f.strip() for f in lbl.anomaly_flags.split("|") if f.strip()]
        for flag in flags:
            if flag not in stats:
                stats[flag] = FlagStats(flag=flag)
            if lbl.label == "tp":
                stats[flag].tp += 1
            elif lbl.label == "fp":
                stats[flag].fp += 1
            elif lbl.label == "unsure":
                stats[flag].unsure += 1

    return stats


def format_feedback_report(
    store: FeedbackStore,
    mandant_id: str | None = None,
) -> str:
    """Formatiert einen Feedback-Bericht als Text.

    Gibt Hinweis zurück wenn zu wenig Labels vorhanden.
    """
    if mandant_id:
        labels = store.load_all(mandant_id)
        scope = f"Mandant {mandant_id}"
    else:
        labels = []
        for path in store.dir.glob("*_feedback.jsonl"):
            mid = path.stem.replace("_feedback", "")
            labels.extend(store.load_all(mid))
        scope = "Alle Mandanten"

    if len(labels) < MIN_LABELS_FOR_STATS:
        return (
            f"⚠️ Zu wenig Labels für Statistiken ({len(labels)}/{MIN_LABELS_FOR_STATS}).\n"
            f"Noch {MIN_LABELS_FOR_STATS - len(labels)} Labels erforderlich."
        )

    stats = compute_feedback_stats(labels)
    if not stats:
        return "Keine Flag-Daten in den Labels gefunden."

    lines = [
        f"📊 Feedback-Statistiken — {scope}",
        f"   Gesamt: {len(labels)} Labels\n",
        f"{'Flag':<30} {'TP':>5} {'FP':>5} {'?':>5} {'Prec':>7} {'FP%':>7}",
        "-" * 65,
    ]
    for flag_name in sorted(stats.keys()):
        fs = stats[flag_name]
        lines.append(
            f"{fs.flag:<30} {fs.tp:>5} {fs.fp:>5} {fs.unsure:>5} "
            f"{fs.precision:>6.1%} {fs.fp_rate:>6.1%}"
        )

    # Gesamt
    total_tp = sum(fs.tp for fs in stats.values())
    total_fp = sum(fs.fp for fs in stats.values())
    total_unsure = sum(fs.unsure for fs in stats.values())
    total_denom = total_tp + total_fp
    overall_prec = total_tp / total_denom if total_denom > 0 else 0.0
    overall_fp_rate = total_fp / total_denom if total_denom > 0 else 0.0
    lines.append("-" * 65)
    lines.append(
        f"{'GESAMT':<30} {total_tp:>5} {total_fp:>5} {total_unsure:>5} "
        f"{overall_prec:>6.1%} {overall_fp_rate:>6.1%}"
    )

    return "\n".join(lines)
