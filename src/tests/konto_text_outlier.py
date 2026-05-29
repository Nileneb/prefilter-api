"""
Buchungs-Anomalie Pre-Filter — KONTO_TEXT_OUTLIER

Pro Konto bilden die Buchungstext-Embeddings ein oder mehrere dichte Cluster
(z.B. "Adressen" für Mieten). DBSCAN markiert Punkte, die in keinem dichten
Cluster liegen, als Noise — das sind die semantischen Ausreißer RELATIV zur
kontoeigenen Verteilung. Unsupervised, kein Vergleich gegen den Kontonamen.
"""

from __future__ import annotations

import numpy as np

from src.config import AnalysisConfig
from src.tests.base import AnomalyTest, EngineStats

try:
    from sklearn.cluster import DBSCAN  # type: ignore[import-untyped]
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def find_text_outliers(
    emb: np.ndarray, eps: float, min_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Findet Ausreißer-Buchungen anhand DBSCAN-Cluster auf Embeddings.

    Args:
        emb: (N, D) L2-normierte Embeddings der Buchungstexte EINES Kontos.
        eps: DBSCAN epsilon (cosine-Distanz).
        min_samples: Mindest-Punkte für ein dichtes Cluster.

    Returns:
        (mask, fit): mask[i]=True → Buchung i ist Ausreißer (DBSCAN-Noise).
        fit[i] = cosine zum nächsten Cluster-Centroid (1.0 wenn kein Cluster).
    """
    n = emb.shape[0]
    if n < min_samples or not HAS_SKLEARN:
        return np.zeros(n, dtype=bool), np.ones(n, dtype=np.float32)

    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(emb).labels_
    mask = labels == -1

    # Centroids je Cluster (Mittelvektor, renormiert) für erklärbaren Fit-Score
    centroids = []
    for lab in sorted(set(labels) - {-1}):
        c = emb[labels == lab].mean(axis=0)
        norm = np.linalg.norm(c)
        centroids.append(c / norm if norm else c)
    if centroids:
        cmat = np.vstack(centroids)
        fit = (emb @ cmat.T).max(axis=1).astype(np.float32)
    else:
        # Kein dichtes Cluster gefunden → kein verlässliches Profil
        return np.zeros(n, dtype=bool), np.ones(n, dtype=np.float32)

    return mask, fit
