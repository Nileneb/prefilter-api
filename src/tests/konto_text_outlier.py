"""
Buchungs-Anomalie Pre-Filter — KONTO_TEXT_OUTLIER

Pro Konto bilden die Buchungstext-Embeddings ein oder mehrere dichte Cluster
(z.B. "Adressen" für Mieten). DBSCAN markiert Punkte, die in keinem dichten
Cluster liegen, als Noise — das sind die semantischen Ausreißer RELATIV zur
kontoeigenen Verteilung. Unsupervised, kein Vergleich gegen den Kontonamen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AnalysisConfig
from src.embedding_store import embed_cached
from src.embeddings import HAS_EMBEDDINGS, get_embedder
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


class KontoTextOutlier(AnomalyTest):
    name = "KONTO_TEXT_OUTLIER"
    weight = 2.0
    critical = False
    required_columns = ["buchungstext", "konto_soll", "_konto_in_scope"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        if not HAS_EMBEDDINGS or not HAS_SKLEARN:
            self.log("SKIP: embeddings/sklearn fehlen")
            return 0
        embedder = get_embedder()
        if embedder is None:
            return 0

        min_bookings = config.konto_text_outlier_min_bookings
        eps = config.konto_text_outlier_eps
        min_samples = config.konto_text_outlier_min_samples

        has_text = df["buchungstext"].astype(str).str.strip().ne("")
        if "_konto_in_scope" in df.columns:
            has_text = has_text & df["_konto_in_scope"].fillna(False).astype(bool)
        sub = df.loc[has_text]
        if sub.empty:
            return 0

        flagged_idx: list = []
        for konto, grp in sub.groupby("konto_soll", observed=True):
            if len(grp) < min_bookings:
                continue
            texts = grp["buchungstext"].astype(str).str.strip().tolist()
            emb = embed_cached(embedder, texts)
            mask, fit = find_text_outliers(emb, eps=eps, min_samples=min_samples)
            n_out = int(mask.sum())
            if n_out:
                self.log("Konto-Ausreißer", konto=str(konto), n=n_out,
                         min_fit=round(float(fit[mask].min()), 3))
                flagged_idx.extend(grp.index[mask].tolist())

        if flagged_idx:
            df.loc[flagged_idx, f"flag_{self.name}"] = True
        return len(flagged_idx)


def get_tests() -> list[AnomalyTest]:
    return [KontoTextOutlier()]
