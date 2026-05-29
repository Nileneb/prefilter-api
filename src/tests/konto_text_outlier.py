"""
Buchungs-Anomalie Pre-Filter — KONTO_TEXT_OUTLIER

Pro Konto bildet das gemeinsame Textprofil (Centroid = Mittelvektor aller
Buchungstext-Embeddings) den "Normalfall". Eine Buchung ist ein semantischer
Ausreißer, wenn ihr Text weit vom Konto-Centroid entfernt liegt (Cosine zum
Centroid < 1 - eps). Unsupervised, kein Vergleich gegen den Kontonamen.

WHY(scale): Bewusst O(n) Zeit / O(d) Speicher (ein Centroid + ein Dot-Product),
KEIN DBSCAN mit n×n-Distanzmatrix — die sprengte bei großen Konten (zehntausende
Buchungen) den RAM (SIGKILL/OOM auf 200MB-Mandanten). So wird JEDE Buchung
verarbeitet, ohne Cap/Sampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AnalysisConfig
from src.embedding_store import embed_cached
from src.embeddings import HAS_EMBEDDINGS, get_embedder
from src.tests.base import AnomalyTest, EngineStats


def find_text_outliers(
    emb: np.ndarray, eps: float, min_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Ausreißer = Buchungstext weit vom gemeinsamen Konto-Textprofil entfernt.

    Profil = Centroid (renormierter Mittelvektor) aller Konto-Embeddings.
    fit[i] = Cosine(emb[i], Centroid). Ausreißer wenn fit < (1 - eps).
    O(n) Zeit, O(d) Speicher — skaliert auf beliebig große Konten.

    Args:
        emb: (N, D) L2-normierte Embeddings der Buchungstexte EINES Kontos.
        eps: max. Cosine-Distanz zum Centroid, ab der eine Buchung als Ausreißer
             gilt (fit < 1 - eps).
        min_samples: Mindest-Buchungen für ein verlässliches Profil; darunter
             keine Wertung (alles False).

    Returns:
        (mask, fit): mask[i]=True → Buchung i ist Ausreißer.
        fit[i] = Cosine zum Konto-Centroid.
    """
    n = emb.shape[0]
    if n < min_samples:
        return np.zeros(n, dtype=bool), np.ones(n, dtype=np.float32)

    centroid = emb.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        return np.zeros(n, dtype=bool), np.ones(n, dtype=np.float32)
    centroid = centroid / norm

    fit = (emb @ centroid).astype(np.float32)  # Cosine zum Centroid, O(n·d)
    mask = fit < np.float32(1.0 - eps)
    return mask, fit


class KontoTextOutlier(AnomalyTest):
    name = "KONTO_TEXT_OUTLIER"
    weight = 2.0
    critical = False
    required_columns = ["buchungstext", "konto_soll", "_konto_in_scope"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        if not HAS_EMBEDDINGS:
            self.log("SKIP: embeddings fehlen")
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

        all_texts = sub["buchungstext"].astype(str).str.strip().tolist()
        all_emb = embed_cached(embedder, all_texts)
        positions = {idx: i for i, idx in enumerate(sub.index)}

        flagged_idx: list = []
        for konto, grp in sub.groupby("konto_soll", observed=True):
            if len(grp) < min_bookings:
                continue
            pos = [positions[i] for i in grp.index]
            emb = all_emb[pos]
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
