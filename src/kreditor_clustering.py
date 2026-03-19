"""
Buchungs-Anomalie Pre-Filter — Kreditor-Clustering

DBSCAN auf Text-Embeddings für kanonische Kreditornamen.
Cluster-Repräsentant = kürzester Name im Cluster.

Graceful Degradation: Ohne Embeddings/sklearn wird ein 1:1-Mapping zurückgegeben.

Public API:
    cluster_kreditors(names, embedder, eps) -> dict[str, str]
"""

from __future__ import annotations

import numpy as np

from src.logging_config import get_logger

logger = get_logger("prefilter.kreditor_clustering")

try:
    from sklearn.cluster import DBSCAN  # type: ignore[import-untyped]
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.info("scikit-learn nicht installiert — Kreditor-Clustering deaktiviert")


def cluster_kreditors(
    names: list[str],
    embedder: object | None = None,
    eps: float = 0.20,
) -> dict[str, str]:
    """Clustert Kreditornamen via DBSCAN auf Embeddings.

    Args:
        names: Liste eindeutiger Kreditornamen.
        embedder: TextEmbedder-Instanz mit embed_texts().
        eps: DBSCAN epsilon (1 - cosine_similarity).

    Returns:
        Mapping {original_name: canonical_name}.
        Canonical = kürzester Name im Cluster.
        Ohne Embeddings/sklearn: 1:1-Identitäts-Mapping.
    """
    if not names:
        return {}

    # Fallback: 1:1-Mapping
    if embedder is None or not HAS_SKLEARN:
        logger.info("Kreditor-Clustering übersprungen (keine Embeddings/sklearn)")
        return {n: n for n in names}

    # Embeddings berechnen
    emb = embedder.embed_texts(names)

    # DBSCAN mit Cosine-Distanz (= 1 - similarity auf normalisierten Vektoren)
    clustering = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit(emb)
    labels = clustering.labels_

    # Mapping bauen: kürzester Name pro Cluster = Canonical
    mapping: dict[str, str] = {}
    clusters: dict[int, list[str]] = {}

    for name, label in zip(names, labels):
        if label == -1:
            # Noise → bleibt eigenständig
            mapping[name] = name
        else:
            clusters.setdefault(label, []).append(name)

    for label, members in clusters.items():
        canonical = min(members, key=len)
        for m in members:
            mapping[m] = canonical

    n_clusters = len(clusters)
    n_noise = int((labels == -1).sum())
    n_merged = sum(len(m) for m in clusters.values()) - n_clusters
    logger.info(
        "Kreditor-Clustering fertig",
        n_unique=len(names),
        n_clusters=n_clusters,
        n_noise=n_noise,
        n_merged=n_merged,
        eps=eps,
    )

    return mapping
