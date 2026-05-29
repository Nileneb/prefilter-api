from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import AnalysisConfig
from src.embeddings import HAS_EMBEDDINGS
from src.tests.base import EngineStats
from src.tests.konto_text_outlier import KontoTextOutlier, find_text_outliers


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_dense_cluster_plus_one_outlier():
    # 6 fast identische Vektoren (ein Cluster) + 1 orthogonaler Ausreißer
    base = _norm(np.tile([1.0, 0.0, 0.0], (6, 1)) + 0.01)
    outlier = _norm(np.array([[0.0, 1.0, 0.0]]))
    emb = np.vstack([base, outlier])
    mask, fit = find_text_outliers(emb, eps=0.15, min_samples=3)
    assert mask.tolist() == [False] * 6 + [True]   # nur der Ausreißer
    assert fit[-1] < fit[0]                          # Ausreißer hat schlechteren Fit


def test_two_legit_clusters_no_false_positive():
    # Zwei legitime Profile → Centroid liegt dazwischen (fit ~0.71 für beide);
    # beim Produktions-eps (0.55 → Schwelle 0.45) wird KEINER als Ausreißer geflaggt.
    a = _norm(np.tile([1.0, 0.0, 0.0], (4, 1)) + 0.01)
    b = _norm(np.tile([0.0, 1.0, 0.0], (4, 1)) + 0.01)
    emb = np.vstack([a, b])
    mask, _ = find_text_outliers(emb, eps=0.55, min_samples=3)
    assert mask.sum() == 0


def test_too_few_rows_returns_all_false():
    emb = _norm(np.random.RandomState(0).rand(2, 3))
    mask, fit = find_text_outliers(emb, eps=0.55, min_samples=3)
    assert mask.tolist() == [False, False]


def test_scales_to_large_account_no_quadratic_blowup():
    """20k Buchungen müssen sofort durchlaufen (O(n), kein n×n) — die DBSCAN-
    Variante hätte hier eine ~1.6GB-Distanzmatrix gebaut (OOM auf 200MB-Mandant)."""
    rng = np.random.RandomState(0)
    dense = _norm(np.tile([1.0, 0.0, 0.0], (20_000, 1)) + rng.rand(20_000, 3) * 0.02)
    outliers = _norm(np.array([[0.0, 1.0, 0.0]] * 5))
    emb = np.vstack([dense, outliers]).astype(np.float32)
    mask, fit = find_text_outliers(emb, eps=0.55, min_samples=8)
    assert mask.shape == (20_005,)
    assert mask[-5:].all()            # die 5 orthogonalen Ausreißer geflaggt
    assert mask[:20_000].sum() == 0   # dichte Masse nicht geflaggt


@pytest.mark.skipif(not HAS_EMBEDDINGS, reason="sentence-transformers nicht installiert")
def test_outlier_booking_flagged_within_account():
    # Konto 4711: 8 Adress-artige Texte + 1 themenfremder ("Gehaltszahlung")
    texts = [
        "Hauptstraße 5 Berlin", "Bahnhofstr 12 Hamburg", "Lindenweg 3 Köln",
        "Marktplatz 1 München", "Gartenstr 8 Bremen", "Ringstr 22 Essen",
        "Seestraße 9 Kiel", "Parkallee 4 Bonn",
        "Gehaltszahlung Lohn März Mitarbeiter",  # Ausreißer
    ]
    df = pd.DataFrame({
        "buchungstext": texts,
        "konto_soll": ["4711"] * len(texts),
        "_konto_in_scope": [True] * len(texts),
    })
    n = KontoTextOutlier().run(df, EngineStats(), AnalysisConfig())
    assert n >= 1
    flagged = df["flag_KONTO_TEXT_OUTLIER"].fillna(False)
    assert flagged.iloc[-1]                 # Gehaltszahlung geflaggt
    assert flagged.iloc[:-1].sum() == 0     # Adressen NICHT geflaggt
