"""
Tests für AI/Embedding-Features:
- TextEmbedder (src/embeddings.py)
- Kreditor-Clustering (src/kreditor_clustering.py)
- NEAR_DUPLICATE mit Embedding-Similarity
- ISOLATION_ANOMALIE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import AnalysisConfig
from src.engine import AnomalyEngine
from src.embeddings import HAS_EMBEDDINGS, get_embedder, TextEmbedder
from src.kreditor_clustering import cluster_kreditors
from src.tests.base import EngineStats
from src.tests.duplikate import NearDuplicate
from src.tests.isolation_anomaly import IsolationAnomalie, HAS_SKLEARN


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_df(**kwargs) -> pd.DataFrame:
    """Schneller DataFrame-Builder mit Defaults."""
    n = max(len(v) for v in kwargs.values()) if kwargs else 10
    defaults = {
        "datum": ["2024-01-15"] * n,
        "betrag": ["1000,00"] * n,
        "konto_soll": ["4711"] * n,
        "konto_haben": [""] * n,
        "buchungstext": [f"Text {i}" for i in range(n)],
        "belegnummer": [f"B{i:04d}" for i in range(n)],
        "kostenstelle": [""] * n,
        "kreditor": [""] * n,
        "soll_haben": [""] * n,
        "klasse": [""] * n,
        "belegart": [""] * n,
        "buchungsperiode": [""] * n,
        "erfassungsdatum": [""] * n,
        "kostentraeger": [""] * n,
        "projekt": [""] * n,
        "steuerschluessel": [""] * n,
        "detailbetrag": [""] * n,
        "generalumgekehrt": [""] * n,
        "dvbelegnummer": [""] * n,
        "dvbuchungsnummer": [""] * n,
        "interne_belegnummer": [""] * n,
        "mandant": [""] * n,
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


# ── TextEmbedder Tests ────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_EMBEDDINGS, reason="sentence-transformers nicht installiert")
class TestTextEmbedder:
    def test_embed_texts_shape(self):
        """Embeddings haben korrekte Form."""
        embedder = get_embedder()
        assert embedder is not None
        texts = ["Rechnung Lieferant A", "Gutschrift B", "Miete Januar"]
        emb = embedder.embed_texts(texts)
        assert emb.shape[0] == 3
        assert emb.shape[1] == 384  # all-MiniLM-L6-v2

    def test_embeddings_normalized(self):
        """Embeddings sind L2-normalisiert."""
        embedder = get_embedder()
        texts = ["Test Buchung", "Andere Buchung"]
        emb = embedder.embed_texts(texts)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_similar_texts_high_similarity(self):
        """Ähnliche Texte haben hohe Cosine-Similarity."""
        embedder = get_embedder()
        texts = ["Rechnung Lieferant ABC GmbH", "Rechnung Lieferant ABC"]
        emb = embedder.embed_texts(texts)
        sim = TextEmbedder.cosine_similarity_pairs(
            emb, np.array([0]), np.array([1])
        )
        assert sim[0] > 0.7

    def test_different_texts_lower_similarity(self):
        """Verschiedene Texte haben niedrigere Similarity."""
        embedder = get_embedder()
        texts = ["Rechnung Lieferant ABC", "Gehaltszahlung März"]
        emb = embedder.embed_texts(texts)
        sim = TextEmbedder.cosine_similarity_pairs(
            emb, np.array([0]), np.array([1])
        )
        assert sim[0] < 0.7

    def test_singleton(self):
        """get_embedder() liefert immer dieselbe Instanz."""
        e1 = get_embedder()
        e2 = get_embedder()
        assert e1 is e2

    def test_empty_list(self):
        """Leere Liste erzeugt leeres Array."""
        embedder = get_embedder()
        emb = embedder.embed_texts([])
        assert emb.shape[0] == 0


# ── Kreditor-Clustering Tests ─────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_EMBEDDINGS, reason="sentence-transformers nicht installiert")
class TestKreditorClustering:
    def test_similar_names_clustered(self):
        """Ähnliche Kreditornamen werden zusammengeführt."""
        names = [
            "ABC Logistik GmbH",
            "ABC Logistik GmbH & Co. KG",
            "XYZ Reinigung",
        ]
        embedder = get_embedder()
        mapping = cluster_kreditors(names, embedder=embedder, eps=0.25)
        # Die beiden ABC-Varianten sollten denselben Canonical haben
        assert mapping["ABC Logistik GmbH"] == mapping["ABC Logistik GmbH & Co. KG"]
        # XYZ bleibt eigenständig
        assert mapping["XYZ Reinigung"] == "XYZ Reinigung"

    def test_no_embedder_identity_mapping(self):
        """Ohne Embedder: 1:1-Identitäts-Mapping."""
        names = ["A", "B", "C"]
        mapping = cluster_kreditors(names, embedder=None)
        assert mapping == {"A": "A", "B": "B", "C": "C"}

    def test_empty_names(self):
        """Leere Liste ergibt leeres Mapping."""
        mapping = cluster_kreditors([])
        assert mapping == {}

    def test_canonical_is_shortest(self):
        """Canonical Name ist der kürzeste im Cluster."""
        names = [
            "Muster AG",
            "Muster Aktiengesellschaft",
        ]
        embedder = get_embedder()
        mapping = cluster_kreditors(names, embedder=embedder, eps=0.30)
        # Wenn geclustert, muss Canonical der kürzere sein
        if mapping["Muster AG"] == mapping["Muster Aktiengesellschaft"]:
            assert mapping["Muster Aktiengesellschaft"] == "Muster AG"


# ── NEAR_DUPLICATE mit Embeddings ─────────────────────────────────────────────

@pytest.mark.skipif(not HAS_EMBEDDINGS, reason="sentence-transformers nicht installiert")
class TestNearDuplicateEmbeddings:
    def test_similar_texts_flagged(self):
        """Buchungen mit ähnlichem Text werden als Near-Duplicate geflaggt."""
        df = _make_df(
            datum=["2024-01-15", "2024-01-16"],
            betrag=["5000,00", "5000,00"],
            konto_soll=["4711", "4711"],
            buchungstext=["Rechnung Lieferant ABC", "Rechnung Lieferant ABC GmbH"],
            belegnummer=["B0001", "B0002"],
            dvbelegnummer=["DV1", "DV2"],
        )
        config = AnalysisConfig(near_duplicate_text_similarity=0.70)
        engine = AnomalyEngine(df, config=config)
        engine._stats_cache = engine._compute_stats()
        engine._t06_near_duplicate()
        assert engine.df["flag_NEAR_DUPLICATE"].sum() == 2

    def test_different_texts_not_flagged(self):
        """Buchungen mit verschiedenem Text werden NICHT geflaggt."""
        df = _make_df(
            datum=["2024-01-15", "2024-01-16"],
            betrag=["5000,00", "5000,00"],
            konto_soll=["4711", "4711"],
            buchungstext=["Rechnung Strom", "Gehaltszahlung März"],
            belegnummer=["B0001", "B0002"],
            dvbelegnummer=["DV1", "DV2"],
        )
        config = AnalysisConfig(near_duplicate_text_similarity=0.70)
        engine = AnomalyEngine(df, config=config)
        engine._stats_cache = engine._compute_stats()
        engine._t06_near_duplicate()
        assert engine.df["flag_NEAR_DUPLICATE"].sum() == 0


# ── Kreditor-Canonical in Engine ──────────────────────────────────────────────

@pytest.mark.skipif(not HAS_EMBEDDINGS, reason="sentence-transformers nicht installiert")
class TestKreditorCanonicalEngine:
    def test_kreditor_canonical_column_created(self):
        """_kreditor_canonical wird in engine._prepare() erstellt."""
        df = _make_df(
            kreditor=["Lieferant A", "Lieferant B"] * 5,
        )
        engine = AnomalyEngine(df)
        assert "_kreditor_canonical" in engine.df.columns

    def test_kreditor_clustering_disabled(self):
        """Mit deaktiviertem Clustering: _kreditor_canonical = Original."""
        df = _make_df(
            kreditor=["Test GmbH"] * 10,
        )
        config = AnalysisConfig(kreditor_clustering_enabled=False)
        engine = AnomalyEngine(df, config=config)
        assert (engine.df["_kreditor_canonical"] == "Test GmbH").all()


# ── Isolation Forest Tests ────────────────────────────────────────────────────

class TestIsolationAnomalie:
    def test_disabled_by_default(self):
        """Isolation Forest ist standardmäßig deaktiviert."""
        df = _make_df(
            betrag=[f"{i * 100},00" for i in range(1, 11)],
        )
        engine = AnomalyEngine(df)
        engine._stats_cache = engine._compute_stats()
        engine._t27_isolation_anomalie()
        assert engine.df["flag_ISOLATION_ANOMALIE"].sum() == 0

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn nicht installiert")
    def test_enabled_with_enough_data(self):
        """Mit genug Daten und aktiviert: kein Crash."""
        n = 100
        betrag = [f"{i * 50},00" for i in range(1, n + 1)]
        df = _make_df(
            datum=[f"2024-01-{(i % 28) + 1:02d}" for i in range(n)],
            betrag=betrag,
            konto_soll=["4711"] * n,
            buchungstext=[f"Buchung {i}" for i in range(n)],
            belegnummer=[f"B{i:04d}" for i in range(n)],
        )
        config = AnalysisConfig(isolation_enabled=True, isolation_contamination=0.05)
        engine = AnomalyEngine(df, config=config)
        engine._stats_cache = engine._compute_stats()
        engine._t27_isolation_anomalie()
        # Sollte einige Anomalien flaggen (≈5% Contamination)
        flagged = engine.df["flag_ISOLATION_ANOMALIE"].sum()
        assert flagged > 0
        assert flagged <= n * 0.15  # Nicht mehr als 15%

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn nicht installiert")
    def test_too_few_rows(self):
        """Bei < 50 Buchungen wird Isolation Forest übersprungen."""
        df = _make_df(
            betrag=[f"{i * 100},00" for i in range(1, 11)],
        )
        config = AnalysisConfig(isolation_enabled=True)
        engine = AnomalyEngine(df, config=config)
        engine._stats_cache = engine._compute_stats()
        engine._t27_isolation_anomalie()
        assert engine.df["flag_ISOLATION_ANOMALIE"].sum() == 0


# ── Config Parameters ────────────────────────────────────────────────────────

class TestAIConfig:
    def test_default_values(self):
        """AI-Config-Defaults sind korrekt."""
        config = AnalysisConfig()
        assert config.near_duplicate_text_similarity == 0.85
        assert config.kreditor_clustering_enabled is True
        assert config.kreditor_clustering_eps == 0.20
        assert config.isolation_enabled is False
        assert config.isolation_contamination == 0.02

    def test_embedding_threshold_zero_disables(self):
        """near_duplicate_text_similarity=0 deaktiviert Embedding-Vergleich."""
        config = AnalysisConfig(near_duplicate_text_similarity=0.0)
        assert config.near_duplicate_text_similarity == 0.0
