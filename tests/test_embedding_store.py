from __future__ import annotations

import numpy as np
import pytest

import src.embedding_store as es


@pytest.fixture
def store_db(tmp_path, monkeypatch):
    db = tmp_path / "emb.db"
    monkeypatch.setattr(es, "CACHE_DB", str(db))
    return db


class _FakeEmbedder:
    """Liefert deterministische 4-dim Vektoren, zählt Embedding-Aufrufe."""
    model_name = "fake-model"

    def __init__(self):
        self.embedded: list[str] = []

    def embed_texts(self, texts):
        self.embedded.extend(texts)
        # deterministisch aus Textlänge, L2-normiert
        out = []
        for t in texts:
            v = np.array([len(t), t.count("a"), t.count("e"), 1.0], dtype=np.float32)
            v /= (np.linalg.norm(v) or 1.0)
            out.append(v)
        return np.vstack(out)


def test_embed_cached_returns_aligned_array(store_db):
    emb = _FakeEmbedder()
    texts = ["miete", "strom", "miete"]  # Duplikat
    out = es.embed_cached(emb, texts)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out[0], out[2])  # gleiches Wort → gleicher Vektor


def test_embed_cached_only_embeds_misses(store_db):
    emb = _FakeEmbedder()
    es.embed_cached(emb, ["miete", "strom"])
    assert sorted(emb.embedded) == ["miete", "strom"]
    # Zweiter Aufruf: alles im Cache → kein erneutes Embedding
    emb.embedded.clear()
    out = es.embed_cached(emb, ["strom", "miete", "neu"])
    assert emb.embedded == ["neu"]  # nur der Miss
    assert out.shape == (3, 4)


def test_cache_keyed_by_model(store_db):
    emb = _FakeEmbedder()
    es.embed_cached(emb, ["miete"])
    emb.embedded.clear()
    emb.model_name = "other-model"
    es.embed_cached(emb, ["miete"])  # anderer Modell-Key → neuer Miss
    assert emb.embedded == ["miete"]


def test_empty_input(store_db):
    out = es.embed_cached(_FakeEmbedder(), [])
    assert out.shape == (0,)
