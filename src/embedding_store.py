"""
Persistenter Embedding-Cache (text_hash → vector) auf SQLite.

WHY: Große Mandanten (200MB+) wurden bei jedem Lauf neu eingebettet. Identische
Buchungstexte/Kontonamen werden hier einmal eingebettet und wiederverwendet.
Exakter Key-Cache (kein ANN) — der Outlier-Test braucht keine Ähnlichkeitssuche.

Public API:
    embed_cached(embedder, texts) -> np.ndarray   # aligned zur Eingabe (inkl. Dups)
"""

from __future__ import annotations

import hashlib
import os
import sqlite3

import numpy as np

from src.logging_config import get_logger

logger = get_logger("prefilter.embedding_store")

CACHE_DB = os.environ.get("EMBEDDING_CACHE_DB", "data/embedding_cache.db")
_CACHE_ENABLED = os.environ.get("EMBEDDING_CACHE_ENABLED", "1") != "0"


def _key(model: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


_SQLITE_BATCH = 900  # WHY: SQLite default SQLITE_LIMIT_VARIABLE_NUMBER=999; stay safely under


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(CACHE_DB) or ".", exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "  key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL"
        ")"
    )
    return conn


def embed_cached(embedder, texts: list[str]) -> np.ndarray:
    """Wie embedder.embed_texts(), aber mit persistentem Cache.

    Gibt ein Array zurück, das 1:1 zur Eingabeliste ausgerichtet ist (inkl.
    Duplikate, Reihenfolge erhalten).
    """
    if not texts:
        return np.empty((0,), dtype=np.float32)

    if not _CACHE_ENABLED:
        return embedder.embed_texts(texts)

    model = embedder.model_name
    unique = list(dict.fromkeys(texts))  # Reihenfolge erhalten, dedupe
    vectors: dict[str, np.ndarray] = {}

    conn = _connect()
    try:
        keys = {t: _key(model, t) for t in unique}
        key_list = [keys[t] for t in unique]
        by_key: dict[str, np.ndarray] = {}
        for i in range(0, len(key_list), _SQLITE_BATCH):
            batch = key_list[i : i + _SQLITE_BATCH]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT key, dim, vector FROM embeddings WHERE key IN ({placeholders})",
                batch,
            ).fetchall()
            by_key.update(
                {k: np.frombuffer(buf, dtype=np.float32).reshape(dim) for k, dim, buf in rows}
            )

        misses = [t for t in unique if keys[t] not in by_key]
        for t in unique:
            if keys[t] in by_key:
                vectors[t] = by_key[keys[t]]

        if misses:
            new = np.asarray(embedder.embed_texts(misses), dtype=np.float32)
            ins = []
            for t, v in zip(misses, new):
                vectors[t] = v
                ins.append((keys[t], int(v.shape[0]), v.tobytes()))
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, dim, vector) VALUES (?, ?, ?)",
                ins,
            )
            conn.commit()
            logger.info("Embedding-Cache", hits=len(unique) - len(misses), misses=len(misses))
    finally:
        conn.close()

    return np.vstack([vectors[t] for t in texts])
