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
        # Gleiche dtype/shape-Semantik wie der Cache-Pfad (float32, 2D).
        return np.asarray(embedder.embed_texts(texts), dtype=np.float32)

    model = embedder.model_name
    unique = list(dict.fromkeys(texts))  # Reihenfolge erhalten, dedupe
    vectors: dict[str, np.ndarray] = {}

    conn = _connect()
    try:
        keys = {t: _key(model, t) for t in unique}
        # Lookup-Keys über eine verbindungslokale TEMP-Tabelle joinen statt eines
        # dynamischen IN(...): alle SQL-Strings sind Literale (keine String-
        # Konkatenation, keine SQLite-Variablen-Limit-Grenze), die Keys sind
        # gebundene Parameter. Robust für sehr viele unique Texte (großer Mandant).
        conn.execute("CREATE TEMP TABLE _lookup_keys (key TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT OR IGNORE INTO _lookup_keys (key) VALUES (?)",
            [(keys[t],) for t in unique],
        )
        rows = conn.execute(
            "SELECT e.key, e.dim, e.vector FROM embeddings e "
            "JOIN _lookup_keys k ON e.key = k.key"
        ).fetchall()
        by_key: dict[str, np.ndarray] = {
            k: np.frombuffer(buf, dtype=np.float32).reshape(dim) for k, dim, buf in rows
        }

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
