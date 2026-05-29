# KONTO_TEXT_OUTLIER + Persistenter Embedding-Store — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Buchungstexte werden pro Konto auf semantische Ausreißer geprüft (statt Text-gegen-Kontoname), und Embeddings werden persistent gecacht, damit große Mandanten nicht bei jedem Lauf neu eingebettet werden.

**Architecture:** Zwei Bausteine. (1) `EmbeddingStore` — SQLite-Cache `text_hash → vector`, gekapselt hinter `embed_cached(embedder, texts)`; eingehängt an allen schweren Embedding-Aufrufen. (2) `KONTO_TEXT_OUTLIER` — pro Konto DBSCAN-Cluster-Set auf den Buchungstext-Embeddings; Buchungen, die in keinem dichten Cluster liegen (DBSCAN-Noise), sind Ausreißer relativ zur kontoeigenen Verteilung. Unsupervised — keine LLM-Ground-Truth nötig.

**Tech Stack:** Python, pandas, numpy, scikit-learn (DBSCAN, vorhanden), sentence-transformers (optional, `HAS_EMBEDDINGS`), SQLite (stdlib), Gradio, pytest.

**Backend-Entscheidung (bestätigungsbedürftig):** Der Embedding-Store ist eine **SQLite-Key-Value-Tabelle** (`text_hash → vector`), kein ANN-Vektor-DB-Server. Begründung: Das erklärte Ziel ist „einmal einbetten" — dafür genügt ein exakter Key-Cache (kein Ops-Overhead, embedded, atomar). Der Centroid/Cluster-Ansatz braucht **keine** ANN-Ähnlichkeitssuche (Centroid = Mittelvektor, Fit = Dot-Product). Ein echter Vektor-DB-Server (Chroma) lohnt erst, wenn kontoübergreifende Ähnlichkeitssuche oder persistente Konto-Profile über Läufe hinweg gebraucht werden → als optionale Phase 4 vermerkt, nicht jetzt.

**Scope-Hinweis:** Phase 1 (Embedding-Store) ist unabhängig wertvoll und kann als eigener PR zuerst gemergt werden. Phase 2/3 (Outlier-Test + UI) bauen darauf auf.

---

## File Structure

| Datei | Verantwortung |
|---|---|
| `src/embedding_store.py` (neu) | SQLite-Cache + `embed_cached(embedder, texts)` |
| `src/embeddings.py` (mod) | `_MODEL_NAME` als `model_name`-Property exponieren |
| `src/engine.py` (mod) | `embed_cached` statt `embed_texts` an den schweren Stellen; neuen Test registrieren |
| `src/tests/konto_text_outlier.py` (neu) | `find_text_outliers()` (pure) + `KontoTextOutlier`-Test |
| `src/tests/text_konto_match.py` (mod) | `embed_cached` statt `embed_texts` |
| `src/kreditor_clustering.py` (mod) | `embed_cached` statt `embed_texts` |
| `src/config.py` (mod) | `konto_text_outlier_*`-Felder + `embedding_cache_enabled` |
| `src/validator.py` (mod) | `KONTO_TEXT_OUTLIER` in TEST_REQUIREMENTS + Embedding-Kategorie |
| `app.py` (mod) | Slider (eps/min_bookings) + config_dict + click-inputs + #22-Persistenz |
| `tests/test_embedding_store.py` (neu) | Cache round-trip, miss/hit, model-key |
| `tests/test_konto_text_outlier.py` (neu) | `find_text_outliers` (synthetische Vektoren, ohne Modell) + Engine-Integration (skipif) |
| `.gitignore` (mod) | `data/embedding_cache.db*` |

---

## Phase 1 — Persistenter Embedding-Store

### Task 1: model_name auf TextEmbedder exponieren

**Files:**
- Modify: `src/embeddings.py`
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_embeddings.py — ans Ende anhängen
def test_embedder_exposes_model_name():
    from src.embeddings import _MODEL_NAME, TextEmbedder
    assert TextEmbedder().model_name == _MODEL_NAME
```

- [ ] **Step 2: Run → FAIL**

Run: `python -m pytest tests/test_embeddings.py::test_embedder_exposes_model_name -v`
Expected: FAIL (`AttributeError: 'TextEmbedder' object has no attribute 'model_name'`)

- [ ] **Step 3: Implement**

```python
# src/embeddings.py — in class TextEmbedder, direkt nach __init__
    @property
    def model_name(self) -> str:
        return _MODEL_NAME
```

- [ ] **Step 4: Run → PASS**

Run: `python -m pytest tests/test_embeddings.py::test_embedder_exposes_model_name -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/embeddings.py tests/test_embeddings.py
git commit -m "feat: expose model_name on TextEmbedder (Embedding-Cache-Key)"
```

### Task 2: EmbeddingStore (SQLite-Cache)

**Files:**
- Create: `src/embedding_store.py`
- Test: `tests/test_embedding_store.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_embedding_store.py
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
```

- [ ] **Step 2: Run → FAIL**

Run: `python -m pytest tests/test_embedding_store.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.embedding_store'`)

- [ ] **Step 3: Implement**

```python
# src/embedding_store.py
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
        placeholders = ",".join("?" for _ in unique)
        rows = conn.execute(
            f"SELECT key, dim, vector FROM embeddings WHERE key IN ({placeholders})",
            [keys[t] for t in unique],
        ).fetchall()
        by_key = {k: np.frombuffer(buf, dtype=np.float32).reshape(dim) for k, dim, buf in rows}

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
```

- [ ] **Step 4: Run → PASS**

Run: `python -m pytest tests/test_embedding_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/embedding_store.py tests/test_embedding_store.py
git commit -m "feat: persistenter Embedding-Cache (SQLite) — embed_cached()"
```

### Task 3: Cache an den schweren Embedding-Stellen einhängen

**Files:**
- Modify: `src/engine.py:247-248`, `src/tests/text_konto_match.py` (Aufruf `embedder.embed_texts(all_texts)`), `src/kreditor_clustering.py:53`
- Test: `tests/test_embeddings.py` (Regression: bestehende Tests bleiben grün)

- [ ] **Step 1: engine._prepare auf Cache umstellen**

```python
# src/engine.py — Import oben ergänzen
from src.embedding_store import embed_cached
# ... in _prepare, statt: self._text_embeddings = embedder.embed_texts(texts)
                    self._text_embeddings = embed_cached(embedder, texts)
```

- [ ] **Step 2: text_konto_match auf Cache umstellen**

```python
# src/tests/text_konto_match.py — Import ergänzen
from src.embedding_store import embed_cached
# ... statt: embeddings = embedder.embed_texts(all_texts)
        embeddings = embed_cached(embedder, all_texts)
```

- [ ] **Step 3: kreditor_clustering auf Cache umstellen**

```python
# src/kreditor_clustering.py — Import ergänzen
from src.embedding_store import embed_cached
# ... statt: emb = embedder.embed_texts(names)
    emb = embed_cached(embedder, names)
```

- [ ] **Step 4: Run gesamte Suite → PASS**

Run: `EMBEDDING_CACHE_DB=/tmp/test_emb_cache.db python -m pytest tests/ -q`
Expected: alle vorher grünen Tests bleiben grün (embedding-Tests skip ohne sentence-transformers)

- [ ] **Step 5: .gitignore + Commit**

```bash
printf '\n# Persistenter Embedding-Cache (#vektordb)\ndata/embedding_cache.db\ndata/embedding_cache.db-*\n' >> .gitignore
git add src/engine.py src/tests/text_konto_match.py src/kreditor_clustering.py .gitignore
git commit -m "feat: Embedding-Cache an allen schweren Embedding-Aufrufen nutzen"
```

---

## Phase 2 — KONTO_TEXT_OUTLIER (per-Konto Cluster-Outlier)

### Task 4: Pure Outlier-Funktion `find_text_outliers`

**Files:**
- Create: `src/tests/konto_text_outlier.py` (zunächst nur die pure Funktion)
- Test: `tests/test_konto_text_outlier.py`

- [ ] **Step 1: Failing tests (synthetische Vektoren, KEIN Modell nötig)**

```python
# tests/test_konto_text_outlier.py
from __future__ import annotations

import numpy as np
import pytest

from src.tests.konto_text_outlier import find_text_outliers


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
    # Zwei legitime Profile (z.B. zwei Adressformate) → keiner ist Ausreißer
    a = _norm(np.tile([1.0, 0.0, 0.0], (4, 1)) + 0.01)
    b = _norm(np.tile([0.0, 1.0, 0.0], (4, 1)) + 0.01)
    emb = np.vstack([a, b])
    mask, _ = find_text_outliers(emb, eps=0.15, min_samples=3)
    assert mask.sum() == 0


def test_too_few_rows_returns_all_false():
    emb = _norm(np.random.RandomState(0).rand(2, 3))
    mask, fit = find_text_outliers(emb, eps=0.15, min_samples=3)
    assert mask.tolist() == [False, False]
```

- [ ] **Step 2: Run → FAIL**

Run: `python -m pytest tests/test_konto_text_outlier.py -v`
Expected: FAIL (`ModuleNotFoundError` / `ImportError: cannot import name 'find_text_outliers'`)

- [ ] **Step 3: Implement pure Funktion**

```python
# src/tests/konto_text_outlier.py
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
```

- [ ] **Step 4: Run → PASS**

Run: `python -m pytest tests/test_konto_text_outlier.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/tests/konto_text_outlier.py tests/test_konto_text_outlier.py
git commit -m "feat: find_text_outliers (per-Konto DBSCAN-Cluster-Outlier, pure)"
```

### Task 5: Config-Felder

**Files:**
- Modify: `src/config.py` (nach dem `text_konto_*`-Block)
- Test: `tests/test_config.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_config.py — ans Ende anhängen
def test_konto_text_outlier_defaults():
    from src.config import AnalysisConfig
    c = AnalysisConfig()
    assert c.konto_text_outlier_min_bookings == 8
    assert c.konto_text_outlier_eps == 0.20
    assert c.konto_text_outlier_min_samples == 3
```

- [ ] **Step 2: Run → FAIL**

Run: `python -m pytest tests/test_config.py::test_konto_text_outlier_defaults -v`
Expected: FAIL (`AttributeError`)

- [ ] **Step 3: Implement**

```python
# src/config.py — direkt nach text_konto_gt_path-Feld
    # ── Konto-Text-Outlier (KONTO_TEXT_OUTLIER) ──────────────────────────────
    konto_text_outlier_min_bookings: int = Field(
        8, ge=3,
        description="Mindest-Buchungen pro Konto für ein stabiles Textprofil (Standard: 8).",
    )
    konto_text_outlier_eps: float = Field(
        0.20, ge=0.01, le=1.0,
        description="DBSCAN epsilon (cosine-Distanz) für die Konto-Textcluster (Standard: 0.20).",
    )
    konto_text_outlier_min_samples: int = Field(
        3, ge=2,
        description="DBSCAN min_samples — Mindestgröße eines dichten Textclusters (Standard: 3).",
    )
```

- [ ] **Step 4: Run → PASS**

Run: `python -m pytest tests/test_config.py::test_konto_text_outlier_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: config-Felder für KONTO_TEXT_OUTLIER"
```

### Task 6: KontoTextOutlier-Test-Klasse

**Files:**
- Modify: `src/tests/konto_text_outlier.py` (Klasse + `get_tests`)
- Test: `tests/test_konto_text_outlier.py` (Engine-Integration, skipif HAS_EMBEDDINGS)

- [ ] **Step 1: Failing test (Integration, nur mit Embeddings)**

```python
# tests/test_konto_text_outlier.py — ans Ende anhängen
from src.embeddings import HAS_EMBEDDINGS
from src.config import AnalysisConfig
from src.tests.base import EngineStats
from src.tests.konto_text_outlier import KontoTextOutlier
import pandas as pd


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
```

- [ ] **Step 2: Run → FAIL (oder SKIP ohne Modell)**

Run: `python -m pytest tests/test_konto_text_outlier.py::test_outlier_booking_flagged_within_account -v`
Expected: FAIL (`ImportError: cannot import name 'KontoTextOutlier'`) — oder SKIP, falls keine Embeddings

- [ ] **Step 3: Implement Klasse**

```python
# src/tests/konto_text_outlier.py — ans Ende anhängen
import pandas as pd

from src.embeddings import HAS_EMBEDDINGS, get_embedder
from src.embedding_store import embed_cached


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
```

- [ ] **Step 4: Run → PASS** (auf Maschine mit sentence-transformers)

Run: `python -m pytest tests/test_konto_text_outlier.py -v`
Expected: PASS (pure Tests + Integration; Integration SKIP ohne Modell)

- [ ] **Step 5: Commit**

```bash
git add src/tests/konto_text_outlier.py tests/test_konto_text_outlier.py
git commit -m "feat: KONTO_TEXT_OUTLIER Test-Klasse"
```

### Task 7: Test in Engine + Validator registrieren

**Files:**
- Modify: `src/engine.py:36-51`, `src/validator.py:19-61`
- Test: `tests/test_engine.py` (`test_all_test_methods_called` deckt Registrierung ab)

- [ ] **Step 1: Engine-Registrierung**

```python
# src/engine.py — Import bei den anderen get_tests
from src.tests.konto_text_outlier import get_tests as get_konto_text_outlier_tests
# ... _ALL_TESTS erweitern
_ALL_TESTS = (
    get_betrag_tests()
    + get_duplikate_tests()
    + get_buchungslogik_tests()
    + get_kreditor_tests()
    + get_zeitreihe_tests()
    + get_isolation_tests()
    + get_text_match_tests()
    + get_konto_text_outlier_tests()
)
```

- [ ] **Step 2: Validator-Registrierung**

```python
# src/validator.py — in TEST_REQUIREMENTS nach TEXT_KONTO_MATCH
    "KONTO_TEXT_OUTLIER":    {"required": ["buchungstext", "konto_soll"]},
# ... in TEST_CATEGORIES "Embedding-Tests" erweitern
    "Embedding-Tests": ["TEXT_KONTO_MATCH", "KONTO_TEXT_OUTLIER"],
```

- [ ] **Step 3: Run → PASS**

Run: `python -m pytest tests/test_engine.py -q`
Expected: PASS (inkl. `test_all_test_methods_called`, das nun KONTO_TEXT_OUTLIER erwartet)

- [ ] **Step 4: ruff**

Run: `ruff check src/ app.py tests/`
Expected: All checks passed!

- [ ] **Step 5: Commit**

```bash
git add src/engine.py src/validator.py
git commit -m "feat: KONTO_TEXT_OUTLIER in Engine + Validator registrieren"
```

---

## Phase 3 — UI-Wiring (kein totes Feature)

### Task 8: Slider + config_dict + click-inputs + #22-Persistenz

**Files:**
- Modify: `app.py` (Slider-Block ~851, `analyze_file`-Signatur ~243, `config_dict` ~306, save_settings-Block, `_SAVED_SETTINGS`-Defaults, `analyze_btn.click` inputs ~1076)
- Test: manuelle UI-Prüfung + `tests/test_user_settings.py` (Persistenz-Keys)

- [ ] **Step 1: Slider in „Erweiterte Einstellungen" ergänzen**

```python
# app.py — im Accordion "Erweiterte Einstellungen", neue Row nach text_konto_slider
        with gr.Row():
            kto_outlier_eps_slider = gr.Slider(
                minimum=0.05, maximum=0.95,
                value=_SAVED_SETTINGS.get("konto_text_outlier_eps", 0.20), step=0.05,
                label="KONTO_TEXT_OUTLIER eps (DBSCAN cosine-Distanz)",
                info="Kleiner = engere Textcluster pro Konto (Standard: 0.20)",
            )
            kto_outlier_min_slider = gr.Slider(
                minimum=3, maximum=50,
                value=_SAVED_SETTINGS.get("konto_text_outlier_min_bookings", 8), step=1,
                label="KONTO_TEXT_OUTLIER min. Buchungen/Konto",
                info="Darunter wird ein Konto übersprungen (Standard: 8)",
            )
```

- [ ] **Step 2: analyze_file-Signatur + config_dict + save_settings**

```python
# app.py — analyze_file(...) Signatur: nach konto_filter_max ergänzen
    kto_outlier_eps: float,
    kto_outlier_min: int,
# ... config_dict ergänzen
        "konto_text_outlier_eps":          float(kto_outlier_eps),
        "konto_text_outlier_min_bookings": int(kto_outlier_min),
# ... save_settings({...}) ergänzen
            "konto_text_outlier_eps":          float(kto_outlier_eps),
            "konto_text_outlier_min_bookings": int(kto_outlier_min),
```

- [ ] **Step 3: click-inputs erweitern**

```python
# app.py — analyze_btn.click inputs=[...] nach konto_filter_max einfügen
            kto_outlier_eps_slider, kto_outlier_min_slider,
```

(Reihenfolge muss exakt zur `analyze_file`-Signatur passen — die beiden neuen Slider VOR `*test_checkboxes` einfügen.)

- [ ] **Step 4: Smoke — App importiert, Slider-Defaults laden**

Run:
```bash
python -c "import os,json,tempfile; d=tempfile.mkdtemp(); p=d+'/s.json'; \
json.dump({'konto_text_outlier_eps':0.3,'konto_text_outlier_min_bookings':12}, open(p,'w')); \
os.environ['USER_SETTINGS_FILE']=p; import app; \
print(app.kto_outlier_eps_slider.value, app.kto_outlier_min_slider.value)"
```
Expected: `0.3 12`

- [ ] **Step 5: ruff + Commit**

```bash
ruff check app.py
git add app.py
git commit -m "feat: KONTO_TEXT_OUTLIER in UI (Slider + Persistenz + config)"
```

### Task 9: End-to-End-Verifikation + PR

- [ ] **Step 1: Volle Suite + ruff**

Run: `EMBEDDING_CACHE_DB=/tmp/e2e_emb.db python -m pytest tests/ -q && ruff check src/ app.py tests/`
Expected: grün

- [ ] **Step 2: Cache-Wirkung belegen (manuell)**

Mit einem großen Sample zweimal `analyze_file` laufen lassen; im Log Run 2 zeigt `Embedding-Cache hits=… misses=0` für unveränderte Texte.

- [ ] **Step 3: PR**

```bash
git push -u origin feature/konto-text-outlier
gh pr create --title "feat: KONTO_TEXT_OUTLIER + persistenter Embedding-Cache" --body "siehe docs/superpowers/plans/2026-05-29-konto-text-outlier-embedding-store.md"
```

---

## Phase 4 (deferred, NICHT jetzt) — Echte Vektor-DB / persistente Konto-Profile

Nur falls später gebraucht: ANN-Vektor-DB (Chroma) statt SQLite-Key-Cache, um (a) kontoübergreifende Ähnlichkeitssuche und (b) **persistente Konto-Centroids über Läufe** zu ermöglichen (Ausreißer-Erkennung auch bei wenigen Buchungen pro Datei). `embed_cached`/`find_text_outliers` bleiben die Schnittstelle; nur der Store wird ausgetauscht. Bewusst ausgeklammert (YAGNI), bis der Centroid-Ansatz sich bewährt hat.

---

## Self-Review

- **Spec-Abdeckung:** Embedding-Cache (Phase 1, Task 2-3) ✓; per-Konto-Centroid/Cluster-Outlier (Phase 2, Task 4-7) ✓; Cluster-Set statt Einzel-Centroid via DBSCAN ✓; UI/Engine-Wiring gegen „totes Feature" (Phase 3) ✓; Vektor-DB-Wunsch adressiert (SQLite jetzt, ANN als Phase 4 mit Begründung) ✓; LLM-GT als unnötig begründet ✓.
- **Platzhalter-Scan:** keine TBD/TODO; jeder Code-Step enthält vollständigen Code.
- **Typ-Konsistenz:** `embed_cached(embedder, texts)->np.ndarray` und `find_text_outliers(emb, eps, min_samples)->(mask, fit)` werden in Task 6 exakt so aufgerufen; Config-Feldnamen (`konto_text_outlier_eps/_min_bookings/_min_samples`) identisch in config.py, Test-Klasse, app.py.
- **Offene Bestätigung:** Backend = SQLite-Key-Cache (kein ANN). Falls explizit Chroma gewünscht → Phase 4 vorziehen.
