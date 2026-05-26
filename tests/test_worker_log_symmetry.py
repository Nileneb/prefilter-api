"""Sequential- vs. Parallel-Pfad: identische _export()-Summary im Redis-Log (#17.1, #17.2, #17.5).

Nutzt einen minimalen In-Memory-Fake-Redis (kein fakeredis-Dependency nötig).
"""

import json
import time

import pandas as pd
import pytest

import src.history as history
import src.worker as worker
from src.engine import _ALL_TESTS, AnomalyEngine


class FakeRedis:
    def __init__(self):
        self.lists: dict[str, list] = {}
        self.hashes: dict[str, dict] = {}
        self.kv: dict[str, str] = {}

    def rpush(self, key, val):
        self.lists.setdefault(key, []).append(val)

    def lrange(self, key, a, b):
        items = self.lists.get(key, [])
        return items[a:] if b == -1 else items[a:b + 1]

    def hset(self, key, field=None, value=None, mapping=None):
        h = self.hashes.setdefault(key, {})
        if mapping:
            h.update({k: str(v) for k, v in mapping.items()})
        elif field is not None:
            h[field] = str(value)

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def get(self, key):
        return self.kv.get(key)

    def set(self, key, val, ex=None):
        self.kv[key] = val

    def exists(self, key):
        return 1 if key in self.kv else 0

    def expire(self, key, ttl):
        pass

    def publish(self, ch, msg):
        pass

    def close(self):
        pass


def _df():
    return pd.DataFrame({
        "datum": ["2024-01-15"] * 12,
        "betrag": [str(i * 100) for i in range(1, 13)],
        "konto_soll": ["4711"] * 12,
        "buchungstext": [f"Buchung {i}" for i in range(12)],
        "belegnummer": [f"{i:04d}" for i in range(12)],
        "kreditor": ["Lieferant A"] * 12,
    })


def _log_lines(fake: FakeRedis, job_id: str) -> list[str]:
    return fake.lists.get(f"job:{job_id}:log", [])


def _count_ergebnis(lines: list[str]) -> int:
    return sum(1 for ln in lines if "ERGEBNIS:" in ln)


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path / "history")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    return tmp_path


def test_sequential_emits_ergebnis_once(isolated_dirs):
    fake = FakeRedis()
    worker._run_sequential(fake, "seq", _df(), {}, time.time(), None)
    lines = _log_lines(fake, "seq")
    assert _count_ergebnis(lines) == 1, lines
    result = json.loads(fake.hashes["job:seq"]["result"])
    assert "stammdaten_report" in result


def test_merge_emits_ergebnis_once_and_matches_sequential(isolated_dirs, tmp_path, monkeypatch):
    df = _df()

    # Sequentielles Referenz-Ergebnis
    seq_fake = FakeRedis()
    worker._run_sequential(seq_fake, "seq", df.copy(), {}, time.time(), None)
    seq_result = json.loads(seq_fake.hashes["job:seq"]["result"])

    # Parallel: prepared parquet + test_results aus einem Engine-Lauf
    prep_engine = AnomalyEngine(df.copy())
    parquet_path = str(tmp_path / "job_par_prepared.parquet")
    prep_engine.df.to_parquet(parquet_path, index=True)

    run_engine = AnomalyEngine(df.copy())
    run_engine.run()
    test_results = [
        {
            "test_name": t.name,
            "flagged": run_engine.df.index[run_engine.df[f"flag_{t.name}"]].tolist(),
            "count": int(run_engine.df[f"flag_{t.name}"].sum()),
        }
        for t in _ALL_TESTS
    ]

    merge_fake = FakeRedis()
    monkeypatch.setattr(worker, "_redis_client", lambda: merge_fake)
    prepare_result = {"parquet_path": parquet_path, "config_dict": {}, "job_id": "par"}
    worker.merge_task.apply(args=[test_results, prepare_result]).get()

    lines = _log_lines(merge_fake, "par")
    assert _count_ergebnis(lines) == 1, lines

    merge_result = json.loads(merge_fake.hashes["job:par"]["result"])
    # Finding #5: stammdaten_report-Struktur stimmt zwischen beiden Pfaden überein
    assert merge_result["stammdaten_report"] == seq_result["stammdaten_report"]
    # Flag-Counts identisch
    assert merge_result["statistics"]["flag_counts"] == seq_result["statistics"]["flag_counts"]
