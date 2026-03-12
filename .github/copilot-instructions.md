# Prefilter-API — Copilot Instructions

## Projektübersicht

Buchungs-Anomalie Pre-Filter v4.1: Gradio-Web-App die CSV/XLS/XLSX-Buchungsdaten durch 14 statistische Anomalie-Tests laufen lässt und verdächtige Buchungen an einen Langdock Agent weiterleitet. Betrieb ausschließlich via Docker Compose.

## Architektur

```
                   ┌─────────┐
User ──► Gradio UI │  app.py │──► Redis ──► Celery Worker(s)
          :7864    └─────────┘              src/worker.py
                                            └─► AnomalyEngine
                                                src/engine.py
                   ┌──────────────┐
     (alternativ)  │ FastAPI REST │──► Redis ──► gleicher Worker
                   │ src/main.py  │
                   │    :8000     │
                   └──────────────┘
```

- **app.py**: Gradio UI, dispatcht Celery-Tasks direkt via Redis. Lokaler Fallback wenn Redis nicht erreichbar (`_LOCAL_MODE`).
- **src/main.py**: FastAPI REST API + WebSocket (`/api/jobs`, `/ws/jobs/{id}`, `/health`). Zweiter Einstiegspunkt, gleicher Worker.
- **src/worker.py**: Celery-Task `prefilter.analyze`. Liest Datei, baut `AnomalyEngine`, veröffentlicht Progress via Redis Pub/Sub.
- **src/engine.py**: Orchestriert 14 Tests aus `src/tests/`. Kein `iterrows()`, alles vektorisiert.
- **src/parser.py**: CSV/XLS-Parsing, Spalten-Mapping (COLUMN_ALIASES), Deutsche Zahlen/Datumsformate.
- **src/config.py**: `AnalysisConfig` (Pydantic) — alle Schwellenwerte konfigurierbar.
- **src/webhook.py**: `push_to_langdock()` mit 3 Retries via httpx.
- **docker-compose.yml**: 3 Services (ui, worker, redis), shared `uploads` Volume.

## Wichtige Konventionen

### Sprache
- Domain-Begriffe sind **Deutsch** (Buchung, Kreditor, Betrag, Belegnummer, etc.)
- Code-Kommentare und Variablennamen mischen Deutsch und Englisch
- README/Docstrings: Deutsch

### Test-Plugin-System
Jeder Anomalie-Test ist eine `AnomalyTest`-Subklasse in `src/tests/`:
```python
class MeinTest(AnomalyTest):
    name = "MEIN_TEST"
    weight = 1.5
    critical = False
    def run(self, df, stats, config) -> int:
        mask = ...  # Boolean-Series
        return self._flag(df, mask)
```
Module exportieren `get_tests() -> list[AnomalyTest]`. Engine sammelt alle via `_ALL_TESTS`.

### Boolean-Flag-Spalten
Jeder Test setzt `df[f"flag_{name}"] = True/False` in-place. Keine Listen-in-Zellen.

### Scoring
`anomaly_score` = gewichtete Summe aller Flags. Output: Score ≥ `output_threshold` (default 2.0) ODER kritisches Flag.

## 14 Tests (src/tests/)

| Modul | Tests | Kritisch |
|---|---|---|
| betrag.py | BETRAG_ZSCORE (2.0), BETRAG_IQR (1.5), KONTO_BETRAG_ANOMALIE (2.0) | ✓, -, ✓ |
| duplikate.py | NEAR_DUPLICATE (2.0), DOPPELTE_BELEGNUMMER (2.0), BELEG_KREDITOR_DUPLIKAT (2.5) | ✓, ✓, ✓ |
| buchungslogik.py | STORNO (1.5), LEERER_BUCHUNGSTEXT (1.0), RECHNUNGSDATUM_PERIODE (1.5), BUCHUNGSTEXT_PERIODE (1.0) | ✓, -, -, - |
| kreditor.py | NEUER_KREDITOR_HOCH (2.5), VELOCITY_ANOMALIE (1.5) | ✓, - |
| zeitreihe.py | MONATS_ENTWICKLUNG (1.5), FEHLENDE_MONATSBUCHUNG (1.0) | -, - |

## AnalysisConfig (src/config.py)

Alle Schwellenwerte sind über `AnalysisConfig` steuerbar (Pydantic mit `ge`/`le`-Validierung). Wichtigste:
- `zscore_threshold` (2.5), `iqr_factor` (1.5), `near_duplicate_days` (3)
- `output_threshold` (2.0), `max_output_rows` (1000)
- Slider in der Gradio-UI steuern `zscore_threshold`, `iqr_factor`, `near_duplicate_days`, `output_threshold`

## Dependency Management

- **requirements.txt**: Loose Pins (minimale Versionsanforderungen)
- **requirements.lock**: `pip freeze` aus `python:3.12-slim` Container — exakte Versionen für reproduzierbare Builds
- **Dockerfile**: Installiert aus `requirements.lock` wenn vorhanden, Fallback `requirements.txt`
- **Lock aktualisieren** nach Dep-Änderungen:
  ```bash
  docker run --rm -v "$PWD/requirements.txt:/app/requirements.txt" \
    python:3.12-slim sh -c \
    "pip install --no-cache-dir -r /app/requirements.txt >/dev/null 2>&1 && pip freeze" \
    > requirements.lock
  ```
- **pyproject.toml**: Muss synchron mit requirements.txt gehalten werden

## Gradio 6.x

- `css=` und `theme=` gehören in `demo.launch()`, NICHT in `gr.Blocks()`
- Alle anderen APIs (File, Slider, Button, Dataframe, Progress, Tabs) sind kompatibel

## Docker Compose

```bash
docker compose up -d --build                  # Standard (2 Worker)
docker compose up -d --build --scale worker=4 # Mehr Worker
```

- `WORKER_REPLICAS` in `deploy.replicas` wird von non-Swarm `docker compose` ignoriert — `--scale` nutzen
- `WORKER_CONCURRENCY` (default 8): Celery-Prozesse pro Worker-Container
- Shared Volume `uploads:/data/uploads` zwischen UI und Worker

## Redis-Keys (Konvention)

| Key | Typ | Inhalt |
|---|---|---|
| `job:{id}` | Hash | status, progress_pct, current_test, result (JSON), error, filename |
| `job:{id}:cancelled` | String | "1" wenn Abbruch angefordert |
| `job:{id}:logs` | Pub/Sub | JSON-Events für WebSocket-Streaming |

## Tests

```bash
pytest tests/ -v                    # 51 Unit-Tests (ohne slow/integration)
pytest tests/ -v -m slow            # Performance-Benchmark (500k Zeilen, <90s)
pytest tests/ -v -m integration     # E2E mit Redis (docker compose up redis)
```

- `pyproject.toml`: `addopts = "-m 'not slow and not integration'"`
- Tests importieren von `src.*`, niemals von `modules.*` (gelöscht)

## Bekannte offene Punkte

1. **Single-Job-Parallelisierung** (HOCH): 500k-Zeilen-Jobs laufen sequentiell auf 1 Worker. Geplant: Celery chord (prepare → 14 Tests parallel → merge). Details in `prefilter_todo_v5.txt`.
2. **FEHLENDE_MONATSBUCHUNG**: Logik in `src/tests/zeitreihe.py` verifizieren — unklar ob `all_periods` via `pd.period_range()` oder nur aus vorhandenen Buchungen erzeugt wird.
3. **deploy.replicas** in docker-compose.yml ist irreführend (non-Swarm ignoriert es).

## Häufige Fehlerquellen

- **HfFolder ImportError**: Gradio < 5.12.0 + huggingface_hub ≥ 1.0.0. Fix: Gradio ≥ 5.12.0.
- **pydantic-core Build-Fehler**: Python 3.13 + pydantic < 2.10. Fix: `pydantic>=2.7.4` (nicht hart pinnen) oder Python 3.12 nutzen.
- **conda-env pollutes pip freeze**: `pip freeze` im conda-Env enthält `@ file://` Pfade. Lock-File IMMER aus Docker generieren.
- **Performance-Test flaky**: 500k-Benchmark ist hardwareabhängig (~68s auf Dev-Laptop). Limit ist 90s.
