# Prefilter-API — Copilot Instructions

> **Stand: 2026-03-19 · Version 5.1**
> Dieses Dokument ist die einzige Wahrheitsquelle für Architektur, Konventionen und Domain-Logik.
PYTHON ENV VERWENDEN!!!!!!!!!!!!!!!!!!!!!!!!!!!!! AUCH FÜR TESTS!!!!!!!!!!!!!!!!!!!!!!!!!!!
---

## Projektübersicht

Buchungs-Anomalie Pre-Filter v5.1: Gradio-Web-App die CSV/XLS/XLSX-Buchungsdaten (inkl. Diamant-Export mit Pipe-Delimiter) durch **13 statistische Anomalie-Tests** laufen lässt und verdächtige Buchungen an einen Langdock Agent weiterleitet. Betrieb ausschließlich via Docker Compose.

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
- **src/engine.py**: Orchestriert 13 Tests aus `src/tests/`. Kein `iterrows()`, alles vektorisiert.
- **src/accounting.py**: **⚠️ ZENTRAL** — Einzige Stelle für Kontoklassen-Grenzen (`kontoklasse()`) und Vorzeichen-Berechnung (`compute_signed_betrag()`). Alle anderen Dateien importieren von hier.
- **src/parser.py**: CSV/XLS-Parsing mit Pipe-Delimiter-Support, Spalten-Mapping (COLUMN_ALIASES inkl. Diamant-Mappings), Deutsche Zahlen/Datumsformate, SQL-Timestamps, NULL-String-Handling.
- **src/config.py**: `AnalysisConfig` (Pydantic) — alle Schwellenwerte konfigurierbar.
- **src/validator.py**: Spalten-Validierung, Test-Blocking, `soll_haben`-Warnungen.
- **src/charts.py**: Plotly-Visualisierungen. Nutzt `_betrag_signed` für PnL-Charts.
- **src/webhook.py**: `push_to_langdock()` mit 3 Retries via httpx.
- **src/logging_config.py**: Zentrales structlog-Setup (JSON/Console via `LOG_FORMAT` ENV).
- **docker-compose.yml**: 3 Services (ui, worker, redis), shared `uploads` Volume.
- **.github/workflows/ci.yml**: CI Pipeline — Unit-Tests + Docker build + smoke test.

---

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
    required_columns = ["_abs", "_datum"]
    def run(self, df, stats, config) -> int:
        mask = ...  # Boolean-Series
        return self._flag(df, mask)
```
Module exportieren `get_tests() -> list[AnomalyTest]`. Engine sammelt alle via `_ALL_TESTS`.

### Boolean-Flag-Spalten
Jeder Test setzt `df[f"flag_{name}"] = True/False` in-place. Keine Listen-in-Zellen.

### Scoring
`anomaly_score` = gewichtete Summe aller Flags. Output: Score >= `output_threshold` (default 2.0) ODER kritisches Flag.

---

## ⚠️ KONTOKLASSEN — ZENTRALE DEFINITION (src/accounting.py)

**Es gibt EINE einzige Implementierung.** Alle anderen Dateien importieren von dort.

| Kontoklasse | Bereich | Beispiel |
|---|---|---|
| **Ertrag** | 40000–59999 | 42000, 50200 |
| **Aufwand** | 60000–79999 | 65000, 70100 |
| **Bestand** | 0–39999 | 1200 (Bank), 10000 |

```python
# src/accounting.py — EINZIGE Definition
ERTRAG_MIN, ERTRAG_MAX = 40000, 59999
AUFWAND_MIN, AUFWAND_MAX = 60000, 79999

def kontoklasse(konto_soll: pd.Series) -> pd.Series:
    \"\"\"Ertrag, Aufwand oder Bestand.\"\"\"
    ...
```

**NIEMALS** Kontoklassen-Grenzen inline in Tests, Charts oder Engine definieren.
Immer `from src.accounting import kontoklasse` verwenden.

---

## ⚠️ SOLL/HABEN-VORZEICHEN — ZENTRALE DEFINITION (src/accounting.py)

Die Spalte `soll_haben` (Diamant-Spalte L, Werte "S"/"H") bestimmt das betriebswirtschaftlich korrekte Vorzeichen:

| Kontoklasse | Soll-Buchung | Haben-Buchung |
|---|---|---|
| **Ertrag** (40000–59999) | − Erlösschmälerung | + normaler Ertrag |
| **Aufwand** (60000–79999) | + normaler Aufwand | − Aufwandsminderung |
| **Bestand** (0–39999) | Originalvorzeichen | Originalvorzeichen |

```python
# src/accounting.py — compute_signed_betrag()
def compute_signed_betrag(df: pd.DataFrame) -> pd.Series:
    \"\"\"Berechnet _betrag_signed aus _abs, konto_soll, soll_haben.
    Fallback wenn soll_haben fehlt: Originalvorzeichen aus _betrag.
    \"\"\"
    ...
```

### Abgeleitete Spalten in `engine._prepare()`

| Spalte | Typ | Berechnung | Wofür |
|---|---|---|---|
| `_betrag` | float32 | `parse_german_number_series(betrag)` | Rohwert mit Originalvorzeichen |
| `_abs` | float32 | `_betrag.abs()` | Alle statistischen Tests (Z-Score, IQR, etc.) |
| `_datum` | datetime64 | `parse_date_series(datum)` | Zeitreihen, Duplikat-Zeitfenster |
| `_kontoklasse` | str | `kontoklasse(konto_soll)` | Getrennte Statistiken je Klasse |
| `_betrag_signed` | float32 | `compute_signed_betrag(df)` | PnL-Charts, Saldierung |
| `_score` | float64 | Summe Flag-Gewichte | Anomalie-Ranking |

### Reihenfolge in `_prepare()` — MUSS eingehalten werden:
1. `map_columns` (Parser) → kanonische Spaltennamen
2. Fehlende Spalten mit `""` initialisieren
3. `_betrag` + `_abs` berechnen (Zahlen-Parsing)
4. `_datum` berechnen (Datums-Parsing)
5. `_kontoklasse` berechnen (aus `konto_soll`)
6. `_betrag_signed` berechnen (aus `_abs` + `_kontoklasse` + `soll_haben`)
7. Kategorische Spalten setzen
8. Flag-Spalten initialisieren

**Erst danach** `compute_stats()` und Tests ausführen.

---

## Diamant-Export — Spalten-Mapping

### Alle Spalten (A–Q)

| Diamant-Spalte | Kanonischer Name | Status |
|---|---|---|
| A: Belegdatum | `datum` | ✅ OK |
| B: FiBuBetrag | `betrag` | ✅ OK |
| C: Kontonummer | `konto_soll` | ✅ OK |
| D: Buchungstext | `buchungstext` | ✅ OK |
| E: Belegnummer | `belegnummer` | ✅ OK |
| F: Bezeichnung | `kreditor` | ✅ OK |
| G: Klasse | `klasse` (K/S/D) | ✅ OK |
| H: Kostenstelle | `kostenstelle` | ✅ OK |
| I: Buchungsperiode | `buchungsperiode` | ✅ OK |
| J: ErfassungAm | `erfassungsdatum` | ✅ OK |
| K: Generalumgekehrt | `generalumgekehrt` | ✅ OK |
| **L: Soll/Haben** | **`soll_haben`** | **✅ NEU** |
| M: Detailbetrag | `detailbetrag` | ℹ️ inaktiv |
| N: Belegart | `belegart` | ℹ️ inaktiv |
| O: Kostenträger | `kostentraeger` | ℹ️ inaktiv |
| P: Projekt | `projekt` | ℹ️ inaktiv |
| Q: Dokumenten-Link | `dokumentenlink` | ℹ️ inaktiv |

### Parser COLUMN_ALIASES

```python
COLUMN_ALIASES = {
    "datum":             ["datum", "date", "buchungsdatum", "belegdatum"],
    "betrag":            ["betrag", "amount", "summe", "wert", "fibubetrag"],
    "konto_soll":        ["konto_soll", "kontosoll", "soll", "sollkonto", "debit", "kontonummer"],
    "konto_haben":       ["konto_haben", "kontohaben", "haben", "habenkonto", "credit"],
    "buchungstext":      ["buchungstext", "text", "beschreibung", "verwendungszweck"],
    "belegnummer":       ["belegnummer", "beleg", "belegnr", "beleg_nr", "voucher"],
    "kostenstelle":      ["kostenstelle", "kst", "cost_center"],
    "kreditor":          ["kreditor", "lieferant", "vendor", "supplier", "creditor", "bezeichnung"],
    "soll_haben":        ["soll_haben", "sollhaben", "s_h", "sh", "soll/haben"],
    "klasse":            ["klasse", "class", "kontoklasse"],
    "generalumgekehrt":  ["generalumgekehrt", "storno_kz", "umkehr"],
    "buchungsperiode":   ["buchungsperiode", "periode", "period"],
    "erfassungsdatum":   ["erfassungsdatum", "erfassungam", "erfassung_am", "created_at"],
    "detailbetrag":      ["detailbetrag", "detail_betrag"],
    # ENTFERNT: "erfasser" — nicht im Diamant-Export vorhanden
}
```

> **Wichtig:** `erfasser` ist NICHT mehr in COLUMN_ALIASES. Diamant liefert keinen Erfasser.

---

## 13 Tests (src/tests/)

> **VELOCITY_ANOMALIE wurde entfernt** — Diamant liefert keinen `erfasser`.

| # | Modul | Test | Gewicht | Kritisch | Benötigte Spalten |
|---|---|---|---|---|---|
| 01 | betrag.py | `BETRAG_ZSCORE` | 2.0 | ✓ | `_abs`, `konto_soll` |
| 02 | betrag.py | `BETRAG_IQR` | 1.5 | — | `_abs`, `konto_soll` |
| 03 | betrag.py | `KONTO_BETRAG_ANOMALIE` | 2.0 | ✓ | `_abs`, `konto_soll` |
| 04 | duplikate.py | `NEAR_DUPLICATE` | 2.0 | ✓ | `_abs`, `_datum`, `konto_soll`, `kreditor` |
| 05 | duplikate.py | `DOPPELTE_BELEGNUMMER` | 2.0 | ✓ | `belegnummer`, `konto_soll`, `_betrag` |
| 06 | duplikate.py | `BELEG_KREDITOR_DUPLIKAT` | 2.5 | ✓ | `belegnummer`, `kreditor`, `_betrag`, `_datum` |
| 07 | buchungslogik.py | `STORNO` | 1.5 | ✓ | `buchungstext`, `_betrag`, `generalumgekehrt` |
| 08 | buchungslogik.py | `LEERER_BUCHUNGSTEXT` | 1.0 | — | `buchungstext` |
| 09 | buchungslogik.py | `RECHNUNGSDATUM_PERIODE` | 1.5 | — | `_datum`, `erfassungsdatum`/`buchungsperiode` |
| 10 | buchungslogik.py | `BUCHUNGSTEXT_PERIODE` | 1.0 | — | `buchungstext`, `_datum` |
| 11 | kreditor.py | `NEUER_KREDITOR_HOCH` | 2.5 | ✓ | `kreditor`, `_abs`, `_datum` |
| 12 | zeitreihe.py | `MONATS_ENTWICKLUNG` | 1.5 | — | `_abs`, `_datum`, `konto_soll` |
| 13 | zeitreihe.py | `FEHLENDE_MONATSBUCHUNG` | 1.0 | — | `_datum`, `konto_soll` |

### Test-Registrierung (src/validator.py)

```python
TEST_REQUIREMENTS: dict[str, dict[str, list[str]]] = {
    "BETRAG_ZSCORE":          {"required": ["betrag"],   "optional": ["konto_soll"]},
    "BETRAG_IQR":             {"required": ["betrag"],   "optional": ["konto_soll"]},
    "KONTO_BETRAG_ANOMALIE":  {"required": ["betrag"],   "optional": ["konto_soll"]},
    "NEAR_DUPLICATE":         {"required": ["kreditor", "betrag"],
                               "optional": ["buchungstext", "konto_soll", "datum"]},
    "DOPPELTE_BELEGNUMMER":   {"required": ["belegnummer"]},
    "BELEG_KREDITOR_DUPLIKAT": {"required": ["belegnummer", "kreditor", "betrag"]},
    "STORNO":                 {"required": ["betrag"],   "optional": ["buchungstext", "generalumgekehrt"]},
    "NEUER_KREDITOR_HOCH":    {"required": ["kreditor", "betrag", "datum"]},
    "LEERER_BUCHUNGSTEXT":    {"required": ["buchungstext"]},
    "RECHNUNGSDATUM_PERIODE": {"required": ["datum"],
                               "optional": ["erfassungsdatum", "buchungsperiode"]},
    "BUCHUNGSTEXT_PERIODE":   {"required": ["buchungstext", "datum"]},
    "MONATS_ENTWICKLUNG":     {"required": ["betrag", "datum"]},
    "FEHLENDE_MONATSBUCHUNG": {"required": ["datum"],     "optional": ["konto_soll"]},
}

ALL_TEST_NAMES: list[str] = list(TEST_REQUIREMENTS.keys())

TEST_CATEGORIES: dict[str, list[str]] = {
    "Betrags-Tests": ["BETRAG_ZSCORE", "BETRAG_IQR", "KONTO_BETRAG_ANOMALIE"],
    "Duplikat-Tests": ["NEAR_DUPLICATE", "DOPPELTE_BELEGNUMMER", "BELEG_KREDITOR_DUPLIKAT"],
    "Buchungslogik": ["STORNO", "LEERER_BUCHUNGSTEXT", "RECHNUNGSDATUM_PERIODE", "BUCHUNGSTEXT_PERIODE"],
    "Kreditor-Tests": ["NEUER_KREDITOR_HOCH"],
    "Zeitreihen-Tests": ["MONATS_ENTWICKLUNG", "FEHLENDE_MONATSBUCHUNG"],
}
```

> **`VELOCITY_ANOMALIE` ist vollständig entfernt** — aus TEST_REQUIREMENTS, TEST_CATEGORIES, ALL_TEST_NAMES, config.py, kreditor.py und allen Unit-Tests.

---

## Betrags-Tests — Kontoklassen-Differenzierung

BETRAG_ZSCORE, BETRAG_IQR und KONTO_BETRAG_ANOMALIE rechnen **getrennt** nach Kontoklasse:
- **Ertrag** (40000–59999): eigene Verteilung
- **Aufwand** (60000–79999): eigene Verteilung
- **Bestand** (0–39999): eigene Verteilung

Alle drei Tests nutzen `_abs` (Absolutbetrag) für die Ausreißer-Erkennung.
Die Kontoklasse wird über `from src.accounting import kontoklasse` bestimmt.

---

## konto_haben — Handling-Policy

`konto_haben` ist im Diamant-Export **nicht vorhanden** (nur eine `Kontonummer`-Spalte -> `konto_soll`).

### Regeln:
- `konto_haben` wird NICHT als `required` in TEST_REQUIREMENTS geführt
- `DoppelteBelegnummer.required_columns` enthält NICHT `konto_haben`
- Soll/Haben-Paar-Erkennung (Gegenbuchungen) nutzt die `soll_haben`-Spalte statt `konto_haben`
- Wenn `soll_haben` verfügbar: Buchung mit S + Buchung mit H + gleicher Belegnr. = Paar -> nicht als Duplikat flaggen
- `engine._export()` gibt `konto_haben` weiterhin im Output aus (bleibt leer bei Diamant)

---

## Charts (src/charts.py)

### Korrekte Betragsverwendung:
- **Statistische Charts** (Score-Verteilung, Betrag vs. Score, Top-Konten): nutzen `_abs`
- **PnL-Charts** (monthly_pnl, ertrag_aufwand_monthly): nutzen `_betrag_signed`
- **Soll/Haben-Balance**: nutzt `_betrag_signed` + `_kontoklasse`
- **Kontoklassen**: immer `from src.accounting import kontoklasse` — NICHT inline definieren

### Fallback wenn `soll_haben` fehlt:
- PnL-Charts zeigen Absolutbeträge mit Warnhinweis-Annotation: *"⚠️ Soll/Haben fehlt — nur Absolutbeträge"*
- Soll/Haben-Balance zeigt leere Figure mit Hinweis

---

## AnalysisConfig (src/config.py)

Alle Schwellenwerte über `AnalysisConfig` steuerbar (Pydantic mit `ge`/`le`-Validierung). Wichtigste:
- `zscore_threshold` (2.5), `iqr_factor` (3.0), `iqr_min_betrag` (10000.0)
- `near_duplicate_days` (3), `near_duplicate_max_group_size` (10)
- `konto_betrag_sigma` (3.0), `konto_min_buchungen` (5)
- `new_kreditor_max_bookings` (2), `new_kreditor_amount_sigma` (1.5)
- `monats_entwicklung_zscore` (2.5), `monats_entwicklung_min_monate` (3)
- `fehlende_buchung_min_quote` (0.3)
- `output_threshold` (2.0), `max_output_rows` (1000)
- Slider in der Gradio-UI steuern `zscore_threshold`, `iqr_factor`, `near_duplicate_days`, `output_threshold`

> **Entfernt:** `velocity_min_months` — war für VELOCITY_ANOMALIE, Test ist gelöscht.

---

## Berechnungslogik (Kurzreferenz)

### 1. Z-Score (BETRAG_ZSCORE)
```
z = (Betrag - Durchschnitt) / Standardabweichung
z > 2,5 -> Ausreisser (Gewicht 2.0, kritisch)
```
Getrennt pro Kontoklasse berechnet.

### 2. IQR-Fence (BETRAG_IQR)
```
IQR = Q3 - Q1
Grenze = Q3 + 3.0 * IQR
Betrag > Grenze UND > 10.000 EUR -> Ausreisser (Gewicht 1.5)
```
Getrennt pro Kontoklasse.

### 3. Konto-Betrag-Anomalie (KONTO_BETRAG_ANOMALIE)
```
Pro Konto (mind. 5 Buchungen):
z = (Betrag - Konto-Durchschnitt) / Konto-Std
z > 3.0 -> Ausreisser (Gewicht 2.0, kritisch)
```

### 4. Near-Duplicate
```
Gleicher Betrag + Konto, Datum <= 3 Tage -> verdaechtig (Gewicht 2.0, kritisch)
Ausnahme: Gruppen > 10 oder regelmaessige Muster (>= 6 Monate)
```

### 5. Doppelte Belegnummer
```
Belegnr. + Konto + Betrag gleich, >= 5x -> verdaechtig (Gewicht 2.0, kritisch)
Soll/Haben-Paare werden ausgeschlossen
```

### 6. Beleg-Kreditor-Duplikat
```
Level 1: Belegnr. + Kreditor + Betrag gleich, >= 3x (Gewicht 2.5, kritisch)
Level 2: Kreditor + Betrag gleich, versch. Belegnr., Datum <= 7 Tage
Dauerschuldverhaeltnisse werden ausgeschlossen
```

### 7. Storno-Erkennung
```
Buchungstext enthaelt "Storno"/"Korrektur"/"Rueckbuchung" (case-insensitive)
ODER Generalumgekehrt = 1/true/J (Gewicht 1.5, kritisch)
```

### 8. Leerer Buchungstext
```
Text leer ODER <= 2 Zeichen ODER generisch ("diverse","xxx","test") (Gewicht 1.0)
```

### 9. Neuer Kreditor mit hohem Betrag
```
Kreditor <= 2 Buchungen UND Betrag > Durchschnitt + 1.5*Std (Gewicht 2.5, kritisch)
```

### 10. Rechnungsdatum-Periode
```
|Monat(Belegdatum) - Monat(ErfassungAm/Buchungsperiode)| > 2 -> verdaechtig (Gewicht 1.5)
```

### 11. Buchungstext-Periode
```
Periodenangabe im Text (z.B. "01/2024") != Monat(Belegdatum) (Gewicht 1.0)
```

### 12. Monats-Entwicklung (GuV-Konten)
```
Pro GuV-Konto (40000-79999, mind. 3 Monate):
z-Score der Monatssumme > 2.5 -> Ausreisser-Monat (Gewicht 1.5)
Alle Buchungen des Ausreisser-Monats werden markiert
```

### 13. Fehlende Monatsbuchung
```
Konto bucht in >= 30% aller Monate -> "regelmaessig"
Fehlender Monat -> Nachbar-Monate werden markiert (Gewicht 1.0)
Mind. 6 Monate Zeitspanne noetig
```

### Ertrag vs. Aufwand (Soll/Haben-Logik) — fuer Charts
```
Ertragskonto (40000-59999): Haben -> + | Soll -> -
Aufwandskonto (60000-79999): Soll -> + | Haben -> -
Bestandskonto (0-39999): Originalvorzeichen

Netto-Ertrag = Summe aller vorzeichenbereinigten Ertrags-Buchungen
Netto-Aufwand = Summe aller vorzeichenbereinigten Aufwands-Buchungen
```

---

## Generalumgekehrt

Wenn `generalumgekehrt` = "1"/"true"/"J" -> automatisch STORNO-Flag.

## Buchungsperiode-Fallback

`RECHNUNGSDATUM_PERIODE` nutzt `buchungsperiode` als Fallback wenn kein `erfassungsdatum` vorhanden.

---

## Dependency Management

- **requirements.txt**: Loose Pins (minimale Versionsanforderungen)
- **requirements.lock**: `pip freeze` aus `python:3.12-slim` Container — exakte Versionen fuer reproduzierbare Builds
- **Dockerfile**: Installiert aus `requirements.lock` wenn vorhanden, Fallback `requirements.txt`
- **pyproject.toml**: Muss synchron mit requirements.txt gehalten werden

## Gradio 6.x

- `css=` und `theme=` gehoeren in `demo.launch()`, NICHT in `gr.Blocks()`
- Alle anderen APIs (File, Slider, Button, Dataframe, Progress, Tabs) sind kompatibel

## Structured Logging

- **structlog** >= 24.1.0 fuer strukturiertes Logging (JSON/Console)
- `src/logging_config.py`: `setup_logging()` + `get_logger(name)`
- ENV-Steuerung:
  - `LOG_FORMAT` = `json` (default) | `console`
  - `LOG_LEVEL` = `INFO` (default) | `DEBUG` | `WARNING` | `ERROR`
- Logger-Konvention: `prefilter.ui`, `prefilter.engine`, `prefilter.worker`, `prefilter.api`, `prefilter.webhook`

## Docker Compose

```bash
docker compose up -d --build                  # Standard
docker compose up -d --build --scale worker=4 # Mehr Worker
```

- `--scale worker=N` nutzen (deploy.replicas wird von non-Swarm docker compose ignoriert)
- `WORKER_CONCURRENCY` (default 4): Celery-Prozesse pro Worker-Container
- Shared Volume `uploads:/data/uploads` zwischen UI und Worker

## Redis-Keys (Konvention)

| Key | Typ | Inhalt |
|---|---|---|
| `job:{id}` | Hash | status, progress_pct, current_test, result (JSON), error, filename |
| `job:{id}:cancelled` | String | "1" wenn Abbruch angefordert |
| `job:{id}:log` | List | Klartext-Log-Zeilen fuer Live-UI |
| `job:{id}:logs` | Pub/Sub | JSON-Events fuer WebSocket-Streaming |

## Tests

```bash
pytest tests/ -v                    # Unit-Tests (ohne slow/integration)
pytest tests/ -v -m slow            # Performance-Benchmark (500k Zeilen, <90s)
pytest tests/ -v -m integration     # E2E mit Redis (docker compose up redis)
```

- `pyproject.toml`: `addopts = "-m 'not slow and not integration'"`
- Tests importieren von `src.*`, niemals von `modules.*` (geloescht)

---

## Projektstruktur

```
prefilter-api/
+-- .env                 # LANGDOCK_WEBHOOK_URL, GRADIO_USERNAME/PASSWORD (nicht im Git!)
+-- docker-compose.yml   # Container-Definition (ui, worker, redis)
+-- Dockerfile           # Python 3.12-slim + Dependencies
+-- pyproject.toml       # Paket-Metadaten + pytest-Konfiguration
+-- requirements.txt     # Python-Abhaengigkeiten (loose pins)
+-- requirements.lock    # pip freeze -> reproduzierbare Builds
+-- app.py               # Gradio UI + Celery-Dispatcher + lokaler Fallback
+-- src/
|   +-- accounting.py    # !!! Kontoklassen + Vorzeichen-Logik (ZENTRAL) !!!
|   +-- config.py        # AnalysisConfig (Pydantic-Schwellenwerte)
|   +-- engine.py        # AnomalyEngine: orchestriert 13 Tests
|   +-- main.py          # FastAPI REST API + WebSocket
|   +-- models.py        # Pydantic-Ergebnis-Modelle
|   +-- parser.py        # CSV/XLS Einlesen + Spalten-Mapping + Serie-Parsing
|   +-- validator.py     # Spalten-Validierung, Test-Blocking
|   +-- charts.py        # Plotly-Visualisierungen (10 Charts)
|   +-- webhook.py       # Langdock Webhook Push (3 Retries)
|   +-- worker.py        # Celery Task (Redis-Backend)
|   +-- logging_config.py
|   +-- tests/           # Test-Module (5 Module, 13 Tests)
|       +-- __init__.py
|       +-- base.py      # AnomalyTest Basisklasse, EngineStats
|       +-- betrag.py    # BETRAG_ZSCORE, BETRAG_IQR, KONTO_BETRAG_ANOMALIE
|       +-- duplikate.py # NEAR_DUPLICATE, DOPPELTE_BELEGNUMMER, BELEG_KREDITOR_DUPLIKAT
|       +-- buchungslogik.py  # STORNO, LEERER_BUCHUNGSTEXT, RECHNUNGSDATUM_PERIODE, BUCHUNGSTEXT_PERIODE
|       +-- kreditor.py  # NEUER_KREDITOR_HOCH (nur noch 1 Test!)
|       +-- zeitreihe.py # MONATS_ENTWICKLUNG, FEHLENDE_MONATSBUCHUNG
+-- tests/
|   +-- test_engine.py       # Unit-Tests
|   +-- test_charts.py       # Chart-Tests mit synthetischen Daten
|   +-- test_performance.py  # 500k-Zeilen Benchmark (@pytest.mark.slow)
|   +-- test_integration.py  # FastAPI+Redis E2E (@pytest.mark.integration)
|   +-- test_worker_parallel.py
+-- .devcontainer/
|   +-- devcontainer.json
|   +-- docker-compose.yml
+-- .github/
    +-- copilot-instructions.md   # <-- DIESES DOKUMENT
    +-- workflows/
        +-- ci.yml
```

---

## Entfernte Features

| Feature | Warum entfernt | Version |
|---|---|---|
| `VELOCITY_ANOMALIE` | Diamant-Export enthaelt keine `erfasser`-Spalte. Test lief immer leer (return 0). | v5.1 |
| `erfasser` in COLUMN_ALIASES | Nicht im Diamant-Export vorhanden. Alle Referenzen entfernt. | v5.1 |
| `velocity_min_months` in AnalysisConfig | Config fuer entfernten Test. | v5.1 |

---

## Bekannte offene Punkte

1. **Single-Job-Parallelisierung** (ERLEDIGT): Celery chord Pipeline implementiert (worker.py v4.2).
2. **FEHLENDE_MONATSBUCHUNG** (ERLEDIGT): Logik verifiziert.
3. **deploy.replicas** (ERLEDIGT): Aus docker-compose.yml entfernt.
4. **Diamant-Integration** (ERLEDIGT v7): Parser, Betrags-Tests, Storno, Rechnungsdatum.
5. **VELOCITY_ANOMALIE** (ERLEDIGT v5.1): Vollstaendig entfernt.
6. **soll_haben** (ERLEDIGT v5.1): Parser-Alias, _betrag_signed, Charts aktualisiert.
7. **Kontoklasse zentralisiert** (ERLEDIGT v5.1): src/accounting.py als Single Source of Truth.
8. **konto_haben-Dependency** (ERLEDIGT v5.1): Aus required_columns entfernt, Soll/Haben-Paare via soll_haben.

---

## Haeufige Fehlerquellen

- **HfFolder ImportError**: Gradio < 5.12.0 + huggingface_hub >= 1.0.0. Fix: Gradio >= 5.12.0.
- **pydantic-core Build-Fehler**: Python 3.13 + pydantic < 2.10. Fix: `pydantic>=2.7.4` oder Python 3.12.
- **conda-env pollutes pip freeze**: Lock-File IMMER aus Docker generieren, nicht aus conda.
- **Performance-Test flaky**: 500k-Benchmark ist hardwareabhaengig (~68s auf Dev-Laptop). Limit ist 90s.
- **Kontoklassen-Inkonsistenz**: NIEMALS Grenzen (40000, 60000 etc.) inline definieren. Immer `from src.accounting import ...`.
- **_betrag vs _betrag_signed**: Tests nutzen `_abs`. Charts fuer PnL nutzen `_betrag_signed`. Verwechslung fuehrt zu falschen Vorzeichen.
- **soll_haben fehlt**: Wenn Spalte L nicht im Input ist, fallen PnL-Charts auf Absolutbetraege zurueck. Validator meldet Warnung.
- **konto_haben leer**: Normal bei Diamant-Export. Tests muessen OHNE konto_haben funktionieren.
- **13 nicht 14 Tests**: VELOCITY_ANOMALIE ist entfernt. Nicht "14 Tests" schreiben.

---

## Codierungs-Regeln fuer Copilot

1. **Keine neuen Kontoklassen-Definitionen** — immer `src/accounting.py` importieren
2. **Neue Tests**: `AnomalyTest`-Subklasse + `get_tests()` + TEST_REQUIREMENTS + TEST_CATEGORIES
3. **Nie `iterrows()`** — alles vektorisiert (pandas vectorized ops)
4. **Flag-Spalten**: `df[f"flag_{name}"] = True/False`, nie Listen in Zellen
5. **Deutsche Betraege**: `parse_german_number_series()` aus `src/parser.py`
6. **Tests aktualisieren**: Bei jeder Aenderung auch `tests/test_engine.py` und ggf. `tests/test_charts.py` anpassen
7. **13 Tests**: Nicht "14 Tests" schreiben — VELOCITY_ANOMALIE ist entfernt
8. **`konto_haben`**: Nicht als required behandeln, ist im Diamant-Export leer
9. **Charts**: PnL-Charts nutzen `_betrag_signed`, nicht `_betrag` oder `_abs`
10. **Validator**: Neue Spalten-Abhaengigkeiten muessen in TEST_REQUIREMENTS eingetragen werden
- **IMMER README.MD aKTUALISIEREN nach Änderungen, damit Copilot die neuesten Infos hat!** 