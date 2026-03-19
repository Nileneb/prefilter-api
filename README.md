# Buchungs-Anomalie Pre-Filter v6.0

Gradio-Web-App, die CSV-/XLS-/XLSX-Dateien mit Buchungsdaten (inkl. Diamant-Export mit Pipe-Delimiter) entgegennimmt, 13 statistische Anomalie-Tests durchführt und verdächtige Buchungen anzeigt + optional per Webhook an einen Langdock Agent sendet.

### v6.0 — Beleg-Aware Refactor

- **Beleg-Ebene**: Buchungszeilen desselben Belegs (DVBelegnummer) werden als zusammengehörig erkannt; Duplikat-Tests ignorieren beleg-interne Zeilen
- **Storno-Ausschluss**: Storno-Buchungen (Generalumgekehrt ≠ leer) werden aus 9 von 13 Tests automatisch ausgeschlossen, um False-Positives zu eliminieren
- **Generalumgekehrt-Fix**: Das Feld enthält DVBelegnummern des Storno-Gegenbelegs (nicht boolean) — jeder nicht-leere Wert = Storno
- **Kontoklasse Kostenrechnung**: Kontonummern ≥ 80000 werden als "Kostenrechnung" klassifiziert (statt fälschlich "Bestand")
- **float64-Präzision**: `_betrag`/`_abs` nutzen float64 statt float32 (keine Rundungsartefakte mehr)
- **Erweiterte Spalten-Aliase**: DVBelegnummer, DVBuchungsnummer, InterneBelegnummer, Mandant, Belegart u.a.

---

## Schnellstart

```bash
# .env anlegen (siehe Abschnitt Konfiguration)
docker compose up -d --build
```

Die Web-UI läuft auf **http://localhost:7864**.

---

## Lokale Entwicklung

> **Docker Compose ist der empfohlene Betriebsmodus.** Ohne Redis/Celery startet
> `app.py` automatisch im lokalen Fallback-Modus (direkte Analyse ohne Worker).

```bash
conda create -n prefilter-dev python=3.12
conda run -n prefilter-dev pip install -r requirements.txt

# Tests ausführen (Unit-Tests, ohne slow/integration)
conda run -n prefilter-dev python -m pytest tests/ -v

# Performance-Test (500k Zeilen)
conda run -n prefilter-dev python -m pytest tests/ -v -m slow

# Integration-Tests (Redis muss laufen)
conda run -n prefilter-dev python -m pytest tests/ -v -m integration

# App starten (lokaler Fallback ohne Redis)
conda run -n prefilter-dev python app.py
```

### requirements.lock aktualisieren

Nach Änderungen an `requirements.txt`:

```bash
docker run --rm -v "$PWD/requirements.txt:/app/requirements.txt" \
  python:3.12-slim sh -c \
  "pip install --no-cache-dir -r /app/requirements.txt >/dev/null 2>&1 && pip freeze" \
  > requirements.lock
```

---

## Konfiguration (.env)

```env
# Langdock Webhook-URL (optional — leer lassen für nur lokale Analyse)
LANGDOCK_WEBHOOK_URL=https://api.langdock.com/webhook/...

# Optionale HTTP-Basic-Auth für die Gradio-UI
GRADIO_USERNAME=admin
GRADIO_PASSWORD=secret

# Root-Path wenn hinter Reverse-Proxy mit Subpath
ROOT_PATH=/prefilter
```

Die Webhook-URL kann auch direkt in der Web-UI überschrieben werden.

---

## Bedienung

1. **Datei hochladen** — CSV, XLS oder XLSX mit Buchungsdaten
2. **Webhook-URL** (optional) — Langdock-Webhook-Ziel eingeben oder leer lassen
3. **"Analyse starten"** klicken
4. Ergebnis in drei Tabs:
   - **Ergebnis** — Zusammenfassung mit Top-3 verdächtigen Buchungen
   - **Verdächtige Buchungen** — Sortierbare Tabelle aller Treffer (Score ≥ 2.0)
   - **Logs** — Detaillierte Engine-Logs aller 13 Tests

---

## Eingabe-Datei

### Unterstützte Formate

| Format | Hinweise |
| ------ | -------- |
| CSV    | Separator wird automatisch erkannt (`;`, `,`, `\t`). Encoding: UTF-8 (BOM wird erkannt) |
| XLS    | Klassisches Excel-Format |
| XLSX   | Modernes Excel-Format |

### Erwartete Spalten

Die Spalten werden automatisch erkannt (case-insensitive, Umlaute werden normalisiert). Es müssen nicht alle Spalten vorhanden sein, aber **`datum`** und **`betrag`** sind für sinnvolle Ergebnisse essenziell.

| Kanonischer Name       | Akzeptierte Aliase                                                          |
| ---------------------- | --------------------------------------------------------------------------- |
| `datum`                | `datum`, `date`, `buchungsdatum`, `belegdatum`                              |
| `betrag`               | `betrag`, `amount`, `summe`, `wert`, `fibubetrag`                           |
| `konto_soll`           | `konto_soll`, `kontosoll`, `soll`, `sollkonto`, `debit`, `kontonummer`      |
| `konto_haben`          | `konto_haben`, `kontohaben`, `haben`, `habenkonto`, `credit`                |
| `buchungstext`         | `buchungstext`, `text`, `beschreibung`, `verwendungszweck`                  |
| `belegnummer`          | `belegnummer`, `beleg`, `belegnr`, `beleg_nr`, `voucher`                    |
| `kostenstelle`         | `kostenstelle`, `kst`, `cost_center`                                        |
| `kreditor`             | `kreditor`, `lieferant`, `vendor`, `supplier`, `creditor`, `bezeichnung`    |
| `soll_haben`           | `soll_haben`, `sollhaben`, `s_h`, `sh`, `soll/haben`                       |
| `dvbelegnummer`        | `dvbelegnummer`, `dv_belegnummer`, `dv_beleg_nr`                            |
| `dvbuchungsnummer`     | `dvbuchungsnummer`, `dv_buchungsnummer`                                     |
| `interne_belegnummer`  | `interne_belegnummer`, `internebelegnummer`, `intern_beleg`                 |
| `generalumgekehrt`     | `generalumgekehrt`, `storno_kz`, `umkehr`                                   |
| `klasse`               | `klasse`, `class`                                                           |
| `buchungsperiode`      | `buchungsperiode`, `periode`, `period`                                      |
| `erfassungsdatum`      | `erfassungsdatum`, `erfassungam`, `erfassung_am`, `created_at`              |
| `belegart`             | `belegart`, `beleg_art`                                                     |
| `mandant`              | `mandant`, `firma`, `mandanten_nr`                                          |
| `detailbetrag`         | `detailbetrag`, `detail_betrag`                                             |

- **Beträge**: Deutsche (`1.234,56`) und englische (`1,234.56`) Formate werden unterstützt.
- **Datumsformate**: `YYYY-MM-DD`, `DD.MM.YYYY`, `DD.MM.YY` und weitere gängige Formate.

### Diamant-Beleg-Struktur

Die Eingabedaten stammen typischerweise aus Diamant/4 Finanzbuchhaltung. Jede Zeile ist eine **Buchungszeile** (nicht ein Beleg). Ein Beleg (DVBelegnummer) besteht aus mindestens 2 Zeilen (Soll + Haben = doppelte Buchführung). Splitbuchungen erzeugen >2 Zeilen pro Beleg.

- **DVBelegnummer**: Gruppiert Zeilen zu einem Beleg (interne ID)
- **Belegnummer**: Externe Rechnungsnummer — kommt natürlich mehrfach vor (min. 2× pro Beleg)
- **Generalumgekehrt**: Enthält die DVBelegnummer des Storno-**Gegenbelegs** (nicht Ja/Nein!)
- **Klasse**: K=Kreditor, D=Debitor, S=Sachkonto

### Kontoklassen (src/accounting.py)

| Kontoklasse        | Bereich       |
| ------------------ | ------------- |
| Bestand            | 0 – 39 999   |
| Ertrag             | 40 000 – 59 999 |
| Aufwand            | 60 000 – 79 999 |
| Kostenrechnung     | ≥ 80 000     |

---

## Webhook-Push

Nach der Analyse wird das komplette Ergebnis-JSON per `POST` an die konfigurierte Langdock-Webhook-URL gesendet.

### Payload-Struktur

```json
{
  "message": "42 verdächtige Buchungen (8.4%)",
  "statistics": {
    "total_input": 500,
    "total_output": 42,
    "filter_ratio": "8.4%",
    "avg_score": 1.23,
    "flag_counts": {
      "BETRAG_ZSCORE": 5,
      "NEAR_DUPLICATE": 12,
      "STORNO": 8
    }
  },
  "verdaechtige_buchungen": [
    {
      "datum": "2025-12-31",
      "konto_soll": "4711",
      "konto_haben": "1200",
      "betrag": 15000.0,
      "buchungstext": "Rechnung XY",
      "belegnummer": "RE-2025-0042",
      "kostenstelle": "100",
      "kreditor": "Lieferant GmbH",
      "soll_haben": "S",
      "anomaly_score": 7.5,
      "anomaly_flags": "BETRAG_ZSCORE|NEAR_DUPLICATE|NEUER_KREDITOR_HOCH"
    }
  ],
  "logs": [
    "Geladen: 500 Buchungen",
    "[01/13] BETRAG_ZSCORE: 5"
  ]
}
```

### Felder pro verdächtige Buchung

| Feld             | Typ      | Beschreibung                                               |
| ---------------- | -------- | ---------------------------------------------------------- |
| `datum`          | `string` | Buchungsdatum (ISO 8601)                                   |
| `konto_soll`     | `string` | Soll-Konto                                                 |
| `konto_haben`    | `string` | Haben-Konto                                                |
| `betrag`         | `float`  | Betrag (geparst, numerisch)                                |
| `buchungstext`   | `string` | Buchungsbeschreibung                                       |
| `belegnummer`    | `string` | Belegnummer                                                |
| `kostenstelle`   | `string` | Kostenstelle                                               |
| `kreditor`       | `string` | Kreditor / Lieferant                                       |
| `soll_haben`     | `string` | Soll/Haben-Kennzeichen (S/H)                               |
| `anomaly_score`  | `float`  | Gewichteter Anomalie-Score (Summe aller Flag-Gewichte)     |
| `anomaly_flags`  | `string` | Pipe-getrennte Liste der ausgelösten Anomalie-Flags        |

---

## Anomalie-Tests (13 Stück)

| #  | Flag                      | Gewicht | Kritisch | Beschreibung                                                               |
| -- | ------------------------- | ------- | -------- | -------------------------------------------------------------------------- |
| 01 | `BETRAG_ZSCORE`           | 2.0     | ✓        | Betrag > 2,5 Standardabweichungen vom Mittelwert (je Kontoklasse)         |
| 02 | `BETRAG_IQR`              | 1.5     |          | Betrag oberhalb des IQR-Fence (Q3 + 3,0 × IQR, je Kontoklasse)           |
| 03 | `KONTO_BETRAG_ANOMALIE`   | 2.0     | ✓        | Betrag ist Ausreißer relativ zur Kontonorm (Z-Score auf Kontoebene)       |
| 04 | `NEAR_DUPLICATE`          | 2.0     | ✓        | Gleicher Betrag + Konten, Buchungsdatum innerhalb von 3 Tagen             |
| 05 | `DOPPELTE_BELEGNUMMER`    | 2.0     | ✓        | Gleiche Belegnummer taucht mehrfach auf                                   |
| 06 | `BELEG_KREDITOR_DUPLIKAT` | 2.5     | ✓        | Gleiche Belegnummer + gleicher Kreditor (mögliche doppelte Zahlung)        |
| 07 | `STORNO`                  | 1.5     | ✓        | Buchungstext enthält Storno/Korrektur/Rückbuchung oder Generalumgekehrt   |
| 08 | `LEERER_BUCHUNGSTEXT`     | 1.0     |          | Buchungstext fehlt oder ist kürzer als 3 Zeichen                          |
| 09 | `RECHNUNGSDATUM_PERIODE`  | 1.5     |          | Erfassungsmonat weicht vom Buchungsmonat ab (Periodenverschiebung)        |
| 10 | `BUCHUNGSTEXT_PERIODE`    | 1.0     |          | Periodenangabe im Buchungstext stimmt nicht mit Buchungsdatum überein     |
| 11 | `NEUER_KREDITOR_HOCH`     | 2.5     | ✓        | Kreditor mit ≤ 2 Buchungen und hohem Betrag (> μ + 1,5σ)                 |
| 12 | `MONATS_ENTWICKLUNG`      | 1.5     |          | Monatlicher Betrag auf GuV-Konto ist Z-Score-Ausreißer                    |
| 13 | `FEHLENDE_MONATSBUCHUNG`  | 1.0     |          | Konto hat regulär monatliche Buchungen, fehlt aber in einem Monat         |

**Kritische Flags** führen dazu, dass die Buchung **immer** im Output erscheint, unabhängig vom Score-Schwellenwert.

### Scoring

- Jedes ausgelöste Flag addiert sein **Gewicht** zum `anomaly_score` der Buchung.
- Buchungen mit `anomaly_score ≥ 2.0` oder mindestens einem kritischen Flag werden ausgegeben.
- Maximal **1.000 Zeilen** im Output (sortiert nach Score absteigend).

---

## Worker-Skalierung (große Datensätze)

Standard: 1 Worker-Replica mit 8 Celery-Prozessen. Für große Uploads:

```bash
# Mehr Worker-Replicas (empfohlen: 4 für Datensätze > 100k Zeilen)
docker compose up -d --build --scale worker=4
```

> **Hinweis:** `deploy.replicas` in `docker-compose.yml` wird von `docker compose` (non-Swarm)
> ignoriert. Nutze immer `--scale worker=N`.

### Parallele Pipeline (Single-Job-Beschleunigung)

Ab **100.000 Zeilen** (konfigurierbar via `PARALLEL_THRESHOLD`) werden die 13 Anomalie-Tests
automatisch parallel über alle verfügbaren Worker verteilt (Celery chord).

**Flow:** `analyze_task` liest die Datei, speichert den vorbereiteten DataFrame als Parquet
auf dem shared Volume. 13 `run_test_task`s laufen parallel (je einer pro Test, jeder liest
das Parquet). `merge_task` sammelt die Flag-Ergebnisse, berechnet Scores und exportiert.

**Konfiguration in `.env`:**

```env
PARALLEL_THRESHOLD=100000    # Ab wann parallel (Default: 100000)
WORKER_CONCURRENCY=8         # Celery-Prozesse pro Worker-Container
```

---

## Projektstruktur

```
prefilter-api/
├── .env                 # LANGDOCK_WEBHOOK_URL, GRADIO_USERNAME/PASSWORD (nicht im Git!)
├── docker-compose.yml   # Container-Definition (ui, worker, redis)
├── Dockerfile           # Python 3.12-slim + Dependencies
├── pyproject.toml       # Paket-Metadaten + pytest-Konfiguration
├── requirements.txt     # Python-Abhängigkeiten (loose pins)
├── requirements.lock    # pip freeze → reproduzierbare Builds
├── app.py               # Gradio UI + Celery-Dispatcher + lokaler Fallback
├── src/
│   ├── accounting.py    # Kontoklassen + Vorzeichen-Logik (ZENTRAL)
│   ├── config.py        # AnalysisConfig (Pydantic-Schwellenwerte)
│   ├── engine.py        # AnomalyEngine: orchestriert 13 Tests
│   ├── main.py          # FastAPI REST API + WebSocket
│   ├── models.py        # Pydantic-Ergebnis-Modelle
│   ├── parser.py        # CSV/XLS Einlesen + Spalten-Mapping + Serie-Parsing
│   ├── webhook.py       # Langdock Webhook Push (3 Retries)
│   ├── worker.py        # Celery Task (Redis-Backend)
│   └── tests/           # Test-Module (betrag, duplikate, buchungslogik, …)
└── tests/
    ├── test_engine.py       # 100 Unit-Tests
    ├── test_performance.py  # 500k-Zeilen Benchmark (@pytest.mark.slow)
    └── test_integration.py  # FastAPI+Redis E2E (@pytest.mark.integration)
```
