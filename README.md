# Buchungs-Anomalie Pre-Filter v4.0

Gradio-Web-App, die CSV-/XLS-/XLSX-Dateien mit Buchungsdaten entgegennimmt, 14 statistische Anomalie-Tests durchführt und verdächtige Buchungen anzeigt + optional per Webhook an einen Langdock Agent sendet.

---

## Schnellstart

```bash
# .env anlegen (siehe Abschnitt Konfiguration)
docker compose up -d --build
```

Die Web-UI läuft auf **http://localhost:7864**.

---

## Lokale Entwicklung

```bash
# conda-Umgebung anlegen (Python 3.13 ist inkompatibel — conda nutzen)
conda create -n prefilter-dev python=3.12
conda run -n prefilter-dev pip install -r requirements.txt

# Tests ausführen
conda run -n prefilter-dev python -m pytest tests/ -v

# App starten
conda run -n prefilter-dev python app.py
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
   - **Logs** — Detaillierte Engine-Logs aller 14 Tests

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

| Kanonischer Name  | Akzeptierte Aliase                                              |
| ----------------- | --------------------------------------------------------------- |
| `datum`           | `datum`, `date`, `buchungsdatum`, `belegdatum`                  |
| `betrag`          | `betrag`, `amount`, `summe`, `wert`                             |
| `konto_soll`      | `konto_soll`, `kontosoll`, `soll`, `sollkonto`, `debit`         |
| `konto_haben`     | `konto_haben`, `kontohaben`, `haben`, `habenkonto`, `credit`    |
| `buchungstext`    | `buchungstext`, `text`, `beschreibung`, `verwendungszweck`      |
| `belegnummer`     | `belegnummer`, `beleg`, `belegnr`, `beleg_nr`, `voucher`        |
| `kostenstelle`    | `kostenstelle`, `kst`, `cost_center`                            |
| `kreditor`        | `kreditor`, `lieferant`, `vendor`, `supplier`, `creditor`       |
| `erfasser`        | `erfasser`, `user`, `benutzer`, `ersteller`, `created_by`       |
| `rechnungsdatum`  | `rechnungsdatum`, `invoice_date`, `rech_datum`, `invoicedate`   |

- **Beträge**: Deutsche (`1.234,56`) und englische (`1,234.56`) Formate werden unterstützt.
- **Datumsformate**: `YYYY-MM-DD`, `DD.MM.YYYY`, `DD.MM.YY` und weitere gängige Formate.

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
      "erfasser": "m.mueller",
      "anomaly_score": 7.5,
      "anomaly_flags": "BETRAG_ZSCORE|NEAR_DUPLICATE|NEUER_KREDITOR_HOCH"
    }
  ],
  "logs": [
    "Geladen: 500 Buchungen",
    "[01/14] BETRAG_ZSCORE: 5"
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
| `erfasser`       | `string` | Buchungserfasser                                           |
| `anomaly_score`  | `float`  | Gewichteter Anomalie-Score (Summe aller Flag-Gewichte)     |
| `anomaly_flags`  | `string` | Pipe-getrennte Liste der ausgelösten Anomalie-Flags        |

---

## Anomalie-Tests (14 Stück)

| #  | Flag                      | Gewicht | Kritisch | Beschreibung                                                               |
| -- | ------------------------- | ------- | -------- | -------------------------------------------------------------------------- |
| 01 | `BETRAG_ZSCORE`           | 2.0     | ✓        | Betrag > 2,5 Standardabweichungen vom Mittelwert                          |
| 02 | `BETRAG_IQR`              | 1.5     |          | Betrag oberhalb des IQR-Fence (Q3 + 1,5 × IQR)                           |
| 03 | `NEAR_DUPLICATE`          | 2.0     | ✓        | Gleicher Betrag + Konten, Buchungsdatum innerhalb von 3 Tagen             |
| 04 | `DOPPELTE_BELEGNUMMER`    | 2.0     | ✓        | Gleiche Belegnummer taucht mehrfach auf                                   |
| 05 | `BELEG_KREDITOR_DUPLIKAT` | 2.5     | ✓        | Gleiche Belegnummer + gleicher Kreditor (mögliche doppelte Zahlung)        |
| 06 | `STORNO`                  | 1.5     | ✓        | Buchungstext enthält Storno/Korrektur/Rückbuchung oder negativer Betrag   |
| 07 | `NEUER_KREDITOR_HOCH`     | 2.5     | ✓        | Kreditor mit ≤ 2 Buchungen und hohem Betrag (> μ + 1,5σ)                 |
| 08 | `KONTO_BETRAG_ANOMALIE`   | 2.0     | ✓        | Betrag ist Ausreißer relativ zur Kontonorm (Z-Score auf Kontoebene)       |
| 09 | `LEERER_BUCHUNGSTEXT`     | 1.0     |          | Buchungstext fehlt oder ist kürzer als 3 Zeichen                          |
| 10 | `VELOCITY_ANOMALIE`       | 1.5     |          | Ungewöhnlich viele Buchungen desselben Erfassers an einem Tag             |
| 11 | `RECHNUNGSDATUM_PERIODE`  | 1.5     |          | Rechnungsmonat weicht vom Buchungsmonat ab (Periodenverschiebung)         |
| 12 | `BUCHUNGSTEXT_PERIODE`    | 1.0     |          | Periodenangabe im Buchungstext stimmt nicht mit Buchungsdatum überein     |
| 13 | `MONATS_ENTWICKLUNG`      | 1.5     |          | Monatlicher Betrag auf GuV-Konto ist Z-Score-Ausreißer (≥ 8 Normalmonate) |
| 14 | `FEHLENDE_MONATSBUCHUNG`  | 1.0     |          | Konto hat regulär monatliche Buchungen, fehlt aber in einem Monat         |

**Kritische Flags** führen dazu, dass die Buchung **immer** im Output erscheint, unabhängig vom Score-Schwellenwert.

### Scoring

- Jedes ausgelöste Flag addiert sein **Gewicht** zum `anomaly_score` der Buchung.
- Buchungen mit `anomaly_score ≥ 2.0` oder mindestens einem kritischen Flag werden ausgegeben.
- Maximal **1.000 Zeilen** im Output (sortiert nach Score absteigend).

---

## Projektstruktur

```
prefilter-api/
├── .env                 # LANGDOCK_WEBHOOK_URL, GRADIO_USERNAME/PASSWORD (nicht im Git!)
├── docker-compose.yml   # Container-Definition mit env_file
├── Dockerfile           # Python 3.12-slim + Dependencies
├── pyproject.toml       # Paket-Metadaten + pytest-Konfiguration
├── requirements.txt     # Python-Abhängigkeiten
├── app.py               # Gradio UI + Handler
├── modules/
│   ├── engine.py        # AnomalyEngine: 14 Tests, vollständig vektorisiert
│   ├── parser.py        # CSV/XLS Einlesen + Spalten-Mapping + serie-Parsing
│   └── webhook.py       # Langdock Webhook Push (3 Retries)
└── tests/
    └── test_engine.py   # 51 Unit-Tests
```
