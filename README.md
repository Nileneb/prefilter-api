# Buchungs-Anomalie Pre-Filter v1.0

Gradio-Web-App, die CSV-/XLS-/XLSX-Dateien mit Buchungsdaten entgegennimmt, 12 statistische Anomalie-Tests durchführt und verdächtige Buchungen anzeigt + optional per Webhook an einen Langdock Agent sendet.

---

## Schnellstart

```bash
# .env anlegen (siehe Abschnitt Konfiguration)
docker compose up -d --build
```

Die Web-UI läuft auf **http://localhost:7864**.

---

## Konfiguration (.env)

```env
# Langdock Webhook-URL (optional — leer lassen für nur lokale Analyse)
LANGDOCK_WEBHOOK_URL=https://api.langdock.com/webhook/...
```

Die Webhook-URL kann auch direkt in der Web-UI überschrieben werden.

---

## Bedienung

1. **Datei hochladen** — CSV, XLS oder XLSX mit Buchungsdaten
2. **Webhook-URL** (optional) — Langdock-Webhook-Ziel eingeben oder leer lassen
3. **"Analyse starten"** klicken
4. Ergebnis in drei Tabs:
   - **Ergebnis** — Zusammenfassung mit Top-3 verdächtigen Buchungen
   - **Verdächtige Buchungen** — Sortierbare Tabelle aller Treffer
   - **Logs** — Detaillierte Engine-Logs aller 12 Tests

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

| Kanonischer Name  | Akzeptierte Aliase                                         |
| ----------------- | ---------------------------------------------------------- |
| `datum`           | `datum`, `date`, `buchungsdatum`, `belegdatum`             |
| `betrag`          | `betrag`, `amount`, `summe`, `wert`                        |
| `konto_soll`      | `konto_soll`, `kontosoll`, `soll`, `sollkonto`, `debit`   |
| `konto_haben`     | `konto_haben`, `kontohaben`, `haben`, `habenkonto`, `credit` |
| `buchungstext`    | `buchungstext`, `text`, `beschreibung`, `verwendungszweck` |
| `belegnummer`     | `belegnummer`, `beleg`, `belegnr`, `beleg_nr`, `voucher`   |
| `kostenstelle`    | `kostenstelle`, `kst`, `cost_center`                       |
| `kreditor`        | `kreditor`, `lieferant`, `vendor`, `supplier`, `creditor`  |
| `erfasser`        | `erfasser`, `user`, `benutzer`, `ersteller`, `created_by`  |

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
      "anomaly_flags": "BETRAG_ZSCORE|NEAR_DUPLICATE|SPLIT_VERDACHT"
    }
  ],
  "logs": [
    "Geladen: 500 Buchungen",
    "[01/12] BETRAG_ZSCORE: 5"
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

## Anomalie-Tests (12 Stück)

| #  | Flag                   | Gewicht | Kritisch | Beschreibung                                                          |
| -- | ---------------------- | ------- | -------- | --------------------------------------------------------------------- |
| 01 | `BETRAG_ZSCORE`        | 2.0     | ✓        | Betrag > 2,5 Standardabweichungen vom Mittelwert                     |
| 02 | `BETRAG_IQR`           | 1.5     |          | Betrag oberhalb des IQR-Fence (Q3 + 1,5 × IQR)                      |
| 03 | `SELTENE_KONTIERUNG`   | 1.5     |          | Soll→Haben-Kombination kommt sehr selten vor (≤ 1% der Buchungen)   |
| 04 | `WOCHENENDE`           | 1.0     |          | Buchung am Samstag oder Sonntag                                      |
| 04 | `MONATSENDE`           | 0.5     |          | Buchung in den letzten 3 Tagen des Monats                            |
| 04 | `QUARTALSENDE`         | 0.5     |          | Monatsende + Quartalsende-Monat (März, Juni, Sept, Dez)              |
| 05 | `NEAR_DUPLICATE`       | 2.0     | ✓        | Gleicher Betrag + Konten, Datum innerhalb von 3 Tagen                |
| 06 | `BENFORD`              | 1.0     |          | Erste Ziffer weicht signifikant von Benfords Gesetz ab               |
| 07 | `RUNDER_BETRAG`        | 1.0     |          | Runde Beträge ≥ 1.000 (500er-Schritte) oder ≥ 5.000 (1000er)       |
| 08 | `ERFASSER_ANOMALIE`    | 1.5     |          | Erfasser mit ungewöhnlich wenigen Buchungen (≤ 3% des Totals)       |
| 09 | `SPLIT_VERDACHT`       | 2.0     | ✓        | ≥ 3 Buchungen am selben Tag, gleicher Kreditor/Erfasser + Soll-Konto |
| 10 | `BELEG_LUECKE`         | 1.0     |          | Lücke > 5 in der Belegnummernsequenz                                 |
| 11 | `STORNO`               | 1.5     | ✓        | Buchungstext enthält Storno/Korrektur/Rückbuchung oder negativer Betrag |
| 12 | `NEUER_KREDITOR_HOCH`  | 2.5     | ✓        | Kreditor mit ≤ 2 Buchungen und hohem Betrag (> μ + 1,5σ)            |

**Kritische Flags** führen dazu, dass die Buchung **immer** im Output erscheint, unabhängig vom Score-Schwellenwert.

### Scoring

- Jedes ausgelöste Flag addiert sein **Gewicht** zum `anomaly_score` der Buchung.
- Buchungen mit `anomaly_score ≥ 1.0` oder mindestens einem kritischen Flag werden ausgegeben.
- Maximal **1.000 Zeilen** im Output (sortiert nach Score absteigend).

---

## Projektstruktur

```
prefilter-api/
├── .env                 # LANGDOCK_WEBHOOK_URL (nicht im Git!)
├── docker-compose.yml   # Container-Definition mit env_file
├── Dockerfile           # Python 3.12 slim + Dependencies
├── main.py              # Gradio UI + Anomaly Engine + Webhook Push
└── requirements.txt     # Python-Abhängigkeiten
```
