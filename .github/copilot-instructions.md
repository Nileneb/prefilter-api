# Prefilter-API — Copilot Instructions

> **Stand: 2026-03-19 · Version 6.2 — OUTPUT-LOGIK-UEBERARBEITUNG**
> Dieses Dokument ist die einzige Wahrheitsquelle für Architektur, Konventionen und Domain-Logik.
PYTHON ENV VERWENDEN!!!!!!!!!!!!!!!!!!!!!!!!!!!!! AUCH FÜR TESTS!!!!!!!!!!!!!!!!!!!!!!!!!!!
---

## Projektuebersicht

Buchungs-Anomalie Pre-Filter v6.2: Gradio-Web-App die CSV/XLS/XLSX-Buchungsdaten (inkl. Diamant-Export mit Pipe-Delimiter) durch **13 statistische Anomalie-Tests** laufen laesst und verdaechtige Buchungen an einen Langdock Agent weiterleitet. Betrieb ausschliesslich via Docker Compose.

### KRITISCH: Diamant-Datenmodell verstehen

Die Eingabedaten stammen aus Diamant/4 Finanzbuchhaltung. Jede Zeile ist eine **Buchungszeile** (NICHT ein Beleg). Die Grundstruktur:

```
EIN BELEG (DVBelegnummer) = MEHRERE BUCHUNGSZEILEN (DVBuchungsnummer)
- Jeder Beleg hat mindestens 2 Zeilen (Soll + Haben = Doppelte Buchfuehrung)
- Splitbuchungen erzeugen >2 Zeilen pro Beleg
- Stornos: Generalumgekehrt enthaelt die DVBelegnummer des Gegenbelegs (NICHT Ja/Nein!)
```

**Felder und ihre Bedeutung:**

| Diamant-Feld | Bedeutung | Wichtig |
|---|---|---|
| DVBelegnummer | Gruppiert Zeilen zu einem Beleg | Interne ID, NICHT die externe Belegnummer |
| DVBuchungsnummer | Identifiziert einzelne Buchungszeilen | Kann bei Kostenrechnungs-Splits mehrfach vorkommen |
| Belegnummer | Externe Rechnungsnummer / Belegnummer | Kommt NATUERLICH mehrfach vor (min. 2x pro Beleg!) |
| InterneBelegnummer | Gruppiert Zeilen eines Belegs | Alternative Beleg-ID |
| Klasse | K=Kreditor, D=Debitor, S=Sachkonto | Bestimmt Kontoart |
| Generalumgekehrt | DVBelegnummer des Storno-GEGENBELEGS | NICHT boolean! Jeder Wert != leer/NULL = STORNO |
| Detailbetrag | Teilbetraege bei Splitbuchungen | |
| FiBuBetrag | Buchungsbetrag | Vorzeichen ist semantisch |
| Kontonummer | Kontonummer (Sachkonto, Kreditor, Debitor) | Bereich bestimmt Kontoklasse |
| Kostenstelle | Kostenstelle (oft 820000) | 32% der Zeilen haben Konto >=80000 (Kostenrechnung) |

---

## Architektur

```
                   +----------+
User --> Gradio UI |  app.py  |--> Redis --> Celery Worker(s)
          :7864    +----------+              src/worker.py
                                              --> AnomalyEngine
                                                  src/engine.py
                   +--------------+
     (alternativ)  | FastAPI REST |--> Redis --> gleicher Worker
                   | src/main.py  |
                   |    :8000     |
                   +--------------+
```

- **app.py**: Gradio UI, dispatcht Celery-Tasks. Lokaler Fallback wenn Redis nicht erreichbar. Integriert file_store fuer Upload-/Ergebnis-Persistenz.
- **src/engine.py**: Orchestriert 13 Tests aus src/tests/. Kein iterrows(), alles vektorisiert.
- **src/accounting.py**: ZENTRAL — Einzige Stelle fuer Kontoklassen-Grenzen und Vorzeichen-Berechnung.
- **src/parser.py**: CSV/XLS-Parsing, Spalten-Mapping, Deutsche Zahlen/Datumsformate, NULL-Handling.
- **src/config.py**: AnalysisConfig (Pydantic) — alle Schwellenwerte konfigurierbar.
- **src/validator.py**: Spalten-Validierung, Test-Blocking.
- **src/charts.py**: Plotly-Visualisierungen. Importiert kontoklasse aus src/accounting.
- **src/file_store.py**: Persistente Speicherung von Uploads + Analyse-Ergebnissen (NEU v6.1).
- **src/history.py**: Monatsdurchschnitte pro Konto persistent speichern, Trend-Erkennung.

---

## KONTOKLASSEN — ZENTRALE DEFINITION (src/accounting.py)

**Es gibt EINE einzige Implementierung.** Alle anderen Dateien importieren von dort.

| Kontoklasse | Bereich | Beispiel |
|---|---|---|
| **Ertrag** | 40000-59999 | 42000, 50200 |
| **Aufwand** | 60000-79999 | 65000, 70100 |
| **Bestand** | 0-39999 | 1200 (Bank), 10000 |
| **Kostenrechnung** | 80000-99999 | 820000 (Kostenstelle) |

> **NEU v6.0:** Kontonummern >= 80000 werden als "Kostenrechnung" klassifiziert (32% der Diamant-Daten!). Vorher fielen sie faelschlich in "Bestand".

**NIEMALS** Kontoklassen-Grenzen inline in Tests, Charts oder Engine definieren.

---

## STORNO/GENERALUMGEKEHRT — KRITISCHER BUGFIX v6.0

### Altes (FALSCHES) Verhalten:
```python
# FALSCH! Generalumgekehrt enthaelt DVBelegnummern, NICHT Ja/Nein!
gu_mask = gu.isin({"1", "true", "j", "ja", "yes", "x"})  # -> 0 Treffer!
```

### Neues (KORREKTES) Verhalten:
```python
# RICHTIG! Jeder nicht-leere/nicht-NULL Wert = STORNO
gu_str = gu.astype(str).str.strip()
gu_mask = (gu_str != "") & (~gu_str.str.lower().isin({"nan", "null", "none"})) & (gu_str != "0")
```

Das Feld generalumgekehrt enthaelt die DVBelegnummer des Storno-GEGENBELEGS. Stornos erfolgen paarweise: Beleg A verweist auf B, Beleg B verweist auf A.

---

## BELEG-EBENE vs. ZEILEN-EBENE — KRITISCH (NEU v6.0)

### Das Problem (v5.1):
Das System arbeitete auf Zeilen-Ebene. Aber im Diamant-Export:
- Ein Beleg = min. 2 Zeilen (Soll + Haben)
- Belegnummer (externe Rechnungsnr) kommt NATUERLICH mehrfach vor
- Soll+Haben-Zeilen eines Belegs haben gleichen Betrag + Datum -> sehen wie Duplikate aus

### Das Ergebnis war MUELL:
- 810 von 1000 Output-Zeilen hatten DOPPELTE_BELEGNUMMER (= normale Beleg-Struktur!)
- 896 hatten BELEG_KREDITOR_DUPLIKAT (= doppelte Buchfuehrung!)
- Storno-Paare erzeugten NEAR_DUPLICATE + DOPPELTE_BELEGNUMMER

### Die Loesung (v6.0):
1. **_beleg_id-Spalte**: Engine mappt dvbelegnummer als Beleg-Gruppierung
2. **Beleg-Kontext**: Tests unterscheiden zwischen Beleg-internen Zeilen (normal) und Beleg-uebergreifenden Mustern (verdaechtig)
3. **Storno-Ausschluss**: Storno-Belege werden aus Duplikat-/Betrags-/Zeitreihen-Tests ausgeschlossen
4. **_is_storno-Spalte**: Boolean, gesetzt in _prepare(), BEVOR Tests laufen

### Reihenfolge in _prepare() — MUSS eingehalten werden:
1. map_columns (Parser) -> kanonische Spaltennamen
2. Fehlende Spalten mit "" initialisieren
3. _betrag + _abs berechnen (Zahlen-Parsing, **float64 statt float32!**)
4. _datum berechnen (Datums-Parsing)
5. _kontoklasse berechnen (aus konto_soll, inkl. Kostenrechnung)
6. _is_storno berechnen (aus generalumgekehrt + Buchungstext)
7. _beleg_id setzen (aus dvbelegnummer wenn vorhanden, sonst Fallback)
8. _betrag_signed berechnen (aus _abs + _kontoklasse + soll_haben)
9. Kategorische Spalten setzen (**mit .str.strip() fuer Diamant fixed-width!**)
10. Flag-Spalten initialisieren

**Erst danach** compute_stats() und Tests ausfuehren.

---

## Diamant-Export — Spalten-Mapping

| Diamant-Spalte | Kanonischer Name | Status |
|---|---|---|
| Mandant | mandant | Info |
| DVBelegnummer | dvbelegnummer | **NEU v6.0 — Beleg-Gruppierung** |
| DVBuchungsnummer | dvbuchungsnummer | Zeilen-ID |
| InterneBelegnummer | interne_belegnummer | Alternative Beleg-ID |
| Belegnummer | belegnummer | Externe Rechnungsnummer |
| Belegart | belegart | KADB/ER/ZAZ/BA/LOHN etc. |
| ErfassungAm | erfassungsdatum | Erfassungszeitpunkt |
| Belegdatum | datum | Buchungsdatum |
| Buchungsperiode | buchungsperiode | Abrechnungsperiode |
| Klasse | klasse (K/S/D) | Kreditor/Sachkonto/Debitor |
| Kontonummer | konto_soll | Kontonummer |
| Bezeichnung | kreditor | Kreditor-/Kontobezeichnung |
| Buchungstext | buchungstext | Freitext |
| Steuerschluessel | steuerschluessel | inaktiv |
| FiBuBetrag | betrag | Buchungsbetrag (deutsches Format!) |
| Kostenstelle | kostenstelle | Kostenstelle |
| Kostentraeger | kostentraeger | inaktiv |
| Projekt | projekt | inaktiv |
| Detailbetrag | detailbetrag | Splitbetraege |
| Generalumgekehrt | generalumgekehrt | **STORNO (DVBelegnummer des Gegenbelegs!)** |

> **ACHTUNG:** Spalte "Soll/Haben" (S/H) existiert im aktuellen Diamant-Export NICHT!
> soll_haben bleibt als optionales Feld im Parser, wird aber bei Diamant LEER sein.

### Parser COLUMN_ALIASES

```python
COLUMN_ALIASES = {
    "datum":              ["datum", "date", "buchungsdatum", "belegdatum"],
    "betrag":             ["betrag", "amount", "summe", "wert", "fibubetrag"],
    "konto_soll":         ["konto_soll", "kontosoll", "soll", "sollkonto", "debit", "kontonummer"],
    "konto_haben":        ["konto_haben", "kontohaben", "haben", "habenkonto", "credit"],
    "buchungstext":       ["buchungstext", "text", "beschreibung", "verwendungszweck"],
    "belegnummer":        ["belegnummer", "beleg", "belegnr", "beleg_nr", "voucher"],
    "kostenstelle":       ["kostenstelle", "kst", "cost_center"],
    "kreditor":           ["kreditor", "lieferant", "vendor", "supplier", "creditor", "bezeichnung"],
    "soll_haben":         ["soll_haben", "sollhaben", "s_h", "sh", "soll/haben"],
    "klasse":             ["klasse", "class"],
    "generalumgekehrt":   ["generalumgekehrt", "storno_kz", "umkehr"],
    "buchungsperiode":    ["buchungsperiode", "periode", "period"],
    "erfassungsdatum":    ["erfassungsdatum", "erfassungam", "erfassung_am", "created_at"],
    "detailbetrag":       ["detailbetrag", "detail_betrag"],
    "dvbelegnummer":      ["dvbelegnummer", "dv_belegnummer", "dv_beleg_nr"],
    "dvbuchungsnummer":   ["dvbuchungsnummer", "dv_buchungsnummer"],
    "interne_belegnummer": ["interne_belegnummer", "internebelegnummer", "intern_beleg"],
    "belegart":           ["belegart", "beleg_art"],
    "mandant":            ["mandant", "firma", "mandanten_nr"],
}
```

---

## 13 Tests (src/tests/)

| # | Modul | Test | Gewicht | Kritisch | Storno-Ausschluss |
|---|---|---|---|---|---|
| 01 | betrag.py | BETRAG_ZSCORE | 2.0 | Ja | JA |
| 02 | betrag.py | BETRAG_IQR | 1.5 | Nein | JA |
| 03 | betrag.py | KONTO_BETRAG_ANOMALIE | 2.0 | Ja | JA |
| 04 | duplikate.py | NEAR_DUPLICATE | 2.0 | Ja | JA + Beleg-intern + Same-day-Skip |
| 05 | duplikate.py | DOPPELTE_BELEGNUMMER | 2.0 | Ja | JA + Beleg-intern |
| 06 | duplikate.py | BELEG_KREDITOR_DUPLIKAT | 2.5 | Ja | JA + Beleg-intern + same-day-of-month Skip |
| 07 | buchungslogik.py | STORNO | 1.5 | **Nein (v6.2)** | Nein (ist der Storno-Test) |
| 08 | buchungslogik.py | LEERER_BUCHUNGSTEXT | 1.0 | Nein | Nein (**nur PnL-Konten v6.2**) |
| 09 | buchungslogik.py | RECHNUNGSDATUM_PERIODE | 1.5 | Nein | JA |
| 10 | buchungslogik.py | BUCHUNGSTEXT_PERIODE | 1.0 | Nein | JA |
| 11 | kreditor.py | NEUER_KREDITOR_HOCH | 2.5 | Ja | JA |
| 12 | zeitreihe.py | MONATS_ENTWICKLUNG | 1.5 | Nein | JA |
| 13 | zeitreihe.py | FEHLENDE_MONATSBUCHUNG | 1.0 | Nein | JA |

### Storno-Ausschluss-Logik (NEU v6.0)

Tests mit Storno-Ausschluss muessen:
1. VOR der Berechnung alle Zeilen mit _is_storno == True herausfiltern
2. Berechnung NUR auf nicht-Storno-Zeilen durchfuehren
3. Storno-Zeilen behalten KEINE Flags aus diesen Tests

**Warum?** Storno-Buchungen haben naturgemaess:
- Gleichen Betrag wie der Original-Beleg -> NEAR_DUPLICATE falsch-positiv
- Gleiche Belegnummer -> DOPPELTE_BELEGNUMMER falsch-positiv
- Gleichen Kreditor + Belegnummer -> BELEG_KREDITOR_DUPLIKAT falsch-positiv
- Oft hohe Betraege -> BETRAG_ZSCORE/IQR falsch-positiv
- "Fehlende" Monate -> FEHLENDE_MONATSBUCHUNG unsinnig
- Periodenabweichung ist bei Stornos NORMAL -> RECHNUNGSDATUM_PERIODE / BUCHUNGSTEXT_PERIODE falsch-positiv

---

## Berechnungslogik (Kurzreferenz)

### 7. Storno-Erkennung (KORRIGIERT v6.0, AKTUALISIERT v6.2)
```
# engine._prepare() UND Storno.run() verwenden DIESELBE Logik:
Buchungstext enthaelt "storno|stornierung|rueckbuchung|gutschrift.*storn" (case-insensitive)
ODER "korrektur" (word-boundary) NUR bei negativem Betrag (_betrag < 0)
ODER generalumgekehrt ist NICHT leer/NULL/nan/0

Gewicht 1.5, critical=FALSE (v6.2 — erzwingt NICHT mehr Output!)
```
**NICHT:** Generalumgekehrt = "1"/"true"/"J" (DAS WAR DER BUG!)
**NICHT:** "korrektur" pauschal als Storno (v6.1 war uneinheitlich zwischen _prepare() und Storno.run())

> **KRITISCH v6.2:** STORNO hat `critical = False`. Das bedeutet: Storno allein (Score 1.5)
> reicht NICHT fuer Output (threshold=2.0). Nur Buchungen mit mehreren Flags oder
> Score >= threshold erscheinen im Output. Das war das Kernproblem der False-Positives!

### 6. BELEG_KREDITOR_DUPLIKAT (AKTUALISIERT v6.2)
```
Level 1: gleiche Belegnr + Kreditor + Betrag(_abs), >= 4 verschiedene _beleg_ids -> verdaechtig
Level 2: gleicher Kreditor + Betrag + Datum <= 1 Tag (v6.2, vorher 3) -> verdaechtig
         PLUS: same-day-of-month Skip (v6.2) — wenn alle Buchungen am gleichen Kalendertag
               fallen = monatliche Regelzahlung -> KEIN Duplikat
ABER: Beleg-interne Zeilen (gleiche _beleg_id) IGNORIEREN
ABER: Stornos ausschliessen
ABER: Regulaere Kreditoren-Muster ausschliessen
```

### 8. LEERER_BUCHUNGSTEXT (AKTUALISIERT v6.2)
```
Buchungstext leer, generisch (diverse, sonstiges, ...) oder <= 2 Zeichen
NUR fuer PnL-Konten (Ertrag + Aufwand) — Bestand/Kostenrechnung werden IGNORIERT
```

### 13. FEHLENDE_MONATSBUCHUNG (AKTUALISIERT v6.2)
```
Konto fehlt in Monat, wo sonst regelmaessig gebucht
min_quote = 0.5 (v6.2, vorher 0.3) — Konto muss in 50% der Monate aktiv sein
```

### 5. Doppelte Belegnummer (KORRIGIERT v6.0)
```
Belegnr. + Konto + Betrag gleich, >= 5x -> verdaechtig
ABER: Beleg-interne Zeilen (gleiche _beleg_id) IGNORIEREN
ABER: Stornos ausschliessen
```

### 4. Near-Duplicate (KORRIGIERT v6.0, AKTUALISIERT v6.2)
```
Gleicher Betrag + Konto, Datum 1-3 Tage Abstand -> verdaechtig
ABER: Same-day (0 Tage) = Soll/Haben-Gegenbuchung oder Korrektur+Neubuchung -> KEIN Duplikat
ABER: Buchungszeilen desselben Belegs (_beleg_id) sind KEIN Duplikat
ABER: Stornos ausschliessen
```

---

## Abgeleitete Spalten in engine._prepare()

| Spalte | Typ | Berechnung | Wofuer |
|---|---|---|---|
| _betrag | **float64** | parse_german_number_series(betrag) | Rohwert mit Originalvorzeichen |
| _abs | **float64** | _betrag.abs() | Alle statistischen Tests |
| _datum | datetime64 | parse_date_series(datum) | Zeitreihen |
| _kontoklasse | str | kontoklasse(konto_soll) | Getrennte Statistiken |
| _is_storno | bool | generalumgekehrt nicht-leer ODER Text-Keywords | Storno-Ausschluss |
| _beleg_id | str | dvbelegnummer oder belegnummer Fallback | Beleg-Gruppierung |
| _betrag_signed | float64 | compute_signed_betrag(df) | PnL-Charts |
| _score | float64 | Summe Flag-Gewichte | Anomalie-Ranking |

> **WICHTIG float64:** v5.1 nutzte float32 -> Rundungsfehler. v6.0 nutzt float64.

---

## Entfernte/Geaenderte Features

| Feature | Warum entfernt/geaendert | Version |
|---|---|---|
| VELOCITY_ANOMALIE | Diamant hat keinen erfasser | v5.1 |
| float32 fuer _betrag/_abs | Rundungsfehler | v6.0 |
| _kontoklasse() inline in charts.py | Muss aus src/accounting importiert werden | v6.0 |
| gu.isin({"1","true","j"}) | Generalumgekehrt enthaelt DVBelegnummern! | v6.0 |
| STORNO critical=True | Erzwang Output fuer JEDE Storno-Buchung → 34.7% FP | v6.2 |
| Storno.run() eigene Regex | Nicht synchron mit engine._prepare() | v6.2 |
| LEERER_BUCHUNGSTEXT alle Konten | Bestand/Kostenrechnung braucht keinen Buchungstext | v6.2 |
| beleg_kreditor_days=3 | Zu grosses Zeitfenster fuer Level 2 | v6.2 |
| fehlende_buchung_min_quote=0.3 | Zu niedrig, flaggt sporadische Konten | v6.2 |
| RECHNUNGSDATUM_PERIODE ohne Storno-Ausschluss | Stornos haben naturgemaess Periodenabweichung -> FP | v6.2 |
| BUCHUNGSTEXT_PERIODE ohne Storno-Ausschluss | Stornos haben naturgemaess Periodenabweichung -> FP | v6.2 |
| NEAR_DUPLICATE same-day (0 Tage) | Soll/Haben-Gegenbuchungen + Korrekturen am selben Tag -> FP | v6.2 |

---

## Haeufige Fehlerquellen

- **Generalumgekehrt != Boolean**: Feld enthaelt DVBelegnummern, nicht Ja/Nein.
- **Belegnummer != Beleg-ID**: Externe Rechnungsnummer kommt natuerlich mehrfach vor.
- **Storno + andere Tests**: Stornos erzeugen Phantom-Flags -> immer ausschliessen!
- **Kontonummern >= 80000**: Sind Kostenrechnungskonten, NICHT Bestandskonten!
- **float32-Artefakte**: -56710.328125 statt -56710.33 -> float64 verwenden.
- **soll_haben fehlt**: Normal bei Diamant -> Fallback auf Originalvorzeichen.
- **13 nicht 14 Tests**: VELOCITY_ANOMALIE ist entfernt.
- **_kontoklasse() inline**: NICHT in charts.py definieren -> aus accounting.py importieren.
- **critical=True Output-Logik**: Flags mit critical=True erzwingen Output UNABHAENGIG vom Score! Nur setzen wenn wirklich JEDE Buchung mit diesem Flag im Output sein soll. (v6.2: STORNO auf False gesetzt weil Storno allein kein Anomalie-Signal ist.)
- **Storno-Regex synchron halten**: engine._prepare() und Storno.run() MUESSEN dieselbe Regex verwenden! Sonst _is_storno != flag_STORNO.
- **LEERER_BUCHUNGSTEXT nur PnL**: Bestandskonten/Kostenrechnung haben oft keinen sinnvollen Buchungstext -> nur Ertrag+Aufwand flaggen.
- **Same-day-of-month Skip**: Monatliche Regelzahlungen am gleichen Kalendertag sind normal -> BELEG_KREDITOR Level 2 ueberspringen.
- **Trailing Whitespace (Diamant fixed-width)**: Diamant exportiert Textfelder mit trailing Spaces. engine._prepare() stripped diese beim Kategorisieren. NICHT separat strippen — das passiert bereits zentral.

---

## Codierungs-Regeln fuer Copilot

1. **Keine neuen Kontoklassen-Definitionen** — immer src/accounting.py importieren
2. **Neue Tests**: AnomalyTest-Subklasse + get_tests() + TEST_REQUIREMENTS + TEST_CATEGORIES
3. **Nie iterrows()** — alles vektorisiert
4. **Flag-Spalten**: df[f"flag_{name}"] = True/False, nie Listen in Zellen
5. **Deutsche Betraege**: parse_german_number_series() aus src/parser.py
6. **Tests aktualisieren**: Bei jeder Aenderung auch tests/test_engine.py anpassen
7. **13 Tests**: Nicht "14 Tests" schreiben
8. **konto_haben**: Nicht als required behandeln
9. **Charts**: PnL-Charts nutzen _betrag_signed, nicht _betrag oder _abs
10. **Storno-Ausschluss**: Tests die _is_storno-Zeilen nicht rausfiltern produzieren Muell
11. **Beleg-Ebene**: Duplikat-Tests muessen Beleg-interne Zeilen ignorieren
12. **float64**: Immer float64 fuer _betrag/_abs, nie float32
13. **Generalumgekehrt**: Pruefen auf nicht-leer, NICHT auf bestimmte Werte
14. **critical sparsam setzen**: Nur wenn Flag ALLEIN den Output erzwingen soll (unabhaengig Score). STORNO ist NICHT critical (v6.2).
15. **Storno-Regex synchron**: engine._prepare() und Storno.run() muessen identische Regex verwenden
16. **LEERER_BUCHUNGSTEXT nur PnL**: Immer _kontoklasse.isin({"Ertrag", "Aufwand"}) pruefen
17. **Config-Defaults beachten**: beleg_kreditor_days=1, fehlende_buchung_min_quote=0.5, monats_entwicklung_zscore=3.0
18. **Trailing Whitespace**: Kategorische Spalten werden in _prepare() mit .str.strip() bereinigt. Keine zusaetzlichen .strip()-Aufrufe in Tests noetig.
- **IMMER README.MD UND copilot-instructions.md AKTUALISIEREN nach Aenderungen, damit Copilot die neuesten Infos hat!**

---

## Konfigurations-Defaults (src/config.py) — Stand v6.2

| Parameter | Default | Beschreibung |
|---|---|---|
| beleg_kreditor_days | **1** | Level-2-Zeitfenster in Tagen |
| beleg_kreditor_max_group_size | 20 | Max. Gruppengroesse Level 2 |
| beleg_kreditor_regular_pct | 0.15 | Anteil Datenmonate fuer regulaere Muster |
| fehlende_buchung_min_quote | **0.5** | Mindestanteil aktiver Monate |
| monats_entwicklung_zscore | **3.0** | Z-Score-Grenze fuer Monatsausreisser |
| output_threshold | 2.0 | Score-Schwelle fuer Output |
| doppelte_beleg_min_count | 10 | Min. identische Belegnummern-Gruppen |
| konto_betrag_sigma | 3.0 | Sigma-Faktor Konto-Betrag-Anomalie |