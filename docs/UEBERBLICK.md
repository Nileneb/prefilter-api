# Wie funktioniert der Pre-Filter? (einfach erklärt)

> Diese Seite erklärt die ganze App so, dass man sie ohne Vorwissen versteht.
> Die technische Detail-Doku steht in [`BERECHNUNG.md`](BERECHNUNG.md) und im
> [README](../README.md).

## Die Idee in einem Satz

Eine Firma bucht jeden Monat **tausende** Geldbewegungen. Ein Mensch kann nicht
jede einzeln prüfen. Diese App ist wie ein **schneller Helfer, der alle Buchungen
durchliest und nur auf die wenigen „komisch aussehenden" einen Zettel klebt** —
damit ein Prüfer nur diese anschauen muss, statt alles.

Wichtig: Die App **entscheidet nichts** und löscht nichts. Sie sagt nur „schau dir
diese hier mal genauer an". Ein Mensch entscheidet am Ende.

## Das große Bild

Stell dir einen **Lehrer vor, der einen Stapel Hausaufgaben korrigiert**. Er hat
eine Liste von Dingen, auf die er achtet („Name vergessen?", „Datum falsch?",
„zweimal dasselbe abgegeben?"). Jeder gefundene Fehler gibt **Minuspunkte**. Am
Ende legt er die Blätter mit den meisten Minuspunkten nach oben — die schaut er
zuerst an.

Genau das macht die App: Die „Dinge, auf die geachtet wird" heißen hier **Tests**
(15 Stück). Die Minuspunkte heißen **Score**. Buchungen mit hohem Score landen oben.

## Der Weg einer Datei — Schritt für Schritt

```mermaid
flowchart TD
    A["1. Du lädst eine Datei hoch (CSV/Excel)"] --> B["2. Einlesen + Spalten erkennen<br/>(versteht auch Diamant-Export)"]
    B --> C["3. Daten-Check: welche Tests sind sinnvoll?<br/>(fehlende Spalten = Test wird geschont)"]
    C --> D["4. Vorbereiten: Beträge, Datum, Kontoart,<br/>Storno-Erkennung, Beleg-Paare bilden"]
    D --> E["5. Die 15 Detektive prüfen jede Buchung"]
    E --> F["6. Punkte zählen: Score = Summe der Treffer x Gewicht"]
    F --> G["7. Nur Buchungen mit genug Punkten behalten,<br/>nach Punkten sortieren"]
    G --> H["8. Ergebnis anzeigen: Liste, Statistik, Diagramme"]
    H --> I["9. Du sagst der App: richtig (tp) / Fehlalarm (fp)"]
    I -.->|App lernt die Gewichte| F
```

## Die 15 Detektive (Tests)

Jeder Detektiv prüft **eine** Sache. Findet er etwas, klebt er seinen Zettel an die
Buchung. ⭐ = besonders wichtig (so eine Buchung kommt **immer** auf den Prüf-Stapel).

| Detektiv (Test) | Prüft, ob… | Punkte | ⭐ |
| --- | --- | --- | --- |
| BETRAG_ZSCORE | der Betrag viel größer/kleiner ist als sonst üblich | 2.0 | ⭐ |
| BETRAG_IQR | der Betrag ein extremer Ausreißer ist | 1.5 | |
| KONTO_BETRAG_ANOMALIE | der Betrag für **dieses Konto** ungewöhnlich ist | 2.0 | ⭐ |
| NEAR_DUPLICATE | fast dieselbe Buchung kurz hintereinander vorkommt | 2.0 | ⭐ |
| DOPPELTE_BELEGNUMMER | dieselbe Belegnummer mehrfach auftaucht | 2.0 | ⭐ |
| BELEG_KREDITOR_DUPLIKAT | derselbe Lieferant evtl. doppelt bezahlt wurde | 2.5 | ⭐ |
| STORNO | es eine Storno-/Rückbuchung ist (auffällig) | 1.5 | |
| LEERER_BUCHUNGSTEXT | der Text fehlt oder nichtssagend ist („diverse") | 1.0 | |
| RECHNUNGSDATUM_PERIODE | Rechnungsmonat ≠ Buchungsmonat (Periode verschoben) | 1.5 | |
| BUCHUNGSTEXT_PERIODE | im Text ein anderer Monat steht als gebucht | 1.0 | |
| NEUER_KREDITOR_HOCH | ein **neuer** Lieferant gleich einen hohen Betrag bekommt | 2.5 | ⭐ |
| MONATS_ENTWICKLUNG | eine Monatssumme aus der Reihe tanzt | 1.5 | |
| FEHLENDE_MONATSBUCHUNG | eine sonst regelmäßige Buchung in einem Monat fehlt | 1.0 | |
| ISOLATION_ANOMALIE | eine KI sie insgesamt „seltsam" findet (Experiment, standardmäßig aus) | 1.5 | |
| TEXT_KONTO_MATCH | der Buchungstext nicht zur Kontobezeichnung passt | 2.0 | |

## Der Verdachts-Stempel (Score)

- Jeder Detektiv-Zettel gibt **Punkte** (sein „Gewicht").
- Alle Punkte einer Buchung werden **zusammengezählt** → das ist der **Score**.
- Ab **2 Punkten** (oder bei einem ⭐-Treffer) kommt die Buchung auf den Prüf-Stapel.
- Höchstmöglich sind **27 Punkte** (alle Detektive zusammen). Angezeigt werden die
  Top-Treffer, sortiert von „am verdächtigsten" nach unten.

Beispiel: Eine Buchung mit `DOPPELTE_BELEGNUMMER` (2.0) **und** `NEUER_KREDITOR_HOCH`
(2.5) hat **4.5 Punkte** → landet weit oben.

## Welche Konten zählen? (GuV-Filter)

Eine Firma hat verschiedene Konto-Arten. Geprüft werden standardmäßig nur die, wo
„echtes Geschäft" passiert (**GuV-Konten**, Nummern **40000–79999** = Einnahmen +
Ausgaben). Konten für z. B. Bankguthaben (unter 40000) oder interne Verrechnung
(ab 80000) werden übersprungen — dort wären die Tests sinnlos.

Das kannst du in der App umstellen („Konten-Bereich" / „Alle Konten einbeziehen").
Es gilt **für alle Detektive gleich**.

## Die App lernt dazu

In der Ergebnis-Liste trägst du bei jeder Buchung ein:
- **tp** = „richtig erkannt" (echte Auffälligkeit)
- **fp** = „Fehlalarm" (war harmlos)
- **unsure** = „weiß nicht"

Wenn genug Bewertungen da sind (500), kann die App die **Punkte (Gewichte)
nachjustieren**: Detektive, die oft danebenliegen, bekommen weniger Gewicht;
zuverlässige mehr. Du kannst die Gewichte auch **selbst per Schieberegler** setzen.

## Spezialfall: Der Isolation Forest (und warum Regeln oft besser sind)

Der 15. Detektiv `ISOLATION_ANOMALIE` arbeitet anders als die anderen 14. Er ist
eine **KI**, die nicht weiß, was „falsch" heißt — er sucht nur das **statistisch
Ungewöhnliche**.

**Wie er funktioniert (Bild):** Stell dir vor, du sollst in einer Menschenmenge die
„komischen" Leute finden, ohne zu wissen, was komisch ist. Du ziehst **zufällige
Trennlinien** („alle über 1,90 m nach links") und schaust, wer nach **ganz wenigen
Schnitten allein** dasteht. Eine Person mit Hut, grünen Haaren und Einrad ist schnell
isoliert — eine Durchschnittsperson erst nach vielen Schnitten. **Schnell isoliert =
verdächtig.** Genau das macht der Algorithmus mit vielen Zufallsbäumen, hier auf den
Merkmalen Betragshöhe, Tag-im-Monat, Wochentag und KI-Textmerkmalen.

**Die naheliegende These:** „Alle anderen Tests könnte man auch mit normaler
SQL-Datenbank bauen — also ist der Isolation Forest die eigentlich wichtige Methode."
Das ist verständlich gedacht, aber **falsch herum**:

- **„In SQL machbar" heißt nicht „schlechter".** Die wirksamsten Indikatoren in der
  Buchhaltung sind gerade die „langweiligen" Regeln, weil sie **Fachwissen** kodieren
  — *so* entstehen Fehler: doppelte Zahlung, neuer Lieferant + sofort hoher Betrag,
  Periodenverschiebung. Diese Treffer sind präzise und **erklärbar**.
- **Betrug ist oft KEIN Ausreißer.** Eine clever doppelt gebuchte Rechnung sieht
  *völlig normal* aus (gleicher Betrag, gleicher Lieferant) → der Forest übersieht sie,
  die Regel `BELEG_KREDITOR_DUPLIKAT` fängt sie. Eine ganze Fehlerklasse ist für den
  Forest **strukturell unsichtbar**.
- **Feste Quote statt Sinn.** Der Forest markiert immer einen festen Anteil (~2 %) als
  „Anomalie" — auch auf sauberen Daten. Die einmalige Jahres-Versicherung ist
  „statistisch selten", aber völlig korrekt → **Fehlalarm**.
- **Nicht erklärbar.** Er sagt „komisch", aber nicht *warum*. Damit kann ein Prüfer
  wenig anfangen. Außerdem braucht er viel Daten (deshalb der ≥1.000-Buchungen-Riegel)
  und Feintuning.

**Die ehrliche Einordnung — es ist kein Entweder-oder:**

| | Regeln (die 14) | Isolation Forest |
| --- | --- | --- |
| Findet | **bekannte** Fehlermuster | **unbekannte** Kombinationen |
| Präzision | hoch, erklärbar | niedrig, viele Fehlalarme |
| Fachwissen | eingebaut | keins |
| Doppelte Zahlungen | ✅ fängt sie | ❌ übersieht sie oft |

Die **Regeln sind das Arbeitspferd**, der Forest ein optionales **Zusatznetz** für
„etwas, an das niemand gedacht hat". Genau deshalb: 14 gezielte Tests + 1 Catch-all,
der standardmäßig **aus** ist.

> Kleine Korrektur zum „alles geht mit SQL": Zwei Tests (`NEAR_DUPLICATE`,
> `TEXT_KONTO_MATCH`) sind **kein** reines SQL — sie nutzen dieselbe KI-Vektor-Technik
> (Embeddings) wie der Forest, um *Bedeutung* von Texten zu vergleichen. Die echte
> Trennlinie ist „Fachregeln + gezielte KI" gegen „blinder statistischer Catch-all".

## Die Knöpfe und Tabs in der App

- **Datei hochladen** + (optional) **Webhook-URL** (Ergebnis automatisch weiterschicken).
- **📋 Datenqualitäts-Check**: zeigt, welche Tests laufen, eingeschränkt oder geblockt
  sind — plus ℹ️-Infos zum Datenmodell (z. B. „Diamant hat kein Gegenkonto, das ist normal").
- **⚙️ Erweiterte Einstellungen**: Schieberegler für Empfindlichkeit + Konten-Bereich.
- **🔧 Test-Konfiguration & Gewichte**: jeden Detektiv an/aus + Gewicht; Knöpfe
  „Defaults" und „Gelernte Gewichte laden".
- **▶️ Analyse starten** / **⛔ Abbrechen**.
- Ergebnis-Tabs:
  - **Ergebnis**: kurze Zusammenfassung.
  - **Verdächtige Buchungen**: die Tabelle + dein tp/fp-Feedback.
  - **📜 Live-Log**: was die App gerade macht (live).
  - **📊 Visualisierungen**: fertige Diagramme (Score-Verteilung, Top-Konten …).
  - **🔬 Eigene Visualisierung**: eigenes Diagramm aus beliebigen Spalten bauen.
  - **📁 History**: frühere Läufe ansehen.
  - **📈 Feedback-Stats**: wie gut die Detektive laut deinen Bewertungen sind.

## Hinter den Kulissen (für Neugierige)

- **Einlesen & Spalten erkennen** (`parser.py`): erkennt deutsche Zahlen (1.234,56),
  verschiedene Datumsformate und ordnet Diamant-Spalten den richtigen Namen zu.
- **Beleg-Paare** (`engine.find_counterpart_rows`): Diamant speichert pro Buchung nur
  **ein** Konto; das Gegenkonto steht in einer zweiten Zeile mit derselben Beleg-ID.
  Die App fügt diese Paare gedanklich zusammen.
- **Vorzeichen** (`accounting.py`): rechnet Soll/Haben in „+ Einnahme / − Ausgabe" um.
- **KI-Textvergleich** (`embeddings.py`): wandelt Texte in Zahlen-Vektoren um, um
  Ähnlichkeit zu messen (für NEAR_DUPLICATE und TEXT_KONTO_MATCH). Es ist ein
  **echtes** Modell — `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers,
  384 Dimensionen, 50+ Sprachen inkl. Deutsch), per ENV `EMBEDDING_MODEL` austauschbar.
  **Kein Dummy/Fake:** ist `sentence-transformers` nicht installiert, werden die zwei
  betroffenen Tests sauber übersprungen bzw. fallen auf exakten Textvergleich zurück —
  es werden **keine** Pseudo-Vektoren erfunden.
- **Lieferanten aufräumen** (`kreditor_clustering.py`): erkennt, dass „Müller GmbH"
  und „Mueller G.m.b.H" derselbe Lieferant sind.
- **History** (`history.py`): speichert jeden Lauf und vergleicht mit dem letzten.
- **Webhook** (`webhook.py`): schickt das Ergebnis (gekürzt) an einen Langdock-Agenten.

## Schnell oder normal? (Worker)

- Kleine Dateien: die App rechnet **direkt** (sequentiell).
- Große Dateien (ab 100.000 Zeilen): die Arbeit wird auf **mehrere Helfer verteilt**
  (Celery-Worker mit Redis) und parallel gerechnet — das Ergebnis ist identisch.
- Ohne Redis läuft alles im **lokalen Fallback-Modus** direkt im Browser-Prozess.

## Zwei Türen hinein

1. **Web-Oberfläche** (Gradio, Port 7864): die bunten Knöpfe für Menschen.
2. **Roboter-Schnittstelle** (FastAPI REST, Port 8000): für andere Programme
   (`POST /api/jobs`, Status abfragen, Live-Logs). `GET /health` sagt „ich lebe",
   `GET /healthz` prüft zusätzlich die Redis-Verbindung.
