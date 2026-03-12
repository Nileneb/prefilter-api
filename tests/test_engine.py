"""
Unit & integration tests for the Buchungs-Anomalie Pre-Filter.
Run with: python -m pytest tests/ -v

Stand nach Phase 0 / 4.2-Refactor:
- 15 Tests entfernt (SELTENE_KONTIERUNG, WOCHENENDE, MONATSENDE, QUARTALSENDE,
  AUSSERHALB_GESCHAEFTSZEIT, BENFORD_*, RUNDER_BETRAG, ERFASSER_ANOMALIE,
  SPLIT_VERDACHT, SCHWELLENWERT_CLUSTER, BELEG_LUECKE, SOLL_GLEICH_HABEN,
  TEXT_KREDITOR_MISMATCH, FUZZY_KREDITOR)
- 10 Tests verbleiben und sind vollständig vektorisiert
- Boolean-Spalten statt Python-Listen: engine.df["flag_X"] statt engine.df["_flags"]
"""

from datetime import datetime

import pandas as pd

from src.parser import parse_german_number, parse_date, map_columns, read_upload
from src.engine import AnomalyEngine


# ══════════════════════════════════════════════════════════════════════════════
# Parser-Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestParseGermanNumber:
    def test_simple_integer(self):
        assert parse_german_number("1234") == 1234.0

    def test_german_decimal(self):
        assert parse_german_number("1,23") == 1.23

    def test_german_thousands(self):
        assert parse_german_number("1.234,56") == 1234.56

    def test_english_format(self):
        assert parse_german_number("1,234.56") == 1234.56

    def test_currency_symbol(self):
        assert parse_german_number("1.234,56€") == 1234.56

    def test_negative(self):
        assert parse_german_number("-500,00") == -500.0

    def test_none(self):
        assert parse_german_number(None) == 0.0

    def test_empty(self):
        assert parse_german_number("") == 0.0

    def test_thousand_separator_comma(self):
        # "1,234" with 3 digits after comma → thousand separator
        assert parse_german_number("1,234") == 1234.0

    def test_spaces(self):
        assert parse_german_number(" 1 234,56 ") == 1234.56

    def test_dollar(self):
        assert parse_german_number("$5000") == 5000.0


class TestParseDate:
    def test_iso(self):
        d = parse_date("2024-01-15")
        assert d == datetime(2024, 1, 15)

    def test_iso_with_time(self):
        d = parse_date("2024-01-15T14:30:00")
        assert d == datetime(2024, 1, 15, 14, 30, 0)

    def test_german(self):
        d = parse_date("15.01.2024")
        assert d == datetime(2024, 1, 15)

    def test_german_with_time(self):
        d = parse_date("15.01.2024 14:30")
        assert d == datetime(2024, 1, 15, 14, 30, 0)

    def test_german_short(self):
        d = parse_date("15.01.24")
        assert d == datetime(2024, 1, 15)

    def test_none(self):
        assert parse_date(None) is None

    def test_empty(self):
        assert parse_date("") is None

    def test_invalid(self):
        assert parse_date("not-a-date") is None


class TestMapColumns:
    def test_basic_mapping(self):
        df = pd.DataFrame({"Betrag": [1], "Datum": ["2024-01-01"], "BuchungsText": ["test"]})
        mapped = map_columns(df)
        assert "betrag" in mapped.columns
        assert "datum" in mapped.columns
        assert "buchungstext" in mapped.columns

    def test_umlaut_mapping(self):
        df = pd.DataFrame({"Beträge": [1]})  # not exact match
        mapped = map_columns(df)
        assert len(mapped.columns) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# Engine-Tests
# ══════════════════════════════════════════════════════════════════════════════

def _make_df(**kwargs) -> pd.DataFrame:
    """Hilfs-Funktion: DataFrame mit sinnvollen Defaults."""
    defaults = {
        "datum":        ["2024-01-15"],
        "betrag":       ["1000,00"],
        "konto_soll":   ["4711"],
        "konto_haben":  ["1200"],
        "buchungstext": ["Test"],
        "belegnummer":  ["001"],
        "kostenstelle": [""],
        "kreditor":     [""],
        "erfasser":     ["TestUser"],
    }
    defaults.update(kwargs)
    n = max(len(v) for v in defaults.values())
    for k, v in defaults.items():
        if len(v) < n:
            defaults[k] = v * n
    return pd.DataFrame(defaults)


class TestEngineZScore:
    def test_extreme_value_flagged(self):
        betraege = ["100,00"] * 99 + ["100000,00"]
        df = _make_df(
            datum=["2024-01-15"] * 100,
            betrag=betraege,
            konto_soll=["4711"] * 100,
            konto_haben=["1200"] * 100,
            buchungstext=["Normal"] * 99 + ["Extrem"],
            belegnummer=[f"{i:04d}" for i in range(100)],
            erfasser=["User"] * 100,
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t01_zscore()
        assert engine.flag_counts["BETRAG_ZSCORE"] >= 1


class TestEngineNearDuplicate:
    def test_counts_bookings_not_pairs(self):
        """4 identische Buchungen → 4 Bookings flaggen, nicht 6 Paare."""
        df = _make_df(
            datum=["2024-01-15"] * 4,
            betrag=["1000,00"] * 4,
            konto_soll=["4711"] * 4,
            konto_haben=["1200"] * 4,
            buchungstext=["Dup"] * 4,
            belegnummer=["001", "002", "003", "004"],
            erfasser=["User"] * 4,
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t06_near_duplicate()
        assert engine.flag_counts["NEAR_DUPLICATE"] == 4

    def test_missing_one_date_not_flagged(self):
        """Nur eine Buchung ohne Datum → nicht flaggen."""
        df = _make_df(
            datum=["2024-01-15", ""],
            betrag=["1000,00", "1000,00"],
            konto_soll=["4711", "4711"],
            konto_haben=["1200", "1200"],
            buchungstext=["A", "B"],
            belegnummer=["001", "002"],
            erfasser=["User", "User"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t06_near_duplicate()
        assert engine.flag_counts["NEAR_DUPLICATE"] == 0

    def test_both_missing_dates_flagged(self):
        """Beide Buchungen ohne Datum → beide flaggen."""
        df = _make_df(
            datum=["", ""],
            betrag=["1000,00", "1000,00"],
            konto_soll=["4711", "4711"],
            konto_haben=["1200", "1200"],
            buchungstext=["A", "B"],
            belegnummer=["001", "002"],
            erfasser=["User", "User"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t06_near_duplicate()
        assert engine.flag_counts["NEAR_DUPLICATE"] == 2


class TestEngineStorno:
    def test_gutschrift_normal_not_flagged(self):
        """Normale Gutschrift mit kleinem Betrag darf nicht flaggen."""
        df = _make_df(
            datum=["2024-01-15"] * 50,
            betrag=["100,00"] * 49 + ["50,00"],
            konto_soll=["4711"] * 50,
            konto_haben=["1200"] * 50,
            buchungstext=["Normal"] * 49 + ["Gutschrift regulär"],
            belegnummer=[f"{i:04d}" for i in range(50)],
            erfasser=["User"] * 50,
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t15_storno()
        # Boolean-Spalte statt _flags-Liste (neue API nach Phase-0-Refactor)
        assert not engine.df.iloc[-1]["flag_STORNO"]

    def test_storno_keyword_always_flagged(self):
        """Explizites Storno-Keyword muss immer flaggen."""
        df = _make_df(
            betrag=["50,00"],
            buchungstext=["Storno Rechnung 123"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t15_storno()
        assert engine.flag_counts["STORNO"] == 1


class TestEngineDoppelteBelegnummer:
    def test_duplicate_flagged(self):
        """Doppelte Belegnummern → beide Zeilen flaggen."""
        df = _make_df(
            datum=["2024-01-15", "2024-01-16"],
            betrag=["100,00", "200,00"],
            belegnummer=["001", "001"],
            konto_soll=["4711", "4720"],
            konto_haben=["1200", "1200"],
            buchungstext=["A", "B"],
            erfasser=["User", "User"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t13_doppelte_belegnummer()
        assert engine.flag_counts["DOPPELTE_BELEGNUMMER"] == 2


class TestEngineBelegKreditorDuplikat:
    def test_same_beleg_same_kreditor(self):
        """Gleiche Belegnr. + gleicher Kreditor → flaggen."""
        df = _make_df(
            datum=["2024-01-15", "2024-01-16"],
            betrag=["500,00", "700,00"],
            belegnummer=["INV-001", "INV-001"],
            kreditor=["Lieferant A", "Lieferant A"],
            konto_soll=["4711", "4720"],
            konto_haben=["1200", "1200"],
            buchungstext=["Einkauf", "Einkauf 2"],
            erfasser=["User", "User"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t14_beleg_kreditor_duplikat()
        assert engine.flag_counts["BELEG_KREDITOR_DUPLIKAT"] == 2

    def test_same_kreditor_amount_within_7days(self):
        """Gleicher Kreditor + Betrag + ≤7 Tage → flaggen."""
        df = _make_df(
            datum=["2024-01-15", "2024-01-20"],
            betrag=["1000,00", "1000,00"],
            belegnummer=["INV-001", "INV-002"],
            kreditor=["Lieferant B", "Lieferant B"],
            konto_soll=["4711", "4711"],
            konto_haben=["1200", "1200"],
            buchungstext=["Rechnung", "Rechnung"],
            erfasser=["User", "User"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t14_beleg_kreditor_duplikat()
        assert engine.flag_counts["BELEG_KREDITOR_DUPLIKAT"] == 2

    def test_different_kreditors_not_flagged(self):
        """Verschiedene Kreditoren → nicht flaggen."""
        df = _make_df(
            datum=["2024-01-15", "2024-01-16"],
            betrag=["500,00", "500,00"],
            belegnummer=["INV-001", "INV-002"],
            kreditor=["Lieferant A", "Lieferant B"],
            konto_soll=["4711", "4711"],
            konto_haben=["1200", "1200"],
            buchungstext=["Einkauf", "Einkauf"],
            erfasser=["User", "User"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t14_beleg_kreditor_duplikat()
        assert engine.flag_counts["BELEG_KREDITOR_DUPLIKAT"] == 0


class TestEngineLeererBuchungstext:
    def test_empty_text_flagged(self):
        df = _make_df(buchungstext=[""])
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t21_leerer_buchungstext()
        assert engine.flag_counts["LEERER_BUCHUNGSTEXT"] == 1

    def test_generic_text_flagged(self):
        df = _make_df(buchungstext=["diverse"])
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t21_leerer_buchungstext()
        assert engine.flag_counts["LEERER_BUCHUNGSTEXT"] == 1

    def test_short_text_flagged(self):
        df = _make_df(buchungstext=["ab"])
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t21_leerer_buchungstext()
        assert engine.flag_counts["LEERER_BUCHUNGSTEXT"] == 1

    def test_proper_text_not_flagged(self):
        df = _make_df(buchungstext=["Bezahlung Rechnung 12345 vom 15.01.2024"])
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t21_leerer_buchungstext()
        assert engine.flag_counts["LEERER_BUCHUNGSTEXT"] == 0


class TestEngineVelocityAnomalie:
    def test_spike_flagged(self):
        """Kreditor mit plötzlichem Monats-Spike → flaggen."""
        dates     = []
        kreditors = []
        for m in range(1, 6):
            for _ in range(2):
                dates.append(f"2024-{m:02d}-15")
                kreditors.append("Lieferant X")
        for _ in range(10):
            dates.append("2024-06-15")
            kreditors.append("Lieferant X")

        n  = len(dates)
        df = _make_df(
            datum=dates,
            betrag=["100,00"] * n,
            konto_soll=["4711"] * n,
            konto_haben=["1200"] * n,
            buchungstext=["Bestellung"] * n,
            belegnummer=[f"{i:04d}" for i in range(n)],
            kreditor=kreditors,
            erfasser=["User"] * n,
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t22_velocity_anomalie()
        assert engine.flag_counts["VELOCITY_ANOMALIE"] >= 1


# ── Phase-2-Tests ─────────────────────────────────────────────────────────────

class TestEngineRechnungsdatumPeriode:
    def test_different_period_flagged(self):
        """Rechnungsdatum in anderem Monat als Buchungsdatum → flaggen."""
        df = _make_df(
            datum=["2024-03-15"],
            rechnungsdatum=["2024-01-10"],  # Januar ≠ März
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t23_rechnungsdatum_periode()
        assert engine.flag_counts["RECHNUNGSDATUM_PERIODE"] == 1

    def test_same_period_not_flagged(self):
        """Gleicher Monat → nicht flaggen."""
        df = _make_df(
            datum=["2024-03-15"],
            rechnungsdatum=["2024-03-01"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t23_rechnungsdatum_periode()
        assert engine.flag_counts["RECHNUNGSDATUM_PERIODE"] == 0

    def test_missing_rechnungsdatum_not_flagged(self):
        """Kein Rechnungsdatum vorhanden → nicht flaggen."""
        df = _make_df(datum=["2024-03-15"])  # rechnungsdatum leer
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t23_rechnungsdatum_periode()
        assert engine.flag_counts["RECHNUNGSDATUM_PERIODE"] == 0


class TestEngineBuchungstextPeriode:
    def test_period_mismatch_flagged(self):
        """Text enthält '01/2024', Buchung ist in März 2024 → flaggen."""
        df = _make_df(
            datum=["2024-03-15"],
            buchungstext=["Rechnung 01/2024 Wartungsvertrag"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t24_buchungstext_periode()
        assert engine.flag_counts["BUCHUNGSTEXT_PERIODE"] == 1

    def test_matching_period_not_flagged(self):
        """Text enthält '03/2024', Buchung ist in März 2024 → nicht flaggen."""
        df = _make_df(
            datum=["2024-03-15"],
            buchungstext=["Rechnung 03/2024 Wartungsvertrag"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t24_buchungstext_periode()
        assert engine.flag_counts["BUCHUNGSTEXT_PERIODE"] == 0

    def test_no_period_in_text_not_flagged(self):
        """Text ohne Periodenangabe → nicht flaggen."""
        df = _make_df(
            datum=["2024-03-15"],
            buchungstext=["Reguläre Wartungsrechnung Heizungsanlage"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t24_buchungstext_periode()
        assert engine.flag_counts["BUCHUNGSTEXT_PERIODE"] == 0


class TestEngineMonatsEntwicklung:
    def test_spike_month_flagged(self):
        """Ausreißer-Monat auf GuV-Konto → alle Buchungen dieses Monats flaggen."""
        # 9 normale Monate (Z = N/sqrt(N+1) = 9/√10 ≈ 2.85 > 2.5)
        dates    = []
        betraege = []
        for m in range(1, 10):           # Monate 1–9 normal
            for _ in range(3):
                dates.append(f"2024-{m:02d}-15")
                betraege.append("1000,00")
        for _ in range(3):               # Monat 11 als Ausreißer
            dates.append("2024-11-15")
            betraege.append("50000,00")

        n  = len(dates)
        df = _make_df(
            datum=dates,
            betrag=betraege,
            konto_soll=["65000"] * n,   # Aufwandskonto
            konto_haben=["1200"] * n,
            buchungstext=["Buchung"] * n,
            belegnummer=[f"{i:04d}" for i in range(n)],
            erfasser=["User"] * n,
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t25_monats_entwicklung()
        assert engine.flag_counts["MONATS_ENTWICKLUNG"] >= 1

    def test_non_pnl_account_not_flagged(self):
        """Bilanzkonto (1200) wird nicht analysiert."""
        n  = 6
        df = _make_df(
            datum=[f"2024-{m:02d}-15" for m in range(1, 7)],
            betrag=["1000,00"] * 5 + ["50000,00"],
            konto_soll=["1200"] * n,   # Bilanzkonto → kein GuV
            konto_haben=["1000"] * n,
            buchungstext=["Buchung"] * n,
            belegnummer=[f"{i:04d}" for i in range(n)],
            erfasser=["User"] * n,
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t25_monats_entwicklung()
        assert engine.flag_counts["MONATS_ENTWICKLUNG"] == 0


class TestEngineFehlendeMonatsbuchung:
    def test_missing_month_neighbors_flagged(self):
        """Konto bucht regelmäßig, Monat fehlt → Nachbarmonate werden geflaggt."""
        # 5 Monate (1,2,4,5,6) × 3 Buchungen = 15 gesamt (>= Mindest-Datenmenge)
        # Monat 3 fehlt → Monate 2 und 4 sind Nachbarn → werden geflaggt
        dates  = []
        konten = []
        for m in [1, 2, 4, 5, 6]:
            for _ in range(3):
                dates.append(f"2024-{m:02d}-15")
                konten.append("4711")

        df = _make_df(
            datum=dates,
            betrag=["500,00"] * len(dates),
            konto_soll=konten,
            konto_haben=["1200"] * len(dates),
            buchungstext=["Buchung"] * len(dates),
            belegnummer=[f"{i:04d}" for i in range(len(dates))],
            erfasser=["User"] * len(dates),
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t26_fehlende_monatsbuchung()
        assert engine.flag_counts["FEHLENDE_MONATSBUCHUNG"] >= 1

    def test_no_regular_account_not_flagged(self):
        """Konto bucht nur sporadisch → kein Flag."""
        # Nur in 2 von 6 Monaten → nicht regelmäßig
        df = _make_df(
            datum=["2024-01-15", "2024-06-15"],
            betrag=["500,00", "500,00"],
            konto_soll=["4711", "4711"],
            konto_haben=["1200", "1200"],
            buchungstext=["Buchung", "Buchung"],
            belegnummer=["001", "002"],
            erfasser=["User", "User"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t26_fehlende_monatsbuchung()
        assert engine.flag_counts["FEHLENDE_MONATSBUCHUNG"] == 0

    def test_short_timespan_not_flagged(self):
        """Datensatz mit < 6 Monaten Zeitspanne → zu kurz, kein Flag."""
        dates  = []
        konten = []
        for m in [1, 2, 4, 5]:  # 5 Monate Spanne, aber nur 4 belegt → < 6
            for _ in range(3):
                dates.append(f"2024-{m:02d}-15")
                konten.append("4711")
        df = _make_df(
            datum=dates,
            betrag=["500,00"] * len(dates),
            konto_soll=konten,
            konto_haben=["1200"] * len(dates),
            buchungstext=["Buchung"] * len(dates),
            belegnummer=[f"{i:04d}" for i in range(len(dates))],
            erfasser=["User"] * len(dates),
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t26_fehlende_monatsbuchung()
        assert engine.flag_counts["FEHLENDE_MONATSBUCHUNG"] == 0

    def test_12_months_missing_7_flagged(self):
        """12 Monate, Konto fehlt in Monat 7 → Nachbarn (6,8) werden geflaggt."""
        dates  = []
        konten = []
        for m in range(1, 13):
            if m == 7:
                continue
            for _ in range(2):
                dates.append(f"2024-{m:02d}-15")
                konten.append("4711")
        df = _make_df(
            datum=dates,
            betrag=["500,00"] * len(dates),
            konto_soll=konten,
            konto_haben=["1200"] * len(dates),
            buchungstext=["Buchung"] * len(dates),
            belegnummer=[f"{i:04d}" for i in range(len(dates))],
            erfasser=["User"] * len(dates),
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t26_fehlende_monatsbuchung()
        assert engine.flag_counts["FEHLENDE_MONATSBUCHUNG"] >= 2  # Monat 6 + 8

    def test_sparse_konto_not_regular(self):
        """Konto nur in 2 von 12 Monaten → nicht regelmäßig, kein Flag."""
        dates  = []
        konten = []
        # Hintergrund-Konto für genügend Datenmenge
        for m in range(1, 13):
            for _ in range(2):
                dates.append(f"2024-{m:02d}-15")
                konten.append("5000")
        # Sparse Konto: nur 2 von 12 Monaten
        for m in [3, 9]:
            dates.append(f"2024-{m:02d}-15")
            konten.append("4711")
        df = _make_df(
            datum=dates,
            betrag=["500,00"] * len(dates),
            konto_soll=konten,
            konto_haben=["1200"] * len(dates),
            buchungstext=["Buchung"] * len(dates),
            belegnummer=[f"{i:04d}" for i in range(len(dates))],
            erfasser=["User"] * len(dates),
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t26_fehlende_monatsbuchung()
        # 4711 ist nicht regulär (2 < max(3, 12*0.6=7.2) → nicht geflaggt)
        # Nur wenn 5000 regulär genug ist und selbst Lücken hat
        flagged_4711 = engine.df[
            (engine.df["konto_soll"] == "4711") & engine.df["flag_FEHLENDE_MONATSBUCHUNG"]
        ]
        assert len(flagged_4711) == 0

    def test_consecutive_missing_months_detected(self):
        """Zwei aufeinanderfolgende fehlende Monate → alle Nachbarn geflaggt."""
        dates  = []
        konten = []
        # 10 Monate, Monate 5+6 fehlen komplett
        for m in [1, 2, 3, 4, 7, 8, 9, 10]:
            for _ in range(3):
                dates.append(f"2024-{m:02d}-15")
                konten.append("4711")
        df = _make_df(
            datum=dates,
            betrag=["500,00"] * len(dates),
            konto_soll=konten,
            konto_haben=["1200"] * len(dates),
            buchungstext=["Buchung"] * len(dates),
            belegnummer=[f"{i:04d}" for i in range(len(dates))],
            erfasser=["User"] * len(dates),
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t26_fehlende_monatsbuchung()
        # Monate 4 und 7 sind Nachbarn der Lücke → deren Buchungen geflaggt
        assert engine.flag_counts["FEHLENDE_MONATSBUCHUNG"] >= 6  # 3+3

    def test_all_months_present_no_flag(self):
        """Konto bucht in allen Monaten → kein Flag."""
        dates  = []
        konten = []
        for m in range(1, 13):
            for _ in range(2):
                dates.append(f"2024-{m:02d}-15")
                konten.append("4711")
        df = _make_df(
            datum=dates,
            betrag=["500,00"] * len(dates),
            konto_soll=konten,
            konto_haben=["1200"] * len(dates),
            buchungstext=["Buchung"] * len(dates),
            belegnummer=[f"{i:04d}" for i in range(len(dates))],
            erfasser=["User"] * len(dates),
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t26_fehlende_monatsbuchung()
        assert engine.flag_counts["FEHLENDE_MONATSBUCHUNG"] == 0


class TestEngineOutputThreshold:
    def test_low_score_not_in_output(self):
        """Buchungen mit Score < 2.0 erscheinen nicht im Output (außer critical)."""
        df = _make_df(
            datum=[f"2024-{m:02d}-{d:02d}" for m, d in
                   [(1,5),(2,10),(3,15),(4,20),(5,5),(6,10),(7,15),(8,20),
                    (9,5),(10,10),(11,15),(12,20),(1,8),(2,12),(3,18),
                    (4,22),(5,8),(6,12),(7,18),(8,22)]],
            betrag=[f"{(i+1)*137},00" for i in range(20)],
            konto_soll=[f"4{700+i}" for i in range(20)],
            konto_haben=[f"1{200+i}" for i in range(20)],
            buchungstext=[f"Spezifische Buchung Nummer {i+1:04d} Detail" for i in range(20)],
            belegnummer=[f"{i:04d}" for i in range(20)],
            kreditor=[f"Lieferant {chr(65 + i % 20)}" for i in range(20)],
            erfasser=["UserAlpha"] * 20,
        )
        engine = AnomalyEngine(df)
        result  = engine.run()
        total   = result["statistics"]["total_input"]
        suspicious = result["statistics"]["total_output"]
        ratio   = suspicious / total * 100 if total > 0 else 0
        assert ratio < 50, f"Ratio {ratio:.1f}% zu hoch für normale Daten"

    def test_critical_flag_always_in_output(self):
        """Buchungen mit Critical-Flag erscheinen auch wenn Score < OUTPUT_THRESHOLD."""
        # STORNO hat Gewicht 1.5 < OUTPUT_THRESHOLD (2.0) → nur via Critical-Pfad im Output
        df = _make_df(
            betrag=["100,00"],
            buchungstext=["Storno Rechnung 001"],
        )
        engine = AnomalyEngine(df)
        result = engine.run()
        assert result["statistics"]["total_output"] >= 1


class TestEngineFullRun:
    def test_full_run_succeeds(self):
        """Integration: Engine läuft ohne Crash durch."""
        df = _make_df(
            datum=["2024-01-15"] * 10,
            betrag=[f"{i * 100}" for i in range(1, 11)],
            konto_soll=["4711"] * 10,
            konto_haben=["1200"] * 10,
            buchungstext=[f"Buchung {i}" for i in range(10)],
            belegnummer=[f"{i:04d}" for i in range(10)],
            erfasser=["User"] * 10,
        )
        engine = AnomalyEngine(df)
        result = engine.run()
        assert "verdaechtige_buchungen" in result
        assert "statistics" in result
        assert result["statistics"]["total_input"] == 10

    def test_empty_df(self):
        """Edge case: leerer DataFrame."""
        df = pd.DataFrame({
            "datum": [], "betrag": [], "konto_soll": [],
            "konto_haben": [], "buchungstext": [], "belegnummer": [],
            "kostenstelle": [], "kreditor": [], "erfasser": [],
        })
        engine = AnomalyEngine(df)
        result = engine.run()
        assert result["statistics"]["total_input"] == 0

    def test_all_test_methods_called(self):
        """Alle 14 Tests liefern einen flag_counts-Eintrag."""
        df = _make_df(
            datum=["2024-01-15"] * 10,
            betrag=[f"{i * 100}" for i in range(1, 11)],
            konto_soll=["4711"] * 10,
            konto_haben=["1200"] * 10,
            buchungstext=[f"Buchung {i}" for i in range(10)],
            belegnummer=[f"{i:04d}" for i in range(10)],
            erfasser=["User"] * 10,
        )
        engine  = AnomalyEngine(df)
        result  = engine.run()
        fc      = result["statistics"]["flag_counts"]

        expected_flags = {
            "BETRAG_ZSCORE", "BETRAG_IQR", "NEAR_DUPLICATE",
            "DOPPELTE_BELEGNUMMER", "BELEG_KREDITOR_DUPLIKAT",
            "STORNO", "NEUER_KREDITOR_HOCH", "KONTO_BETRAG_ANOMALIE",
            "LEERER_BUCHUNGSTEXT", "VELOCITY_ANOMALIE",
            "RECHNUNGSDATUM_PERIODE", "BUCHUNGSTEXT_PERIODE",
            "MONATS_ENTWICKLUNG", "FEHLENDE_MONATSBUCHUNG",
        }
        assert expected_flags == set(fc.keys()), (
            f"Fehlend: {expected_flags - set(fc.keys())}, "
            f"Überschuss: {set(fc.keys()) - expected_flags}"
        )
        assert "stammdaten_report"       in result
        assert "fuzzy_kreditor_matches"  in result["stammdaten_report"]


# ── v7-Tests (Diamant-Integration) ───────────────────────────────────────────

class TestParserDiamantAliases:
    """Testet Diamant-spezifische Spalten-Mappings."""

    def test_diamant_column_mapping(self):
        """Diamant-Spaltennamen werden korrekt auf kanonische Namen gemappt."""
        df = pd.DataFrame({
            "Belegdatum": ["2024-01-15"],
            "FiBuBetrag": ["1000,00"],
            "Kontonummer": ["4711"],
            "Buchungstext": ["Test"],
            "Belegnummer": ["001"],
            "Bezeichnung": ["Lieferant A"],
            "Klasse": ["K"],
            "Generalumgekehrt": ["0"],
        })
        mapped = map_columns(df)
        assert "datum" in mapped.columns
        assert "betrag" in mapped.columns
        assert "konto_soll" in mapped.columns
        assert "kreditor" in mapped.columns
        assert "klasse" in mapped.columns
        assert "generalumgekehrt" in mapped.columns

    def test_pipe_delimited_csv(self, tmp_path):
        """Pipe-delimitierte CSV wird korrekt gelesen."""
        csv_content = "Belegdatum|FiBuBetrag|Kontonummer|Buchungstext\n2024-01-15|1000.00|4711|Test\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content, encoding="utf-8")
        df = read_upload(str(csv_file))
        assert len(df) == 1
        assert len(df.columns) >= 4

    def test_null_string_handling(self, tmp_path):
        """'NULL'-Strings werden als NaN/NA eingelesen."""
        csv_content = "datum;betrag;kreditor\n2024-01-15;1000;NULL\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content, encoding="utf-8")
        df = read_upload(str(csv_file))
        assert pd.isna(df.iloc[0]["kreditor"])

    def test_sql_timestamp_parsing(self):
        """SQL-Timestamps mit Millisekunden werden korrekt geparst."""
        from src.parser import parse_date_series
        s = pd.Series(["2023-12-15 07:24:06.000", "2024-01-20 14:30:00.123"])
        result = parse_date_series(s)
        assert result.notna().all()
        assert result.iloc[0].day == 15
        assert result.iloc[1].day == 20


class TestBetragByKontoklasse:
    """BETRAG_ZSCORE und BETRAG_IQR rechnen jetzt getrennt pro Kontoklasse."""

    def test_aufwandskonto_extreme_flagged(self):
        """Extremer Betrag auf Aufwandskonto (5-7xxx) wird erkannt."""
        betraege = ["100,00"] * 49 + ["100000,00"]
        df = _make_df(
            datum=["2024-01-15"] * 50,
            betrag=betraege,
            konto_soll=["65000"] * 50,
            konto_haben=["1200"] * 50,
            buchungstext=["Normal"] * 49 + ["Extrem"],
            belegnummer=[f"{i:04d}" for i in range(50)],
            erfasser=["User"] * 50,
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t01_zscore()
        assert engine.flag_counts["BETRAG_ZSCORE"] >= 1

    def test_mixed_kontoklassen_separate_stats(self):
        """Aufwand und Ertrag bekommen getrennte Statistiken —
        ein Betrag der bei Aufwand normal wäre, kann bei Ertrag auffällig sein."""
        # Aufwand: 50x 10000, Ertrag: 49x 100 + 1x 10000 (Ausreißer bei Ertrag)
        n = 100
        betraege = ["10000,00"] * 50 + ["100,00"] * 49 + ["10000,00"]
        konten = ["60000"] * 50 + ["40000"] * 50
        df = _make_df(
            datum=["2024-01-15"] * n,
            betrag=betraege,
            konto_soll=konten,
            konto_haben=["1200"] * n,
            buchungstext=["Buchung"] * n,
            belegnummer=[f"{i:04d}" for i in range(n)],
            erfasser=["User"] * n,
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t01_zscore()
        # Der 10000-Betrag auf Ertragskonto sollte auffallen
        assert engine.flag_counts["BETRAG_ZSCORE"] >= 1
        # Prüfe dass nicht alle Aufwandsbuchungen geflaggt sind (die sind normal dort)
        aufwand_flags = engine.df[
            (engine.df["konto_soll"] == "60000") & engine.df["flag_BETRAG_ZSCORE"]
        ]
        assert len(aufwand_flags) == 0


class TestGeneralumgekehrtStorno:
    """Generalumgekehrt-Kennzeichen triggert STORNO-Flag."""

    def test_generalumgekehrt_1_flags_storno(self):
        """Generalumgekehrt='1' setzt STORNO-Flag."""
        df = _make_df(
            betrag=["500,00"],
            buchungstext=["Normaler Text"],
            generalumgekehrt=["1"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t15_storno()
        assert engine.flag_counts["STORNO"] == 1

    def test_generalumgekehrt_empty_no_storno(self):
        """Leeres Generalumgekehrt löst keinen STORNO-Flag aus."""
        df = _make_df(
            betrag=["500,00"],
            buchungstext=["Normaler Text"],
            generalumgekehrt=[""],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t15_storno()
        assert engine.flag_counts["STORNO"] == 0


class TestRechnungsdatumBuchungsperiodeFallback:
    """RECHNUNGSDATUM_PERIODE nutzt Buchungsperiode als Fallback."""

    def test_buchungsperiode_mismatch_flagged(self):
        """Buchungsperiode ≠ Belegdatum → Flag (wenn kein Rechnungsdatum)."""
        df = _make_df(
            datum=["2024-03-15"],
            buchungsperiode=["2024-01-02"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t23_rechnungsdatum_periode()
        assert engine.flag_counts["RECHNUNGSDATUM_PERIODE"] == 1

    def test_rechnungsdatum_takes_priority(self):
        """Wenn Rechnungsdatum vorhanden, wird Buchungsperiode ignoriert."""
        df = _make_df(
            datum=["2024-03-15"],
            rechnungsdatum=["2024-03-01"],
            buchungsperiode=["2024-01-02"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t23_rechnungsdatum_periode()
        # Rechnungsdatum gleicher Monat → kein Flag
        assert engine.flag_counts["RECHNUNGSDATUM_PERIODE"] == 0
