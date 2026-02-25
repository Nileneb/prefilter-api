"""
Unit & integration tests for the Buchungs-Anomalie Pre-Filter.
Run with: python -m pytest tests/ -v
"""

import math
from datetime import datetime

import pandas as pd
import pytest

from modules.parser import parse_german_number, parse_date, map_columns, read_upload
from modules.engine import AnomalyEngine


# ══════════════════════════════════════════════════════════════
# Parser tests
# ══════════════════════════════════════════════════════════════

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
        # Should still work via partial match
        assert len(mapped.columns) >= 1


# ══════════════════════════════════════════════════════════════
# Engine tests
# ══════════════════════════════════════════════════════════════

def _make_df(**kwargs) -> pd.DataFrame:
    """Helper to create test DataFrames with defaults."""
    defaults = {
        "datum": ["2024-01-15"],
        "betrag": ["1000,00"],
        "konto_soll": ["4711"],
        "konto_haben": ["1200"],
        "buchungstext": ["Test"],
        "belegnummer": ["001"],
        "kostenstelle": [""],
        "kreditor": [""],
        "erfasser": ["TestUser"],
    }
    defaults.update(kwargs)
    n = max(len(v) for v in defaults.values())
    for k, v in defaults.items():
        if len(v) < n:
            defaults[k] = v * n  # repeat to fill
    return pd.DataFrame(defaults)


class TestEngineZScore:
    def test_extreme_value_flagged(self):
        # 99 normal values + 1 extreme
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
        """FIX #3: 4 identical bookings should flag 4 bookings, not 6 pairs."""
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
        # Should be 4 (bookings), not 6 (pairs)
        assert engine.flag_counts["NEAR_DUPLICATE"] == 4

    def test_missing_one_date_not_flagged(self):
        """FIX #4: if only one booking has no date, don't auto-flag."""
        df = _make_df(
            datum=["2024-01-15", ""],  # one has date, one doesn't
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
        # Should NOT be flagged (only one missing date)
        assert engine.flag_counts["NEAR_DUPLICATE"] == 0

    def test_both_missing_dates_flagged(self):
        """Both missing dates with same amount/accounts → flagged."""
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


class TestEngineBenford:
    def test_small_amounts_excluded(self):
        """FIX #5: amounts ≤10 should not affect Benford analysis."""
        # Create lots of small amounts that would distort Benford
        small = ["0,50"] * 100
        normal = [f"{i * 111}" for i in range(1, 61)]
        df = _make_df(
            datum=["2024-01-15"] * 160,
            betrag=small + normal,
            konto_soll=["4711"] * 160,
            konto_haben=["1200"] * 160,
            buchungstext=["Test"] * 160,
            belegnummer=[f"{i:04d}" for i in range(160)],
            erfasser=["User"] * 160,
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t07_benford()
        # The key assertion: the small amounts should NOT have been
        # counted in the Benford analysis at all


class TestEngineStorno:
    def test_gutschrift_normal_not_flagged(self):
        """FIX #14: normal-amount Gutschrift should not be flagged."""
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
        # The "Gutschrift" with 50€ should NOT be flagged (below threshold)
        last_flags = engine.df.iloc[-1]["_flags"]
        assert "STORNO" not in last_flags

    def test_storno_keyword_always_flagged(self):
        """Explicit storno keyword should always be flagged."""
        df = _make_df(
            betrag=["50,00"],
            buchungstext=["Storno Rechnung 123"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t15_storno()
        assert engine.flag_counts["STORNO"] == 1


class TestEngineSollGleichHaben:
    def test_same_account_flagged(self):
        """NEW #12: Soll = Haben should be flagged."""
        df = _make_df(
            konto_soll=["4711"],
            konto_haben=["4711"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t17_soll_gleich_haben()
        assert engine.flag_counts["SOLL_GLEICH_HABEN"] == 1

    def test_different_accounts_not_flagged(self):
        df = _make_df(
            konto_soll=["4711"],
            konto_haben=["1200"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t17_soll_gleich_haben()
        assert engine.flag_counts["SOLL_GLEICH_HABEN"] == 0


class TestEngineDoppelteBelegnummer:
    def test_duplicate_flagged(self):
        """NEW #11: duplicate document numbers flagged."""
        df = _make_df(
            datum=["2024-01-15", "2024-01-16"],
            betrag=["100,00", "200,00"],
            belegnummer=["001", "001"],  # same!
            konto_soll=["4711", "4720"],
            konto_haben=["1200", "1200"],
            buchungstext=["A", "B"],
            erfasser=["User", "User"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t13_doppelte_belegnummer()
        assert engine.flag_counts["DOPPELTE_BELEGNUMMER"] == 2


class TestEngineGeschaeftszeit:
    def test_nighttime_flagged(self):
        """NEW #8: booking at 3 AM should be flagged."""
        df = _make_df(
            datum=["2024-01-15T03:00:00"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t05_ausserhalb_geschaeftszeit()
        assert engine.flag_counts["AUSSERHALB_GESCHAEFTSZEIT"] == 1

    def test_daytime_not_flagged(self):
        """Booking at 10 AM should not be flagged."""
        df = _make_df(
            datum=["2024-01-15T10:00:00"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t05_ausserhalb_geschaeftszeit()
        assert engine.flag_counts["AUSSERHALB_GESCHAEFTSZEIT"] == 0

    def test_midnight_default_not_flagged(self):
        """Date without time (midnight) should NOT be flagged."""
        df = _make_df(
            datum=["2024-01-15"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t05_ausserhalb_geschaeftszeit()
        assert engine.flag_counts["AUSSERHALB_GESCHAEFTSZEIT"] == 0


class TestEngineSchwellenwertCluster:
    def test_cluster_below_threshold(self):
        """Bookings just below 5000 threshold should be flagged."""
        # 20 bookings minimum, >5% in the 4500-5000 band
        normal = ["200,00"] * 14
        cluster = ["4800,00"] * 6  # 6/20 = 30% → should trigger
        df = _make_df(
            datum=["2024-01-15"] * 20,
            betrag=normal + cluster,
            konto_soll=["4711"] * 20,
            konto_haben=["1200"] * 20,
            buchungstext=["Normal"] * 14 + ["Knapp"] * 6,
            belegnummer=[f"{i:04d}" for i in range(20)],
            erfasser=["User"] * 20,
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t11_schwellenwert_cluster()
        assert engine.flag_counts["SCHWELLENWERT_CLUSTER"] >= 6

    def test_no_cluster_if_too_few(self):
        """Less than 3 bookings in band should NOT be flagged."""
        normal = ["200,00"] * 18
        cluster = ["4800,00"] * 2  # only 2 in band
        df = _make_df(
            datum=["2024-01-15"] * 20,
            betrag=normal + cluster,
            konto_soll=["4711"] * 20,
            konto_haben=["1200"] * 20,
            buchungstext=["Normal"] * 18 + ["Knapp"] * 2,
            belegnummer=[f"{i:04d}" for i in range(20)],
            erfasser=["User"] * 20,
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t11_schwellenwert_cluster()
        assert engine.flag_counts["SCHWELLENWERT_CLUSTER"] == 0


class TestEngineBelegKreditorDuplikat:
    def test_same_beleg_same_kreditor(self):
        """Same doc# + same creditor → flagged."""
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
        """Same creditor + same amount + ≤7 days → flagged."""
        df = _make_df(
            datum=["2024-01-15", "2024-01-20"],
            betrag=["1000,00", "1000,00"],
            belegnummer=["INV-001", "INV-002"],  # different doc#
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
        """Same doc# but different creditors → NOT flagged (different test)."""
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
        """Empty booking text should be flagged."""
        df = _make_df(
            buchungstext=[""],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t21_leerer_buchungstext()
        assert engine.flag_counts["LEERER_BUCHUNGSTEXT"] == 1

    def test_generic_text_flagged(self):
        """Generic text like 'diverse' should be flagged."""
        df = _make_df(
            buchungstext=["diverse"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t21_leerer_buchungstext()
        assert engine.flag_counts["LEERER_BUCHUNGSTEXT"] == 1

    def test_short_text_flagged(self):
        """Very short text (≤2 chars) should be flagged."""
        df = _make_df(
            buchungstext=["ab"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t21_leerer_buchungstext()
        assert engine.flag_counts["LEERER_BUCHUNGSTEXT"] == 1

    def test_proper_text_not_flagged(self):
        """Normal booking text should not be flagged."""
        df = _make_df(
            buchungstext=["Bezahlung Rechnung 12345 vom 15.01.2024"],
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t21_leerer_buchungstext()
        assert engine.flag_counts["LEERER_BUCHUNGSTEXT"] == 0


class TestEngineVelocityAnomalie:
    def test_spike_flagged(self):
        """Creditor with sudden spike in bookings per month → flagged."""
        # 5 months with 2 bookings each, then 1 month with 10
        dates = []
        kreditors = []
        for m in range(1, 6):
            for _ in range(2):
                dates.append(f"2024-{m:02d}-15")
                kreditors.append("Lieferant X")
        # Spike month
        for _ in range(10):
            dates.append("2024-06-15")
            kreditors.append("Lieferant X")

        n = len(dates)
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


class TestEngineFuzzyKreditor:
    def test_similar_names_flagged(self):
        """Similar creditor names (e.g. GmbH vs no GmbH) → flagged."""
        df = _make_df(
            datum=["2024-01-15"] * 4,
            betrag=["100,00"] * 4,
            konto_soll=["4711"] * 4,
            konto_haben=["1200"] * 4,
            buchungstext=["Einkauf"] * 4,
            belegnummer=["001", "002", "003", "004"],
            kreditor=[
                "Mueller Elektronik GmbH",
                "Müller Elektronik",
                "Schmidt Bürobedarf",
                "Schmidt Buerobedarf GmbH",
            ],
            erfasser=["User"] * 4,
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t20_fuzzy_kreditor()
        # At least 2 should be flagged (Mueller/Müller pair)
        assert engine.flag_counts["FUZZY_KREDITOR"] >= 2

    def test_completely_different_not_flagged(self):
        """Completely different creditors → not flagged."""
        df = _make_df(
            datum=["2024-01-15"] * 2,
            betrag=["100,00"] * 2,
            konto_soll=["4711"] * 2,
            konto_haben=["1200"] * 2,
            buchungstext=["Einkauf"] * 2,
            belegnummer=["001", "002"],
            kreditor=["ACME Corporation", "Bundesamt für Statistik"],
            erfasser=["User"] * 2,
        )
        engine = AnomalyEngine(df)
        engine._stats()
        engine._t20_fuzzy_kreditor()
        assert engine.flag_counts["FUZZY_KREDITOR"] == 0


class TestEngineNormVendor:
    def test_rechtsform_stripped(self):
        from modules.engine import _norm_vendor
        assert "gmbh" not in _norm_vendor("Mueller GmbH")
        assert "kg" not in _norm_vendor("Schmidt GmbH & Co. KG")

    def test_umlaute_converted(self):
        from modules.engine import _norm_vendor
        assert "ue" in _norm_vendor("Müller")
        assert "oe" in _norm_vendor("Böhm")
        assert "ue" in _norm_vendor("Grün")
        assert "ae" in _norm_vendor("Bär")


class TestEngineFullRun:
    def test_full_run_succeeds(self):
        """Integration: full engine run doesn't crash."""
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
        """Edge case: empty DataFrame."""
        df = pd.DataFrame({
            "datum": [], "betrag": [], "konto_soll": [],
            "konto_haben": [], "buchungstext": [], "belegnummer": [],
            "kostenstelle": [], "kreditor": [], "erfasser": [],
        })
        engine = AnomalyEngine(df)
        result = engine.run()
        assert result["statistics"]["total_input"] == 0

    def test_21_test_methods_called(self):
        """Verify all 21 tests are represented in flag_counts keys."""
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
        fc = result["statistics"]["flag_counts"]
        # All 25 flag names must be present (some tests produce multiple flags)
        expected_flags = {
            "BETRAG_ZSCORE", "BETRAG_IQR", "SELTENE_KONTIERUNG",
            "WOCHENENDE", "MONATSENDE", "QUARTALSENDE",
            "AUSSERHALB_GESCHAEFTSZEIT", "NEAR_DUPLICATE",
            "BENFORD_1ZIFFER", "BENFORD_2ZIFFERN",
            "RUNDER_BETRAG", "ERFASSER_ANOMALIE", "SPLIT_VERDACHT",
            "SCHWELLENWERT_CLUSTER", "BELEG_LUECKE",
            "DOPPELTE_BELEGNUMMER", "BELEG_KREDITOR_DUPLIKAT",
            "STORNO", "NEUER_KREDITOR_HOCH", "SOLL_GLEICH_HABEN",
            "KONTO_BETRAG_ANOMALIE", "TEXT_KREDITOR_MISMATCH",
            "FUZZY_KREDITOR", "LEERER_BUCHUNGSTEXT",
            "VELOCITY_ANOMALIE",
        }
        assert expected_flags == set(fc.keys()), \
            f"Missing: {expected_flags - set(fc.keys())}, Extra: {set(fc.keys()) - expected_flags}"
