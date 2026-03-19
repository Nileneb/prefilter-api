"""Tests für src/charts.py — ChartBuilder mit synthetischen Daten."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytest

from src.charts import ChartBuilder, _empty_figure
from src.accounting import kontoklasse, compute_signed_betrag


@pytest.fixture
def sample_df():
    """Synthetischer DataFrame wie nach engine._prepare() + Tests."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    konten = np.random.choice(["40100", "50200", "65000", "70100", "10000"], n)
    betraege = np.random.lognormal(mean=6, sigma=2, size=n).round(2)
    kreditoren = np.random.choice(["Lieferant A", "Lieferant B", "Lieferant C", ""], n)
    sh = np.random.choice(["S", "H"], n)

    df = pd.DataFrame({
        "datum": dates.strftime("%Y-%m-%d"),
        "konto_soll": pd.Categorical(konten),
        "betrag": betraege,
        "buchungstext": pd.Categorical(["Text"] * n),
        "belegnummer": pd.Categorical([f"B{i:04d}" for i in range(n)]),
        "kreditor": pd.Categorical(kreditoren),
        "soll_haben": sh,
        "klasse": "K",
        "kostenstelle": "",
        "konto_haben": "",
        "_datum": dates,
        "_betrag": betraege.astype("float32"),
        "_abs": np.abs(betraege).astype("float32"),
        "_score": np.random.uniform(0, 5, n).round(2),
    })
    # Abgeleitete Spalten
    df["_kontoklasse"] = kontoklasse(df["konto_soll"])
    df["_betrag_signed"] = compute_signed_betrag(df)
    # Flag-Spalten
    for flag in ["BETRAG_ZSCORE", "BETRAG_IQR", "NEAR_DUPLICATE", "STORNO",
                 "DOPPELTE_BELEGNUMMER", "BELEG_KREDITOR_DUPLIKAT",
                 "LEERER_BUCHUNGSTEXT", "RECHNUNGSDATUM_PERIODE",
                 "BUCHUNGSTEXT_PERIODE", "NEUER_KREDITOR_HOCH",
                 "KONTO_BETRAG_ANOMALIE",
                 "MONATS_ENTWICKLUNG", "FEHLENDE_MONATSBUCHUNG"]:
        df[f"flag_{flag}"] = np.random.choice([True, False], n, p=[0.1, 0.9])

    return df


@pytest.fixture
def sample_result():
    """Synthetisches Engine-Result-Dict."""
    return {
        "statistics": {
            "total_input": 200,
            "total_suspicious": 20,
            "total_output": 20,
            "filter_ratio": "10.0%",
            "avg_score": 2.5,
            "flag_counts": {
                "BETRAG_ZSCORE": 15,
                "BETRAG_IQR": 8,
                "NEAR_DUPLICATE": 5,
                "STORNO": 3,
                "DOPPELTE_BELEGNUMMER": 2,
                "BELEG_KREDITOR_DUPLIKAT": 1,
                "LEERER_BUCHUNGSTEXT": 0,
                "RECHNUNGSDATUM_PERIODE": 0,
                "BUCHUNGSTEXT_PERIODE": 0,
                "NEUER_KREDITOR_HOCH": 4,
                "KONTO_BETRAG_ANOMALIE": 6,
                "MONATS_ENTWICKLUNG": 2,
                "FEHLENDE_MONATSBUCHUNG": 1,
            },
        },
        "verdaechtige_buchungen": [],
    }


@pytest.fixture
def builder(sample_df, sample_result):
    return ChartBuilder(sample_df, sample_result)


class TestChartBuilder:
    def test_score_distribution(self, builder):
        fig = builder.score_distribution()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_flag_frequency(self, builder):
        fig = builder.flag_frequency()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_monthly_pnl(self, builder):
        fig = builder.monthly_pnl()
        assert isinstance(fig, go.Figure)

    def test_top_accounts(self, builder):
        fig = builder.top_accounts()
        assert isinstance(fig, go.Figure)

    def test_ertrag_aufwand_monthly(self, builder):
        fig = builder.ertrag_aufwand_monthly()
        assert isinstance(fig, go.Figure)

    def test_volume_heatmap(self, builder):
        fig = builder.volume_heatmap()
        assert isinstance(fig, go.Figure)

    def test_betrag_vs_score(self, builder):
        fig = builder.betrag_vs_score()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_kreditor_treemap(self, builder):
        fig = builder.kreditor_treemap()
        assert isinstance(fig, go.Figure)

    def test_zeitreihe_konto(self, builder):
        fig = builder.zeitreihe_konto()
        assert isinstance(fig, go.Figure)

    def test_zeitreihe_konto_specific(self, builder):
        fig = builder.zeitreihe_konto(konto="40100")
        assert isinstance(fig, go.Figure)

    def test_soll_haben_balance(self, builder):
        fig = builder.soll_haben_balance()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_all_charts(self, builder):
        charts = builder.all_charts()
        assert isinstance(charts, dict)
        assert len(charts) == 10
        for name, fig in charts.items():
            assert isinstance(fig, go.Figure), f"{name} ist keine Figure"


class TestEmptyData:
    """Charts mit Minimal-/Leerdaten geben leere Figures statt Fehler."""

    def test_empty_df(self, sample_result):
        empty_df = pd.DataFrame({
            "konto_soll": pd.Series(dtype="category"),
            "_datum": pd.Series(dtype="datetime64[ns]"),
            "_betrag": pd.Series(dtype="float32"),
            "_abs": pd.Series(dtype="float32"),
            "_score": pd.Series(dtype="float64"),
            "kreditor": pd.Series(dtype="category"),
            "soll_haben": pd.Series(dtype="str"),
        })
        sample_result["statistics"]["flag_counts"] = {}
        builder = ChartBuilder(empty_df, sample_result)
        charts = builder.all_charts()
        assert len(charts) == 10
        for fig in charts.values():
            assert isinstance(fig, go.Figure)

    def test_empty_figure_helper(self):
        fig = _empty_figure("Test-Nachricht")
        assert isinstance(fig, go.Figure)
        assert fig.layout.annotations[0].text == "Test-Nachricht"
