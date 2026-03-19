"""Tests mit echten Buchungsdaten — nur wenn Realdaten vorhanden."""

from __future__ import annotations

import pytest


@pytest.mark.real
class TestRealData:
    """Smoke-Tests mit echten Buchungsdaten."""

    def test_engine_runs_without_error(self, real_engine):
        result = real_engine.run()
        assert "statistics" in result
        assert result["statistics"]["total_input"] > 0

    def test_suspicious_under_15_pct(self, real_engine):
        result = real_engine.run()
        ratio = result["statistics"]["total_suspicious"] / result["statistics"]["total_input"]
        assert ratio < 0.15, f"Verdächtig: {ratio:.1%} — Ziel: <15%"

    def test_konto_betrag_under_100(self, real_engine):
        result = real_engine.run()
        assert result["statistics"]["flag_counts"].get("KONTO_BETRAG_ANOMALIE", 0) < 100

    def test_monats_entwicklung_under_200(self, real_engine):
        result = real_engine.run()
        assert result["statistics"]["flag_counts"].get("MONATS_ENTWICKLUNG", 0) < 200

    def test_beleg_kreditor_under_300(self, real_engine):
        result = real_engine.run()
        assert result["statistics"]["flag_counts"].get("BELEG_KREDITOR_DUPLIKAT", 0) < 300

    def test_doppelte_beleg_under_50(self, real_engine):
        result = real_engine.run()
        assert result["statistics"]["flag_counts"].get("DOPPELTE_BELEGNUMMER", 0) < 50
