"""Tests für AnalysisConfig nach Entfernen der Legacy-Felder (#4)."""

from src.config import AnalysisConfig


def test_legacy_fields_removed():
    fields = set(AnalysisConfig.model_fields.keys())
    assert "ertrag_abweichung_pct" not in fields
    assert "aufwand_abweichung_pct" not in fields


def test_old_api_callers_do_not_crash():
    # WHY(#4): pydantic ignoriert unbekannte Felder per Default → kein Breaking Change
    cfg = AnalysisConfig.model_validate({"ertrag_abweichung_pct": 0.5, "zscore_threshold": 3.0})
    assert cfg.zscore_threshold == 3.0
    assert not hasattr(cfg, "ertrag_abweichung_pct")


def test_konto_text_outlier_defaults():
    from src.config import AnalysisConfig
    c = AnalysisConfig()
    assert c.konto_text_outlier_min_bookings == 8
    assert c.konto_text_outlier_eps == 0.20
    assert c.konto_text_outlier_min_samples == 3
