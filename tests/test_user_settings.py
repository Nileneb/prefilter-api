"""
Tests für persistente UI-Einstellungen (#22): src/user_settings.py
"""

from __future__ import annotations

import json

import pytest

import src.user_settings as us


@pytest.fixture
def settings_path(tmp_path, monkeypatch):
    """SETTINGS_FILE auf eine Temp-Datei umbiegen."""
    path = tmp_path / "user_settings.json"
    monkeypatch.setattr(us, "SETTINGS_FILE", str(path))
    return path


def test_load_missing_returns_empty(settings_path):
    """Keine Datei vorhanden → leeres dict (Fallback auf Defaults)."""
    assert not settings_path.exists()
    assert us.load_settings() == {}


def test_save_then_load_roundtrip(settings_path):
    """Gespeicherte Werte werden beim Laden 1:1 zurückgegeben."""
    payload = {
        "zscore_threshold": 3.1,
        "text_konto_threshold": 0.45,
        "konto_filter_all": True,
        "konto_filter_min": 40000,
        "prefix_ignore": "RW, SB",
    }
    us.save_settings(payload)
    assert settings_path.exists()
    assert us.load_settings() == payload


def test_save_overwrites_previous(settings_path):
    """Erneutes Speichern ersetzt die alten Werte vollständig."""
    us.save_settings({"zscore_threshold": 2.0})
    us.save_settings({"zscore_threshold": 4.0, "iqr_factor": 1.2})
    assert us.load_settings() == {"zscore_threshold": 4.0, "iqr_factor": 1.2}


def test_load_corrupt_returns_empty(settings_path):
    """Kaputte JSON-Datei → leeres dict statt Crash."""
    settings_path.write_text("{ this is not valid json", encoding="utf-8")
    assert us.load_settings() == {}


def test_load_non_dict_returns_empty(settings_path):
    """JSON, das kein Objekt ist (z.B. Liste) → leeres dict."""
    settings_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert us.load_settings() == {}


def test_save_no_tmp_leftover(settings_path, tmp_path):
    """Nach erfolgreichem Speichern bleibt keine .tmp-Datei liegen."""
    us.save_settings({"output_threshold": 2.5})
    assert not (tmp_path / "user_settings.json.tmp").exists()
