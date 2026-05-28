"""
Buchungs-Anomalie Pre-Filter — Persistente UI-Einstellungen (#22)

Speichert die zuletzt genutzten Schwellenwert-Slider als neue Defaults, damit
sie beim nächsten App-Start vorgeladen sind. Bei fehlender oder unlesbarer Datei
greifen die eingebauten UI-Defaults.

Public API:
    load_settings() -> dict
    save_settings(settings: dict) -> None
"""

from __future__ import annotations

import json
import os

from src.logging_config import get_logger

logger = get_logger("prefilter.user_settings")

# WHY(#22): Pfad per ENV überschreibbar (Tests, Container-Volume).
SETTINGS_FILE = os.environ.get("USER_SETTINGS_FILE", "user_settings.json")


def load_settings() -> dict:
    """Lädt gespeicherte UI-Einstellungen. Leeres dict, wenn keine vorhanden."""
    if not os.path.exists(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # WHY(#22): Eine kaputte Settings-Datei darf die UI nicht crashen — laut
        # loggen und auf Defaults zurückfallen (kein stilles Schlucken).
        logger.warning("user_settings.json unlesbar → Defaults",
                       error=str(e), path=SETTINGS_FILE)
        return {}
    if not isinstance(data, dict):
        logger.warning("user_settings.json hat unerwartetes Format → Defaults",
                       path=SETTINGS_FILE)
        return {}
    return data


def save_settings(settings: dict) -> None:
    """Schreibt die aktuellen UI-Einstellungen atomar nach SETTINGS_FILE.

    Atomar (tmp + os.replace), damit ein abgebrochener Schreibvorgang keine
    halbe/korrupte Datei hinterlässt.
    """
    tmp = f"{SETTINGS_FILE}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
    os.replace(tmp, SETTINGS_FILE)
