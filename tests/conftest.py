"""
Pytest Fixtures mit echten Buchungsdaten.

Verwendet die persistent gespeicherten Uploads wenn vorhanden,
sonst Fallback auf synthetische Daten.
"""

from __future__ import annotations

import os

import pytest
import pandas as pd
from pathlib import Path

from src.parser import read_upload, map_columns
from src.engine import AnomalyEngine
from src.config import AnalysisConfig

REAL_DATA_DIR = Path(os.environ.get("UPLOAD_STORE_DIR", "data/uploads"))
FIXTURE_FILE = Path("tests/fixtures/150_buchungsdaten_sample.csv")


@pytest.fixture(scope="session")
def real_df():
    """Echte Buchungsdaten wenn vorhanden, sonst skip."""
    # 1. Direkt im Fixture-Verzeichnis
    if FIXTURE_FILE.exists():
        df = read_upload(str(FIXTURE_FILE))
        return map_columns(df)
    # 2. Aus Upload-Store (letzter Upload von Mandant 150)
    mandant_dir = REAL_DATA_DIR / "150"
    if mandant_dir.exists():
        csvs = sorted(
            [
                f
                for f in mandant_dir.glob("*.csv")
                if "_result" not in f.name and "_verdaechtig" not in f.name
            ],
            reverse=True,
        )
        if csvs:
            df = read_upload(str(csvs[0]))
            return map_columns(df)
    pytest.skip("Keine Realdaten vorhanden")


@pytest.fixture(scope="session")
def real_engine(real_df):
    """Engine mit echten Buchungsdaten."""
    return AnomalyEngine(real_df, config=AnalysisConfig())
