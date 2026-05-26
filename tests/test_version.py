"""Erzwingt Versions- und Flag-Namens-Konsistenz (#7, #11)."""

import tomllib
from pathlib import Path

from src import __version__
from src.engine import _ALL_TESTS
from src.models import FlagCounts
from src.validator import ALL_TEST_NAMES


def test_pyproject_version_matches_package():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["version"] == __version__


def test_validator_and_engine_flag_names_match():
    # Single source of truth: validator-Liste == registrierte Engine-Tests
    assert set(ALL_TEST_NAMES) == {t.name for t in _ALL_TESTS}


def test_flagcounts_covers_all_tests():
    fields = set(FlagCounts.model_fields.keys())
    assert fields == {t.name for t in _ALL_TESTS}
