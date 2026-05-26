"""
Buchungs-Anomalie Pre-Filter — Pydantic Request/Response Schemas

Wird von src/main.py (FastAPI) genutzt.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, create_model

from src.validator import ALL_TEST_NAMES


# ── Job lifecycle ─────────────────────────────────────────────────────────────

class JobResponse(BaseModel):
    job_id: str
    status: str   # "queued" | "running" | "done" | "failed" | "cancelled"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_pct: float = 0.0
    current_test: str = ""
    elapsed_s: float = 0.0
    partial_results: dict[str, Any] | None = None
    error: str | None = None


# ── Analysis result (matches src/engine.py run() return dict) ─────────────────
# Jeder Test setzt flag_<NAME> in-place. Score = gewichtete Summe der Flags.

# WHY(#7): FlagCounts driftete (es fehlten ISOLATION_ANOMALIE/TEXT_KONTO_MATCH).
# Jetzt dynamisch aus der kanonischen Testliste erzeugt → kann nicht mehr abweichen.
FlagCounts: type[BaseModel] = create_model(
    "FlagCounts",
    **{name: (int, 0) for name in ALL_TEST_NAMES},
)


class Statistics(BaseModel):
    total_input: int
    total_output: int
    filter_ratio: str
    avg_score: float
    flag_counts: dict[str, int]


class VerdaechtigeBuchung(BaseModel):
    datum: str = ""
    konto_soll: str = ""
    konto_haben: str = ""
    betrag: float = 0.0
    buchungstext: str = ""
    belegnummer: str = ""
    kostenstelle: str = ""
    kreditor: str = ""
    anomaly_score: float = 0.0
    anomaly_flags: str = ""


class AnalysisResult(BaseModel):
    message: str
    statistics: Statistics
    verdaechtige_buchungen: list[VerdaechtigeBuchung]
    stammdaten_report: dict[str, list] = {"fuzzy_kreditor_matches": []}
    logs: list[str]
