"""
Buchungs-Anomalie Pre-Filter — Pydantic Request/Response Schemas

Wird von src/main.py (FastAPI) genutzt.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel


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


# ── Analysis result (matches modules/engine.py run() return dict) ─────────────

class FlagCounts(BaseModel):
    BETRAG_ZSCORE: int = 0
    BETRAG_IQR: int = 0
    NEAR_DUPLICATE: int = 0
    DOPPELTE_BELEGNUMMER: int = 0
    BELEG_KREDITOR_DUPLIKAT: int = 0
    STORNO: int = 0
    NEUER_KREDITOR_HOCH: int = 0
    KONTO_BETRAG_ANOMALIE: int = 0
    LEERER_BUCHUNGSTEXT: int = 0
    VELOCITY_ANOMALIE: int = 0
    RECHNUNGSDATUM_PERIODE: int = 0
    BUCHUNGSTEXT_PERIODE: int = 0
    MONATS_ENTWICKLUNG: int = 0
    FEHLENDE_MONATSBUCHUNG: int = 0


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
    erfasser: str = ""
    anomaly_score: float = 0.0
    anomaly_flags: str = ""


class AnalysisResult(BaseModel):
    message: str
    statistics: Statistics
    verdaechtige_buchungen: list[VerdaechtigeBuchung]
    stammdaten_report: dict[str, list] = {"fuzzy_kreditor_matches": []}
    logs: list[str]
