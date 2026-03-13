"""
Buchungs-Anomalie Pre-Filter — Daten-Validierung

Prüft nach dem Parsing, welche Spalten befüllt sind und welche Tests
sinnvoll laufen können.

Public API:
    validate_columns(df) -> ValidationResult
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


# ── Welcher Test braucht welche Spalten ──────────────────────────────────────

TEST_REQUIREMENTS: dict[str, dict[str, list[str]]] = {
    "BETRAG_ZSCORE":          {"required": ["betrag"],   "optional": ["konto_soll"]},
    "BETRAG_IQR":             {"required": ["betrag"],   "optional": ["konto_soll"]},
    "KONTO_BETRAG_ANOMALIE":  {"required": ["betrag"],   "optional": ["konto_soll", "konto_haben"]},
    "NEAR_DUPLICATE":         {"required": ["kreditor", "betrag"],
                               "optional": ["buchungstext", "konto_haben", "konto_soll", "datum"]},
    "DOPPELTE_BELEGNUMMER":   {"required": ["belegnummer"]},
    "BELEG_KREDITOR_DUPLIKAT": {"required": ["belegnummer", "kreditor", "betrag"]},
    "STORNO":                 {"required": ["betrag"],   "optional": ["buchungstext", "generalumgekehrt"]},
    "NEUER_KREDITOR_HOCH":    {"required": ["kreditor", "betrag", "datum"]},
    "LEERER_BUCHUNGSTEXT":    {"required": ["buchungstext"]},
    "VELOCITY_ANOMALIE":      {"required": ["erfasser", "datum"]},
    "RECHNUNGSDATUM_PERIODE": {"required": ["datum"],
                               "optional": ["rechnungsdatum", "erfassungsdatum", "buchungsperiode"]},
    "BUCHUNGSTEXT_PERIODE":   {"required": ["buchungstext", "datum"]},
    "MONATS_ENTWICKLUNG":     {"required": ["betrag", "datum"]},
    "FEHLENDE_MONATSBUCHUNG": {"required": ["datum"],     "optional": ["konto_soll"]},
}

# ── Alle 14 Tests in UI-Reihenfolge ─────────────────────────────────────────

ALL_TEST_NAMES: list[str] = list(TEST_REQUIREMENTS.keys())

# UI-Kategorien für die Checkbox-Gruppierung
TEST_CATEGORIES: dict[str, list[str]] = {
    "Betrags-Tests": ["BETRAG_ZSCORE", "BETRAG_IQR", "KONTO_BETRAG_ANOMALIE"],
    "Duplikat-Tests": ["NEAR_DUPLICATE", "DOPPELTE_BELEGNUMMER", "BELEG_KREDITOR_DUPLIKAT"],
    "Buchungslogik": ["STORNO", "LEERER_BUCHUNGSTEXT", "RECHNUNGSDATUM_PERIODE", "BUCHUNGSTEXT_PERIODE"],
    "Kreditor-Tests": ["NEUER_KREDITOR_HOCH", "VELOCITY_ANOMALIE"],
    "Zeitreihen-Tests": ["MONATS_ENTWICKLUNG", "FEHLENDE_MONATSBUCHUNG"],
}


def _col_fill_rate(df: pd.DataFrame, col: str) -> float:
    """Gibt den Füllgrad einer Spalte zurück (0.0 – 100.0)."""
    if col not in df.columns:
        return 0.0
    vals = df[col].astype(str).str.strip()
    filled = (vals != "").sum()
    return round(filled / len(df) * 100, 1) if len(df) > 0 else 0.0


@dataclass
class ValidationResult:
    total_rows: int = 0
    columns_found: list[str] = field(default_factory=list)
    columns_empty: list[str] = field(default_factory=list)
    columns_sparse: dict[str, float] = field(default_factory=dict)  # col -> fill%
    tests_ok: list[str] = field(default_factory=list)
    tests_blocked: dict[str, str] = field(default_factory=dict)     # test -> reason
    tests_degraded: dict[str, str] = field(default_factory=dict)    # test -> reason


def validate_columns(df: pd.DataFrame) -> ValidationResult:
    """Prüft Spalten-Füllgrade und leitet ab, welche Tests sinnvoll sind."""
    result = ValidationResult(total_rows=len(df))

    # Füllgrade aller relevanten Spalten berechnen
    all_cols = set()
    for reqs in TEST_REQUIREMENTS.values():
        all_cols.update(reqs.get("required", []))
        all_cols.update(reqs.get("optional", []))

    fill_rates: dict[str, float] = {}
    for col in sorted(all_cols):
        rate = _col_fill_rate(df, col)
        fill_rates[col] = rate
        if rate > 0:
            result.columns_found.append(col)
        if rate == 0.0:
            result.columns_empty.append(col)
        elif rate < 50.0:
            result.columns_sparse[col] = rate

    # Pro Test: blockiert / degradiert / ok
    for test_name, reqs in TEST_REQUIREMENTS.items():
        required = reqs.get("required", [])
        optional = reqs.get("optional", [])

        blocked_cols = [c for c in required if fill_rates.get(c, 0.0) == 0.0]
        sparse_cols = [c for c in required if 0 < fill_rates.get(c, 0.0) < 50.0]
        empty_opt = [c for c in optional if fill_rates.get(c, 0.0) == 0.0]

        if blocked_cols:
            result.tests_blocked[test_name] = ", ".join(blocked_cols) + " leer"
        elif sparse_cols:
            details = ", ".join(f"{c} ({fill_rates[c]:.0f}%)" for c in sparse_cols)
            result.tests_degraded[test_name] = details + " dünn besetzt"
            result.tests_ok.append(test_name)
        elif empty_opt:
            result.tests_degraded[test_name] = ", ".join(empty_opt) + " leer → eingeschränkt"
            result.tests_ok.append(test_name)
        else:
            result.tests_ok.append(test_name)

    return result


def format_validation_report(v: ValidationResult) -> str:
    """Erzeugt einen lesbaren Validierungsbericht für die UI."""
    lines: list[str] = []
    lines.append(f"Datei: {v.total_rows:,} Zeilen".replace(",", "."))
    lines.append("")

    if v.columns_empty:
        lines.append(f"❌ LEER: {', '.join(v.columns_empty)}")
    if v.columns_sparse:
        parts = [f"{c} ({p:.0f}%)" for c, p in v.columns_sparse.items()]
        lines.append(f"⚠️ DÜNN: {', '.join(parts)}")

    if v.tests_blocked:
        lines.append("")
        lines.append("Blockierte Tests:")
        for t, reason in v.tests_blocked.items():
            lines.append(f"  ❌ {t} → {reason}")

    if v.tests_degraded:
        lines.append("")
        lines.append("Eingeschränkte Tests:")
        for t, reason in v.tests_degraded.items():
            lines.append(f"  ⚠️ {t} → {reason}")

    ok_count = len([t for t in v.tests_ok if t not in v.tests_degraded])
    lines.append("")
    lines.append(f"✅ {ok_count} Tests voll einsatzfähig, "
                 f"{len(v.tests_degraded)} eingeschränkt, "
                 f"{len(v.tests_blocked)} blockiert")
    return "\n".join(lines)
