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
    "KONTO_BETRAG_ANOMALIE":  {"required": ["betrag"],   "optional": ["konto_soll"]},
    "NEAR_DUPLICATE":         {"required": ["kreditor", "betrag"],
                               "optional": ["buchungstext", "konto_soll", "datum", "soll_haben"]},
    "DOPPELTE_BELEGNUMMER":   {"required": ["belegnummer"]},
    "BELEG_KREDITOR_DUPLIKAT": {"required": ["belegnummer", "kreditor", "betrag"]},
    "STORNO":                 {"required": ["betrag"],   "optional": ["buchungstext", "generalumgekehrt"]},
    "NEUER_KREDITOR_HOCH":    {"required": ["kreditor", "betrag", "datum"]},
    "LEERER_BUCHUNGSTEXT":    {"required": ["buchungstext"]},
    # WHY(#15): Diamant liefert kein separates rechnungsdatum. Der Test vergleicht
    # _datum gegen erfassungsdatum ODER buchungsperiode — fehlen beide, läuft er
    # immer leer. required_any blockiert ihn dann (statt still 0 Treffer).
    "RECHNUNGSDATUM_PERIODE": {"required": ["datum"],
                               "required_any": ["erfassungsdatum", "buchungsperiode"]},
    "BUCHUNGSTEXT_PERIODE":   {"required": ["buchungstext", "datum"]},
    "MONATS_ENTWICKLUNG":     {"required": ["betrag", "datum"]},
    "FEHLENDE_MONATSBUCHUNG": {"required": ["datum"],     "optional": ["konto_soll"]},
    "ISOLATION_ANOMALIE":    {"required": ["betrag"],    "optional": ["datum", "konto_soll"]},
    "TEXT_KONTO_MATCH":      {"required": ["buchungstext", "konto_soll"]},
}

# ── Alle Tests in UI-Reihenfolge ─────────────────────────────────────────────

ALL_TEST_NAMES: list[str] = list(TEST_REQUIREMENTS.keys())

# WHY: Im Diamant-Buchungszeilen-Modell sind soll_haben/konto_haben erwartbar
# abwesend (Gegenseite = eigene Zeile je DVBeleg). Ihr Fehlen darf einen Test
# nicht als "eingeschränkt" markieren — die Engine fällt sauber zurück (signierter
# Betrag + Beleg-Paar-Heuristik). Andere fehlende Optionals bleiben echte Hinweise.
_MODEL_OPTIONAL: set[str] = {"soll_haben", "konto_haben"}

# UI-Kategorien für die Checkbox-Gruppierung
TEST_CATEGORIES: dict[str, list[str]] = {
    "Betrags-Tests": ["BETRAG_ZSCORE", "BETRAG_IQR", "KONTO_BETRAG_ANOMALIE"],
    "Duplikat-Tests": ["NEAR_DUPLICATE", "DOPPELTE_BELEGNUMMER", "BELEG_KREDITOR_DUPLIKAT"],
    "Buchungslogik": ["STORNO", "LEERER_BUCHUNGSTEXT", "RECHNUNGSDATUM_PERIODE", "BUCHUNGSTEXT_PERIODE"],
    "Kreditor-Tests": ["NEUER_KREDITOR_HOCH"],
    "Zeitreihen-Tests": ["MONATS_ENTWICKLUNG", "FEHLENDE_MONATSBUCHUNG"],
    "Experimentell": ["ISOLATION_ANOMALIE"],
    "Embedding-Tests": ["TEXT_KONTO_MATCH"],
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
    warnings: list[str] = field(default_factory=list)               # Echte Qualitätsprobleme (⚠️)
    infos: list[str] = field(default_factory=list)                  # Erwartbare Modell-Eigenschaften (ℹ️)


def validate_columns(df: pd.DataFrame) -> ValidationResult:
    """Prüft Spalten-Füllgrade und leitet ab, welche Tests sinnvoll sind."""
    result = ValidationResult(total_rows=len(df))

    # Füllgrade aller relevanten Spalten berechnen
    all_cols = set()
    required_anywhere = set()
    for reqs in TEST_REQUIREMENTS.values():
        req = reqs.get("required", [])
        all_cols.update(req)
        required_anywhere.update(req)
        all_cols.update(reqs.get("optional", []))
        all_cols.update(reqs.get("required_any", []))

    fill_rates: dict[str, float] = {}
    for col in sorted(all_cols):
        rate = _col_fill_rate(df, col)
        fill_rates[col] = rate
        if rate > 0:
            result.columns_found.append(col)
        if rate == 0.0 and col in required_anywhere:
            result.columns_empty.append(col)
        elif rate < 50.0 and rate > 0:
            result.columns_sparse[col] = rate

    # Pro Test: blockiert / degradiert / ok
    for test_name, reqs in TEST_REQUIREMENTS.items():
        required = reqs.get("required", [])
        optional = reqs.get("optional", [])
        required_any = reqs.get("required_any", [])

        blocked_cols = [c for c in required if fill_rates.get(c, 0.0) == 0.0]
        # required_any: mindestens eine dieser Spalten muss befüllt sein
        any_satisfied = (not required_any) or any(
            fill_rates.get(c, 0.0) > 0.0 for c in required_any
        )
        sparse_cols = [c for c in required if 0 < fill_rates.get(c, 0.0) < 50.0]
        # Modellbedingt-abwesende Optionals (soll_haben/konto_haben) NICHT als
        # Degradierung werten — die Engine degradiert dafür sauber.
        empty_opt = [
            c for c in optional
            if fill_rates.get(c, 0.0) == 0.0 and c not in _MODEL_OPTIONAL
        ]

        if blocked_cols:
            result.tests_blocked[test_name] = ", ".join(blocked_cols) + " leer"
        elif not any_satisfied:
            result.tests_blocked[test_name] = (
                "keine Vergleichsdatumsquelle (" + ", ".join(required_any) + " leer)"
            )
        elif sparse_cols:
            details = ", ".join(f"{c} ({fill_rates[c]:.0f}%)" for c in sparse_cols)
            result.tests_degraded[test_name] = details + " dünn besetzt"
            result.tests_ok.append(test_name)
        elif empty_opt:
            result.tests_degraded[test_name] = ", ".join(empty_opt) + " leer → eingeschränkt"
            result.tests_ok.append(test_name)
        else:
            result.tests_ok.append(test_name)

    # ── Hinweise: Diamant-Modell vs. echte Probleme ───────────────
    # WHY: Diamant exportiert Buchungszeilen (je Zeile EIN Konto, Gegenseite =
    # andere Zeile desselben DVBelegs). Das Fehlen eines Gegenkonto-Spalte bzw.
    # einer separaten Soll/Haben-Spalte ist KEIN Defekt, sondern das Modell —
    # darf also nicht als ⚠️ "eingeschränkt" alarmieren (sonst unlogisch).
    betrag_signed = False
    if "betrag" in df.columns:
        from src.parser import parse_german_number_series
        _vals = parse_german_number_series(df["betrag"])
        betrag_signed = bool((_vals < 0).any())

    sh_rate = _col_fill_rate(df, "soll_haben")
    if sh_rate == 0.0:
        if betrag_signed:
            result.infos.append(
                "Keine separate soll_haben-Spalte — Vorzeichen kommt aus dem signierten "
                "Betrag (Diamant-Standard). Kein Qualitätsproblem."
            )
        else:
            result.warnings.append(
                "soll_haben fehlt UND Betrag ist unsigniert → Soll/Haben-Richtung unbekannt, "
                "Ertrag/Aufwand-Vorzeichen ggf. ungenau."
            )
    elif sh_rate < 50.0:
        result.warnings.append(
            f"soll_haben nur {sh_rate:.0f}% befüllt → Vorzeichen-Berechnung teilweise ungenau"
        )

    kh_rate = _col_fill_rate(df, "konto_haben")
    if kh_rate == 0.0:
        result.infos.append(
            "Kein Gegenkonto je Zeile — entspricht dem Diamant-Buchungszeilen-Modell "
            "(Soll/Haben sind getrennte Zeilen je DVBeleg). Gegenseite wird über das "
            "Beleg-Paar ermittelt. Kein Test benötigt konto_haben direkt."
        )
    elif kh_rate < 50.0:
        result.warnings.append(
            f"konto_haben nur teilweise befüllt ({kh_rate:.0f}%) → uneinheitliche Belegstruktur."
        )

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

    if v.warnings:
        lines.append("")
        lines.append("Hinweise:")
        for w in v.warnings:
            lines.append(f"  ⚠️ {w}")

    if v.infos:
        lines.append("")
        lines.append("Info (erwartetes Datenmodell):")
        for inf in v.infos:
            lines.append(f"  ℹ️ {inf}")

    return "\n".join(lines)
