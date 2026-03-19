"""
Buchungs-Anomalie Pre-Filter — Kontoklassen & Vorzeichen-Logik

⚠️ ZENTRALE DEFINITION — Einzige Stelle für Kontoklassen-Grenzen und
Vorzeichen-Berechnung. Alle anderen Dateien importieren von hier.

Kontoklassen:
    Ertrag:           40000–59999
    Aufwand:          60000–79999
    Kostenrechnung:   80000–99999
    Bestand:          0–39999

Soll/Haben-Vorzeichen:
    Ertrag + Haben → +abs  (normaler Ertrag)
    Ertrag + Soll  → −abs  (Erlösschmälerung)
    Aufwand + Soll → +abs  (normaler Aufwand)
    Aufwand + Haben → −abs (Aufwandsminderung)
    Bestand / Kostenrechnung → Originalvorzeichen aus _betrag
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# ── Kontoklassen-Konstanten ──────────────────────────────────────────────────
ERTRAG_MIN = 40000
ERTRAG_MAX = 59999
AUFWAND_MIN = 60000
AUFWAND_MAX = 79999
KOSTENRECHNUNG_MIN = 80000


def kontoklasse(konto_soll: pd.Series) -> pd.Series:
    """Bestimmt Kontoklasse aus der Kontonummer.

    Ertrag: 40000–59999, Aufwand: 60000–79999,
    Kostenrechnung: >= 80000, Bestand: 0–39999
    """
    num = pd.to_numeric(
        konto_soll.astype(str).str.strip().str.replace(r"\D", "", regex=True),
        errors="coerce",
    )
    klasse = pd.Series("Bestand", index=konto_soll.index)
    klasse = klasse.where(~((num >= ERTRAG_MIN) & (num <= ERTRAG_MAX)), "Ertrag")
    klasse = klasse.where(~((num >= AUFWAND_MIN) & (num <= AUFWAND_MAX)), "Aufwand")
    klasse = klasse.where(~(num >= KOSTENRECHNUNG_MIN), "Kostenrechnung")
    return klasse


def compute_signed_betrag(df: pd.DataFrame) -> pd.Series:
    """Berechnet _betrag_signed aus _abs, _kontoklasse, soll_haben.

    Fallback wenn soll_haben fehlt: Originalvorzeichen aus _betrag.
    """
    signed = df["_betrag"].copy()

    sh_col = df.get("soll_haben")
    if sh_col is None:
        return signed

    sh = sh_col.astype(str).str.strip().str.upper()
    has_sh = sh.isin(["S", "SOLL", "H", "HABEN"])
    if not has_sh.any():
        return signed

    is_soll = sh.isin(["S", "SOLL"])
    is_haben = sh.isin(["H", "HABEN"])
    kl = df["_kontoklasse"] if "_kontoklasse" in df.columns else kontoklasse(df["konto_soll"])

    # Ertrag: Haben → +abs, Soll → −abs
    ertrag_mask = kl == "Ertrag"
    signed.loc[has_sh & ertrag_mask & is_haben] = df.loc[has_sh & ertrag_mask & is_haben, "_abs"]
    signed.loc[has_sh & ertrag_mask & is_soll] = -df.loc[has_sh & ertrag_mask & is_soll, "_abs"]

    # Aufwand: Soll → +abs, Haben → −abs
    aufwand_mask = kl == "Aufwand"
    signed.loc[has_sh & aufwand_mask & is_soll] = df.loc[has_sh & aufwand_mask & is_soll, "_abs"]
    signed.loc[has_sh & aufwand_mask & is_haben] = -df.loc[has_sh & aufwand_mask & is_haben, "_abs"]

    # Bestand: Originalvorzeichen bleibt

    return signed
