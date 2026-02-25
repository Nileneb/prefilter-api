"""
Buchungs-Anomalie Pre-Filter — Parsing & Column Mapping
"""

import io
import re
from typing import Optional
from datetime import datetime

import pandas as pd


# ── Column normalisation ────────────────────────────────────
COLUMN_ALIASES = {
    "datum":        ["datum", "date", "buchungsdatum", "belegdatum"],
    "betrag":       ["betrag", "amount", "summe", "wert"],
    "konto_soll":   ["konto_soll", "kontosoll", "soll", "sollkonto", "debit"],
    "konto_haben":  ["konto_haben", "kontohaben", "haben", "habenkonto", "credit"],
    "buchungstext": ["buchungstext", "text", "beschreibung", "verwendungszweck"],
    "belegnummer":  ["belegnummer", "beleg", "belegnr", "beleg_nr", "voucher"],
    "kostenstelle": ["kostenstelle", "kst", "cost_center"],
    "kreditor":     ["kreditor", "lieferant", "vendor", "supplier", "creditor"],
    "erfasser":     ["erfasser", "user", "benutzer", "ersteller", "created_by"],
}


def _norm(name: str) -> str:
    """Normalise a column name for fuzzy matching."""
    s = name.lower().strip()
    for old, new in [("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")]:
        s = s.replace(old, new)
    s = re.sub(r"[^a-z0-9]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using COLUMN_ALIASES."""
    normed = {_norm(c): c for c in df.columns}
    rename = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normed:
                rename[normed[alias]] = canon
                break
        else:
            for alias in aliases:
                for n, orig in normed.items():
                    if alias in n and orig not in rename:
                        rename[orig] = canon
                        break
                if canon in rename.values():
                    break
    return df.rename(columns=rename)


# ── Number parsing ──────────────────────────────────────────
def parse_german_number(val) -> float:
    """Parse a German-style number string to float.

    Handles:
      - Currency symbols (€, $, £)
      - Mixed dot/comma separators (1.234,56 → 1234.56)
      - Ambiguous: if comma-separated with ≤2 decimals → treated as decimal
        e.g. "1,23" → 1.23 (not 123). This is a known ambiguity for values
        like "1,23" which could mean 1.23€ or 123 units.
        For Buchungsdaten this is usually correct (currency amounts).
    """
    if pd.isna(val):
        return 0.0
    s = str(val).strip().replace("€", "").replace("$", "").replace("£", "").replace(" ", "")
    if not s:
        return 0.0
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


# ── Date parsing ────────────────────────────────────────────
def parse_date(val) -> Optional[datetime]:
    """Parse common German/ISO date formats, returns None on failure."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    # ISO: 2024-01-15 or 2024-01-15T10:30:00
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?", s)
    if m:
        try:
            return datetime(
                int(m[1]), int(m[2]), int(m[3]),
                int(m[4] or 0), int(m[5] or 0), int(m[6] or 0),
            )
        except (ValueError, TypeError):
            pass
    # German: 15.01.2024
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})(?:\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?", s)
    if m:
        try:
            return datetime(
                int(m[3]), int(m[2]), int(m[1]),
                int(m[4] or 0), int(m[5] or 0), int(m[6] or 0),
            )
        except (ValueError, TypeError):
            pass
    # German short: 15.01.24
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{2})$", s)
    if m:
        try:
            return datetime(2000 + int(m[3]), int(m[2]), int(m[1]))
        except ValueError:
            pass
    try:
        result = pd.to_datetime(s).to_pydatetime()
        return result if not pd.isna(result) else None
    except Exception:
        return None


# ── File ingestion ──────────────────────────────────────────
def read_upload(filepath: str) -> pd.DataFrame:
    """Read CSV / XLS / XLSX with auto-detection."""
    name = filepath.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(filepath, engine="openpyxl", dtype=str)
    if name.endswith(".xls"):
        return pd.read_excel(filepath, engine="xlrd", dtype=str)
    # CSV — auto-detect separator
    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
            if len(df.columns) >= 3:
                return df
        except Exception:
            continue
    raise ValueError("CSV konnte nicht geparst werden — mindestens 3 Spalten erwartet.")
