"""
Buchungs-Anomalie Pre-Filter — Parsing & Column Mapping

Public API
----------
Scalar (row-by-row, for tests and single-value use):
    parse_german_number(val)  -> float
    parse_date(val)           -> Optional[datetime]

Vectorised (column-level, use in engine for large DataFrames):
    parse_german_number_series(series) -> pd.Series[float]   (NaN for unparseable)
    parse_date_series(series)          -> pd.Series[datetime] (NaT for unparseable)

Column mapping / file ingestion:
    map_columns(df)   -> pd.DataFrame
    read_upload(path) -> pd.DataFrame
"""

import io
import re
from typing import Optional
from datetime import datetime

import pandas as pd


# ── Public exports ───────────────────────────────────────────
__all__ = [
    "COLUMN_ALIASES",
    "map_columns",
    "read_upload",
    "parse_german_number",
    "parse_german_number_series",
    "parse_date",
    "parse_date_series",
]


# ── Column normalisation ────────────────────────────────────
COLUMN_ALIASES = {
    "datum":           ["datum", "date", "buchungsdatum", "belegdatum"],
    "betrag":          ["betrag", "amount", "summe", "wert"],
    "konto_soll":      ["konto_soll", "kontosoll", "soll", "sollkonto", "debit"],
    "konto_haben":     ["konto_haben", "kontohaben", "haben", "habenkonto", "credit"],
    "buchungstext":    ["buchungstext", "text", "beschreibung", "verwendungszweck"],
    "belegnummer":     ["belegnummer", "beleg", "belegnr", "beleg_nr", "voucher"],
    "kostenstelle":    ["kostenstelle", "kst", "cost_center"],
    "kreditor":        ["kreditor", "lieferant", "vendor", "supplier", "creditor"],
    "erfasser":        ["erfasser", "user", "benutzer", "ersteller", "created_by"],
    "rechnungsdatum":  ["rechnungsdatum", "invoice_date", "rech_datum", "invoicedate"],
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

    Note: scalar version kept for backwards compatibility with tests.
    For bulk processing use parse_german_number_series() instead.
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


def parse_german_number_series(series: pd.Series) -> pd.Series:
    """Vektorisiertes Parsing einer kompletten Betrag-Spalte.

    Bei 500k Zeilen ca. 50-100x schneller als .apply(parse_german_number).
    Rückgabe: pd.Series[float] mit NaN für nicht-parseable Werte.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r'[€$£\s]', '', regex=True)
    german_mask = s.str.match(r'^-?[\d\.]*,\d{1,2}$', na=False)
    s_german = (
        s.str.replace(r'\.(?=\d{3})', '', regex=True)
         .str.replace(',', '.', regex=False)
    )
    s_english = s.str.replace(r',(?=\d{3})', '', regex=True)
    return pd.to_numeric(
        s_german.where(german_mask, s_english),
        errors='coerce'
    )


# ── Date parsing ────────────────────────────────────────────
def parse_date(val) -> Optional[datetime]:
    """Parse common German/ISO date formats, returns None on failure.

    Note: scalar version kept for backwards compatibility with tests.
    For bulk processing use parse_date_series() instead.
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?", s)
    if m:
        try:
            return datetime(
                int(m[1]), int(m[2]), int(m[3]),
                int(m[4] or 0), int(m[5] or 0), int(m[6] or 0),
            )
        except (ValueError, TypeError):
            pass
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})(?:\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?", s)
    if m:
        try:
            return datetime(
                int(m[3]), int(m[2]), int(m[1]),
                int(m[4] or 0), int(m[5] or 0), int(m[6] or 0),
            )
        except (ValueError, TypeError):
            pass
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


def parse_date_series(series: pd.Series) -> pd.Series:
    """Vektorisiertes Parsing einer kompletten Datum-Spalte mit Format-Fallbacks.

    Bei 500k Zeilen ca. 50-100x schneller als .apply(parse_date).
    Rückgabe: pd.Series[datetime64] mit pd.NaT für nicht-parseable Werte.
    """
    s = series.astype(str).str.strip()
    formats = [
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d.%m.%y",
        "%Y-%m-%dT%H:%M:%S",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
    ]
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = pd.Series(True, index=series.index)
    for fmt in formats:
        if not remaining.any():
            break
        parsed = pd.to_datetime(s[remaining], format=fmt, errors='coerce')
        newly_parsed = parsed.notna()
        if newly_parsed.any():
            result[remaining] = result[remaining].where(~newly_parsed, parsed[newly_parsed])
            remaining = remaining & result.isna()
    return result


# ── File ingestion ──────────────────────────────────────────
def read_upload(filepath: str) -> pd.DataFrame:
    """Read CSV / XLS / XLSX with auto-detection."""
    name = filepath.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(filepath, engine="openpyxl", dtype=str)
    if name.endswith(".xls"):
        return pd.read_excel(filepath, engine="xlrd", dtype=str)
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
