"""
Buchungs-Anomalie Pre-Filter — Run-History (Monatsdurchschnitte persistent)

Speichert Lauf-Ergebnisse (Monatssummen pro Konto, Flag-Counts, Score)
als JSON unter data/history/{mandant_id}/.

Public API:
    save_run(mandant_id, filename, df, flag_counts) → path
    load_last_run(mandant_id) → dict | None
    compare_runs(current, previous) → dict
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.logging_config import get_logger

logger = get_logger("prefilter.history")

HISTORY_DIR = Path(os.environ.get("HISTORY_DIR", "data/history"))


def _monthly_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Berechnet Monatssummen pro Konto aus dem DataFrame."""
    if "_datum" not in df.columns or "_abs" not in df.columns:
        return {}
    has = df["_datum"].notna() & (df["_abs"] > 0)
    if not has.any():
        return {}
    sub = df.loc[has].copy()
    sub["_ym"] = sub["_datum"].dt.to_period("M").astype(str)
    monthly = (
        sub.groupby(["konto_soll", "_ym"], observed=True)["_abs"]
        .sum()
        .reset_index(name="summe")
    )
    result: dict[str, dict[str, float]] = {}
    for _, row in monthly.iterrows():
        konto = str(row["konto_soll"])
        ym = str(row["_ym"])
        result.setdefault(konto, {})[ym] = round(float(row["summe"]), 2)
    return result


def save_run(
    mandant_id: str,
    filename: str,
    df: pd.DataFrame,
    flag_counts: dict[str, int],
    suspicious_pct: float = 0.0,
) -> Path:
    """Speichert einen Analyse-Lauf als JSON."""
    mandant_dir = HISTORY_DIR / mandant_id
    mandant_dir.mkdir(parents=True, exist_ok=True)

    run_date = datetime.now().isoformat(timespec="seconds")
    data = {
        "mandant": mandant_id,
        "run_date": run_date,
        "file": filename,
        "total_rows": len(df),
        "monthly_stats": _monthly_stats(df),
        "flag_counts": flag_counts,
        "suspicious_pct": round(suspicious_pct, 2),
    }

    safe_ts = run_date.replace(":", "-")
    path = mandant_dir / f"run_{safe_ts}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("History gespeichert", path=str(path), mandant=mandant_id)
    return path


def load_last_run(mandant_id: str) -> dict | None:
    """Lädt den letzten Lauf für einen Mandanten."""
    mandant_dir = HISTORY_DIR / mandant_id
    if not mandant_dir.exists():
        return None
    files = sorted(mandant_dir.glob("run_*.json"), reverse=True)
    if not files:
        return None
    # Zweiter Eintrag = vorheriger Lauf (erster ist der gerade gespeicherte)
    target = files[1] if len(files) > 1 else files[0]
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("History-Datei nicht lesbar", path=str(target), error=str(e))
        return None


def compare_runs(current: dict, previous: dict) -> dict:
    """Vergleicht zwei Läufe und gibt Deltas zurück."""
    cur_flags = current.get("flag_counts", {})
    prev_flags = previous.get("flag_counts", {})

    flag_deltas = {}
    all_keys = set(cur_flags) | set(prev_flags)
    for k in sorted(all_keys):
        c = cur_flags.get(k, 0)
        p = prev_flags.get(k, 0)
        if c != p:
            flag_deltas[k] = {"current": c, "previous": p, "delta": c - p}

    cur_pct = current.get("suspicious_pct", 0)
    prev_pct = previous.get("suspicious_pct", 0)

    # Neue/entfernte Konten
    cur_konten = set(current.get("monthly_stats", {}).keys())
    prev_konten = set(previous.get("monthly_stats", {}).keys())

    return {
        "suspicious_pct_delta": round(cur_pct - prev_pct, 2),
        "flag_deltas": flag_deltas,
        "new_konten": sorted(cur_konten - prev_konten),
        "removed_konten": sorted(prev_konten - cur_konten),
        "previous_run_date": previous.get("run_date", "unbekannt"),
    }


def format_comparison(comparison: dict) -> str:
    """Formatiert den Vergleich als lesbaren String."""
    lines = [f"Vergleich mit letztem Lauf ({comparison['previous_run_date']}):"]

    delta_pct = comparison["suspicious_pct_delta"]
    sign = "+" if delta_pct > 0 else ""
    lines.append(f"  Verdächtig: {sign}{delta_pct:.1f} Prozentpunkte")

    if comparison["flag_deltas"]:
        lines.append("  Flag-Änderungen:")
        for k, v in comparison["flag_deltas"].items():
            s = "+" if v["delta"] > 0 else ""
            lines.append(f"    {k}: {v['previous']} → {v['current']} ({s}{v['delta']})")

    if comparison["new_konten"]:
        lines.append(f"  Neue Konten: {', '.join(comparison['new_konten'][:10])}")
    if comparison["removed_konten"]:
        lines.append(f"  Entfernte Konten: {', '.join(comparison['removed_konten'][:10])}")

    return "\n".join(lines)
