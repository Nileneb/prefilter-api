"""
Persistent File Storage für hochgeladene Buchungsdateien + Ergebnisse.

Verzeichnisstruktur:
    data/uploads/{mandant_id}/
        {timestamp}_{original_filename}.csv      — Original-Upload
        {timestamp}_{original_filename}_result.json  — Analyse-Ergebnis
        {timestamp}_{original_filename}_verdaechtig.csv  — Verdächtige Buchungen

Konfiguration:
    UPLOAD_STORE_DIR=/data/uploads  (ENV, Default: data/uploads)
    MAX_STORED_FILES=20             (pro Mandant, älteste werden gelöscht)
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

UPLOAD_STORE_DIR = Path(os.environ.get("UPLOAD_STORE_DIR", "data/uploads"))
MAX_STORED_FILES = int(os.environ.get("MAX_STORED_FILES", "20"))


def store_upload(
    mandant_id: str,
    original_path: str,
    original_filename: str,
) -> Path:
    """Kopiert Upload-Datei in persistenten Store. Gibt Zielpfad zurück."""
    dest_dir = UPLOAD_STORE_DIR / _safe_dirname(mandant_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_filename(Path(original_filename).stem)
    ext = Path(original_filename).suffix
    dest = dest_dir / f"{ts}_{stem}{ext}"
    shutil.copy2(original_path, dest)
    _cleanup_old(dest_dir)
    return dest


def store_result(
    mandant_id: str,
    original_filename: str,
    result: dict,
    verdaechtig_df=None,
) -> Path:
    """Speichert Analyse-Ergebnis neben der Upload-Datei."""
    dest_dir = UPLOAD_STORE_DIR / _safe_dirname(mandant_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_filename(Path(original_filename).stem)

    result_path = dest_dir / f"{ts}_{stem}_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if verdaechtig_df is not None and not verdaechtig_df.empty:
        csv_path = dest_dir / f"{ts}_{stem}_verdaechtig.csv"
        verdaechtig_df.to_csv(csv_path, index=False, sep=";", encoding="utf-8-sig")

    return result_path


def list_uploads(mandant_id: str) -> list[dict]:
    """Listet alle gespeicherten Uploads für einen Mandanten."""
    dest_dir = UPLOAD_STORE_DIR / _safe_dirname(mandant_id)
    if not dest_dir.exists():
        return []
    uploads = []
    for f in sorted(dest_dir.glob("*_result.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            uploads.append({
                "file": data.get("file", f.stem),
                "run_date": data.get("run_date", ""),
                "total_rows": data.get("statistics", {}).get("total_input", 0),
                "suspicious": data.get("statistics", {}).get("total_suspicious", 0),
                "filter_ratio": data.get("statistics", {}).get("filter_ratio", ""),
                "path": str(f),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return uploads


def get_latest_upload_path(mandant_id: str) -> Path | None:
    """Gibt den Pfad des letzten Uploads zurück (für Tests)."""
    dest_dir = UPLOAD_STORE_DIR / _safe_dirname(mandant_id)
    if not dest_dir.exists():
        return None
    csvs = sorted(
        [
            f
            for f in dest_dir.iterdir()
            if f.suffix in (".csv", ".xlsx", ".xls")
            and "_result" not in f.stem
            and "_verdaechtig" not in f.stem
        ],
        reverse=True,
    )
    return csvs[0] if csvs else None


def _safe_dirname(name: str) -> str:
    """Sanitize directory name to prevent path traversal."""
    safe = "".join(c for c in name if c.isalnum() or c in ("_", "-"))
    return safe or "unknown"


def _safe_filename(name: str) -> str:
    """Sanitize filename stem to prevent path traversal."""
    safe = "".join(c for c in name if c.isalnum() or c in ("_", "-", "."))
    return safe or "upload"


def _cleanup_old(dest_dir: Path) -> None:
    """Löscht älteste Dateien wenn MAX_STORED_FILES überschritten."""
    all_files = sorted(dest_dir.iterdir(), key=lambda f: f.stat().st_mtime)
    # Gruppiere nach Timestamp-Prefix (erste 15 Zeichen = YYYYMMDD_HHMMSS)
    groups: dict[str, list[Path]] = {}
    for f in all_files:
        prefix = f.name[:15]
        groups.setdefault(prefix, []).append(f)
    if len(groups) > MAX_STORED_FILES:
        excess = sorted(groups.keys())[: len(groups) - MAX_STORED_FILES]
        for prefix in excess:
            for f in groups[prefix]:
                f.unlink(missing_ok=True)
