"""
Buchungs-Anomalie Pre-Filter — Kontierungs-IST-Analyse + Präzedenz-Vorschlag

Ein geteilter Kern für drei Konsumenten aus denselben Aggregaten (keine doppelte
Pipeline):

  1. IST-Zustand-Report (`ist_report`): wie wird faktisch kontiert? Zeigt
     Kreditor→Konto-Konsistenz, Konto→Buchungstext-Cluster und GT- vs.
     DIAMANT-Namensdrift. Grundlage zum *Schreiben* einer Kontierungsrichtlinie
     — funktioniert ohne dass eine Richtlinie existiert.
  2. KontoSuggester (`KontoSuggester.suggest`): schlägt zu (Kreditor,
     Buchungstext) die wahrscheinlichen Sachkonten vor — für beleg-capture
     (Erfassung vorne).
  3. Persistenter Index (`KontoIndex.save/load`): einmal aus Altdaten gebaut,
     von beiden gelesen.

Embeddings sind OPTIONAL (graceful degradation wie `kreditor_clustering`): ohne
Embedder trägt die Kreditor-Häufigkeit weiter; nur der Text-kNN-Fallback
entfällt. Wird ein Embedder übergeben und schlägt fehl, propagiert der Fehler
(fail-loud) — kein stummes Schlucken.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger("prefilter.kontierung")

DEFAULT_GT_PATH = "docs/gt_lookup.csv"


def _norm_kreditor(name: object) -> str:
    return " ".join(str(name).strip().lower().split())


def load_gt_lookup(path: str | Path | None) -> dict[str, str]:
    """Lädt den Ground-Truth-Kontenrahmen (konto_soll → gt_bezeichnung).

    Fehlt die (bewusst privaten) Datei, wird ein leeres Mapping zurückgegeben —
    der Report läuft dann ohne GT-Drift-Spalte.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.info("GT-Lookup nicht gefunden — Report ohne GT-Drift", path=str(p))
        return {}
    gt = pd.read_csv(p, dtype=str)
    if "konto_soll" not in gt.columns or "gt_bezeichnung" not in gt.columns:
        logger.warning("GT-Lookup hat unerwartete Spalten", columns=list(gt.columns))
        return {}
    return {
        str(k).strip(): str(v).strip()
        for k, v in zip(gt["konto_soll"], gt["gt_bezeichnung"])
        if str(k).strip()
    }


@dataclass
class KontoIndex:
    """Aggregierter Präzedenz-Index aus Altdaten. Serialisierbar."""

    kreditor_konto: dict[str, dict[str, int]]  # norm_kreditor -> {konto: count}
    texts: list[str]                           # unique Buchungstexte (∥ zu emb-Zeilen)
    text_konto: list[dict[str, int]]           # je texts[i]: {konto: count}
    emb: np.ndarray | None                     # (n_texts, dim) oder None
    model_name: str
    gt: dict[str, str]                         # konto -> GT-Bezeichnung
    diamant_bezeichnung: dict[str, str]        # konto -> häufigste DIAMANT-Bezeichnung

    def save(self, out_dir: str | Path) -> Path:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "kreditor_konto": self.kreditor_konto,
            "texts": self.texts,
            "text_konto": self.text_konto,
            "model_name": self.model_name,
            "gt": self.gt,
            "diamant_bezeichnung": self.diamant_bezeichnung,
            "has_emb": self.emb is not None,
        }
        (p / "index.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        if self.emb is not None:
            np.save(p / "emb.npy", self.emb)
        logger.info("Kontierungs-Index gespeichert", path=str(p),
                    n_kreditoren=len(self.kreditor_konto), n_texte=len(self.texts),
                    embeddings=self.emb is not None)
        return p

    @staticmethod
    def load(out_dir: str | Path) -> "KontoIndex":
        p = Path(out_dir)
        meta = json.loads((p / "index.json").read_text(encoding="utf-8"))
        emb = None
        if meta.get("has_emb") and (p / "emb.npy").exists():
            emb = np.load(p / "emb.npy")
        return KontoIndex(
            kreditor_konto=meta["kreditor_konto"],
            texts=meta["texts"],
            text_konto=meta["text_konto"],
            emb=emb,
            model_name=meta.get("model_name", ""),
            gt=meta.get("gt", {}),
            diamant_bezeichnung=meta.get("diamant_bezeichnung", {}),
        )


def build_index(
    df: pd.DataFrame,
    embedder: object | None = None,
    gt: dict[str, str] | None = None,
    gt_path: str | Path | None = DEFAULT_GT_PATH,
) -> KontoIndex:
    """Baut den Präzedenz-Index aus einem (kanonisierten) Buchungs-DataFrame.

    Erwartet (nach map_columns) mindestens `konto_soll`. `kreditor`,
    `buchungstext` und `bezeichnung` werden genutzt wenn vorhanden.
    """
    from src.parser import map_columns

    df = map_columns(df.copy())
    if "konto_soll" not in df.columns:
        raise ValueError("Spalte konto_soll fehlt — Kontierungs-Index nicht baubar.")

    n = len(df)
    konto = df["konto_soll"].astype(str).str.strip()
    kred = (df["kreditor"] if "kreditor" in df.columns else pd.Series([""] * n)).astype(str).map(_norm_kreditor)
    text = (df["buchungstext"] if "buchungstext" in df.columns else pd.Series([""] * n)).astype(str).str.strip()
    bez = (df["bezeichnung"] if "bezeichnung" in df.columns else pd.Series([""] * n)).astype(str).str.strip()

    kreditor_konto: dict[str, Counter] = {}
    for k, ko in zip(kred, konto):
        if k and ko:
            kreditor_konto.setdefault(k, Counter())[ko] += 1

    text_map: dict[str, Counter] = {}
    for t, ko in zip(text, konto):
        if t and ko:
            text_map.setdefault(t, Counter())[ko] += 1

    diamant_bez: dict[str, Counter] = {}
    for ko, b in zip(konto, bez):
        if ko and b:
            diamant_bez.setdefault(ko, Counter())[b] += 1

    texts = list(text_map.keys())
    text_konto = [dict(text_map[t]) for t in texts]

    emb: np.ndarray | None = None
    model_name = ""
    if embedder is not None and texts:
        # fail-loud: wenn ein Embedder explizit übergeben wird, darf er nicht
        # stumm wegfallen.
        from src.embedding_store import embed_cached

        emb = embed_cached(embedder, texts)
        model_name = getattr(embedder, "model_name", "")
    elif embedder is None:
        logger.info("Kein Embedder — Index nur mit Kreditor-Signal (kein Text-kNN)")

    resolved_gt = gt if gt is not None else load_gt_lookup(gt_path)

    return KontoIndex(
        kreditor_konto={k: dict(c) for k, c in kreditor_konto.items()},
        texts=texts,
        text_konto=text_konto,
        emb=emb,
        model_name=model_name,
        gt=resolved_gt,
        diamant_bezeichnung={k: c.most_common(1)[0][0] for k, c in diamant_bez.items()},
    )


@dataclass
class Suggestion:
    konto: str
    bezeichnung: str
    score: float
    reason: str


class KontoSuggester:
    """Schlägt Sachkonten zu (Kreditor, Buchungstext) vor.

    Kreditor-Häufigkeit ist das primäre Signal (gleicher Lieferant → meist
    gleiches Konto). Buchungstext-kNN ergänzt/überbrückt, wenn der Kreditor
    unbekannt ist — nur falls der Index Embeddings hat UND ein Embedder
    übergeben wurde.
    """

    def __init__(self, index: KontoIndex, embedder: object | None = None) -> None:
        self.idx = index
        self.embedder = embedder

    def suggest(
        self,
        kreditor: str = "",
        buchungstext: str = "",
        top_k: int = 3,
        text_sim_min: float = 0.55,
        knn: int = 8,
    ) -> list[Suggestion]:
        scores: dict[str, float] = {}
        reasons: dict[str, list[str]] = {}

        nk = _norm_kreditor(kreditor)
        kc = self.idx.kreditor_konto.get(nk)
        if kc:
            total = sum(kc.values())
            for ko, c in sorted(kc.items(), key=lambda kv: -kv[1]):
                share = c / total
                scores[ko] = scores.get(ko, 0.0) + share
                reasons.setdefault(ko, []).append(
                    f"{c}× bei Kreditor „{kreditor.strip()}“ ({share:.0%})"
                )

        if (
            buchungstext.strip()
            and self.idx.emb is not None
            and self.embedder is not None
            and self.idx.texts
        ):
            from src.embedding_store import embed_cached

            q = np.asarray(embed_cached(self.embedder, [buchungstext.strip()])[0], dtype=np.float32)
            mat = self.idx.emb
            qn = q / (np.linalg.norm(q) + 1e-9)
            mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
            sims = mn @ qn
            for i in np.argsort(-sims)[:knn]:
                sim = float(sims[i])
                if sim < text_sim_min:
                    break
                for ko, c in self.idx.text_konto[i].items():
                    scores[ko] = scores.get(ko, 0.0) + sim * 0.6
                    reasons.setdefault(ko, []).append(
                        f"ähnlich zu „{self.idx.texts[i][:40]}“ (sim {sim:.2f})"
                    )

        out: list[Suggestion] = []
        for ko, sc in sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]:
            bez = self.idx.gt.get(ko) or self.idx.diamant_bezeichnung.get(ko, "")
            out.append(
                Suggestion(
                    konto=ko,
                    bezeichnung=bez,
                    score=round(min(sc, 1.0), 3),
                    reason="; ".join(reasons[ko][:3]),
                )
            )
        return out


def ist_report(index: KontoIndex) -> dict:
    """Verdichtet den Index zum IST-Zustand-Report (Richtlinien-Entwurfshilfe).

    Returns dict mit:
      kreditoren: list[row]  — Konsistenz je Kreditor (uneinheitliche zuerst)
      konten:     list[row]  — Buchungs-Volumen, Text-Cluster, GT-vs-DIAMANT-Drift
      Kennzahlen: n_*, embeddings
    """
    kreditor_rows = []
    for k, kc in index.kreditor_konto.items():
        total = sum(kc.values())
        konten = sorted(kc.items(), key=lambda kv: -kv[1])
        kreditor_rows.append(
            {
                "kreditor": k,
                "buchungen": total,
                "n_konten": len(kc),
                "dominant_konto": konten[0][0],
                "dominant_anteil": round(konten[0][1] / total, 3),
                "konsistent": len(kc) == 1,
                "verteilung": ", ".join(f"{ko}:{c}" for ko, c in konten[:5]),
            }
        )
    # uneinheitliche Kreditoren zuerst (Richtlinien-Kandidaten), dann nach Volumen
    kreditor_rows.sort(key=lambda r: (r["konsistent"], -r["n_konten"], -r["buchungen"]))

    konto_count: Counter = Counter()
    konto_texts: dict[str, Counter] = {}
    for t, tk in zip(index.texts, index.text_konto):
        for ko, c in tk.items():
            konto_count[ko] += c
            konto_texts.setdefault(ko, Counter())[t] += c

    konto_rows = []
    for ko, cnt in konto_count.most_common():
        gt = index.gt.get(ko, "")
        dia = index.diamant_bezeichnung.get(ko, "")
        drift = bool(gt and dia and gt.strip().lower() != dia.strip().lower())
        konto_rows.append(
            {
                "konto": ko,
                "buchungen": cnt,
                "gt_bezeichnung": gt,
                "diamant_bezeichnung": dia,
                "namens_drift": drift,
                "top_texte": "; ".join(t for t, _ in konto_texts.get(ko, Counter()).most_common(3)),
            }
        )

    return {
        "kreditoren": kreditor_rows,
        "konten": konto_rows,
        "n_kreditoren": len(kreditor_rows),
        "n_konten": len(konto_rows),
        "n_inkonsistente_kreditoren": sum(1 for r in kreditor_rows if not r["konsistent"]),
        "n_namens_drifts": sum(1 for r in konto_rows if r["namens_drift"]),
        "embeddings": index.emb is not None,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli() -> None:
    import argparse

    from src.parser import read_upload

    ap = argparse.ArgumentParser(description="Kontierungs-IST-Analyse + Präzedenz-Index")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Index aus Altdaten-Export bauen")
    b.add_argument("csv")
    b.add_argument("--out", default="data/konto_index")
    b.add_argument("--gt", default=DEFAULT_GT_PATH)
    b.add_argument("--no-embed", action="store_true", help="Ohne Text-Embeddings (nur Kreditor)")

    r = sub.add_parser("report", help="IST-Zustand-Report ausgeben")
    r.add_argument("csv")
    r.add_argument("--gt", default=DEFAULT_GT_PATH)
    r.add_argument("--out-dir", default=None, help="CSVs hierhin schreiben (kreditoren/konten)")

    s = sub.add_parser("suggest", help="Sachkonto zu Kreditor/Text vorschlagen")
    s.add_argument("kreditor")
    s.add_argument("buchungstext", nargs="?", default="")
    s.add_argument("--index", default="data/konto_index")

    args = ap.parse_args()

    if args.cmd == "build":
        embedder = None if args.no_embed else _try_embedder()
        idx = build_index(read_upload(args.csv), embedder=embedder, gt_path=args.gt)
        idx.save(args.out)
        print(f"Index gebaut: {len(idx.kreditor_konto)} Kreditoren, "
              f"{len(idx.texts)} Texte, Embeddings={idx.emb is not None} → {args.out}")

    elif args.cmd == "report":
        idx = build_index(read_upload(args.csv), embedder=None, gt_path=args.gt)
        rep = ist_report(idx)
        print(f"\nIST-Zustand: {rep['n_kreditoren']} Kreditoren "
              f"({rep['n_inkonsistente_kreditoren']} uneinheitlich), "
              f"{rep['n_konten']} Konten ({rep['n_namens_drifts']} Namens-Drifts)\n")
        print("Uneinheitlich kontierte Kreditoren (Richtlinien-Kandidaten):")
        for row in rep["kreditoren"]:
            if not row["konsistent"]:
                print(f"  {row['kreditor']:<30} {row['n_konten']} Konten | {row['verteilung']}")
        if args.out_dir:
            outp = Path(args.out_dir)
            outp.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rep["kreditoren"]).to_csv(outp / "ist_kreditoren.csv", index=False)
            pd.DataFrame(rep["konten"]).to_csv(outp / "ist_konten.csv", index=False)
            print(f"\nCSVs geschrieben → {outp}")

    elif args.cmd == "suggest":
        idx = KontoIndex.load(args.index)
        embedder = _try_embedder() if idx.emb is not None else None
        for sug in KontoSuggester(idx, embedder).suggest(args.kreditor, args.buchungstext):
            print(f"  {sug.konto:<10} {sug.bezeichnung:<35} score={sug.score:<5} | {sug.reason}")


def _try_embedder() -> object | None:
    from src.embeddings import get_embedder

    emb = get_embedder()
    if emb is None:
        logger.warning("Embeddings nicht verfügbar — fahre ohne Text-kNN fort")
    return emb


if __name__ == "__main__":
    _cli()
