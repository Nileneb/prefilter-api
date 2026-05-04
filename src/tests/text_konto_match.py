"""
Buchungs-Anomalie Pre-Filter — Text-Konto-Match Test

Vergleicht Buchungstext mit Kontobezeichnung per Cosine-Similarity.
Unterstützt Ground-Truth-Kontenplan als optionalen Override.

Standalone:
    python -m src.tests.text_konto_match input.csv --sweep
    python -m src.tests.text_konto_match input.csv --gt gt_lookup.csv --sweep

Engine-Integration:
    from src.tests.text_konto_match import get_tests as get_text_match_tests
    _ALL_TESTS += get_text_match_tests()
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AnalysisConfig
from src.embeddings import HAS_EMBEDDINGS, get_embedder
from src.tests.base import AnomalyTest, EngineStats


class TextKontoMatch(AnomalyTest):
    """Prüft ob Buchungstext semantisch zur Kontobezeichnung passt.

    Berechnet Cosine-Similarity zwischen Buchungstext-Embedding und
    Bezeichnung-Embedding (GT oder Diamant). Niedrige Similarity = Anomalie.

    Config:
        text_konto_threshold: float (default 0.3)
        text_konto_min_bookings: int (default 5)
        text_konto_gt_path: str | None — CSV mit konto_soll,gt_bezeichnung
    """

    name = "TEXT_KONTO_MATCH"
    weight = 1.5
    critical = False
    required_columns = ["buchungstext", "konto_soll"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        threshold = getattr(config, "text_konto_threshold", 0.3)
        min_bookings = getattr(config, "text_konto_min_bookings", 5)
        gt_lookup_path = getattr(config, "text_konto_gt_path", None)

        self.log("Config", threshold=threshold, min_bookings=min_bookings,
                 has_embeddings=HAS_EMBEDDINGS, gt_path=gt_lookup_path)

        if not HAS_EMBEDDINGS:
            self.log("SKIP: sentence-transformers nicht verfügbar")
            return 0

        embedder = get_embedder()
        if embedder is None:
            return 0

        # ── Bezeichnung bestimmen: GT > Diamant ──
        bez_col = self._resolve_bezeichnung(df, gt_lookup_path)
        if bez_col is None:
            self.log("SKIP: Keine Bezeichnung verfügbar")
            return 0

        # Nur Zeilen mit Buchungstext UND Bezeichnung
        has_text = (
            df["buchungstext"].astype(str).str.strip().ne("")
            & df[bez_col].astype(str).str.strip().ne("")
        )

        # Nur Ertrags- und Aufwandskonten
        if "_kontoklasse" in df.columns:
            has_text = has_text & df["_kontoklasse"].isin(["Ertrag", "Aufwand"])

        n_eligible = int(has_text.sum())
        self.log("Eligible", n_eligible=n_eligible, total=len(df))
        if n_eligible == 0:
            return 0

        sub = df.loc[has_text].copy()

        # ── Embeddings ──
        buchungstexte = sub["buchungstext"].astype(str).str.strip().values
        bezeichnungen = sub[bez_col].astype(str).str.strip().values

        all_texts = list(set(buchungstexte) | set(bezeichnungen))
        self.log("Embedding", n_unique_texts=len(all_texts))

        embeddings = embedder.embed_texts(all_texts)
        text_to_idx = {t: i for i, t in enumerate(all_texts)}

        idx_buch = np.array([text_to_idx[t] for t in buchungstexte])
        idx_bez = np.array([text_to_idx[t] for t in bezeichnungen])
        similarities = embedder.cosine_similarity_pairs(embeddings, idx_buch, idx_bez)
        sub["_text_konto_sim"] = similarities

        # ── Statistiken ──
        self.log("Similarity-Verteilung",
                 mean=round(float(similarities.mean()), 3),
                 median=round(float(np.median(similarities)), 3),
                 std=round(float(similarities.std()), 3),
                 min=round(float(similarities.min()), 3),
                 max=round(float(similarities.max()), 3))

        for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            val = float(np.percentile(similarities, pct))
            self.log(f"P{pct}", percentile=pct, value=round(val, 3),
                     n_below=int((similarities < val).sum()))
            self.metric(f"p{pct}", round(val, 3))

        # ── Flaggen ──
        anomaly_mask = similarities < threshold

        konto_counts = sub["konto_soll"].value_counts()
        small_konten = konto_counts[konto_counts < min_bookings].index
        skip_mask = sub["konto_soll"].isin(small_konten)

        final_mask = anomaly_mask & ~skip_mask
        n_flagged = int(final_mask.sum())

        self.log("Flagged",
                 n_below_threshold=int(anomaly_mask.sum()),
                 n_skipped_small_konto=int(skip_mask.sum()),
                 n_final=n_flagged)

        if n_flagged > 0:
            df.loc[sub.index[final_mask], f"flag_{self.name}"] = True
            df.loc[sub.index, "_text_konto_sim"] = similarities

        return n_flagged

    def _resolve_bezeichnung(self, df: pd.DataFrame, gt_path: str | None) -> str | None:
        """Bestimmt welche Bezeichnung-Spalte verwendet wird."""
        if gt_path:
            gt = pd.read_csv(gt_path)
            gt["konto_soll"] = gt["konto_soll"].astype(str).str.strip()
            gt_map = dict(zip(gt["konto_soll"], gt["gt_bezeichnung"]))
            df["_gt_bezeichnung"] = df["konto_soll"].astype(str).str.strip().map(gt_map)
            # Fallback auf Diamant wo GT fehlt
            if "bezeichnung" in df.columns:
                missing = df["_gt_bezeichnung"].isna() & df["bezeichnung"].astype(str).str.strip().ne("")
                df.loc[missing, "_gt_bezeichnung"] = df.loc[missing, "bezeichnung"]
            self.log("GT geladen", n_konten=len(gt_map),
                     n_matched=int(df["_gt_bezeichnung"].notna().sum()))
            return "_gt_bezeichnung"

        if "bezeichnung" in df.columns:
            self.log("Verwende Diamant-Bezeichnung (kein GT)")
            return "bezeichnung"

        return None


def get_tests() -> list[AnomalyTest]:
    return [TextKontoMatch()]


# ── Sweep-Hilfsfunktion ──────────────────────────────────────────────────────

def sweep_thresholds(similarities: np.ndarray, buchungstexte: np.ndarray,
                     bezeichnungen: np.ndarray, konten: np.ndarray) -> list[dict]:
    """Analysiert verschiedene Thresholds mit Beispielen."""
    results = []
    for threshold in np.arange(0.05, 0.95, 0.05):
        mask = similarities < threshold
        n = int(mask.sum())
        pct = n / len(similarities) * 100

        examples = []
        if n > 0:
            worst_idx = np.argsort(similarities[mask])[:3]
            actual_idx = np.where(mask)[0][worst_idx]
            for i in actual_idx:
                examples.append({
                    'sim': round(float(similarities[i]), 3),
                    'konto': str(konten[i]),
                    'bezeichnung': str(bezeichnungen[i])[:40],
                    'buchungstext': str(buchungstexte[i])[:40],
                })

        results.append({
            'threshold': round(float(threshold), 2),
            'n_anomalies': n,
            'pct': round(pct, 1),
            'examples': examples,
        })
    return results


# ── Standalone-Modus ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Text-Konto-Match Analyse')
    parser.add_argument('input', help='CSV-Datei (pipe-separated)')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--sweep', action='store_true', help='Threshold-Sweep 0.05-0.95')
    parser.add_argument('--output', default='text_konto_anomalies.csv')
    parser.add_argument('--delimiter', default='|')
    parser.add_argument('--gt', default=None,
                        help='Ground-Truth CSV (konto_soll,gt_bezeichnung)')

    args = parser.parse_args()

    from src.parser import normalize_columns
    from src.accounting import kontoklasse

    print(f"Lade {args.input}...")
    df = pd.read_csv(args.input, sep=args.delimiter, encoding='utf-8-sig', low_memory=False)
    df = normalize_columns(df)

    # GT laden
    if args.gt:
        gt = pd.read_csv(args.gt)
        gt["konto_soll"] = gt["konto_soll"].astype(str).str.strip()
        gt_map = dict(zip(gt["konto_soll"], gt["gt_bezeichnung"]))
        df["_gt_bezeichnung"] = df["konto_soll"].astype(str).str.strip().map(gt_map)
        if "bezeichnung" in df.columns:
            missing = df["_gt_bezeichnung"].isna() & df["bezeichnung"].astype(str).str.strip().ne("")
            df.loc[missing, "_gt_bezeichnung"] = df.loc[missing, "bezeichnung"]
        bez_col = "_gt_bezeichnung"
        print(f"GT geladen: {len(gt_map)} Konten, {df[bez_col].notna().sum():,} gemappt")
    else:
        bez_col = "bezeichnung"
        print("Kein GT — verwende Diamant-Bezeichnung")

    # Filter
    df['_kontoklasse'] = kontoklasse(df['konto_soll'])
    ea = df[df['_kontoklasse'].isin(['Ertrag', 'Aufwand'])].copy()
    ea = ea[ea['buchungstext'].astype(str).str.strip().ne('')]
    ea = ea[ea[bez_col].astype(str).str.strip().ne('')]
    print(f"Ertrag+Aufwand: {len(ea):,}")

    embedder = get_embedder()
    if embedder is None:
        print("ERROR: sentence-transformers nicht installiert!")
        sys.exit(1)

    buchungstexte = ea['buchungstext'].astype(str).str.strip().values
    bezeichnungen = ea[bez_col].astype(str).str.strip().values
    konten = ea['konto_soll'].astype(str).str.strip().values

    all_texts = list(set(buchungstexte) | set(bezeichnungen))
    print(f"Embedding {len(all_texts):,} unique Texte...")

    embeddings = embedder.embed_texts(all_texts)
    text_to_idx = {t: i for i, t in enumerate(all_texts)}

    idx_buch = np.array([text_to_idx[t] for t in buchungstexte])
    idx_bez = np.array([text_to_idx[t] for t in bezeichnungen])
    similarities = embedder.cosine_similarity_pairs(embeddings, idx_buch, idx_bez)

    print(f"\nSimilarity: mean={similarities.mean():.3f} median={np.median(similarities):.3f} "
          f"std={similarities.std():.3f} min={similarities.min():.3f} max={similarities.max():.3f}")

    if args.sweep:
        results = sweep_thresholds(similarities, buchungstexte, bezeichnungen, konten)
        print(f"\n{'Threshold':>10} | {'Anomalien':>10} | {'%':>6} | Beispiele")
        print("-" * 80)
        for r in results:
            ex_str = "; ".join([
                f"sim={e['sim']} {e['bezeichnung']}↔{e['buchungstext']}"
                for e in r['examples'][:2]
            ]) if r['examples'] else ""
            print(f"{r['threshold']:>10.02f} | {r['n_anomalies']:>10,} | {r['pct']:>5.1f}% | {ex_str[:60]}")

    ea['_similarity'] = similarities
    ea['_bezeichnung_used'] = bezeichnungen
    anomalies = ea[similarities < args.threshold].sort_values('_similarity')
    anomalies.to_csv(args.output, index=False)
    print(f"\n✓ {len(anomalies):,} Anomalien: {args.output}")
