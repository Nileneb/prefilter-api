"""
Buchungs-Anomalie Pre-Filter — Konfiguration

Alle konfigurierbaren Schwellenwerte in einem Pydantic-Modell.
Defaults entsprechen den bisherigen Hardcode-Werten in src/engine.py.
"""

from pydantic import BaseModel, Field


class AnalysisConfig(BaseModel):
    # ── Betrags-Statistik ────────────────────────────────────
    zscore_threshold: float = Field(
        2.5, ge=0.5,
        description="Z-Score-Grenze für BETRAG_ZSCORE (Standard: 2.5)",
    )
    iqr_factor: float = Field(
        3.0, ge=0.5,
        description="IQR-Faktor für BETRAG_IQR (Standard: 3.0 → Q3 + 3.0×IQR)",
    )
    iqr_min_betrag: float = Field(
        10000.0, ge=0.0,
        description="Mindestbetrag für BETRAG_IQR — nur flaggen wenn Betrag über Fence UND über diesem Wert (Standard: 10000)",
    )

    # ── Duplikat-Erkennung ───────────────────────────────────
    near_duplicate_days: int = Field(
        3, ge=1,
        description="Zeitfenster in Tagen für NEAR_DUPLICATE (Standard: 3)",
    )
    near_duplicate_max_group_size: int = Field(
        10, ge=2,
        description="Max. Gruppengröße für NEAR_DUPLICATE — größere Gruppen sind reguläre Muster (Standard: 10)",
    )
    near_duplicate_regular_months: int = Field(
        6, ge=2,
        description="Mindest-Monate für reguläres Zahlungsmuster bei NEAR_DUPLICATE (Standard: 6)",
    )
    doppelte_beleg_min_count: int = Field(
        10, ge=2,
        description="Mindestanzahl identischer Belegnummern-Gruppen für DOPPELTE_BELEGNUMMER (Standard: 10)",
    )
    doppelte_beleg_regular_months: int = Field(
        6, ge=2,
        description="Mindest-Monate für reguläres Muster bei DOPPELTE_BELEGNUMMER (Standard: 6)",
    )
    doppelte_beleg_prefix_ignore: str = Field(
        "",
        description="Komma-getrennte Belegnummer-Präfixe die ignoriert werden (z.B. 'RW, SB')",
    )
    beleg_kreditor_days: int = Field(
        1, ge=1,
        description="Zeitfenster in Tagen für BELEG_KREDITOR_DUPLIKAT Level 2 (Standard: 1)",
    )
    beleg_kreditor_max_group_size: int = Field(
        20, ge=2,
        description="Max. Gruppengröße für BELEG_KREDITOR_DUPLIKAT Level 2 (Standard: 20)",
    )
    beleg_kreditor_regular_pct: float = Field(
        0.15, ge=0.05, le=0.80,
        description="Anteil der Datenmonate für reguläre Zahlungsmuster bei BELEG_KREDITOR_DUPLIKAT (Standard: 0.15 = 15%)",
    )

    # ── Neuer Kreditor ───────────────────────────────────────
    new_kreditor_max_bookings: int = Field(
        2, ge=1,
        description="Max. Buchungen um als 'neu' zu gelten für NEUER_KREDITOR_HOCH (Standard: 2)",
    )
    new_kreditor_amount_sigma: float = Field(
        1.5, ge=0.0,
        description="Sigma-Faktor für NEUER_KREDITOR_HOCH Betragsschwelle (Standard: 1.5)",
    )

    # ── Konto-Betrag-Anomalie ────────────────────────────────
    konto_betrag_sigma: float = Field(
        3.0, ge=1.0,
        description="Sigma-Faktor für KONTO_BETRAG_ANOMALIE: Betrag > mean ± sigma×std → Flag (Standard: 3.0)",
    )
    konto_min_buchungen: int = Field(
        5, ge=2,
        description="Mindestanzahl Buchungen pro Konto für KONTO_BETRAG_ANOMALIE (Standard: 5)",
    )

    # ── Monats-Entwicklung ───────────────────────────────────
    monats_entwicklung_zscore: float = Field(
        3.0, ge=1.0,
        description="Z-Score-Grenze für MONATS_ENTWICKLUNG: Monatssumme > mean ± zscore×std → Flag (Standard: 3.0)",
    )
    monats_entwicklung_min_monate: int = Field(
        3, ge=2,
        description="Mindest-Monate pro Konto für MONATS_ENTWICKLUNG (Standard: 3)",
    )

    # ── Fehlende Monatsbuchung ───────────────────────────────
    fehlende_buchung_min_quote: float = Field(
        0.5, ge=0.1, le=1.0,
        description="Mindestanteil aktiver Monate für FEHLENDE_MONATSBUCHUNG (Standard: 0.5 = 50%)",
    )

    # ── Globaler Konto-Bereichsfilter (GuV) ──────────────────
    # WHY: Die Anomalie-Erkennung zielt fachlich auf GuV-Konten — Ertrag
    # (40000–59999) + Aufwand (60000–79999). Bestandskonten (<40000) und
    # Kostenrechnung (≥80000) sind keine sinnvollen Prüfziele. Dieser EINE
    # Filter gilt für ALLE Tests (statt früher: hardcoded bei Betrag, separat
    # bei TEXT_KONTO_MATCH, gar nicht bei den übrigen).
    konto_filter_enabled: bool = Field(
        True,
        description="Konto-Bereichsfilter aktiv. False = alle Konten einbeziehen.",
    )
    konto_filter_min: int = Field(
        40000, ge=0,
        description="Untergrenze konto_soll (inklusive). Standard: 40000 (Erträge ab).",
    )
    konto_filter_max: int = Field(
        80000, ge=0,
        description="Obergrenze konto_soll (exklusive). Standard: 80000 (Aufwände bis 79999).",
    )

    # ── Output-Steuerung ─────────────────────────────────────
    output_threshold: float = Field(
        2.0, ge=0.0,
        description="Score-Schwellenwert für Output (Standard: 2.0)",
    )
    max_output_rows: int = Field(
        0, ge=0,
        description="Maximale Anzahl Ausgabezeilen (0 = unbegrenzt)",
    )

    # ── AI / Embedding-Features ──────────────────────────────
    near_duplicate_text_similarity: float = Field(
        0.85, ge=0.0, le=1.0,
        description="Cosine-Similarity-Schwelle für NEAR_DUPLICATE Buchungstext-Vergleich (Standard: 0.85). 0 = deaktiviert.",
    )
    kreditor_clustering_enabled: bool = Field(
        True,
        description="Kreditor-Clustering via DBSCAN auf Embeddings aktivieren (Standard: True)",
    )
    kreditor_clustering_eps: float = Field(
        0.20, ge=0.01, le=1.0,
        description="DBSCAN epsilon für Kreditor-Clustering (1-cosine_similarity, Standard: 0.20)",
    )
    isolation_enabled: bool = Field(
        False,
        description="Isolation-Forest Catch-All-Test aktivieren (Standard: False, experimentell)",
    )
    isolation_contamination: float = Field(
        0.02, ge=0.001, le=0.5,
        description="Erwarteter Anomalie-Anteil für Isolation Forest (Standard: 0.02 = 2%)",
    )
    isolation_min_bookings: int = Field(
        1000, ge=50,
        description="Mindestanzahl Buchungen für ISOLATION_ANOMALIE — darunter zu unzuverlässig (Standard: 1000). Sonst 0 + Warnung.",
    )

    # ── Text-Konto-Match ─────────────────────────────────────────────────────
    text_konto_threshold: float = Field(
        0.12, ge=0.0, le=1.0,
        description="TEXT_KONTO_MATCH (Konto-Ebene): liegt die MITTLERE Cosine-Similarity "
                    "aller Buchungstexte eines Kontos zum Kontonamen unter diesem Wert, gilt das "
                    "Konto als systematisch namens-fremd genutzt → alle Buchungen geflaggt. "
                    "Standard 0.12 (konservativ). WHY: Diamant-Buchungstexte sind oft Namen/"
                    "Referenzen statt Konto-Beschreibungen → höhere Schwellen flaggen legitime "
                    "Konten (Lohn, Abschreibung). Höher stellen, um verdächtigere Konten zu sehen.",
    )
    text_konto_min_bookings: int = Field(
        5, ge=1,
        description="Mindestanzahl Buchungen pro Konto für TEXT_KONTO_MATCH (Standard: 5).",
    )
    text_konto_gt_path: str | None = Field(
        "docs/gt_lookup.csv",
        description="Pfad zur Ground-Truth-CSV (konto_soll,gt_bezeichnung). None = Diamant-Bezeichnung.",
    )
    # Konto-Bereich kommt jetzt aus dem globalen konto_filter_* (s.o.), nicht mehr test-spezifisch.

    # ── Konto-Text-Outlier (KONTO_TEXT_OUTLIER) ──────────────────────────────
    konto_text_outlier_min_bookings: int = Field(
        8, ge=3,
        description="Mindest-Buchungen pro Konto für ein stabiles Textprofil (Standard: 8).",
    )
    konto_text_outlier_eps: float = Field(
        0.55, ge=0.01, le=1.0,
        description="DBSCAN epsilon (cosine-Distanz) für die Konto-Textcluster (Standard: 0.55). "
                    "WHY: thematisch ähnliche Buchungstexte (z.B. Adressen) haben mit dem "
                    "multilingualen MiniLM Distanzen ~0.3-0.6; ein themenfremder Text liegt ~0.9 "
                    "entfernt. 0.55 clustert das Konto-Profil und isoliert Ausreißer robust. "
                    "(Nicht 0.20 wie beim Kreditor-Clustering — das vergleicht fast identische Namen.)",
    )
    konto_text_outlier_min_samples: int = Field(
        3, ge=2,
        description="DBSCAN min_samples — Mindestgröße eines dichten Textclusters (Standard: 3).",
    )

    # ── Gelernte Gewichte (via Feedback-Training) ─────────────
    custom_weights: dict[str, float] | None = Field(
        None,
        description="Optional: Vom Trainer gelernte Flag-Gewichte. Überschreibt Defaults pro Flag.",
    )
