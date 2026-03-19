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
        5, ge=2,
        description="Mindestanzahl identischer Belegnummern-Gruppen für DOPPELTE_BELEGNUMMER (Standard: 5)",
    )
    doppelte_beleg_prefix_ignore: str = Field(
        "",
        description="Komma-getrennte Belegnummer-Präfixe die ignoriert werden (z.B. 'RW, SB')",
    )
    beleg_kreditor_days: int = Field(
        7, ge=1,
        description="Zeitfenster in Tagen für BELEG_KREDITOR_DUPLIKAT Level 2 (Standard: 7)",
    )
    beleg_kreditor_max_group_size: int = Field(
        20, ge=2,
        description="Max. Gruppengröße für BELEG_KREDITOR_DUPLIKAT Level 2 (Standard: 20)",
    )
    beleg_kreditor_regular_pct: float = Field(
        0.20, ge=0.05, le=0.80,
        description="Anteil der Datenmonate für reguläre Zahlungsmuster bei BELEG_KREDITOR_DUPLIKAT (Standard: 0.20 = 20%)",
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
        description="Sigma-Faktor für KONTO_BETRAG_ANOMALIE (Standard: 3.0)",
    )
    konto_min_buchungen: int = Field(
        5, ge=2,
        description="Mindestanzahl Buchungen pro Konto für KONTO_BETRAG_ANOMALIE (Standard: 5)",
    )

    # ── Monats-Entwicklung ───────────────────────────────────
    monats_entwicklung_zscore: float = Field(
        2.5, ge=1.0,
        description="Z-Score-Grenze für MONATS_ENTWICKLUNG (Standard: 2.5)",
    )
    monats_entwicklung_min_monate: int = Field(
        3, ge=2,
        description="Mindest-Monate pro Konto für MONATS_ENTWICKLUNG (Standard: 3)",
    )

    # ── Fehlende Monatsbuchung ───────────────────────────────
    fehlende_buchung_min_quote: float = Field(
        0.3, ge=0.1, le=1.0,
        description="Mindestanteil aktiver Monate für FEHLENDE_MONATSBUCHUNG (Standard: 0.3 = 30%)",
    )

    # ── Output-Steuerung ─────────────────────────────────────
    output_threshold: float = Field(
        2.0, ge=0.0,
        description="Score-Schwellenwert für Output (Standard: 2.0)",
    )
    max_output_rows: int = Field(
        1000, ge=1,
        description="Maximale Anzahl Ausgabezeilen (Standard: 1000)",
    )
