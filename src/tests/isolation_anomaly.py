"""
Buchungs-Anomalie Pre-Filter — Isolation Forest Test (experimentell)

Test:
    ISOLATION_ANOMALIE — Catch-All Anomalie-Erkennung via Isolation Forest

Standardmäßig DEAKTIVIERT (config.isolation_enabled = False).
Benötigt scikit-learn und optionale Text-Embeddings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AnalysisConfig
from src.logging_config import get_logger
from src.tests.base import AnomalyTest, EngineStats

logger = get_logger("prefilter.tests.isolation")

try:
    from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class IsolationAnomalie(AnomalyTest):
    name = "ISOLATION_ANOMALIE"
    weight = 1.5
    critical = False
    required_columns = ["_abs", "_datum", "konto_soll", "_is_storno"]

    def run(self, df: pd.DataFrame, stats: EngineStats, config: AnalysisConfig) -> int:
        if not getattr(config, "isolation_enabled", False):
            self.log("Deaktiviert (config.isolation_enabled=False)")
            return 0

        if not HAS_SKLEARN:
            self.log("scikit-learn nicht verfügbar — übersprungen")
            return 0

        # Stornos ausschließen
        is_storno = df.get("_is_storno", pd.Series(False, index=df.index))
        valid = (~is_storno) & (df["_abs"] > 0)
        n_valid = int(valid.sum())

        if n_valid < 50:
            self.log("Zu wenig Daten", n_valid=n_valid)
            return 0

        # Feature-Matrix bauen
        features = self._build_features(df, valid, stats)
        if features is None or features.shape[1] == 0:
            self.log("Keine Features extrahierbar")
            return 0

        self.log("Features", shape=features.shape)
        self.metric("n_features", features.shape[1])
        self.metric("n_samples", features.shape[0])

        contamination = getattr(config, "isolation_contamination", 0.02)
        clf = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        preds = clf.fit_predict(features)
        anomaly_mask_local = preds == -1

        # Zurück auf df-Index mappen
        valid_idx = df.index[valid]
        mask = pd.Series(False, index=df.index)
        mask.loc[valid_idx[anomaly_mask_local]] = True

        count = self._flag(df, mask)
        self.log("TOTAL", total_flagged=count, contamination=contamination)
        return count

    @staticmethod
    def _build_features(df: pd.DataFrame, valid: pd.Series, stats: EngineStats) -> np.ndarray | None:
        """Baut Feature-Matrix aus numerischen und optionalen Embedding-Features."""
        sub = df.loc[valid]
        feature_cols: list[np.ndarray] = []

        # Betrag (log-transformiert + zscore)
        abs_vals = sub["_abs"].values.astype(np.float64)
        log_abs = np.log1p(abs_vals)
        if log_abs.std() > 0:
            log_abs = (log_abs - log_abs.mean()) / log_abs.std()
        feature_cols.append(log_abs.reshape(-1, 1))

        # Tag im Monat (zyklisch kodiert)
        if "_datum" in sub.columns:
            dates = sub["_datum"]
            has_date = dates.notna()
            day_of_month = np.zeros(len(sub))
            if has_date.any():
                day_of_month[has_date.values] = dates[has_date].dt.day.values
            day_sin = np.sin(2 * np.pi * day_of_month / 31)
            day_cos = np.cos(2 * np.pi * day_of_month / 31)
            feature_cols.append(day_sin.reshape(-1, 1))
            feature_cols.append(day_cos.reshape(-1, 1))

        # Wochentag (zyklisch kodiert)
        if "_datum" in sub.columns:
            dow = np.zeros(len(sub))
            if has_date.any():
                dow[has_date.values] = dates[has_date].dt.dayofweek.values
            dow_sin = np.sin(2 * np.pi * dow / 7)
            dow_cos = np.cos(2 * np.pi * dow / 7)
            feature_cols.append(dow_sin.reshape(-1, 1))
            feature_cols.append(dow_cos.reshape(-1, 1))

        # Text-Embeddings (erste 10 Dimensionen via Projektion)
        text_embeddings = getattr(stats, "text_embeddings", None)
        if text_embeddings is not None:
            valid_idx = np.where(valid.values)[0]
            emb_sub = text_embeddings[valid_idx]
            # Erste 10 Dimensionen als Features (billige Dimensionsreduktion)
            n_dims = min(10, emb_sub.shape[1])
            feature_cols.append(emb_sub[:, :n_dims].astype(np.float64))

        if not feature_cols:
            return None

        return np.hstack(feature_cols)


def get_tests() -> list[AnomalyTest]:
    return [IsolationAnomalie()]
