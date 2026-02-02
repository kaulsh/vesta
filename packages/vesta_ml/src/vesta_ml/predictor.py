"""
Prediction utilities for Vesta.
Shared by CLI scripts and the web app.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Converts raw cycle history into model-ready features.
    Must match the feature engineering in preprocess.py.
    """

    def __init__(self, lookback_window: int = 5) -> None:
        self.lookback_window = lookback_window

    def engineer_features_from_history(
        self, cycle_history: Union[List[Dict[str, str]], pd.DataFrame]
    ) -> np.ndarray:
        """
        Create features from cycle history for prediction.

        Args:
            cycle_history: List of dicts with 'start_date' and 'end_date' keys
                          OR pandas DataFrame with those columns

        Returns:
            np.ndarray: Feature vector ready for prediction
        """
        # Convert to DataFrame if needed
        if isinstance(cycle_history, list):
            df = pd.DataFrame(cycle_history)
        else:
            df = cycle_history.copy()

        # Parse dates
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        df = df.sort_values("start_date").reset_index(drop=True)

        # Need at least lookback_window + 1 cycles
        if len(df) < self.lookback_window + 1:
            raise ValueError(
                f"Insufficient cycle history. You provided {len(df)} cycles, "
                f"but need at least {self.lookback_window + 1} complete cycles to make a prediction."
            )

        # Calculate base features
        df["period_duration"] = (df["end_date"] - df["start_date"]).dt.days
        df["cycle_length"] = (df["start_date"].shift(-1) - df["start_date"]).dt.days

        # For prediction, the LAST cycle's length is unknown (we're predicting it!)
        # So we use the second-to-last cycle as our reference point
        # The features should be: what we knew BEFORE the last cycle started

        # Temporal features (from when the last cycle started)
        # Actually, we want features from the CURRENT cycle (the last one)
        last_cycle = df.iloc[-1]
        day_of_year = last_cycle["start_date"].dayofyear
        month = last_cycle["start_date"].month
        period_duration = last_cycle["period_duration"]

        # Lag features: get the last N COMPLETE cycle lengths
        # lag_1 = cycle before the current one (the one we're predicting)
        # lag_2 = cycle before that, etc.
        features: Dict[str, float] = {}

        for i in range(1, self.lookback_window + 1):
            # Index from the second-to-last cycle backwards
            idx = -(i + 1)  # -2, -3, -4, -5, -6
            if abs(idx) <= len(df):
                cycle_len = df["cycle_length"].iloc[idx]
                period_dur = df["period_duration"].iloc[idx]
                features[f"cycle_length_lag_{i}"] = cycle_len if pd.notna(cycle_len) else np.nan
                features[f"period_duration_lag_{i}"] = period_dur
            else:
                features[f"cycle_length_lag_{i}"] = np.nan
                features[f"period_duration_lag_{i}"] = np.nan

        # Rolling statistics: use the last 3 COMPLETE cycles (excluding the current one)
        # Get cycles -2, -3, -4 (indices relative to end)
        recent_cycle_lengths: List[float] = []
        for i in range(2, min(5, len(df) + 1)):  # indices -2, -3, -4
            cycle_len = df["cycle_length"].iloc[-i]
            if pd.notna(cycle_len):
                recent_cycle_lengths.append(cycle_len)

        recent_period_durations: List[float] = []
        for i in range(2, min(5, len(df) + 1)):
            recent_period_durations.append(df["period_duration"].iloc[-i])

        if len(recent_cycle_lengths) > 0:
            features["cycle_length_mean_3"] = np.mean(recent_cycle_lengths)
            features["cycle_length_std_3"] = (
                np.std(recent_cycle_lengths) if len(recent_cycle_lengths) > 1 else 0.0
            )
        else:
            features["cycle_length_mean_3"] = np.nan
            features["cycle_length_std_3"] = 0.0

        features["period_duration_mean_3"] = (
            np.mean(recent_period_durations) if len(recent_period_durations) > 0 else np.nan
        )

        # Current cycle features
        features["period_duration"] = period_duration
        features["day_of_year"] = day_of_year
        features["month"] = month

        # Convert to array in correct order (must match training feature order)
        feature_order = [
            "period_duration",
            "day_of_year",
            "month",
            "cycle_length_lag_1",
            "period_duration_lag_1",
            "cycle_length_lag_2",
            "period_duration_lag_2",
            "cycle_length_lag_3",
            "period_duration_lag_3",
            "cycle_length_lag_4",
            "period_duration_lag_4",
            "cycle_length_lag_5",
            "period_duration_lag_5",
            "cycle_length_mean_3",
            "cycle_length_std_3",
            "period_duration_mean_3",
        ]

        feature_vector = np.array([features.get(f, np.nan) for f in feature_order])

        # Check for NaN values in critical features
        if np.any(np.isnan(feature_vector)):
            missing_features = [f for f, v in zip(feature_order, feature_vector) if np.isnan(v)]
            raise ValueError(
                "Insufficient cycle history. Missing features: "
                f"{missing_features}\n"
                f"You need at least {self.lookback_window + 1} complete cycles to make a prediction.\n"
                f"You provided {len(df)} cycles."
            )

        return feature_vector.reshape(1, -1)


def _default_package_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class CyclePredictor:
    """
    Main predictor class that loads models and makes predictions.
    """

    models_dir: Optional[Path] = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.models_dir is None:
            package_root = _default_package_root()
            self.models_dir = package_root / "models"
        else:
            self.models_dir = Path(self.models_dir)

        self.feature_engineer = FeatureEngineer(lookback_window=5)

        # Load scaler
        scaler_path = self.models_dir.parent / "data" / "processed" / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. "
                "Please run preprocess.py first."
            )
        self.scaler = joblib.load(scaler_path)

        # Load Random Forest models
        self._load_random_forest()

    def _load_random_forest(self) -> None:
        """Load Random Forest models."""
        rf_dir = self.models_dir / "random_forest"

        self.model_mean = joblib.load(rf_dir / "rf_mean.pkl")
        self.model_lower = joblib.load(rf_dir / "rf_lower.pkl")
        self.model_upper = joblib.load(rf_dir / "rf_upper.pkl")

        if self.verbose:
            print(f"✓ Loaded Random Forest models from {rf_dir}")

    def predict_next_cycle(self, cycle_history, return_date: bool = True) -> Dict[str, object]:
        """
        Predict the next cycle length given historical data.

        Args:
            cycle_history: List of dicts with 'start_date' and 'end_date' keys
                          Format: [
                              {'start_date': '2023-01-15', 'end_date': '2023-01-20'},
                              {'start_date': '2023-02-12', 'end_date': '2023-02-17'},
                              ...
                          ]
            return_date: If True, also return predicted next start date

        Returns:
            dict: Prediction results with mean, lower, upper bounds, and optionally date
        """
        # Engineer features
        X = self.feature_engineer.engineer_features_from_history(cycle_history)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get predictions from Random Forest
        pred_mean = self.model_mean.predict(X_scaled)[0]
        pred_lower = self.model_lower.predict(X_scaled)[0]
        pred_upper = self.model_upper.predict(X_scaled)[0]

        result = {
            "predicted_cycle_length": round(pred_mean, 1),
            "confidence_interval_lower": round(pred_lower, 1),
            "confidence_interval_upper": round(pred_upper, 1),
            "model_used": "random_forest",
        }

        # Calculate predicted next start date
        if return_date:
            if isinstance(cycle_history, list):
                last_start = pd.to_datetime(cycle_history[-1]["start_date"])
            else:
                last_start = pd.to_datetime(cycle_history["start_date"].iloc[-1])

            next_start = last_start + timedelta(days=int(round(pred_mean)))
            next_start_lower = last_start + timedelta(days=int(round(pred_lower)))
            next_start_upper = last_start + timedelta(days=int(round(pred_upper)))

            result["predicted_next_start_date"] = next_start.strftime("%Y-%m-%d")
            result["date_range_lower"] = next_start_lower.strftime("%Y-%m-%d")
            result["date_range_upper"] = next_start_upper.strftime("%Y-%m-%d")

        return result

    def predict_from_csv(self, csv_path: Union[str, Path]) -> Dict[str, object]:
        """
        Convenience method to predict from a CSV file.

        Args:
            csv_path: Path to CSV with start_date and end_date columns

        Returns:
            dict: Prediction results
        """
        df = pd.read_csv(csv_path)
        return self.predict_next_cycle(df)
