"""
Data Preprocessing Pipeline for Vesta
======================================
This script converts raw menstrual cycle dates into a supervised learning dataset
with engineered features suitable for Random Forest models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class CycleDataPreprocessor:
    """
    Transforms raw menstrual cycle dates into a supervised learning problem.

    The key transformation is:
    - From: List of (start_date, end_date) pairs
    - To: Dataset where each row represents a cycle with features from previous cycles
    """

    def __init__(self, lookback_window=5):
        """
        Args:
            lookback_window (int): Number of previous cycles to use as features
        """
        self.lookback_window = lookback_window
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_raw_data(self, filepath):
        """
        Load raw cycle data from CSV.

        Expected format:
        start_date,end_date
        2023-01-15,2023-01-20
        2023-02-12,2023-02-17
        ...
        """
        df = pd.read_csv(filepath)
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        return df.sort_values('start_date').reset_index(drop=True)

    def calculate_base_features(self, df):
        """
        Calculate cycle-level features from raw dates.

        Features:
        - cycle_length: Days from this cycle start to next cycle start
        - period_duration: Days from start to end of period
        - days_since_last: Days from previous cycle start to this cycle start
        - day_of_year: Day of year
        - month: Month
        """
        df = df.copy()

        # Period duration (how long the bleeding lasts)
        df['period_duration'] = (df['end_date'] - df['start_date']).dt.days

        # Cycle length (start-to-start interval)
        df['cycle_length'] = (
            df['start_date'].shift(-1) - df['start_date']).dt.days

        # Days since last cycle started
        df['days_since_last'] = (
            df['start_date'] - df['start_date'].shift(1)).dt.days

        # Day of year (seasonal pattern)
        df['day_of_year'] = df['start_date'].dt.dayofyear

        # Month (another way to capture seasonality)
        df['month'] = df['start_date'].dt.month

        return df

    def create_lag_features(self, df, n_lags=None):
        """
        Create lag features: previous N cycle lengths and period durations.
        """
        if n_lags is None:
            n_lags = self.lookback_window

        df = df.copy()

        # Create lag features for cycle length
        for i in range(1, n_lags + 1):
            df[f'cycle_length_lag_{i}'] = df['cycle_length'].shift(i)
            df[f'period_duration_lag_{i}'] = df['period_duration'].shift(i)

        # Rolling statistics (captures trend)
        df['cycle_length_mean_3'] = df['cycle_length'].shift(
            1).rolling(window=3, min_periods=1).mean()
        df['cycle_length_std_3'] = df['cycle_length'].shift(
            1).rolling(window=3, min_periods=1).std()

        df['period_duration_mean_3'] = df['period_duration'].shift(
            1).rolling(window=3, min_periods=1).mean()

        return df

    def prepare_supervised_dataset(self, df, target_col='cycle_length'):
        """
        Prepare the final supervised learning dataset.
        """
        # Drop rows with NaN (from lag features and last row with no next cycle)
        df = df.dropna()

        # Select feature columns (exclude dates and target)
        exclude_cols = ['start_date', 'end_date',
                        'cycle_length', 'days_since_last']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].values
        y = df[target_col].values

        self.feature_names = feature_cols

        return X, y, feature_cols, df['start_date'].values

    def fit_transform(self, filepath, scale=True):
        """
        Complete preprocessing pipeline: load -> engineer -> scale.
        """
        # Load and process
        df = self.load_raw_data(filepath)
        df = self.calculate_base_features(df)
        df = self.create_lag_features(df)

        X, y, feature_names, dates = self.prepare_supervised_dataset(df)

        # Scale features
        if scale:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        return {
            'X': X_scaled,
            'y': y,
            'X_unscaled': X,
            'dates': dates,
            'feature_names': feature_names,
            'n_samples': len(y)
        }
    
    def save_scaler(self, filepath):
        """Save the fitted scaler for later use."""
        import joblib
        joblib.dump(self.scaler, filepath)


def create_sample_data(output_path, n_cycles=36):
    """
    Create sample menstrual cycle data for demonstration.
    """
    np.random.seed(42)

    # Start date
    start = datetime(2021, 1, 15)

    cycles = []
    current_date = start

    for _ in range(n_cycles):
        # Period duration: 3-7 days, average 5
        period_duration = np.random.randint(3, 8)

        # Cycle length: 25-32 days, average 28
        cycle_length = int(np.random.normal(28, 2))
        # Clip to realistic range
        cycle_length = max(25, min(32, cycle_length))

        end_date = current_date + timedelta(days=period_duration)

        cycles.append({
            'start_date': current_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        })

        # Move to next cycle
        current_date = current_date + timedelta(days=cycle_length)

    # Save to CSV
    df = pd.DataFrame(cycles)
    df.to_csv(output_path, index=False)
    print(f"Created sample data with {n_cycles} cycles at: {output_path}")
    print(
        f"Date range: {cycles[0]['start_date']} to {cycles[-1]['start_date']}")

    return df


def main():
    """
    Main execution: Create sample data and preprocess it.
    """
    package_root = Path(__file__).resolve().parents[1]

    raw_data_path = package_root / 'data' / 'raw' / 'cycles.csv'
    processed_data_path = package_root / 'data' / \
        'processed' / 'cycles_processed.npz'
    scaler_path = package_root / 'data' / 'processed' / 'scaler.pkl'

    # Ensure directories exist
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    # Create sample data if it doesn't exist
    if not raw_data_path.exists():
        print("Creating sample data...")
        create_sample_data(raw_data_path, n_cycles=36)
    else:
        print(f"Using existing data: {raw_data_path}")

    # Preprocess
    print("\n" + "="*60)
    print("PREPROCESSING CYCLE DATA")
    print("="*60)

    preprocessor = CycleDataPreprocessor(lookback_window=5)
    data = preprocessor.fit_transform(str(raw_data_path), scale=True)

    print(f"\nDataset Info:")
    print(f"  Total samples: {data['n_samples']}")
    print(f"  Features: {len(data['feature_names'])}")
    print(f"  Feature names: {data['feature_names']}")
    print(f"\nTarget statistics (cycle length in days):")
    print(f"  Mean: {data['y'].mean():.2f}")
    print(f"  Std: {data['y'].std():.2f}")
    print(f"  Min: {data['y'].min():.2f}")
    print(f"  Max: {data['y'].max():.2f}")

    # Save processed data
    np.savez(processed_data_path,
             X=data['X'],
             y=data['y'],
             X_unscaled=data['X_unscaled'],
             dates=data['dates'],
             feature_names=data['feature_names'])

    preprocessor.save_scaler(str(scaler_path))

    print(f"\nProcessed data saved to: {processed_data_path}")
    print(f"Scaler saved to: {scaler_path}")
    print("\n✓ Preprocessing complete!")


if __name__ == '__main__':
    main()
