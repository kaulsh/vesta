"""
Training Suite for Vesta
=========================
This script trains Random Forest models for menstrual cycle prediction
with quantile regression for confidence intervals.
"""

import warnings
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
import json
from pathlib import Path
from datetime import datetime


# ============================================================================
# RANDOM FOREST BASELINE
# ============================================================================

class RandomForestTrainer:
    """
    Random Forest baseline with GridSearchCV for hyperparameter tuning.
    """
    
    def __init__(self, n_jobs=-1):
        """
        Args:
            n_jobs (int): Number of parallel jobs (-1 = use all CPU threads)
        """
        self.n_jobs = n_jobs
        self.model_mean = None
        self.model_lower = None
        self.model_upper = None
        self.best_params = None
        
    def train(self, X_train, y_train, use_gridsearch=True):
        """
        Train Random Forest with optional GridSearch.
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST BASELINE")
        print("="*60)
        
        if use_gridsearch:
            # GridSearch parameter space
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            # Base model
            rf = RandomForestRegressor(random_state=42, n_jobs=self.n_jobs)
            
            # GridSearchCV
            print(f"Running GridSearchCV with {self.n_jobs} parallel jobs...")
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=self.n_jobs,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.best_params = grid_search.best_params_
            self.model_mean = grid_search.best_estimator_
            
            print(f"\nBest parameters: {self.best_params}")
            print(f"Best CV MAE: {-grid_search.best_score_:.3f} days")
        else:
            # Use default good parameters
            self.model_mean = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=self.n_jobs
            )
            self.model_mean.fit(X_train, y_train)
        
        # Train quantile models for confidence intervals
        print("\nTraining quantile models for confidence intervals...")
        
        # Use Gradient Boosting for quantile regression (RF doesn't support it natively)
        self.model_lower = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            loss='quantile',
            alpha=0.1,  # 10th percentile
            random_state=42
        )
        self.model_lower.fit(X_train, y_train)
        
        self.model_upper = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            loss='quantile',
            alpha=0.9,  # 90th percentile
            random_state=42
        )
        self.model_upper.fit(X_train, y_train)
        
        print("✓ Random Forest training complete!")
        
    def save(self, directory):
        """Save all models."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model_mean, directory / 'rf_mean.pkl')
        joblib.dump(self.model_lower, directory / 'rf_lower.pkl')
        joblib.dump(self.model_upper, directory / 'rf_upper.pkl')
        
        if self.best_params:
            with open(directory / 'rf_params.json', 'w') as f:
                json.dump(self.best_params, f, indent=2)


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """
    Main training pipeline.
    """
    print("\n" + "="*60)
    print("VESTA TRAINING PIPELINE")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    package_root = Path(__file__).resolve().parents[1]
    data_dir = package_root / 'data'
    models_dir = package_root / 'models'
    
    processed_data_path = data_dir / 'processed' / 'cycles_processed.npz'
    test_data_path = data_dir / 'processed' / 'test_data.npz'
    rf_model_dir = models_dir / 'random_forest'
    
    # Load processed data
    print("\nLoading processed data...")
    data = np.load(processed_data_path, allow_pickle=True)
    
    X = data['X']
    y = data['y']
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle time series
    )
    
    print(f"Train: {len(y_train)} samples")
    print(f"Test: {len(y_test)} samples")
    
    # Train Random Forest
    rf_trainer = RandomForestTrainer()
    rf_trainer.train(X_train, y_train, use_gridsearch=True)
    rf_trainer.save(rf_model_dir)
    
    # Save test data for evaluation
    np.savez(test_data_path,
             X_test=X_test,
             y_test=y_test)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nModels saved:")
    print(f"  - Random Forest: {rf_model_dir}")
    print("\nNext step: Run evaluate.py to assess model performance")


if __name__ == '__main__':
    main()
