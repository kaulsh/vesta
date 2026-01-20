"""
Training Suite for Vesta
=========================
This script trains both Random Forest (baseline) and LSTM (challenger) models
for menstrual cycle prediction with quantile regression for confidence intervals.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from optuna.trial import TrialState
import joblib
import json
import os
from datetime import datetime
from tqdm import tqdm


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
            n_jobs (int): Number of parallel jobs (-1 = use all 16 CPU threads)
        """
        self.n_jobs = n_jobs
        self.model_mean = None
        self.model_lower = None
        self.model_upper = None
        self.best_params = None
        
    def train(self, X_train, y_train, use_gridsearch=True):
        """
        Train Random Forest with optional GridSearch.
        
        Args:
            X_train: Training features
            y_train: Training targets
            use_gridsearch: Whether to use GridSearchCV
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
        
    def predict(self, X, return_intervals=True):
        """
        Predict cycle lengths with optional confidence intervals.
        
        Args:
            X: Features
            return_intervals: Whether to return 10th and 90th percentile predictions
            
        Returns:
            dict or array: Predictions (and intervals if requested)
        """
        pred_mean = self.model_mean.predict(X)
        
        if return_intervals:
            pred_lower = self.model_lower.predict(X)
            pred_upper = self.model_upper.predict(X)
            
            return {
                'mean': pred_mean,
                'lower': pred_lower,
                'upper': pred_upper
            }
        
        return pred_mean
    
    def save(self, directory):
        """Save all models."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model_mean, f'{directory}/rf_mean.pkl')
        joblib.dump(self.model_lower, f'{directory}/rf_lower.pkl')
        joblib.dump(self.model_upper, f'{directory}/rf_upper.pkl')
        
        if self.best_params:
            with open(f'{directory}/rf_params.json', 'w') as f:
                json.dump(self.best_params, f, indent=2)


# ============================================================================
# LSTM CHALLENGER
# ============================================================================

class CycleDataset(Dataset):
    """PyTorch Dataset for cycle data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """
    LSTM model for cycle length prediction.
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, features) -> reshape to (batch, seq_len=1, features)
        x = x.unsqueeze(1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected
        out = self.fc(lstm_out)
        
        return out.squeeze()


class QuantileLoss(nn.Module):
    """
    Quantile loss for training quantile regression models.
    """
    
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
        
    def forward(self, pred, target):
        errors = target - pred
        loss = torch.max(
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
        return loss.mean()


class LSTMTrainer:
    """
    LSTM trainer with Optuna hyperparameter optimization.
    """
    
    def __init__(self, device=None):
        """
        Args:
            device: torch device (cuda or cpu)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        self.model_mean = None
        self.model_lower = None
        self.model_upper = None
        self.best_params = None
        
    def create_model(self, input_size, hidden_size, num_layers, dropout):
        """Create LSTM model with given hyperparameters."""
        return LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
    
    def train_single_model(self, model, train_loader, val_loader, 
                          epochs, learning_rate, quantile=None):
        """
        Train a single LSTM model.
        
        Args:
            model: LSTM model
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            quantile: If specified, use quantile loss (for confidence intervals)
            
        Returns:
            float: Best validation MAE
        """
        # Loss and optimizer
        if quantile is not None:
            criterion = QuantileLoss(quantile=quantile)
        else:
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        
        best_val_mae = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    outputs = model(X_batch)
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(y_batch.numpy())
            
            val_mae = mean_absolute_error(val_targets, val_preds)
            scheduler.step(val_mae)
            
            # Early stopping
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        return best_val_mae
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Optuna objective function for hyperparameter tuning.
        """
        # Suggest hyperparameters
        hidden_size = trial.suggest_int('hidden_size', 32, 128, step=32)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        
        # Create data loaders
        train_dataset = CycleDataset(X_train, y_train)
        val_dataset = CycleDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create and train model
        model = self.create_model(
            input_size=X_train.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        val_mae = self.train_single_model(
            model, train_loader, val_loader,
            epochs=100, learning_rate=learning_rate
        )
        
        return val_mae
    
    def train(self, X_train, y_train, X_val, y_val, n_trials=500):
        """
        Train LSTM with Optuna hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of Optuna trials
        """
        print("\n" + "="*60)
        print("TRAINING LSTM CHALLENGER")
        print("="*60)
        
        # Optuna study
        print(f"\nRunning Optuna optimization with {n_trials} trials...")
        print("This may take a while...")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best validation MAE: {study.best_value:.3f} days")
        
        # Train final models with best parameters
        print("\nTraining final models with best parameters...")
        
        batch_size = self.best_params['batch_size']
        train_dataset = CycleDataset(X_train, y_train)
        val_dataset = CycleDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Mean model
        print("  Training mean prediction model...")
        self.model_mean = self.create_model(
            input_size=X_train.shape[1],
            hidden_size=self.best_params['hidden_size'],
            num_layers=self.best_params['num_layers'],
            dropout=self.best_params['dropout']
        )
        self.train_single_model(
            self.model_mean, train_loader, val_loader,
            epochs=200, learning_rate=self.best_params['learning_rate']
        )
        
        # Lower quantile model (10th percentile)
        print("  Training lower bound model (10th percentile)...")
        self.model_lower = self.create_model(
            input_size=X_train.shape[1],
            hidden_size=self.best_params['hidden_size'],
            num_layers=self.best_params['num_layers'],
            dropout=self.best_params['dropout']
        )
        self.train_single_model(
            self.model_lower, train_loader, val_loader,
            epochs=200, learning_rate=self.best_params['learning_rate'],
            quantile=0.1
        )
        
        # Upper quantile model (90th percentile)
        print("  Training upper bound model (90th percentile)...")
        self.model_upper = self.create_model(
            input_size=X_train.shape[1],
            hidden_size=self.best_params['hidden_size'],
            num_layers=self.best_params['num_layers'],
            dropout=self.best_params['dropout']
        )
        self.train_single_model(
            self.model_upper, train_loader, val_loader,
            epochs=200, learning_rate=self.best_params['learning_rate'],
            quantile=0.9
        )
        
        print("✓ LSTM training complete!")
    
    def predict(self, X, return_intervals=True):
        """
        Predict cycle lengths with optional confidence intervals.
        
        Args:
            X: Features
            return_intervals: Whether to return 10th and 90th percentile predictions
            
        Returns:
            dict or array: Predictions (and intervals if requested)
        """
        self.model_mean.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            pred_mean = self.model_mean(X_tensor).cpu().numpy()
            
            if return_intervals:
                self.model_lower.eval()
                self.model_upper.eval()
                
                pred_lower = self.model_lower(X_tensor).cpu().numpy()
                pred_upper = self.model_upper(X_tensor).cpu().numpy()
                
                return {
                    'mean': pred_mean,
                    'lower': pred_lower,
                    'upper': pred_upper
                }
        
        return pred_mean
    
    def save(self, directory):
        """Save all models."""
        os.makedirs(directory, exist_ok=True)
        
        torch.save(self.model_mean.state_dict(), f'{directory}/lstm_mean.pt')
        torch.save(self.model_lower.state_dict(), f'{directory}/lstm_lower.pt')
        torch.save(self.model_upper.state_dict(), f'{directory}/lstm_upper.pt')
        
        # Save architecture params
        params = {
            'input_size': self.model_mean.lstm.input_size,
            'best_params': self.best_params
        }
        
        with open(f'{directory}/lstm_params.json', 'w') as f:
            json.dump(params, f, indent=2)


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
    
    # Load processed data
    print("\nLoading processed data...")
    data = np.load('../data/processed/cycles_processed.npz', allow_pickle=True)
    
    X = data['X']
    y = data['y']
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle time series
    )
    
    # Further split train into train/val for LSTM
    X_train_lstm, X_val, y_train_lstm, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )
    
    print(f"Train: {len(y_train)} samples")
    print(f"  LSTM Train: {len(y_train_lstm)} samples")
    print(f"  LSTM Val: {len(y_val)} samples")
    print(f"Test: {len(y_test)} samples")
    
    # Train Random Forest
    rf_trainer = RandomForestTrainer(n_jobs=-1)
    rf_trainer.train(X_train, y_train, use_gridsearch=True)
    rf_trainer.save('../models/random_forest')
    
    # Train LSTM
    lstm_trainer = LSTMTrainer()
    lstm_trainer.train(X_train_lstm, y_train_lstm, X_val, y_val, n_trials=500)
    lstm_trainer.save('../models/lstm')
    
    # Save test data for evaluation
    np.savez('../data/processed/test_data.npz',
             X_test=X_test,
             y_test=y_test)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nModels saved:")
    print("  - Random Forest: ../models/random_forest/")
    print("  - LSTM: ../models/lstm/")
    print("\nNext step: Run evaluate.py to assess model performance")


if __name__ == '__main__':
    main()
