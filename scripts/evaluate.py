"""
Evaluation & Visualization for Vesta
=====================================
This script evaluates both Random Forest and LSTM models, calculates metrics,
and generates visualizations comparing predictions to actual cycle dates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import joblib
import json
import os
from datetime import datetime, timedelta


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


# ============================================================================
# MODEL LOADING
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM model architecture (must match train.py)."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out.squeeze()


def load_random_forest_models(directory):
    """Load Random Forest models."""
    return {
        'mean': joblib.load(f'{directory}/rf_mean.pkl'),
        'lower': joblib.load(f'{directory}/rf_lower.pkl'),
        'upper': joblib.load(f'{directory}/rf_upper.pkl')
    }


def load_lstm_models(directory, device):
    """Load LSTM models."""
    # Load parameters
    with open(f'{directory}/lstm_params.json', 'r') as f:
        params = json.load(f)
    
    input_size = params['input_size']
    best_params = params['best_params']
    
    # Create models
    models = {}
    for model_type in ['mean', 'lower', 'upper']:
        model = LSTMModel(
            input_size=input_size,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout']
        ).to(device)
        
        model.load_state_dict(torch.load(f'{directory}/lstm_{model_type}.pt'))
        model.eval()
        
        models[model_type] = model
    
    return models


# ============================================================================
# PREDICTION & METRICS
# ============================================================================

def predict_rf(models, X):
    """Get predictions from Random Forest models."""
    return {
        'mean': models['mean'].predict(X),
        'lower': models['lower'].predict(X),
        'upper': models['upper'].predict(X)
    }


def predict_lstm(models, X, device):
    """Get predictions from LSTM models."""
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        predictions = {}
        for model_type, model in models.items():
            pred = model(X_tensor).cpu().numpy()
            predictions[model_type] = pred
    
    return predictions


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def calculate_interval_coverage(y_true, y_lower, y_upper, target_coverage=0.8):
    """
    Calculate what percentage of actual values fall within predicted intervals.
    Target is 80% (10th to 90th percentile).
    """
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    return coverage


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_predictions_comparison(y_true, rf_preds, lstm_preds, save_path):
    """
    Plot comparing predicted vs actual cycle lengths.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Random Forest - Mean predictions
    ax = axes[0, 0]
    ax.scatter(y_true, rf_preds['mean'], alpha=0.6, s=80, edgecolors='k', linewidth=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Cycle Length (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Cycle Length (days)', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest: Predicted vs Actual', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # LSTM - Mean predictions
    ax = axes[0, 1]
    ax.scatter(y_true, lstm_preds['mean'], alpha=0.6, s=80, 
               edgecolors='k', linewidth=0.5, color='orange')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Cycle Length (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Cycle Length (days)', fontsize=12, fontweight='bold')
    ax.set_title('LSTM: Predicted vs Actual', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residuals - Random Forest
    ax = axes[1, 0]
    residuals_rf = y_true - rf_preds['mean']
    ax.scatter(rf_preds['mean'], residuals_rf, alpha=0.6, s=80, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Cycle Length (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals (days)', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest: Residual Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Residuals - LSTM
    ax = axes[1, 1]
    residuals_lstm = y_true - lstm_preds['mean']
    ax.scatter(lstm_preds['mean'], residuals_lstm, alpha=0.6, s=80, 
               edgecolors='k', linewidth=0.5, color='orange')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Cycle Length (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals (days)', fontsize=12, fontweight='bold')
    ax.set_title('LSTM: Residual Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_time_series_predictions(dates, y_true, rf_preds, lstm_preds, save_path):
    """
    Plot predictions over time with confidence intervals.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Convert numpy dates to datetime if needed
    if isinstance(dates[0], np.datetime64):
        dates = pd.to_datetime(dates)
    
    # Random Forest
    ax = axes[0]
    ax.plot(dates, y_true, 'o-', label='Actual', linewidth=2, markersize=8, color='black')
    ax.plot(dates, rf_preds['mean'], 's-', label='Predicted (Mean)', 
            linewidth=2, markersize=6, color='blue', alpha=0.7)
    ax.fill_between(dates, rf_preds['lower'], rf_preds['upper'], 
                     alpha=0.2, color='blue', label='80% Confidence Interval')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cycle Length (days)', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest: Time Series Prediction', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # LSTM
    ax = axes[1]
    ax.plot(dates, y_true, 'o-', label='Actual', linewidth=2, markersize=8, color='black')
    ax.plot(dates, lstm_preds['mean'], 's-', label='Predicted (Mean)', 
            linewidth=2, markersize=6, color='orange', alpha=0.7)
    ax.fill_between(dates, lstm_preds['lower'], lstm_preds['upper'], 
                     alpha=0.2, color='orange', label='80% Confidence Interval')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cycle Length (days)', fontsize=12, fontweight='bold')
    ax.set_title('LSTM: Time Series Prediction', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_error_distribution(y_true, rf_preds, lstm_preds, save_path):
    """
    Plot distribution of prediction errors.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    rf_errors = y_true - rf_preds['mean']
    lstm_errors = y_true - lstm_preds['mean']
    
    # Random Forest
    ax = axes[0]
    ax.hist(rf_errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=rf_errors.mean(), color='green', linestyle='--', linewidth=2, 
               label=f'Mean Error: {rf_errors.mean():.2f} days')
    ax.set_xlabel('Prediction Error (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest: Error Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # LSTM
    ax = axes[1]
    ax.hist(lstm_errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=lstm_errors.mean(), color='green', linestyle='--', linewidth=2,
               label=f'Mean Error: {lstm_errors.mean():.2f} days')
    ax.set_xlabel('Prediction Error (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('LSTM: Error Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_interval_analysis(y_true, rf_preds, lstm_preds, save_path):
    """
    Analyze confidence interval quality.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest interval widths
    ax = axes[0]
    rf_widths = rf_preds['upper'] - rf_preds['lower']
    indices = np.arange(len(rf_widths))
    
    colors = ['green' if (y_true[i] >= rf_preds['lower'][i] and y_true[i] <= rf_preds['upper'][i]) 
              else 'red' for i in range(len(y_true))]
    
    ax.bar(indices, rf_widths, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Interval Width (days)', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest: Confidence Interval Width\n(Green = Actual in range)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # LSTM interval widths
    ax = axes[1]
    lstm_widths = lstm_preds['upper'] - lstm_preds['lower']
    
    colors = ['green' if (y_true[i] >= lstm_preds['lower'][i] and y_true[i] <= lstm_preds['upper'][i]) 
              else 'red' for i in range(len(y_true))]
    
    ax.bar(indices, lstm_widths, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Interval Width (days)', fontsize=12, fontweight='bold')
    ax.set_title('LSTM: Confidence Interval Width\n(Green = Actual in range)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    """
    Main evaluation pipeline.
    """
    print("\n" + "="*70)
    print("VESTA MODEL EVALUATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create output directory
    os.makedirs('../results', exist_ok=True)
    
    # Load test data
    print("\nLoading test data...")
    test_data = np.load('../data/processed/test_data.npz')
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    # Load dates for visualization
    full_data = np.load('../data/processed/cycles_processed.npz', allow_pickle=True)
    dates = full_data['dates'][-len(y_test):]  # Get dates corresponding to test set
    
    print(f"Test samples: {len(y_test)}")
    
    # Load models
    print("\nLoading models...")
    rf_models = load_random_forest_models('../models/random_forest')
    lstm_models = load_lstm_models('../models/lstm', device)
    print("✓ Models loaded")
    
    # Get predictions
    print("\nGenerating predictions...")
    rf_preds = predict_rf(rf_models, X_test)
    lstm_preds = predict_lstm(lstm_models, X_test, device)
    print("✓ Predictions generated")
    
    # Calculate metrics
    print("\n" + "="*70)
    print("METRICS")
    print("="*70)
    
    print("\nRANDOM FOREST:")
    rf_metrics = calculate_metrics(y_test, rf_preds['mean'])
    for metric, value in rf_metrics.items():
        print(f"  {metric:10s}: {value:.3f}")
    
    rf_coverage = calculate_interval_coverage(y_test, rf_preds['lower'], rf_preds['upper'])
    print(f"  {'Coverage':10s}: {rf_coverage*100:.1f}% (target: 80%)")
    print(f"  Avg Interval Width: {np.mean(rf_preds['upper'] - rf_preds['lower']):.2f} days")
    
    print("\nLSTM:")
    lstm_metrics = calculate_metrics(y_test, lstm_preds['mean'])
    for metric, value in lstm_metrics.items():
        print(f"  {metric:10s}: {value:.3f}")
    
    lstm_coverage = calculate_interval_coverage(y_test, lstm_preds['lower'], lstm_preds['upper'])
    print(f"  {'Coverage':10s}: {lstm_coverage*100:.1f}% (target: 80%)")
    print(f"  Avg Interval Width: {np.mean(lstm_preds['upper'] - lstm_preds['lower']):.2f} days")
    
    # Winner
    print("\n" + "="*70)
    if lstm_metrics['MAE'] < rf_metrics['MAE']:
        improvement = ((rf_metrics['MAE'] - lstm_metrics['MAE']) / rf_metrics['MAE']) * 100
        print(f"🏆 WINNER: LSTM (MAE improved by {improvement:.1f}%)")
    else:
        improvement = ((lstm_metrics['MAE'] - rf_metrics['MAE']) / lstm_metrics['MAE']) * 100
        print(f"🏆 WINNER: Random Forest (MAE {improvement:.1f}% better)")
    print("="*70)
    
    # Save metrics to file
    metrics_summary = {
        'random_forest': {
            **rf_metrics,
            'coverage': float(rf_coverage),
            'avg_interval_width': float(np.mean(rf_preds['upper'] - rf_preds['lower']))
        },
        'lstm': {
            **lstm_metrics,
            'coverage': float(lstm_coverage),
            'avg_interval_width': float(np.mean(lstm_preds['upper'] - lstm_preds['lower']))
        }
    }
    
    with open('../results/metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print("\n✓ Metrics saved to: ../results/metrics.json")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_predictions_comparison(y_test, rf_preds, lstm_preds, '../results/predictions_comparison.png')
    plot_time_series_predictions(dates, y_test, rf_preds, lstm_preds, '../results/time_series.png')
    plot_error_distribution(y_test, rf_preds, lstm_preds, '../results/error_distribution.png')
    plot_interval_analysis(y_test, rf_preds, lstm_preds, '../results/interval_analysis.png')
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\nResults saved in: ../results/")
    print("  - metrics.json")
    print("  - predictions_comparison.png")
    print("  - time_series.png")
    print("  - error_distribution.png")
    print("  - interval_analysis.png")


if __name__ == '__main__':
    main()
