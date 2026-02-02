"""
Evaluation & Visualization for Vesta
=====================================
This script evaluates Random Forest models, calculates metrics,
and generates visualizations comparing predictions to actual cycle dates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from pathlib import Path
from datetime import datetime


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_random_forest_models(directory):
    """Load Random Forest models."""
    directory = Path(directory)
    return {
        'mean': joblib.load(directory / 'rf_mean.pkl'),
        'lower': joblib.load(directory / 'rf_lower.pkl'),
        'upper': joblib.load(directory / 'rf_upper.pkl')
    }


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

def plot_predictions_comparison(y_true, rf_preds, save_path):
    """
    Plot comparing predicted vs actual cycle lengths.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest - Mean predictions
    ax = axes[0]
    ax.scatter(y_true, rf_preds['mean'], alpha=0.6, s=80, edgecolors='k', linewidth=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Cycle Length (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Cycle Length (days)', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest: Predicted vs Actual', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residuals - Random Forest
    ax = axes[1]
    residuals_rf = y_true - rf_preds['mean']
    ax.scatter(rf_preds['mean'], residuals_rf, alpha=0.6, s=80, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Cycle Length (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals (days)', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest: Residual Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_time_series_predictions(dates, y_true, rf_preds, save_path):
    """
    Plot predictions over time with confidence intervals.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    # Convert numpy dates to datetime if needed
    if isinstance(dates[0], np.datetime64):
        dates = pd.to_datetime(dates)
    
    # Random Forest
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
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_error_distribution(y_true, rf_preds, save_path):
    """
    Plot distribution of prediction errors.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    rf_errors = y_true - rf_preds['mean']
    
    # Random Forest
    ax.hist(rf_errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=rf_errors.mean(), color='green', linestyle='--', linewidth=2, 
               label=f'Mean Error: {rf_errors.mean():.2f} days')
    ax.set_xlabel('Prediction Error (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest: Error Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_interval_analysis(y_true, rf_preds, save_path):
    """
    Analyze confidence interval quality.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Random Forest interval widths
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
    
    package_root = Path(__file__).resolve().parents[1]
    data_dir = package_root / 'data'
    models_dir = package_root / 'models'
    results_dir = package_root / 'results'
    
    test_data_path = data_dir / 'processed' / 'test_data.npz'
    processed_data_path = data_dir / 'processed' / 'cycles_processed.npz'
    rf_model_dir = models_dir / 'random_forest'
    
    # Create output directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print("\nLoading test data...")
    test_data = np.load(test_data_path)
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    # Load dates for visualization
    full_data = np.load(processed_data_path, allow_pickle=True)
    dates = full_data['dates'][-len(y_test):]  # Get dates corresponding to test set
    
    print(f"Test samples: {len(y_test)}")
    
    # Load models
    print("\nLoading models...")
    rf_models = load_random_forest_models(rf_model_dir)
    print("✓ Models loaded")
    
    # Get predictions
    print("\nGenerating predictions...")
    rf_preds = predict_rf(rf_models, X_test)
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
    
    # Save metrics to file
    metrics_summary = {
        'random_forest': {
            **rf_metrics,
            'coverage': float(rf_coverage),
            'avg_interval_width': float(np.mean(rf_preds['upper'] - rf_preds['lower']))
        }
    }
    
    metrics_path = results_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_predictions_comparison(y_test, rf_preds, results_dir / 'predictions_comparison.png')
    plot_time_series_predictions(dates, y_test, rf_preds, results_dir / 'time_series.png')
    plot_error_distribution(y_test, rf_preds, results_dir / 'error_distribution.png')
    plot_interval_analysis(y_test, rf_preds, results_dir / 'interval_analysis.png')
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {results_dir}")
    print("  - metrics.json")
    print("  - predictions_comparison.png")
    print("  - time_series.png")
    print("  - error_distribution.png")
    print("  - interval_analysis.png")


if __name__ == '__main__':
    main()
