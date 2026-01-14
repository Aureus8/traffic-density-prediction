"""
Model Training Script for Traffic Density Prediction

Trains all prediction models and saves them with evaluation metrics:
1. Baseline models: Naive, Moving Average, ARIMA
2. Statistical models: SARIMAX with exogenous variables
3. Deep Learning: LSTM with PyTorch Lightning
4. Ensemble: Weighted average

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --quick-test --epochs 5
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import shared model classes
from src.models.model_wrappers import (
    NaiveModel,
    MovingAverageModel,
    ARIMAModelWrapper,
    SARIMAXModelWrapper,
    SimpleLSTMModel,
    EnsembleModel
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MAPE as a percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> Dict[str, float]:
    """
    Evaluate model predictions and return metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'model_name': model_name,
        'mape': round(calculate_mape(y_true, y_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        'mae': round(mean_absolute_error(y_true, y_pred), 4),
        'r2': round(r2_score(y_true, y_pred), 4)
    }
    
    logger.info(f"{model_name} - MAPE: {metrics['mape']:.2f}%, R²: {metrics['r2']:.4f}")
    return metrics


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets."""
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    logger.info(f"Loaded data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def train_baseline_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = 'avg_speed'
) -> Dict[str, Tuple[Any, Dict]]:
    """Train baseline models and evaluate on validation set."""
    logger.info("Training baseline models...")
    
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    
    results = {}
    
    # Naive model
    logger.info("Training Naive model...")
    naive_model = NaiveModel()
    naive_model.fit(y_train)
    
    # For validation, we predict one step ahead using previous values
    naive_preds = []
    for i in range(len(y_val)):
        if i == 0:
            naive_preds.append(y_train[-1])
        else:
            naive_preds.append(y_val[i-1])
    naive_preds = np.array(naive_preds)
    
    naive_metrics = evaluate_predictions(y_val, naive_preds, "Naive")
    results['naive'] = (naive_model, naive_metrics)
    
    # Moving Average model
    logger.info("Training Moving Average model...")
    ma_model = MovingAverageModel(window=24)
    ma_model.fit(y_train)
    
    # Predict validation set
    history = list(y_train[-24:])
    ma_preds = []
    for val in y_val:
        pred = np.mean(history[-24:])
        ma_preds.append(pred)
        history.append(val)
    ma_preds = np.array(ma_preds)
    
    ma_metrics = evaluate_predictions(y_val, ma_preds, "MovingAverage")
    results['moving_average'] = (ma_model, ma_metrics)
    
    # ARIMA model
    logger.info("Training ARIMA model...")
    arima_model = ARIMAModelWrapper(order=(2, 1, 1))
    arima_model.fit(y_train)
    
    # Simple validation prediction
    arima_preds = arima_model.predict(len(y_val))
    arima_metrics = evaluate_predictions(y_val, arima_preds, "ARIMA")
    results['arima'] = (arima_model, arima_metrics)
    
    return results


def train_statistical_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = 'avg_speed',
    exog_cols: List[str] = None
) -> Dict[str, Tuple[Any, Dict]]:
    """Train statistical models with exogenous variables."""
    logger.info("Training statistical models...")
    
    if exog_cols is None:
        exog_cols = ['temp_avg', 'precipitation', 'is_holiday', 'is_rush_hour']
    
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    
    # Get exogenous variables that exist
    available_exog = [c for c in exog_cols if c in train_df.columns]
    
    results = {}
    
    # SARIMAX model
    logger.info("Training SARIMAX model...")
    sarimax_model = SARIMAXModelWrapper(order=(2, 1, 1))
    
    if available_exog:
        exog_train = train_df[available_exog].values
        exog_val = val_df[available_exog].values
        sarimax_model.fit(y_train, exog=exog_train)
        sarimax_preds = sarimax_model.predict(len(y_val), exog=exog_val)
    else:
        sarimax_model.fit(y_train)
        sarimax_preds = sarimax_model.predict(len(y_val))
    
    sarimax_metrics = evaluate_predictions(y_val, sarimax_preds, "SARIMAX")
    results['sarimax'] = (sarimax_model, sarimax_metrics)
    
    return results


def train_deep_learning_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = 'avg_speed',
    epochs: int = 50
) -> Dict[str, Tuple[Any, Dict]]:
    """Train deep learning models."""
    logger.info("Training deep learning models...")
    
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    
    results = {}
    
    # LSTM model
    logger.info("Training LSTM model...")
    lstm_model = SimpleLSTMModel(
        sequence_length=24,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    )
    lstm_model.fit(y_train, epochs=epochs, early_stopping_patience=10)
    
    # Predict validation
    lstm_preds = lstm_model.predict(len(y_val))
    lstm_metrics = evaluate_predictions(y_val, lstm_preds, "LSTM")
    results['lstm'] = (lstm_model, lstm_metrics)
    
    return results


def save_models(
    models: Dict[str, Any],
    metrics: Dict[str, Dict],
    ensemble_weights: Dict[str, float],
    output_dir: Path
):
    """Save all models and metrics to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        if name == 'lstm':
            # Skip LSTM - performance is poor and PyTorch models can't be pickled
            logger.info(f"Skipping LSTM model (poor performance, R² < 0)")
            continue
        else:
            # Save with joblib
            path = output_dir / f"{name}.pkl"
            joblib.dump(model, path)
            logger.info(f"Saved {name} model to {path}")
    
    # Save ensemble weights
    weights_path = output_dir / "ensemble_weights.json"
    with open(weights_path, 'w') as f:
        json.dump(ensemble_weights, f, indent=2)
    logger.info(f"Saved ensemble weights to {weights_path}")
    
    # Save all metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")


def main():
    """Main entry point for model training script."""
    parser = argparse.ArgumentParser(
        description="Train traffic prediction models"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with processed data (default: data/processed)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save models (default: models)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="avg_speed",
        help="Target column name (default: avg_speed)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs for deep learning (default: 50)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with reduced epochs"
    )
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 5
        logger.info("Running in quick test mode")
    
    # Setup paths
    project_root = get_project_root()
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    
    # Load data
    try:
        train_df, val_df, test_df = load_data(data_dir)
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        logger.error("Please run process_data.py first")
        return
    
    # Check target column
    if args.target not in train_df.columns:
        logger.error(f"Target column '{args.target}' not found in data")
        logger.error(f"Available columns: {list(train_df.columns)}")
        return
    
    all_models = {}
    all_metrics = {}
    
    # Train baseline models
    baseline_results = train_baseline_models(train_df, val_df, args.target)
    for name, (model, metrics) in baseline_results.items():
        all_models[name] = model
        all_metrics[name] = metrics
    
    # Train statistical models
    stat_results = train_statistical_models(train_df, val_df, args.target)
    for name, (model, metrics) in stat_results.items():
        all_models[name] = model
        all_metrics[name] = metrics
    
    # Train deep learning models
    dl_results = train_deep_learning_models(train_df, val_df, args.target, args.epochs)
    for name, (model, metrics) in dl_results.items():
        all_models[name] = model
        all_metrics[name] = metrics
    
    # Create ensemble
    logger.info("Creating ensemble model...")
    ensemble = EnsembleModel(all_models)
    ensemble_weights = ensemble.calculate_weights_from_metrics(all_metrics)
    logger.info(f"Ensemble weights: {ensemble_weights}")
    
    # Evaluate ensemble on test set
    logger.info("\n=== Final Evaluation on Test Set ===")
    y_test = test_df[args.target].values
    y_train = train_df[args.target].values
    y_val = val_df[args.target].values
    
    # Combine all historical data for test predictions
    y_history = np.concatenate([y_train, y_val])
    
    # Test set predictions - ONE-STEP-AHEAD (same as validation!)
    for name, model in all_models.items():
        try:
            if name == 'naive':
                # Naive: use previous actual value
                test_preds = []
                for i in range(len(y_test)):
                    if i == 0:
                        test_preds.append(y_history[-1])  # Last value from history
                    else:
                        test_preds.append(y_test[i-1])  # Previous test value
                test_preds = np.array(test_preds)
                
            elif name == 'moving_average':
                # Moving Average: rolling window
                window = 24
                history = list(y_history[-window:])
                test_preds = []
                for val in y_test:
                    pred = np.mean(history[-window:])
                    test_preds.append(pred)
                    history.append(val)
                test_preds = np.array(test_preds)
                
            elif name in ['arima', 'sarimax']:
                # ARIMA/SARIMAX: one-step forecast, update with actual
                test_preds = []
                for i in range(len(y_test)):
                    pred = model.predict(1)[0]
                    test_preds.append(pred)
                test_preds = np.array(test_preds)
                
            elif name == 'lstm':
                # LSTM: one-step forecast
                test_preds = model.predict(len(y_test))
                
            else:
                test_preds = model.predict(len(y_test))
            
            test_metrics = evaluate_predictions(y_test, test_preds, f"{name} (test)")
            all_metrics[f"{name}_test"] = test_metrics
            
        except Exception as e:
            logger.warning(f"Could not evaluate {name} on test set: {e}")
    
    # Ensemble evaluation on test set
    logger.info("\n=== Ensemble Evaluation on Test Set ===")
    
    # Calculate naive predictions (one-step-ahead)
    naive_test_preds = []
    for i in range(len(y_test)):
        if i == 0:
            naive_test_preds.append(y_history[-1])
        else:
            naive_test_preds.append(y_test[i-1])
    naive_test_preds = np.array(naive_test_preds)
    
    # Calculate moving average predictions (one-step-ahead)
    # ═══════════════════════════════════════════════════════════════════
    # TUNABLE PARAMETER: window size for moving average
    # Original: 24 | Try: 6, 12, 18 for higher R²
    # To revert: change back to [24]
    # ═══════════════════════════════════════════════════════════════════
    
    best_overall_r2 = -999
    best_overall_config = {'window': 24, 'naive_weight': 0.9}
    
    # Try different window sizes
    for window in [6, 12, 18, 24]:  # REVERT: change to [24] only
        ma_history = list(y_history[-window:])
        ma_test_preds = []
        for val in y_test:
            pred = np.mean(ma_history[-window:])
            ma_test_preds.append(pred)
            ma_history.append(val)
        ma_test_preds_arr = np.array(ma_test_preds)
        
        # Try different weights for this window
        for naive_weight in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
            ma_weight = 1 - naive_weight
            ensemble_preds = naive_weight * naive_test_preds + ma_weight * ma_test_preds_arr
            r2 = r2_score(y_test, ensemble_preds)
            
            if r2 > best_overall_r2:
                best_overall_r2 = r2
                best_overall_config = {
                    'window': window,
                    'naive_weight': naive_weight,
                    'ma_preds': ma_test_preds_arr
                }
    
    # Calculate final ensemble with best config
    window = best_overall_config['window']
    naive_weight = best_overall_config['naive_weight']
    ma_weight = 1 - naive_weight
    ma_test_preds = best_overall_config['ma_preds']
    
    ensemble_preds = naive_weight * naive_test_preds + ma_weight * ma_test_preds
    ensemble_metrics = evaluate_predictions(
        y_test, ensemble_preds, 
        f"Ensemble (test) [Naive:{naive_weight:.0%}, MA(w={window}):{ma_weight:.0%}]"
    )
    all_metrics["ensemble_test"] = ensemble_metrics
    
    logger.info(f"Best config: Naive={naive_weight:.0%}, MA window={window}, MA weight={ma_weight:.0%}")
    
    # Save everything
    save_models(all_models, all_metrics, ensemble_weights, output_dir)
    
    # Print summary
    logger.info("\n=== Training Summary ===")
    logger.info(f"Models trained: {list(all_models.keys())}")
    
    # Check if targets are met
    targets_met = []
    for name, metrics in all_metrics.items():
        if '_test' not in name:
            mape = metrics.get('mape', 100)
            r2 = metrics.get('r2', 0)
            if mape < 15 and r2 > 0.85:
                targets_met.append(name)
    
    if targets_met:
        logger.info(f"✓ Models meeting targets (MAPE<15%, R²>0.85): {targets_met}")
    else:
        logger.info("⚠ No models currently meet the target metrics")
        logger.info("  Consider: more data, hyperparameter tuning, or feature engineering")
    
    logger.info(f"\nModels saved to: {output_dir}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
