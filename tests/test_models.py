"""
Model Tests

Unit tests for all models.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataLoader:
    """DataLoader tests."""
    
    def test_generate_synthetic_data(self):
        from src.data.loader import DataLoader
        
        loader = DataLoader()
        df = loader.generate_synthetic_data(
            start_date="2023-01-01",
            end_date="2023-01-31",
            num_zones=3
        )
        
        # Check shape
        assert len(df) > 0
        assert 'datetime' in df.columns
        assert 'density' in df.columns
        assert 'zone_id' in df.columns
        
        # Check zones
        assert df['zone_id'].nunique() == 3
        
        # Check density is positive
        assert (df['density'] >= 0).all()
    
    def test_data_summary(self):
        from src.data.loader import DataLoader
        
        loader = DataLoader()
        df = loader.generate_synthetic_data(num_zones=1)
        summary = loader.get_data_summary(df)
        
        assert 'total_records' in summary
        assert 'density_stats' in summary
        assert summary['zones'] == 1


class TestDataPreprocessor:
    """DataPreprocessor tests."""
    
    def test_time_series_split(self):
        from src.data.loader import DataLoader
        from src.data.preprocessor import DataPreprocessor
        
        loader = DataLoader()
        df = loader.generate_synthetic_data(num_zones=1)
        
        preprocessor = DataPreprocessor()
        train, val, test = preprocessor.time_series_split(df)
        
        # Check proportions
        total = len(train) + len(val) + len(test)
        assert abs(total - len(df)) < 5  # Allow small rounding differences
        
        # Check chronological order
        assert train['datetime'].max() < val['datetime'].min()
        assert val['datetime'].max() < test['datetime'].min()
    
    def test_handle_outliers(self):
        from src.data.preprocessor import DataPreprocessor
        
        # Create data with outlier
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=100, freq='h'),
            'density': np.concatenate([
                np.random.normal(100, 10, 99),
                [500]  # Outlier
            ])
        })
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_outliers(df, method='clip')
        
        # Outlier should be clipped
        assert df_clean['density'].max() < 500


class TestFeatureEngineer:
    """FeatureEngineer tests."""
    
    def test_add_time_features(self):
        from src.data.feature_engineer import FeatureEngineer
        
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=24, freq='h'),
            'density': np.random.uniform(50, 150, 24)
        })
        
        engineer = FeatureEngineer()
        df = engineer.add_time_features(df)
        
        assert 'hour' in df.columns
        assert 'day_of_week' in df.columns
        assert 'is_weekend' in df.columns
        assert 'is_rush_hour' in df.columns
    
    def test_add_lag_features(self):
        from src.data.feature_engineer import FeatureEngineer
        
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=100, freq='h'),
            'density': np.random.uniform(50, 150, 100)
        })
        
        engineer = FeatureEngineer()
        df = engineer.add_lag_features(df, lags=[1, 24])
        
        assert 'density_lag_1' in df.columns
        assert 'density_lag_24' in df.columns


class TestBaselineModels:
    """Baseline model tests."""
    
    def test_naive_model(self):
        from src.models.baseline import NaiveModel
        
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=100, freq='h'),
            'density': np.random.uniform(50, 150, 100)
        })
        
        model = NaiveModel()
        model.fit(df)
        
        assert model.is_fitted
        
        predictions = model.predict(5)
        assert len(predictions) == 5
        assert all(p == model.last_value for p in predictions)
    
    def test_moving_average_model(self):
        from src.models.baseline import MovingAverageModel
        
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=100, freq='h'),
            'density': np.random.uniform(50, 150, 100)
        })
        
        model = MovingAverageModel(window=10)
        model.fit(df)
        
        predictions = model.predict(5)
        assert len(predictions) == 5
    
    def test_arima_model(self):
        from src.models.baseline import ARIMAModel
        
        # Stationary series
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=200, freq='h'),
            'density': np.random.uniform(90, 110, 200)
        })
        
        model = ARIMAModel(order=(1, 0, 1))
        model.fit(df)
        
        predictions = model.predict(5)
        assert len(predictions) == 5


class TestMetrics:
    """Metrics tests."""
    
    def test_mape(self):
        from src.evaluation.metrics import ModelMetrics
        
        metrics = ModelMetrics()
        
        y_true = np.array([100, 100, 100, 100])
        y_pred = np.array([90, 110, 95, 105])
        
        mape = metrics.mape(y_true, y_pred)
        
        # Expected: (10 + 10 + 5 + 5) / 4 = 7.5%
        assert abs(mape - 7.5) < 0.1
    
    def test_r2_perfect(self):
        from src.evaluation.metrics import ModelMetrics
        
        metrics = ModelMetrics()
        
        y_true = np.array([100, 120, 80, 90])
        y_pred = y_true.copy()
        
        r2 = metrics.r2_score(y_true, y_pred)
        assert r2 == 1.0
    
    def test_calculate_all(self):
        from src.evaluation.metrics import ModelMetrics
        
        metrics = ModelMetrics()
        
        y_true = np.array([100, 120, 80, 90, 110])
        y_pred = np.array([102, 118, 82, 88, 112])
        
        all_metrics = metrics.calculate_all(y_true, y_pred)
        
        assert 'mae' in all_metrics
        assert 'mse' in all_metrics
        assert 'rmse' in all_metrics
        assert 'mape' in all_metrics
        assert 'r2' in all_metrics
    
    def test_check_targets(self):
        from src.evaluation.metrics import ModelMetrics
        
        metrics = ModelMetrics()
        
        # Good metrics
        good_metrics = {'mape': 10.0, 'r2': 0.90}
        targets = metrics.check_targets(good_metrics)
        assert targets['overall_pass'] == True
        
        # Bad metrics
        bad_metrics = {'mape': 20.0, 'r2': 0.70}
        targets = metrics.check_targets(bad_metrics)
        assert targets['overall_pass'] == False


class TestCalendarFeatures:
    """Calendar features tests."""
    
    def test_is_holiday(self):
        from src.external.calendar import CalendarFeatures
        
        calendar = CalendarFeatures(country='US')
        
        # New Year
        assert calendar.is_holiday(datetime(2023, 1, 1)) == True
        
        # Regular day
        assert calendar.is_holiday(datetime(2023, 6, 15)) == False
    
    def test_add_calendar_features(self):
        from src.external.calendar import CalendarFeatures
        
        calendar = CalendarFeatures(country='US')
        
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=48, freq='h')
        })
        
        df = calendar.add_calendar_features(df)
        
        assert 'is_holiday' in df.columns
        assert 'is_rush_hour' in df.columns
        assert 'is_weekend' in df.columns


class TestWeatherAPI:
    """Weather API tests."""
    
    def test_generate_historical_weather(self):
        from src.external.weather_api import WeatherAPI
        
        api = WeatherAPI()
        
        df = api.generate_historical_weather(
            start_date="2023-01-01",
            end_date="2023-01-07"
        )
        
        assert len(df) > 0
        assert 'temperature' in df.columns
        assert 'precipitation' in df.columns


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
