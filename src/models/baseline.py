"""
Baseline Models Module

Karşılaştırma için baseline modeller.
Bu modeller yeni geliştirilen modellerin performansını değerlendirmek için kullanılır.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Tüm modeller için abstract base class."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.train_history = []
        
    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> 'BaseModel':
        """Modeli eğitir."""
        pass
    
    @abstractmethod
    def predict(self, n_steps: int) -> np.ndarray:
        """n_steps adım ileriye tahmin yapar."""
        pass
    
    def get_name(self) -> str:
        return self.name


class NaiveModel(BaseModel):
    """
    Naive (Basit) Model
    
    Strateji: Son gözlenen değeri tahmin olarak kullan.
    En basit baseline - herhangi bir model bundan iyi olmalı.
    """
    
    def __init__(self):
        super().__init__("Naive")
        self.last_value = None
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'density') -> 'NaiveModel':
        """
        Son değeri kaydeder.
        
        Args:
            train_data: Eğitim verisi
            target_col: Hedef sütun
        """
        self.last_value = train_data[target_col].iloc[-1]
        self.is_fitted = True
        logger.info(f"Naive model fitted. Last value: {self.last_value:.2f}")
        return self
    
    def predict(self, n_steps: int = 1) -> np.ndarray:
        """Son değeri n_steps kez tekrarlar."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.full(n_steps, self.last_value)


class SeasonalNaiveModel(BaseModel):
    """
    Seasonal Naive Model
    
    Strateji: Geçen haftanın aynı saatindeki değeri kullan.
    Haftalık seasonality yakalamak için ideal baseline.
    """
    
    def __init__(self, seasonal_period: int = 168):  # 168 = 24*7 (haftalık)
        super().__init__("Seasonal Naive")
        self.seasonal_period = seasonal_period
        self.seasonal_values = None
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'density') -> 'SeasonalNaiveModel':
        """
        Son seasonal_period değeri kaydeder.
        
        Args:
            train_data: Eğitim verisi
            target_col: Hedef sütun
        """
        values = train_data[target_col].values
        if len(values) >= self.seasonal_period:
            self.seasonal_values = values[-self.seasonal_period:]
        else:
            # Veri yetersizse, mevcut veriyi tekrarla
            self.seasonal_values = np.tile(values, 
                                          int(np.ceil(self.seasonal_period / len(values))))[:self.seasonal_period]
        self.is_fitted = True
        logger.info(f"Seasonal Naive model fitted. Period: {self.seasonal_period}")
        return self
    
    def predict(self, n_steps: int = 1) -> np.ndarray:
        """Seasonal pattern'ı tekrarlayarak tahmin yapar."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        for i in range(n_steps):
            idx = i % self.seasonal_period
            predictions.append(self.seasonal_values[idx])
        return np.array(predictions)


class MovingAverageModel(BaseModel):
    """
    Moving Average (Hareketli Ortalama) Model
    
    Strateji: Son n değerin ortalamasını tahmin olarak kullan.
    Kısa vadeli noise'u filtrelemek için kullanışlı.
    """
    
    def __init__(self, window: int = 24):
        super().__init__(f"Moving Average (window={window})")
        self.window = window
        self.values = None
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'density') -> 'MovingAverageModel':
        """
        Son window değeri kaydeder.
        
        Args:
            train_data: Eğitim verisi
            target_col: Hedef sütun
        """
        self.values = train_data[target_col].values[-self.window:]
        self.is_fitted = True
        avg = np.mean(self.values)
        logger.info(f"Moving Average model fitted. Window avg: {avg:.2f}")
        return self
    
    def predict(self, n_steps: int = 1) -> np.ndarray:
        """Hareketli ortalamayı tahmin olarak döndürür."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        current_values = list(self.values)
        
        for _ in range(n_steps):
            pred = np.mean(current_values[-self.window:])
            predictions.append(pred)
            current_values.append(pred)
            
        return np.array(predictions)


class ARIMAModel(BaseModel):
    """
    ARIMA Model
    
    AutoRegressive Integrated Moving Average.
    Klasik zaman serisi tahmin modeli.
    
    Parametreler:
    - p: AR order (autoregressive)
    - d: Differencing order
    - q: MA order (moving average)
    """
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 2)):
        super().__init__(f"ARIMA{order}")
        self.order = order
        self.model = None
        self.fitted_model = None
        
    def check_stationarity(self, data: np.ndarray) -> bool:
        """
        ADF testi ile durağanlık kontrolü.
        
        Args:
            data: Test edilecek zaman serisi
            
        Returns:
            bool: Durağan mı?
        """
        result = adfuller(data, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < 0.05
        logger.info(f"ADF test p-value: {p_value:.4f}. Stationary: {is_stationary}")
        return is_stationary
    
    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str = 'density',
        auto_order: bool = False
    ) -> 'ARIMAModel':
        """
        ARIMA modelini eğitir.
        
        Args:
            train_data: Eğitim verisi
            target_col: Hedef sütun
            auto_order: Otomatik order seçimi (henüz implement edilmedi)
        """
        values = train_data[target_col].values
        
        # Durağanlık kontrolü
        self.check_stationarity(values)
        
        # Model eğitimi
        try:
            self.model = ARIMA(values, order=self.order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            # Model özeti
            aic = self.fitted_model.aic
            bic = self.fitted_model.bic
            logger.info(f"ARIMA{self.order} fitted. AIC: {aic:.2f}, BIC: {bic:.2f}")
            
        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            raise
            
        return self
    
    def predict(self, n_steps: int = 1) -> np.ndarray:
        """n_steps adım ileriye tahmin yapar."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        forecast = self.fitted_model.forecast(steps=n_steps)
        return forecast
    
    def get_residuals(self) -> np.ndarray:
        """Model residual'larını döndürür (model diagnostics için)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.fitted_model.resid


def compare_baseline_models(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_col: str = 'density'
) -> Dict[str, Dict[str, float]]:
    """
    Tüm baseline modelleri karşılaştırır.
    
    Args:
        train_data: Eğitim verisi
        test_data: Test verisi
        target_col: Hedef sütun
        
    Returns:
        Dict: Her model için metrikler
    """
    from src.evaluation.metrics import ModelMetrics
    
    models = [
        NaiveModel(),
        SeasonalNaiveModel(),
        MovingAverageModel(window=24),
        ARIMAModel(order=(5, 1, 2))
    ]
    
    results = {}
    n_steps = len(test_data)
    y_true = test_data[target_col].values
    
    for model in models:
        try:
            model.fit(train_data, target_col)
            y_pred = model.predict(n_steps)
            
            metrics = ModelMetrics()
            model_metrics = metrics.calculate_all(y_true, y_pred)
            results[model.get_name()] = model_metrics
            
            logger.info(f"{model.get_name()}: MAPE={model_metrics['mape']:.2f}%, "
                       f"RMSE={model_metrics['rmse']:.2f}")
        except Exception as e:
            logger.error(f"Failed to evaluate {model.get_name()}: {e}")
            results[model.get_name()] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Test
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    
    # Veri yükle
    loader = DataLoader()
    df = loader.generate_synthetic_data(num_zones=1)
    
    # Ön işleme
    preprocessor = DataPreprocessor()
    train, val, test = preprocessor.time_series_split(df)
    
    # Baseline modelleri test et
    print("\n=== Baseline Model Tests ===\n")
    
    # Naive
    naive = NaiveModel()
    naive.fit(train)
    print(f"Naive prediction (5 steps): {naive.predict(5)}")
    
    # Moving Average
    ma = MovingAverageModel(window=24)
    ma.fit(train)
    print(f"MA prediction (5 steps): {ma.predict(5)}")
    
    # ARIMA
    arima = ARIMAModel(order=(2, 1, 1))
    arima.fit(train)
    print(f"ARIMA prediction (5 steps): {arima.predict(5)}")
