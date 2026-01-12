"""
Statistical Models Module

Harici değişken destekli istatistiksel modeller.
SARIMAX ve Prophet modelleri.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prophet opsiyonel
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed. ProphetModel will not be available.")


class SARIMAXModel:
    """
    SARIMAX Model
    
    Seasonal ARIMA with eXogenous variables.
    Harici değişkenleri (hava durumu, tatil vb.) modele entegre eder.
    
    Parametreler:
    - order: (p, d, q) - ARIMA parametreleri
    - seasonal_order: (P, D, Q, s) - Seasonal parametreler
    - exog: Harici değişkenler DataFrame
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (2, 1, 2),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
        name: str = None
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.name = name or f"SARIMAX{order}x{seasonal_order}"
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.exog_columns = None
        
    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str = 'density',
        exog_cols: Optional[List[str]] = None,
        verbose: bool = True
    ) -> 'SARIMAXModel':
        """
        SARIMAX modelini eğitir.
        
        Args:
            train_data: Eğitim verisi
            target_col: Hedef sütun
            exog_cols: Harici değişken sütunları (None ise SARIMA olarak çalışır)
            verbose: Eğitim çıktısı göster
        """
        y = train_data[target_col].values
        
        # Harici değişkenler
        exog = None
        if exog_cols:
            exog = train_data[exog_cols].values
            self.exog_columns = exog_cols
            logger.info(f"Using exogenous variables: {exog_cols}")
        
        try:
            self.model = SARIMAX(
                y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.fitted_model = self.model.fit(disp=verbose)
            self.is_fitted = True
            
            # Model diagnostics
            aic = self.fitted_model.aic
            bic = self.fitted_model.bic
            logger.info(f"{self.name} fitted. AIC: {aic:.2f}, BIC: {bic:.2f}")
            
        except Exception as e:
            logger.error(f"SARIMAX fitting failed: {e}")
            raise
            
        return self
    
    def predict(
        self,
        n_steps: int = 1,
        exog: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        n_steps adım ileriye tahmin yapar.
        
        Args:
            n_steps: Tahmin adım sayısı
            exog: Tahmin dönemi için harici değişkenler (eğer fit'te kullanıldıysa gerekli)
            
        Returns:
            np.ndarray: Tahminler
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.exog_columns and exog is None:
            raise ValueError("Model was trained with exogenous variables. "
                           "Please provide exog for prediction.")
        
        forecast = self.fitted_model.forecast(steps=n_steps, exog=exog)
        return forecast
    
    def get_summary(self) -> str:
        """Model özetini döndürür."""
        if not self.is_fitted:
            return "Model not fitted."
        return str(self.fitted_model.summary())
    
    def get_residuals(self) -> np.ndarray:
        """Model residual'larını döndürür."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self.fitted_model.resid
    
    def get_name(self) -> str:
        return self.name


class ProphetModel:
    """
    Facebook Prophet Model
    
    Özellikler:
    - Otomatik trend detection
    - Mevsimsellik (günlük, haftalık, yıllık)
    - Tatil etkileri
    - Harici değişkenler (regressors)
    
    Prophet, harici değişkenleri "additional regressors" olarak kabul eder.
    """
    
    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        holidays: Optional[pd.DataFrame] = None,
        name: str = "Prophet"
    ):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")
        
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.name = name
        self.model = None
        self.is_fitted = False
        self.regressor_cols = []
        
    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str = 'density',
        datetime_col: str = 'datetime',
        regressor_cols: Optional[List[str]] = None
    ) -> 'ProphetModel':
        """
        Prophet modelini eğitir.
        
        Prophet'ın beklediği format:
        - 'ds': datetime column
        - 'y': target column
        - additional columns: regressors
        
        Args:
            train_data: Eğitim verisi
            target_col: Hedef sütun
            datetime_col: Tarih sütunu
            regressor_cols: Harici değişken sütunları
        """
        # Prophet formatına çevir
        df_prophet = train_data[[datetime_col, target_col]].copy()
        df_prophet.columns = ['ds', 'y']
        
        # Model oluştur
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            holidays=self.holidays
        )
        
        # Harici değişkenler ekle
        if regressor_cols:
            self.regressor_cols = regressor_cols
            for col in regressor_cols:
                self.model.add_regressor(col)
                df_prophet[col] = train_data[col].values
            logger.info(f"Added regressors: {regressor_cols}")
        
        # Eğit
        try:
            self.model.fit(df_prophet)
            self.is_fitted = True
            logger.info(f"Prophet model fitted with "
                       f"yearly={self.yearly_seasonality}, "
                       f"weekly={self.weekly_seasonality}, "
                       f"daily={self.daily_seasonality}")
        except Exception as e:
            logger.error(f"Prophet fitting failed: {e}")
            raise
            
        return self
    
    def predict(
        self,
        n_steps: int = 1,
        future_regressors: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        n_steps adım ileriye tahmin yapar.
        
        Args:
            n_steps: Tahmin adım sayısı
            future_regressors: Gelecek dönem için regressor değerleri
            
        Returns:
            np.ndarray: Tahminler
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Future dataframe oluştur
        future = self.model.make_future_dataframe(periods=n_steps, freq='h')
        
        # Regressors ekle
        if self.regressor_cols:
            if future_regressors is None:
                raise ValueError("Model was trained with regressors. "
                               "Please provide future_regressors.")
            # Son n_steps satırı al
            future_end = future.tail(n_steps)
            for col in self.regressor_cols:
                future.loc[future_end.index, col] = future_regressors[col].values[-n_steps:]
        
        # Tahmin
        forecast = self.model.predict(future)
        
        # Son n_steps tahmini döndür
        return forecast['yhat'].values[-n_steps:]
    
    def plot_components(self, filepath: Optional[str] = None):
        """
        Prophet component planını görselleştirir.
        
        Trend, mevsimsellik vb. bileşenleri gösterir.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        import matplotlib.pyplot as plt
        
        future = self.model.make_future_dataframe(periods=0)
        forecast = self.model.predict(future)
        
        fig = self.model.plot_components(forecast)
        
        if filepath:
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Components plot saved to {filepath}")
        
        return fig
    
    def get_name(self) -> str:
        return self.name


def create_sarimax_with_exog(
    train_data: pd.DataFrame,
    target_col: str = 'density',
    weather_cols: List[str] = None,
    time_cols: List[str] = None
) -> Tuple[SARIMAXModel, List[str]]:
    """
    Harici değişkenlerle SARIMAX modeli oluşturur.
    
    Önerilen harici değişkenler:
    - Hava durumu: temperature, precipitation, wind_speed
    - Zaman: is_weekend, is_rush_hour, is_holiday
    
    Args:
        train_data: Eğitim verisi
        target_col: Hedef sütun
        weather_cols: Hava durumu sütunları
        time_cols: Zaman özellik sütunları
        
    Returns:
        Tuple: (fitted model, exog columns)
    """
    exog_cols = []
    
    if weather_cols:
        exog_cols.extend([c for c in weather_cols if c in train_data.columns])
    
    if time_cols:
        exog_cols.extend([c for c in time_cols if c in train_data.columns])
    
    # Varsayılan zaman özellikleri
    if not exog_cols:
        default_time_cols = ['is_weekend', 'is_rush_hour', 'hour_sin', 'hour_cos']
        exog_cols = [c for c in default_time_cols if c in train_data.columns]
    
    model = SARIMAXModel(
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 24)
    )
    
    model.fit(train_data, target_col, exog_cols if exog_cols else None)
    
    return model, exog_cols


if __name__ == "__main__":
    # Test
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    from src.data.feature_engineer import FeatureEngineer
    
    # Veri hazırla
    loader = DataLoader()
    df = loader.generate_synthetic_data(num_zones=1, 
                                        start_date="2023-01-01",
                                        end_date="2023-03-31")
    
    engineer = FeatureEngineer()
    df = engineer.add_time_features(df)
    
    preprocessor = DataPreprocessor()
    train, val, test = preprocessor.time_series_split(df)
    
    print("\n=== Statistical Model Tests ===\n")
    
    # SARIMAX (exog olmadan)
    print("Testing SARIMAX...")
    sarimax = SARIMAXModel(order=(1, 1, 1), seasonal_order=(1, 0, 1, 24))
    sarimax.fit(train, verbose=False)
    pred = sarimax.predict(5)
    print(f"SARIMAX prediction (5 steps): {pred}")
    
    # SARIMAX (exog ile)
    print("\nTesting SARIMAX with exogenous variables...")
    exog_cols = ['is_weekend', 'is_rush_hour']
    sarimax_exog = SARIMAXModel(order=(1, 1, 1), seasonal_order=(1, 0, 1, 24))
    sarimax_exog.fit(train, exog_cols=exog_cols, verbose=False)
    
    # Test için exog değerleri
    test_exog = test[exog_cols].values[:5]
    pred_exog = sarimax_exog.predict(5, exog=test_exog)
    print(f"SARIMAX+exog prediction (5 steps): {pred_exog}")
    
    # Prophet (varsa)
    if PROPHET_AVAILABLE:
        print("\nTesting Prophet...")
        prophet = ProphetModel(yearly_seasonality=False)
        prophet.fit(train)
        pred_prophet = prophet.predict(5)
        print(f"Prophet prediction (5 steps): {pred_prophet}")
