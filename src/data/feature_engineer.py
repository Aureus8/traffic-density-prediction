"""
Feature Engineering Module

Zaman serisi ve harici değişkenler için feature engineering.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Trafik yoğunluğu tahmini için feature engineering.
    
    Özellik kategorileri:
    - Zaman özellikleri
    - Lag özellikleri
    - Rolling statistics
    - Harici değişkenler
    """
    
    def __init__(self):
        self.feature_columns = []
        
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zaman bazlı özellikler ekler.
        
        Args:
            df: DataFrame (datetime sütunu olmalı)
            
        Returns:
            DataFrame: Yeni özellikler eklenmiş veri
        """
        df = df.copy()
        
        # Temel zaman özellikleri
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)
        
        # Kategorik zaman özellikleri
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = df['datetime'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['datetime'].dt.is_month_end.astype(int)
        
        # Rush hour tanımlamaları
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
        
        # Gün dilimi
        conditions = [
            (df['hour'] >= 0) & (df['hour'] < 6),
            (df['hour'] >= 6) & (df['hour'] < 12),
            (df['hour'] >= 12) & (df['hour'] < 18),
            (df['hour'] >= 18) & (df['hour'] < 24)
        ]
        choices = [0, 1, 2, 3]  # gece, sabah, öğlen, akşam
        df['time_of_day'] = np.select(conditions, choices, default=0)
        
        # Mevsim
        df['season'] = (df['month'] % 12 // 3)  # 0: Kış, 1: İlkbahar, 2: Yaz, 3: Sonbahar
        
        # Cyclical encoding (sin/cos) - daha iyi pattern yakalama için
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        new_features = [
            'hour', 'day_of_week', 'day_of_month', 'month', 'year', 'week_of_year',
            'is_weekend', 'is_month_start', 'is_month_end',
            'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
            'time_of_day', 'season',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]
        self.feature_columns.extend(new_features)
        
        logger.info(f"Added {len(new_features)} time features")
        return df
    
    def add_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'density',
        lags: List[int] = [1, 2, 3, 4, 6, 12, 24, 48, 168]
    ) -> pd.DataFrame:
        """
        Lag (gecikmeli) özellikler ekler.
        
        Args:
            df: DataFrame
            target_col: Hedef sütun
            lags: Lag değerleri (saat cinsinden)
                  1, 2, 3, 4: Son 4 saat
                  6, 12: Yarım gün patterns
                  24: Dün aynı saat
                  48: 2 gün önce
                  168: Geçen hafta aynı saat
                  
        Returns:
            DataFrame: Lag özellikleri eklenmiş veri
        """
        df = df.copy()
        
        for lag in lags:
            col_name = f'{target_col}_lag_{lag}'
            df[col_name] = df[target_col].shift(lag)
            self.feature_columns.append(col_name)
            
        logger.info(f"Added {len(lags)} lag features")
        return df
    
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'density',
        windows: List[int] = [3, 6, 12, 24]
    ) -> pd.DataFrame:
        """
        Rolling (hareketli) istatistik özellikleri ekler.
        
        Args:
            df: DataFrame
            target_col: Hedef sütun
            windows: Pencere boyutları (saat)
            
        Returns:
            DataFrame: Rolling özellikleri eklenmiş veri
        """
        df = df.copy()
        
        for window in windows:
            # Ortalama
            col_mean = f'{target_col}_rolling_mean_{window}'
            df[col_mean] = df[target_col].shift(1).rolling(window=window).mean()
            self.feature_columns.append(col_mean)
            
            # Standart sapma
            col_std = f'{target_col}_rolling_std_{window}'
            df[col_std] = df[target_col].shift(1).rolling(window=window).std()
            self.feature_columns.append(col_std)
            
            # Min
            col_min = f'{target_col}_rolling_min_{window}'
            df[col_min] = df[target_col].shift(1).rolling(window=window).min()
            self.feature_columns.append(col_min)
            
            # Max
            col_max = f'{target_col}_rolling_max_{window}'
            df[col_max] = df[target_col].shift(1).rolling(window=window).max()
            self.feature_columns.append(col_max)
        
        logger.info(f"Added {len(windows) * 4} rolling features")
        return df
    
    def add_diff_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'density',
        periods: List[int] = [1, 24, 168]
    ) -> pd.DataFrame:
        """
        Diferansiyel (değişim) özellikleri ekler.
        
        Args:
            df: DataFrame
            target_col: Hedef sütun
            periods: Fark alınacak periodlar
                     1: Saatlik değişim
                     24: Günlük değişim
                     168: Haftalık değişim
                     
        Returns:
            DataFrame: Diff özellikleri eklenmiş veri
        """
        df = df.copy()
        
        for period in periods:
            col_diff = f'{target_col}_diff_{period}'
            df[col_diff] = df[target_col].diff(period)
            self.feature_columns.append(col_diff)
            
            # Yüzdesel değişim
            col_pct = f'{target_col}_pct_change_{period}'
            df[col_pct] = df[target_col].pct_change(period)
            self.feature_columns.append(col_pct)
        
        logger.info(f"Added {len(periods) * 2} diff features")
        return df
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Etkileşim özellikleri ekler.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame: Etkileşim özellikleri eklenmiş veri
        """
        df = df.copy()
        
        # Rush hour + weekend etkileşimi
        if 'is_rush_hour' in df.columns and 'is_weekend' in df.columns:
            df['rush_weekend_interaction'] = df['is_rush_hour'] * (1 - df['is_weekend'])
            self.feature_columns.append('rush_weekend_interaction')
        
        # Saat + gün etkileşimi
        if 'hour' in df.columns and 'day_of_week' in df.columns:
            df['hour_day_interaction'] = df['hour'] * df['day_of_week']
            self.feature_columns.append('hour_day_interaction')
        
        logger.info("Added interaction features")
        return df
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        add_time: bool = True,
        add_lags: bool = True,
        add_rolling: bool = True,
        add_diff: bool = True,
        add_interaction: bool = True,
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Tam feature engineering pipeline'ı.
        
        Args:
            df: Raw DataFrame
            add_*: Hangi özelliklerin ekleneceği
            drop_na: NaN satırları sil (lag/rolling için gerekli)
            
        Returns:
            DataFrame: Tüm özellikler eklenmiş veri
        """
        self.feature_columns = []
        logger.info("Starting feature engineering pipeline...")
        
        if add_time:
            df = self.add_time_features(df)
            
        if add_lags:
            df = self.add_lag_features(df)
            
        if add_rolling:
            df = self.add_rolling_features(df)
            
        if add_diff:
            df = self.add_diff_features(df)
            
        if add_interaction:
            df = self.add_interaction_features(df)
            
        if drop_na:
            initial_len = len(df)
            df = df.dropna()
            dropped = initial_len - len(df)
            logger.info(f"Dropped {dropped} rows with NaN values")
            
        logger.info(f"Feature engineering completed. Total features: {len(self.feature_columns)}")
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Oluşturulan feature sütunlarını döndürür."""
        return self.feature_columns.copy()


if __name__ == "__main__":
    # Test
    from loader import DataLoader
    
    loader = DataLoader()
    df = loader.generate_synthetic_data(num_zones=1)
    
    engineer = FeatureEngineer()
    df_featured = engineer.engineer_features(df)
    
    print(f"\nOriginal shape: {loader.generate_synthetic_data(num_zones=1).shape}")
    print(f"Featured shape: {df_featured.shape}")
    print(f"\nFeature columns ({len(engineer.get_feature_columns())}):")
    for col in engineer.get_feature_columns()[:10]:
        print(f"  - {col}")
    print("  ...")
