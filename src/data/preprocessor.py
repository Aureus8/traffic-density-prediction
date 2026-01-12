"""
Data Preprocessor Module

Veri ön işleme, temizleme ve split işlemleri.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Trafik yoğunluğu verilerini ön işleyen sınıf.
    
    İşlemler:
    - Eksik veri doldurma
    - Outlier tespiti ve işleme
    - Normalizasyon
    - Time series split
    """
    
    def __init__(self, scaler_type: str = 'minmax'):
        """
        Args:
            scaler_type: 'minmax' veya 'standard'
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.is_fitted = False
        
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = 'interpolate'
    ) -> pd.DataFrame:
        """
        Eksik değerleri işler.
        
        Args:
            df: DataFrame
            method: 'interpolate', 'ffill', 'bfill', 'mean'
            
        Returns:
            DataFrame: Eksik değerler işlenmiş veri
        """
        df = df.copy()
        
        missing_before = df.isnull().sum().sum()
        
        if method == 'interpolate':
            # Zaman serisi için en uygun yöntem
            df['density'] = df['density'].interpolate(method='time')
        elif method == 'ffill':
            df['density'] = df['density'].ffill()
        elif method == 'bfill':
            df['density'] = df['density'].bfill()
        elif method == 'mean':
            df['density'] = df['density'].fillna(df['density'].mean())
        
        # Hala kalan eksikler için ffill + bfill
        df['density'] = df['density'].ffill().bfill()
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
        
        return df
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        column: str = 'density',
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.Series:
        """
        Outlier tespiti yapar.
        
        Args:
            df: DataFrame
            column: Kontrol edilecek sütun
            method: 'iqr' veya 'zscore'
            threshold: IQR için çarpan, zscore için eşik
            
        Returns:
            Series: Outlier boolean mask
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold
        else:
            raise ValueError(f"Unknown method: {method}")
            
        logger.info(f"Detected {outliers.sum()} outliers ({outliers.mean()*100:.2f}%)")
        return outliers
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        column: str = 'density',
        method: str = 'clip'
    ) -> pd.DataFrame:
        """
        Outlier'ları işler.
        
        Args:
            df: DataFrame
            column: İşlenecek sütun
            method: 'clip', 'remove', 'interpolate'
            
        Returns:
            DataFrame: Outlier'lar işlenmiş veri
        """
        df = df.copy()
        outliers = self.detect_outliers(df, column)
        
        if method == 'clip':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].clip(lower_bound, upper_bound)
        elif method == 'remove':
            df = df[~outliers]
        elif method == 'interpolate':
            df.loc[outliers, column] = np.nan
            df[column] = df[column].interpolate(method='time')
            
        return df
    
    def normalize(
        self,
        df: pd.DataFrame,
        columns: List[str] = ['density'],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Verileri normalize eder.
        
        Args:
            df: DataFrame
            columns: Normalize edilecek sütunlar
            fit: True ise scaler'ı fit et
            
        Returns:
            DataFrame: Normalize edilmiş veri
        """
        df = df.copy()
        
        if fit or self.scaler is None:
            if self.scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
            self.scaler.fit(df[columns])
            self.is_fitted = True
            
        df[columns] = self.scaler.transform(df[columns])
        
        logger.info(f"Normalized columns: {columns}")
        return df
    
    def inverse_normalize(
        self,
        df: pd.DataFrame,
        columns: List[str] = ['density']
    ) -> pd.DataFrame:
        """
        Normalize işlemini tersine çevirir.
        
        Args:
            df: DataFrame
            columns: Tersine çevrilecek sütunlar
            
        Returns:
            DataFrame: Orijinal ölçekteki veri
        """
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted. Call normalize() first.")
            
        df = df.copy()
        df[columns] = self.scaler.inverse_transform(df[columns])
        return df
    
    def time_series_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Zaman serisi için kronolojik split yapar.
        
        **ÖNEMLİ**: Zaman serilerinde random split YAPILMAZ!
        Veri kronolojik sırayla bölünmelidir.
        
        Args:
            df: DataFrame (datetime'a göre sıralı olmalı)
            train_ratio: Eğitim verisi oranı
            val_ratio: Validation verisi oranı
            test_ratio: Test verisi oranı
            
        Returns:
            Tuple: (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Tarih sıralaması garantile
        df = df.sort_values('datetime').reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Train period: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
        logger.info(f"Val period: {val_df['datetime'].min()} to {val_df['datetime'].max()}")
        logger.info(f"Test period: {test_df['datetime'].min()} to {test_df['datetime'].max()}")
        
        return train_df, val_df, test_df
    
    def create_sequences(
        self,
        data: np.ndarray,
        seq_length: int = 24,
        target_length: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derin öğrenme için sequence'lar oluşturur.
        
        Args:
            data: Input array (n_samples, n_features)
            seq_length: Input sequence uzunluğu
            target_length: Tahmin edilecek adım sayısı
            
        Returns:
            Tuple: (X, y) arrays
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length - target_length + 1):
            X.append(data[i:(i + seq_length)])
            y.append(data[(i + seq_length):(i + seq_length + target_length)])
            
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created sequences - X: {X.shape}, y: {y.shape}")
        return X, y
    
    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        handle_missing: bool = True,
        handle_outliers: bool = True,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Tam ön işleme pipeline'ı.
        
        Args:
            df: Raw DataFrame
            handle_missing: Eksik değerleri işle
            handle_outliers: Outlier'ları işle
            normalize: Normalize et
            
        Returns:
            DataFrame: İşlenmiş veri
        """
        logger.info("Starting preprocessing pipeline...")
        
        df = df.copy()
        
        if handle_missing:
            df = self.handle_missing_values(df)
            
        if handle_outliers:
            df = self.handle_outliers(df)
            
        if normalize:
            df = self.normalize(df)
            
        logger.info("Preprocessing pipeline completed")
        return df


if __name__ == "__main__":
    # Test
    from loader import DataLoader
    
    loader = DataLoader()
    df = loader.generate_synthetic_data(num_zones=1)
    
    preprocessor = DataPreprocessor()
    
    # Pipeline
    df_processed = preprocessor.preprocess_pipeline(df)
    
    # Split
    train, val, test = preprocessor.time_series_split(df_processed)
    
    print(f"\nProcessed data shape: {df_processed.shape}")
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
