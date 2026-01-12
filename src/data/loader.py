"""
Data Loader Module

NYC Taxi veri setini indirme ve yükleme işlemleri.
Sentetik veri üretimi desteği de içerir.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Trafik yoğunluğu verilerini yükleyen ve yöneten sınıf.
    
    Desteklenen veri kaynakları:
    - NYC Taxi dataset (parquet format)
    - Sentetik veri üretimi
    - CSV dosyaları
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: Veri dosyalarının saklanacağı ana dizin
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Dizinleri oluştur
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_synthetic_data(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2023-12-31",
        num_zones: int = 5,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Gerçekçi sentetik trafik yoğunluğu verisi üretir.
        
        Özellikler:
        - Günlük seasonality (rush hour patterns)
        - Haftalık seasonality (hafta içi/sonu farkı)
        - Yıllık seasonality (mevsimsel değişimler)
        - Rastgele noise
        - Tatil etkileri
        
        Args:
            start_date: Başlangıç tarihi (YYYY-MM-DD)
            end_date: Bitiş tarihi (YYYY-MM-DD)
            num_zones: Bölge sayısı
            seed: Random seed
            
        Returns:
            DataFrame: Saatlik trafik yoğunluğu verileri
        """
        np.random.seed(seed)
        logger.info(f"Generating synthetic data from {start_date} to {end_date}")
        
        # Tarih aralığı oluştur
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
        data = []
        for zone_id in range(1, num_zones + 1):
            for dt in date_range:
                # Base density
                base_density = 100
                
                # Günlük pattern (rush hours)
                hour = dt.hour
                if 7 <= hour <= 9:  # Sabah rush hour
                    daily_factor = 2.5 + np.random.uniform(-0.3, 0.3)
                elif 17 <= hour <= 19:  # Akşam rush hour
                    daily_factor = 2.8 + np.random.uniform(-0.3, 0.3)
                elif 10 <= hour <= 16:  # Gündüz
                    daily_factor = 1.5 + np.random.uniform(-0.2, 0.2)
                elif 20 <= hour <= 23:  # Akşam
                    daily_factor = 1.2 + np.random.uniform(-0.2, 0.2)
                else:  # Gece (00:00 - 06:00)
                    daily_factor = 0.3 + np.random.uniform(-0.1, 0.1)
                
                # Haftalık pattern
                day_of_week = dt.dayofweek
                if day_of_week < 5:  # Hafta içi
                    weekly_factor = 1.0
                elif day_of_week == 5:  # Cumartesi
                    weekly_factor = 0.7
                else:  # Pazar
                    weekly_factor = 0.5
                
                # Mevsimsel pattern
                month = dt.month
                if month in [6, 7, 8]:  # Yaz
                    seasonal_factor = 0.85  # Tatil sezonu, daha az trafik
                elif month in [12, 1, 2]:  # Kış
                    seasonal_factor = 0.9  # Kötü hava koşulları
                else:
                    seasonal_factor = 1.0
                
                # Bölge faktörü (farklı bölgeler farklı yoğunluk)
                zone_factor = 0.5 + (zone_id / num_zones) * 1.5
                
                # Noise
                noise = np.random.normal(0, 10)
                
                # Final density hesapla
                density = max(0, (
                    base_density * 
                    daily_factor * 
                    weekly_factor * 
                    seasonal_factor * 
                    zone_factor + 
                    noise
                ))
                
                data.append({
                    'datetime': dt,
                    'zone_id': zone_id,
                    'density': round(density, 2),
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'month': month,
                    'is_weekend': day_of_week >= 5
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} records for {num_zones} zones")
        
        return df
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        CSV dosyasından veri yükler.
        
        Args:
            filepath: CSV dosya yolu
            
        Returns:
            DataFrame: Yüklenen veri
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath, parse_dates=['datetime'])
        return df
    
    def save_data(
        self,
        df: pd.DataFrame,
        filename: str,
        processed: bool = True
    ) -> str:
        """
        Veriyi dosyaya kaydeder.
        
        Args:
            df: Kaydedilecek DataFrame
            filename: Dosya adı
            processed: True ise processed dizinine, False ise raw dizinine kaydet
            
        Returns:
            str: Kaydedilen dosya yolu
        """
        target_dir = self.processed_dir if processed else self.raw_dir
        filepath = target_dir / filename
        
        if filename.endswith('.parquet'):
            df.to_parquet(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
            
        logger.info(f"Data saved to {filepath}")
        return str(filepath)
    
    def load_data(
        self,
        filename: str,
        processed: bool = True
    ) -> pd.DataFrame:
        """
        Kaydedilmiş veriyi yükler.
        
        Args:
            filename: Dosya adı
            processed: True ise processed dizininden, False ise raw dizininden yükle
            
        Returns:
            DataFrame: Yüklenen veri
        """
        target_dir = self.processed_dir if processed else self.raw_dir
        filepath = target_dir / filename
        
        if filename.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath, parse_dates=['datetime'])
            
        logger.info(f"Data loaded from {filepath}: {len(df)} records")
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Veri seti hakkında özet bilgi döndürür.
        
        Args:
            df: Analiz edilecek DataFrame
            
        Returns:
            dict: Özet istatistikler
        """
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': str(df['datetime'].min()),
                'end': str(df['datetime'].max())
            },
            'zones': df['zone_id'].nunique() if 'zone_id' in df.columns else 1,
            'density_stats': {
                'mean': round(df['density'].mean(), 2),
                'std': round(df['density'].std(), 2),
                'min': round(df['density'].min(), 2),
                'max': round(df['density'].max(), 2)
            },
            'missing_values': df.isnull().sum().to_dict()
        }
        return summary


if __name__ == "__main__":
    # Test
    loader = DataLoader()
    
    # Sentetik veri üret
    df = loader.generate_synthetic_data(
        start_date="2022-01-01",
        end_date="2023-12-31",
        num_zones=5
    )
    
    # Özet bilgi
    summary = loader.get_data_summary(df)
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Kaydet
    loader.save_data(df, "traffic_data.csv")
