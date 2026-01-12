"""
Weather API Module

OpenWeatherMap API entegrasyonu.
Geçmiş ve tahmin hava durumu verileri.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherAPI:
    """
    OpenWeatherMap API entegrasyonu.
    
    Özellikler:
    - Güncel hava durumu
    - 5 günlük tahmin
    - Geçmiş veri (simüle)
    - Cache mekanizması
    """
    
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/external/weather_cache"
    ):
        """
        Args:
            api_key: OpenWeatherMap API anahtarı
            cache_dir: Cache dizini
        """
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.api_key:
            logger.warning("API key not provided. Using simulated weather data.")
    
    def get_current_weather(
        self,
        lat: float = 40.7128,  # New York
        lon: float = -74.0060
    ) -> Dict:
        """
        Güncel hava durumu verisi alır.
        
        Args:
            lat: Enlem
            lon: Boylam
            
        Returns:
            Dict: Hava durumu bilgileri
        """
        if not self.api_key:
            return self._generate_simulated_weather()
        
        try:
            url = f"{self.BASE_URL}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_weather_data(data)
            
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return self._generate_simulated_weather()
    
    def get_forecast(
        self,
        lat: float = 40.7128,
        lon: float = -74.0060,
        hours: int = 24
    ) -> pd.DataFrame:
        """
        Hava durumu tahmini alır (5 güne kadar).
        
        Args:
            lat: Enlem
            lon: Boylam
            hours: Tahmin saati sayısı
            
        Returns:
            DataFrame: Saatlik hava durumu tahmini
        """
        if not self.api_key:
            return self._generate_simulated_forecast(hours)
        
        try:
            url = f"{self.BASE_URL}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # API 3 saatlik veriler döndürür, interpole et
            forecasts = []
            for item in data['list'][:hours // 3 + 1]:
                weather = self._parse_weather_data(item)
                weather['datetime'] = datetime.fromtimestamp(item['dt'])
                forecasts.append(weather)
            
            return pd.DataFrame(forecasts)
            
        except Exception as e:
            logger.error(f"Forecast request failed: {e}")
            return self._generate_simulated_forecast(hours)
    
    def _parse_weather_data(self, data: Dict) -> Dict:
        """API yanıtını parse eder."""
        main = data.get('main', {})
        weather = data.get('weather', [{}])[0]
        wind = data.get('wind', {})
        
        return {
            'temperature': main.get('temp', 20),
            'feels_like': main.get('feels_like', 20),
            'humidity': main.get('humidity', 50),
            'pressure': main.get('pressure', 1013),
            'wind_speed': wind.get('speed', 0),
            'weather_condition': weather.get('main', 'Clear'),
            'weather_description': weather.get('description', 'clear sky'),
            'precipitation': data.get('rain', {}).get('1h', 0),
            'cloudiness': data.get('clouds', {}).get('all', 0)
        }
    
    def _generate_simulated_weather(self) -> Dict:
        """Simüle edilmiş hava durumu üretir."""
        hour = datetime.now().hour
        month = datetime.now().month
        
        # Mevsimsel sıcaklık
        base_temp = 15 + 10 * np.sin(2 * np.pi * (month - 4) / 12)
        
        # Günlük değişim
        daily_var = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        return {
            'temperature': round(base_temp + daily_var + np.random.uniform(-2, 2), 1),
            'feels_like': round(base_temp + daily_var + np.random.uniform(-3, 1), 1),
            'humidity': round(50 + np.random.uniform(-20, 30)),
            'pressure': round(1013 + np.random.uniform(-10, 10)),
            'wind_speed': round(np.random.uniform(0, 10), 1),
            'weather_condition': np.random.choice(['Clear', 'Clouds', 'Rain'], p=[0.5, 0.35, 0.15]),
            'weather_description': 'simulated weather',
            'precipitation': round(np.random.uniform(0, 2) if np.random.random() > 0.8 else 0, 1),
            'cloudiness': round(np.random.uniform(0, 100))
        }
    
    def _generate_simulated_forecast(self, hours: int) -> pd.DataFrame:
        """Simüle edilmiş tahmin üretir."""
        forecasts = []
        base_time = datetime.now()
        
        for h in range(hours):
            forecast_time = base_time + timedelta(hours=h)
            weather = self._generate_simulated_weather()
            weather['datetime'] = forecast_time
            forecasts.append(weather)
        
        return pd.DataFrame(forecasts)
    
    def generate_historical_weather(
        self,
        start_date: str,
        end_date: str,
        lat: float = 40.7128,
        lon: float = -74.0060
    ) -> pd.DataFrame:
        """
        Geçmiş hava durumu verisi üretir (simüle).
        
        Not: OpenWeatherMap historical API ücretlidir.
        Bu fonksiyon gerçekçi simüle veri üretir.
        
        Args:
            start_date: Başlangıç tarihi (YYYY-MM-DD)
            end_date: Bitiş tarihi  
            lat: Enlem
            lon: Boylam
            
        Returns:
            DataFrame: Saatlik hava durumu verileri
        """
        logger.info(f"Generating simulated historical weather from {start_date} to {end_date}")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
        weather_data = []
        for dt in date_range:
            hour = dt.hour
            month = dt.month
            day_of_year = dt.dayofyear
            
            # Mevsimsel sıcaklık (NYC için gerçekçi)
            # Kış: -2°C, Yaz: 25°C ortalama
            seasonal_temp = 11.5 + 13.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            # Günlük değişim (gece soğuk, gündüz sıcak)
            daily_var = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            # Rastgele değişim
            noise = np.random.normal(0, 2)
            
            temperature = seasonal_temp + daily_var + noise
            
            # Yağış olasılığı (mevsime bağlı)
            rain_prob = 0.1 + 0.1 * np.sin(2 * np.pi * (month - 4) / 12)
            is_raining = np.random.random() < rain_prob
            
            weather_data.append({
                'datetime': dt,
                'temperature': round(temperature, 1),
                'feels_like': round(temperature - np.random.uniform(0, 3), 1),
                'humidity': round(50 + 20 * np.sin(2 * np.pi * (hour - 6) / 24) + np.random.uniform(-10, 10)),
                'pressure': round(1013 + np.random.uniform(-10, 10)),
                'wind_speed': round(max(0, np.random.normal(5, 3)), 1),
                'precipitation': round(np.random.uniform(0.5, 5), 1) if is_raining else 0,
                'cloudiness': round(min(100, max(0, 30 + np.random.normal(0, 20)))),
                'is_raining': int(is_raining)
            })
        
        df = pd.DataFrame(weather_data)
        logger.info(f"Generated {len(df)} weather records")
        
        return df
    
    def get_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hava durumu verilerinden model özellikleri çıkarır.
        
        Args:
            df: Hava durumu DataFrame
            
        Returns:
            DataFrame: Model için hazır özellikler
        """
        features = df.copy()
        
        # Sıcaklık kategorileri
        features['temp_category'] = pd.cut(
            features['temperature'],
            bins=[-np.inf, 0, 10, 20, 30, np.inf],
            labels=['freezing', 'cold', 'mild', 'warm', 'hot']
        )
        
        # Binary özellikler
        features['is_cold'] = (features['temperature'] < 10).astype(int)
        features['is_hot'] = (features['temperature'] > 25).astype(int)
        features['is_windy'] = (features['wind_speed'] > 7).astype(int)
        features['is_humid'] = (features['humidity'] > 70).astype(int)
        
        # Kötü hava koşulları (trafik etkisi yüksek)
        features['bad_weather'] = (
            (features['precipitation'] > 0) | 
            (features['temperature'] < 0) |
            (features['wind_speed'] > 10)
        ).astype(int)
        
        return features


if __name__ == "__main__":
    print("\n=== Weather API Tests ===\n")
    
    api = WeatherAPI()
    
    # Güncel hava durumu (simüle)
    current = api.get_current_weather()
    print("Current weather:", current)
    
    # Geçmiş veri
    historical = api.generate_historical_weather("2023-01-01", "2023-01-31")
    print(f"\nHistorical data shape: {historical.shape}")
    print(historical.head())
    
    # Features
    features = api.get_weather_features(historical)
    print(f"\nWeather features columns: {list(features.columns)}")
