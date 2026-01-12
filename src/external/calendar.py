"""
Calendar Features Module

Tatil ve özel gün bilgileri.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Holiday kütüphanesi
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    logger.warning("holidays library not installed. Using basic holiday detection.")


class CalendarFeatures:
    """
    Takvim bazlı özellikler.
    
    Özellikler:
    - Resmi tatiller
    - Okul tatilleri
    - Rush hour tanımlamaları
    - Özel etkinlikler
    """
    
    def __init__(self, country: str = 'US', state: Optional[str] = 'NY'):
        """
        Args:
            country: Ülke kodu (ISO 3166-1 alpha-2)
            state: Eyalet/bölge kodu (opsiyonel)
        """
        self.country = country
        self.state = state
        self.holiday_calendar = None
        
        if HOLIDAYS_AVAILABLE:
            try:
                self.holiday_calendar = holidays.country_holidays(country, state=state)
            except Exception as e:
                logger.warning(f"Could not create holiday calendar: {e}")
    
    def is_holiday(self, dt: datetime) -> bool:
        """
        Verilen tarih tatil mi?
        
        Args:
            dt: Kontrol edilecek tarih
            
        Returns:
            bool: Tatil mi?
        """
        if self.holiday_calendar is not None:
            return dt.date() in self.holiday_calendar
        
        # Fallback: Temel ABD tatilleri
        return self._is_basic_holiday(dt)
    
    def _is_basic_holiday(self, dt: datetime) -> bool:
        """Temel tatil kontrolü."""
        month, day = dt.month, dt.day
        
        # Fixed holidays
        fixed_holidays = [
            (1, 1),   # New Year
            (7, 4),   # Independence Day
            (12, 25), # Christmas
            (12, 31), # New Year's Eve
        ]
        
        if (month, day) in fixed_holidays:
            return True
        
        # Thanksgiving (4th Thursday in November)
        if month == 11:
            # Ayın 4. Perşembe günü
            first_day = datetime(dt.year, 11, 1).weekday()
            thanksgiving = 22 + (3 - first_day) % 7
            if day == thanksgiving:
                return True
        
        return False
    
    def get_holiday_name(self, dt: datetime) -> Optional[str]:
        """Tatil adını döndürür."""
        if self.holiday_calendar is not None:
            return self.holiday_calendar.get(dt.date())
        return None
    
    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame'e takvim özelliklerini ekler.
        
        Args:
            df: DataFrame (datetime sütunu olmalı)
            
        Returns:
            DataFrame: Takvim özellikleri eklenmiş veri
        """
        df = df.copy()
        
        # Holiday
        df['is_holiday'] = df['datetime'].apply(lambda x: int(self.is_holiday(x)))
        
        # Holiday adı (analiz için)
        if self.holiday_calendar is not None:
            df['holiday_name'] = df['datetime'].apply(
                lambda x: self.holiday_calendar.get(x.date(), '')
            )
        
        # Pre-holiday (tatilden önceki gün)
        df['is_pre_holiday'] = df['is_holiday'].shift(-1).fillna(0).astype(int)
        
        # Post-holiday (tatilden sonraki gün)
        df['is_post_holiday'] = df['is_holiday'].shift(1).fillna(0).astype(int)
        
        # Rush hour tanımları
        df['is_morning_rush'] = (
            (df['datetime'].dt.hour >= 7) & 
            (df['datetime'].dt.hour <= 9) &
            (df['datetime'].dt.dayofweek < 5)
        ).astype(int)
        
        df['is_evening_rush'] = (
            (df['datetime'].dt.hour >= 17) & 
            (df['datetime'].dt.hour <= 19) &
            (df['datetime'].dt.dayofweek < 5)
        ).astype(int)
        
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
        
        # Weekend
        df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
        
        # Business hours
        df['is_business_hours'] = (
            (df['datetime'].dt.hour >= 9) & 
            (df['datetime'].dt.hour <= 17) &
            (df['datetime'].dt.dayofweek < 5)
        ).astype(int)
        
        # Night hours (trafik düşük)
        df['is_night'] = (
            (df['datetime'].dt.hour >= 22) | 
            (df['datetime'].dt.hour <= 5)
        ).astype(int)
        
        # Mevsimler
        df['season'] = df['datetime'].dt.month.apply(self._get_season)
        
        # Ayın günü kategorisi
        df['is_month_start'] = (df['datetime'].dt.day <= 3).astype(int)
        df['is_month_end'] = (df['datetime'].dt.day >= 28).astype(int)
        
        # Payday effect (15 ve ayın sonu)
        df['is_payday'] = (
            (df['datetime'].dt.day == 15) | 
            (df['datetime'].dt.day == df['datetime'].dt.daysinmonth)
        ).astype(int)
        
        logger.info("Calendar features added")
        return df
    
    def _get_season(self, month: int) -> int:
        """Ay numarasından mevsim döndürür."""
        # 0: Kış, 1: İlkbahar, 2: Yaz, 3: Sonbahar
        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        else:
            return 3
    
    def add_event_features(
        self,
        df: pd.DataFrame,
        events: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Özel etkinlik özellikleri ekler.
        
        Args:
            df: DataFrame
            events: Etkinlik listesi [{'date': 'YYYY-MM-DD', 'type': 'concert', 'impact': 1.5}, ...]
            
        Returns:
            DataFrame: Etkinlik özellikleri eklenmiş veri
        """
        df = df.copy()
        
        # Varsayılan: etkinlik yok
        df['has_event'] = 0
        df['event_impact'] = 1.0
        
        if events:
            for event in events:
                event_date = pd.to_datetime(event['date']).date()
                mask = df['datetime'].dt.date == event_date
                df.loc[mask, 'has_event'] = 1
                df.loc[mask, 'event_impact'] = event.get('impact', 1.5)
        
        return df
    
    def simulate_events(
        self,
        start_date: str,
        end_date: str,
        avg_events_per_month: int = 5
    ) -> List[Dict]:
        """
        Rastgele etkinlikler simüle eder.
        
        Args:
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            avg_events_per_month: Aylık ortalama etkinlik sayısı
            
        Returns:
            List: Etkinlik listesi
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        n_months = len(date_range) / 30
        n_events = int(n_months * avg_events_per_month)
        
        event_types = [
            ('concert', 1.3),
            ('sports_game', 1.5),
            ('convention', 1.4),
            ('parade', 1.6),
            ('festival', 1.7)
        ]
        
        events = []
        event_dates = np.random.choice(date_range, size=min(n_events, len(date_range)), replace=False)
        
        for event_date in event_dates:
            event_type, base_impact = event_types[np.random.randint(len(event_types))]
            impact = base_impact + np.random.uniform(-0.2, 0.3)
            
            events.append({
                'date': str(event_date.date()),
                'type': event_type,
                'impact': round(impact, 2)
            })
        
        logger.info(f"Simulated {len(events)} events")
        return events


if __name__ == "__main__":
    print("\n=== Calendar Features Tests ===\n")
    
    calendar = CalendarFeatures(country='US', state='NY')
    
    # Test dates
    test_dates = [
        datetime(2023, 1, 1),   # New Year
        datetime(2023, 7, 4),   # Independence Day
        datetime(2023, 12, 25), # Christmas
        datetime(2023, 6, 15),  # Regular day
    ]
    
    for dt in test_dates:
        is_holiday = calendar.is_holiday(dt)
        name = calendar.get_holiday_name(dt)
        print(f"{dt.date()}: Holiday={is_holiday}, Name={name}")
    
    # DataFrame test
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq='h')
    df = pd.DataFrame({'datetime': dates})
    
    df = calendar.add_calendar_features(df)
    print(f"\nDataFrame with calendar features: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nHolidays found: {df['is_holiday'].sum()}")
    print(f"Rush hours: {df['is_rush_hour'].sum()}")
    
    # Simulated events
    events = calendar.simulate_events("2023-01-01", "2023-03-31", avg_events_per_month=3)
    print(f"\nSimulated events: {len(events)}")
    for e in events[:3]:
        print(f"  {e}")
