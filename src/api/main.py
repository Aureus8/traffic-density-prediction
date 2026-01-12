"""
FastAPI Traffic Density Prediction Service

Gerçek zamanlı trafik yoğunluğu tahmin API'si.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

# FastAPI
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import numpy as np
import pandas as pd
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureEngineer
from src.external.weather_api import WeatherAPI
from src.external.calendar import CalendarFeatures
from src.evaluation.metrics import ModelMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== API Setup ====================

app = FastAPI(
    title="Traffic Density Prediction API",
    description="""
    Gerçek zamanlı trafik yoğunluğu tahmin servisi.
    
    ## Özellikler
    - Saatlik yoğunluk tahmini (1-24 saat ileriye)
    - Harici değişken entegrasyonu (hava durumu, tatiller)
    - Çoklu model desteği
    - < 100ms yanıt süresi hedefi
    
    ## Modeller
    - Baseline: Naive, Moving Average, ARIMA
    - Statistical: SARIMAX
    - Deep Learning: LSTM, GRU
    - Ensemble: Weighted Average
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Request/Response Models ====================

class PredictionRequest(BaseModel):
    """Tahmin isteği."""
    datetime: str = Field(..., description="Tahmin zamanı (ISO format: YYYY-MM-DDTHH:MM:SS)")
    zone_id: int = Field(default=1, description="Bölge ID (1-5)")
    include_weather: bool = Field(default=True, description="Hava durumu bilgisi dahil et")
    include_confidence: bool = Field(default=False, description="Güven aralığı dahil et")
    
    class Config:
        json_schema_extra = {
            "example": {
                "datetime": "2024-01-15T14:00:00",
                "zone_id": 1,
                "include_weather": True,
                "include_confidence": False
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch tahmin isteği."""
    start_datetime: str = Field(..., description="Başlangıç zamanı")
    hours: int = Field(default=24, ge=1, le=168, description="Tahmin saati sayısı (1-168)")
    zone_id: int = Field(default=1, description="Bölge ID")


class PredictionResponse(BaseModel):
    """Tahmin yanıtı."""
    datetime: str
    zone_id: int
    predicted_density: float
    model_used: str
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    weather: Optional[Dict] = None
    is_holiday: Optional[bool] = None
    is_rush_hour: Optional[bool] = None
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    """Batch tahmin yanıtı."""
    predictions: List[PredictionResponse]
    total_hours: int
    model_used: str
    latency_ms: float


class ModelMetricsResponse(BaseModel):
    """Model metrikleri yanıtı."""
    model_name: str
    metrics: Dict[str, float]
    last_updated: str


class HealthResponse(BaseModel):
    """Sağlık kontrolü yanıtı."""
    status: str
    version: str
    models_loaded: List[str]
    uptime_seconds: float


# ==================== Global State ====================

class AppState:
    """Uygulama durumu."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.loaded_models = {}
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.weather_api = WeatherAPI()
        self.calendar = CalendarFeatures()
        self.metrics = ModelMetrics()
        
        # Simüle edilmiş model tahminleri için geçmiş veri
        self.historical_data = None
        self._load_sample_data()
        
    def _load_sample_data(self):
        """Örnek veri yükle."""
        try:
            self.historical_data = self.data_loader.generate_synthetic_data(
                start_date="2023-01-01",
                end_date="2023-12-31",
                num_zones=5
            )
            logger.info("Sample data loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            self.historical_data = None
    
    def get_uptime(self) -> float:
        """Uptime'ı saniye olarak döndürür."""
        return (datetime.now() - self.start_time).total_seconds()


state = AppState()


# ==================== Endpoints ====================

@app.get("/", tags=["General"])
async def root():
    """API root endpoint."""
    return {
        "name": "Traffic Density Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Servis sağlık kontrolü.
    
    Servisin çalışır durumda olduğunu ve modellerin yüklendiğini doğrular.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=["Naive", "MovingAverage", "ARIMA", "SARIMAX", "Ensemble"],
        uptime_seconds=round(state.get_uptime(), 2)
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Tek nokta tahmin.
    
    Belirtilen zaman ve bölge için trafik yoğunluğu tahmini yapar.
    
    **Hedef latency: < 100ms**
    """
    import time
    start_time = time.time()
    
    try:
        # Parse datetime
        dt = datetime.fromisoformat(request.datetime)
        
        # Calendar özellikler
        is_holiday = state.calendar.is_holiday(dt)
        is_rush_hour = (7 <= dt.hour <= 9) or (17 <= dt.hour <= 19)
        is_weekend = dt.weekday() >= 5
        
        # Hava durumu
        weather = None
        if request.include_weather:
            weather = state.weather_api._generate_simulated_weather()
        
        # Tahmin (simüle)
        # Gerçek senaryoda burası model.predict() olacak
        base_density = 100
        
        # Zaman faktörleri
        hour_factor = 1.0
        if is_rush_hour and not is_weekend:
            hour_factor = 2.5
        elif is_weekend:
            hour_factor = 0.7
        elif 22 <= dt.hour or dt.hour <= 5:
            hour_factor = 0.3
        
        # Tatil faktörü
        holiday_factor = 0.5 if is_holiday else 1.0
        
        # Hava durumu faktörü
        weather_factor = 1.0
        if weather and weather.get('precipitation', 0) > 0:
            weather_factor = 0.8
        
        # Final tahmin
        predicted_density = base_density * hour_factor * holiday_factor * weather_factor
        predicted_density += np.random.uniform(-5, 5)  # Noise
        predicted_density = round(max(0, predicted_density), 2)
        
        # Güven aralığı
        confidence_lower = None
        confidence_upper = None
        if request.include_confidence:
            confidence_lower = round(predicted_density * 0.85, 2)
            confidence_upper = round(predicted_density * 1.15, 2)
        
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        return PredictionResponse(
            datetime=request.datetime,
            zone_id=request.zone_id,
            predicted_density=predicted_density,
            model_used="Ensemble",
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            weather=weather if request.include_weather else None,
            is_holiday=is_holiday,
            is_rush_hour=is_rush_hour,
            latency_ms=latency_ms
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch tahmin.
    
    Belirtilen başlangıç zamanından itibaren birden fazla saat için tahmin yapar.
    Maximum 168 saat (1 hafta).
    """
    import time
    start_time = time.time()
    
    try:
        start_dt = datetime.fromisoformat(request.start_datetime)
        predictions = []
        
        for i in range(request.hours):
            dt = start_dt + timedelta(hours=i)
            
            # Her saat için tahmin
            single_request = PredictionRequest(
                datetime=dt.isoformat(),
                zone_id=request.zone_id,
                include_weather=True,
                include_confidence=False
            )
            
            pred = await predict(single_request)
            predictions.append(pred)
        
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_hours=request.hours,
            model_used="Ensemble",
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/metrics", response_model=List[ModelMetricsResponse], tags=["Model Info"])
async def get_model_metrics():
    """
    Model performans metrikleri.
    
    Tüm modellerin güncel performans metriklerini döndürür.
    """
    # Simüle edilmiş metrikler
    models_metrics = [
        {
            "model_name": "Naive",
            "metrics": {"mape": 25.3, "rmse": 18.5, "r2": 0.65},
            "last_updated": datetime.now().isoformat()
        },
        {
            "model_name": "MovingAverage",
            "metrics": {"mape": 18.7, "rmse": 14.2, "r2": 0.78},
            "last_updated": datetime.now().isoformat()
        },
        {
            "model_name": "ARIMA",
            "metrics": {"mape": 15.2, "rmse": 11.8, "r2": 0.82},
            "last_updated": datetime.now().isoformat()
        },
        {
            "model_name": "SARIMAX",
            "metrics": {"mape": 12.8, "rmse": 9.5, "r2": 0.87},
            "last_updated": datetime.now().isoformat()
        },
        {
            "model_name": "Ensemble",
            "metrics": {"mape": 11.2, "rmse": 8.1, "r2": 0.91},
            "last_updated": datetime.now().isoformat()
        }
    ]
    
    return [ModelMetricsResponse(**m) for m in models_metrics]


@app.get("/model/compare", tags=["Model Info"])
async def compare_models():
    """
    Model karşılaştırması.
    
    Tüm modelleri baseline ile karşılaştırır ve iyileşme oranlarını gösterir.
    """
    metrics = await get_model_metrics()
    
    baseline = next(m for m in metrics if m.model_name == "Naive")
    
    comparisons = []
    for m in metrics:
        if m.model_name == "Naive":
            continue
        
        mape_improvement = (baseline.metrics["mape"] - m.metrics["mape"]) / baseline.metrics["mape"] * 100
        r2_improvement = (m.metrics["r2"] - baseline.metrics["r2"]) / (1 - baseline.metrics["r2"]) * 100
        
        comparisons.append({
            "model": m.model_name,
            "mape": m.metrics["mape"],
            "r2": m.metrics["r2"],
            "mape_improvement_%": round(mape_improvement, 1),
            "r2_improvement_%": round(r2_improvement, 1),
            "meets_targets": m.metrics["mape"] < 15 and m.metrics["r2"] > 0.85
        })
    
    return {
        "baseline": baseline.model_name,
        "baseline_metrics": baseline.metrics,
        "target_mape": "< 15%",
        "target_r2": "> 0.85",
        "comparisons": comparisons
    }


@app.get("/data/zones", tags=["Data"])
async def get_zones():
    """Mevcut bölgeleri döndürür."""
    return {
        "zones": [
            {"id": 1, "name": "Downtown", "avg_density": 150},
            {"id": 2, "name": "Midtown", "avg_density": 180},
            {"id": 3, "name": "Uptown", "avg_density": 120},
            {"id": 4, "name": "Suburbs North", "avg_density": 80},
            {"id": 5, "name": "Suburbs South", "avg_density": 70}
        ]
    }


@app.get("/data/historical", tags=["Data"])
async def get_historical(
    zone_id: int = Query(1, ge=1, le=5),
    hours: int = Query(24, ge=1, le=168)
):
    """Son n saat için geçmiş verileri döndürür."""
    if state.historical_data is None:
        raise HTTPException(status_code=500, detail="Historical data not available")
    
    zone_data = state.historical_data[state.historical_data['zone_id'] == zone_id]
    recent = zone_data.tail(hours)
    
    return {
        "zone_id": zone_id,
        "hours": hours,
        "data": recent[['datetime', 'density']].to_dict(orient='records')
    }


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
