"""
API Tests

FastAPI endpoint tests.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Health endpoint tests."""
    
    def test_health_check(self):
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models_loaded" in data
        assert "uptime_seconds" in data


class TestPredictionEndpoint:
    """Prediction endpoint tests."""
    
    def test_single_prediction(self):
        response = client.post(
            "/predict",
            json={
                "datetime": "2024-01-15T14:00:00",
                "zone_id": 1,
                "include_weather": True,
                "include_confidence": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predicted_density" in data
        assert "datetime" in data
        assert "zone_id" in data
        assert "latency_ms" in data
        assert data["latency_ms"] < 1000  # Should be fast
    
    def test_prediction_with_confidence(self):
        response = client.post(
            "/predict",
            json={
                "datetime": "2024-01-15T14:00:00",
                "zone_id": 1,
                "include_weather": False,
                "include_confidence": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["confidence_lower"] is not None
        assert data["confidence_upper"] is not None
        assert data["confidence_lower"] < data["predicted_density"]
        assert data["confidence_upper"] > data["predicted_density"]
    
    def test_invalid_datetime(self):
        response = client.post(
            "/predict",
            json={
                "datetime": "invalid-date",
                "zone_id": 1
            }
        )
        
        assert response.status_code == 400


class TestBatchPredictionEndpoint:
    """Batch prediction endpoint tests."""
    
    def test_batch_prediction(self):
        response = client.post(
            "/predict/batch",
            json={
                "start_datetime": "2024-01-15T00:00:00",
                "hours": 5,
                "zone_id": 1
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["predictions"]) == 5
        assert data["total_hours"] == 5
    
    def test_batch_max_hours(self):
        # Test that hours > 168 fails validation
        response = client.post(
            "/predict/batch",
            json={
                "start_datetime": "2024-01-15T00:00:00",
                "hours": 200,
                "zone_id": 1
            }
        )
        
        assert response.status_code == 422  # Validation error


class TestModelEndpoints:
    """Model info endpoint tests."""
    
    def test_get_metrics(self):
        response = client.get("/model/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) > 0
        for model in data:
            assert "model_name" in model
            assert "metrics" in model
            assert "mape" in model["metrics"]
            assert "r2" in model["metrics"]
    
    def test_compare_models(self):
        response = client.get("/model/compare")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "baseline" in data
        assert "comparisons" in data
        assert len(data["comparisons"]) > 0


class TestDataEndpoints:
    """Data endpoint tests."""
    
    def test_get_zones(self):
        response = client.get("/data/zones")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "zones" in data
        assert len(data["zones"]) == 5
    
    def test_get_historical(self):
        response = client.get("/data/historical?zone_id=1&hours=10")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["zone_id"] == 1
        assert data["hours"] == 10


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
