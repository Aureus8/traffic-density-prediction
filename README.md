# Traffic Density Prediction System

Gerçek zamanlı trafik yoğunluğu tahmin sistemi. Harici değişkenleri (hava durumu, tatiller) entegre ederek zaman serisi tahmin hatalarını minimize eder.

## 🎯 Proje Hedefleri

| Metrik | Hedef | Açıklama |
|--------|-------|----------|
| MAPE | < 15% | Mean Absolute Percentage Error |
| R² | > 0.85 | Coefficient of Determination |
| Latency | < 100ms | API yanıt süresi |
| Improvement | > 20% | Baseline'a göre iyileşme |

## 🏗️ Proje Yapısı

```
traffic-density-prediction/
├── src/
│   ├── data/           # Veri yükleme ve ön işleme
│   ├── models/         # ML modelleri (Baseline, SARIMAX, LSTM, Ensemble)
│   ├── external/       # Harici API'ler (hava durumu, tatiller)
│   ├── evaluation/     # Metrikler ve görselleştirme
│   └── api/            # FastAPI servisi
├── data/               # Veri dosyaları
├── notebooks/          # Jupyter notebooks
├── tests/              # Unit testler
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🚀 Kurulum

### 1. Repository'yi klonla
```bash
git clone <repository-url>
cd traffic-density-prediction
```

### 2. Virtual environment oluştur
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
.\venv\Scripts\activate  # Windows
```

### 3. Bağımlılıkları yükle
```bash
pip install -r requirements.txt
```

### 4. Environment değişkenlerini ayarla
```bash
cp .env.example .env
# .env dosyasını düzenle (opsiyonel)
```

## 💻 Kullanım

### API Servisi Başlatma

```bash
# Development
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production (Docker)
docker-compose up --build
```

### API Endpoints

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/health` | GET | Sağlık kontrolü |
| `/predict` | POST | Tek nokta tahmin |
| `/predict/batch` | POST | Batch tahmin (1-168 saat) |
| `/model/metrics` | GET | Model performans metrikleri |
| `/model/compare` | GET | Model karşılaştırması |

### Örnek API İsteği

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "datetime": "2024-01-15T14:00:00",
    "zone_id": 1,
    "include_weather": true
  }'
```

### Python ile Kullanım

```python
from src.data.loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.models.baseline import ARIMAModel
from src.models.ensemble import EnsembleModel
from src.evaluation.metrics import ModelMetrics

# Veri yükle
loader = DataLoader()
df = loader.generate_synthetic_data()

# Feature engineering
engineer = FeatureEngineer()
df = engineer.engineer_features(df)

# Model eğit
model = ARIMAModel(order=(2, 1, 1))
model.fit(df)

# Tahmin
predictions = model.predict(24)  # 24 saat
```

## 📊 Modeller

### Baseline Modeller
- **Naive**: Son değeri kullan
- **Moving Average**: Hareketli ortalama
- **ARIMA**: AutoRegressive Integrated Moving Average

### İstatistiksel Modeller
- **SARIMAX**: Seasonal ARIMA + harici değişkenler
- **Prophet**: Facebook Prophet (trend + seasonality)

### Derin Öğrenme
- **LSTM**: Long Short-Term Memory
- **GRU**: Gated Recurrent Unit

### Ensemble
- **Weighted Average**: Performansa göre ağırlıklı ortalama
- **Stacking**: Meta-learner ile kombinasyon

## ⚠️ Overfitting Kontrolü

Proje kapsamında uygulanan önlemler:

1. **Early Stopping**: Validation loss iyileşmezse eğitimi durdur
2. **Dropout**: Neural network katmanlarında %20-30 dropout
3. **L2 Regularization**: Weight decay ile regularization
4. **Time Series Split**: Kronolojik cross-validation
5. **Learning Curves**: Train vs validation loss izleme

```python
# Overfitting analizi
from src.models.deep_learning import LSTMModel

model = LSTMModel(dropout=0.3, weight_decay=0.0001)
history = model.fit(X_train, y_train, X_val, y_val, early_stopping_patience=10)

# Analiz
analysis = model.analyze_overfitting()
print(analysis['status'])  # 'GOOD FIT', 'OVERFITTING', 'UNDERFITTING'
```

## 🧪 Testler

```bash
# Tüm testleri çalıştır
pytest tests/ -v

# Coverage ile
pytest tests/ --cov=src --cov-report=html
```

## 📈 Performans Görselleştirme

```python
from src.evaluation.visualizer import PerformanceVisualizer

viz = PerformanceVisualizer()

# Learning curves (overfitting detection)
viz.plot_learning_curves(train_losses, val_losses)

# Model karşılaştırma
viz.plot_model_comparison(comparison_results)

# Residual analizi
viz.plot_residuals(y_true, y_pred)
```

## 🐳 Docker

```bash
# Build
docker build -t traffic-density-api .

# Run
docker run -p 8000:8000 traffic-density-api

# Docker Compose
docker-compose up --build
```

## 📝 Lisans

MIT License

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın
