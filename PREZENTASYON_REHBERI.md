# 🎓 Trafik Tahmin Projesi - Prezentasyon Rehberi

Bu rehber, projenizi başka bir bilgisayarda (örn. hocanızın bilgisayarında) çalıştırmak için gereken tüm adımları içerir.

## 📋 Ön Hazırlık (Sunum Öncesi)

### Gerekli Programlar
Sunum yapacağınız bilgisayarda şunların kurulu olduğundan emin olun:

1. **Python 3.8 veya üzeri** 
   - İndirme: https://www.python.org/downloads/
   - Kurulum sırasında "Add Python to PATH" seçeneğini işaretleyin

2. **Git** (Opsiyonel - internet varsa)
   - İndirme: https://git-scm.com/downloads

3. **İnternet Bağlantısı** (İlk kurulum için gerekli)

---

## 🚀 Hızlı Kurulum (Adım Adım)

### Seçenek 1: GitHub'dan İndirme (İNTERNET VARSA - ÖNERİLEN)

#### Adım 1: Projeyi İndirin
```bash
# Terminal veya PowerShell açın
git clone https://github.com/Aureus8/traffic-density-prediction.git
cd traffic-density-prediction
```

Alternatif (Git yoksa):
- GitHub'dan ZIP olarak indirin: https://github.com/Aureus8/traffic-density-prediction
- ZIP'i masaüstüne çıkarın
- Terminal'de klasöre gidin: `cd Desktop/traffic-density-prediction`

#### Adım 2: Virtual Environment Oluşturun
```bash
# Windows için
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux için
python3 -m venv venv
source venv/bin/activate
```

**Başarılı olduğunu nasıl anlarım?**
Terminalde komut satırının başında `(venv)` yazısını görmelisiniz.

#### Adım 3: Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```

⏱️ Bu işlem 2-5 dakika sürebilir. Bekleyin...

#### Adım 4: Environment Dosyasını Hazırlayın
```bash
# Windows için
copy .env.example .env

# Mac/Linux için
cp .env.example .env
```

#### Adım 5: API'yi Başlatın
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Başarılı olduğunu nasıl anlarım?**
Şu mesajı görmelisiniz:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

#### Adım 6: Tarayıcıda Test Edin
Tarayıcınızı açın ve şu adresleri ziyaret edin:

1. **API Dokümantasyonu**: http://localhost:8000/docs
2. **Health Check**: http://localhost:8000/health

---

### Seçenek 2: USB ile Taşıma (İNTERNET YOKSA)

#### Hazırlık (Kendi bilgisayarınızda yapın):

1. **Projeyi USB'ye kopyalayın**
```bash
# Venv klasörünü hariç tutarak kopyalayın
robocopy traffic-density-prediction E:\presentation-project /E /XD venv __pycache__ .git
```

2. **Requirements.txt'i önceden indirin**
```bash
# Wheel dosyalarını indirin
pip download -r requirements.txt -d packages/
```
Bu `packages/` klasörünü de USB'ye kopyalayın.

#### Sunum Bilgisayarında:

1. **USB'den masaüstüne kopyalayın**

2. **Virtual Environment oluşturun**
```bash
cd Desktop/presentation-project
python -m venv venv
.\venv\Scripts\activate  # Windows
```

3. **Paketleri USB'den yükleyin**
```bash
pip install --no-index --find-links=packages/ -r requirements.txt
```

4. **API'yi başlatın**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 📱 Demo Senaryoları

### Demo 1: Basit Tahmin (Tekli)

Tarayıcıda http://localhost:8000/docs adresine gidin.

1. `/predict` endpoint'ini açın
2. "Try it out" butonuna tıklayın
3. Şu JSON'u girin:
```json
{
  "datetime": "2024-01-15T14:00:00",
  "zone_id": 1,
  "include_weather": true
}
```
4. "Execute" butonuna tıklayın
5. Sonucu gösterin!

**Beklenen Çıktı:**
```json
{
  "datetime": "2024-01-15T14:00:00",
  "zone_id": 1,
  "predicted_density": 87.5,
  "confidence_interval": {
    "lower": 75.2,
    "upper": 99.8
  },
  "model_used": "ensemble",
  "weather_included": true
}
```

---

### Demo 2: Toplu Tahmin (Batch)

1. `/predict/batch` endpoint'ini açın
2. "Try it out" butonuna tıklayın
3. Şu JSON'u girin:
```json
{
  "start_datetime": "2024-01-15T00:00:00",
  "zone_id": 1,
  "hours_ahead": 24,
  "include_weather": true
}
```
4. "Execute" butonuna tıklayın
5. 24 saatlik tahmin sonuçlarını gösterin!

---

### Demo 3: Model Performansı

1. `/model/metrics` endpoint'ini açın
2. "Try it out" ve "Execute" butonlarına tıklayın
3. Model performans metriklerini gösterin:
   - MAPE (Ortalama Mutlak Yüzde Hatası)
   - R² (Belirleme Katsayısı)
   - RMSE (Kök Ortalama Kare Hatası)

**İyi sonuçlar:**
- MAPE < 15%
- R² > 0.85
- RMSE düşük

---

### Demo 4: Model Karşılaştırması

1. `/model/compare` endpoint'ini açın
2. "Execute" butonuna tıklayın
3. Farklı modellerin performansını karşılaştırın
4. Ensemble modelinin en iyi sonucu verdiğini gösterin

---

## 🎯 Sunum İpuçları

### Söyleyeceğiniz Şeyler:

1. **Proje açıklaması:**
   > "Bu proje, gerçek zamanlı trafik yoğunluğunu tahmin eden bir sistem. Hava durumu ve tatiller gibi harici faktörleri de dikkate alıyor."

2. **Teknoloji stack:**
   > "Python, FastAPI, LSTM, SARIMAX ve Ensemble öğrenme yöntemlerini kullandım. API ile kolay entegrasyon sağlıyor."

3. **Model seçimi:**
   > "Baseline, istatistiksel ve derin öğrenme modellerini karşılaştırdım. Ensemble model en iyi performansı gösterdi."

4. **Overfitting kontrolü:**
   > "Early stopping, dropout ve cross-validation ile overfitting'i önledim. Train/validation loss farkına dikkat ettim."

### Gösterilecek Önemli Noktalar:

✅ API dokümantasyonu (Swagger UI)
✅ Gerçek zamanlı tahmin yapabilme
✅ Model performans metrikleri
✅ Batch tahmin özelliği (24-168 saat)
✅ Health check endpoint

---

## ⚠️ Olası Sorunlar ve Çözümler

### Sorun 1: "uvicorn: command not found"
**Çözüm:**
```bash
# Virtual environment'i aktif etmeyi unutmuş olabilirsiniz
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Veya doğrudan çalıştırın
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Sorun 2: "Port 8000 already in use"
**Çözüm:**
```bash
# Farklı bir port kullanın
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8080
# Tarayıcıda: http://localhost:8080/docs
```

### Sorun 3: "ModuleNotFoundError: No module named 'src'"
**Çözüm:**
```bash
# Doğru klasörde olduğunuzdan emin olun
cd traffic-density-prediction

# PYTHONPATH ayarlayın
set PYTHONPATH=.  # Windows
export PYTHONPATH=.  # Mac/Linux
```

### Sorun 4: Model dosyaları bulunamıyor
**Çözüm:**
```bash
# Model eğitim scriptini çalıştırın
python scripts/train_models.py

# Veya önceden eğitilmiş modelleri kullanın (USB'de taşıyın)
```

### Sorun 5: İnternet bağlantısı yok (Hava durumu API)
**Çözüm:**
API'de include_weather parametresini false yapın:
```json
{
  "datetime": "2024-01-15T14:00:00",
  "zone_id": 1,
  "include_weather": false
}
```

---

## 📦 Tam Offline Paket Hazırlama

Hiç internet olmayacak bir ortamda sunum yapacaksanız:

1. **Kendi bilgisayarınızda şunları yapın:**

```bash
# 1. Tüm Python paketlerini indirin
pip download -r requirements.txt -d packages/

# 2. Projeyi hazırlayın (venv'siz)
# Git klasörünü, cache'leri temizleyin
```

2. **USB'ye şunları kopyalayın:**
   - Proje klasörü (venv hariç)
   - packages/ klasörü
   - Bu rehber (PREZENTASYON_REHBERI.md)

3. **Sunum bilgisayarında:**
   - USB'den kopyalayın
   - `pip install --no-index --find-links=packages/ -r requirements.txt`
   - API'yi başlatın

---

## ⏱️ Zaman Planlaması

**Toplam süre: ~10 dakika**

| Adım | Süre |
|------|------|
| Projeyi indirme/kopyalama | 1 dk |
| Virtual environment oluşturma | 1 dk |
| Paket kurulumu | 3-5 dk |
| API başlatma | 30 sn |
| Demo/test | 3-5 dk |

**İpucu:** Kurulum adımlarını sunum öncesi yapın, sadece API'yi başlatıp demo yapın!

---

## 🎬 Hızlı Başlangıç (Tek Komut)

**Tek seferde her şeyi yapmak için:**

```bash
git clone https://github.com/Aureus8/traffic-density-prediction.git && \
cd traffic-density-prediction && \
python -m venv venv && \
.\venv\Scripts\activate && \
pip install -r requirements.txt && \
copy .env.example .env && \
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 📧 Acil Durum

Eğer hiçbir şey çalışmazsa:

1. **Video gösterimi hazırlayın**: Kendi bilgisayarınızda çalışırken ekran kaydı alın
2. **Ekran görüntüleri**: API dokümantasyonu ve sonuçların ekran görüntülerini alın
3. **Postman Collection**: Hazır API isteklerini Postman'de kaydedin

---

## ✅ Son Kontrol Listesi

Sunum öncesi kontrol edin:

- [ ] Python kurulu (3.8+)
- [ ] Git kurulu veya ZIP indirildi
- [ ] İnternet bağlantısı var (ilk kurulum için)
- [ ] Proje GitHub'dan erişilebilir
- [ ] Bu rehber USB'de
- [ ] Tarayıcı hazır
- [ ] Demo senaryoları ezberinde
- [ ] Postman kurulu (opsiyonel)

---

## 🎓 Başarılar!

Bu rehberi takip ederseniz, projenizi sorunsuz bir şekilde gösterebilirsiniz. Bol şans! 🚀
