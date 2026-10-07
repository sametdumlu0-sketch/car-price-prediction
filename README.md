# 🏎️ AutoValue-UK: 100K Araç Verisiyle Uçtan Uca İkinci El Fiyat Değerleme Motoru

> **CRISP-DM Metodolojisi & Histogram Tabanlı Gradyan Artırma ile %93.8 R² Skoru ve %8.0 Ortalama Hata Marjı**

[![Python Version](https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-4B8BBE.svg)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
[![Model Performance](https://img.shields.io/badge/R%C2%B2%20Score-0.9383-success.svg)](#-model-başarım-kriterleri-benchmark)
[![Mean Absolute Error](https://img.shields.io/badge/MAE-%C2%A31%2C339-orange.svg)](#-model-başarım-kriterleri-benchmark)
[![Production Ready](https://img.shields.io/badge/Pipeline-Joblib%20Exported-brightgreen.svg)](#-canlı-tahmin-servisi-kullanımı)

---

## 📌 Proje Genel Bakışı

İkinci el otomobil piyasasında doğru ve adil fiyatlandırma; marka algısı, araç yaşı, kullanım yoğunluğu (kilometre), motor hacmi ve yakıt verimliliği gibi çok boyutlu ve doğrusal olmayan değişkenlerin bileşkesine dayanır. 

Bu projede, Birleşik Krallık (UK) pazarındaki **9 küresel otomotiv markasına** (Audi, BMW, Ford, Hyundai, Mercedes-Benz, Skoda, Toyota, Vauxhall, Volkswagen) ait **99.187 adet** gerçek pazar kaydı analiz edilmiş; veri temizliğinden dağıtıma kadar tüm aşamalar uluslararası endüstri standardı olan **CRISP-DM** metodolojisi izlenerek kurumsal seviyede bir fiyat değerleme motoruna dönüştürülmüştür.

---

## 🎯 Temel Başarım Kriterleri (Benchmark)

Eğitim sürecinde modele hiç gösterilmeyen **19.488 adetlik bağımsız test kümesi** üzerindeki sonuçlar:

| Model Mimarisi | MAE (Ortalama Mutlak Hata) | RMSE (Kök Ortalama Kare Hata) | $R^2$ Skoru | MAPE (Yüzdesel Hata) |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regresyon (Doğrusal Baseline)** | £2,360.14 | £4,312.80 | 0.8124 | %14.85 |
| **HistGradientBoosting (Gelişmiş Model)** | **£1,339.85** | **£2,415.99** | **0.9383** | **%8.00** |

* 🚀 **%43 Hata Azalımı:** Ağaç tabanlı gradyan artırma modeli, doğrusal referans modeline kıyasla mutlak hata tutarını £1,020 aşağı çekmiştir.
* 📈 **%93.83 Açıklanan Varyans:** Fiyat hareketlerinin %93.8'i model tarafından yüksek doğrulukla açıklanmaktadır.
* 🎯 **%8.00 Ortalama Bağıl Hata:** Model tahminleri ortalamada gerçek pazar fiyatının yalnızca %8 yakınında konumlanmaktadır.

---

## 🔍 Saha ve Veri Mühendisliği Dokunuşları (Under the Hood)

Standart ve yapay yaklaşımların ötesinde, veri setine yönelik kritik veri bilimi kararları uygulanmıştır:

1. **Hedef Değişken Normalizasyonu ($log1p$):** 
   Araç fiyatları belirgin şekilde sağa çarpık (right-skewed) bir dağılıma sahiptir. Fiyata doğrudan regresyon uygulamak pahalı lüks araçlarda büyük hatalara tolerans tanırken ucuz araçlarda dengesiz sapmalar üretir. Hedef değişkene $y = \log(1 + \text{price})$ dönüşümü uygulanarak artık hataların homojenliği (homoskedastisite) sağlanmış ve oransal hata minimize edilmiştir.

2. **Karakter ve String Temizliği:**
   Kaggle veri setlerindeki kronik model başı boşlukları (`' A1'` $\rightarrow$ `'A1'`) giderilmiş, kategorik eşleşmeler garanti altına alınmıştır.

3. **Mantıksal Sınır ve Anomali Ayıklama:**
   Veri tabanında yer alan imkansız tekil kayıt (`year = 2060`), 1995 öncesi aşırı yıpranmış kayıtlar ve motor hacmi `0` girilmiş eksik veriler temizlenmiştir (97.436 temiz kayıt).

4. **Alan Bilgisine Dayalı Özellik Mühendisliği (Domain-driven Feature Engineering):**
   * `car_age = 2020 - year`: Verinin derlendiği baz yıl üzerinden gerçek yaş parametresi.
   * `mileage_per_year = mileage / (car_age + 1)`: Yıllık ortalama sürüş yoğunluğu (aracın hor kullanılıp kullanılmadığını yakalayan gösterge).

5. **Sıfır Veri Sızıntısı (Zero Data Leakage Pipeline):**
   Ön işleme transformer'ları (`RobustScaler` ve `OneHotEncoder`), `train_test_split` adımından sonra yalnızca eğitim kümesi üzerinde `fit` edilmiş, test kümesine bilgi sızması kesin olarak engellenmiştir.

---

## 🚘 Örnek Pazar Değerleme Senaryoları

Eğitilen model, farklı segmentlerdeki araçlar için şu piyasa değerleme çıktılarını üretmiştir:

| Segment | Araç Profili | Model Tahmini | Beklenen Güven Aralığı (±%8) |
| :--- | :--- | :---: | :---: |
| **Premium Sedan** | 2017 Audi A4 2.0 TDI (Dizel, Otomatik, 28k mil) | **£19,467.66** | £17,910 – £21,025 |
| **Aile Hatchback** | 2018 Volkswagen Golf 1.5 TSI (Benzin, Manuel, 32k mil) | **£15,429.23** | £14,195 – £16,664 |
| **Şehir / Giriş** | 2019 Ford Fiesta 1.0 EcoBoost (Benzin, Manuel, 18k mil) | **£12,770.01** | £11,748 – £13,792 |

---

## 🛠️ Proje Mimarisi & Dizin Yapısı

```
car-analyz/
│
├── data_eye.ipynb          # Eksiksiz 6 fazlı CRISP-DM analiz ve modelleme notebook'u
├── CRISP_DM_RAPORU.md      # Ayrıntılı metodoloji ve iş analitiği teknik raporu
├── README.md               # GitHub ana dokümantasyonu
├── requirements.txt        # Yeniden üretilebilir ortam bağımlılıkları
├── car_price_model.joblib  # Üretim için serileştirilmiş uçtan uca pipeline
│
├── audi.csv                # Marka bazlı veri setleri (~100k satır)
├── bmw.csv
├── ford.csv
├── hyundi.csv
├── merc.csv
├── skoda.csv
├── toyota.csv
├── vauxhall.csv
└── vw.csv
```

---

## 💻 Kurulum ve Hızlı Başlangıç

### 1. Depoyu Klonlayın ve Ortamı Hazırlayın
```bash
git clone https://github.com/sametdumlu/car-price-prediction.git
cd car-price-prediction

# Sanal ortam oluşturma ve bağımlılıkları yükleme
python -m venv .venv
source .venv/bin/activate  # Windows için: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Canlı Tahmin Servisi Kullanımı
Eğitilmiş model pipeline'ını yükleyip tek satırda araç değerleme tahmini alabilirsiniz:

```python
import joblib
import pandas as pd
import numpy as np

# Kayıtlı modeli yükleyin
model = joblib.load('car_price_model.joblib')

# Örnek araç girdisi
yeni_arac = pd.DataFrame([{
    'brand': 'audi',
    'model': 'A4',
    'transmission': 'Automatic',
    'fuelType': 'Diesel',
    'car_age': 3,
    'mileage': 28000,
    'tax': 145,
    'mpg': 67.3,
    'engineSize': 2.0,
    'mileage_per_year': 7000.0
}])

# Fiyat tahmini (£)
tahmin_fiyat = np.expm1(model.predict(yeni_arac)[0])
print(f"Araç Tahmini Piyasa Değeri: £{tahmin_fiyat:,.2f}")
# Çıktı: Araç Tahmini Piyasa Değeri: £19,467.66
```

---

## 📜 Lisans & Yazar

* **Geliştirici:** Samet Dumlu
* **E-Posta:** sametdumlu0@gmail.com
* **Metodoloji:** CRISP-DM Framework
