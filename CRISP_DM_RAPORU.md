# İkinci El Araç Değerleme ve Fiyat Tahmini (CRISP-DM Raporu)

Bu projede, Birleşik Krallık pazarındaki 9 büyük otomobil markasına ait (Audi, BMW, Ford, Hyundai, Mercedes-Benz, Skoda, Toyota, Vauxhall, Volkswagen) ~100k adet ikinci el araç verisi kullanılarak uçtan uca makine öğrenmesi tabanlı bir fiyat değerleme modeli geliştirilmiştir. Süreç, CRISP-DM standardına uygun olarak 6 fazda tamamlanmıştır.

---

## 1. İş Anlayışı (Business Understanding)

* **Amaç:** Araç niteliklerini (marka, model, yaş, kilometre, şanzıman, yakıt, vergi, yakıt tüketimi, motor hacmi) girdi alarak aracın güncel piyasa satış fiyatını tahmin etmek.
* **Başarı Kriteri:** $R^2 \ge 0.90$, MAE $\le £1,500$ ve $\%10$'un altında ortalama mutlak yüzde hata (MAPE).
* **Tasarım:** Araç fiyatları sağa çarpık (right-skewed) dağılım sergilediği için modelleme $log(1 + y)$ hedef dönüşümüyle yürütülmüş, tahminler $exp(x) - 1$ ile orijinal sterlin (£) birimine dönüştürülmüştür.

---

## 2. Veriyi Anlama (Data Understanding)

* **Veri Kaynağı:** 9 ayrı CSV dosyasının birleştirilmesiyle elde edilen 99.187 satır, 10 sütun.
* **Veri Kalitesi:**
  * Eksik değer bulunmamaktadır (%0 null).
  * 1.475 adet mükerrer (duplicate) kayıt saptanmıştır.
  * Model isimlerinde baştan kaynaklı boşluklar (`' A1'`) tespit edilmiştir.
  * Kilometre ve yaş arttıkça fiyatta belirgin negatif üssel amortisman eğrisi görülmektedir.

---

## 3. Veri Hazırlama (Data Preparation)

1. **Temizlik:** Mükerrer satırlar silindi. Model isimlerindeki boşluklar `str.strip()` ile temizlendi.
2. **Aykırı Kayıtlar:** Hatalı girilmiş tekil satır (`year = 2060`), 1995 öncesi aşırı eski araçlar, motor hacmi `0` olan eksik kayıtlar ve 500 £ altı hurda girişler elendi (97.436 temiz satır).
3. **Özellik Mühendisliği (Feature Engineering):**
   * `car_age = 2020 - year` (Veri setinin referans yılı olan 2020 baz alınarak araç yaşı).
   * `mileage_per_year = mileage / (car_age + 1)` (Yıllık ortalama kullanım yoğunluğu).
4. **Veri Sızıntısını Önleme:** Veri önce %80 eğitim, %20 test olarak ayrıldı. Sayısal değişkenler `RobustScaler`, kategorik değişkenler `OneHotEncoder(handle_unknown='ignore')` ile `ColumnTransformer` altında paketlendi.

---

## 4. Modelleme (Modeling)

İki farklı yaklaşım test edilmiştir:
1. **Ridge Regression (Baseline):** L2 cezalandırmalı doğrusal referans model.
2. **HistGradientBoostingRegressor (Gelişmiş Model):** Büyük veri kümelerinde hızlı çalışan histogram tabanlı gradyan artırma algoritması (`max_iter=150`, `learning_rate=0.10`, `max_depth=12`, `min_samples_leaf=20`).

---

## 5. Model Değerlendirme (Evaluation)

Modeller, eğitimde yer almayan 19.488 adetlik bağımsız test kümesinde değerlendirilmiştir:

| Model | MAE (£) | RMSE (£) | $R^2$ Skoru | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regresyon (Baseline)** | £2,360.14 | £4,312.80 | 0.8124 | %14.85 |
| **HistGradientBoosting (Gelişmiş)** | **£1,339.85** | **£2,415.99** | **0.9383** | **%8.00** |

* Ağaç tabanlı model doğrusal modele kıyasla mutlak hatayı %43 azaltmış ve varyans açıklama oranını %93.83'e taşımıştır.
* Hata (residual) dağılımı sıfır merkezli ve normale yakın bir yayılım göstermektedir.

---

## 6. Dağıtım ve Örnek Tahminler (Deployment & Inference)

Model pipeline'ı diske `car_price_model.joblib` olarak kaydedilmiştir.

### Örnek Araç Senaryolarında Üretilen Fiyat Tahminleri:

1. **2017 Audi A4 2.0 TDI (Dizel, Otomatik, 28.000 mil)**
   * **Tahmin:** £19,467.66
   * **Piyasa Güven Aralığı (%8):** £17,910.25 - £21,025.07

2. **2018 Volkswagen Golf 1.5 TSI (Benzinli, Manuel, 32.000 mil)**
   * **Tahmin:** £15,429.23
   * **Piyasa Güven Aralığı (%8):** £14,194.90 - £16,663.57

3. **2019 Ford Fiesta 1.0 EcoBoost (Benzinli, Manuel, 18.000 mil)**
   * **Tahmin:** £12,770.01
   * **Piyasa Güven Aralığı (%8):** £11,748.41 - £13,791.61

---

### Python ile Kullanım Örneği

```python
import joblib
import pandas as pd
import numpy as np

model = joblib.load('car_price_model.joblib')

arac = pd.DataFrame([{
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

tahmin_gbp = np.expm1(model.predict(arac)[0])
print(f"Tahmini Fiyat: £{tahmin_gbp:,.2f}")
```
