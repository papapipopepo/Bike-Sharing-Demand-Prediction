# Capstone Module 2 — Bike Sharing Demand Prediction (Regression)

Membangun model regresi untuk memprediksi jumlah peminjaman sepeda per jam (`cnt`) dari data cuaca dan waktu. Output model bisa dipakai sebagai baseline untuk perencanaan operasional (rebalancing sepeda, estimasi demand jam sibuk, dll).

---

## Project Overview

**Target**
- `cnt`: total rental per jam

**Catatan penting**
- Kolom `casual` dan `registered` **tidak dipakai** karena leakage (`cnt = casual + registered`).

**Approach**
- Feature engineering berbasis waktu (timestamp, month, day_of_week, year, time_of_day, is_peak_hour).
- Split data **berdasarkan waktu** (time-based split) untuk menghindari data leakage antar waktu.
- Pipeline preprocessing dengan `ColumnTransformer`:
  - numerik: imputer (median) + scaler
  - kategorikal: imputer (most_frequent) + OneHotEncoder (dense output)
- Benchmark beberapa model + hyperparameter tuning untuk kandidat terbaik.

---

## Repository Structure

- `Capstone Modul 2.ipynb` : notebook utama (end-to-end)
- `data_bike_sharing.csv` : dataset (letakkan di folder yang sama atau sesuaikan path)
- `bike_sharing_xgboost.pkl` (atau nama serupa) : model final hasil training (pickle)

---

## Environment & Requirements

Disarankan pakai Python 3.9+.

Dependencies utama:
- numpy, pandas
- matplotlib, seaborn
- scikit-learn
- xgboost

Install cepat:

```bash
pip install -U numpy pandas matplotlib seaborn scikit-learn xgboost
```

## How to Run

1. Pastikan file dataset ada di folder yang sama dengan notebook:
   - `data_bike_sharing.csv`

2. Jalankan notebook dari atas sampai bawah.

3. Notebook akan melakukan:
   - preprocessing + feature engineering  
   - benchmark + tuning (HGB dan XGBoost)  
   - evaluasi test set + residual plots  
   - segment analysis  
   - menyimpan model final ke file `.pkl`

---

## Results

Model final yang terpilih: **XGBoost (tuned)**

### Test set performance
- **RMSE**: ~65.36  
- **MAE**: ~43.00  
- **MAPE**: ~62.37%  
- **R²**: ~0.912  

### Catatan metrik
- **MAPE** cenderung tinggi untuk data demand per jam karena saat nilai aktual kecil (mis. malam), persentase error mudah membesar.
- Untuk konteks operasional, **MAE** dan **RMSE** lebih representatif.

---

## Key Insights

### Feature importance (high level)
Fitur waktu sangat dominan (contoh: `time_of_day`, `hr`, `is_peak_hour`), disusul fitur kalender dan cuaca.

### Segment performance
- Error jauh lebih tinggi di **jam sibuk (peak hours)** dibanding non-peak.
- Jam yang paling “sulit” umumnya sekitar jam commute (misal 7–9 dan 16–19).
- Error meningkat pada kondisi tertentu seperti **musim Winter** dan **cuaca buruk** (mis. `weathersit = 3`).

---

## Limitations
- Model belum memakai fitur historis (lag/rolling) yang biasanya sangat membantu forecasting demand time series.
- **MAPE** kurang stabil ketika `cnt` kecil.

---

## Future Improvements
- Tambahkan fitur time-series yang aman (tanpa leakage), misalnya:
  - `lag_1`, `lag_24`
  - rolling mean (3 jam / 24 jam) menggunakan `shift`
- Tambahkan metrik yang lebih stabil untuk persentase error:
  - **sMAPE** / **WAPE**
- Evaluasi per segmen lebih lanjut (holiday, weather extremes, peak vs non-peak) untuk strategi operasional yang lebih presisi.

---

## Model Saving (Pickle)

Model disimpan sebagai **Pipeline** (preprocess + model), jadi setelah load bisa langsung `predict(X)`.

```python
import pickle

with open("bike_sharing_xgboost.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_input)

