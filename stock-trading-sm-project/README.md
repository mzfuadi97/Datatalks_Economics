# Trading Strategy ML Project

Project ini saya buat untuk tugas end-to-end machine learning di bidang
trading: mulai dari ambil data, bikin fitur, latih model, backtest strategi,
sampai optimasi portofolio. Semua langkah dijalankan oleh satu pipeline
(`main.py`) dan bisa dijalankan di Docker.

## Ringkasan

- Universe: 25 saham/ETF AS (AAPL, MSFT, NVDA, TSLA, XOM, dst.)
- Benchmark: S&P 500 (SPY)
- Periode: 2018-01-02 sampai 2023-12-29 (harian)
- Dataset: 37.725 baris, 81 kolom (73 fitur teknikal + kolom benchmark)
- Target: arah return 5 hari ke depan, dibagi 3 kelas (turun / tahan / naik)
- Model terbaik: XGBoost (F1 tertinggi; akurasi 0.46, ROC-AUC 0.60)

## Struktur

```
stock-trading-sm-project/
├── main.py                        # jalankan semua langkah
├── Dockerfile
├── requirements.txt
├── scripts/
│   ├── config.py                  # path & logging
│   ├── data_collector.py          # unduh OHLCV + indikator + benchmark SPY
│   ├── data_processor.py          # bersihkan data + bikin target
│   ├── train_ml_models.py         # latih DT / RF / XGBoost
│   ├── simulate_trading.py        # backtest strategi vs SPY
│   └── portfolio_optimization.py  # equal-weight / inv-vol / max-Sharpe / min-vol
├── data/
│   ├── raw/
│   └── processed/
└── models/
```

## Alur

**1. Ambil data** (`data_collector.py`)
Pakai `yfinance` untuk OHLCV harian, lalu hitung 73 fitur dengan library `ta`:
tren (SMA/EMA/MACD/ADX/KST), momentum (RSI, ROC, Williams %R, Stochastic),
volatilitas (ATR, Bollinger Bands) dan volume (OBV, MFI, force index). Saya
juga unduh SPY sebagai benchmark dan menyimpan `BENCHMARK_returns` (dipakai
untuk perbandingan, bukan sebagai fitur model).

**2. Praproses** (`data_processor.py`)
Bersihkan NaN, buang kolom yang bocor (leakage), lalu bentuk target dari
return 5 hari ke depan: `-1` untuk turun, `0` untuk tahan, `1` untuk naik.

Distribusi target: tahan sekitar 42%, naik 33%, turun 25%.

**3. Latih model** (`train_ml_models.py`)
Split 80/20 pada batas tanggal (bukan jumlah baris), supaya tidak ada tanggal
yang masuk ke train sekaligus test. Coba Decision Tree, Random Forest, dan
XGBoost dengan grid kecil. Kolom `future_returns`, `date`, dan kolom
`BENCHMARK_*` tidak dipakai sebagai fitur.

| Model | Akurasi | Precision | F1 | ROC-AUC | Recall (naik) |
|-------|---------|-----------|-----|---------|---------------|
| Decision Tree | 0.407 | 0.399 | 0.398 | 0.552 | 0.247 |
| Random Forest | 0.467 | 0.433 | 0.424 | 0.597 | 0.363 |
| XGBoost | 0.463 | 0.438 | 0.432 | 0.597 | 0.294 |

ROC-AUC di sini dihitung untuk event "naik" (target == 1) terhadap `prob_up`.
Model XGBoost dipilih karena F1 tertinggi (0.432), walaupun Random Forest
memiliki akurasi sedikit lebih tinggi (0.467 vs 0.463). F1 lebih diprioritaskan
karena dataset tidak seimbang dan saya ingin model cukup baik di kelas minoritas.

Rincian per kelas pada data out-of-sample (Oktober 2022 - Desember 2023):

| Kelas | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Turun (-1) | 0.311 | 0.188 | 0.234 | 1773 |
| Tahan (0) | 0.504 | 0.746 | 0.602 | 3246 |
| Naik (1) | 0.443 | 0.294 | 0.353 | 2531 |

Akurasi OOS 0.463. Model paling andal mengenali kondisi "tahan" dan masih
lemah membedakan naik/turun. Ini wajar untuk prediksi arah saham harian.

Fitur yang paling berpengaruh (top-10, dari `models/feature_importance.csv`):

| Fitur | Gain | Permutation |
|-------|------|-------------|
| ROLLING_VOL_20 | 0.053 | 0.027 |
| ROLLING_VOL_50 | 0.036 | 0.025 |
| MA_ratio_14 | 0.020 | -0.001 |
| EMA_200 | 0.019 | 0.005 |
| MA_ratio_50 | 0.018 | 0.000 |
| Force_Index | 0.018 | 0.001 |
| MA_200 | 0.018 | 0.001 |
| EMA_30 | 0.017 | 0.000 |
| OBV | 0.017 | 0.004 |
| EMA_21 | 0.017 | 0.000 |

Interpretasi: model paling bergantung pada **volatilitas** (rolling vol 20/50),
lalu rasio moving average dan indikator tren/volume. Dua fitur volatilitas
juga paling kuat di permutation importance, jadi sinyalnya memang benar-benar
digunakan, bukan sekadar artefak. Beberapa fitur teratas berupa level harga
non-stasioner (`EMA_200`, `MA_200`, `OBV`), jadi ada risiko model menghafal
level harga historis; ini menjadi catatan tersendiri di bagian akhir.

**4. Simulasi trading** (`simulate_trading.py`)
Backtest di 20% data terakhir (out-of-sample). Sinyal digeser satu hari supaya
tidak memakai harga hari yang sama. Ada tiga strategi:

1. Long-only vectorized - beli saat `prob_up > 0.5`, bobot sama rata, ambil 5 saham teratas.
2. Long/short vectorized - long saat sinyal naik kuat, short saat sinyal turun kuat.
3. Iterative reinvestment - simulasi baris per baris, modal diinvestasikan ulang
   ke 5 sinyal teratas dengan komisi 0.1%.

Hasil out-of-sample (modal awal 100rb, benchmark S&P 500):

| Strategi | Total Return | CAGR | Sharpe | Max DD |
|----------|-------------|------|--------|--------|
| Long-only | +35.7% | 29.1% | 1.39 | -10.4% |
| Long/short | +38.2% | 31.0% | 1.42 | -10.4% |
| Iterative | +73.1% | 58.2% | 1.78 | -9.6% |
| Benchmark S&P 500 (SPY) | +30.6% | 25.0% | - | - |

Ketiga strategi mengalahkan SPY di periode ini. Sharpe long-only dan long/short
mirip, sedangkan strategi iterative paling tinggi return-nya (tapi volatilitas
juga lebih besar).

**5. Optimasi portofolio** (`portfolio_optimization.py`)
Bobot di-fit hanya dari periode train, lalu dievaluasi di periode hold-out
yang tidak dipakai saat fitting. Jadi angka di bawah ini tidak ada look-ahead.

| Portofolio | Total Return | Sharpe | Max DD |
|-----------|-------------|--------|--------|
| Equal weight | +54.6% | 2.02 | -11.2% |
| Inverse volatility | +46.8% | 1.97 | -9.9% |
| Max Sharpe | +36.8% | 1.18 | -20.8% |
| Min volatility | +14.8% | 1.02 | -11.9% |

Sebelumnya max-Sharpe sempat terlihat jauh lebih bagus, tapi itu karena bobot
dioptimasi dan diuji di periode yang sama (bocor). Setelah diperbaiki jadi
fit-on-train / test-on-holdout, hasilnya turun dan lebih masuk akal.

## Cara menjalankan

Lokal:
```bash
pip install -r requirements.txt
python main.py                 # semua langkah, termasuk unduh data
python main.py --skip-data     # pakai data raw yang sudah ada
python main.py --steps 3 4 5   # pilih langkah tertentu
```

Docker:
```bash
docker build -t stock-ml .
docker run --rm -v "%cd%/data:/app/data" -v "%cd%/models:/app/models" stock-ml
```

Cron (contoh, tiap hari jam 6 pagi):
```cron
0 6 * * * cd /app && python main.py --skip-data --steps 3 4 5 >> logs/pipeline.log 2>&1
```

## Yang belum beres / catatan

- Scaling (StandardScaler) tidak diterapkan; untuk model berbasis tree
  sebenarnya tidak wajib, jadi saya biarkan data dalam skala aslinya.
- LightGBM sempat dicoba tapi gagal di-import karena bentrok `dask` dengan
  pandas 2.2, jadi untuk sementara di-skip lewat optional import.
- `shap` belum terpasang, jadi interpretasi fitur memakai gain + permutation
  importance dari scikit-learn (`models/feature_importance.csv`).
- Beberapa fitur penting berupa level harga non-stasioner (`EMA_200`, `MA_200`,
  `OBV`). Bisa jadi model lebih banyak belajar dari level harga historis
  daripada pola return murni, jadi perlu hati-hati untuk data di luar periode.
- Model masih lemah memprediksi kelas "turun" (recall sekitar 0.19), karena
  kelas ini paling sedikit jumlahnya.
- Dataset masih 37ribu baris. Kalau butuh lebih dari 1 juta baris, tinggal
  tambah jumlah simbol atau rentang tanggal di `data_collector.py`.
