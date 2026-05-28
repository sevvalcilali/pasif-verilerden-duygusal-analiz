# cleaned_data/ — Dosya Kataloğu

Pipeline'ın ürettiği temiz veri ve modelleme tablolarını içerir.
Tüm CSV/JSON dosyaları `main.py` ve modelleme scriptleri tarafından üretilir;
büyük dosyalar `.gitignore` ile dışlanmıştır (kaynak veriden yeniden üretilebilir).

## Modelleme girdileri (feature tabloları)

| Dosya | Satır × Sütun | Boyut | Kullanan model |
|---|---|---|---|
| `features_core_clean.csv` | 34.913 × 30 | 7.4M | Core feature seti (26 feature) |
| `features_extended_clean.csv` | 34.913 × 131 | 28M | Genişletilmiş feature seti (127 feature) |
| `features_clean_labels.csv` | 16.973 × 164 | 20M | `clean_labels.py` (baseline), `personalized_v2.py` (ana model) |
| `features_extended_temporal.csv` | 34.913 × 164 | 42M | `clean_labels.py`, `hybrid_model.py` (lag/rolling feature) |
| `features_hybrid.csv` | 34.913 × 187 | 46M | `hybrid_model.py`, `forecasting_model.py` |
| `features_forecasting.csv` | 2.554 × 193 | 3.5M | `forecasting_model.py` (bugün → yarın) |

## Ara çıktılar (pipeline aşamaları)

| Dosya | Satır × Sütun | Boyut | Üreten modül |
|---|---|---|---|
| `sensing_cleaned.csv` | 216.065 × 646 | 685M | `cleaner.py` + `post_clean.py` |
| `steps_cleaned.csv` | 176.458 × 30 | 22M | `cleaner.py` |
| `ema_cleaned.csv` | 35.561 × 9 | 1.9M | `EMA/src/ema_cleaner.py` |
| `master_dataset.csv` | 34.914 × 675 | 119M | `master_dataset.py` (sensing + EMA) |
| `master_imputed.csv` | 34.914 × 672 | 124M | `imputation.py` (NaN doldurulmuş) |

Bu dosyalar yalnızca feature tablolarını üretmek için gereklidir; eğitilmiş
modellerin çalışması (predict/server) için gerekli değildir.

## Yardımcı dosyalar (JSON)

| Dosya | Boyut | Açıklama |
|---|---|---|
| `cv_splits.json` | 52K | 5-fold StratifiedGroupKFold bölünmeleri + 22 hold-out uid. Anahtarlar: `meta`, `holdout_uids`, `cv_pool_uids`, `folds`. |
| `feature_medyanlar.json` | 8K | Feature popülasyon medyanları. `server.py` çalışma anında eksik feature'ları bununla doldurur. |

## Hedef değişken: `final_risk_4`

`features_*_clean.csv` tablolarındaki hedef sütun. `EMA/src/risk_classifier.py`
tarafından üretilir: `final_risk_4 = max(cascade_risk, klinik_risk)`.

| Değer | Anlam | n | % |
|---|---|---|---|
| 0 | İyi Durum | 14.915 | 42.72 |
| 1 | Hafif Risk | 10.062 | 28.82 |
| 2 | Orta Risk | 7.878 | 22.56 |
| 3 | Yüksek Risk | 2.057 | 5.89 |
