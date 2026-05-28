# Raporlar Kataloğu

Pipeline'ın ürettiği analiz çıktıları (CSV tablolar + PNG grafikler), aşama
sırasına göre 6 klasörde. Dosyalar numara önekiyle sıralı.

## 01_sensing_eda/ — Sensing genel bakış
Veri seti özeti, describe(), kullanıcı/gün dağılımı, uyku/aktivite/telefon
kullanımı grafikleri. Modül: `src/eda.py`.

## 02_data_quality/ — Eksik veri, aykırı değer, temizleme
Sütun/grup/kullanıcı bazında eksik veri, missing heatmap, IQR aykırı değerler,
temizleme ve post-clean logları. Modüller: `missing_analysis.py`,
`outlier_detector.py`, `cleaner.py`, `post_clean.py`.

## 03_correlation/ — Korelasyon (multicollinearity)
Pearson/Spearman matrisleri, VIF skorları, yüksek korelasyon çiftleri, 8
davranışsal boyut heatmap'i. Bulgu: 4 sıfır-varyanslı feature, VIF>10 yok.
Modül: `correlation_analysis.py`.

## 04_ema_analysis/ — EMA keşif + temizleme
EMA özeti, eksik veri, kullanıcı başına dolu gün, dağılım grafikleri ve
sensing↔EMA uid eşleşmesi (37 — kritik). Modüller: `EMA/src/ema_eda.py`,
`ema_cleaner.py`.

## 05_master_dataset/ — Sensing + EMA birleştirme
Birleştirme logu, 4-sınıf risk dağılımı (40-41), 12 profil dağılımı (cascade
ve manhattan), cascade-manhattan çapraz tablosu (uyum %31.9).
Modül: `master_dataset.py`.

## 06_modeling_prep/ — Modelleme hazırlığı ve model sonuçları

Hazırlık logları (45-51): imputation, feature engineering, drop edilen
sütunlar, data quality, dışlanan kullanıcı, CV fold dağılımı, hold-out özeti.

Model deneyleri özeti: `70_all_models_comparison.csv` tüm denenen modelleri
(RF, HGB, binary, voting, stacking, optuna temporal, maximized) tek tabloda
karşılaştırır. Bu deneylerin tam anlatısı `docs/MODEL_DOKUMANTASYON.md`
Bölüm 8'de; tek tek metrik/grafik dosyaları repoyu sade tutmak için tutulmadı.

Nihai sonuçlar (99-120): clean labels (saf pasif), hybrid (leakage demosu),
SHAP karşılaştırması (109-113), forecasting (114-117), kişiselleştirilmiş
model (118-120, ana sonuç F1=0.42).

## Sunumda öne çıkan raporlar

| Konu | Dosya |
|---|---|
| Veri kapsamı | `01_sensing_eda/01_dataset_overview.csv` |
| Eksik veri yapısı | `02_data_quality/23_nan_coverage.csv` |
| Multicollinearity | `03_correlation/27_high_corr_pairs.csv` |
| EMA ↔ sensing eşleşme | `04_ema_analysis/37_sensing_ema_uid_match.csv` |
| Risk sınıf dağılımı | `05_master_dataset/41_risk_class_distribution.csv` |
| Drop edilen sütunlar (leakage) | `06_modeling_prep/47_dropped_columns.csv` |
| Tüm model karşılaştırması | `06_modeling_prep/108_MASTER_COMPARISON.csv` |
| SHAP (hibrit vs pasif) | `06_modeling_prep/113_shap_comparison.md` |
