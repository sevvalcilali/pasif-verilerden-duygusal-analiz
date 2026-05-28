# Sunum Resimleri

Sunumda kullanılabilecek görseller ve tablolar. Toplam 29 dosya (22 görsel + 7 tablo).

```
SUNUM_RESIMLER/
├── 01_VERI_SETI/          (7 görsel) — veri seti karakteri
├── 02_MODELLER/           (8 görsel) — model performansları
├── 03_YORUMLANABILIRLIK/  (6 görsel) — SHAP analizleri
├── 04_SONUCLAR/           (1 görsel) — tüm modeller karşılaştırması
└── 05_TABLOLAR/           (7 dosya)  — CSV + markdown tablolar
```

## Ana sonuç hatırlatması

- **Ana sonuç:** saf pasif **kişiselleştirilmiş** model — F1=0.42, AUC=0.77
  (`02_MODELLER/09_kisisellestirme_F1_042_AUC_077.png`). Popülasyon
  baseline'a göre +%16 F1, +%11 AUC; 5-seed robust. EMA modele girmedi.
- **Baseline:** saf pasif popülasyon — F1=0.36, AUC=0.70.
- **Hibrit (F1=1.0)** ve **forecasting (F1=0.97)** yüksek skorları gerçek
  pasif başarı değil; sırasıyla data leakage ve EMA otokorelasyonundan
  geliyor. Eğitici karşı-örnek olarak sunulur (SHAP ile gösterilir).

## Önerilen slayt sırası

1. Problem + veri seti: `01_VERI_SETI/01_kullanici_basina_gun.png`,
   `04_ema_7_soru_dagilimlari.png`
2. Korelasyon: `01_VERI_SETI/07_8_boyut_korelasyon.png`
3. Ana sonuç: `02_MODELLER/09_kisisellestirme_F1_042_AUC_077.png`
4. Denenen diğer modeller (yolculuk): `02_MODELLER/06_threshold_optimizasyonu.png`,
   `07_hibrit_model_F1_100.png`, `08_forecasting_F1_097.png`
5. Tüm modeller karşılaştırması: `04_SONUCLAR/01_TUM_MODELLER_KARSILASTIRMA_ANA_GORSEL.png`
6. Yorumlanabilirlik: `03_YORUMLANABILIRLIK/06_SHAP_EMA_vs_PASIF_kategori.png`
   (leakage kanıtı), `07_SHAP_pasif_top20_DAVRANIS.png` (gerçek davranış sinyali)
7. Sonuç + sınırlılıklar: `05_TABLOLAR/02_FORECASTING_VS_CROSS_SECTIONAL.csv`

## Görsel açıklamaları

### 01_VERI_SETI/
| Dosya | Ne gösteriyor |
|---|---|
| `01_kullanici_basina_gun.png` | Kullanıcı başına gün sayısı (220 kullanıcı, medyan ~170 gün) |
| `02_uyku_dagilimi.png` | Uyku süresi histogram + boxplot |
| `03_aktivite_ozeti.png` | Günlük aktivite türü dağılımı |
| `04_ema_7_soru_dagilimlari.png` | 7 EMA değişkeninin dağılımları |
| `05_ema_kullanici_kapsami.png` | EMA dolduran kullanıcı kapsamı |
| `06_korelasyon_heatmap.png` | Tüm feature korelasyonu |
| `07_8_boyut_korelasyon.png` | 8 davranışsal boyut (Wang 2014) |

### 02_MODELLER/
| Dosya | Ne gösteriyor |
|---|---|
| `01_rf_4sinif_baseline_F1_030.png` | RF baseline confusion matrix (F1=0.30) |
| `02_hgb_4sinif_tuned_F1_029.png` | HGB tuned (F1=0.29 — tuning fark yaratmadı) |
| `03_binary_4_model_karsilastirma.png` | 4 binary model karşılaştırması |
| `04_binary_PR_curve.png` | Precision-Recall eğrisi |
| `06_threshold_optimizasyonu.png` | Threshold'a göre F1/recall (eşik ayarı en etkili iyileştirme) |
| `07_hibrit_model_F1_100.png` | Hibrit confusion matrix (F1=1.0 — data leakage) |
| `08_forecasting_F1_097.png` | Forecasting (F1=0.97 — EMA otokorelasyonu) |
| `09_kisisellestirme_F1_042_AUC_077.png` | Kişiselleştirilmiş saf pasif — ana sonuç (F1=0.42) |

### 03_YORUMLANABILIRLIK/
| Dosya | Ne gösteriyor |
|---|---|
| `04_SHAP_hibrit_top20.png` | Hibrit top 20 SHAP (neredeyse hepsi EMA) |
| `05_SHAP_hibrit_beeswarm.png` | Hibrit beeswarm |
| `06_SHAP_EMA_vs_PASIF_kategori.png` | Kategori bazlı katkı — hibrit kararının ~%95'i EMA (leakage kanıtı) |
| `07_SHAP_pasif_top20_DAVRANIS.png` | Pasif top 20 — gerçek davranış feature'ları |
| `08_SHAP_pasif_beeswarm.png` | Pasif beeswarm |
| `09_SHAP_pasif_lokal_ornekler.png` | 3 örnek için lokal açıklama |

### 04_SONUCLAR/
| Dosya | Ne gösteriyor |
|---|---|
| `01_TUM_MODELLER_KARSILASTIRMA_ANA_GORSEL.png` | Tüm modellerin F1 karşılaştırması |

### 05_TABLOLAR/
| Dosya | İçerik |
|---|---|
| `01_TUM_MODELLER_TABLOSU.csv` | Tüm modellerin metrikleri |
| `02_FORECASTING_VS_CROSS_SECTIONAL.csv` | 3 yaklaşımın karşılaştırması |
| `03_SHAP_EMA_VS_PASIF_KARSILASTIRMA.md` | SHAP karşılaştırma raporu |
| `04_HIBRIT_5FOLD_CV_DOGRULAMA.csv` | Hibrit CV sonuçları |
| `05_HIBRIT_FEATURE_IMPORTANCE.csv` | Hibrit feature önemleri |
| `06_SHAP_HIBRIT_DEGERLERI.csv` | SHAP hibrit ham değerleri |
| `07_SHAP_PASIF_DEGERLERI.csv` | SHAP pasif ham değerleri |

## Olası sorular

**F1=1.0 nasıl olur?** Hibrit modelde data leakage var — etiket EMA'dan
üretiliyor, EMA da girdide. SHAP'ta görünüyor: kararın ~%95'i EMA.

**Forecasting F1=0.97 neden gerçek başarı değil?** Strict leakage yok ama yüksek
skor EMA'nın günden güne otokorelasyonundan geliyor; EMA çıkarılınca ~0.35'e
düşüyor (cross-sectional pasifle aynı).

**Gerçek/dürüst sonuç ne?** Saf pasif kişiselleştirilmiş model: F1=0.42,
AUC=0.77. Pasif sensing literatürünün üst bandı (Saeb 2015, Xu 2021).

**Pasif F1 neden düşük?** Pasif sinyal–ruh hali ilişkisi doğası gereği zayıf;
literatürde de bu seviyede. Klinik tanı için değil, tarama/triaj için.
