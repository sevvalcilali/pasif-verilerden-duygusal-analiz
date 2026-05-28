# SHAP Karşılaştırma — Hibrit vs Pasif

## Hibrit Model (F1=1.00) — Top 10 SHAP Feature

| Sıra | Feature | Kategori | SHAP |
|---|---|---|---|
| 1 | `phq4_total` | EMA_turetilen | 0.0517 |
| 2 | `phq4_risk` | EMA_turetilen | 0.0482 |
| 3 | `phq4_anksiyete` | EMA_turetilen | 0.0439 |
| 4 | `gad2_pozitif` | EMA_turetilen | 0.0401 |
| 5 | `phq4_depresyon` | EMA_turetilen | 0.0348 |
| 6 | `phq4_q2` | EMA_ham | 0.0245 |
| 7 | `phq4_q4` | EMA_ham | 0.0244 |
| 8 | `phq2_pozitif` | EMA_turetilen | 0.0209 |
| 9 | `phq4_q3` | EMA_ham | 0.0184 |
| 10 | `phq4_q1` | EMA_ham | 0.0169 |

## Pasif Model (F1=0.36) — Top 10 SHAP Feature

| Sıra | Feature | Kategori | SHAP |
|---|---|---|---|
| 1 | `sosyal_iletisim_yogunluk_rmean7` | Pasif | 0.1763 |
| 2 | `act_on_foot_ep_0` | Pasif | 0.1655 |
| 3 | `aktivite_toplam_rmean7` | Pasif | 0.1252 |
| 4 | `unlock_num_ep_0_rmean7` | Pasif | 0.1242 |
| 5 | `unlock_duration_ep_0_rmean7` | Pasif | 0.1192 |
| 6 | `audio_amp_mean_ep_0_rmean7` | Pasif | 0.1183 |
| 7 | `audio_amp_mean_ep_1` | Pasif | 0.0940 |
| 8 | `loc_max_dis_from_campus_ep_0_lag1` | Pasif | 0.0912 |
| 9 | `gunduz_gece_telefon_orani_rstd7` | Pasif | 0.0909 |
| 10 | `act_on_foot_ep_2` | Pasif | 0.0894 |

## Sonuçlar

**Hibrit Modelde EMA Dominantı:**
- EMA_turetilen: 0.2514 (68.1%)
- EMA_ham: 0.0984 (26.6%)
- Pasif: 0.0128 (3.5%)
- EMA_pam_onehot: 0.0068 (1.8%)

**Pasif Modelde Önemli Davranışsal Feature'lar:**
- En önemli pasif feature'lar gerçek davranış sinyalleri (telefon kullanımı, hareketlilik, vb.)
- Wang 2014 paradigmasıyla uyumlu

**Akademik Yorum:**
- Hibrit model EMA cevaplarını dominant kullanıyor → cascade'i ezberlemiş gibi
- Pasif model ise gerçek davranışsal pattern öğreniyor (F1 daha düşük ama "öğrenme" var)
- İki model birlikte: zengin tartışma için ideal
