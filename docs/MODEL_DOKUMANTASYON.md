# Modelleme Dokümantasyonu

Her adımda ne yapıldığı, neden yapıldığı, sonucu ve anlamı açıklanır. Sayılar
`reports/` klasöründeki çıktılardan alınmıştır; metodolojik sınırlamalar
(test-eşiği yanlılığı, otokorelasyon, mobil pasif sınırı) açıkça belirtilmiştir.

Sunum formatındaki versiyon: `docs/SUNUM_RAPORU_TAM.md`.

---

# BÖLÜM 0 — Problemin Tanımı

**Soru:** Üniversite öğrencisinin ruhsal risk durumunu (depresyon/anksiyete eğilimi) (a) günlük 7 soruluk EMA anketi ve (b) telefonun pasif sensör verisinden tahmin edebilir miyiz?

**Neden zor:** Ruh hali ↔ telefon davranışı ilişkisi doğası gereği zayıf ve gürültülüdür. Bu, projenin baştan kabul edilen gerçeğidir; düşük skorların kök nedeni de budur.

**Veri:** Dartmouth StudentLife (2017–2022), 220 kullanıcı.

---

# BÖLÜM 1 — VERİ SETİ

| Alt veri | Satır | Sütun | Kullanıcı |
|---|---|---|---|
| sensing | 216.065 | 651 | 220 |
| steps | 176.458 | 30 | 198 |
| calllog | 130.366 | 6 | 35 |
| smslog | 1.177.575 | 7 | 35 |
| unlock | 22.506.500 | 3 | 146 |
| running_apps | 884.102 | 3 | 24 |
| **EMA** | 217.155 | 19 | 220 |

**Neden bu veri seti:** Mental sağlık + pasif sensing literatüründe referans (Wang ve ark. 2014). Gerçek öğrenci, uzun süreli, EMA = etiket kaynağı. Sentetik veri YOK.

---

# BÖLÜM 2 — VERİ TEMİZLEME (neden her adım)

### 2.1 Sensing temizleme
- 651 → **646 sütun**: 5 sütun aşırı eksik (model için gürültü) → atıldı
- Negatif/imkânsız süreler (>86400 sn) → NaN: **Neden:** bir gün 86400 sn'dir, fazlası sensör hatası
- Satır **silinmedi** (216.065 korundu): **Neden:** kullanıcı-gün bütünlüğü; eksikler imputation'a bırakıldı
- Kalan NaN: 12.955.555 (sensör kapalı/izin yok = pasifte normal)

### 2.2 EMA temizleme
- 217.155 → **35.561 satır**: 181.594 satır tamamen boş (o gün anket doldurulmamış = **etiketsiz, kullanılamaz**). Bu veri kaybı değil, gürültü temizliği.
- 19 → 9 sütun: 10 metadata sütunu modelle ilgisiz
- Sonuç: 35.561 temiz EMA, 218 kullanıcı

### 2.3 Eşleştirme
- 220 ortak kullanıcı; (uid, gün) inner join → 216.057 eşleşmiş satır
- **Neden inner join:** hem pasif hem EMA'sı olan günler kullanılabilir; biri yoksa ya girdi ya etiket eksiktir

---

# BÖLÜM 3 — KORELASYON ANALİZİ (neden gerekli)

**Neden:** Yüksek korelasyonlu feature'lar modelde tekrar/kararsızlık yaratır.

- **9 yüksek korelasyon çifti** (örn. `audio_convo_duration ↔ audio_convo_num` r=0.998)
- **4 sıfır-varyanslı feature** (hep aynı değer = bilgi yok)
- **VIF > 10 olan YOK** → ciddi multicollinearity yok

**Karar:** Korelasyonlu fazlalıklar + sıfır-varyans feature engineering'de düşürüldü. VIF temiz olduğu için agresif eleme yapılmadı (ağaç modelleri korelasyona dayanıklıdır).

---

# BÖLÜM 4 — ETİKETLEME (riskin nasıl tanımlandığı)

EMA'dan **kural-tabanlı cascade** ile 4 risk sınıfı + 13 profil:

- **PHQ-4** (Kroenke 2009): anksiyete=q1+q2, depresyon=q3+q4; ≥3 pozitif, ≥5 akut. Kesim noktaları **literatürden** (uydurma değil).
- **PAM** (Pollak 2011 + Russell 1980 circumplex): 16'lık ruh hali → valence/arousal → Q1–Q4
- Hiyerarşik kural: en kritik durumdan (akut depresif + Q4) en sağlıklıya (optimum denge)

**Final risk dağılımı:**

| Sınıf | İsim | n | % |
|---|---|---|---|
| 0 | İyi | 14.916 | 42.7 |
| 1 | Hafif | 10.063 | 28.8 |
| 2 | Orta | 7.878 | 22.6 |
| 3 | **Yüksek** | **2.057** | **5.9** |

**KRİTİK:** Yüksek risk **%5.9**. Bu **şiddetli sınıf dengesizliğidir** ve tüm düşük F1'lerin **kök nedenidir**. Model "herkese İyi" derse %42 doğruluk alır ama hiçbir riskliyi yakalamaz.

> **Not:** Cascade'in *bileşenleri* (PHQ-4/PAM eşikleri, Russell modeli) valide; ama 13 profili birleştiren *kural mantığı* senin tasarladığın bir **risk katmanlama heuristiğidir** — klinik tanı aracı değil, tarama/triaj amaçlı.

---

# BÖLÜM 5 — IMPUTATION (eksik veri doldurma)

| İşlem | Detay | Neden |
|---|---|---|
| Başlangıç NaN | 1.941.708 | Sensör kapalı/izin yok |
| Zero-fill | 199 sütun, 1.913.093 hücre | Sayım feature'larında "yok = 0" doğru (arama yoksa 0) |
| User-median | 440 sütun, 957 hücre | Sürekli feature'da kişinin kendi medyanı en doğru tahmin |
| Final NaN | **0** | Model NaN kabul etmez |

**Neden tek strateji değil:** "Arama sayısı boşsa 0", ama "ışık boşsa kişinin ortalaması". Tek strateji yanlış sinyal verirdi.

---

# BÖLÜM 6 — FEATURE ENGINEERING

672 ara sütundan:
- Leakage drop 28 (EMA'dan sızıntı), hourly drop 480 (saatlik→epoch), nzv 21, redundant 5, light 9
- **Core: 30 feature** (güçlü, yorumlanabilir) · **Extended: 131 feature** (tüm sinyaller)
- **Türetilmiş:** sedanter_saat, aktivite_toplam, mobilite_skoru, gunduz_gece_telefon_orani
- **Zamansal:** her güçlü feature'ın `_lag1` (dün), `_rmean7` (7-gün ort), `_rstd7` (7-gün std) — *bunlar pasif modelin en güçlü sinyalleri olacak (Bölüm 9'da kritik)*

**Neden core/extended ikilisi:** "az ama öz mü, çok ama gürültülü mü?" sorusunu deneysel test etmek için. (Cevap: extended marjinal, fazlası gürültü.)

---

# BÖLÜM 7 — CROSS-VALIDATION KURGUSU (akademik dürüstlüğün temeli)

**Yöntem: StratifiedGroupKFold (5-fold) + ayrı hold-out test.**

| Set | Kullanıcı | Satır |
|---|---|---|
| Hold-out test | 22 | 3.799 |
| Train (fold başına) | ~155-157 | ~25K |
| Validation | ~38-40 | ~6K |
| Dışlanan | 1 | (veri kalitesi yetersiz) |

**Neden GROUP fold:** Aynı kullanıcının günleri hem train hem test'e düşerse model **kişiyi ezberler** → sahte yüksek skor. Kullanıcıyı grup alarak bu engellendi.

> **Önemli sınırlılık (Bölüm 8'de tekrar):** Eşik (threshold) seçimi bazı modellerde **hold-out test üzerinde** yapıldı → raporlanan F1'ler **iyimser yanlı**. Eşikten bağımsız **AUC daha güvenilir** metrik.

---

# BÖLÜM 8 — MODELLER: NE, NEDEN, SONUÇ (sıralı düşünce zinciri)

> Mantık zinciri: baseline kötü → binary'e geç → eşik ayarla → ensemble dene → başarısızları öğren → clean-labels → hibrit (leakage) → forecasting (otokorelasyon). Her adım bir öncekinin **neden**ine cevap.

### 8.1 RF 4-Sınıf Baseline — *neden ilk bu?*
**Neden:** Referans nokta gerek. RF hızlı, yorumlanabilir, class_weight'le dengesizliğe dayanıklı.
**Sonuç:** f1_macro=**0.30**, acc=0.37, sınıf-3 F1=**0.108** (recall %11)
**Yorum:** Riskli kişilerin %89'unu kaçırıyor. 4-sınıf + %5.9 dengesizlik = çıkmaz.

### 8.2 HGB 4-Sınıf + Optuna — *neden daha güçlü model?*
**Neden:** "Belki daha güçlü model + hiperparametre optimizasyonu kurtarır." (XGBoost macOS'ta OpenMP hatası → sklearn HGB, eşdeğer.)
**Sonuç:** f1_macro=**0.29** (RF'den marjinal kötü)
**Yorum:** Optuna bile geçemedi → **problem model değil, problem formülasyonu**. 4-sınıfta sıkışmak hata.

### 8.3 Binary + Threshold Tuning — *neden binary?*
**Neden:** Klinik asıl soru "risk var mı/yok mu". 4-sınıf → binary.
**Sonuç:** Binary RF eşik=0.5 → F1=**0.07** (felaket). RF_ext **F1-opt eşik** → F1=**0.196**
**Yorum:** Dengesiz veride 0.5 eşiği yanlış. Eşik optimize → F1 **3 KAT** (0.069→0.196). **Tek en etkili iyileştirme.** (Ama eşik test'te seçildi → iyimser; aşağıya bak.)

### 8.4 Voting Ensemble V1 — *neden ensemble?*
**Neden:** İki modelin (RF+HGB) hataları farklı; ortalaması daha kararlı.
**Sonuç:** F1=**0.23**, AUC=0.67 — saf pasif + tüm sınıf kategorisinde **lider**
**Yorum:** İşe yaradı çünkü farklı modeller farklı hata yapar.

### 8.5 Lag Features (HGB temporal) — *neden zamansal?*
**Neden:** Ruh hali zamansaldır; dünkü davranış bugünü etkiler.
**Sonuç:** F1=**0.22**, +%11 (önceki adıma göre)
**Yorum:** Zamansal trend gerçek sinyal taşıyor (Bölüm 9'da kritik olacak).

### 8.6 Stacking (RF+HGB→Lojistik meta) — BAŞARISIZ
**Neden denendi:** Meta-öğrenici teoride en iyisi.
**Sonuç:** F1=0.13, **AUC=0.33** (rastgeleden kötü!)
**Yorum:** Meta-learner dengesiz veride base olasılıkları yanlış öğrendi. **Karmaşıklık her zaman iyi değil.** Terk.

### 8.7 Voting V2 — Calibrated (Platt) — BAŞARISIZ
**Neden:** Olasılık kalibrasyonu eşik seçimini iyileştirir (teoride).
**Sonuç:** F1=0.13, AUC=0.41 (recall=1.0 ama precision=0.07 → her şeye "riskli")
**Yorum:** Platt **dengesiz veride** olasılık dağılımını bozdu. Geri alındı.

### 8.8 Voting V3 (3-model) — BAŞARISIZ
**Sonuç:** F1=0.18 — yine V1'in (0.23) altında. **En basit ensemble en iyiymiş.** V1'e dönüldü.

### 8.9 Maximized Features (194) — BAŞARISIZ
**Neden:** "Daha çok feature daha iyi" hipotezi.
**Sonuç:** F1=0.15, **AUC=0.53** (neredeyse rastgele)
**Yorum:** 194 feature aşırı gürültü → AUC 0.70'ten 0.53'e. **Feature sayısı ≠ performans.**

### 8.10 Kişiselleştirilmiş Pasif Model (en iyi saf pasif sonuç)
**Fikir:** Popülasyon modeli "ortalama kişi"yi öğrenir, kişiler-arası varyansta boğulur. Her feature için kullanıcının kendi baseline'ından sapma feature'ları üret.
**Yöntem:** Her feature için 4 paralel görünüm:
- orijinal değer
- `_dev`: kullanıcı medyanından sapma (`x - user_median`)
- `_zsc`: z-score (`(x - user_mean) / user_std`)
- `_rzs`: robust z-score MAD-tabanlı (`(x - user_median) / user_MAD`)
- 159 → 636 feature; HGB k=8 ile eğitildi; EMA modele HİÇ girmedi.

**Sonuç (hold-out, seed=42):** F1=**0.42**, AUC=**0.77**, Recall=0.53, Precision=0.35
**5-seed sanity check (42, 0, 7, 100, 2024):** F1 mean=0.416 std=0.034; AUC mean=0.758 std=0.028 → **robust, lucky-seed değil**
**Baseline'a göre:** F1 +0.06 (+%16), AUC +0.07 (+%11)
**Yorum:** Kişiselleştirme **gerçek bir kazanç** verdi — pasif sinyali popülasyon tavanından (0.36) literatürün üst bandına (0.77 AUC = Saeb 2015 / Xu 2021 seviyesi) taşıdı. EMA modele girmediği için leakage yok. **Projenin gerçek ana bilimsel sonucu bu.**
**Kayıt:** `models/best_personalized.pkl` (636 feature, HGB k=8, threshold 0.22 F1-opt)

### 8.11 CLEAN LABELS — *neden saf pasifte güçlü?*
**Fikir:** Belirsiz orta sınıfları (1,2) **at**; sadece net uçlar: sınıf 0 (kesin iyi) vs 3 (kesin riskli).
**Sonuç:** CLEAN HGB k=8 → F1=**0.36**, AUC=**0.70**; CLEAN RF k=5 F2-opt → recall=**%91**
**Yorum:** F1 0.23→0.36 sıçradı. Neden? Sınıf 1-2 "gri bölge" modeli karıştırıyordu. Net uçlarla gerçek sinyal öğrenildi. **Bu, saf pasifin dürüst akademik sınırıdır (F1≈0.35).**

### 8.12 HİBRİT — EMA+Pasif (183 feature) — *neden mükemmel ama sahte?*
**Sonuç:** F1=**1.00**, AUC=**1.00** (5 fold'da da)
**Yorum:** Çok iyi = şüpheli. **Data leakage:** etiket EMA'dan cascade ile üretiliyor, hibrit modele de EMA giriyor → model target'ı girdiden okuyor. SHAP kanıtı: kararın **%94.7'si EMA** (Bölüm 9). Bu bir hata değil, **eğitici karşı-örnek.**

### 8.13 FORECASTING — *neden bu da yanıltıcı?*
**Kurgu:** Bugün EMA+pasif → YARIN riski (etiket farklı zaman = strict leakage yok).
**Sonuç:** EMA+Pasif F1=**0.97**, AUC=0.99 · **SADECE PASİF F1=0.34-0.37**
**Yorum (en kritik):** Strict leakage yok AMA F1=0.97 **EMA otokorelasyonundan** geliyor: bugün class-3 olan kullanıcı yarın da **%70.5** class-3, class-0→class-0 **%78.2**. EMA çıkarılınca forecasting **0.35'e çöküyor** = cross-sectional pasifle aynı. Yani "pasif yarını tahmin ediyor" YANLIŞ; tahmini yapan, dünkü depresyonun bugünküne benzemesidir (trivial süreklilik). **ANA sonuç değil, eğitici karşı-örnek.**

### Tüm Modeller — Özet Tablo (hold-out)

| # | Model | Girdi | F1 | AUC | Durum |
|---|---|---|---|---|---|
| 1 | RF 4-sınıf | Pasif 26 | 0.30* | — | Baseline, imbalance |
| 2 | HGB Optuna | Pasif 26 | 0.29* | — | RF'den kötü |
| 3 | Binary RF eşik=0.5 | Pasif 26 | 0.07 | 0.66 | Eşiksiz çöp |
| 4 | RF_ext F1-opt | Pasif 127 | 0.20 | 0.67 | Eşik 3× artırdı |
| 5 | Voting V1 | Pasif 159 | 0.23 | 0.67 | Ensemble lideri |
| 6 | HGB temporal | Pasif+Lag 159 | 0.22 | 0.67 | Lag +%11 |
| 7 | Stacking | Pasif 159 | 0.13 | 0.33 | Meta çöktü |
| 8 | Voting V2 calib | Pasif 159 | 0.13 | 0.41 | Kalibrasyon bozdu |
| 9 | Voting V3 | Pasif 159 | 0.18 | 0.62 | V1 altında |
| 10 | Maximized 194 | Pasif 194 | 0.15 | 0.53 | Gürültü |
| 11 | CLEAN HGB k=8 | Pasif 159 | **0.36** | 0.70 | **En iyi SAF PASİF** |
| 12 | CLEAN RF k=5 F2 | Pasif 159 | 0.34 | 0.70 | Recall %91 (triaj) |
| 13 | HİBRİT RF | EMA+Pasif 183 | 1.00 | 1.00 | Data leakage |
| 14 | Forecasting EMA+Pasif | Bugün→Yarın | 0.97 | 0.99 | EMA otokorelasyonu |
| 15 | **Forecasting SADECE Pasif** | Bugün→Yarın | **~0.35** | **~0.70** | **Pasifin gerçek sınırı** |

*f1_macro

---

# BÖLÜM 9 — SHAP (modelin neye baktığının kanıtı)

**Hibrit model — kategori katkısı:**
- EMA türetilmiş: **%68.1** · EMA ham: **%26.6** · Pasif: **%3.5**
→ Hibrit kararının %94.7'si EMA = **data leakage'ın görsel/sayısal kanıtı**

**Pasif model — en güçlü feature'lar:** `sosyal_iletisim_yogunluk_rmean7`, `unlock_num_ep_0_rmean7`, `aktivite_toplam_rmean7`, `gunduz_gece_telefon_orani_rstd7` → **çoğu 7-GÜNLÜK rolling istatistik** (Bölüm 11'de mobil için kritik).

---

# BÖLÜM 10 — Metodolojik Sınırlar

1. **Sınıf dengesizliği %5.9** → tüm düşük F1'lerin kök nedeni
2. **Eşik test'te seçildi** → F1'ler iyimser yanlı; **AUC birincil metrik** olmalı
3. **Küçük test:** Forecasting holdout 133 satır, **sadece 17 pozitif** → precision=1.00 kırılgan
4. **Clean-labels kolaylaştırması:** tüm iyi F1'ler class 0-vs-3 alt-kümesinde; tam 4-sınıf ≈0.30
5. **Forecasting otokorelasyonu:** 0.97 davranış tahmini değil, EMA sürekliliği
6. **Etiket öznel:** EMA self-report; cascade heuristik (tanı değil)

**Bunlar projeyi zayıflatmaz** — bilerek tespit edip raporlamak metodolojik olgunluktur.

---

# BÖLÜM 11 — Ürün: Mobil Pasifin Gerçek Durumu

Telefon ~30 günlük-anlık feature toplar. Modelin EN güçlü pasif feature'ları ise **7-günlük rolling** (Bölüm 9).

**StudentLife holdout simülasyonu (kanıtlı):**

| Senaryo | F1 | Recall | Çıktı dağılımı | Anlam |
|---|---|---|---|---|
| Tek-gün mod (rolling medyanla dolu) | **0.08** | %4 | std 0.10 (~sabit) | **Fiilen bozuk = "herkese aynı"** |
| 7-gün mod (telefon 7 günlük seri yollar, backend rolling'i eğitimle birebir hesaplar) | **0.28** | %47 | std 0.18 (**kişiye özel**) | Gerçek |
| Geçmiş 7 gün UsageStats backfill (anında) | **0.31** | %34 | std 0.13 | En pratik |

**Mimari karar (neden böyle):** Rolling/lag hesabı **Kotlin'de değil backend'de** yapılıyor — eğitimle **birebir aynı** kod (doğrulama: 861/861 birebir eşleşme). Kotlin'de yeniden yazmak train/serve uyumsuzluğu riskiydi.

**Backfill (neden 7 gün beklemeye gerek yok):** Android `UsageStatsManager` ~7-10 günlük geçmişi tutar → telefon kullanım feature'ları açılışta geriye dönük doldurulabilir. Aktivite/konum geçmişi Android'de **yok** → medyan kalır (dürüst donanım sınırı). Kanıt: kullanım kümesi sinyalin çoğunu taşıdığı için backfill F1=0.31 (tam 7-gün ileri toplamadan bile iyi).

**Tavan neden 0.36 değil:** Ses/ışık/arama/SMS telefonla fiziksel olarak toplanamıyor (159 feature'ın 87'si medyan). Bu kusur değil, dürüst sınır.

**Ekran kararı (final_risk):** = SADECE cascade (klinik, EMA). Hibrit ML leakage'lı olduğu için UI'da HİÇ gösterilmiyor; pasif ML + forecasting destekleyici panel. Uç EMA testinde doğru: maks kötü→Yüksek/kırmızı, sağlıklı→İyi/yeşil.

---

# BÖLÜM 12 — "DOĞRULUK ORANI" NEDİR (net cevap)

- **Telefonda ölçülemez** — gerçek kullanıcının klinik tanısı (ground truth) yok
- **Akademik doğruluk = StudentLife hold-out** (22 görülmemiş kullanıcı):
  - **Saf pasif KİŞİSELLEŞTİRİLMİŞ (yeni ana sonuç): F1=0.42, AUC=0.77** (5-seed robust)
  - Saf pasif popülasyon (eski baseline): F1≈0.35, AUC≈0.70
  - Mobil pasif (7-gün/backfill): **F1≈0.28-0.31** (gerçek cihaz koşulu)
  - Hibrit 1.00 / Forecasting 0.97 = EMA kaynaklı, pasif katkısı değil
- **Neden accuracy değil F1/AUC:** %5.9 dengesizlikte "herkese İyi" %42 accuracy alır ama 0 riskli yakalar; F1/AUC bu hileyi yakalar

---

# BÖLÜM 13 — Özet

> *"Pasif telefon sensörüyle ruhsal risk tahmininin popülasyon modeli sınırı F1≈0.35, AUC≈0.70'tir (Wang 2014 literatür tutarlı). **Kişiselleştirme uyguladım** — her feature için kullanıcının kendi medyan/ortalama/varyansından sapma görünümleri eklenerek (EMA modele girmedi) **F1=0.42, AUC=0.77'ye çıkardım**, bu Saeb 2015 / Xu 2021 literatür üst bandı. 5 farklı random seed'de robust (F1 std=0.034). EMA-bazlı yüksek skorlar (hibrit 1.00, forecasting 0.97) leakage ve otokorelasyondan kaynaklanır — bunu pasif-only ablasyonuyla kendim tespit ettim. Mobil pasifin tek-gün modunda işlevsiz olduğunu fark edip 7-günlük train/serve-tutarlı mimariyle düzelttim. Katkım: kişiselleştirme ile pasifi popülasyon tavanından literatür üst bandına taşımak + metodolojik dürüstlük + çalışan uçtan uca sistem."*

---

*Sayılar: `reports/06_modeling_prep/` (70_all_models, 108_MASTER, 113_shap, 115_forecasting_holdout) + 2026-05-18 mobil pasif simülasyonları. Görseller: `SUNUM_RESIMLER/`. Kısa versiyon: `docs/SUNUM_RAPORU_TAM.md`.*
