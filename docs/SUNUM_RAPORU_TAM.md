# BİTİRME PROJESİ — TAM SUNUM RAPORU
### Pasif Telefon Sensörü + EMA ile Üniversite Öğrencilerinde Ruhsal Risk Tahmini

> Bu belge projenin **baştan sona** tüm adımlarını, kullanılan **her algoritmayı**, neyin **işe yaradığını/yaramadığını** ve **neden** olduğunu sebep-sonuç ilişkisiyle anlatır. Sunumda doğrudan kullanılabilir.

---

## 0. ÖZET (Sunumun İlk Slaytı İçin)

| | |
|---|---|
| **Problem** | Üniversite öğrencilerinin ruhsal risk durumunu (depresyon/anksiyete) telefon davranışından ve günlük EMA anketinden tahmin etmek |
| **Veri** | Dartmouth StudentLife (2017–2022), 220 kullanıcı, 217K+ satır |
| **ANA SONUÇ (kişiselleştirilmiş pasif)** | **F1=0.42, AUC=0.77** — kişi-bazlı çoklu-bakış (orig+median dev+z-score+robust z-score) ile saf pasif, 5-seed sanity check'te robust (F1 std=0.03) |
| **Baseline (popülasyon pasif)** | F1=0.36, AUC=0.70 — eski clean-labels saf pasif, **+0.06 F1 ve +0.07 AUC kazanç** |
| **Eğitici karşı-örnek** | Hibrit F1=**1.00** = data leakage; Forecasting F1=**0.97** = EMA günlük otokorelasyonu (pasif katkısı DEĞİL — pasif-only forecasting F1≈0.35'e düşer) |
| **Ürün** | Kotlin Android app → pasif veri + EMA → FastAPI + ML → ekranda renkli risk kartı (uçtan uca çalışıyor) |

**Özet:** *"Saf pasif telefon verisiyle ruhsal risk tahmininin popülasyon modeli sınırı F1≈0.35, AUC≈0.70'tir (literatür tutarlı). Kişiselleştirme ile (her kullanıcının medyan/ortalama/varyansından sapma feature'ları eklendi, EMA modele girmedi) bu **F1=0.42, AUC=0.77'ye çıktı** — pasif sensing literatür bandının üst kısmı (Wang 2014, Saeb 2015, Xu 2021). EMA eklenince çıkan yüksek skorlar (hibrit 1.00, forecasting 0.97) pasif katkısı değil, sırasıyla data leakage ve EMA otokorelasyonudur; bunu pasif-only ablasyon + SHAP ile gösterdim. Katkım: kişiselleştirilmiş pasif ML modeli + çalışan uçtan uca sistem + leakage'ın metodolojik tespiti."*

> **Metrik uyarısı (her slaytta geçerli):** Tüm F1 değerleri (i) eşiği hold-out test üzerinde maksimize ederek seçtiği için iyimser yanlıdır — eşikten bağımsız **AUC daha güvenilir**; (ii) sadece "clean-labels" (net class 0 vs 3) alt-kümesinde, küçük test setinde hesaplanmıştır (forecasting holdout: 133 satır, **yalnız 17 pozitif**). Tam 4-sınıf problemi çok daha zordur (f1_macro≈0.30).

---

## 1. VERİ SETİ — Ne Kullandık, Neden

**Kaynak:** Dartmouth College StudentLife — pasif telefon sensörü + EMA anketleri.

| Alt veri | Satır | Sütun | Kullanıcı | Ne içerir |
|---|---|---|---|---|
| sensing | 216.065 | 651 | 220 | Aktivite, konum, ses, ışık, telefon kullanımı, uyku |
| steps | 176.458 | 30 | 198 | Adım sayısı |
| calllog | 130.366 | 6 | 35 | Arama kayıtları |
| smslog | 1.177.575 | 7 | 35 | SMS kayıtları |
| unlock | 22.506.500 | 3 | 146 | Ekran açma/kapama event'leri |
| running_apps | 884.102 | 3 | 24 | Açık uygulamalar |
| **EMA** | 217.155 | 19 | 220 | Günlük 7 soru: stres, ruh hali (PAM), sosyallik, PHQ-4 |

**Neden bu veri seti:** Mental sağlık + pasif sensing literatüründe (Wang 2014, StudentLife) referans veri seti. Gerçek öğrencilerden, uzun süreli (5 yıl), etiketli (EMA = ground truth). **Sentetik veri kullanılmadı** — tamamı gerçek StudentLife verisi.

---

## 2. VERİ TEMİZLEME — Adım Adım, Sebep-Sonuç

### 2.1 Sensing Temizleme (`src/cleaner.py`, `src/post_clean.py`)

| İşlem | Önce | Sonra | **Neden** |
|---|---|---|---|
| Tamamen boş sütun | 651 | 651 | Yoktu (0 kaldırıldı) |
| Yüksek-eksik sütun (%>eşik) | 651 | **646** | 5 sütun çok fazla NaN içeriyordu → model için gürültü, kaldırıldı |
| Negatif/overflow → NaN | — | 0 | Süre değerleri 86400s'i aşamaz; mantıksızları NaN'a çevirdik |
| Uyku sanity | — | 0 | Fizyolojik olmayan uyku değerleri kontrol edildi |

**Sonuç:** 216.065 satır korundu (satır silmedik — kullanıcı-gün bütünlüğü için), kalan NaN = 12.955.555 (sonra imputation ile dolduruldu).
**Sebep:** Sensing'de eksik veri *normaldir* (sensör kapalı, izin yok). Satır silmek yerine sütun bazında temizleyip imputation'a bıraktık → veri kaybı minimum.

### 2.2 EMA Temizleme (`src/cleaner.py`)

| İşlem | Önce | Sonra | **Neden** |
|---|---|---|---|
| Boş satır silme | 217.155 | **35.561** | 181.594 satır tamamen boştu (kullanıcı o gün anketi doldurmamış) → etiketsiz, kullanılamaz |
| Gereksiz sütun | 19 | **9** | 10 sütun (id, metadata) modelle ilgisiz |
| Aralık-dışı → NaN | — | 0 | PHQ-4 (0-3), stres (1-5) vb. aralık dışı yok |
| Kalan NaN | — | 1.653 | Kısmi doldurulmuş anketler (sonra ele alındı) |

**Sonuç:** 35.561 temiz EMA satırı, 218 kullanıcı.
**Sebep-sonuç:** %83.6 EMA satırı boştu → bu satırlarda **etiket yok**, modelin öğreneceği bir şey yok. Bunları atmak veri kaybı değil, **gürültü temizliğidir.**

### 2.3 Sensing ↔ EMA Eşleştirme (`src/master_dataset.py`)

- 220 sensing uid ∩ 220 EMA uid = **220 ortak kullanıcı**
- Tarih örtüşmesi: sensing 2017-09-07→2022-06-15, EMA 2017-09-07→2022-07-04
- (uid, gün) inner join → **216.057 eşleşmiş satır**

**Neden inner join:** Sadece *hem pasif hem EMA'sı olan* gün-kullanıcı çiftlerini kullanabiliriz; biri eksikse o gün için ne girdi ne etiket vardır.

---

## 3. KORELASYON ANALİZİ — Multicollinearity (`src/correlation_analysis.py`)

**Neden yaptık:** Yüksek korelasyonlu feature'lar modelde gereksiz tekrar ve kararsızlık yaratır (özellikle lineer modellerde). Feature engineering öncesi temizlik için.

**Bulgular:**
- **9 yüksek korelasyon çifti** tespit edildi (örn. `audio_convo_duration ↔ audio_convo_num` r=0.998; `sms_in_num ↔ sms_out_num` r=0.94)
- **4 sıfır-varyanslı feature** (hep aynı değer → bilgi yok)
- **VIF > 10 olan feature YOK** → ciddi multicollinearity yok, ağaç-tabanlı modeller zaten dayanıklı

**Sonuç → karar:** Korelasyonlu çiftlerden biri ve sıfır-varyanslı feature'lar feature engineering'de düşürüldü. VIF temiz olduğu için agresif feature eleme yapmadık (ağaç modelleri multicollinearity'ye dayanıklı).

---

## 4. ETİKETLEME — Risk Sınıfı Nasıl Üretildi (`src/master_dataset.py`)

EMA'dan **kural-tabanlı cascade** ile 4 risk sınıfı + 13 klinik profil üretildi.

**Cascade mantığı (literatür-temelli):**
- **PHQ-4** (Kroenke 2009): anksiyete (q1+q2), depresyon (q3+q4); ≥3 pozitif, ≥5 akut
- **PAM** (Pollak 2011): 16'lık ruh hali → valence/arousal → 4 quadrant (Q1 Coşkulu, Q2 Sakin, Q3 Anksiyete, Q4 Depresif)
- Stres + sosyallik ile birleşik hiyerarşik kural

**Final risk dağılımı (büyük problem burada görünüyor):**

| Sınıf | İsim | n | % |
|---|---|---|---|
| 0 | İyi Durum | 14.916 | %42.7 |
| 1 | Hafif Risk | 10.063 | %28.8 |
| 2 | Orta Risk | 7.878 | %22.6 |
| 3 | **Yüksek Risk** | **2.057** | **%5.9** |

**13 profil** (A–L): en sık `C Normal Akademik Stres` %18.7, `F Hafif Depresif` %18.2; en nadir `J Akut Depresif` sadece **144 satır (%0.4)**.

**KRİTİK SEBEP-SONUÇ:** Yüksek risk sınıfı verinin sadece **%5.9'u**. Bu **şiddetli sınıf dengesizliğidir** ve projenin ana zorluğudur — tüm düşük F1 skorlarının kök nedeni budur. Model "herkese İyi Durum de" derse %42 doğruluk alır ama hiçbir riskli kişiyi yakalayamaz.

---

## 5. IMPUTATION — Eksik Veri Doldurma (`src/imputation.py`)

| İşlem | Detay | **Neden** |
|---|---|---|
| Başlangıç NaN | 1.941.708 | Sensör kapalı/izin yok |
| `has_light` flag | 6.542 var / 28.372 yok | Eksikliğin kendisi bilgi → flag ile koru |
| Zero-fill | 199 sütun, 1.913.093 hücre | Sayım/süre feature'larında "yok = 0" mantıklı (örn. arama yoksa 0) |
| User-median fill | 440 sütun, 957 hücre | Sürekli feature'larda kişinin kendi medyanı en doğru tahmin |
| **Final NaN** | **0** | Model NaN kabul etmez |

**Sebep-sonuç:** Farklı feature tiplerine farklı strateji uyguladık. "Arama sayısı boşsa 0'dır" ama "ışık seviyesi boşsa kişinin ortalaması". Tek strateji (örn. hep 0) yanlış sinyal verirdi.

---

## 6. FEATURE ENGINEERING (`src/feature_engineering.py`)

672 ara sütundan iki feature seti üretildi:

| Düşürülen | Sayı | **Neden** |
|---|---|---|
| Leakage drop | 28 | EMA'dan türeyen, target'a sızıntı yapan feature'lar |
| Light drop | 9 | Korelasyon analizinde elenenler |
| Redundant drop | 5 | Yüksek korelasyon çiftlerinden fazlalık |
| Hourly drop | 480 | Saatlik ham sütunlar → epoch'a (ep_0/1/2/3) toplandı, ham gereksiz |
| Near-zero-variance | 21 | Bilgi taşımayan feature'lar |

**İki set:**
- **Core (30 feature):** En güçlü, yorumlanabilir feature'lar
- **Extended (131 feature):** Tüm davranışsal sinyaller

**Sebep:** Core/Extended ikilisi ile "az ama öz mü, çok ama gürültülü mü daha iyi?" sorusunu deneysel test ettik (cevap aşağıda — extended marjinal, fazlası gürültü).

---

## 7. CROSS-VALIDATION KURGUSU (`src/cv_setup.py`)

**Yöntem: StratifiedGroupKFold (5-fold) + ayrı hold-out test.**

| Set | Kullanıcı | Satır | **Neden bu kurgu** |
|---|---|---|---|
| Hold-out test | 22 | 3.799 | Hiç görülmeyen kullanıcılar → gerçek genelleme testi |
| Train (fold başına) | ~155-157 | ~25K | |
| Validation | ~38-40 | ~6K | Eşik/hiperparametre seçimi |
| Dışlanan | 1 | — | `ad15fc...` — veri kalitesi yetersiz |

**Neden GROUP fold:** Aynı kullanıcının günleri hem train hem test'e düşerse model kişiyi ezberler → sahte yüksek skor. Kullanıcıyı **grup** alarak bunu engelledik. Bu, akademik dürüstlüğün temelidir.

---

## 8. MODELLER — Her Biri: Ne, Neden, Sonuç, Neden İşe Yaradı/Yaramadı

> Not: "F1" pozitif sınıf (yüksek risk) F1'i, aksi belirtilmedikçe. Hold-out test sonuçları.

### 8.1 Random Forest — 4-Sınıf Baseline
- **Sonuç:** f1_macro=**0.3022**, accuracy=0.3743, sınıf-3 F1=**0.108**
- **Neden yaptık:** İlk referans nokta (baseline).
- **Neden zayıf:** Sınıf dengesizliği. Sınıf 3 recall sadece %11 → riskli kişilerin %89'unu kaçırıyor. **Çıkarım:** 4-sınıf + dengesizlik = çıkmaz sokak.

### 8.2 HistGradientBoosting — 4-Sınıf + Optuna
- **Sonuç:** f1_macro=**0.2934**, accuracy=0.3464
- **Neden yaptık:** XGBoost macOS'ta OpenMP hatası verdi → sklearn HGB (eşdeğer gradient boosting) + Optuna ile hiperparametre optimizasyonu.
- **Neden işe yaramadı:** Optuna bile RF'yi geçemedi (marjinal daha kötü). **Çıkarım:** Problem algoritma değil, problem kurgusu (4-sınıf + imbalance). Daha güçlü model kurtarmıyor.

### 8.3 Binary Dönüşüm + Threshold Tuning
- **Problem yeniden tanımı:** 4-sınıf → binary (riskli mi / değil mi). **Neden:** Klinik olarak da asıl soru "risk var mı?".
- **Binary RF default (eşik=0.5):** F1=**0.0693** — felaket.
- **RF_ext F1-opt (eşik ayarlı):** F1=**0.1964**, recall=0.50, AUC=0.6697
- **SEBEP-SONUÇ:** Eşik 0.5 dengesiz veride yanlış. Eşiği F1'e göre optimize edince F1 **3 KAT** arttı (0.069 → 0.196). **En önemli tek iyileştirme buydu.**

### 8.4 Voting Ensemble V1
- **Sonuç:** F1=**0.2298**, AUC=0.6671 (RF+HGB soft voting, eşit ağırlık, kalibrasyon yok)
- **Neden yaptık:** İki modelin hatalarını dengelemek.
- **Sonuç:** Saf pasif + tüm sınıflar kategorisinde **lider** oldu. İşe yaradı çünkü RF ve HGB farklı hatalar yapıyor, ortalaması daha kararlı.

### 8.5 Lag Features — HGB Temporal k=3
- **Sonuç:** F1=**0.2173**, AUC=0.6672 (geçmiş günlerin değerleri feature olarak eklendi)
- **Neden yaptık:** Ruh hali zamansal — dünkü davranış bugünü etkiler.
- **Sonuç:** Önceki adıma göre +%11 iyileşme. Lag features davranışsal trendi yakalıyor.

### 8.6 Stacking (RF+HGB → Lojistik Regresyon meta) — BAŞARISIZ
- **Sonuç:** F1=0.1327, AUC=**0.334** (rastgeleden kötü!)
- **Neden denedik:** Meta-öğrenici teorik olarak en iyisi.
- **Neden ÇÖKTÜ:** Meta-learner dengesiz veride base modellerin olasılıklarını yanlış öğrendi, AUC 0.33'e düştü. **Çıkarım:** Karmaşıklık her zaman iyi değil. Terk edildi.

### 8.7 Voting V2 — Calibrated (Platt) — BAŞARISIZ
- **Sonuç:** F1=0.1327, AUC=**0.4072**
- **Neden denedik:** Olasılık kalibrasyonu eşik seçimini iyileştirir (teoride).
- **Neden ÇÖKTÜ:** Platt calibration **dengesiz veride** olasılık dağılımını bozdu (recall=1.0 ama precision=0.07 → her şeye "riskli" diyor). **Çıkarım:** Kalibrasyon imbalanced veride tehlikeli. Geri alındı.

### 8.8 Voting V3 — 3-model, CV-threshold — BAŞARISIZ
- **Sonuç:** F1=0.139–0.176, AUC=0.6230
- **Neden denedik:** V2'nin kalibrasyon hatası olmadan 3 model.
- **Sonuç:** Yine basit Voting V1'in (0.23) altında. **Çıkarım:** En basit ensemble en iyisiydi. V1'e geri dönüldü.

### 8.9 Maximized Features (194 feature) — BAŞARISIZ
- **Sonuç:** F1=0.145, AUC=**0.527** (neredeyse rastgele)
- **Neden denedik:** "Daha çok feature daha iyi" hipotezi.
- **Neden ÇÖKTÜ:** 194 feature aşırı gürültü kattı, AUC 0.70'ten 0.53'e düştü. **Çıkarım:** Feature sayısı ≠ performans. Az ve öz daha iyi (core/extended kararını doğruladı).

### 8.10 CLEAN LABELS — Saf Pasif En İyi (`src/clean_labels.py`)
- **Fikir:** Belirsiz orta sınıfları (1, 2) **at**, sadece net uçları kullan: Sınıf 0 (kesin iyi) vs Sınıf 3 (kesin riskli).
- **CLEAN HGB k=8 F1-opt:** F1=**0.3599**, AUC=**0.6976**
- **CLEAN RF k=5 F2-opt:** F1=0.3358, **recall=0.9074** (klinik triaj)
- **SEBEP-SONUÇ:** F1 0.23 → **0.36** sıçradı. Neden? Sınıf 1 ve 2 "gri bölge" — etiketleri belirsiz, modeli karıştırıyordu. Sadece net uçlarla eğitince model gerçek sinyali öğrendi. **Bu, saf pasif verinin akademik sınırıdır (F1≈0.36).**

### 8.11 HİBRİT — EMA + Pasif (183 feature) (`src/hybrid_model.py`)
- **Sonuç:** F1=**1.0000**, precision=1.00, recall=1.00, AUC=**1.0000** — 5 fold'da da mükemmel
- **Neden denedik:** EMA + pasif birleşince performans artar mı?
- **KRİTİK BULGU — DATA LEAKAGE:** F1=1.00 *çok iyi olduğu için şüpheli*. Sebep: risk etiketi **EMA'dan cascade ile üretiliyor**, hibrit modele de **EMA feature'ları giriyor** → model target'ı girdiden "okuyor". `phq4_total` zaten target'ı belirliyor.
- **Bu bir hata değil, eğitici bir bulgu:** Sunumda data leakage'ın ne olduğunu **canlı kanıtla** gösteriyoruz (bkz. SHAP).

### 8.12 FORECASTING — Yüksek Skor ama YANILTICI (`src/forecasting_model.py`)
- **Kurgu:** **Bugünün** EMA+pasif verisi → **YARININ** riskini tahmin et.
- **EMA+Pasif sonuç:** F1=0.9697, precision=1.00, recall=0.9412, AUC=0.9904
- **SADECE PASİF sonuç (aynı kurgu, EMA çıkarılmış):** **F1≈0.34–0.37, AUC≈0.69–0.71**
- **Veri:** holdout 133 satır, **sadece 17 yarın-pozitif**, 21 kullanıcı (clean-labels)
- **NEDEN YANILTICI (dürüst):** Strict data leakage YOK (yarının etiketi yarının EMA'sından gelir, girdiye girmez). AMA F1=0.97 **pasif sensörden değil, EMA'nın günden güne otokorelasyonundan** gelir: bugün class-3 olan kullanıcı yarın da %70.5 class-3, bugün class-0 olan yarın %78.2 class-0. EMA çıkarılınca forecasting **F1≈0.35'e çöküyor** = cross-sectional pasifle aynı. Yani "pasif yarınki riski tahmin ediyor" iddiası YANLIŞ; tahmini yapan, dünkü depresyon skorunun bugünküne benzemesidir (trivial süreklilik).
- **Ek uyarılar:** Eşik hold-out test üzerinde maksimize edilmiş (iyimser yanlı); n_pozitif=17 → precision=1.00 istatistiksel olarak kırılgan (16/17).
- **Sunumdaki rolü:** ANA sonuç DEĞİL. Hibrit (leakage) ile birlikte **eğitici karşı-örnek**: "yüksek skor her zaman başarı değildir; pasif-only ablasyonu gerçeği gösterir → F1≈0.35."

### Tüm Modeller — Tek Tablo (Sunum İçin)

| # | Model | Girdi | F1 | AUC | Durum |
|---|---|---|---|---|---|
| 1 | RF 4-sınıf | Pasif 26 | 0.302* | — | Baseline, imbalance sorunu |
| 2 | HGB 4-sınıf Optuna | Pasif 26 | 0.293* | — | RF'den kötü |
| 3 | Binary RF eşik=0.5 | Pasif 26 | 0.069 | 0.656 | Eşiksiz çöp |
| 4 | RF_ext F1-opt | Pasif 127 | 0.196 | 0.670 | Eşik tuning 3× artırdı |
| 5 | Voting V1 | Pasif 159 | 0.230 | 0.667 | Ensemble lideri |
| 6 | HGB temporal (lag) | Pasif+Lag 159 | 0.217 | 0.667 | Lag +%11 |
| 7 | Stacking | Pasif 159 | 0.133 | 0.334 | Meta-learner çöktü |
| 8 | Voting V2 calib | Pasif 159 | 0.133 | 0.407 | Kalibrasyon bozdu |
| 9 | Voting V3 | Pasif 159 | 0.176 | 0.623 | V1'in altında |
| 10 | Maximized 194 | Pasif 194 | 0.145 | 0.527 | Fazla feature gürültü |
| 11 | CLEAN HGB k=8 (popülasyon) | Pasif 159 | 0.360 | 0.698 | Saf pasif tavanı (eski) |
| **11b** | **Personalized MEGA HGB** | **Pasif 636 (4 view)** | **0.42** | **0.77** | **YENİ ANA SONUÇ — saf pasif + kişiselleştirme, 5-seed robust** |
| 12 | CLEAN RF k=5 F2 | Pasif 159 | 0.336 | 0.697 | Klinik recall %91 |
| 13 | HİBRİT RF | EMA+Pasif 183 | 1.000 | 1.000 | Data leakage (eğitici karşı-örnek) |
| 14 | Forecasting (EMA+Pasif) | Bugün→Yarın 183 | 0.970 | 0.990 | EMA otokorelasyonu, pasif katkısı değil |
| 15 | **Forecasting (SADECE PASİF)** | Bugün→Yarın ~159 | **~0.35** | **~0.70** | **Pasifin yarın-tahmin gerçek sınırı** |
| | **ANA DÜRÜST SONUÇ** | Pasif | **≈0.35** | **≈0.70** | **Saf pasif sınırı — tüm dürüst pasif sonuçlar burada birleşiyor** |

*f1_macro

---

## 9. SHAP — Yorumlanabilirlik & Leakage'ın GÖRSEL Kanıtı (`src/shap_analysis.py`)

**Hibrit Model SHAP — kategori bazında katkı:**

| Kategori | SHAP Katkısı | % |
|---|---|---|
| EMA türetilmiş | 0.2514 | **%68.1** |
| EMA ham | 0.0984 | **%26.6** |
| **Pasif** | 0.0128 | **%3.5** |

→ **Hibrit model kararının %94.7'sini EMA'dan veriyor, pasif sadece %3.5.** Bu, data leakage'ın **sayısal ve görsel kanıtıdır.** Model cascade kuralını ezberlemiş.

**Pasif Model SHAP — en önemli feature'lar:** sosyal iletişim yoğunluğu, yürüme (`act_on_foot`), toplam aktivite, telefon açma sayısı → **gerçek davranışsal sinyaller**, Wang 2014 paradigmasıyla uyumlu. Pasif model F1'i düşük ama **gerçekten öğreniyor**.

**Sunum mesajı:** "İki SHAP grafiğini yan yana koyuyoruz: hibrit EMA'yı ezberliyor (%94.7 EMA), pasif gerçek davranış öğreniyor. Bu yüzden forecasting'e geçtik."

---

## 10. ÜRÜN — Backend + Mobil Entegrasyon

### Backend (`src/server.py`, `src/predict.py`)
- **FastAPI** REST API. `predict_mood()` = **3 paralel sistem:**
  1. **Cascade** (kural-tabanlı, klinik validasyonlu) — ANA KARAR
  2. **Hibrit ML** — destekleyici sinyal
  3. **Pasif ML** — destekleyici sinyal
- Endpoint: `POST /predict/mobile` — iki mod:
  - **tek-gün (eski):** sadece o günün ~30 ham feature'ı; rolling/lag medyanla dolar
  - **7-günlük (yeni):** telefon son ~7 günün günlük epoch verisini gönderir, backend **eğitimle BİREBİR aynı fonksiyonla** lag1/rmean7/rstd7/türetilmiş feature'ları hesaplar (train/serve tutarlılığı %100 doğrulandı, 861/861 birebir)
- **2026-05-16 tasarım kararı:** `final_risk = SADECE cascade sınıfı`. Sebep: hibrit modelin leakage'ı + agresif eşik (0.11) hafif vakaları "Yüksek Risk" gösteriyordu. ML çıktıları ekranda **destekleyici** olarak kalır, final kararı ezmez. Test: hafif EMA → "Hafif Risk/sarı" , ağır EMA → "Yüksek Risk/kırmızı" .
- **2026-05-18 dürüst bulgu — mobil pasifin gerçek durumu:** Modelin en güçlü pasif feature'ları 7-günlük rolling istatistikler. Tek-gün modunda bunlar medyanla dolduğu için mobil pasif **fiilen bozuktu** (StudentLife holdout simülasyonu: F1=**0.08**, recall %4, çıktı ~sabit = "herkese aynı"). 7-günlük mod ile telefon kendi geçmişinden rolling hesaplattığında **F1=0.28, recall %47, çıktı kişiye özel** (std 0.10→0.18). Teorik tavan 0.36 değil çünkü ses/ışık/arama/SMS telefonla toplanamıyor (159 feature'ın 87'si). Bu **dürüst bir donanım sınırı**, kusur değil — ve mobil pasifin ancak ~7 gün veri biriktikten sonra anlamlı olduğu açıkça raporlanıyor.

### Mobil (Kotlin Android)
- Kullanıcı 7 EMA sorusu doldurur + app pasif veri toplar:
  - **UsageStats** → telefon açma/süre
  - **ActivityRecognition** → STILL/WALKING/RUNNING/BIKE/VEHICLE süreleri
  - **FusedLocation** → konum/mesafe
- Ham veri saatlik toplanır → **epoch'a bölünür** (ep_0 tüm gün, ep_1 00-09, ep_2 09-18, ep_3 18-24, SUM aggregation)
- EMA + pasif → `/predict/mobile` → ekranda **renkli risk kartı + top 5 neden + öneri**
- Offline için Room veritabanına kaydedilir.

**Akış:** Telefon → pasif+EMA → FastAPI → ML → ekranda risk. **Uçtan uca çalışıyor.**

---

## 11. Kişiselleştirme — Saf Pasifte Gerçek Sıçrama

| Sonuç | F1 | AUC | Akademik Anlam |
|---|---|---|---|
| **① Saf Pasif popülasyon** (eski baseline) | 0.36 | 0.70 | Popülasyon modeli sınırı — literatür alt-bandı |
| ** ② Saf Pasif KİŞİSELLEŞTİRİLMİŞ** (YENİ ANA) | **0.42** | **0.77** | **Çoklu-bakış kişiselleştirme; EMA modele girmedi; 5-seed robust (std 0.03)** |
| **③ Hibrit Cross-sectional** | 1.00 | 1.00 | Data leakage (etiket EMA'dan, EMA da girdide) — eğitici karşı-örnek |
| **④ Forecasting EMA+Pasif** | 0.97 | 0.99 | EMA günlük otokorelasyonu — pasif katkısı DEĞİL |
| **⑤ Forecasting SADECE Pasif** | ≈0.35 | ≈0.70 | ③④'ün maskesi düşünce kalan, popülasyon seviyesi |

**Sıçrama nasıl oldu (kişiselleştirme = ana katkı):**
- Her feature için 4 paralel görünüm üretildi: orijinal değer, kullanıcı medyanından sapma (`dev`), z-score (`zsc` = (x−µ)/σ), robust z-score (`rzs` = (x−medyan)/MAD)
- Model her bakıştan farklı sinyal çekti → 159 → 636 feature
- F1 +0.06, AUC +0.07 — 5 farklı random seed'de tutarlı (std F1=0.034, std AUC=0.028)
- EMA modele HİÇ girmedi → leakage yok, dürüst
- **Projenin gerçek bilimsel katkısı bu**

**Anlatı:** "②③'teki yüksek skorlar cazip ama yanıltıcı. Pasif-only ablasyonu (①④) gerçeği gösteriyor: F1≈0.35. **Katkım yüksek bir skor değil; bu yanılgıyı kendim tespit edip dürüstçe karakterize etmem** + çalışan uçtan uca sistem." Bu farkındalık, projeyi 'model eğittim' projesinden metodolojik olgunluğu olan bir çalışmaya dönüştürür.

---

## 12. Sınırlılıklar

1. **Sınıf dengesizliği:** Yüksek risk %5.9 → tüm düşük F1'lerin kök nedeni.
2. **Saf pasif sınır:** F1≈0.35 — pasif veri tek başına klinik tanı için yetersiz, **tarama/triaj** aracı olabilir.
3. **Eşik test setinde seçildi:** `egit_test()` eşiği hold-out test üzerinde F1-maksimize ediyor → raporlanan F1'ler **iyimser yanlı**. Eşikten bağımsız **AUC daha güvenilir** metrik. (Düzeltme: eşik validation'da seçilmeli — bilinçli sınırlılık olarak raporlanıyor.)
4. **Küçük test seti:** Forecasting holdout'ta sadece **17 pozitif** → precision=1.00 istatistiksel olarak kırılgan, geniş güven aralığı.
5. **Clean-labels kolaylaştırması:** Tüm "iyi" F1'ler sadece class 0-vs-3 alt-kümesinde; belirsiz orta sınıflar (1,2) atılmış. Tam 4-sınıf problemi çok daha zor (f1_macro≈0.30).
6. **Forecasting otokorelasyonu:** F1=0.97 davranışsal tahmin değil, EMA'nın günden güne sürekliliğidir; pasif-only düşünce F1≈0.35.
7. **Etiket öznel:** EMA self-report; cascade kural-tabanlı (klinik tanı değil).
8. **Cihaz çeşitliliği:** StudentLife eski Android; toplanacak veri farklı cihaz/sürüm.

---

## 13. Sık Sorulan Sorular

**S: F1=1.00 nasıl olur, overfitting değil mi?**
C: Overfitting değil, **data leakage**. Etiket EMA'dan üretiliyor, hibrit modele de EMA giriyor. SHAP'ta görünüyor: kararın %94.7'si EMA. Bunu eğitici karşı-örnek olarak sunuyorum.

**S: Forecasting F1=0.97 leakage yoksa neden gerçek başarı değil?**
C: Strict leakage yok ama F1=0.97 **EMA'nın günden güne otokorelasyonundan** geliyor — bugün class-3 olan kullanıcı yarın da %70.5 class-3. EMA'yı çıkarıp pasif-only forecasting yaptığımda F1 **0.35'e düşüyor**. Yani tahmini yapan pasif sensör değil, dünkü depresyon skorunun bugünküne benzemesi. Bunu pasif-only ablasyonuyla **kendim gösterdim** — yüksek skoru olduğu gibi sunmadım.

**S: O zaman gerçek/dürüst sonucun ne?**
C: Popülasyon saf pasif F1=0.36, AUC=0.70 (literatür alt bandı). **Kişiselleştirme uyguladıktan sonra** (her kullanıcının medyan/ortalama/varyansından sapma feature'ları — EMA modele girmedi) **F1=0.42, AUC=0.77** elde ettim — pasif sensing literatürünün üst bandı (Saeb 2015 AUC 0.71-0.74, Xu 2021 AUC 0.74-0.81). 5 farklı random seed'de robust (F1 std=0.034). Katkım: kişiselleştirme ile pasif sinyali popülasyon modelinin ötesine taşımak + leakage'ın metodolojik tespiti + çalışan sistem.

**S: Kişiselleştirmeyi nasıl yaptın?**
C: Her feature için 4 paralel görünüm: orijinal değer + kullanıcı medyanından sapma + z-score + robust z-score (MAD-tabanlı). 159 → 636 feature. Model her bakıştan farklı sinyal çekti. EMA modele HİÇ girmedi (leakage'sız). 5-seed sanity check ile gain gerçek olduğu doğrulandı.

**S: F1 değerlerine neden tam güvenmemeliyim?**
C: Haklısınız — eşik hold-out test üzerinde F1-maksimize edilerek seçildi, iyimser yanlı. Bu yüzden **eşikten bağımsız AUC'yi** birincil metrik alıyorum ve bunu sınırlılık olarak açıkça belirtiyorum. Ayrıca forecasting test setinde sadece 17 pozitif var → precision=1.00 kırılgan.

**S: Neden bu kadar model denediniz?**
C: Sistematik metodoloji: baseline → threshold → ensemble → temporal → clean labels. Başarısızlar da (stacking, calibration, maximized) **bilimsel değerli** — neyin neden çalışmadığını gösteriyor.

**S: Gerçek hayatta kullanılır mı?**
C: Klinik tanı için değil, **erken tarama/triaj** için, sınırlılıklarıyla. Pasif sinyal zayıf; sistem EMA + pasifi birleştiren bir farkındalık aracı, tanı koymaz.

---

## 14. SAYISAL ÖZET (Son Slayt)

| Metrik | Değer |
|---|---|
| Toplam kullanıcı | 220 |
| Temiz EMA satırı | 35.561 |
| Eğitilen model | 14+ |
| Başarısız ama öğretici deney | 4 (stacking, calib, V3, maximized) |
| **ANA SONUÇ — Saf pasif KİŞİSELLEŞTİRİLMİŞ** | **F1=0.42, AUC=0.77** (5-seed robust) |
| Saf pasif popülasyon (eski baseline) | F1=0.36, AUC=0.70 |
| Hibrit (leakage, eğitici) | F1=1.00 |
| Forecasting EMA+Pasif (otokorelasyon) | F1=0.97 / pasif-only düşünce **F1≈0.35** |
| Forecasting test pozitif sayısı | sadece 17 (precision=1.00 kırılgan) |
| Metodolojik not | Eşik test'te seçildi → AUC birincil metrik |
| Ürün | Kotlin app + FastAPI + ML (uçtan uca çalışıyor) |

---

*Bu rapor `reports/` klasöründeki gerçek CSV/PNG çıktılarından üretilmiştir. Görseller: `SUNUM_RESIMLER/` klasörü.*
