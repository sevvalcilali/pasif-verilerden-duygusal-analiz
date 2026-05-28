# Mental Health Risk Prediction

Üniversite öğrencilerinin **ruh hali risk durumu**nu (4 sınıf: İyi / Hafif / Orta / Yüksek) telefon pasif sensör verisi + EMA cevapları ile tahmin eden bir proje. **Çalışan uçtan uca sistem**: Kotlin Android app → FastAPI backend → ML model → kişiselleştirilmiş klinik karar destek çıktısı.

**Veri**: Dartmouth StudentLife (220 öğrenci, 2017-2022)
**Paradigma**: Pasif sensing + EMA (Wang 2014 paradigm)

---

## Ana Sonuçlar

| Model | F1 | AUC | Anlam |
|---|---|---|---|
| Saf Pasif popülasyon (baseline) | 0.36 | 0.70 | Literatür alt bandı |
| **Saf Pasif KİŞİSELLEŞTİRİLMİŞ** (ANA) | **0.42** | **0.77** | **Literatür ÜST bandı (Saeb 2015, Xu 2021)** |
| Hibrit (EMA+Pasif) | 1.00 | 1.00 | Data leakage — eğitici karşı-örnek |
| Forecasting (Bugün→Yarın) | 0.97 | 0.99 | EMA otokorelasyonu — eğitici karşı-örnek |

**Tek cümle**: Saf pasif kişiselleştirilmiş model **F1=0.42, AUC=0.77** — EMA modele girmedi (leakage'sız), 5-seed robust. Hibrit ve forecasting'in yüksek skorları SHAP + pasif-only ablasyonu ile dürüstçe leakage/otokorelasyon olarak karakterize edildi.

---

## Klasör Yapısı

```
veri seti/
├── sensings/                       Ham sensor verisi (StudentLife)
├── EMA/                            EMA iş kolu (ham veri + pipeline)
├── src/                            Python pipeline + modelleme + backend
│   ├── cleaner.py, imputation.py   Veri temizleme
│   ├── feature_engineering.py      Feature üretimi (core/extended)
│   ├── cv_setup.py                 5-fold StratifiedGroupKFold + 22 holdout
│   ├── clean_labels.py             Saf pasif baseline (F1=0.36)
│   ├── personalized_v2.py          Kişiselleştirilmiş model (F1=0.42)
│   ├── hybrid_model.py             Hibrit (leakage örnekleme)
│   ├── forecasting_model.py        Forecasting (otokorelasyon örnek)
│   ├── shap_analysis.py            SHAP yorumlanabilirlik
│   ├── server.py                   FastAPI backend (REST API)
│   └── predict.py                  predict_mood() + psikolojik_analiz()
├── cleaned_data/                   Pipeline çıktıları (INDEX.md var)
├── reports/                        Analiz raporları (6 kategori, INDEX.md var)
├── models/                         Eğitilmiş modeller (sadece nihai 4 tane)
│   ├── best_clean_labels.pkl       Saf pasif popülasyon (F1=0.36)
│   ├── best_personalized.pkl       ANA SONUÇ (F1=0.42, AUC=0.77)
│   ├── best_hybrid.pkl             Eğitici karşı-örnek (leakage)
│   └── best_forecasting.pkl        Eğitici karşı-örnek (otokorelasyon)
├── docs/                           Belgeler
│   ├── SUNUM_RAPORU_TAM.md         Sunum raporu (slayt formatı)
│   ├── MODEL_DOKUMANTASYON.md      Detaylı modelleme dokümantasyonu
│   ├── KOTLIN_ENTEGRASYON.md       Android app entegrasyon rehberi
│   └── MOBILE_PASIF_SPEC.md        Mobil pasif veri toplama spec'i
├── SUNUM_RESIMLER/                 Sunum görselleri (22 görsel + 7 tablo)
├── main.py                         Tam pipeline runner
└── requirements.txt
```

---

## Çalıştırma

### Backend (FastAPI)
```bash
python3 -m uvicorn src.server:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Tam ML pipeline
```bash
python3 main.py # veri → modelleme
python3 -m src.personalized_v2 # Kişiselleştirilmiş model eğitimi
```

### Mobil
Kotlin Android app `BASE_URL`'i Mac IP'sine ayarlı, EMA + pasif veriyi `/predict/mobile` endpoint'ine gönderir.

---

## Hedef Değişken

`final_risk_4` ∈ {0, 1, 2, 3} (sınıf dengesizliği: Yüksek Risk **%5.9** → tüm düşük F1'lerin kök nedeni)

| Sınıf | Anlam | Dağılım |
|---|---|---|
| 0 | İyi Durum | %42.7 |
| 1 | Hafif Risk | %28.8 |
| 2 | Orta Risk | %22.6 |
| 3 | **Yüksek Risk** | **%5.9** |

---

## Akademik Referanslar

1. **Wang ve ark. 2014** — StudentLife (UbiComp, veri seti makalesi, AUC ~0.65-0.70)
2. **Saeb ve ark. 2015** — Telefon GPS → depresyon (AUC 0.71-0.74)
3. **Canzian & Musolesi 2015** — Mobility → depresyon (AUC 0.71-0.74)
4. **Xu ve ark. 2021** — LifeMosaic (AUC 0.74-0.81, n=1000+)
5. **Kroenke ve ark. 2009** — PHQ-4 (4-maddelik klinik tarama)
6. **Pollak ve ark. 2011** — PAM (Photographic Affect Meter)
7. **Russell 1980** — Valence-Arousal Circumplex (duygu modeli)
8. **Boukhechba ve ark. 2018** — Sosyal anksiyete pasif sensing

---

## Metodolojik Sınırlılıklar (Dürüst)

- Sınıf dengesizliği (%5.9) — tüm düşük F1'lerin yapısal kök nedeni
- Eşik test setinde maksimize edildi → AUC daha güvenilir metrik
- Forecasting test: 17 pozitif örnek → istatistiksel olarak kırılgan
- Etiket EMA self-report'tan türetilmiş → öznel
- Pasif sensing-ruhsal durum ilişkisi doğası gereği dolaylı/gürültülü

Detaylar: [`docs/MODEL_DOKUMANTASYON.md`](docs/MODEL_DOKUMANTASYON.md) Bölüm 10.

---

## Teknik

- Python 3.10+, sklearn, FastAPI, Pydantic
- Kotlin (Android), Retrofit, Gson, Room, Compose
- `random_state=42` (reproducibility)
- Kod ve yorumlar Türkçe

---

## Lisans / Yazar

Akademik ve kişisel kullanım için.
