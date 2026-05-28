"""
FastAPI backend — mobil app risk tahmin API'si.
Kotlin app POST ile 7 EMA + pasif veri yollar, 4-sınıf risk döner.

Çalıştırma: python3 -m uvicorn src.server:app --host 0.0.0.0 --port 8000
Endpoint listesi ve örnek istekler: http://localhost:8000/docs
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Optional, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# src/predict modülünü import et
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import predict_mood

# FastAPI uygulaması

app = FastAPI(
    title="Ruh Hali Risk Tahmin API",
    description=(
        "Mobil uygulama için REST API.\n\n"
        "Kullanıcının günlük EMA cevapları ve pasif sensör verilerini alıp "
        "**4-sınıflı risk tahmini** döndürür (0=İyi, 1=Hafif, 2=Orta, 3=Yüksek)."
    ),
    version="1.0.0",
)

# CORS — Kotlin app'inden çağrılabilmesi için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Production'da spesifik domain kullan
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic modelleri (request/response şemaları)

class EMAGirdi(BaseModel):
    """7 EMA sorusunun cevapları."""
    stress:       int = Field(..., ge=1, le=5,  description="Stres seviyesi (1-5)")
    pam_score:    int = Field(..., ge=1, le=16, description="PAM 1-16 (Pollak grid)")
    social_level: int = Field(..., ge=1, le=5,  description="Sosyallik (1-5)")
    phq4_q1:      int = Field(..., ge=0, le=3,  description="Gergin/kaygılı (0-3)")
    phq4_q2:      int = Field(..., ge=0, le=3,  description="Endişe kontrolü (0-3)")
    phq4_q3:      int = Field(..., ge=0, le=3,  description="İlgi/zevk kaybı (0-3)")
    phq4_q4:      int = Field(..., ge=0, le=3,  description="Çökkün/depresif (0-3)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "stress": 4, "pam_score": 12, "social_level": 2,
                "phq4_q1": 2, "phq4_q2": 2, "phq4_q3": 3, "phq4_q4": 3,
            }
        }
    }

class TahminIstegi(BaseModel):
    """Mobil uygulamadan gelen tahmin isteği."""
    uid:   str = Field(..., description="Kullanıcı kimliği")
    gun:   str = Field(..., description="Tarih (YYYY-MM-DD veya YYYYMMDD)")
    ema:   EMAGirdi = Field(..., description="7 EMA cevabı")
    pasif: Dict[str, float] = Field(
        default_factory=dict,
        description="Pasif sensör verileri (feature_adi: deger çiftleri). Boş olabilir."
    )
    obj_iletisim: float = Field(
        default=0.0,
        description="Sensing'den gelen toplam iletişim sayısı (call+sms)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "uid": "user_001",
                "gun": "2026-05-15",
                "ema": {
                    "stress": 4, "pam_score": 12, "social_level": 2,
                    "phq4_q1": 2, "phq4_q2": 2, "phq4_q3": 3, "phq4_q4": 3,
                },
                "pasif": {
                    "unlock_num_ep_0": 120.0,
                    "sedanter_saat": 14.0,
                    "aktivite_toplam": 5400.0,
                    "mobilite_skoru": 4500.0,
                },
                "obj_iletisim": 3.0,
            }
        }
    }

class TahminCevabi(BaseModel):
    """Mobil uygulamaya dönülecek tahmin cevabı."""
    kullanici:    str
    gun:          str
    final_risk:   int = Field(..., description="0-3 arası risk sınıfı")
    final_isim:   str
    final_renk:   str
    guvenilirlik: str
    cascade:      dict
    hibrit_ml:    Optional[Dict] = None
    pasif_ml:     Optional[Dict] = None
    top_5_neden:  List[str]
    aciklama:     str
    oneri:        str

    model_config = {
        "json_schema_extra": {
            "example": {
                "kullanici": "user_001",
                "gun": "2026-05-15",
                "final_risk": 3,
                "final_isim": "Yüksek Risk",
                "final_renk": "kırmızı",
                "guvenilirlik": "yüksek (cascade + hibrit + pasif aynı yönde)",
                "cascade": {"risk_sinifi": 3, "profil_id": "J",
                            "profil_isim": "Akut Depresif (Kriz Modu 2)"},
                "top_5_neden": ["PHQ-4 toplam=10 (≥9 şiddetli)", "GAD-2=4 pozitif"],
                "aciklama": "Yüksek risk sinyali — klinik değerlendirme önerilir.",
                "oneri": "Acil destek hattı veya klinik psikolog başvurusu önerilir.",
            }
        }
    }

# Endpoint: Sağlık kontrolü

@app.get("/", tags=["Sistem"])
def root() -> dict:
    """API çalışıyor mu? Basit sağlık kontrolü."""
    return {
        "status":  "ok",
        "service": "Ruh Hali Risk Tahmin API",
        "version": "1.0.0",
        "zaman":   datetime.now().isoformat(),
        "docs":    "/docs",
    }

@app.get("/info", tags=["Sistem"])
def info() -> dict:
    """Yüklü modeller hakkında bilgi."""
    models_dir = os.path.join(BASE_DIR, "models")
    mevcut_modeller = []
    if os.path.isdir(models_dir):
        for fn in os.listdir(models_dir):
            if fn.endswith(".pkl"):
                path = os.path.join(models_dir, fn)
                mevcut_modeller.append({
                    "dosya": fn,
                    "boyut_kb": round(os.path.getsize(path) / 1024, 1),
                })
    return {
        "modeller":       mevcut_modeller,
        "ana_model":      "best_hybrid.pkl (predict_mood içinde)",
        "ek_model":       "best_clean_labels.pkl (pasif-only cross-check)",
        "predict_mood":   "src/predict.py",
        "akademik_referans": "Wang 2014, Kroenke 2009, Pollak 2011",
    }

# Endpoint: Ana tahmin

@app.post("/predict", response_model=TahminCevabi, tags=["Tahmin"])
def tahmin_yap(istek: TahminIstegi) -> dict:
    """
    Ana tahmin endpoint'i.

    Kullanıcının 7 EMA cevabı ve pasif sensör verisini alır,
    3 paralel sistemden (cascade + hibrit ML + pasif ML) tahmin üretir,
    birleşik risk değerlendirmesini döndürür.

    **Mobil app kullanımı:**
    - Kotlin Retrofit ile POST request
    - Cevap JSON olarak parse edilir
    - `final_risk` ve `final_isim` UI'da gösterilir
    - `top_5_neden` kullanıcıya açıklama olarak gösterilir
    """
    try:
        sonuc = predict_mood(
            uid=istek.uid,
            gun=istek.gun,
            ema=istek.ema.model_dump(),
            pasif=istek.pasif if istek.pasif else None,
            obj_iletisim=istek.obj_iletisim,
        )
        return sonuc
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Eksik EMA alanı: {e}. 7 madde de gerekli."
        )
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Geçersiz değer: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin sırasında hata: {type(e).__name__}: {e}"
        )

# Endpoint: Sadece EMA ile hızlı tahmin

class HizliIstek(BaseModel):
    """Sadece EMA — pasif veri olmadan hızlı tahmin."""
    uid: str = "anonim"
    gun: str = ""
    ema: EMAGirdi

@app.post("/predict/quick", tags=["Tahmin"])
def hizli_tahmin(istek: HizliIstek) -> dict:
    """
    Hızlı tahmin — sadece EMA cevapları (pasif veri olmadan).

    Bu endpoint cascade'ı çalıştırır (kural-tabanlı).
    Pasif veri vermediği için ML model çağrılmaz.

    Kotlin app demosunda kullanılabilir:
    - Sensor izinleri henüz alınmamışsa
    - Hızlı test için
    """
    if not istek.gun:
        istek.gun = datetime.now().strftime("%Y-%m-%d")

    sonuc = predict_mood(
        uid=istek.uid,
        gun=istek.gun,
        ema=istek.ema.model_dump(),
        pasif=None,
    )
    return sonuc

# Endpoint: Mobil — kısmi pasif veri + medyan doldurma

import json as _json

# Popülasyon medyanlarını yükle (eksik feature doldurma için)
_MEDYAN_PATH = os.path.join(BASE_DIR, "cleaned_data", "feature_medyanlar.json")
try:
    with open(_MEDYAN_PATH) as _f:
        _FEATURE_MEDYANLAR = _json.load(_f)
except FileNotFoundError:
    _FEATURE_MEDYANLAR = {}

# Telefonun toplayabileceği pasif feature'lar (whitelist)
_TELEFON_PASIF_FEATURES = {
    "unlock_num_ep_0", "unlock_num_ep_1", "unlock_num_ep_2", "unlock_num_ep_3",
    "unlock_duration_ep_0", "unlock_duration_ep_1", "unlock_duration_ep_2", "unlock_duration_ep_3",
    "act_still_ep_0", "act_still_ep_1", "act_still_ep_2", "act_still_ep_3",
    "act_walking_ep_0", "act_walking_ep_1", "act_walking_ep_2", "act_walking_ep_3",
    "act_on_foot_ep_0", "act_on_foot_ep_1", "act_on_foot_ep_2", "act_on_foot_ep_3",
    "act_running_ep_0", "act_on_bike_ep_0", "act_in_vehicle_ep_0",
    "loc_dist_ep_0", "loc_dist_ep_1", "loc_dist_ep_2", "loc_dist_ep_3",
    "loc_visit_num_ep_0",
    "other_playing_duration_ep_0", "other_playing_num_ep_0",
}

class MobilIstek(BaseModel):
    """
    Mobil uygulamadan gelen istek.
    Pasif veride sadece TELEFONUN TOPLAYABİLDİĞİ feature'lar gelir.
    Eksik feature'lar backend tarafından popülasyon medyanıyla doldurulur.
    """
    uid: str = Field(..., description="Kullanıcı kimliği")
    gun: str = Field(default="", description="Tarih (boşsa bugün)")
    ema: EMAGirdi = Field(..., description="7 EMA cevabı")
    pasif: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Telefonun topladığı pasif feature'lar (kısmi olabilir). "
            "Sadece toplanabilen gönderilir; eksikler backend'de medyanla doldurulur."
        ),
    )
    pasif_gunluk: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Son ~7 günün GÜNLÜK epoch feature map listesi. Her eleman: "
            "{'gun': 'YYYY-MM-DD', '<feature>': değer, ...}. Backend bundan "
            "lag1/rmean7/rstd7/türetilmiş feature'ları EĞİTİMLE AYNI fonksiyonla "
            "hesaplar (train/serve tutarlılığı). Boşsa tek-gün moduna düşer."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "uid": "telefon_user_001",
                "gun": "2026-05-16",
                "ema": {
                    "stress": 4, "pam_score": 12, "social_level": 2,
                    "phq4_q1": 2, "phq4_q2": 2, "phq4_q3": 3, "phq4_q4": 3,
                },
                "pasif": {
                    "unlock_num_ep_0": 145.0,
                    "unlock_num_ep_1": 12.0,
                    "unlock_num_ep_2": 80.0,
                    "unlock_num_ep_3": 53.0,
                    "act_still_ep_0": 64800.0,
                    "act_walking_ep_0": 3200.0,
                    "loc_dist_ep_0": 8500.0,
                },
            }
        }
    }

def _mobil_zaman_serisi_feature(
    gunluk: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Telefonun son ~7 günlük GÜNLÜK epoch verisinden, EĞİTİMDE KULLANILAN
    ASIL fonksiyonlarla (add_behavioral_features + temporal_features_olustur)
    lag1/rmean7/rstd7/türetilmiş feature'ları hesaplar ve EN SON günün
    (bugünün) feature vektörünü döndürür.

    Train/serve tutarlılığı: rolling/lag mantığı Kotlin'de YENİDEN YAZILMAZ;
    burada eğitimin birebir aynı kodu çağrılır.

    Yetersiz gün (rmean7 min_periods=3) → ilgili feature NaN → atlanır →
    çağıran tarafta popülasyon medyanıyla dolar (graceful degradation).
    """
    import pandas as pd
    from src.feature_engineering import add_behavioral_features
    from src.config import LAG_FEATURES, ROLLING_STD_FEATURES

    rows = []
    for g in gunluk:
        r: Dict[str, Any] = {
            k: float(v)
            for k, v in g.items()
            if k in _TELEFON_PASIF_FEATURES
        }
        r["uid"] = "_mobil_"
        r["gun"] = str(g.get("gun", ""))
        rows.append(r)

    if not rows:
        return {}

    df = pd.DataFrame(rows).sort_values("gun").reset_index(drop=True)

    # 1) Davranışsal türetme: eğitimin ASIL fonksiyonu (sedanter_saat,
    #    aktivite_toplam, mobilite_skoru, gunduz_gece_telefon_orani, ...)
    df = add_behavioral_features(df)

    # 2) lag1/rmean7/rstd7: eğitimle BİREBİR AYNI pandas işlemleri.
    #    Feature tanımları burada inline replike edilir; train/serve
    #    tutarlılığı için aynı window=7, min_periods=3, ddof kullanılır.
    for kol in LAG_FEATURES:
        if kol in df.columns:
            df[f"{kol}_lag1"] = df.groupby("uid")[kol].shift(1)
    for kol in LAG_FEATURES:
        if kol in df.columns:
            df[f"{kol}_rmean7"] = (
                df.groupby("uid")[kol]
                  .rolling(window=7, min_periods=3)
                  .mean()
                  .reset_index(level=0, drop=True)
            )
    for kol in ROLLING_STD_FEATURES:
        if kol in df.columns:
            df[f"{kol}_rstd7"] = (
                df.groupby("uid")[kol]
                  .rolling(window=7, min_periods=3)
                  .std()
                  .reset_index(level=0, drop=True)
            )

    # 3) dow / is_weekend (model bunları da kullanıyor). Eğitimde gun
    #    %Y%m%d idi; mobilde YYYY-MM-DD. Haftanın günü her iki formatta
    #    da aynı → esnek parse.
    dt = pd.to_datetime(df["gun"].astype(str), errors="coerce")
    df["dow"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype("Int64")

    # 4) Hedef gün = en son satır. NaN feature'lar (yetersiz pencere)
    #    dahil edilmez → çağıran tarafta popülasyon medyanıyla dolar.
    son = df.sort_values("gun").iloc[-1]
    cikti: Dict[str, float] = {}
    for c in df.columns:
        if c in ("uid", "gun"):
            continue
        v = son[c]
        if pd.notna(v):
            cikti[c] = float(v)
    return cikti

@app.post("/predict/mobile", tags=["Tahmin"])
def mobil_tahmin(istek: MobilIstek) -> dict:
    """
    MOBİL endpoint — kısmi pasif veri kabul eder.

    Kotlin app şunu yapar:
    1. 7 EMA sorusu doldurulur
    2. Telefonun TOPLAYABİLDİĞİ pasif feature'lar gönderilir
       (~30 feature; izin/sensör yoksa eksik gelebilir)
    3. Backend EKSİK feature'ları popülasyon medyanıyla doldurur
    4. Hibrit model (EMA + tam pasif) çalışır
    5. Risk + açıklama döner

    Eksik pasif feature stratejisi:
    - Kotlin sadece toplayabildiğini gönderir (null/0 GÖNDERME)
    - Backend feature_medyanlar.json'dan eksikleri tamamlar
    - Bilinmeyen/whitelist dışı feature'lar yok sayılır
    """
    if not istek.gun:
        istek.gun = datetime.now().strftime("%Y-%m-%d")

    # 1. Pasif veri kaynağı: 7-günlük mod (tercih edilen) veya tek-gün (fallback)
    if istek.pasif_gunluk:
        # 7-GÜNLÜK MOD: eğitimle AYNI fonksiyonla lag1/rmean7/rstd7/türetilmiş
        # hesaplanır → pasif feature'lar GERÇEK ve kişiye özel olur.
        gelen_pasif = _mobil_zaman_serisi_feature(istek.pasif_gunluk)
        reddedilen = []
        kullanilan_mod = "7-gunluk"
        gun_sayisi = len(istek.pasif_gunluk)
    else:
        # TEK-GÜN MOD (geriye uyumlu): sadece whitelist ham feature.
        # rolling/lag feature'lar medyanla dolar → pasif zayıf/sabit.
        gelen_pasif = {
            k: float(v) for k, v in istek.pasif.items()
            if k in _TELEFON_PASIF_FEATURES
        }
        reddedilen = [k for k in istek.pasif if k not in _TELEFON_PASIF_FEATURES]
        kullanilan_mod = "tek-gun"
        gun_sayisi = 1 if istek.pasif else 0

    # 2. predict_mood için tam pasif feature seti oluştur
    #    (Tüm pasif feature'lar: gelen + medyan ile doldurulan)
    tam_pasif: Dict[str, float] = {}
    medyan_doldurulan = 0
    for fname, medyan_deger in _FEATURE_MEDYANLAR.items():
        # EMA feature'larını atla (onlar ema dict'inden gelir)
        if fname in (
            "stress", "pam_score", "social_level",
            "phq4_q1", "phq4_q2", "phq4_q3", "phq4_q4",
            "stress_z", "pam_valence", "pam_arousal",
            "phq4_anksiyete", "phq4_depresyon", "phq4_total", "phq4_risk",
            "gad2_pozitif", "phq2_pozitif",
            "social_subj_norm", "social_obj_norm", "social_delta", "obj_iletisim",
            "pam_q_Q1", "pam_q_Q2", "pam_q_Q3", "pam_q_Q4",
        ):
            continue
        if fname in gelen_pasif:
            tam_pasif[fname] = gelen_pasif[fname]
        else:
            tam_pasif[fname] = float(medyan_deger)
            medyan_doldurulan += 1

    # 3. predict_mood çağır (tam pasif ile → hibrit model çalışır)
    try:
        sonuc = predict_mood(
            uid=istek.uid,
            gun=istek.gun,
            ema=istek.ema.model_dump(),
            pasif=tam_pasif,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin hatası: {type(e).__name__}: {e}"
        )

    # 4. Şeffaflık bilgisi ekle (kullanıcı/Kotlin görsün)
    sonuc["mobil_meta"] = {
        "telefondan_gelen_feature": len(gelen_pasif),
        "medyanla_doldurulan_feature": medyan_doldurulan,
        "toplam_pasif_feature": len(tam_pasif),
        "reddedilen_bilinmeyen_feature": reddedilen,
        "gelen_feature_listesi": sorted(gelen_pasif.keys()),
        "mod": kullanilan_mod,
        "pasif_gun_sayisi": gun_sayisi,
    }
    return sonuc

# Yerel çalıştırma

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  Ruh Hali Risk Tahmin API")
    print("=" * 60)
    print("\n  Sunucu başlatılıyor...")
    print("  Yerel:       http://localhost:8000")
    print("  Docs (UI):   http://localhost:8000/docs")
    print("  Test:        curl http://localhost:8000/")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
