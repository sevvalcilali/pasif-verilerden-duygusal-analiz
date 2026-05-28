"""
Feature Engineering Modülü — Modelleme İçin Hazır Feature Setleri
Bağımsız çalışır:

    python -m src.feature_engineering

Akış (sıralı):
  1. Leakage kolonlarını drop et (27 EMA-türetilmiş + ham EMA + risk etiketleri)
  2. Light sütunlarını drop et (is_ios platform sinyalini taşıyor)
  3. Korelasyon analizinden gelen redundant sütunları drop et
  4. Hourly (_hr_X) sütunları drop et (ep_X özetleri zaten var)
  5. Near-zero-variance sütunları drop et (%95+ aynı değer)
  6. Seyrek lokasyon sütunlarını tek bir özete birleştir
  7. Wang 2014 paradigm davranışsal feature'lar türet
  8. İki alternatif çıktı:
       - features_core.csv     (~30 sütun: ep_0 + türetilmiş + meta)
       - features_extended.csv (~80 sütun: core + ep_1/2/3 + IS/IV)

Çıktılar:
  cleaned_data/features_core.csv      — ana modelleme tablosu
  cleaned_data/features_extended.csv  — kapsamlı modelleme tablosu
  reports/46_feature_engineering_log.csv — yapılan işlem özeti
  reports/47_dropped_columns.csv      — drop edilen tüm sütunların listesi
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.config import PATHS

# Yollar

INPUT_PATH      = os.path.join(PATHS["cleaned_data"], "master_imputed.csv")
CORE_OUT        = os.path.join(PATHS["cleaned_data"], "features_core.csv")
EXTENDED_OUT    = os.path.join(PATHS["cleaned_data"], "features_extended.csv")
REPORTS         = PATHS["reports"]

# Sabitler — drop listeleri

# Leakage: hedef değişkeni doğrudan üreten / EMA bilgisi içeren tüm sütunlar
LEAKAGE_KOLONLAR = [
    # Ham EMA (input olarak verilirse data leakage)
    "stress", "pam_score", "social_level",
    "phq4_q1", "phq4_q2", "phq4_q3", "phq4_q4",
    # Türetilmiş EMA — hedefi belirleyen ara skorlar
    "stress_z", "pam_valence", "pam_arousal", "pam_quadrant",
    "phq4_anksiyete", "phq4_depresyon", "phq4_total", "phq4_risk",
    "gad2_pozitif", "phq2_pozitif",
    "social_subj_norm", "social_obj_norm", "social_delta",
    # Risk etiketleri (hedefin kendisi veya türetmesi)
    "profil_id", "profil_isim", "profil_risk_4", "profil_guven",
    "profil_id_manhattan", "profil_risk_4_manhattan",
    "klinik_risk_4",
    # obj_iletisim sensing'den türetildi ama EMA tarafına bağlı
    "obj_iletisim",
]

# Korelasyon analizinden gelen redundant feature'lar (|r| >= 0.85)
REDUNDANT_KOLONLAR = [
    "audio_convo_num_ep_0",     # audio_convo_duration ile r=0.998
    "audio_voice_ep_0",         # audio_convo_duration ile r=0.984
    "call_in_num_ep_0",         # call_in_duration ile r=0.96
    "call_out_num_ep_0",        # call_out_duration ile r=0.93
    "audio_amp_std_ep_0",       # audio_amp_mean ile r=0.95
]

# Hedef + meta — modele girmeyecek ama tabloda kalmalı
META_KOLONLAR = ["uid", "gun", "is_ios"]
HEDEF_KOLON   = "final_risk_4"

# Near-zero-variance eşiği
NZV_ESIK = 0.95   # bir sütunda en yaygın değerin oranı bu eşikten büyükse drop

# 1. Leakage drop

def drop_leakage(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Hedef değişkeni belirleyen tüm EMA-bağlantılı sütunları drop et."""
    print("\n── 1. Leakage kolonları drop ediliyor ──")
    var_olan = [c for c in LEAKAGE_KOLONLAR if c in df.columns]
    df = df.drop(columns=var_olan)
    print(f"  Drop edilen: {len(var_olan)} sütun")
    return df, var_olan

# 2. Light + has_light_sensor drop

def drop_light_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Light_* sütunlarını ve has_light_sensor flag'ini drop et.
    is_ios zaten platform sinyalini taşıyor; %81 iOS verisinde light_* hep 0.
    """
    print("\n── 2. Light sütunları drop ediliyor (iOS dominantı nedeniyle) ──")
    light_kollar = [c for c in df.columns
                    if c.startswith("light_") or c == "has_light_sensor"]
    df = df.drop(columns=light_kollar)
    print(f"  Drop edilen: {len(light_kollar)} sütun")
    return df, light_kollar

# 3. Korelasyon redundancy drop

def drop_redundant(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Korelasyon analizinden gelen yüksek korelasyonlu çiftlerden birini at."""
    print("\n── 3. Redundant (yüksek korelasyonlu) sütunlar drop ediliyor ──")
    var_olan = [c for c in REDUNDANT_KOLONLAR if c in df.columns]
    df = df.drop(columns=var_olan)
    print(f"  Drop edilen: {len(var_olan)} sütun")
    return df, var_olan

# 4. Hourly drop

def drop_hourly_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    _hr_X (saatlik) sütunları drop et.
    ep_1 (00-09), ep_2 (09-18), ep_3 (18-24) zaten epoch özetleri sağlıyor.
    """
    print("\n── 4. Hourly (_hr_X) sütunları drop ediliyor ──")
    hr_kollar = [c for c in df.columns if "_hr_" in c]
    df = df.drop(columns=hr_kollar)
    print(f"  Drop edilen: {len(hr_kollar)} sütun (ep_1/2/3 özetleri yeterli)")
    return df, hr_kollar

# 5. Near-zero-variance drop

def drop_near_zero_variance(df: pd.DataFrame, esik: float = NZV_ESIK) -> tuple[pd.DataFrame, list[str]]:
    """
    Bir sütunda en yaygın değerin oranı esik üzerindeyse drop et.
    Modele ayırıcı sinyal vermiyor.
    """
    print(f"\n── 5. Near-zero-variance drop (en yaygın değer > %{esik*100:.0f}) ──")
    sayisal = df.select_dtypes(include="number").columns.tolist()
    sayisal = [c for c in sayisal if c not in META_KOLONLAR + [HEDEF_KOLON]]

    drop_kollar = []
    for kol in sayisal:
        en_yaygin_oran = df[kol].value_counts(normalize=True, dropna=False).iloc[0]
        if en_yaygin_oran > esik:
            drop_kollar.append(kol)

    df = df.drop(columns=drop_kollar)
    print(f"  Drop edilen: {len(drop_kollar)} sütun")
    if drop_kollar[:10]:
        print(f"    İlk 10: {drop_kollar[:10]}")
    return df, drop_kollar

# 6. Seyrek lokasyonları birleştir

def merge_minor_locations(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Seyrek lokasyon sütunlarını (workout/health/worship/leisure) tek bir
    minor_location_dur özeti olarak topla.
    """
    print("\n── 6. Seyrek lokasyonlar birleştiriliyor ──")
    minor_kollar = [c for c in df.columns
                    if c.startswith(("loc_workout_", "loc_health_",
                                     "loc_worship_", "loc_leisure_"))]
    if not minor_kollar:
        return df, []

    df["loc_minor_locations_total"] = df[minor_kollar].sum(axis=1)
    df = df.drop(columns=minor_kollar)
    print(f"  Birleştirilen: {len(minor_kollar)} sütun → 1 (loc_minor_locations_total)")
    return df, minor_kollar

# 7. Wang 2014 davranışsal feature türetme

def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wang 2014 ve Jacobson 2022 paradigm'ından gelen davranışsal feature'lar.
    Hepsi ep_0 (full_day) bazlı.
    """
    print("\n── 7. Davranışsal feature'lar türetiliyor ──")
    eklenen = []

    # 1. Sedanter saat (hareketsizlik saat cinsinden)
    if "act_still_ep_0" in df.columns:
        df["sedanter_saat"] = df["act_still_ep_0"] / 3600.0
        eklenen.append("sedanter_saat")

    # 2. Gece telefon yoğunluğu (00-09 saat aralığı)
    if "unlock_num_ep_1" in df.columns:
        df["gece_telefon_yogunluk"] = df["unlock_num_ep_1"]
        eklenen.append("gece_telefon_yogunluk")

    # 3. Sosyal iletişim yoğunluğu (sensing tarafından, EMA'ya bakmadan)
    iletisim_kollar = [c for c in ["call_in_duration_ep_0", "call_out_duration_ep_0",
                                    "sms_in_num_ep_0", "sms_out_num_ep_0"]
                       if c in df.columns]
    if iletisim_kollar:
        df["sosyal_iletisim_yogunluk"] = df[iletisim_kollar].sum(axis=1)
        eklenen.append("sosyal_iletisim_yogunluk")

    # 4. Aktivite toplamı
    aktivite_kollar = [c for c in ["act_walking_ep_0", "act_running_ep_0",
                                    "act_on_foot_ep_0", "act_on_bike_ep_0"]
                       if c in df.columns]
    if aktivite_kollar:
        df["aktivite_toplam"] = df[aktivite_kollar].sum(axis=1)
        eklenen.append("aktivite_toplam")

    # 5. Mobilite skoru
    if "loc_dist_ep_0" in df.columns:
        df["mobilite_skoru"] = df["loc_dist_ep_0"]
        eklenen.append("mobilite_skoru")

    # 6. Gündüz-gece telefon kullanım oranı (sirkadiyen ipucu)
    if all(c in df.columns for c in ["unlock_num_ep_1", "unlock_num_ep_2", "unlock_num_ep_3"]):
        gunduz = df["unlock_num_ep_2"] + df["unlock_num_ep_3"]
        gece   = df["unlock_num_ep_1"]
        df["gunduz_gece_telefon_orani"] = (gunduz + 1) / (gece + 1)   # +1 smoothing
        eklenen.append("gunduz_gece_telefon_orani")

    print(f"  Eklenen: {len(eklenen)} davranışsal feature")
    for e in eklenen:
        print(f"    + {e}")
    return df

# 8. Core ve Extended setleri ayır

def build_core_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core set — sadece ep_0 + türetilmiş feature'lar + meta + target.
    Ana modelleme tablosu.
    """
    print("\n── 8a. features_core üretiliyor ──")

    turetilen = ["sedanter_saat", "gece_telefon_yogunluk", "sosyal_iletisim_yogunluk",
                 "aktivite_toplam", "mobilite_skoru", "gunduz_gece_telefon_orani",
                 "loc_minor_locations_total"]
    ep0_kollar = [c for c in df.columns if c.endswith("_ep_0")]

    secilen = META_KOLONLAR + ep0_kollar + [k for k in turetilen if k in df.columns] + [HEDEF_KOLON]
    secilen = [c for c in secilen if c in df.columns]   # var olan

    core = df[secilen].copy()
    print(f"  Boyut: {core.shape}")
    print(f"  Sütun yapısı: {len(META_KOLONLAR)} meta + {len(ep0_kollar)} ep_0 + "
          f"{sum(1 for k in turetilen if k in df.columns)} türetilmiş + 1 target")
    return core

def build_extended_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended set — core + ep_1/2/3 (epoch farkları) + lokasyon özetleri.
    Daha kapsamlı modelleme tablosu.
    """
    print("\n── 8b. features_extended üretiliyor ──")

    turetilen = ["sedanter_saat", "gece_telefon_yogunluk", "sosyal_iletisim_yogunluk",
                 "aktivite_toplam", "mobilite_skoru", "gunduz_gece_telefon_orani",
                 "loc_minor_locations_total"]
    ep_kollar = [c for c in df.columns
                 if any(c.endswith(f"_ep_{i}") for i in [0, 1, 2, 3])]
    loc_kollar = [c for c in df.columns
                  if c.startswith(("loc_food_", "loc_home_",
                                   "loc_self_dorm_", "loc_other_dorm_",
                                   "loc_social_", "loc_study_"))]

    secilen = (META_KOLONLAR + ep_kollar + loc_kollar
               + [k for k in turetilen if k in df.columns] + [HEDEF_KOLON])
    secilen = list(dict.fromkeys(secilen))   # tekrarsız sıralı
    secilen = [c for c in secilen if c in df.columns]

    extended = df[secilen].copy()
    print(f"  Boyut: {extended.shape}")
    print(f"  Sütun yapısı: {len(META_KOLONLAR)} meta + {len(ep_kollar)} epoch + "
          f"{len(loc_kollar)} lokasyon + {sum(1 for k in turetilen if k in df.columns)} türetilmiş + 1 target")
    return extended

# Ana akış

def run_feature_engineering() -> None:
    """Feature engineering pipeline'ı."""
    print("\n" + "═" * 60)
    print("  FEATURE ENGINEERING — Modelleme İçin Hazır Setler")
    print("═" * 60)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"master_imputed.csv yok: {INPUT_PATH}\n"
            f"Önce 'python -m src.imputation' çalıştır."
        )

    os.makedirs(REPORTS, exist_ok=True)

    print(f"\n  Yükleniyor: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"  Boyut: {df.shape}")

    # Hedef target sayım kayıt et
    target_dagilim_baslangic = df[HEDEF_KOLON].value_counts(normalize=True).sort_index().to_dict()

    log: dict = {"baslangic_sutun": df.shape[1]}
    drop_log: list[dict] = []

    # 1. Leakage drop
    df, leak = drop_leakage(df)
    log["leakage_drop"] = len(leak)
    drop_log.extend([{"asama": "leakage", "sutun": k} for k in leak])

    # 2. Light drop
    df, light = drop_light_columns(df)
    log["light_drop"] = len(light)
    drop_log.extend([{"asama": "light", "sutun": k} for k in light])

    # 3. Redundant drop
    df, redundant = drop_redundant(df)
    log["redundant_drop"] = len(redundant)
    drop_log.extend([{"asama": "redundant", "sutun": k} for k in redundant])

    # 4. Hourly drop
    df, hourly = drop_hourly_columns(df)
    log["hourly_drop"] = len(hourly)
    drop_log.extend([{"asama": "hourly", "sutun": k} for k in hourly])

    # 5. Near-zero-variance drop
    df, nzv = drop_near_zero_variance(df)
    log["nzv_drop"] = len(nzv)
    drop_log.extend([{"asama": "near_zero_variance", "sutun": k} for k in nzv])

    # 6. Seyrek lokasyon birleştirme
    df, minor = merge_minor_locations(df)
    log["minor_merge_birlestirilen"] = len(minor)

    # 7. Davranışsal feature türetme
    df = add_behavioral_features(df)

    log["ara_sutun_sayisi"] = df.shape[1]

    # 8. Core ve Extended setleri
    core = build_core_set(df)
    extended = build_extended_set(df)

    # Kaydet
    print("\n── 9. Kaydediliyor ──")
    core.to_csv(CORE_OUT, index=False)
    print(f"   features_core:     {CORE_OUT} ({core.shape[0]:,} × {core.shape[1]})")

    extended.to_csv(EXTENDED_OUT, index=False)
    print(f"   features_extended: {EXTENDED_OUT} ({extended.shape[0]:,} × {extended.shape[1]})")

    log["core_satir"]     = core.shape[0]
    log["core_sutun"]     = core.shape[1]
    log["extended_satir"] = extended.shape[0]
    log["extended_sutun"] = extended.shape[1]
    log["core_nan"]       = int(core.isnull().sum().sum())
    log["extended_nan"]   = int(extended.isnull().sum().sum())

    # Target dağılım kontrol
    print("\n── 10. Target dağılım kontrolü (değişmemiş olmalı) ──")
    for setadi, tablo in [("core", core), ("extended", extended)]:
        dag = tablo[HEDEF_KOLON].value_counts(normalize=True).sort_index()
        for k, v in dag.items():
            baslangic = target_dagilim_baslangic[k]
            esit = abs(v - baslangic) < 1e-6
            print(f"  {setadi:8s} sınıf {k}: %{v*100:5.2f}  {'' if esit else ' DEĞİŞTİ'}")

    # Logları kaydet
    pd.DataFrame([log]).to_csv(os.path.join(REPORTS, "46_feature_engineering_log.csv"), index=False)
    pd.DataFrame(drop_log).to_csv(os.path.join(REPORTS, "47_dropped_columns.csv"), index=False)
    print("\n  → kaydedildi: reports/46_feature_engineering_log.csv")
    print("  → kaydedildi: reports/47_dropped_columns.csv")

    print("\n" + "═" * 60)
    print("  FEATURE ENGINEERING TAMAMLANDI")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    run_feature_engineering()
