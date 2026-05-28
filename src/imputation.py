"""
Imputation Modülü — Master Dataset NaN Doldurma
Bağımsız çalışır:

    python -m src.imputation

Yapılan işlemler (sırayla):
  1. iOS light sensörü için binary flag türet (`has_light_sensor`)
  2. quality_* sütunlarını drop (modelleme için kritik değil)
  3. call_*, sms_*, unlock_*, light_*, loc_<spesifik>_*, sleep_* → 0 doldur
     (yokluk = "o aktivite/lokasyon yok" anlamına gelir)
  4. loc_dist_*, audio_*, act_*, step_* → kullanıcı medyanı
     (aynı kullanıcının diğer günlerinden tahmin)
  5. Hâlâ NaN kalan varsa → global medyan (fallback)
  6. NaN = 0 verifikasyonu

Çıktılar:
  cleaned_data/master_imputed.csv     — NaN'sız master tablo
  reports/45_imputation_log.csv       — sütun bazında uygulanan yöntem
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.config import PATHS

# Yollar ve sabitler

MASTER_IN  = os.path.join(PATHS["cleaned_data"], "master_dataset.csv")
MASTER_OUT = os.path.join(PATHS["cleaned_data"], "master_imputed.csv")
REPORTS    = PATHS["reports"]

# Drop edilecek prefix'ler (modelleme için kritik değil)
DROP_PREFIX = ("quality_",)

# 0 ile doldurulacak prefix'ler — yokluk = aktivite yok demek
ZERO_FILL_PREFIX = (
    "call_", "sms_", "unlock_",
    "light_",   # iOS NaN'leri 0 yapılır; binary flag has_light_sensor zaten taşıyor
    "loc_food_", "loc_home_",
    "loc_self_dorm_", "loc_other_dorm_",
    "loc_social_", "loc_study_",
    "loc_leisure_", "loc_health_",
    "loc_workout_", "loc_worship_",
    "other_playing_",
    "sleep_",   # sleep verisi yoksa o gün ölçüm yok demek (0 mantıklı)
)

# Kullanıcı medyanı ile doldurulacak prefix'ler
USER_MEDIAN_PREFIX = (
    "loc_dist_",        # mesafe verisi: aynı kullanıcı için tipik bir mesafe vardır
    "loc_max_dis_",     # kampüsten max uzaklık
    "loc_visit_",       # ziyaret sayısı
    "audio_",           # ses ortamı
    "act_",             # aktivite süresi
    "step_",            # adım
)

# Korunan meta/target/EMA sütunları (dokunma)
KORUNAN_SUTUNLAR = {
    "uid", "gun", "is_ios", "day",
    # Ham EMA
    "stress", "pam_score", "social_level", "phq4_q1", "phq4_q2", "phq4_q3", "phq4_q4",
    # Türetilmiş EMA
    "stress_z", "pam_valence", "pam_arousal", "pam_quadrant",
    "phq4_anksiyete", "phq4_depresyon", "phq4_total", "phq4_risk",
    "gad2_pozitif", "phq2_pozitif",
    "social_subj_norm", "social_obj_norm", "social_delta", "obj_iletisim",
    # Profil/risk
    "profil_id", "profil_isim", "profil_risk_4", "profil_guven",
    "profil_id_manhattan", "profil_risk_4_manhattan",
    "klinik_risk_4", "final_risk_4",
}

# Adım 1 — iOS light flag

def has_light_sensor_flag(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """light_mean_ep_0 dolu mu? Binary feature türet. iOS'ta sensör yok."""
    print("\n── 1. iOS light sensörü binary flag türetiliyor ──")

    if "light_mean_ep_0" not in df.columns:
        print("   light_mean_ep_0 yok, flag atlanıyor")
        return df, {"has_light_eklendi": False}

    df["has_light_sensor"] = df["light_mean_ep_0"].notna().astype(int)
    dagilim = df["has_light_sensor"].value_counts().to_dict()
    print(f"  has_light_sensor = 1 (Android, sensör var): {dagilim.get(1, 0):,}")
    print(f"  has_light_sensor = 0 (iOS, sensör yok):      {dagilim.get(0, 0):,}")

    return df, {"has_light_eklendi": True, "has_light_1": dagilim.get(1, 0),
                "has_light_0": dagilim.get(0, 0)}

# Adım 2 — quality_* drop

def drop_quality_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """quality_* sütunlarını modelleme için drop et."""
    print("\n── 2. quality_* sütunları drop ediliyor ──")
    drop_kollar = [c for c in df.columns if any(c.startswith(p) for p in DROP_PREFIX)]
    df = df.drop(columns=drop_kollar)
    print(f"  Drop edilen: {len(drop_kollar)} sütun")
    for k in drop_kollar:
        print(f"    - {k}")
    return df, drop_kollar

# Adım 3 — Sıfır doldurma

def zero_fill(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """call/sms/unlock/light/loc_<spesifik>/sleep prefix'lerini 0 doldur."""
    print("\n── 3. Sıfır doldurma (call/sms/unlock/light/loc_spesifik/sleep) ──")
    hedef = [
        c for c in df.columns
        if c not in KORUNAN_SUTUNLAR
        and any(c.startswith(p) for p in ZERO_FILL_PREFIX)
    ]
    nan_once = int(df[hedef].isnull().sum().sum())
    df[hedef] = df[hedef].fillna(0)
    nan_sonra = int(df[hedef].isnull().sum().sum())
    print(f"  Sütun sayısı: {len(hedef)}, doldurulan hücre: {nan_once - nan_sonra:,}")
    return df, {"zero_fill_sutun": len(hedef), "zero_fill_hucre": nan_once - nan_sonra}

# Adım 4 — Kullanıcı medyanı doldurma

def user_median_fill(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """loc_dist/audio/act/step prefix'lerini kullanıcı medyanıyla doldur."""
    print("\n── 4. Kullanıcı medyanı doldurma (loc_dist/audio/act/step) ──")
    hedef = [
        c for c in df.columns
        if c not in KORUNAN_SUTUNLAR
        and any(c.startswith(p) for p in USER_MEDIAN_PREFIX)
    ]
    nan_once = int(df[hedef].isnull().sum().sum())

    for kol in hedef:
        df[kol] = df.groupby("uid")[kol].transform(
            lambda x: x.fillna(x.median())
        )

    nan_sonra = int(df[hedef].isnull().sum().sum())
    print(f"  Sütun sayısı: {len(hedef)}, doldurulan hücre: {nan_once - nan_sonra:,}")
    print(f"  Kullanıcı medyanı bulunamayan (HİÇ değer yok): {nan_sonra:,} hücre kaldı")
    return df, {
        "user_median_sutun": len(hedef),
        "user_median_hucre": nan_once - nan_sonra,
        "user_median_fallback_kalan": nan_sonra,
    }

# Adım 5 — Global medyan fallback

def global_median_fallback(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Hâlâ NaN kalan sayısal sütunları global medyan ile doldur."""
    print("\n── 5. Global medyan fallback (hala NaN olanlar) ──")
    sayisal = df.select_dtypes(include="number").columns.tolist()
    sayisal = [c for c in sayisal if c not in KORUNAN_SUTUNLAR]

    nan_once = int(df[sayisal].isnull().sum().sum())
    if nan_once == 0:
        print("  Tüm sayısal sütunlar dolu, fallback gereksiz.")
        return df, {"global_median_hucre": 0}

    for kol in sayisal:
        if df[kol].isnull().any():
            med = df[kol].median()
            if pd.notna(med):
                df[kol] = df[kol].fillna(med)
            else:
                # Tüm sütun NaN — 0 doldur
                df[kol] = df[kol].fillna(0)

    nan_sonra = int(df[sayisal].isnull().sum().sum())
    print(f"  Doldurulan hücre: {nan_once - nan_sonra:,}")
    return df, {"global_median_hucre": nan_once - nan_sonra}

# Adım 6 — Verifikasyon

def verify_zero_nan(df: pd.DataFrame) -> dict:
    """Sayısal sütunlarda NaN kalmadığını doğrula."""
    print("\n── 6. NaN verifikasyonu ──")
    sayisal = df.select_dtypes(include="number")
    toplam_nan = int(sayisal.isnull().sum().sum())
    nan_sutunlar = sayisal.columns[sayisal.isnull().any()].tolist()

    if toplam_nan == 0:
        print("   Sayısal sütunlarda NaN sıfır")
    else:
        print(f"   Hâlâ {toplam_nan:,} NaN var, etkilenen sütun sayısı: {len(nan_sutunlar)}")
        print(f"    Etkilenen sütunlar (ilk 10): {nan_sutunlar[:10]}")

    return {
        "final_nan_sayisi": toplam_nan,
        "etkilenen_sutun":  len(nan_sutunlar),
    }

# Ana akış

def run_imputation() -> None:
    """Imputation pipeline'ı."""
    print("\n" + "═" * 60)
    print("  IMPUTATION — Master Dataset NaN Doldurma")
    print("═" * 60)

    if not os.path.exists(MASTER_IN):
        raise FileNotFoundError(f"Master dataset yok: {MASTER_IN}")

    os.makedirs(REPORTS, exist_ok=True)

    print(f"\n  Yükleniyor: {MASTER_IN}")
    df = pd.read_csv(MASTER_IN, low_memory=False)
    print(f"  Boyut: {df.shape}, başlangıç NaN: {df.isnull().sum().sum():,}")

    log: dict = {
        "baslangic_satir": df.shape[0],
        "baslangic_sutun": df.shape[1],
        "baslangic_nan":   int(df.isnull().sum().sum()),
    }

    # Adım 1 — has_light_sensor
    df, l1 = has_light_sensor_flag(df)
    log.update(l1)

    # Adım 2 — quality drop
    df, drop_quality = drop_quality_columns(df)
    log["drop_quality_sayisi"] = len(drop_quality)

    # Adım 3 — zero fill
    df, l3 = zero_fill(df)
    log.update(l3)

    # Adım 4 — user median
    df, l4 = user_median_fill(df)
    log.update(l4)

    # Adım 5 — global median fallback
    df, l5 = global_median_fallback(df)
    log.update(l5)

    # Adım 6 — verify
    l6 = verify_zero_nan(df)
    log.update(l6)
    log["final_satir"] = df.shape[0]
    log["final_sutun"] = df.shape[1]

    # Hedef target dağılım kontrol
    if "final_risk_4" in df.columns:
        dag = df["final_risk_4"].value_counts(normalize=True).sort_index()
        print("\n  Target dağılımı kontrolü (final_risk_4):")
        for k, v in dag.items():
            print(f"    Sınıf {k}: %{v*100:.2f}")

    # Kaydet
    print(f"\n  Kaydediliyor: {MASTER_OUT}")
    df.to_csv(MASTER_OUT, index=False)
    print(f"   Yazıldı. Boyut: {df.shape}")

    # Log kaydet
    log_df = pd.DataFrame([log])
    log_path = os.path.join(REPORTS, "45_imputation_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"      → kaydedildi: reports/45_imputation_log.csv")

    print("\n" + "═" * 60)
    print("  IMPUTATION TAMAMLANDI")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    run_imputation()
