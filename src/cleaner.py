import os
import pandas as pd
import numpy as np
from src.config import PATHS, NON_NEGATIVE_PREFIXES

CLEANED_DIR = PATHS["cleaned_data"]
REPORT_DIR  = PATHS["reports"]

# Sensing temizleme

def drop_fully_missing_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Tamamen boş (100% eksik) sütunları kaldır."""
    fully_missing = df.columns[df.isnull().all()].tolist()
    df = df.drop(columns=fully_missing)
    print(f"  Tamamen boş sütun kaldırıldı: {len(fully_missing)}")
    return df, fully_missing

def drop_high_missing_columns(df: pd.DataFrame,
                               threshold: float = 0.9) -> tuple[pd.DataFrame, list[str]]:
    """
    Belirtilen eşiğin üzerinde eksik değere sahip sütunları kaldır.
    Varsayılan eşik: %90
    """
    miss_ratio = df.isnull().mean()
    high_miss = miss_ratio[miss_ratio >= threshold].index.tolist()
    df = df.drop(columns=high_miss)
    print(f"  >%{int(threshold*100)} eksik sütun kaldırıldı: {len(high_miss)}")
    return df, high_miss

def fix_negative_values(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Negatif olmaması gereken sütunlardaki negatif değerleri NaN ile değiştir.
    Sensör verilerinde negatif süre/sayı fiziksel olarak mümkün değildir.
    """
    num_cols = df.select_dtypes(include="number").columns
    target_cols = [c for c in num_cols
                   if any(c.startswith(p) for p in NON_NEGATIVE_PREFIXES)]
    total_fixed = 0
    for col in target_cols:
        neg_mask = df[col] < 0
        count = neg_mask.sum()
        if count > 0:
            df.loc[neg_mask, col] = np.nan
            total_fixed += count
    print(f"  Negatif değer → NaN dönüştürüldü: {total_fixed:,} hücre")
    return df, total_fixed

def fix_duration_overflow(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Günlük süre sütunlarında 86400 saniyeyi (1 gün) aşan değerleri NaN yap.
    """
    # Sadece gerçek süre sütunları — mesafe sütunları (loc_dist_, loc_max_dis_) hariç
    dur_cols = [c for c in df.columns
                if c.endswith("_ep_0") and
                any(c.startswith(p) for p in ["act_", "unlock_duration_", "loc_"]) and
                not any(c.startswith(p) for p in ["loc_dist_", "loc_max_dis_"])]
    total_fixed = 0
    for col in dur_cols:
        over_mask = df[col] > 86400
        count = over_mask.sum()
        if count > 0:
            df.loc[over_mask, col] = np.nan
            total_fixed += count
    print(f"  Gün sınırı aşımı → NaN dönüştürüldü: {total_fixed:,} hücre")
    return df, total_fixed

def fix_sleep_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """24 saatten uzun uyku sürelerini NaN yap."""
    count = 0
    if "sleep_duration" in df.columns:
        mask = df["sleep_duration"] > 86400
        count = mask.sum()
        df.loc[mask, "sleep_duration"] = np.nan
    print(f"  Uyku süresi aşımı (>24h) → NaN: {count:,} hücre")
    return df, count

def fill_missing_sensing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eksik değer doldurma stratejileri:
    - Sayı/adet sütunları (count, num)  → 0 ile doldur (gözlem yoksa 0 mantıklı)
    - Süre sütunları (duration)         → 0 ile doldur
    - Kalite sütunları (quality_)       → NaN bırak (kalite bilinmiyorsa doldurmak yanıltıcı)
    - sleep_* sütunları                 → NaN bırak
    - Diğer sayısal                     → sütun medyanı ile doldur
    """
    num_cols = df.select_dtypes(include="number").columns

    zero_fill_prefixes = [
        "act_", "step_",
        "unlock_num", "unlock_duration",
        "other_playing_num", "other_playing_duration",
        "loc_visit_num",
        "audio_convo_num",
    ]
    skip_prefixes = ["quality_", "sleep_", "light_",
                     "call_", "sms_",
                     "loc_food_", "loc_home_",
                     "loc_self_dorm_", "loc_other_dorm_", "loc_social_",
                     "loc_study_", "loc_leisure_", "loc_health_",
                     "loc_workout_", "loc_worship_"]

    zero_filled = 0
    median_filled = 0

    for col in num_cols:
        if df[col].isnull().sum() == 0:
            continue
        if any(col.startswith(p) for p in skip_prefixes):
            continue
        if any(col.startswith(p) for p in zero_fill_prefixes):
            df[col] = df[col].fillna(0)
            zero_filled += 1
        else:
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)
                median_filled += 1

    print(f"  Eksik doldurma: {zero_filled} sütun → 0, {median_filled} sütun → medyan")
    return df

def clean_sensing(df: pd.DataFrame) -> pd.DataFrame:
    """Sensing veri seti için tam temizleme pipeline'ı."""
    print("\n── sensing.csv temizleniyor ──")
    original_shape = df.shape

    df, dropped_full   = drop_fully_missing_columns(df)
    df, dropped_high   = drop_high_missing_columns(df, threshold=0.90)
    df, neg_count      = fix_negative_values(df)
    df, overflow_count = fix_duration_overflow(df)
    df, sleep_count    = fix_sleep_outliers(df)
    df                 = fill_missing_sensing(df)

    print(f"\n  Özet: {original_shape} → {df.shape}")
    print(f"  Kaldırılan sütun: {original_shape[1] - df.shape[1]}")
    print(f"  Kalan NaN: {df.isnull().sum().sum():,}")

    # Temizlik logu
    log = {
        "orijinal_satır": original_shape[0],
        "orijinal_sütun": original_shape[1],
        "temiz_satır":    df.shape[0],
        "temiz_sütun":    df.shape[1],
        "tamamen_boş_sütun_kaldırıldı": len(dropped_full),
        "yüksek_eksik_sütun_kaldırıldı": len(dropped_high),
        "negatif→nan": neg_count,
        "overflow→nan": overflow_count,
        "sleep→nan": sleep_count,
        "kalan_nan": int(df.isnull().sum().sum()),
    }
    pd.DataFrame([log]).to_csv(
        os.path.join(REPORT_DIR, "22_sensing_clean_log.csv"), index=False
    )
    print("      → kaydedildi: reports/22_sensing_clean_log.csv")
    return df

# Kaydetme

def save_cleaned(df: pd.DataFrame, name: str) -> None:
    """Temizlenmiş veri setini CSV olarak kaydet."""
    os.makedirs(CLEANED_DIR, exist_ok=True)
    path = os.path.join(CLEANED_DIR, f"{name}_cleaned.csv")
    df.to_csv(path, index=False)
    print(f"  Kaydedildi: cleaned_data/{name}_cleaned.csv  ({len(df):,} satır)")

# Ana fonksiyon

def run_cleaning(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Tüm veri setlerini temizle ve kaydet."""
    print("\n=== VERİ TEMİZLEME ===")
    os.makedirs(CLEANED_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    cleaned = {}

    # Sensing
    cleaned["sensing"] = clean_sensing(datasets["sensing"].copy())
    save_cleaned(cleaned["sensing"], "sensing")

    # Steps — zaten temiz, sadece kopyala
    if "steps" in datasets and not datasets["steps"].empty:
        cleaned["steps"] = datasets["steps"].copy()
        no_time = cleaned["steps"]["day"].isnull().sum()
        cleaned["steps"] = cleaned["steps"].dropna(subset=["day"])
        print(f"\n── steps.csv: {no_time} geçersiz tarih silindi ──")
        save_cleaned(cleaned["steps"], "steps")

    print("\n=== TEMİZLEME TAMAMLANDI ===\n")
    return cleaned
