"""
Bağlamsal son temizleme adımı.
cleaner.py'den sonra çalışır — kullanıcı/platform bağlamına göre düzeltmeler yapar.

Ele alınan durumlar:
1. light_* sütunları iOS kullanıcılarda geçersiz → NaN
2. loc_dist_ep_0 fizik dışı değerler (>1000 km/gün) → NaN
3. Kalan NaN'ların kategori dağılımı raporu
"""

import os
import pandas as pd
import numpy as np
from src.config import PATHS

CLEANED_DIR = PATHS["cleaned_data"]
REPORT_DIR  = PATHS["reports"]

# Fiziksel sınırlar (metre)
LOC_DIST_DAILY_LIMIT = 1_000_000      # 1000 km/gün — bunun üstü cihaz hatası

# 1. Light sütunları — sadece Android'de geçerli

def mask_light_for_ios(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """iOS kullanıcılarında light_* değerlerini NaN'a çevir."""
    if "is_ios" not in df.columns:
        return df, 0
    light_cols = [c for c in df.columns if c.startswith("light_")]
    if not light_cols:
        return df, 0
    ios_mask = df["is_ios"] == 1
    before = df.loc[ios_mask, light_cols].notnull().sum().sum()
    df.loc[ios_mask, light_cols] = np.nan
    print(f"  Light → iOS NaN: {before:,} hücre ({len(light_cols)} sütun × "
          f"{ios_mask.sum():,} iOS satır)")
    return df, int(before)

# 2. Konum mesafesi — fiziksel sınır kontrolü

def cap_loc_distance(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    loc_dist_* sütunlarında günlük 1000 km'den fazla mesafe → NaN.
    loc_max_dis_from_campus_* uluslararası seyahat olabilir; dokunulmaz.
    """
    dist_cols = [c for c in df.columns
                 if c.startswith("loc_dist_") and not c.startswith("loc_dist_max")]
    total = 0
    for col in dist_cols:
        mask = df[col] > LOC_DIST_DAILY_LIMIT
        n = int(mask.sum())
        if n > 0:
            df.loc[mask, col] = np.nan
            total += n
            print(f"    {col}: >{LOC_DIST_DAILY_LIMIT/1000:.0f}km → NaN ({n:,} hücre)")
    print(f"  loc_dist fiziksel sınır → NaN: toplam {total:,} hücre")
    return df, total

# 4. NaN coverage raporu — kalan NaN'ların kaynağını sınıflandır

def nan_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Kalan NaN'ların hangi kategoriden geldiğini raporla."""
    rows = []
    total_nan = int(df.isnull().sum().sum())

    categories = [
        ("light_*       (iOS sensörü yok)",      lambda c: c.startswith("light_")),
        ("call_*/sms_*  (tracking yok)",         lambda c: c.startswith("call_") or c.startswith("sms_")),
        ("quality_*     (kalite raporu yok)",    lambda c: c.startswith("quality_")),
        ("sleep_*       (uyku ölçümü yok)",      lambda c: c.startswith("sleep_")),
        ("loc_zone_*    (bölgeye uğramadı)",     lambda c: any(c.startswith(p) for p in [
                                                    "loc_food_", "loc_home_",
                                                    "loc_self_dorm_", "loc_other_dorm_",
                                                    "loc_social_", "loc_study_",
                                                    "loc_leisure_", "loc_health_",
                                                    "loc_workout_", "loc_worship_"])),
        ("loc_dist_*    (fizik dışı kırpıldı)",  lambda c: c.startswith("loc_dist_")),
        ("diğer",                                lambda c: True),
    ]

    counted = set()
    for label, predicate in categories:
        cols = [c for c in df.columns if c not in counted and predicate(c)]
        if not cols:
            continue
        nan_count = int(df[cols].isnull().sum().sum())
        rows.append({
            "kategori": label,
            "sütun_sayısı": len(cols),
            "nan_adet": nan_count,
            "toplam_nan_%": round(nan_count / total_nan * 100, 2) if total_nan else 0.0,
        })
        counted.update(cols)

    report = pd.DataFrame(rows)
    print("\n  Kalan NaN kategori dağılımı:")
    print(report.to_string(index=False))
    print(f"\n  TOPLAM kalan NaN: {total_nan:,}")

    out = os.path.join(REPORT_DIR, "23_nan_coverage.csv")
    report.to_csv(out, index=False)
    print(f"      → kaydedildi: reports/23_nan_coverage.csv")
    return report

# Ana fonksiyon

def run_post_clean(cleaned: dict) -> dict:
    """Bağlamsal son temizleme adımlarını uygula."""
    print("\n=== BAĞLAMSAL SON TEMİZLEME ===")
    os.makedirs(REPORT_DIR, exist_ok=True)

    sensing = cleaned["sensing"].copy()
    print(f"\n  Başlangıç: {sensing.shape}, NaN={sensing.isnull().sum().sum():,}")

    # 1. Light → iOS NaN
    print("\n── 1. Light sütunları (iOS sensörü yok) ──")
    sensing, light_n = mask_light_for_ios(sensing)

    # 2. loc_dist fiziksel sınır
    print("\n── 2. loc_dist fiziksel sınır ──")
    sensing, loc_n = cap_loc_distance(sensing)

    # 3. NaN coverage raporu
    print("\n── 3. NaN coverage raporu ──")
    coverage = nan_coverage_report(sensing)

    # Kaydet
    print(f"\n  Sonuç: {sensing.shape}, NaN={sensing.isnull().sum().sum():,}")
    out_path = os.path.join(CLEANED_DIR, "sensing_cleaned.csv")
    sensing.to_csv(out_path, index=False)
    print(f"      → güncellendi: cleaned_data/sensing_cleaned.csv")

    # Log
    log = {
        "light_ios_nan":   light_n,
        "loc_dist_capped": loc_n,
        "kalan_nan":       int(sensing.isnull().sum().sum()),
    }
    pd.DataFrame([log]).to_csv(
        os.path.join(REPORT_DIR, "24_post_clean_log.csv"), index=False
    )
    print(f"      → kaydedildi: reports/24_post_clean_log.csv")

    cleaned["sensing"] = sensing
    print("\n=== SON TEMİZLEME TAMAMLANDI ===\n")
    return cleaned
