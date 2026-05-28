"""
Data Quality Modülü — Modelleme Öncesi Son Kontrol
Bağımsız çalışır:

    python -m src.data_quality

Akış:
  1. Tek sınıflı kullanıcıları drop (LOSO-CV mantığı)
  2. < 7 gün verisi olan kullanıcıları drop (Wang 2014 minimum)
  3. Duplicate (uid + gun) kontrolü
  4. Target değer validasyonu (sadece 0,1,2,3)
  5. Türetilmiş feature outlier cap (mantıklı aralık)
  6. features_core_clean.csv + features_extended_clean.csv üret

Çıktılar:
  cleaned_data/features_core_clean.csv
  cleaned_data/features_extended_clean.csv
  reports/48_data_quality_log.csv
  reports/49_excluded_users.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.config import PATHS

# Yollar

CORE_IN     = os.path.join(PATHS["cleaned_data"], "features_core.csv")
EXTEND_IN   = os.path.join(PATHS["cleaned_data"], "features_extended.csv")
CORE_OUT    = os.path.join(PATHS["cleaned_data"], "features_core_clean.csv")
EXTEND_OUT  = os.path.join(PATHS["cleaned_data"], "features_extended_clean.csv")
REPORTS     = PATHS["reports"]

# Sabitler

HEDEF_KOLON     = "final_risk_4"
GECERLI_TARGET  = {0, 1, 2, 3}
MIN_GUN_ESIK    = 7        # Wang 2014 minimum
GUN_SANIYE      = 86400    # 24 saat

# Outlier cap kuralları (sütun → (min, max))
OUTLIER_CAP_KURALLARI = {
    "sedanter_saat":              (0,   24),
    "aktivite_toplam":            (0,   GUN_SANIYE),
    "gece_telefon_yogunluk":      (0,   500),     # mantıklı üst sınır
    "sosyal_iletisim_yogunluk":   (0,   10000),
    "mobilite_skoru":             (0,   1_000_000),  # 1000 km (zaten post_clean'de uygulandı)
    "gunduz_gece_telefon_orani":  (0,   1000),    # +1/+1 smoothing var ama astronomik olabilir
}

# 1. Tek sınıflı kullanıcıları drop

def drop_single_class_users(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """final_risk_4 değişkeninde sadece tek sınıfa sahip kullanıcıları çıkar."""
    print("\n── 1. Tek sınıflı kullanıcı kontrolü ──")
    sinif_sayisi = df.groupby("uid")[HEDEF_KOLON].nunique()
    tek_sinif = sinif_sayisi[sinif_sayisi == 1].index.tolist()
    if not tek_sinif:
        print("   Tek sınıflı kullanıcı yok")
        return df, []

    once_satir = len(df)
    df = df[~df["uid"].isin(tek_sinif)].copy()
    print(f"  Drop edilen kullanıcı: {len(tek_sinif)}")
    print(f"  Drop edilen satır: {once_satir - len(df):,}")
    return df, tek_sinif

# 2. Az veri olan kullanıcıları drop

def drop_low_data_users(df: pd.DataFrame,
                        min_gun: int = MIN_GUN_ESIK) -> tuple[pd.DataFrame, list[str]]:
    """min_gun altında veri üreten kullanıcıları çıkar."""
    print(f"\n── 2. < {min_gun} gün verisi olan kullanıcı kontrolü (Wang 2014 minimum) ──")
    gun_sayisi = df.groupby("uid").size()
    az_veri = gun_sayisi[gun_sayisi < min_gun].index.tolist()
    if not az_veri:
        print(f"   < {min_gun} gün verisi olan kullanıcı yok")
        return df, []

    once_satir = len(df)
    df = df[~df["uid"].isin(az_veri)].copy()
    print(f"  Drop edilen kullanıcı: {len(az_veri)}")
    print(f"  Drop edilen satır: {once_satir - len(df):,}")
    return df, az_veri

# 3. Duplicate kontrolü

def check_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """uid+gun çiftinin tekrar etmediğini doğrula."""
    print("\n── 3. Duplicate (uid+gun) kontrolü ──")
    dup_mask = df.duplicated(subset=["uid", "gun"], keep="first")
    dup_sayisi = int(dup_mask.sum())
    if dup_sayisi == 0:
        print("   Duplicate yok")
        return df, 0
    print(f"   {dup_sayisi:,} duplicate bulundu, drop ediliyor")
    df = df[~dup_mask].copy()
    return df, dup_sayisi

# 4. Target validasyonu

def validate_target(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """final_risk_4 sadece {0,1,2,3} olmalı."""
    print("\n── 4. Target değer validasyonu ──")
    invalid_mask = ~df[HEDEF_KOLON].isin(GECERLI_TARGET) | df[HEDEF_KOLON].isnull()
    invalid_sayisi = int(invalid_mask.sum())
    if invalid_sayisi == 0:
        print("   Tüm target değerleri valid {0,1,2,3}")
        return df, 0
    print(f"   {invalid_sayisi:,} invalid target bulundu, drop ediliyor")
    df = df[~invalid_mask].copy()
    return df, invalid_sayisi

# 5. Outlier cap

def cap_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Türetilmiş feature'ları mantıklı aralığa clip et."""
    print("\n── 5. Outlier cap (clipping) ──")
    cap_log: dict[str, int] = {}
    for kol, (lo, hi) in OUTLIER_CAP_KURALLARI.items():
        if kol not in df.columns:
            continue
        before_low  = int((df[kol] < lo).sum())
        before_high = int((df[kol] > hi).sum())
        if before_low + before_high == 0:
            continue
        df[kol] = df[kol].clip(lower=lo, upper=hi)
        cap_log[kol] = before_low + before_high
        print(f"  {kol}: cap aralığı=[{lo}, {hi}] | clip edilen: {before_low + before_high:,} hücre")
    if not cap_log:
        print("   Hiç outlier cap gerekmedi")
    return df, cap_log

# Tek tablo işleme

def process_table(df: pd.DataFrame, set_adi: str) -> tuple[pd.DataFrame, dict]:
    """Tek bir tabloyu (core veya extended) işle."""
    print(f"\n{'═' * 60}\n  {set_adi.upper()} işleniyor\n{'═' * 60}")
    baslangic = {
        "baslangic_satir":     len(df),
        "baslangic_kullanici": df["uid"].nunique(),
    }
    log = dict(baslangic)

    # Hedef dağılım
    target_dag_oncesi = df[HEDEF_KOLON].value_counts(normalize=True).sort_index().to_dict()

    df, tek_sinif = drop_single_class_users(df)
    log["tek_sinifli_drop"] = len(tek_sinif)

    df, az_veri = drop_low_data_users(df)
    log["az_veri_drop"] = len(az_veri)

    df, dup = check_duplicates(df)
    log["duplicate_drop"] = dup

    df, invalid = validate_target(df)
    log["invalid_target_drop"] = invalid

    df, cap_log = cap_outliers(df)
    log["cap_toplam_hucre"] = sum(cap_log.values())

    log["final_satir"]     = len(df)
    log["final_kullanici"] = df["uid"].nunique()

    # Drop edilen kullanıcılar (tekil liste)
    drop_uid = sorted(set(tek_sinif + az_veri))
    log["drop_uid_listesi"] = ";".join(drop_uid) if drop_uid else ""

    # Target dağılım kontrolü
    target_dag_sonra = df[HEDEF_KOLON].value_counts(normalize=True).sort_index().to_dict()
    print(f"\n  {set_adi} target dağılımı:")
    for k in sorted(set(target_dag_oncesi) | set(target_dag_sonra)):
        once  = target_dag_oncesi.get(k, 0) * 100
        sonra = target_dag_sonra.get(k, 0) * 100
        print(f"    Sınıf {k}: %{once:5.2f} → %{sonra:5.2f}")

    print(f"\n  {set_adi} SONUÇ: {log['final_satir']:,} satır × {df.shape[1]} sütun, "
          f"{log['final_kullanici']} kullanıcı")
    return df, log

# Ana akış

def run_data_quality() -> None:
    """Data quality pipeline'ı — core ve extended için ayrı."""
    print("\n" + "═" * 60)
    print("  DATA QUALITY — Modelleme Öncesi Son Kontrol")
    print("═" * 60)

    if not os.path.exists(CORE_IN):
        raise FileNotFoundError(
            f"features_core.csv yok: {CORE_IN}\n"
            f"Önce 'python -m src.feature_engineering' çalıştır."
        )

    os.makedirs(REPORTS, exist_ok=True)

    print(f"\n  Yükleniyor: features_core.csv")
    core = pd.read_csv(CORE_IN, low_memory=False)

    print(f"  Yükleniyor: features_extended.csv")
    extended = pd.read_csv(EXTEND_IN, low_memory=False)

    # İşle
    core, core_log = process_table(core, "features_core")
    extended, ext_log = process_table(extended, "features_extended")

    # Kaydet
    print("\n── Kaydediliyor ──")
    core.to_csv(CORE_OUT, index=False)
    print(f"   {CORE_OUT}")
    extended.to_csv(EXTEND_OUT, index=False)
    print(f"   {EXTEND_OUT}")

    # Log
    log_df = pd.DataFrame([
        {"set": "core",     **core_log},
        {"set": "extended", **ext_log},
    ])
    log_df.to_csv(os.path.join(REPORTS, "48_data_quality_log.csv"), index=False)
    print("  → kaydedildi: reports/48_data_quality_log.csv")

    # Drop edilen kullanıcılar tek liste
    drop_uid_full = set()
    for uid_str in [core_log.get("drop_uid_listesi", ""), ext_log.get("drop_uid_listesi", "")]:
        if uid_str:
            drop_uid_full.update(uid_str.split(";"))
    if drop_uid_full:
        pd.DataFrame({"uid": sorted(drop_uid_full)}).to_csv(
            os.path.join(REPORTS, "49_excluded_users.csv"), index=False
        )
        print(f"  → kaydedildi: reports/49_excluded_users.csv ({len(drop_uid_full)} kullanıcı)")

    print("\n" + "═" * 60)
    print("  DATA QUALITY TAMAMLANDI")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    run_data_quality()
