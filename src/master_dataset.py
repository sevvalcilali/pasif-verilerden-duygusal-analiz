"""
Master Dataset Birleştirme — Modelleme İçin Final Tablo
Bağımsız çalışır:

    python -m src.master_dataset

Akış:
  1. sensing_cleaned.csv yükle
  2. ema_cleaned.csv yükle
  3. Sensing'den günlük obj_iletisim hesapla (call + sms toplamı)
  4. uid + gün üzerinden inner join (sensing ⊕ ema)
  5. EMA matematiksel post-processing uygula (stress_z, pam_quadrant, phq4_risk, social_delta)
  6. Risk classifier her satıra uygula (profil_id + 4-sınıflı risk)
  7. cleaned_data/master_dataset.csv olarak kaydet

Çıktılar:
  cleaned_data/master_dataset.csv    — modelleme için final birleşik tablo
  reports/40_master_log.csv          — birleştirme adım adım özeti
  reports/41_risk_class_distribution.csv — 4-sınıflı risk dağılımı
  reports/42_profile_distribution.csv    — 12 profil dağılımı
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from src.config import PATHS
from EMA.src.ema_processing import tum_islemleri_uygula
from EMA.src.risk_classifier import (
    en_yakin_profil_bul,
    kural_tabanli_profil_eslemesi,
    risk_sinifi_4_hesapla,
)

# Yollar

SENSING_CLEANED = os.path.join(PATHS["cleaned_data"], "sensing_cleaned.csv")
EMA_CLEANED     = os.path.join(PATHS["cleaned_data"], "ema_cleaned.csv")
MASTER_OUT      = os.path.join(PATHS["cleaned_data"], "master_dataset.csv")
REPORTS         = PATHS["reports"]

# Sensing'de obj_iletisim'i oluşturan günlük (ep_0) sütunlar
ILETISIM_SUTUNLARI = [
    "call_in_num_ep_0",
    "call_out_num_ep_0",
    "sms_in_num_ep_0",
    "sms_out_num_ep_0",
]

# 1. Yükleme + tarih normalleştirme

def sensing_yukle() -> pd.DataFrame:
    """sensing_cleaned.csv yükle, gün sütununu YYYYMMDD int'e normalize et."""
    print("\n── 1. Sensing yükleniyor ──")
    df = pd.read_csv(SENSING_CLEANED, low_memory=False)
    print(f"  Boyut: {df.shape}")

    # day → YYYYMMDD int (EMA ile uyumlu olsun)
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df["gun"] = df["day"].dt.strftime("%Y%m%d").astype("Int64")
    df = df.drop(columns=["day"])
    print(f"  uid: {df['uid'].nunique()} | gün aralığı: {df['gun'].min()} → {df['gun'].max()}")
    return df

def ema_yukle() -> pd.DataFrame:
    """ema_cleaned.csv yükle. Sütunlar zaten ema_processing ile uyumlu."""
    print("\n── 2. EMA yükleniyor ──")
    df = pd.read_csv(EMA_CLEANED, low_memory=False)
    print(f"  Boyut: {df.shape}")
    print(f"  uid: {df['uid'].nunique()} | gün aralığı: {df['gun'].min()} → {df['gun'].max()}")
    return df

# 2. obj_iletisim hesabı

def obj_iletisim_hesapla(sensing: pd.DataFrame) -> pd.DataFrame:
    """
    Sensing günlük tablosundan obj_iletisim (objektif sosyal iletişim) hesapla.

    Tanım:  call_in + call_out + sms_in + sms_out  (günlük sayım)
    Bu, kullanıcının o gün gerçekten kurduğu/aldığı iletişim sayısıdır.
    EMA'daki sübjektif `social_level` ile birlikte social_delta üretiminde kullanılır.

    Eksik (NaN) ele alma: yokluk = 0 iletişim sayılır (tracking yoksa konuşma yok varsayımı).
    """
    print("\n── 3. obj_iletisim hesaplanıyor (sensing'den) ──")
    eksik_sutun = [k for k in ILETISIM_SUTUNLARI if k not in sensing.columns]
    if eksik_sutun:
        raise ValueError(f"Sensing'de iletişim sütunları eksik: {eksik_sutun}")

    iletisim = sensing[ILETISIM_SUTUNLARI].fillna(0).sum(axis=1)
    sensing = sensing.copy()
    sensing["obj_iletisim"] = iletisim.astype(float)
    print(f"  obj_iletisim ortalama: {iletisim.mean():.2f}, max: {iletisim.max():.0f}, NaN: 0")
    return sensing

# 3. Inner join

def master_join(sensing: pd.DataFrame, ema: pd.DataFrame) -> pd.DataFrame:
    """Sensing + EMA inner join (uid + gun)."""
    print("\n── 4. Inner join (uid + gun) ──")
    once_sens = len(sensing)
    once_ema  = len(ema)

    # Sensing tarafında is_ios gibi non-EMA meta sütunları korunur
    master = ema.merge(sensing, on=["uid", "gun"], how="inner")
    print(f"  Sensing: {once_sens:,} | EMA: {once_ema:,} | Master: {len(master):,}")
    print(f"  Master'da kullanıcı: {master['uid'].nunique()}")
    return master

# 4. EMA matematiksel post-processing

def ema_post_processing(master: pd.DataFrame) -> pd.DataFrame:
    """
    ema_processing.tum_islemleri_uygula çağır:
      - stress_z (within-person)
      - pam_valence, pam_arousal, pam_quadrant
      - phq4_anksiyete, phq4_depresyon, phq4_total, phq4_risk, gad2_pozitif, phq2_pozitif
      - social_subj_norm, social_obj_norm, social_delta
    """
    print("\n── 5. EMA matematiksel post-processing ──")

    # tum_islemleri_uygula float bekliyor; Int64 nullable türü np.int'e çevir
    sayisal_kollar = ["stress", "pam_score", "social_level",
                      "phq4_q1", "phq4_q2", "phq4_q3", "phq4_q4"]
    for kol in sayisal_kollar:
        if kol in master.columns:
            master[kol] = pd.to_numeric(master[kol], errors="coerce")

    # Bu noktada bazı satırlar kısmi NaN olabilir; tum_islemleri_uygula NaN'lı çalışır
    # ama PHQ-4 toplamı için 4 q kolonu da gerekli. Kısmi olanları sileceğiz.
    eksik_mask = master[sayisal_kollar].isnull().any(axis=1)
    eksik_sayisi = int(eksik_mask.sum())
    if eksik_sayisi:
        print(f"  Kısmi NaN nedeniyle çıkarılan satır: {eksik_sayisi:,}")
        master = master[~eksik_mask].copy()

    master = tum_islemleri_uygula(master)
    print(f"  EMA türetilen sütunlar eklendi. Master boyut: {master.shape}")
    return master

# 5. Risk classifier — her satıra uygula

def risk_etiketle(master: pd.DataFrame) -> pd.DataFrame:
    """
    Her satır için risk etiketleme — iki paralel profil eşleme + bir klinik risk:

      profil_id          : ema.md cascade (PRIMARY — hiyerarşik kural-tabanlı)
      profil_isim        : profil Türkçe ismi
      profil_risk_4      : cascade'den türeyen 4-sınıflı risk
      profil_guven       : cascade kuralının güven açıklaması

      profil_id_manhattan: 12 profil centroid'lerine Manhattan mesafesi (alternatif)
      profil_risk_4_manhattan: Manhattan eşlemeden 4-sınıflı risk

      klinik_risk_4      : bağımsız kural-tabanlı risk (stress_z+phq4+delta)
      final_risk_4       : max(cascade, klinik) — modelleme HEDEFİ
    """
    print("\n── 6. Risk classifier uygulanıyor (cascade primary + Manhattan alternatif) ──")
    t0 = time.time()

    stress_ham     = master["stress"].to_numpy()
    stress_z       = master["stress_z"].to_numpy()
    pam_quadrant   = master["pam_quadrant"].to_numpy()
    social_level   = master["social_level"].to_numpy()
    social_delta   = master["social_delta"].to_numpy()
    phq4_anksiyete = master["phq4_anksiyete"].to_numpy()
    phq4_depresyon = master["phq4_depresyon"].to_numpy()
    phq4_total     = master["phq4_total"].to_numpy()

    n = len(master)
    profil_id              = np.empty(n, dtype=object)
    profil_isim            = np.empty(n, dtype=object)
    profil_risk_4          = np.empty(n, dtype=np.int8)
    profil_guven           = np.empty(n, dtype=object)
    profil_id_manhattan    = np.empty(n, dtype=object)
    profil_risk_4_manhattan = np.empty(n, dtype=np.int8)
    klinik_risk_4          = np.empty(n, dtype=np.int8)

    for i in range(n):
        # Cascade (primary)
        c_id, c_isim, c_risk, c_guven = kural_tabanli_profil_eslemesi(
            stress_ham=int(stress_ham[i]),
            pam_quadrant=str(pam_quadrant[i]),
            social_level=int(social_level[i]),
            phq4_anksiyete=int(phq4_anksiyete[i]),
            phq4_depresyon=int(phq4_depresyon[i]),
            phq4_total=int(phq4_total[i]),
        )
        # Manhattan (alternatif)
        m_id, m_isim, m_risk = en_yakin_profil_bul(
            stress_z=float(stress_z[i]),
            pam_quadrant=str(pam_quadrant[i]),
            social_delta=float(social_delta[i]),
            phq4_anksiyete=int(phq4_anksiyete[i]),
            phq4_depresyon=int(phq4_depresyon[i]),
            phq4_total=int(phq4_total[i]),
        )
        # Klinik risk
        k_risk, _ = risk_sinifi_4_hesapla(
            stress_z=float(stress_z[i]),
            pam_quadrant=str(pam_quadrant[i]),
            social_delta=float(social_delta[i]),
            phq4_total=int(phq4_total[i]),
        )

        profil_id[i]               = c_id
        profil_isim[i]             = c_isim
        profil_risk_4[i]           = c_risk
        profil_guven[i]            = c_guven
        profil_id_manhattan[i]     = m_id
        profil_risk_4_manhattan[i] = m_risk
        klinik_risk_4[i]           = k_risk

    master["profil_id"]               = profil_id
    master["profil_isim"]             = profil_isim
    master["profil_risk_4"]           = profil_risk_4
    master["profil_guven"]            = profil_guven
    master["profil_id_manhattan"]     = profil_id_manhattan
    master["profil_risk_4_manhattan"] = profil_risk_4_manhattan
    master["klinik_risk_4"]           = klinik_risk_4
    master["final_risk_4"]            = np.maximum(profil_risk_4, klinik_risk_4).astype(np.int8)

    # Cascade ↔ Manhattan uyum oranı
    uyum = (profil_id == profil_id_manhattan).mean()
    print(f"  Cascade ↔ Manhattan profil uyum oranı: {uyum*100:.1f}%")

    sure = time.time() - t0
    print(f"  Etiketleme süresi: {sure:.1f}s ({n:,} satır)")
    return master

# 6. Dağılım raporları

def dagilim_raporlari(master: pd.DataFrame) -> None:
    """Risk sınıfı ve profil dağılımlarını rapor olarak kaydet."""
    print("\n── 7. Dağılım raporları ──")

    # 4-sınıflı risk dağılımı (cascade primary)
    risk_meta = {0: "İyi Durum", 1: "Hafif Risk", 2: "Orta Risk", 3: "Yüksek Risk"}
    risk_dagilim = pd.DataFrame({
        "risk_sinifi": [0, 1, 2, 3],
        "risk_isim":   [risk_meta[i] for i in range(4)],
    })
    for ad in ("profil_risk_4", "profil_risk_4_manhattan", "klinik_risk_4", "final_risk_4"):
        sayim = master[ad].value_counts().reindex([0, 1, 2, 3]).fillna(0).astype(int)
        risk_dagilim[f"{ad}_n"]     = sayim.values
        risk_dagilim[f"{ad}_yuzde"] = (sayim.values / len(master) * 100).round(2)

    risk_dagilim.to_csv(os.path.join(REPORTS, "41_risk_class_distribution.csv"), index=False)
    print("      → kaydedildi: reports/41_risk_class_distribution.csv")
    print("\n  4-sınıflı risk dağılımı:")
    print(risk_dagilim.to_string(index=False))

    # 12 profil dağılımı — cascade
    profil_dagilim = (
        master.groupby(["profil_id", "profil_isim"])
              .size()
              .reset_index(name="n")
              .sort_values("n", ascending=False)
    )
    profil_dagilim["yuzde"] = (profil_dagilim["n"] / len(master) * 100).round(2)
    profil_dagilim.to_csv(os.path.join(REPORTS, "42_profile_distribution.csv"), index=False)
    print("\n      → kaydedildi: reports/42_profile_distribution.csv (cascade primary)")
    print("\n  12 profil dağılımı (cascade primary):")
    print(profil_dagilim.to_string(index=False))

    # 12 profil dağılımı — Manhattan (karşılaştırma)
    profil_dagilim_man = (
        master.groupby("profil_id_manhattan")
              .size()
              .reset_index(name="n")
              .rename(columns={"profil_id_manhattan": "profil_id"})
              .sort_values("n", ascending=False)
    )
    profil_dagilim_man["yuzde"] = (profil_dagilim_man["n"] / len(master) * 100).round(2)
    profil_dagilim_man.to_csv(os.path.join(REPORTS, "43_profile_distribution_manhattan.csv"), index=False)
    print("      → kaydedildi: reports/43_profile_distribution_manhattan.csv (Manhattan alternatif)")

    # Cascade ↔ Manhattan çapraz tablo (uyumsuzluk analizi)
    capraz = pd.crosstab(master["profil_id"], master["profil_id_manhattan"])
    capraz.to_csv(os.path.join(REPORTS, "44_cascade_vs_manhattan.csv"))
    print("      → kaydedildi: reports/44_cascade_vs_manhattan.csv (çapraz tablo)")

# 7. Ana akış

def run_master_dataset() -> None:
    """Master dataset'i sıfırdan üret."""
    print("\n" + "═" * 60)
    print("  MASTER DATASET BİRLEŞTİRME")
    print("═" * 60)

    if not os.path.exists(SENSING_CLEANED):
        raise FileNotFoundError(f"Sensing temiz veri yok: {SENSING_CLEANED}")
    if not os.path.exists(EMA_CLEANED):
        raise FileNotFoundError(
            f"EMA temiz veri yok: {EMA_CLEANED}\n"
            f"Önce 'python -m EMA.src.ema_cleaner' ile temizleme adımını çalıştır."
        )

    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(os.path.dirname(MASTER_OUT), exist_ok=True)

    # Yükle
    sensing = sensing_yukle()
    ema     = ema_yukle()
    sensing_satir_once = len(sensing)
    ema_satir_once     = len(ema)

    # obj_iletisim
    sensing = obj_iletisim_hesapla(sensing)

    # Join
    master = master_join(sensing, ema)
    join_satir = len(master)

    # EMA post-processing
    master = ema_post_processing(master)
    proc_satir = len(master)

    # Risk classifier
    master = risk_etiketle(master)

    # Kaydet
    print("\n── 8. Master dataset kaydediliyor ──")
    master.to_csv(MASTER_OUT, index=False)
    print(f"  Çıktı: {MASTER_OUT}")
    print(f"  Final boyut: {master.shape[0]:,} satır × {master.shape[1]} sütun")
    print(f"  Kullanıcı: {master['uid'].nunique()}")
    print(f"  Bellekte: ~{master.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Dağılım raporları
    dagilim_raporlari(master)

    # Log
    log = pd.DataFrame([{
        "sensing_baslangic":      sensing_satir_once,
        "ema_baslangic":          ema_satir_once,
        "inner_join_sonrasi":     join_satir,
        "kismi_nan_sonrasi":      proc_satir,
        "final_satir":            master.shape[0],
        "final_sutun":            master.shape[1],
        "final_kullanici":        master["uid"].nunique(),
    }])
    log.to_csv(os.path.join(REPORTS, "40_master_log.csv"), index=False)
    print("\n      → kaydedildi: reports/40_master_log.csv")

    print("\n" + "═" * 60)
    print("  MASTER DATASET TAMAMLANDI")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    run_master_dataset()
