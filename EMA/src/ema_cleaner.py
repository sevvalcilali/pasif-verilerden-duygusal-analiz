"""
EMA temizleme. Bağımsız çalışır: python -m EMA.src.ema_cleaner

Sırayla: kapsam dışı sütunları at, tamamen boş EMA satırlarını sil,
sütunları ema_processing'in beklediği isimlere çevir, aralık dışı değerleri
NaN yap, türleri Int64'e çevir, kısmi NaN desenini raporla ve kaydet.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMA_RAW      = os.path.join(BASE, "EMA", "veri seti", "general_ema.csv")
EMA_CLEANED  = os.path.join(BASE, "cleaned_data", "ema_cleaned.csv")
REPORTS      = os.path.join(BASE, "reports")

# Kullanılmayan sütunlar. phq4_score'u da atıyoruz; ema_processing yeniden hesaplıyor.
KAPSAM_DISI_SUTUNLAR = [
    "sse3-1", "sse3-2", "sse3-3", "sse3-4",
    "sse3_resp_mean", "sse3_resp_median",
    "phq4_resp_mean", "phq4_resp_median",
    "avg_ema_spent_time",
    "phq4_score",
]

# 7 EMA değişkeni — hepsi boşsa satır silinir
EMA_DEGISKENLER_HAM = ["stress", "pam", "social_level", "phq4-1", "phq4-2", "phq4-3", "phq4-4"]

# ema_processing'in beklediği isimlere eşleme
SUTUN_YENIDEN_ADLANDIRMA = {
    "day":    "gun",
    "pam":    "pam_score",
    "phq4-1": "phq4_q1",
    "phq4-2": "phq4_q2",
    "phq4-3": "phq4_q3",
    "phq4-4": "phq4_q4",
}

# Ölçekler doğal olarak tamsayı
INT_SUTUNLAR = ["stress", "pam_score", "social_level", "phq4_q1", "phq4_q2", "phq4_q3", "phq4_q4"]

# Validasyon için geçerli aralıklar
GECERLI_ARALIK = {
    "stress":       (1, 5),
    "pam_score":    (1, 16),
    "social_level": (1, 5),
    "phq4_q1":      (0, 3),
    "phq4_q2":      (0, 3),
    "phq4_q3":      (0, 3),
    "phq4_q4":      (0, 3),
}


def kapsam_disi_sutunlari_kaldir(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Kullanılmayan sütunları at."""
    mevcut = [k for k in KAPSAM_DISI_SUTUNLAR if k in df.columns]
    df = df.drop(columns=mevcut)
    print(f"  Kaldırılan sütun: {len(mevcut)}")
    for k in mevcut:
        print(f"    - {k}")
    return df, mevcut


def bos_ema_satirlari_sil(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """7 EMA değişkeninin hepsi NaN olan satırları sil (o gün EMA doldurulmamış)."""
    mevcut_ema = [k for k in EMA_DEGISKENLER_HAM if k in df.columns]
    bos_mask = df[mevcut_ema].isnull().all(axis=1)
    silinen = int(bos_mask.sum())
    df = df[~bos_mask].copy()
    print(f"  Tamamen boş EMA satırı silindi: {silinen:,}")
    print(f"  Kalan satır: {len(df):,}")
    return df, silinen


def sutunlari_yeniden_adlandir(df: pd.DataFrame) -> pd.DataFrame:
    """ema_processing'in beklediği isimlere çevir."""
    yapilanlar = {k: v for k, v in SUTUN_YENIDEN_ADLANDIRMA.items() if k in df.columns}
    df = df.rename(columns=yapilanlar)
    print(f"  Yeniden adlandırılan sütun: {len(yapilanlar)}")
    for eski, yeni in yapilanlar.items():
        print(f"    - {eski:10s} → {yeni}")
    return df


def kismi_nan_raporu(df: pd.DataFrame) -> pd.DataFrame:
    """Bazı cevaplı bazı boş satırların dolu/boş deseni dağılımı."""
    ema_kollari = [k for k in INT_SUTUNLAR if k in df.columns]
    dolu_mask_per_kol = df[ema_kollari].notnull().astype(int)
    desen = dolu_mask_per_kol.astype(str).agg("".join, axis=1)
    dagilim = desen.value_counts().reset_index()
    dagilim.columns = ["dolu_deseni", "satir_sayisi"]
    dagilim["dolu_deseni_yorum"] = dagilim["dolu_deseni"].apply(
        lambda s: "TAM DOLU" if s == "1" * len(ema_kollari) else f"KISMİ ({s.count('1')}/{len(ema_kollari)})"
    )
    print("\n  Kısmi NaN deseni:")
    print(dagilim.head(15).to_string(index=False))
    return dagilim


def aralik_disi_degerleri_temizle(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Aralık dışı değerleri NaN yap (örn. stress=7 olamaz). Emniyet kontrolü."""
    toplam = 0
    for kol, (lo, hi) in GECERLI_ARALIK.items():
        if kol not in df.columns:
            continue
        mask = (df[kol] < lo) | (df[kol] > hi)
        n = int(mask.sum())
        if n > 0:
            df.loc[mask, kol] = np.nan
            toplam += n
            print(f"    {kol}: aralık [{lo}, {hi}] dışı → NaN ({n} hücre)")
    if toplam == 0:
        print("  Aralık dışı değer bulunamadı.")
    return df, toplam


def tur_donusumleri(df: pd.DataFrame) -> pd.DataFrame:
    """Tamsayı ölçekleri Int64 (nullable) yap; 2.0 yerine 2 görünür."""
    for kol in INT_SUTUNLAR:
        if kol in df.columns:
            df[kol] = df[kol].astype("Int64")
    return df


def run_ema_cleaning() -> None:
    """EMA temizleme pipeline'ı."""
    print("\n" + "═" * 60)
    print("  EMA TEMİZLEME")
    print("═" * 60)

    if not os.path.exists(EMA_RAW):
        raise FileNotFoundError(f"EMA verisi bulunamadı: {EMA_RAW}")

    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(os.path.dirname(EMA_CLEANED), exist_ok=True)

    print(f"\n  Yükleniyor: {EMA_RAW}")
    df = pd.read_csv(EMA_RAW, low_memory=False)
    baslangic_satir, baslangic_sutun = df.shape
    print(f"  Başlangıç: {baslangic_satir:,} satır × {baslangic_sutun} sütun")

    print("\n── 1. Kapsam dışı sütunları kaldır ──")
    df, kaldirilan = kapsam_disi_sutunlari_kaldir(df)

    print("\n── 2. Tamamen boş EMA satırlarını sil ──")
    df, silinen = bos_ema_satirlari_sil(df)

    print("\n── 3. Sütun yeniden adlandırma ──")
    df = sutunlari_yeniden_adlandir(df)

    print("\n── 4. Geçersiz aralık değerleri ──")
    df, aralik_disi = aralik_disi_degerleri_temizle(df)

    print("\n── 5. Tür dönüşümleri (Int64) ──")
    df = tur_donusumleri(df)

    print("\n── 6. Kısmi NaN deseni ──")
    desen = kismi_nan_raporu(df)
    desen.to_csv(os.path.join(REPORTS, "39_ema_partial_nan_pattern.csv"), index=False)
    print("      → kaydedildi: reports/39_ema_partial_nan_pattern.csv")

    print("\n── 7. Temizlenmiş EMA kaydediliyor ──")
    df.to_csv(EMA_CLEANED, index=False)
    print(f"  Çıktı: {EMA_CLEANED}")
    print(f"  Final: {df.shape[0]:,} satır × {df.shape[1]} sütun")
    print(f"  Sütunlar: {list(df.columns)}")

    log = pd.DataFrame([{
        "baslangic_satir":       baslangic_satir,
        "baslangic_sutun":       baslangic_sutun,
        "kaldirilan_sutun":      len(kaldirilan),
        "silinen_bos_satir":     silinen,
        "aralik_disi_nan":       aralik_disi,
        "kalan_satir":           df.shape[0],
        "kalan_sutun":           df.shape[1],
        "kalan_nan_toplam":      int(df.isnull().sum().sum()),
        "benzersiz_uid":         df["uid"].nunique(),
    }])
    log.to_csv(os.path.join(REPORTS, "38_ema_clean_log.csv"), index=False)
    print("      → kaydedildi: reports/38_ema_clean_log.csv")

    print("\n" + "═" * 60)
    print("  EMA TEMİZLEME TAMAMLANDI")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    run_ema_cleaning()
