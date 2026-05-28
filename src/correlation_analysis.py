"""
Korelasyon Analizi Modülü — Modelleme Öncesi Aşama
Bağımsız çalışan modül. Pipeline'a entegre değil, doğrudan çağrılır:

    python -m src.correlation_analysis

İki amaç birden:

1) MULTICOLLINEARITY TARAMASI
   - sensing_cleaned.csv içindeki sayısal pasif feature'lar arası
     Pearson ve Spearman korelasyonu
   - |r| >= 0.85 olan çiftler "atılabilir aday" olarak işaretlenir
   - Variance Inflation Factor (VIF) hesaplanır; VIF > 10 problemli sayılır
   - Amaç: modele girmeden önce gereksiz/tekrarlanan feature'ları tespit

2) WANG 2014 BOYUTSAL KORELASYON
   - Pasif feature'lar 8 davranışsal boyuta indirgenir (uyku, konuşma,
     aktivite, mobilite, telefon kullanımı, sosyallik, ses, ışık)
   - Boyutlar arası korelasyon matrisi Wang et al. 2014'teki yapıyı
     karşılaştırmaya uygun temsile dönüşür
   - Amaç: pasif sinyallerin birbirleriyle nasıl konuştuğunu görmek,
     ileride sentetik EMA için weak-supervision kurallarına temel hazırlamak

Çıktılar (reports/ altına):
    25_pearson_ep0.csv         — full_day feature'ları arası Pearson matrisi
    26_spearman_ep0.csv        — full_day feature'ları arası Spearman matrisi
    27_high_corr_pairs.csv     — |r| >= 0.85 çiftleri (multicollinearity)
    28_vif_scores.csv          — VIF skorları (yüksek olanlar problemli)
    29_dimensional_corr.csv    — 8 davranışsal boyut arası korelasyon
    30_ep0_heatmap.png         — full_day Pearson heatmap
    31_dimensional_heatmap.png — 8 boyut korelasyon heatmap
"""

from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

from src.config import PATHS

sns.set_theme(style="white", palette="muted")

# Sabitler

CLEANED_SENSING = os.path.join(PATHS["cleaned_data"], "sensing_cleaned.csv")
REPORT_DIR      = PATHS["reports"]

# Multicollinearity için eşikler
YUKSEK_KORELASYON_ESIK = 0.85   # |r| üstü "atılabilir aday"
VIF_PROBLEM_ESIK       = 10.0   # VIF üstü problemli

# 8 davranışsal boyut → hangi sütunlardan ortalamayla toplanır
# ep_0 (full_day) bazlı; bir boyut birden fazla sütundan oluşabilir
DAVRANIS_BOYUTLARI: dict[str, list[str]] = {
    "konusma":          ["audio_convo_duration_ep_0", "audio_convo_num_ep_0", "audio_voice_ep_0"],
    "ses_ortam":        ["audio_amp_mean_ep_0", "audio_amp_std_ep_0"],
    "aktivite":         ["act_walking_ep_0", "act_running_ep_0", "act_on_foot_ep_0", "act_on_bike_ep_0"],
    "hareketsizlik":    ["act_still_ep_0"],
    "mobilite":         ["loc_dist_ep_0", "loc_visit_num_ep_0", "loc_max_dis_from_campus_ep_0"],
    "telefon_kullanim": ["unlock_num_ep_0", "unlock_duration_ep_0"],
    "iletisim":         ["call_in_num_ep_0", "call_out_num_ep_0", "call_in_duration_ep_0",
                         "call_out_duration_ep_0", "sms_in_num_ep_0", "sms_out_num_ep_0"],
    "isik":             ["light_mean_ep_0", "light_std_ep_0"],
}

# 1. Sütun seçimi

def full_day_feature_secimi(df: pd.DataFrame) -> list[str]:
    """
    Korelasyon analizi için _ep_0 (full_day) ile biten sayısal sütunları seç.
    27 sütun gelir; matris görseli okunabilir kalır.
    """
    secilen = [
        c for c in df.columns
        if c.endswith("_ep_0") and pd.api.types.is_numeric_dtype(df[c])
    ]
    return secilen

# 2. Korelasyon matrisleri

def korelasyon_matrisleri_uret(
    df: pd.DataFrame,
    sutunlar: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Verilen sütunlar için Pearson ve Spearman korelasyon matrislerini üret.
    NaN değerler pairwise işlenir (her çift kendi geçerli satırlarıyla).
    """
    veri = df[sutunlar]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pearson  = veri.corr(method="pearson",  min_periods=30)
        spearman = veri.corr(method="spearman", min_periods=30)

    return pearson, spearman

# 3. Yüksek korelasyon çiftleri (multicollinearity adayları)

def yuksek_korelasyon_ciftleri(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    esik: float = YUKSEK_KORELASYON_ESIK,
) -> pd.DataFrame:
    """
    |r| >= esik olan tüm (a, b) çiftlerini listele.
    Hem Pearson hem Spearman skoru aynı tabloda.
    """
    kayitlar: list[dict] = []
    sutunlar = list(pearson.columns)

    for i in range(len(sutunlar)):
        for j in range(i + 1, len(sutunlar)):
            a, b = sutunlar[i], sutunlar[j]
            r_p = pearson.iat[i, j]
            r_s = spearman.iat[i, j]

            if pd.isna(r_p) and pd.isna(r_s):
                continue

            max_abs = np.nanmax([abs(r_p) if not pd.isna(r_p) else 0,
                                 abs(r_s) if not pd.isna(r_s) else 0])
            if max_abs >= esik:
                kayitlar.append({
                    "feature_a":   a,
                    "feature_b":   b,
                    "pearson_r":   round(float(r_p), 4) if not pd.isna(r_p) else np.nan,
                    "spearman_r":  round(float(r_s), 4) if not pd.isna(r_s) else np.nan,
                    "max_abs":     round(float(max_abs), 4),
                })

    tablo = pd.DataFrame(kayitlar)
    if not tablo.empty:
        tablo = tablo.sort_values("max_abs", ascending=False).reset_index(drop=True)
    return tablo

# 4. VIF (Variance Inflation Factor)

def vif_hesapla(df: pd.DataFrame, sutunlar: list[str]) -> pd.DataFrame:
    """
    Her feature için VIF = 1 / (1 - R²) hesapla.
    R², feature_i'yi diğer tüm feature'lardan regresyonla tahminden gelir.

    Ön işlem:
        - NaN içeren satırlar listwise atılır
        - Sıfır varyanslı sütunlar atlanır (VIF tanımsız)
        - Standardizasyon yapılmaz (R² ölçek-bağımsızdır)

    VIF yorumu:
        1     → tamamen bağımsız
        1-5   → kabul edilebilir
        5-10  → orta düzeyde çoklu doğrusallık
        >10   → ciddi sorun
    """
    veri = df[sutunlar].dropna()
    if veri.empty:
        return pd.DataFrame({"feature": sutunlar, "vif": np.nan, "not": "Tüm satırlar NaN"})

    # Sıfır-varyanslı sütunları ele
    varyanslar = veri.var()
    gecerli_sutunlar = [c for c in sutunlar if varyanslar.get(c, 0) > 0]
    atlanan = [c for c in sutunlar if c not in gecerli_sutunlar]

    sonuclar: list[dict] = []
    for atlanan_kol in atlanan:
        sonuclar.append({"feature": atlanan_kol, "vif": np.nan, "not": "Sıfır varyans"})

    X = veri[gecerli_sutunlar].to_numpy()
    for idx, feature in enumerate(gecerli_sutunlar):
        y = X[:, idx]
        X_diger = np.delete(X, idx, axis=1)
        try:
            model = LinearRegression()
            model.fit(X_diger, y)
            r2 = float(model.score(X_diger, y))
            if r2 >= 1.0:
                vif = np.inf
            else:
                vif = 1.0 / (1.0 - r2)
        except Exception as hata:  # noqa: BLE001
            vif = np.nan
            not_bilgi = f"Hata: {hata}"
        else:
            not_bilgi = "problem" if vif > VIF_PROBLEM_ESIK else "ok"

        sonuclar.append({
            "feature": feature,
            "vif":     round(vif, 3) if np.isfinite(vif) else vif,
            "not":     not_bilgi,
        })

    tablo = pd.DataFrame(sonuclar)
    # Problemli olanlar tepede
    tablo["_sort"] = tablo["vif"].apply(lambda v: -np.inf if pd.isna(v) else -v)
    tablo = tablo.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return tablo

# 5. Davranışsal boyutlara indirgeme + boyutlar arası korelasyon

def boyutsal_ozet_uret(df: pd.DataFrame) -> pd.DataFrame:
    """
    8 davranışsal boyut için satır-bazlı (gün-bazlı) toplam değer üret.
    Her boyut, kendi sütunlarının z-standardize edilmiş ortalaması olur.
    Z standardizasyonu farklı birimlerin (saniye, sayım, lux) eşitlenmesi için.
    """
    cikti = pd.DataFrame(index=df.index)

    for boyut_adi, sutunlar in DAVRANIS_BOYUTLARI.items():
        mevcut = [c for c in sutunlar if c in df.columns]
        if not mevcut:
            print(f"  Uyarı — '{boyut_adi}' için hiçbir sütun bulunamadı, atlanıyor")
            continue

        # Z standardizasyon (global, kişi-içi değil — boyut karşılaştırması için)
        alt = df[mevcut].copy()
        for kol in mevcut:
            ort = alt[kol].mean()
            std = alt[kol].std(ddof=1)
            if pd.isna(std) or std == 0:
                alt[kol] = 0.0
            else:
                alt[kol] = (alt[kol] - ort) / std

        cikti[boyut_adi] = alt.mean(axis=1, skipna=True)

    return cikti

def boyutsal_korelasyon(boyutlar: pd.DataFrame) -> pd.DataFrame:
    """8 davranışsal boyut arası Pearson korelasyon matrisi."""
    return boyutlar.corr(method="pearson", min_periods=30)

# 6. Görselleştirme

def heatmap_kaydet(
    matris: pd.DataFrame,
    baslik: str,
    dosya_adi: str,
    annot: bool = False,
    figsize: tuple[float, float] = (14, 12),
) -> None:
    """Korelasyon matrisini renkli heatmap olarak diske kaydet."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matris,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "Korelasyon (r)"},
        ax=ax,
    )
    ax.set_title(baslik, fontsize=13, pad=12)
    ax.tick_params(axis="x", rotation=75)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    yol = os.path.join(REPORT_DIR, dosya_adi)
    fig.savefig(yol, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"      → kaydedildi: reports/{dosya_adi}")

# 7. Ana akış

def run_correlation_analysis() -> None:
    """Korelasyon analizinin tam akışını çalıştır ve çıktıları raporla."""
    print("\n" + "═" * 60)
    print("  KORELASYON ANALİZİ (Modelleme Öncesi)")
    print("═" * 60)

    if not os.path.exists(CLEANED_SENSING):
        raise FileNotFoundError(
            f"Temizlenmiş sensing verisi bulunamadı: {CLEANED_SENSING}\n"
            f"Önce 'python main.py' ile temizleme pipeline'ını çalıştır."
        )

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Yükle
    print("\n── 1. Temizlenmiş veri yükleniyor ──")
    df = pd.read_csv(CLEANED_SENSING, low_memory=False)
    print(f"  Satır: {len(df):,}, Sütun: {df.shape[1]}")

    # Full-day feature'ları seç
    print("\n── 2. Full-day (ep_0) feature seçimi ──")
    sutunlar = full_day_feature_secimi(df)
    print(f"  Seçilen sayısal feature: {len(sutunlar)}")

    # Korelasyon matrisleri
    print("\n── 3. Pearson + Spearman korelasyon matrisi ──")
    pearson, spearman = korelasyon_matrisleri_uret(df, sutunlar)
    pearson.to_csv(os.path.join(REPORT_DIR, "25_pearson_ep0.csv"))
    spearman.to_csv(os.path.join(REPORT_DIR, "26_spearman_ep0.csv"))
    print("      → kaydedildi: reports/25_pearson_ep0.csv")
    print("      → kaydedildi: reports/26_spearman_ep0.csv")

    # Heatmap
    heatmap_kaydet(
        pearson,
        baslik="Pasif Feature'lar Arası Pearson Korelasyonu (ep_0)",
        dosya_adi="30_ep0_heatmap.png",
        annot=False,
    )

    # Yüksek korelasyon çiftleri
    print(f"\n── 4. Yüksek korelasyon çiftleri (|r| ≥ {YUKSEK_KORELASYON_ESIK}) ──")
    ciftler = yuksek_korelasyon_ciftleri(pearson, spearman)
    ciftler.to_csv(os.path.join(REPORT_DIR, "27_high_corr_pairs.csv"), index=False)
    print(f"  Bulunan çift sayısı: {len(ciftler)}")
    if not ciftler.empty:
        print("\n  İlk 10 en yüksek çift:")
        print(ciftler.head(10).to_string(index=False))
    print("      → kaydedildi: reports/27_high_corr_pairs.csv")

    # VIF
    print("\n── 5. VIF (Variance Inflation Factor) ──")
    vif = vif_hesapla(df, sutunlar)
    vif.to_csv(os.path.join(REPORT_DIR, "28_vif_scores.csv"), index=False)
    print(f"  Problemli (VIF > {VIF_PROBLEM_ESIK}) feature sayısı: "
          f"{(vif['not'] == 'problem').sum()}")
    print("\n  En yüksek 10 VIF:")
    print(vif.head(10).to_string(index=False))
    print("      → kaydedildi: reports/28_vif_scores.csv")

    # Boyutsal indirgeme
    print("\n── 6. 8 davranışsal boyuta indirgeme ──")
    boyutlar = boyutsal_ozet_uret(df)
    print(f"  Üretilen boyut sayısı: {boyutlar.shape[1]}")

    boyut_corr = boyutsal_korelasyon(boyutlar)
    boyut_corr.to_csv(os.path.join(REPORT_DIR, "29_dimensional_corr.csv"))
    print("      → kaydedildi: reports/29_dimensional_corr.csv")

    heatmap_kaydet(
        boyut_corr,
        baslik="Davranışsal Boyutlar Arası Korelasyon (Wang 2014 Yapısı)",
        dosya_adi="31_dimensional_heatmap.png",
        annot=True,
        figsize=(9, 8),
    )

    print("\n  Boyutsal korelasyon matrisi:")
    print(boyut_corr.round(3).to_string())

    print("\n" + "═" * 60)
    print("  KORELASYON ANALİZİ TAMAMLANDI")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    run_correlation_analysis()
