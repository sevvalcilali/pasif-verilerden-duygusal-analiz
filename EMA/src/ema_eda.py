"""
EMA keşifsel analiz. Bağımsız çalışır: python -m EMA.src.ema_eda

Üretilenler: 32_ema_overview, 33_ema_missing, 34_ema_days_per_user,
35_ema_distributions.png, 36_ema_user_coverage.png, 37_sensing_ema_uid_match.
Sonuncusu kritik: sensing ve EMA uid'leri kesişmezse join imkansız.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMA_CSV         = os.path.join(BASE, "EMA", "veri seti", "general_ema.csv")
SENSING_CLEANED = os.path.join(BASE, "cleaned_data", "sensing_cleaned.csv")
REPORTS         = os.path.join(BASE, "reports")

# Kullanılan EMA değişkenleri (sse3 ve meta hariç)
EMA_DEGISKENLER = ["stress", "pam", "social_level", "phq4-1", "phq4-2", "phq4-3", "phq4-4"]

# Dağılım grafiği için eksen aralıkları ve etiketler
EMA_OLCEKLER = {
    "stress":       (1, 5,  "Stres (1=hiç, 5=çok)"),
    "pam":          (1, 16, "PAM Skoru (1-16)"),
    "social_level": (1, 5,  "Sosyallik (1=yalnız, 5=hep birlikte)"),
    "phq4-1":       (0, 3,  "PHQ4-1 Anksiyete"),
    "phq4-2":       (0, 3,  "PHQ4-2 Endişe"),
    "phq4-3":       (0, 3,  "PHQ4-3 Çökkünlük"),
    "phq4-4":       (0, 3,  "PHQ4-4 İlgi Kaybı"),
}


def genel_bakis(df: pd.DataFrame) -> pd.DataFrame:
    """Satır, kullanıcı, tarih aralığı ve boş satır oranı."""
    print("\n── 1. Genel Bakış ──")

    df["day_dt"] = pd.to_datetime(df["day"], format="%Y%m%d", errors="coerce")

    ozet = {
        "toplam_satir":     len(df),
        "benzersiz_uid":    df["uid"].nunique(),
        "ilk_tarih":        str(df["day_dt"].min().date()),
        "son_tarih":        str(df["day_dt"].max().date()),
        "toplam_gun_araligi": (df["day_dt"].max() - df["day_dt"].min()).days,
        "bos_satir_orani":  round(df[EMA_DEGISKENLER].isnull().all(axis=1).mean() * 100, 2),
    }
    print(f"  Satır:        {ozet['toplam_satir']:,}")
    print(f"  Kullanıcı:    {ozet['benzersiz_uid']}")
    print(f"  Tarih:        {ozet['ilk_tarih']} → {ozet['son_tarih']} ({ozet['toplam_gun_araligi']} gün)")
    print(f"  7 değişken tamamen boş satır oranı: %{ozet['bos_satir_orani']}")

    tablo = pd.DataFrame([ozet])
    tablo.to_csv(os.path.join(REPORTS, "32_ema_overview.csv"), index=False)
    print("      → kaydedildi: reports/32_ema_overview.csv")
    return df


def eksik_veri_raporu(df: pd.DataFrame) -> None:
    """Sütun bazında eksik veri yüzdesi."""
    print("\n── 2. Eksik Veri (sütun bazında) ──")

    eksik = pd.DataFrame({
        "sutun":       df.columns,
        "eksik_adet":  df.isnull().sum().values,
        "eksik_yuzde": (df.isnull().mean() * 100).round(2).values,
        "dolu_adet":   df.notnull().sum().values,
    }).sort_values("eksik_yuzde", ascending=False).reset_index(drop=True)

    print(eksik.to_string(index=False))

    eksik.to_csv(os.path.join(REPORTS, "33_ema_missing.csv"), index=False)
    print("      → kaydedildi: reports/33_ema_missing.csv")


def kullanici_basina_gun(df: pd.DataFrame) -> None:
    """Kullanıcı başına kaç günde EMA dolu (en az bir değişken dolu)."""
    print("\n── 3. Kullanıcı başına dolu EMA gün sayısı ──")

    dolu_mask = df[EMA_DEGISKENLER].notnull().any(axis=1)
    dolu_df   = df[dolu_mask]

    gun_sayisi = dolu_df.groupby("uid")["day"].nunique().sort_values()
    tablo = pd.DataFrame({
        "uid": gun_sayisi.index,
        "dolu_gun_sayisi": gun_sayisi.values,
    })

    print(f"  Dolu EMA olan kullanıcı: {len(gun_sayisi)} / {df['uid'].nunique()}")
    print(f"  Gün sayısı istatistikleri:")
    print(f"    min:    {gun_sayisi.min()}")
    print(f"    median: {gun_sayisi.median():.0f}")
    print(f"    mean:   {gun_sayisi.mean():.1f}")
    print(f"    max:    {gun_sayisi.max()}")

    tablo.to_csv(os.path.join(REPORTS, "34_ema_days_per_user.csv"), index=False)
    print("      → kaydedildi: reports/34_ema_days_per_user.csv")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(gun_sayisi.values, bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("Dolu EMA gün sayısı")
    ax.set_ylabel("Kullanıcı sayısı")
    ax.set_title("Kullanıcı Başına Dolu EMA Gün Sayısı Dağılımı")
    ax.axvline(gun_sayisi.median(), color="red", linestyle="--", label=f"Medyan = {gun_sayisi.median():.0f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS, "36_ema_user_coverage.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("      → kaydedildi: reports/36_ema_user_coverage.png")


def ema_dagilimlari(df: pd.DataFrame) -> None:
    """7 EMA değişkeni için ayrı dağılım grafiği + sayısal özet."""
    print("\n── 4. EMA değişkenlerinin dağılımları ──")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for idx, kol in enumerate(EMA_DEGISKENLER):
        ax = axes[idx]
        veri = df[kol].dropna()
        if veri.empty:
            ax.set_title(f"{kol} — TÜMÜ BOŞ")
            continue

        lo, hi, etiket = EMA_OLCEKLER[kol]
        bins = np.arange(lo - 0.5, hi + 1.5, 1)
        ax.hist(veri, bins=bins, color="mediumpurple", edgecolor="white", alpha=0.85)
        ax.set_xticks(range(lo, hi + 1))
        ax.set_xlabel(etiket)
        ax.set_ylabel("Frekans")
        ax.set_title(f"{kol}  (N={len(veri):,})")

        ax.axvline(veri.mean(),   color="red",   linestyle="--", linewidth=1, label=f"ort={veri.mean():.2f}")
        ax.axvline(veri.median(), color="green", linestyle=":",  linewidth=1, label=f"med={veri.median():.1f}")
        ax.legend(fontsize=8)

    axes[-1].axis("off")  # 8. panel boş

    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS, "35_ema_distributions.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("      → kaydedildi: reports/35_ema_distributions.png")

    ozet = df[EMA_DEGISKENLER].describe().T.round(2)
    print("\n  Sayısal özet:")
    print(ozet.to_string())


def sensing_ema_eslesme(ema_df: pd.DataFrame) -> None:
    """Sensing ve EMA uid + tarih kesişimi. Kesişim küçükse join mümkün değil."""
    print("\n── 5. Sensing ↔ EMA uid + tarih eşleşmesi ──")

    if not os.path.exists(SENSING_CLEANED):
        print(f"  sensing_cleaned.csv bulunamadı, atlanıyor.")
        return

    sens = pd.read_csv(SENSING_CLEANED, usecols=["uid", "day"], low_memory=False)
    sens["day"] = pd.to_datetime(sens["day"], errors="coerce")

    ema = ema_df[["uid", "day_dt"]].rename(columns={"day_dt": "day"}).dropna()

    sens_uids = set(sens["uid"].unique())
    ema_uids  = set(ema["uid"].unique())
    ortak_uids = sens_uids & ema_uids

    print(f"  Sensing uid sayısı: {len(sens_uids)}")
    print(f"  EMA uid sayısı:     {len(ema_uids)}")
    print(f"  KESİŞİM:            {len(ortak_uids)}  (sıfırsa join imkansız)")

    sens_min, sens_max = sens["day"].min(), sens["day"].max()
    ema_min,  ema_max  = ema["day"].min(),  ema["day"].max()
    print(f"\n  Sensing tarih aralığı: {sens_min.date()} → {sens_max.date()}")
    print(f"  EMA tarih aralığı:     {ema_min.date()} → {ema_max.date()}")

    tarih_kesisim_var = sens_max >= ema_min and ema_max >= sens_min
    print(f"  Tarih aralıkları örtüşüyor mu? {'EVET' if tarih_kesisim_var else 'HAYIR'}")

    if ortak_uids:
        merged = sens.merge(ema, on=["uid", "day"], how="inner")
        print(f"\n  Inner join (uid+day) sonucu: {len(merged):,} satır")
        if len(merged) > 0:
            ortak_kullanici_join = merged["uid"].nunique()
            print(f"  Join'de gerçekten eşleşen kullanıcı: {ortak_kullanici_join}")
    else:
        merged = pd.DataFrame()
        print("  Hiç ortak uid yok, join atlanıyor.")

    rapor = pd.DataFrame([{
        "sensing_uid_sayisi":     len(sens_uids),
        "ema_uid_sayisi":         len(ema_uids),
        "kesisim_uid_sayisi":     len(ortak_uids),
        "sensing_ilk_tarih":      str(sens_min.date()),
        "sensing_son_tarih":      str(sens_max.date()),
        "ema_ilk_tarih":          str(ema_min.date()),
        "ema_son_tarih":          str(ema_max.date()),
        "tarih_ortusu":           "evet" if tarih_kesisim_var else "hayir",
        "uid_day_inner_join_satir": len(merged),
    }])
    rapor.to_csv(os.path.join(REPORTS, "37_sensing_ema_uid_match.csv"), index=False)
    print("      → kaydedildi: reports/37_sensing_ema_uid_match.csv")

    print("\n  Yorum:")
    if not ortak_uids:
        print("     Hiç ortak uid yok — sensing ve EMA farklı kohortlardan, birleştirilemez.")
    elif len(ortak_uids) < 10:
        print(f"     Sadece {len(ortak_uids)} ortak kullanıcı — istatistiksel olarak çok az.")
    elif not tarih_kesisim_var:
        print("     uid'ler eşleşse de tarih aralıkları örtüşmüyor — gün bazlı join boş çıkar.")
    elif len(merged) == 0:
        print("     uid + day eşleşmesi 0 — aynı kullanıcılar farklı zamanlarda kayıtlı.")
    else:
        print(f"     {len(merged):,} satır eşleşiyor, modelleme mümkün.")


def run_ema_eda() -> None:
    """EMA EDA'nın tüm akışı."""
    print("\n" + "═" * 60)
    print("  EMA KEŞİFSEL ANALİZİ")
    print("═" * 60)

    if not os.path.exists(EMA_CSV):
        raise FileNotFoundError(f"EMA verisi bulunamadı: {EMA_CSV}")

    os.makedirs(REPORTS, exist_ok=True)

    print(f"\n  Yükleniyor: {EMA_CSV}")
    df = pd.read_csv(EMA_CSV, low_memory=False)
    print(f"  Yüklendi: {len(df):,} satır × {df.shape[1]} sütun")

    df = genel_bakis(df)
    eksik_veri_raporu(df)
    kullanici_basina_gun(df)
    ema_dagilimlari(df)
    sensing_ema_eslesme(df)

    print("\n" + "═" * 60)
    print("  EMA EDA TAMAMLANDI")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    run_ema_eda()
