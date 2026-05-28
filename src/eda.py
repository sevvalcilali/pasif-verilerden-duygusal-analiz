import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless ortamda grafik kaydet
import seaborn as sns
from src.config import PATHS, COLUMN_GROUPS

sns.set_theme(style="whitegrid", palette="muted")
REPORT_DIR = PATHS["reports"]

# Yardımcı

def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(REPORT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"      → kaydedildi: reports/{name}")

def _cols_by_group(df: pd.DataFrame) -> dict[str, list[str]]:
    """Sütunları prefix grubuna göre ayır."""
    result: dict[str, list[str]] = {}
    for group, prefixes in COLUMN_GROUPS.items():
        cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
        if cols:
            result[group] = cols
    return result

# EDA adımları

def dataset_overview(datasets: dict[str, pd.DataFrame]) -> None:
    """Her veri setinin temel bilgilerini yazdır."""
    print("\n── GENEL BAKIŞ ──")
    rows = []
    for name, df in datasets.items():
        uid_col = "uid" if "uid" in df.columns else None
        user_count = df[uid_col].nunique() if uid_col else "-"
        rows.append({
            "Veri Seti": name,
            "Satır": f"{len(df):,}",
            "Sütun": df.shape[1],
            "Kullanıcı": user_count,
            "Bellek (MB)": f"{df.memory_usage(deep=True).sum() / 1e6:.1f}",
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    # Kaydet
    summary.to_csv(os.path.join(REPORT_DIR, "01_dataset_overview.csv"), index=False)
    print("      → kaydedildi: reports/01_dataset_overview.csv")

def sensing_basic_stats(df: pd.DataFrame) -> None:
    """Sensing.csv için temel istatistikler."""
    print("\n── SENSING TEMEL İSTATİSTİKLER ──")

    # Kullanıcı başına gün sayısı
    days_per_user = df.groupby("uid")["day"].nunique().describe()
    print("  Kullanıcı başına gün sayısı:\n", days_per_user.to_string())

    # Tarih aralığı
    print(f"\n  Tarih aralığı: {df['day'].min().date()} → {df['day'].max().date()}")

    # iOS vs Android
    if "is_ios" in df.columns:
        platform = df.drop_duplicates("uid")["is_ios"].value_counts()
        print(f"\n  Platform: Android={platform.get(0,0)}, iOS={platform.get(1,0)}")

    # Grup bazlı sütun sayısı
    groups = _cols_by_group(df)
    print("\n  Sütun grupları:")
    for g, cols in groups.items():
        print(f"    {g:15s}: {len(cols):4d} sütun")

    # Rapor: describe()
    num_df = df.select_dtypes(include="number")
    desc = num_df.describe().T
    desc.to_csv(os.path.join(REPORT_DIR, "02_sensing_describe.csv"))
    print("      → kaydedildi: reports/02_sensing_describe.csv")

def plot_days_per_user(df: pd.DataFrame) -> None:
    """Kullanıcı başına veri günü dağılımı."""
    days = df.groupby("uid")["day"].nunique().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(days)), days.values, color="steelblue", width=1.0)
    ax.set_xlabel("Kullanıcı (sıralı)")
    ax.set_ylabel("Gün sayısı")
    ax.set_title("Kullanıcı Başına Kayıtlı Gün Sayısı")
    _save(fig, "03_days_per_user.png")

def plot_sleep_distribution(df: pd.DataFrame) -> None:
    """Uyku süresi dağılımı."""
    if "sleep_duration" not in df.columns:
        return
    data = df["sleep_duration"].dropna()
    data = data[data > 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(data / 3600, bins=50, color="mediumpurple", edgecolor="white")
    axes[0].set_xlabel("Uyku Süresi (saat)")
    axes[0].set_ylabel("Frekans")
    axes[0].set_title("Uyku Süresi Dağılımı")
    axes[1].boxplot(data / 3600, vert=False, patch_artist=True,
                    boxprops=dict(facecolor="mediumpurple", alpha=0.6))
    axes[1].set_xlabel("Uyku Süresi (saat)")
    axes[1].set_title("Uyku Süresi Box Plot")
    plt.tight_layout()
    _save(fig, "04_sleep_distribution.png")

def plot_activity_summary(df: pd.DataFrame) -> None:
    """Günlük aktivite sürelerinin ortalama dağılımı."""
    act_cols = [c for c in df.columns if c.startswith("act_") and c.endswith("_ep_0")]
    if not act_cols:
        return
    means = df[act_cols].mean().sort_values(ascending=False)
    means = means[means > 0]
    labels = [c.replace("act_", "").replace("_ep_0", "") for c in means.index]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(labels, means.values / 3600, color="teal")
    ax.set_xlabel("Ortalama Süre (saat/gün)")
    ax.set_title("Aktivite Türü — Günlük Ortalama Süre")
    plt.tight_layout()
    _save(fig, "05_activity_summary.png")

def plot_hourly_unlock(df: pd.DataFrame) -> None:
    """Saat bazlı telefon kullanım yoğunluğu."""
    unlock_hr_cols = [c for c in df.columns
                      if c.startswith("unlock_num_hr_")]
    if not unlock_hr_cols:
        return
    hours = list(range(24))
    col_map = {c: int(c.split("_hr_")[1]) for c in unlock_hr_cols}
    hour_means = {col_map[c]: df[c].mean() for c in unlock_hr_cols}
    ordered = [hour_means.get(h, 0) for h in hours]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(hours, ordered, color="darkorange")
    ax.set_xticks(hours)
    ax.set_xlabel("Saat")
    ax.set_ylabel("Ortalama kilit açma sayısı")
    ax.set_title("Saatlik Ortalama Telefon Kullanımı (Kilit Açma)")
    plt.tight_layout()
    _save(fig, "08_hourly_unlock.png")

def run_eda(datasets: dict[str, pd.DataFrame]) -> None:
    """Tüm EDA adımlarını çalıştır."""
    print("\n=== KEŞİFSEL ANALİZ (EDA) ===")
    os.makedirs(REPORT_DIR, exist_ok=True)

    dataset_overview(datasets)

    sensing = datasets["sensing"]
    sensing_basic_stats(sensing)
    plot_days_per_user(sensing)
    plot_sleep_distribution(sensing)
    plot_activity_summary(sensing)
    plot_hourly_unlock(sensing)

    print("=== EDA TAMAMLANDI ===\n")
