import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PATHS, COLUMN_GROUPS

REPORT_DIR = PATHS["reports"]

def _group_of(col: str) -> str:
    for group, prefixes in COLUMN_GROUPS.items():
        if any(col.startswith(p) for p in prefixes):
            return group
    return "other"

# Sütun bazlı eksik analizi

def column_missing_report(df: pd.DataFrame, name: str = "sensing") -> pd.DataFrame:
    """Her sütun için eksik değer istatistiklerini hesapla."""
    total = len(df)
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / total * 100).round(2)

    report = pd.DataFrame({
        "sütun": df.columns,
        "eksik_adet": missing_count.values,
        "eksik_%": missing_pct.values,
        "dolu_adet": (total - missing_count).values,
        "veri_tipi": df.dtypes.astype(str).values,
        "grup": [_group_of(c) for c in df.columns],
    })
    report = report.sort_values("eksik_%", ascending=False).reset_index(drop=True)

    out = os.path.join(REPORT_DIR, f"09_{name}_column_missing.csv")
    report.to_csv(out, index=False)
    print(f"      → kaydedildi: reports/09_{name}_column_missing.csv")

    # Özet yazdır
    fully_missing = (report["eksik_%"] == 100).sum()
    high_missing  = ((report["eksik_%"] >= 50) & (report["eksik_%"] < 100)).sum()
    low_missing   = ((report["eksik_%"] > 0) & (report["eksik_%"] < 50)).sum()
    complete      = (report["eksik_%"] == 0).sum()
    print(f"\n  Sütun eksiklik özeti ({name}):")
    print(f"    Tamamen boş  (100%)   : {fully_missing:4d}")
    print(f"    Yüksek eksik (50-99%) : {high_missing:4d}")
    print(f"    Düşük eksik  (1-49%)  : {low_missing:4d}")
    print(f"    Tam dolu     (0%)     : {complete:4d}")

    return report

def group_missing_report(col_report: pd.DataFrame, name: str = "sensing") -> pd.DataFrame:
    """Grup bazlı ortalama eksiklik oranlarını hesapla."""
    group_report = (
        col_report.groupby("grup")["eksik_%"]
        .agg(["mean", "max", "min", "count"])
        .round(2)
        .rename(columns={"mean": "ort_eksik_%", "max": "maks_%",
                         "min": "min_%", "count": "sütun_sayısı"})
        .sort_values("ort_eksik_%", ascending=False)
    )
    out = os.path.join(REPORT_DIR, f"10_{name}_group_missing.csv")
    group_report.to_csv(out)
    print(f"      → kaydedildi: reports/10_{name}_group_missing.csv")
    print("\n  Grup bazlı ortalama eksiklik:\n", group_report.to_string())
    return group_report

def user_missing_report(df: pd.DataFrame, name: str = "sensing") -> pd.DataFrame:
    """Kullanıcı başına eksiklik oranlarını hesapla."""
    non_uid_cols = [c for c in df.columns if c != "uid"]
    user_missing = (
        df.groupby("uid")[non_uid_cols]
        .apply(lambda g: g.isnull().mean().mean() * 100)
        .rename("ort_eksik_%")
        .round(2)
        .reset_index()
        .sort_values("ort_eksik_%", ascending=False)
    )
    out = os.path.join(REPORT_DIR, f"11_{name}_user_missing.csv")
    user_missing.to_csv(out, index=False)
    print(f"      → kaydedildi: reports/11_{name}_user_missing.csv")

    print(f"\n  Kullanıcı başına eksiklik (ilk 10):\n",
          user_missing.head(10).to_string(index=False))
    return user_missing

# Görselleştirme

def plot_missing_heatmap(df: pd.DataFrame, name: str = "sensing") -> None:
    """Eksik değerlerin ısı haritası (yüksek eksikli sütunlar)."""
    high_missing_cols = df.columns[df.isnull().mean() > 0.3].tolist()
    if not high_missing_cols:
        print("      Isı haritası: >%30 eksik sütun yok, atlanıyor.")
        return
    # Max 60 sütun göster
    sample_cols = high_missing_cols[:60]
    missing_matrix = df[sample_cols].isnull().astype(int)

    fig, ax = plt.subplots(figsize=(min(len(sample_cols) * 0.4, 20), 6))
    sns.heatmap(
        missing_matrix.T,
        cmap="YlOrRd", cbar=False,
        xticklabels=False,
        yticklabels=[c[:30] for c in sample_cols],
        ax=ax,
    )
    ax.set_xlabel("Satırlar")
    ax.set_title(f"Eksik Değer Isı Haritası — {name} (>%30 eksik sütunlar, max 60)")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, f"12_{name}_missing_heatmap.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"      → kaydedildi: reports/12_{name}_missing_heatmap.png")

def plot_missing_by_group(group_report: pd.DataFrame, name: str = "sensing") -> None:
    """Grup bazlı ortalama eksiklik bar grafiği."""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#d62728" if v >= 50 else "#ff7f0e" if v >= 20 else "#1f77b4"
              for v in group_report["ort_eksik_%"]]
    ax.barh(group_report.index, group_report["ort_eksik_%"], color=colors)
    ax.axvline(50, color="red", linestyle="--", linewidth=0.8, label="%50 eşiği")
    ax.set_xlabel("Ortalama Eksiklik (%)")
    ax.set_title(f"Sütun Grubu Bazlı Ortalama Eksiklik — {name}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, f"13_{name}_missing_by_group.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"      → kaydedildi: reports/13_{name}_missing_by_group.png")

def plot_user_missing_distribution(user_report: pd.DataFrame, name: str = "sensing") -> None:
    """Kullanıcı başına eksiklik dağılımı histogramı."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(user_report["ort_eksik_%"], bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("Ortalama Eksiklik (%)")
    ax.set_ylabel("Kullanıcı Sayısı")
    ax.set_title(f"Kullanıcı Bazlı Eksiklik Dağılımı — {name}")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, f"14_{name}_user_missing_dist.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"      → kaydedildi: reports/14_{name}_user_missing_dist.png")

# Ana fonksiyon

def run_missing_analysis(datasets: dict[str, pd.DataFrame]) -> dict:
    """Sensing veri seti için eksik veri analizini çalıştır."""
    print("\n=== EKSİK VERİ ANALİZİ ===")
    os.makedirs(REPORT_DIR, exist_ok=True)
    results = {}

    sensing = datasets["sensing"]
    print("\n── sensing.csv ──")
    col_report   = column_missing_report(sensing, "sensing")
    group_report = group_missing_report(col_report, "sensing")
    user_report  = user_missing_report(sensing, "sensing")
    plot_missing_heatmap(sensing, "sensing")
    plot_missing_by_group(group_report, "sensing")
    plot_user_missing_distribution(user_report, "sensing")
    results["sensing"] = {"col": col_report, "group": group_report, "user": user_report}

    print("\n=== EKSİK VERİ ANALİZİ TAMAMLANDI ===\n")
    return results
