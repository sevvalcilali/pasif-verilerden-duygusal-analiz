import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import PATHS, NON_NEGATIVE_PREFIXES

REPORT_DIR = PATHS["reports"]

# Alan bilgisi kontrolleri

def check_negative_values(df: pd.DataFrame, name: str = "sensing") -> pd.DataFrame:
    """Negatif olmaması gereken sütunlardaki negatif değerleri bul."""
    num_cols = df.select_dtypes(include="number").columns
    neg_cols = [c for c in num_cols
                if any(c.startswith(p) for p in NON_NEGATIVE_PREFIXES)]

    rows = []
    for col in neg_cols:
        neg_mask = df[col] < 0
        count = neg_mask.sum()
        if count > 0:
            rows.append({
                "sütun": col,
                "negatif_adet": int(count),
                "negatif_%": round(count / len(df) * 100, 4),
                "min_değer": df.loc[neg_mask, col].min(),
            })

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values("negatif_adet", ascending=False)
    if report.empty:
        print(f"  {name}: negatif değer bulunamadı.")
    else:
        print(f"\n  {name} — negatif değerli sütunlar ({len(report)}):")
        print(report.to_string(index=False))
        out = os.path.join(REPORT_DIR, f"16_{name}_negative_values.csv")
        report.to_csv(out, index=False)
        print(f"      → kaydedildi: reports/16_{name}_negative_values.csv")
    return report

def check_duration_limits(df: pd.DataFrame, name: str = "sensing") -> pd.DataFrame:
    """
    Süre sütunlarında fiziksel sınır kontrolü:
    Günlük süre sütunları (_ep_0) 86400 saniyeyi (1 gün) aşamaz.
    """
    # Mesafe sütunları (loc_dist_, loc_max_dis_) hariç — bunlar saniye değil metre
    dur_cols = [c for c in df.columns
                if c.endswith("_ep_0") and
                any(c.startswith(p) for p in ["act_", "unlock_duration_", "loc_"]) and
                not any(c.startswith(p) for p in ["loc_dist_", "loc_max_dis_"])]
    rows = []
    for col in dur_cols:
        over = (df[col] > 86400).sum()
        if over > 0:
            rows.append({
                "sütun": col,
                "aşım_adet": int(over),
                "aşım_%": round(over / len(df) * 100, 4),
                "max_değer": df[col].max(),
            })
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values("aşım_adet", ascending=False)
    if report.empty:
        print(f"  {name}: gün sınırı (86400s) aşımı bulunamadı.")
    else:
        print(f"\n  {name} — gün sınırı aşımları ({len(report)} sütun):")
        print(report.to_string(index=False))
        out = os.path.join(REPORT_DIR, f"17_{name}_duration_limit.csv")
        report.to_csv(out, index=False)
        print(f"      → kaydedildi: reports/17_{name}_duration_limit.csv")
    return report

def check_sleep_sanity(df: pd.DataFrame) -> pd.DataFrame:
    """Uyku tutarsızlıklarını kontrol et."""
    rows = []
    if "sleep_duration" in df.columns:
        # Negatif uyku
        neg = (df["sleep_duration"] < 0).sum()
        # 24 saatten uzun uyku
        over24 = (df["sleep_duration"] > 86400).sum()
        # 20 saatten uzun uyku (şüpheli)
        over20 = ((df["sleep_duration"] > 72000) &
                  (df["sleep_duration"] <= 86400)).sum()
        rows = [
            {"kontrol": "negatif uyku süresi",        "adet": int(neg)},
            {"kontrol": "24 saat+ uyku",               "adet": int(over24)},
            {"kontrol": "20-24 saat arası (şüpheli)",  "adet": int(over20)},
        ]
    report = pd.DataFrame(rows)
    print("\n  Uyku tutarsızlık kontrolü:")
    print(report.to_string(index=False))
    out = os.path.join(REPORT_DIR, "18_sleep_sanity.csv")
    report.to_csv(out, index=False)
    print(f"      → kaydedildi: reports/18_sleep_sanity.csv")
    return report

# İstatistiksel aykırı değer tespiti (IQR)

def iqr_outliers(df: pd.DataFrame,
                 cols: list[str],
                 factor: float = 3.0) -> pd.DataFrame:
    """
    IQR yöntemiyle (factor * IQR) aykırı değer oranını hesapla.
    Mobilite/sensör verisi için factor=3 kullanılır (daha az agresif).
    """
    rows = []
    for col in cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        out_count = ((series < lower) | (series > upper)).sum()
        if out_count > 0:
            rows.append({
                "sütun": col,
                "aykırı_adet": int(out_count),
                "aykırı_%": round(out_count / len(series) * 100, 3),
                "alt_sınır": round(lower, 2),
                "üst_sınır": round(upper, 2),
                "min": round(series.min(), 2),
                "max": round(series.max(), 2),
            })
    return pd.DataFrame(rows).sort_values("aykırı_%", ascending=False)

def run_iqr_analysis(df: pd.DataFrame, name: str = "sensing") -> pd.DataFrame:
    """Sayısal sütunlara IQR analizi uygula."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    # uid ve day hariç tut
    num_cols = [c for c in num_cols if c not in ("is_ios",)]
    report = iqr_outliers(df, num_cols, factor=3.0)

    if report.empty:
        print(f"  {name}: IQR aykırı değer bulunamadı.")
    else:
        print(f"\n  {name} — IQR aykırı değerli sütun sayısı: {len(report)}")
        print(f"  En yüksek 10 aykırı sütun:")
        print(report.head(10).to_string(index=False))
        out = os.path.join(REPORT_DIR, f"20_{name}_iqr_outliers.csv")
        report.to_csv(out, index=False)
        print(f"      → kaydedildi: reports/20_{name}_iqr_outliers.csv")
    return report

# Görselleştirme

def plot_outlier_summary(iqr_report: pd.DataFrame, name: str = "sensing") -> None:
    """Aykırı değer yüzdesine göre top-20 sütun bar grafiği."""
    if iqr_report.empty:
        return
    top = iqr_report.head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top["sütun"].str[-35:], top["aykırı_%"], color="tomato")
    ax.set_xlabel("Aykırı Değer Oranı (%)")
    ax.set_title(f"Top-20 Aykırı Değer Sütunu — {name}")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, f"21_{name}_outlier_top20.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"      → kaydedildi: reports/21_{name}_outlier_top20.png")

# Ana fonksiyon

def run_outlier_detection(datasets: dict[str, pd.DataFrame]) -> dict:
    """Tüm veri setleri için aykırı değer analizini çalıştır."""
    print("\n=== AYKIRI DEĞER TESPİTİ ===")
    os.makedirs(REPORT_DIR, exist_ok=True)
    results = {}

    sensing = datasets["sensing"]
    print("\n── sensing.csv ──")
    results["sensing_negative"] = check_negative_values(sensing, "sensing")
    results["sensing_duration"] = check_duration_limits(sensing, "sensing")
    results["sensing_sleep"]    = check_sleep_sanity(sensing)
    iqr_report = run_iqr_analysis(sensing, "sensing")
    results["sensing_iqr"] = iqr_report
    plot_outlier_summary(iqr_report, "sensing")

    print("\n=== AYKIRI DEĞER TESPİTİ TAMAMLANDI ===\n")
    return results
