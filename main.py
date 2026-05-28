"""
Veri Seti — Tam Pipeline Runner
================================
Sensing + EMA + master dataset birleştirme dahil tüm akışı çalıştırır.

Kullanım:
    python main.py                          → tam pipeline (sensing + ema + master)
    python main.py --skip-eda               → EDA adımlarını atla (sensing + ema)
    python main.py --only-clean             → sadece sensing temizleme
    python main.py --only-sensing           → sadece sensing tarafı
    python main.py --only-ema               → sadece EMA tarafı (temiz sensing varsa)
    python main.py --only-master            → sadece master dataset birleştirme
    python main.py --with-correlation       → korelasyon analizini de ekle

Bağımsız modül çağırıları:
    python -m src.correlation_analysis
    python -m EMA.src.ema_eda
    python -m EMA.src.ema_cleaner
    python -m src.master_dataset
"""

import sys
import time


def banner(text: str) -> None:
    line = "═" * 60
    print(f"\n{line}\n  {text}\n{line}")


def sensing_pipeline(skip_eda: bool = False, only_clean: bool = False) -> None:
    """Sensing tarafı: yükle → EDA → eksik → aykırı → temizle → post-clean."""
    from src.data_loader import load_all
    from src.eda import run_eda
    from src.missing_analysis import run_missing_analysis
    from src.outlier_detector import run_outlier_detection
    from src.cleaner import run_cleaning
    from src.post_clean import run_post_clean

    banner("SENSING — ADIM 1: VERİ YÜKLEME")
    datasets = load_all()

    if not only_clean:
        if not skip_eda:
            banner("SENSING — ADIM 2: KEŞİFSEL ANALİZ (EDA)")
            run_eda(datasets)

        banner("SENSING — ADIM 3: EKSİK VERİ ANALİZİ")
        run_missing_analysis(datasets)

        banner("SENSING — ADIM 4: AYKIRI DEĞER TESPİTİ")
        run_outlier_detection(datasets)

    banner("SENSING — ADIM 5: VERİ TEMİZLEME")
    cleaned = run_cleaning(datasets)

    banner("SENSING — ADIM 6: BAĞLAMSAL SON TEMİZLEME")
    run_post_clean(cleaned)


def correlation_pipeline() -> None:
    """Pasif feature korelasyon + multicollinearity analizi."""
    from src.correlation_analysis import run_correlation_analysis

    banner("KORELASYON — Pasif Feature Multicollinearity + Boyutsal Analiz")
    run_correlation_analysis()


def ema_pipeline(skip_eda: bool = False) -> None:
    """EMA tarafı: EDA → temizleme."""
    from EMA.src.ema_cleaner import run_ema_cleaning

    if not skip_eda:
        from EMA.src.ema_eda import run_ema_eda
        banner("EMA — ADIM 1: KEŞİFSEL ANALİZ")
        run_ema_eda()

    banner("EMA — ADIM 2: TEMİZLEME")
    run_ema_cleaning()


def master_pipeline() -> None:
    """Master dataset — sensing + EMA + risk etiketleri."""
    from src.master_dataset import run_master_dataset

    banner("MASTER DATASET — Sensing ⊕ EMA ⊕ Risk Etiketleri")
    run_master_dataset()


def main() -> None:
    t0 = time.time()
    args = set(sys.argv[1:])

    skip_eda          = "--skip-eda"        in args
    only_clean        = "--only-clean"      in args
    only_sensing      = "--only-sensing"    in args
    only_ema          = "--only-ema"        in args
    only_master       = "--only-master"     in args
    with_correlation  = "--with-correlation" in args

    banner("VERİ SETİ — TAM PIPELINE")
    print(f"  Argümanlar: {sorted(args) if args else '(yok — tam pipeline)'}")

    if only_master:
        master_pipeline()
    elif only_ema:
        ema_pipeline(skip_eda=skip_eda)
    elif only_sensing or only_clean:
        sensing_pipeline(skip_eda=skip_eda, only_clean=only_clean)
    else:
        # Tam pipeline
        sensing_pipeline(skip_eda=skip_eda, only_clean=only_clean)
        if with_correlation:
            correlation_pipeline()
        ema_pipeline(skip_eda=skip_eda)
        master_pipeline()

    elapsed = time.time() - t0
    banner(f"TAMAMLANDI — {elapsed:.1f} saniye")
    print("\nÇıktılar:")
    print("  reports/                              → EDA grafikleri ve analiz raporları")
    print("  cleaned_data/sensing_cleaned.csv       → temiz sensing")
    print("  cleaned_data/ema_cleaned.csv           → temiz EMA")
    print("  cleaned_data/master_dataset.csv        → modelleme için birleşik tablo")
    print()


if __name__ == "__main__":
    main()
