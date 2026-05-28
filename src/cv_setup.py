"""
Cross-Validation Setup Modülü — Modele Son Hazırlık
Bağımsız çalışır:

    python -m src.cv_setup

Yapılan işlemler:
  1. features_core_clean.csv yükle (target etiketleri için)
  2. Her kullanıcının baskın final_risk_4 sınıfını hesapla
  3. STRATIFIED HOLD-OUT split — 22 kullanıcı (%10) test setine ayır
  4. CV pool (195 kullanıcı) üzerinde 5-fold StratifiedGroupKFold
  5. Hiçbir kullanıcının fold'lar arası çakışmadığını doğrula
  6. Her fold için class dağılımını raporla
  7. cv_splits.json olarak kaydet (modelleme aşamasında okunacak)

Çıktılar:
  cleaned_data/cv_splits.json          — fold tanımları (model eğitiminde kullanılır)
  reports/50_cv_fold_distribution.csv  — her fold'un class dağılımı
  reports/51_holdout_summary.csv       — hold-out set istatistikleri
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.config import PATHS

# Yollar ve sabitler

CORE_PATH       = os.path.join(PATHS["cleaned_data"], "features_core_clean.csv")
SPLITS_OUT      = os.path.join(PATHS["cleaned_data"], "cv_splits.json")
REPORTS         = PATHS["reports"]

# Reproducibility
RANDOM_STATE    = 42

# Bölme parametreleri
HOLDOUT_RATIO   = 0.10   # %10 hold-out test
N_FOLDS         = 5      # 5-fold CV (LOSO yerine — hızlı)
HEDEF_KOLON     = "final_risk_4"

# 1. Kullanıcı baskın sınıfı

def kullanici_baskin_sinif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her kullanıcı için en yaygın final_risk_4 sınıfını hesapla.
    Stratified bölme bu sınıfa göre yapılacak — kullanıcı genel risk profilini
    temsil eder.
    """
    print("\n── 1. Kullanıcı baskın sınıfı hesaplanıyor ──")
    baskin = (df.groupby("uid")[HEDEF_KOLON]
              .agg(lambda x: int(x.mode().iloc[0]))
              .reset_index()
              .rename(columns={HEDEF_KOLON: "baskin_sinif"}))
    dag = baskin["baskin_sinif"].value_counts().sort_index()
    print(f"  Toplam kullanıcı: {len(baskin)}")
    print(f"  Baskın sınıf dağılımı:")
    for k, v in dag.items():
        print(f"    Sınıf {k}: {v} kullanıcı (%{v/len(baskin)*100:.1f})")
    return baskin

# 2. Stratified hold-out split

def holdout_split(baskin_df: pd.DataFrame,
                  oran: float = HOLDOUT_RATIO,
                  seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train_test_split — kullanıcı bazında, baskın sınıfa göre dengeli.
    Returns:
        cv_pool_df: CV için kalan kullanıcılar
        holdout_df: tamamen ayrılan test kullanıcıları
    """
    print(f"\n── 2. Hold-out split — %{oran*100:.0f} test, stratified ──")

    cv_pool, holdout = train_test_split(
        baskin_df,
        test_size=oran,
        stratify=baskin_df["baskin_sinif"],
        random_state=seed,
    )
    print(f"  Hold-out kullanıcı: {len(holdout)}")
    print(f"  CV pool kullanıcı:  {len(cv_pool)}")

    print(f"\n  Hold-out class dağılımı:")
    for k, v in holdout["baskin_sinif"].value_counts().sort_index().items():
        print(f"    Sınıf {k}: {v} kullanıcı (%{v/len(holdout)*100:.1f})")

    return cv_pool.reset_index(drop=True), holdout.reset_index(drop=True)

# 3. 5-fold StratifiedGroupKFold (row-level class dengeli + uid grup korumalı)

def build_folds(df: pd.DataFrame,
                cv_pool_uids: list[str],
                n_folds: int = N_FOLDS,
                seed: int = RANDOM_STATE) -> list[dict]:
    """
    CV pool satırları üzerinde StratifiedGroupKFold.
    - groups=uid: aynı kullanıcının satırları tek fold'a düşer
    - stratify=final_risk_4: row-level class dağılımı fold'lar arası dengeli

    Standart StratifiedKFold'dan farkı: o user-majority'ye göre dengeli,
    bu ise row-level final_risk_4'e göre dengeli — Yüksek Risk satırlarının
    fold'lar arası varyansını minimize eder.
    """
    print(f"\n── 3. {n_folds}-fold StratifiedGroupKFold (row-level + group constraint) ──")

    pool_df = df[df["uid"].isin(cv_pool_uids)].copy()
    print(f"  CV pool satır sayısı: {len(pool_df):,}")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_list: list[dict] = []

    for fold_id, (tr_idx, va_idx) in enumerate(
        sgkf.split(pool_df, pool_df[HEDEF_KOLON], groups=pool_df["uid"])
    ):
        train_uids = pool_df.iloc[tr_idx]["uid"].unique().tolist()
        val_uids   = pool_df.iloc[va_idx]["uid"].unique().tolist()
        val_class_dist = (
            pool_df.iloc[va_idx][HEDEF_KOLON]
            .value_counts(normalize=True)
            .round(3)
            .to_dict()
        )
        fold_list.append({
            "fold_id":          fold_id,
            "train_uid_sayisi": len(train_uids),
            "val_uid_sayisi":   len(val_uids),
            "train_uids":       train_uids,
            "val_uids":         val_uids,
            "val_class_dist":   val_class_dist,
        })
        print(f"  Fold {fold_id}: train uid={len(train_uids):3d}, val uid={len(val_uids):3d}, "
              f"val class dist={val_class_dist}")
    return fold_list

# 4. Doğrulama — kullanıcı çakışması yok

def verify_no_overlap(holdout: pd.DataFrame,
                      cv_pool: pd.DataFrame,
                      folds: list[dict]) -> None:
    """Holdout ↔ CV pool ve fold'lar arası kullanıcı çakışması olmadığını doğrula."""
    print("\n── 4. Çakışma kontrolü ──")

    holdout_set = set(holdout["uid"])
    cv_set      = set(cv_pool["uid"])
    cakisma     = holdout_set & cv_set
    if cakisma:
        raise ValueError(f"Hold-out ↔ CV pool çakışması: {cakisma}")
    print(f"   Hold-out ↔ CV pool: çakışma yok")

    # Her kullanıcı tam 1 fold'un val setinde olmalı
    val_sayisi = {uid: 0 for uid in cv_set}
    for fold in folds:
        for uid in fold["val_uids"]:
            val_sayisi[uid] += 1

    yanlis = [uid for uid, n in val_sayisi.items() if n != 1]
    if yanlis:
        raise ValueError(f"Bazı kullanıcılar tek fold'da val değil: {yanlis[:5]}")
    print(f"   Her kullanıcı tam 1 fold'un val setinde")

    # Her fold'da train val ayrık mı
    for fold in folds:
        t = set(fold["train_uids"])
        v = set(fold["val_uids"])
        if t & v:
            raise ValueError(f"Fold {fold['fold_id']}: train ↔ val çakışıyor")
    print(f"   Her fold'da train ↔ val ayrık")

# 5. Satır bazında istatistik

def per_fold_row_stats(df: pd.DataFrame, folds: list[dict],
                       holdout_uids: list[str]) -> pd.DataFrame:
    """Her fold ve hold-out için satır bazında class dağılımı."""
    print("\n── 5. Satır bazında class dağılımı ──")
    rows = []

    holdout_df = df[df["uid"].isin(holdout_uids)]
    dag = holdout_df[HEDEF_KOLON].value_counts(normalize=True).sort_index().to_dict()
    rows.append({
        "set":           "holdout",
        "fold_id":       -1,
        "uid_sayisi":    len(holdout_uids),
        "satir_sayisi":  len(holdout_df),
        **{f"sinif_{k}_yuzde": round(v*100, 2) for k, v in dag.items()},
    })

    for fold in folds:
        for adi, uid_list in [("train", fold["train_uids"]), ("val", fold["val_uids"])]:
            alt = df[df["uid"].isin(uid_list)]
            dag = alt[HEDEF_KOLON].value_counts(normalize=True).sort_index().to_dict()
            rows.append({
                "set":           adi,
                "fold_id":       fold["fold_id"],
                "uid_sayisi":    len(uid_list),
                "satir_sayisi":  len(alt),
                **{f"sinif_{k}_yuzde": round(v*100, 2) for k, v in dag.items()},
            })

    tablo = pd.DataFrame(rows)
    print(tablo.to_string(index=False))
    return tablo

# Ana akış

def run_cv_setup() -> None:
    """CV setup pipeline'ı."""
    print("\n" + "═" * 60)
    print("  CV SETUP — Modele Son Hazırlık")
    print("═" * 60)

    if not os.path.exists(CORE_PATH):
        raise FileNotFoundError(
            f"features_core_clean.csv yok: {CORE_PATH}\n"
            f"Önce 'python -m src.data_quality' çalıştır."
        )

    os.makedirs(REPORTS, exist_ok=True)

    print(f"\n  Yükleniyor: {CORE_PATH}")
    df = pd.read_csv(CORE_PATH, low_memory=False)
    print(f"  Boyut: {df.shape}, kullanıcı: {df['uid'].nunique()}")

    # 1. Baskın sınıf
    baskin = kullanici_baskin_sinif(df)

    # 2. Hold-out
    cv_pool, holdout = holdout_split(baskin)

    # 3. Folds — row-level StratifiedGroupKFold
    folds = build_folds(df, cv_pool["uid"].tolist())

    # 4. Doğrula
    verify_no_overlap(holdout, cv_pool, folds)

    # 5. Satır istatistiği
    row_stats = per_fold_row_stats(df, folds, holdout["uid"].tolist())

    # 6. JSON kaydet
    print("\n── 6. cv_splits.json kaydediliyor ──")
    splits_data = {
        "meta": {
            "random_state":  RANDOM_STATE,
            "n_folds":       N_FOLDS,
            "holdout_ratio": HOLDOUT_RATIO,
            "stratify_by":   "kullanici_baskin_final_risk_4",
            "toplam_kullanici": int(df["uid"].nunique()),
        },
        "holdout_uids": holdout["uid"].tolist(),
        "cv_pool_uids": cv_pool["uid"].tolist(),
        "folds": [
            {
                "fold_id":     f["fold_id"],
                "train_uids":  f["train_uids"],
                "val_uids":    f["val_uids"],
            }
            for f in folds
        ],
    }
    with open(SPLITS_OUT, "w") as f:
        json.dump(splits_data, f, indent=2)
    print(f"   {SPLITS_OUT}")

    # 7. Raporlar
    row_stats.to_csv(os.path.join(REPORTS, "50_cv_fold_distribution.csv"), index=False)
    print(f"      → reports/50_cv_fold_distribution.csv")

    # Hold-out özet
    holdout_summary = pd.DataFrame([{
        "kullanici_sayisi":       len(holdout),
        "satir_sayisi":           int(df[df["uid"].isin(holdout["uid"])].shape[0]),
        **{f"baskin_sinif_{k}":   int(v)
           for k, v in holdout["baskin_sinif"].value_counts().sort_index().items()},
    }])
    holdout_summary.to_csv(os.path.join(REPORTS, "51_holdout_summary.csv"), index=False)
    print(f"      → reports/51_holdout_summary.csv")

    print("\n" + "═" * 60)
    print("  CV SETUP TAMAMLANDI — Modelleme için her şey hazır")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    run_cv_setup()
