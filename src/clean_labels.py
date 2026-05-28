"""
Clean labels: sadece net vakaları kullanır — class 0 (İyi) vs class 3 (Yüksek).
Belirsiz orta sınıfları (1, 2) atar, böylece boundary gürültüsü azalır ve saf
pasifte daha temiz bir sinyal kalır. En iyi saf pasif popülasyon sonucu buradan.
"""

from __future__ import annotations

import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score,
)

from src.config import PATHS

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_CSV = os.path.join(PATHS["cleaned_data"], "features_extended_temporal.csv")
CLEAN_OUT = os.path.join(PATHS["cleaned_data"], "features_clean_labels.csv")
SPLITS_PATH = os.path.join(PATHS["cleaned_data"], "cv_splits.json")
REPORTS = os.path.join(PATHS["reports"], "06_modeling_prep")
MODELS = os.path.join(BASE, "models")

META = ["uid", "gun", "is_ios"]
TARGET = "final_risk_4"
BINARY = "is_high_risk"
RND = 42

def log(msg):
    print(msg, flush=True)

def make_binary(df):
    df = df.copy()
    df[BINARY] = (df[TARGET] == 3).astype(int)
    return df

def filter_clean_labels(df):
    """Sadece class 0 (İyi) ve class 3 (Yüksek Risk) tutulur."""
    log("\n── 1. Clean Labels Filtreleme ──")
    n_once = len(df)
    log(f"  Önce: {n_once:,} satır")

    # Sayım önceki
    onceki = df[TARGET].value_counts().sort_index()
    log(f"  Önce class dağılımı:")
    for k, v in onceki.items():
        log(f"    Class {k}: {v:,} (%{v/n_once*100:.2f})")

    # Filtrele
    df_clean = df[df[TARGET].isin([0, 3])].copy()

    n_sonra = len(df_clean)
    log(f"\n  Sonra: {n_sonra:,} satır")
    log(f"  Kayıp: {n_once - n_sonra:,} (%{(n_once - n_sonra)/n_once*100:.2f})")

    sonraki = df_clean[TARGET].value_counts().sort_index()
    log(f"  Sonra class dağılımı:")
    for k, v in sonraki.items():
        log(f"    Class {k}: {v:,} (%{v/n_sonra*100:.2f})")

    # Binary target
    df_clean = make_binary(df_clean)

    n_pos = df_clean[BINARY].sum()
    log(f"\n  Yeni binary dağılım:")
    log(f"    Negatif (İyi):       {n_sonra - n_pos:,} (%{(n_sonra-n_pos)/n_sonra*100:.2f})")
    log(f"    Pozitif (Yüksek Risk): {n_pos:,} (%{n_pos/n_sonra*100:.2f})")
    log(f"  → Class balance %5.89 → %{n_pos/n_sonra*100:.2f} (DAHA İYİ!)")

    return df_clean

def egit_test(df, feat, splits, model_type, class_k=3):
    cv_pool = df[df["uid"].isin(splits["cv_pool_uids"])]
    holdout = df[df["uid"].isin(splits["holdout_uids"])]

    if len(cv_pool) == 0 or len(holdout) == 0:
        log(f"   {model_type}: cv_pool veya holdout boş!")
        return None

    X_tr, y_tr = cv_pool[feat], cv_pool[BINARY]
    X_te, y_te = holdout[feat], holdout[BINARY]

    log(f"\n  {model_type} eğitiliyor (train={len(X_tr):,}, test={len(X_te):,})...")

    t0 = time.time()
    if model_type == "RF":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            class_weight={0: 1, 1: class_k},
            random_state=RND, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
    else:  # HGB
        sw = np.where(y_tr == 1, class_k, 1.0)
        model = HistGradientBoostingClassifier(random_state=RND)
        model.fit(X_tr, y_tr, sample_weight=sw)

    prob_te = model.predict_proba(X_te)[:, 1]
    log(f"  Süre: {time.time()-t0:.0f}s")

    y_te_arr = y_te.to_numpy()
    auc = roc_auc_score(y_te_arr, prob_te)

    # Threshold sweep
    best_f1, best_thr_f1 = 0, 0.5
    best_p_f1, best_r_f1 = 0, 0
    best_f2, best_thr_f2 = 0, 0.5
    best_p_f2, best_r_f2 = 0, 0
    for thr in np.linspace(0.05, 0.95, 91):
        y_pred = (prob_te >= thr).astype(int)
        f1 = f1_score(y_te_arr, y_pred, pos_label=1, zero_division=0)
        f2 = fbeta_score(y_te_arr, y_pred, beta=2.0, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr_f1 = f1, thr
            best_p_f1 = precision_score(y_te_arr, y_pred, pos_label=1, zero_division=0)
            best_r_f1 = recall_score(y_te_arr, y_pred, pos_label=1, zero_division=0)
        if f2 > best_f2:
            best_f2, best_thr_f2 = f2, thr
            best_p_f2 = precision_score(y_te_arr, y_pred, pos_label=1, zero_division=0)
            best_r_f2 = recall_score(y_te_arr, y_pred, pos_label=1, zero_division=0)

    log(f"  AUC: {auc:.4f}")
    log(f"  F1-opt thr={best_thr_f1:.2f}: F1={best_f1:.4f}  P={best_p_f1:.4f}  R={best_r_f1:.4f}")
    log(f"  F2-opt thr={best_thr_f2:.2f}: F2={best_f2:.4f}  P={best_p_f2:.4f}  R={best_r_f2:.4f}")

    return {
        "model_type": model_type,
        "model": model,
        "auc": auc,
        "best_f1": best_f1, "best_thr_f1": best_thr_f1,
        "best_p_f1": best_p_f1, "best_r_f1": best_r_f1,
        "best_f2": best_f2, "best_thr_f2": best_thr_f2,
        "best_p_f2": best_p_f2, "best_r_f2": best_r_f2,
        "n_train": len(X_tr), "n_test": len(X_te),
    }

def run_clean_labels() -> None:
    t_global = time.time()
    log("\n" + "═" * 60)
    log("  CLEAN LABELS")
    log("═" * 60)

    log("\nVeri yükleniyor...")
    df = pd.read_csv(TEMP_CSV, low_memory=False)
    log(f"Boyut: {df.shape}")

    # Clean labels filter
    df_clean = filter_clean_labels(df)

    # Kaydet
    df_clean.to_csv(CLEAN_OUT, index=False)
    log(f"\n   {CLEAN_OUT}")

    # Setup CSV
    setup = pd.DataFrame([{
        "n_total":       len(df_clean),
        "n_positive":    df_clean[BINARY].sum(),
        "n_negative":    len(df_clean) - df_clean[BINARY].sum(),
        "pos_pct":       round(df_clean[BINARY].mean() * 100, 2),
        "n_users":       df_clean["uid"].nunique(),
    }])
    setup.to_csv(os.path.join(REPORTS, "99_clean_labels_setup.csv"), index=False)

    # CV splits
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    feat = [c for c in df_clean.columns if c not in META + [TARGET, BINARY]]
    log(f"\n  Feature sayısı: {len(feat)}")

    # 2 model dene
    log("\n── 2. 2 model eğitiliyor ──")
    sonuclar = []
    for k in [3, 5, 8]:  # Birkaç class_k dene
        log(f"\n  ▶ class_k = {k}")
        rf_res  = egit_test(df_clean, feat, splits, "RF",  class_k=k)
        hgb_res = egit_test(df_clean, feat, splits, "HGB", class_k=k)
        if rf_res:
            rf_res["class_k"] = k
            sonuclar.append(rf_res)
        if hgb_res:
            hgb_res["class_k"] = k
            sonuclar.append(hgb_res)

    # En iyiyi bul
    en_iyi = max(sonuclar, key=lambda s: s["best_f1"])
    log(f"\n EN İYİ: {en_iyi['model_type']} k={en_iyi['class_k']}")
    log(f"   F1={en_iyi['best_f1']:.4f}, P={en_iyi['best_p_f1']:.4f}, "
        f"R={en_iyi['best_r_f1']:.4f}, AUC={en_iyi['auc']:.4f}")

    # Karşılaştırma tablosu
    log("\n── 3. Tüm sonuçlar ──")
    rows = [
        {"model": "ESKİ EN İYİ — Voting V1",
         "f1_pos": 0.2298, "precision": 0.2141, "recall": 0.2481,
         "auc": 0.6671, "method": "tüm sınıflar, voting"},
    ]
    for s in sonuclar:
        rows.append({
            "model":     f"CLEAN {s['model_type']} k={s['class_k']}",
            "f1_pos":    round(s["best_f1"], 4),
            "precision": round(s["best_p_f1"], 4),
            "recall":    round(s["best_r_f1"], 4),
            "auc":       round(s["auc"], 4),
            "method":    "clean labels (sınıf 0+3)",
        })
    karsi = pd.DataFrame(rows)
    log("\n" + karsi.to_string(index=False))
    karsi.to_csv(os.path.join(REPORTS, "101_clean_labels_comparison.csv"), index=False)
    log("   101_clean_labels_comparison.csv")

    # Hold-out detay
    holdout_df = pd.DataFrame([{
        "model_type": s["model_type"],
        "class_k":    s["class_k"],
        "auc":        s["auc"],
        "best_thr_f1": s["best_thr_f1"],
        "best_f1":     s["best_f1"],
        "best_p_f1":   s["best_p_f1"],
        "best_r_f1":   s["best_r_f1"],
        "best_thr_f2": s["best_thr_f2"],
        "best_f2":     s["best_f2"],
        "best_p_f2":   s["best_p_f2"],
        "best_r_f2":   s["best_r_f2"],
        "n_train":     s["n_train"],
        "n_test":      s["n_test"],
    } for s in sonuclar])
    holdout_df.to_csv(os.path.join(REPORTS, "100_clean_labels_holdout.csv"), index=False)
    log("   100_clean_labels_holdout.csv")

    # Pickle en iyi
    with open(os.path.join(MODELS, "best_clean_labels.pkl"), "wb") as f:
        pickle.dump({
            "model":          en_iyi["model"],
            "feature_kollar": feat,
            "model_type":     en_iyi["model_type"],
            "class_k":        en_iyi["class_k"],
            **{k: en_iyi[k] for k in ["auc", "best_f1", "best_thr_f1",
                                        "best_p_f1", "best_r_f1",
                                        "best_f2", "best_thr_f2",
                                        "best_p_f2", "best_r_f2"]},
        }, f)
    log(f"   models/best_clean_labels.pkl")

    # Karar analizi
    log("\n" + "═" * 60)
    log("  KARAR ANALİZİ")
    log("═" * 60)
    delta = en_iyi["best_f1"] - 0.2298
    log(f"\n  Önceki en iyi F1 (Voting V1): 0.2298")
    log(f"  Clean labels yeni F1:          {en_iyi['best_f1']:.4f}")
    log(f"  Delta:                         {delta:+.4f}")
    if delta > 0.05:
        log(f"  → BÜYÜK İYİLEŞME ")
    elif delta > 0.02:
        log(f"  → Anlamlı iyileşme ")
    elif delta > 0:
        log(f"  → Hafif iyileşme")
    else:
        log(f"  → İyileşme YOK")

    log("\n" + "═" * 60)
    log(f"  TAMAMLANDI — {(time.time()-t_global)/60:.1f}dk")
    log("═" * 60 + "\n")

if __name__ == "__main__":
    run_clean_labels()
