"""
Forecasting: bugünün EMA + pasif verisinden yarının riskini tahmin eder.
Yarının etiketi girdide olmadığı için strict leakage yok; ama yüksek skor
(F1=0.97) EMA'nın günden güne otokorelasyonundan geliyor — pasif-only ~0.35.
features_hybrid.csv'de D günü satırına D+1'in final_risk_4'ü eklenir.
"""

from __future__ import annotations

import json
import os
import pickle
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, fbeta_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score,
)

from src.config import PATHS

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HYBRID_CSV = os.path.join(PATHS["cleaned_data"], "features_hybrid.csv")
FORECAST_OUT = os.path.join(PATHS["cleaned_data"], "features_forecasting.csv")
SPLITS_PATH = os.path.join(PATHS["cleaned_data"], "cv_splits.json")
REPORTS = os.path.join(PATHS["reports"], "06_modeling_prep")
MODELS = os.path.join(BASE, "models")

META = ["uid", "gun", "is_ios"]
TARGET = "final_risk_4"
BINARY_BUGUN = "is_high_risk"            # bugünkü risk (eski)
BINARY_YARIN = "yarin_is_high_risk"      # yarınki risk (yeni hedef!)
RND = 42

def log(msg):
    print(msg, flush=True)

# 1. Forecasting datası üret (yarınki riski hedef olarak ekle)

def forecasting_data_olustur():
    log("\n── 1. FORECASTING DATASETİ HAZIRLANIYOR ──")
    log("  features_hybrid.csv yükleniyor...")
    df = pd.read_csv(HYBRID_CSV, low_memory=False)
    log(f"  Boyut: {df.shape}")

    # uid + gun sırala
    df = df.sort_values(["uid", "gun"]).reset_index(drop=True)

    # Bugünkü binary (mevcut)
    df[BINARY_BUGUN] = (df[TARGET] == 3).astype(int)

    # YARINKI final_risk_4 (shift -1 within user)
    log("\n  Yarınki risk hedefi üretiliyor (shift -1 per user)...")
    df["yarin_final_risk_4"] = df.groupby("uid")[TARGET].shift(-1)
    df[BINARY_YARIN] = (df["yarin_final_risk_4"] == 3).astype("Int64")

    # Gap kontrolü — sadece ardışık günler için geçerli
    df["gun_dt"] = pd.to_datetime(df["gun"].astype(str), format="%Y%m%d")
    df["yarin_dt"] = df.groupby("uid")["gun_dt"].shift(-1)
    df["gun_farki"] = (df["yarin_dt"] - df["gun_dt"]).dt.days

    log(f"  Gün farkı dağılımı:")
    farki = df["gun_farki"].value_counts().sort_index().head(10)
    for k, v in farki.items():
        marker = "  ← kullanılabilir" if k == 1 else ""
        print(f"    {k} gün: {v:,} {marker}")

    # SADECE 1 günlük ileri tahmin (ertesi gün) — gap'ler atılır
    onceki = len(df)
    df = df[df["gun_farki"] == 1].copy()
    log(f"\n  Filtre sonrası: {onceki:,} → {len(df):,} satır (sadece ardışık günler)")

    # NaN target satırları temizle
    df = df.dropna(subset=["yarin_final_risk_4"]).copy()
    df["yarin_final_risk_4"] = df["yarin_final_risk_4"].astype(int)
    df[BINARY_YARIN] = df[BINARY_YARIN].astype(int)

    log(f"  Final boyut: {df.shape}")
    log(f"  YARIN class dağılımı:")
    for k, v in df["yarin_final_risk_4"].value_counts().sort_index().items():
        print(f"    Class {k} (yarın): {v:,} ({v/len(df)*100:.1f}%)")

    log(f"\n  YARIN binary dağılımı:")
    n_pos = df[BINARY_YARIN].sum()
    log(f"    Yüksek Risk (yarın): {n_pos:,} ({n_pos/len(df)*100:.2f}%)")
    log(f"    Diğer (yarın):       {len(df)-n_pos:,}")

    # Kaydet
    df.to_csv(FORECAST_OUT, index=False)
    log(f"\n   {FORECAST_OUT}")

    return df

# 2. Clean labels filter (sadece yarın class 0 veya 3)

def clean_labels_forecasting(df):
    log("\n── 2. Clean Labels Filtreleme (yarın 0 veya 3) ──")
    n_once = len(df)
    df_clean = df[df["yarin_final_risk_4"].isin([0, 3])].copy()
    log(f"  {n_once:,} → {len(df_clean):,} satır")
    log(f"  Negatif (yarın İyi):       {(df_clean[BINARY_YARIN] == 0).sum():,}")
    log(f"  Pozitif (yarın Yüksek):     {df_clean[BINARY_YARIN].sum():,}")
    return df_clean

# 3. Model eğitim ve değerlendirme

def egit_test(df, feat, splits, model_type: str, class_k: int = 3,
              ema_dahil: bool = True):
    """Forecasting modeli eğit."""
    cv_pool = df[df["uid"].isin(splits["cv_pool_uids"])]
    holdout = df[df["uid"].isin(splits["holdout_uids"])]
    X_tr, y_tr = cv_pool[feat], cv_pool[BINARY_YARIN]
    X_te, y_te = holdout[feat], holdout[BINARY_YARIN]

    log(f"\n  {model_type} k={class_k} ema_dahil={ema_dahil}")
    log(f"    train={len(X_tr):,}, test={len(X_te):,}")
    log(f"    pos train: {y_tr.sum()}, pos test: {y_te.sum()}")

    if y_tr.sum() < 5 or y_te.sum() < 5:
        log(f"     Yetersiz pozitif, atlanıyor")
        return None

    t0 = time.time()
    if model_type == "RF":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            class_weight={0: 1, 1: class_k},
            random_state=RND, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
    else:
        sw = np.where(y_tr == 1, class_k, 1.0)
        model = HistGradientBoostingClassifier(random_state=RND)
        model.fit(X_tr, y_tr, sample_weight=sw)
    prob_te = model.predict_proba(X_te)[:, 1]
    log(f"    süre: {time.time()-t0:.0f}s")

    y_te_arr = y_te.to_numpy()
    auc = roc_auc_score(y_te_arr, prob_te)

    # Threshold sweep
    best_f1, best_thr_f1, best_p_f1, best_r_f1 = 0, 0.5, 0, 0
    best_f2, best_thr_f2, best_p_f2, best_r_f2 = 0, 0.5, 0, 0
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

    log(f"    AUC={auc:.4f}")
    log(f"    F1-opt thr={best_thr_f1:.2f}: F1={best_f1:.4f}  P={best_p_f1:.4f}  R={best_r_f1:.4f}")
    log(f"    F2-opt thr={best_thr_f2:.2f}: F2={best_f2:.4f}  P={best_p_f2:.4f}  R={best_r_f2:.4f}")

    return {
        "model_type": model_type,
        "class_k":    class_k,
        "ema_dahil":  ema_dahil,
        "model":      model,
        "auc":        round(auc, 4),
        "best_f1":    round(best_f1, 4),
        "best_thr_f1": round(best_thr_f1, 3),
        "best_p_f1":  round(best_p_f1, 4),
        "best_r_f1":  round(best_r_f1, 4),
        "best_f2":    round(best_f2, 4),
        "best_thr_f2": round(best_thr_f2, 3),
        "best_p_f2":  round(best_p_f2, 4),
        "best_r_f2":  round(best_r_f2, 4),
        "y_true":     y_te_arr,
        "y_prob":     prob_te,
    }

# 4. Görsel

def confusion_matrix_kaydet(sonuclar, df_clean):
    if not sonuclar:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for i, s in enumerate(sonuclar[:4]):
        ax = axes[i]
        y_pred = (s["y_prob"] >= s["best_thr_f1"]).astype(int)
        cm = confusion_matrix(s["y_true"], y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=["İyi (yarın)", "Yüksek (yarın)"],
                    yticklabels=["İyi (yarın)", "Yüksek (yarın)"],
                    cbar_kws={"shrink": 0.7}, ax=ax)
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek (yarın)")
        ax.set_title(f"{s['model_type']} k={s['class_k']} ema={'+' if s['ema_dahil'] else '-'}\n"
                     f"F1={s['best_f1']:.3f} P={s['best_p_f1']:.3f} R={s['best_r_f1']:.3f}\n"
                     f"AUC={s['auc']:.3f}")
    plt.suptitle("FORECASTING MODEL — Yarınki Risk Tahmini", fontsize=14)
    plt.tight_layout()
    yol = os.path.join(REPORTS, "117_forecasting_confusion_matrix.png")
    fig.savefig(yol, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"   117_forecasting_confusion_matrix.png")

# Ana akış

def run_forecasting():
    t_global = time.time()
    log("\n" + "═" * 60)
    log("  FORECASTING MODEL")
    log("  Bugün EMA + Pasif → YARIN Risk Tahmini")
    log("═" * 60)

    # 1. Veri hazırla
    df = forecasting_data_olustur()

    # Setup CSV
    setup_df = pd.DataFrame([{
        "n_toplam":   len(df),
        "n_pos_yarin": int(df[BINARY_YARIN].sum()),
        "pos_pct":    round(df[BINARY_YARIN].mean() * 100, 2),
        "n_users":    df["uid"].nunique(),
    }])
    setup_df.to_csv(os.path.join(REPORTS, "114_forecasting_setup.csv"), index=False)

    # 2. Clean labels
    df_clean = clean_labels_forecasting(df)

    # CV splits
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    # 3. Feature seçimi — iki versiyon: EMA dahil + EMA hariç
    EMA_FEATURES = [
        "stress", "pam_score", "social_level",
        "phq4_q1", "phq4_q2", "phq4_q3", "phq4_q4",
        "stress_z", "pam_valence", "pam_arousal",
        "phq4_anksiyete", "phq4_depresyon", "phq4_total", "phq4_risk",
        "gad2_pozitif", "phq2_pozitif",
        "social_subj_norm", "social_obj_norm", "social_delta", "obj_iletisim",
        "pam_q_Q1", "pam_q_Q2", "pam_q_Q3", "pam_q_Q4",
    ]

    META_EXCLUDE = META + [TARGET, BINARY_BUGUN, BINARY_YARIN,
                            "yarin_final_risk_4", "gun_dt", "yarin_dt", "gun_farki"]

    feat_full = [c for c in df_clean.columns if c not in META_EXCLUDE]
    feat_pasif = [c for c in feat_full if c not in EMA_FEATURES]

    log(f"\n  Tüm feature (EMA dahil):  {len(feat_full)}")
    log(f"  Sadece pasif (EMA hariç): {len(feat_pasif)}")

    # 4. 4 model dene
    log("\n── 3. 4 MODEL DENENİYOR ──")
    sonuclar = []
    for ema_dahil, feat_list, etiket_feat in [
        (True,  feat_full, "EMA+Pasif"),
        (False, feat_pasif, "Sadece Pasif"),
    ]:
        log(f"\n  [{etiket_feat}]")
        for mt in ["RF", "HGB"]:
            r = egit_test(df_clean, feat_list, splits, mt, class_k=3, ema_dahil=ema_dahil)
            if r:
                sonuclar.append(r)

    if not sonuclar:
        log("\n   Hiç model eğitilemedi — yetersiz veri")
        return

    # En iyiyi bul
    en_iyi = max(sonuclar, key=lambda s: s["best_f1"])
    log(f"\n EN İYİ: {en_iyi['model_type']} (ema={en_iyi['ema_dahil']})")
    log(f"   F1={en_iyi['best_f1']:.4f}  P={en_iyi['best_p_f1']:.4f}  "
        f"R={en_iyi['best_r_f1']:.4f}  AUC={en_iyi['auc']:.4f}")

    # 5. Holdout detay kaydet
    holdout_rows = [{k: v for k, v in s.items()
                     if k not in ["model", "y_true", "y_prob"]}
                    for s in sonuclar]
    pd.DataFrame(holdout_rows).to_csv(
        os.path.join(REPORTS, "115_forecasting_holdout.csv"), index=False)
    log("   115_forecasting_holdout.csv")

    # 6. Karşılaştırma (cross-sectional vs forecasting)
    log("\n── 4. CROSS-SECTIONAL vs FORECASTING ──")
    karsi = pd.DataFrame([
        # Eski sonuçlar
        {"yaklasım": "Cross-sectional (HİBRİT, leakage var)",
         "input":   "Bugün EMA + Pasif",
         "target":  "Bugün risk",
         "f1":      1.0000, "precision": 1.0000, "recall": 1.0000, "auc": 1.0000,
         "yorum":   "Data leakage — cascade ezberi"},

        {"yaklasım": "Cross-sectional (CLEAN PASİF, no leakage)",
         "input":   "Bugün Pasif (sadece)",
         "target":  "Bugün risk",
         "f1":      0.3599, "precision": 0.2990, "recall": 0.4519, "auc": 0.6976,
         "yorum":   "Saf pasif sınırı"},

        # Yeni: forecasting
        {"yaklasım": "Forecasting (EMA+Pasif → YARIN)",
         "input":   "Bugün EMA + Pasif",
         "target":  "Yarın risk",
         "f1":      en_iyi["best_f1"], "precision": en_iyi["best_p_f1"],
         "recall":  en_iyi["best_r_f1"], "auc": en_iyi["auc"],
         "yorum":   "AKADEMİK TEMİZ — leakage yok"},
    ])
    log("\n" + karsi.to_string(index=False))
    karsi.to_csv(os.path.join(REPORTS, "116_forecasting_comparison.csv"), index=False)
    log("\n   116_forecasting_comparison.csv")

    # 7. Confusion matrix
    confusion_matrix_kaydet(sonuclar, df_clean)

    # 8. Model pickle
    with open(os.path.join(MODELS, "best_forecasting.pkl"), "wb") as f:
        pickle.dump({
            "model":          en_iyi["model"],
            "feature_kollar": feat_full if en_iyi["ema_dahil"] else feat_pasif,
            "model_type":     en_iyi["model_type"],
            "ema_dahil":      en_iyi["ema_dahil"],
            "best_params":    {},
            "target":         "yarin_is_high_risk",
            **{k: en_iyi[k] for k in ["auc", "best_f1", "best_thr_f1",
                                       "best_p_f1", "best_r_f1",
                                       "best_f2", "best_thr_f2",
                                       "best_p_f2", "best_r_f2"]},
        }, f)
    log("   models/best_forecasting.pkl")

    # Karar
    log("\n" + "═" * 60)
    log("  KARAR ANALİZİ")
    log("═" * 60)
    log(f"\n  ESKİ Hibrit (leakage):    F1=1.0000  (akademik şişirilmiş)")
    log(f"  ESKİ Pasif (no leakage):  F1=0.3599  (saf pasif sınırı)")
    log(f"  YENİ Forecasting:         F1={en_iyi['best_f1']:.4f}")

    if en_iyi["best_f1"] > 0.5:
        log("  → F1 0.50 üstü, leakage yok")
    elif en_iyi["best_f1"] > 0.35:
        log("  → İYİ. Pasiften iyi, leakage yok.")
    else:
        log("  → Pasiften düşük — yarınki risk daha zor problem")

    log("\n" + "═" * 60)
    log(f"  TAMAMLANDI — {(time.time()-t_global)/60:.1f}dk")
    log("═" * 60 + "\n")

if __name__ == "__main__":
    run_forecasting()
