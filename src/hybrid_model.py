"""
Hibrit model: ~181 feature ile EMA + pasif veriyi birlikte kullanır.
F1=1.0 çıkıyor çünkü etiket EMA'dan üretiliyor ve EMA da girdide — data leakage.
Eğitici karşı-örnek olarak tutuluyor (SHAP ile leakage gösteriliyor).
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
    confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score,
)

from src.config import PATHS

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_CSV   = os.path.join(PATHS["cleaned_data"], "features_extended_temporal.csv")
MASTER_CSV = os.path.join(PATHS["cleaned_data"], "master_dataset.csv")
HYBRID_OUT = os.path.join(PATHS["cleaned_data"], "features_hybrid.csv")
SPLITS_PATH = os.path.join(PATHS["cleaned_data"], "cv_splits.json")
REPORTS = os.path.join(PATHS["reports"], "06_modeling_prep")
MODELS = os.path.join(BASE, "models")

META = ["uid", "gun", "is_ios"]
TARGET = "final_risk_4"
BINARY = "is_high_risk"
RND = 42

# EMA feature'ları (master_dataset'ten alınacak)
EMA_HAM = ["stress", "pam_score", "social_level",
           "phq4_q1", "phq4_q2", "phq4_q3", "phq4_q4"]
EMA_TURETILEN = ["stress_z", "pam_valence", "pam_arousal",
                  "phq4_anksiyete", "phq4_depresyon", "phq4_total", "phq4_risk",
                  "gad2_pozitif", "phq2_pozitif",
                  "social_subj_norm", "social_obj_norm", "social_delta",
                  "obj_iletisim"]

def log(msg):
    print(msg, flush=True)

def make_binary(df):
    df = df.copy()
    df[BINARY] = (df[TARGET] == 3).astype(int)
    return df

# 1. EMA + Pasif birleştirme

def hibrit_veri_olustur():
    log("\n── 1. EMA + Pasif Veri Birleştirme ──")

    # Master (EMA dahil)
    log("  master_dataset.csv yükleniyor...")
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    log(f"    {master.shape}")

    # Pasif (temporal features dahil)
    log("  features_extended_temporal.csv yükleniyor...")
    pasif = pd.read_csv(TEMP_CSV, low_memory=False)
    log(f"    {pasif.shape}")

    # Master'dan EMA feature'larını seç
    ema_secilen_kollar = META + EMA_HAM + EMA_TURETILEN + ["pam_quadrant", TARGET]
    ema_secilen_kollar = [c for c in ema_secilen_kollar if c in master.columns]
    ema_df = master[ema_secilen_kollar].copy()
    log(f"  EMA tarafı: {len(ema_secilen_kollar)} sütun seçildi")

    # Pasif'ten meta'yı çıkar (zaten master'da var), feature'ları al
    pasif_feat = [c for c in pasif.columns if c not in META + [TARGET, BINARY]]
    pasif_df = pasif[["uid", "gun"] + pasif_feat].copy()
    log(f"  Pasif tarafı: {len(pasif_feat)} feature")

    # Birleştir
    hibrit = ema_df.merge(pasif_df, on=["uid", "gun"], how="inner")
    log(f"\n  Birleştirilmiş: {hibrit.shape}")

    # pam_quadrant one-hot
    log("\n  pam_quadrant → one-hot encoding...")
    pam_dummies = pd.get_dummies(hibrit["pam_quadrant"], prefix="pam_q").astype(int)
    hibrit = pd.concat([hibrit.drop(columns="pam_quadrant"), pam_dummies], axis=1)
    log(f"  Yeni one-hot sütunlar: {list(pam_dummies.columns)}")

    # bool sütunları int'e çevir
    for c in ["gad2_pozitif", "phq2_pozitif"]:
        if c in hibrit.columns:
            hibrit[c] = hibrit[c].astype(int)

    # NaN kontrolü
    nan_count = hibrit.isnull().sum().sum()
    log(f"\n  NaN: {nan_count}")
    if nan_count > 0:
        log("  Kullanıcı medyanı ile dolduruluyor...")
        for c in hibrit.columns:
            if hibrit[c].dtype.kind in 'fi' and hibrit[c].isnull().any():
                hibrit[c] = hibrit.groupby("uid")[c].transform(
                    lambda x: x.fillna(x.median()))
                hibrit[c] = hibrit[c].fillna(hibrit[c].median())

    log(f"  Final NaN: {hibrit.isnull().sum().sum()}")
    log(f"  Final boyut: {hibrit.shape}")
    return hibrit

# 2. Clean labels filter

def clean_labels(df):
    log("\n── 2. Clean Labels Filtreleme (class 0 + class 3) ──")
    n_once = len(df)
    df_clean = df[df[TARGET].isin([0, 3])].copy()
    df_clean = make_binary(df_clean)
    log(f"  {n_once:,} → {len(df_clean):,} satır")
    log(f"  Negatif: {(df_clean[BINARY] == 0).sum():,}")
    log(f"  Pozitif: {df_clean[BINARY].sum():,}")
    return df_clean

# 3. Tek model eğitim + değerlendirme

def egit_test(df, feat, splits, model_type, class_k):
    cv_pool = df[df["uid"].isin(splits["cv_pool_uids"])]
    holdout = df[df["uid"].isin(splits["holdout_uids"])]
    X_tr, y_tr = cv_pool[feat], cv_pool[BINARY]
    X_te, y_te = holdout[feat], holdout[BINARY]

    log(f"\n  {model_type} k={class_k} (train={len(X_tr):,}, test={len(X_te):,})")
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
    ap  = average_precision_score(y_te_arr, prob_te)

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

    log(f"    AUC={auc:.4f}, AP={ap:.4f}")
    log(f"    F1-opt thr={best_thr_f1:.2f}: F1={best_f1:.4f}  P={best_p_f1:.4f}  R={best_r_f1:.4f}")
    log(f"    F2-opt thr={best_thr_f2:.2f}: F2={best_f2:.4f}  P={best_p_f2:.4f}  R={best_r_f2:.4f}")

    return {
        "model_type": model_type, "class_k": class_k, "model": model,
        "auc": auc, "ap": ap,
        "best_f1": best_f1, "best_thr_f1": best_thr_f1,
        "best_p_f1": best_p_f1, "best_r_f1": best_r_f1,
        "best_f2": best_f2, "best_thr_f2": best_thr_f2,
        "best_p_f2": best_p_f2, "best_r_f2": best_r_f2,
        "y_true": y_te_arr, "y_prob": prob_te,
    }

# 4. Confusion matrix görsel

def confusion_matrix_kaydet(en_iyi):
    log("\n── 4. Confusion matrix ──")
    y_t = en_iyi["y_true"]
    y_prob = en_iyi["y_prob"]
    y_p_f1 = (y_prob >= en_iyi["best_thr_f1"]).astype(int)
    y_p_f2 = (y_prob >= en_iyi["best_thr_f2"]).astype(int)

    cm_f1 = confusion_matrix(y_t, y_p_f1, labels=[0,1])
    cm_f2 = confusion_matrix(y_t, y_p_f2, labels=[0,1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cm, baslik in [
        (axes[0], cm_f1, f"HİBRİT F1-opt (thr={en_iyi['best_thr_f1']:.2f})\n"
                         f"F1={en_iyi['best_f1']:.3f} P={en_iyi['best_p_f1']:.3f} R={en_iyi['best_r_f1']:.3f}"),
        (axes[1], cm_f2, f"HİBRİT F2-opt (thr={en_iyi['best_thr_f2']:.2f})\n"
                         f"F2={en_iyi['best_f2']:.3f} P={en_iyi['best_p_f2']:.3f} R={en_iyi['best_r_f2']:.3f}"),
    ]:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=["İyi", "Yüksek Risk"],
                    yticklabels=["İyi", "Yüksek Risk"],
                    cbar_kws={"shrink": 0.7}, ax=ax)
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
        ax.set_title(baslik, fontsize=11)
    plt.suptitle(f"HİBRİT MODEL ({en_iyi['model_type']} k={en_iyi['class_k']}) — "
                 f"AUC={en_iyi['auc']:.3f}", fontsize=14)
    plt.tight_layout()
    yol = os.path.join(REPORTS, "105_hybrid_confusion_matrix.png")
    fig.savefig(yol, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"   105_hybrid_confusion_matrix.png")

# Ana akış

def run_hybrid():
    t_global = time.time()
    log("\n" + "═" * 60)
    log("  HİBRİT MODEL (EMA + Pasif)")
    log("═" * 60)

    # 1. Hibrit veri
    hibrit = hibrit_veri_olustur()
    hibrit.to_csv(HYBRID_OUT, index=False)
    log(f"   {HYBRID_OUT}")

    # 2. Clean labels
    hibrit_clean = clean_labels(hibrit)

    # 3. Feature list
    feat = [c for c in hibrit_clean.columns if c not in META + [TARGET, BINARY]]
    log(f"\n  Toplam feature: {len(feat)}")
    log(f"    EMA ham (7) + EMA türetilmiş (~18) + Pasif (~159)")

    # Feature dağılımı kayıt
    feat_log = []
    for f in feat:
        if f in EMA_HAM:
            kat = "ema_ham"
        elif f in EMA_TURETILEN:
            kat = "ema_turetilen"
        elif f.startswith("pam_q"):
            kat = "ema_pam_onehot"
        else:
            kat = "pasif"
        feat_log.append({"feature": f, "kategori": kat})
    pd.DataFrame(feat_log).to_csv(
        os.path.join(REPORTS, "102_hybrid_features.csv"), index=False)
    log(f"   102_hybrid_features.csv")

    # CV splits
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    # 4. Modeller (RF + HGB × k=3, 5, 8)
    log("\n── 3. Model eğitimi (RF + HGB × class_k) ──")
    sonuclar = []
    for k in [3, 5, 8]:
        for tip in ["RF", "HGB"]:
            res = egit_test(hibrit_clean, feat, splits, tip, k)
            sonuclar.append(res)

    # En iyiyi bul
    en_iyi = max(sonuclar, key=lambda s: s["best_f1"])
    log(f"\n EN İYİ: {en_iyi['model_type']} k={en_iyi['class_k']}")
    log(f"   F1={en_iyi['best_f1']:.4f}  P={en_iyi['best_p_f1']:.4f}  "
        f"R={en_iyi['best_r_f1']:.4f}  AUC={en_iyi['auc']:.4f}")

    # Hold-out detay
    holdout_rows = []
    for s in sonuclar:
        holdout_rows.append({
            "model_type": s["model_type"],
            "class_k":    s["class_k"],
            "auc":        round(s["auc"], 4),
            "ap":         round(s["ap"], 4),
            "best_thr_f1": s["best_thr_f1"],
            "best_f1":    round(s["best_f1"], 4),
            "best_p_f1":  round(s["best_p_f1"], 4),
            "best_r_f1":  round(s["best_r_f1"], 4),
            "best_thr_f2": s["best_thr_f2"],
            "best_f2":    round(s["best_f2"], 4),
            "best_p_f2":  round(s["best_p_f2"], 4),
            "best_r_f2":  round(s["best_r_f2"], 4),
        })
    pd.DataFrame(holdout_rows).to_csv(
        os.path.join(REPORTS, "103_hybrid_holdout.csv"), index=False)
    log("   103_hybrid_holdout.csv")

    # 5. Karşılaştırma (pasif vs hibrit)
    log("\n── 4. PASIF vs HİBRİT Karşılaştırma ──")
    karsi = pd.DataFrame([
        {"model": "ÖNCEKİ — Voting V1 (tüm sınıflar, pasif only)",
         "f1": 0.2298, "precision": 0.2141, "recall": 0.2481, "auc": 0.6671,
         "n_features": 159, "ema_input": False},
        {"model": "ÖNCEKİ — CLEAN HGB k=8 (pasif only, clean)",
         "f1": 0.3599, "precision": 0.2990, "recall": 0.4519, "auc": 0.6976,
         "n_features": 159, "ema_input": False},
        {"model": f"HİBRİT — {en_iyi['model_type']} k={en_iyi['class_k']} F1-opt ",
         "f1": round(en_iyi["best_f1"], 4),
         "precision": round(en_iyi["best_p_f1"], 4),
         "recall":    round(en_iyi["best_r_f1"], 4),
         "auc":       round(en_iyi["auc"], 4),
         "n_features": len(feat),
         "ema_input": True},
        {"model": f"HİBRİT — {en_iyi['model_type']} k={en_iyi['class_k']} F2-opt (klinik)",
         "f1": round(en_iyi["best_f2"], 4),
         "precision": round(en_iyi["best_p_f2"], 4),
         "recall":    round(en_iyi["best_r_f2"], 4),
         "auc":       round(en_iyi["auc"], 4),
         "n_features": len(feat),
         "ema_input": True},
    ])
    log("\n" + karsi.to_string(index=False))
    karsi.to_csv(os.path.join(REPORTS, "104_hybrid_comparison.csv"), index=False)
    log("   104_hybrid_comparison.csv")

    # 6. Confusion matrix
    confusion_matrix_kaydet(en_iyi)

    # 7. Pickle
    with open(os.path.join(MODELS, "best_hybrid.pkl"), "wb") as f:
        pickle.dump({
            "model":          en_iyi["model"],
            "feature_kollar": feat,
            "model_type":     en_iyi["model_type"],
            "class_k":        en_iyi["class_k"],
            **{k: en_iyi[k] for k in ["auc", "ap", "best_f1", "best_thr_f1",
                                       "best_p_f1", "best_r_f1",
                                       "best_f2", "best_thr_f2",
                                       "best_p_f2", "best_r_f2"]},
        }, f)
    log(f"   models/best_hybrid.pkl")

    # Karar analizi
    log("\n" + "═" * 60)
    log("  KARAR ANALİZİ")
    log("═" * 60)
    onceki_best = 0.3599   # CLEAN HGB k=8
    yeni_best = en_iyi["best_f1"]
    delta = yeni_best - onceki_best
    log(f"\n  Önceki best (clean pasif HGB k=8 F1): {onceki_best:.4f}")
    log(f"  Hibrit best F1:                        {yeni_best:.4f}")
    log(f"  Delta:                                 {delta:+.4f}")
    if yeni_best >= 0.70:
        log("  → F1 0.70 hedefini geçti")
    elif yeni_best >= 0.50:
        log("  → iyi")
    elif delta >= 0.10:
        log("  → BÜYÜK İYİLEŞME ")
    elif delta > 0:
        log("  → Hafif iyileşme")
    else:
        log("  → İyileşme YOK")

    log("\n" + "═" * 60)
    log(f"  TAMAMLANDI — {(time.time()-t_global)/60:.1f}dk")
    log("═" * 60 + "\n")

if __name__ == "__main__":
    run_hybrid()
