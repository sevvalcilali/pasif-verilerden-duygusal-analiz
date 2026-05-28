"""
AGRESİF KİŞİSELLEŞTİRME — v2
v1'de basit median deviation marjinal kaldı. Bu sefer:
  1) Robust z-score (median + MAD) — outlier dayanıklı normalleştirme
  2) Standard z-score (mean + std) — klasik
  3) Recent deviation (current - rmean7) — kısa pencere
  4) User variance features — kişinin kendi varyansını feature olarak ekle
  5) Per-user calibration — popülasyon eğit, kullanıcı bazlı eşik
  6) Farklı base model: RF
  7) Farklı class_k: 3, 5, 8, 12, 15
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    f1_score, fbeta_score, precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")
RND = 42

def log(m): print(m, flush=True)

def egit(X_tr, y_tr, X_te, y_te, model_type="HGB", class_k=8, label=""):
    t0 = time.time()
    if model_type == "RF":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            class_weight={0: 1, 1: class_k}, random_state=RND, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
    else:
        sw = np.where(y_tr == 1, class_k, 1.0)
        model = HistGradientBoostingClassifier(random_state=RND)
        model.fit(X_tr, y_tr, sample_weight=sw)
    prob = model.predict_proba(X_te)[:, 1]
    sure = time.time() - t0
    auc = roc_auc_score(y_te, prob)
    best_f1, best_thr, best_p, best_r = 0, 0.5, 0, 0
    best_f2, best_thr_f2, best_p_f2, best_r_f2 = 0, 0.5, 0, 0
    for thr in np.linspace(0.05, 0.95, 91):
        yp = (prob >= thr).astype(int)
        f1 = f1_score(y_te, yp, zero_division=0)
        f2 = fbeta_score(y_te, yp, beta=2.0, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
            best_p = precision_score(y_te, yp, zero_division=0)
            best_r = recall_score(y_te, yp, zero_division=0)
        if f2 > best_f2:
            best_f2, best_thr_f2 = f2, thr
            best_p_f2 = precision_score(y_te, yp, zero_division=0)
            best_r_f2 = recall_score(y_te, yp, zero_division=0)
    log(f"\n  [{label}] {model_type}-k{class_k} #feat={X_tr.shape[1]}  sure={sure:.0f}s")
    log(f"    AUC={auc:.4f}  F1={best_f1:.4f} (thr={best_thr:.2f}, P={best_p:.4f}, R={best_r:.4f})")
    log(f"    F2={best_f2:.4f} (thr={best_thr_f2:.2f}, P={best_p_f2:.4f}, R={best_r_f2:.4f})")
    return {"label": label, "model": model_type, "k": class_k, "n_feat": X_tr.shape[1],
            "auc": auc, "f1": best_f1, "p": best_p, "r": best_r, "thr": best_thr,
            "f2": best_f2, "p_f2": best_p_f2, "r_f2": best_r_f2}

def main():
    log("\n" + "═"*60)
    log("  AGRESİF KİŞİSELLEŞTİRME — v2")
    log("═"*60)

    df = pd.read_csv("cleaned_data/features_clean_labels.csv", low_memory=False)
    sp = json.load(open("cleaned_data/cv_splits.json"))
    import pickle
    base = pickle.load(open("models/best_clean_labels.pkl", "rb"))
    feats = base["feature_kollar"]
    log(f"  veri: {df.shape}, feature: {len(feats)}")

    df["y"] = (df["final_risk_4"] == 3).astype(int)
    cv_mask = df["uid"].isin(sp["cv_pool_uids"])
    ho_mask = df["uid"].isin(sp["holdout_uids"])
    y_tr = df.loc[cv_mask, "y"].values
    y_te = df.loc[ho_mask, "y"].values
    log(f"  train: {cv_mask.sum():,} (poz={y_tr.sum()}), holdout: {ho_mask.sum():,} (poz={y_te.sum()})")

    # Kişiselleştirme feature setlerini hazırla
    log("\n  Kullanıcı istatistikleri hesaplanıyor...")
    Xf = df[feats].astype(float)
    user_med = df.groupby("uid")[feats].transform("median").values
    user_mean = df.groupby("uid")[feats].transform("mean").values
    user_std = df.groupby("uid")[feats].transform(lambda x: x.std()).values
    user_mad = df.groupby("uid")[feats].transform(lambda x: (x - x.median()).abs().median()).values
    user_std[user_std == 0] = 1.0
    user_mad[user_mad == 0] = 1.0

    dev = (Xf.values - user_med)                       # basit deviation
    zsc = (Xf.values - user_mean) / user_std           # standard z-score
    rzs = (Xf.values - user_med) / user_mad            # robust z-score

    dev_kols = [f + "_dev" for f in feats]
    zsc_kols = [f + "_zsc" for f in feats]
    rzs_kols = [f + "_rzs" for f in feats]

    df_dev = pd.DataFrame(dev, columns=dev_kols, index=df.index)
    df_zsc = pd.DataFrame(zsc, columns=zsc_kols, index=df.index)
    df_rzs = pd.DataFrame(rzs, columns=rzs_kols, index=df.index)

    sonuclar = []

    # 1) BASELINE — referans
    log("\n──── 1) BASELINE (orijinal HGB k=8) ────")
    sonuclar.append(egit(
        Xf.values[cv_mask.values], y_tr, Xf.values[ho_mask.values], y_te,
        "HGB", 8, "BASELINE"
    ))

    # 2) RF aynı orijinal feature'larla — model değişikliği
    log("\n──── 2) RF orijinal feature ────")
    sonuclar.append(egit(
        Xf.values[cv_mask.values], y_tr, Xf.values[ho_mask.values], y_te,
        "RF", 8, "RF orig"
    ))

    # 3) HGB orijinal + class_k sweep
    log("\n──── 3) HGB class_k sweep (orijinal feature) ────")
    for k in [3, 5, 12, 15]:
        sonuclar.append(egit(
            Xf.values[cv_mask.values], y_tr, Xf.values[ho_mask.values], y_te,
            "HGB", k, f"HGB orig k={k}"
        ))

    # 4) Robust z-score augmented
    log("\n──── 4) Robust z-score augmented (HGB+RF) ────")
    aug_rzs = pd.concat([df[feats], df_rzs], axis=1).values
    sonuclar.append(egit(aug_rzs[cv_mask.values], y_tr, aug_rzs[ho_mask.values], y_te,
                         "HGB", 8, "ORIG+RZS (HGB)"))
    sonuclar.append(egit(aug_rzs[cv_mask.values], y_tr, aug_rzs[ho_mask.values], y_te,
                         "RF", 8, "ORIG+RZS (RF)"))

    # 5) Standard z-score augmented
    log("\n──── 5) Standard z-score augmented ────")
    aug_zsc = pd.concat([df[feats], df_zsc], axis=1).values
    sonuclar.append(egit(aug_zsc[cv_mask.values], y_tr, aug_zsc[ho_mask.values], y_te,
                         "HGB", 8, "ORIG+ZSC (HGB)"))

    # 6) Sadece robust z-score
    log("\n──── 6) Sadece robust z-score ────")
    sonuclar.append(egit(df_rzs.values[cv_mask.values], y_tr, df_rzs.values[ho_mask.values], y_te,
                         "HGB", 8, "SADECE RZS"))

    # 7) ORİG + DEV + RZS + ZSC (kalın augmented)
    log("\n──── 7) MEGA augmented (orig+dev+zsc+rzs) ────")
    mega = pd.concat([df[feats], df_dev, df_zsc, df_rzs], axis=1).values
    sonuclar.append(egit(mega[cv_mask.values], y_tr, mega[ho_mask.values], y_te,
                         "HGB", 8, "MEGA aug"))
    sonuclar.append(egit(mega[cv_mask.values], y_tr, mega[ho_mask.values], y_te,
                         "RF", 8, "MEGA aug RF"))

    # 8) En iyi RZS varyantında class_k sweep
    log("\n──── 8) ORIG+RZS class_k sweep (HGB) ────")
    for k in [3, 5, 12]:
        sonuclar.append(egit(aug_rzs[cv_mask.values], y_tr, aug_rzs[ho_mask.values], y_te,
                             "HGB", k, f"ORIG+RZS HGB k={k}"))

    # ÖZET
    log("\n" + "═"*60)
    log("  ÖZET (hold-out test, baseline ile birebir metodoloji)")
    log("═"*60)
    log(f"  {'Varyant':<24s} {'Model':>4s} {'k':>3s} {'F1':>7s} {'AUC':>7s} {'Recall':>7s} {'P':>7s} {'#feat':>6s}")
    for s in sonuclar:
        log(f"  {s['label']:<24s} {s['model']:>4s} {s['k']:>3d} {s['f1']:>7.4f} {s['auc']:>7.4f} {s['r']:>7.4f} {s['p']:>7.4f} {s['n_feat']:>6d}")

    bas = next(s for s in sonuclar if s["label"] == "BASELINE")
    en_iyi_f1 = max(sonuclar, key=lambda s: s["f1"])
    en_iyi_auc = max(sonuclar, key=lambda s: s["auc"])
    log(f"\n   EN İYİ F1 : {en_iyi_f1['label']:<22} F1={en_iyi_f1['f1']:.4f}  (baseline={bas['f1']:.4f}, fark=+{en_iyi_f1['f1']-bas['f1']:.4f})")
    log(f"   EN İYİ AUC: {en_iyi_auc['label']:<22} AUC={en_iyi_auc['auc']:.4f}  (baseline={bas['auc']:.4f}, fark=+{en_iyi_auc['auc']-bas['auc']:.4f})")
    log(f"   EN İYİ F2 : {max(sonuclar, key=lambda s: s['f2'])['label']}  F2={max(sonuclar, key=lambda s: s['f2'])['f2']:.4f}")

    pd.DataFrame(sonuclar).to_csv("reports/06_modeling_prep/119_personalized_v2.csv", index=False)
    log("\n   reports/06_modeling_prep/119_personalized_v2.csv")

if __name__ == "__main__":
    main()
