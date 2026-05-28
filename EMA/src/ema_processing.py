"""
EMA cevaplarını türetilmiş skorlara çevirir.

7 ham sorudan: stress z-skoru, PAM valence/arousal/quadrant,
PHQ-4 alt-ölçek + risk sınıfı, social delta üretir.
Cevapların hazır geldiğini varsayar.
"""

import numpy as np
import pandas as pd


# Eşikler: Kroenke 2009 (PHQ-4), Russell 1980 (PAM)
PHQ4_NORMAL_USTU = 2       # toplam 0-2 normal, 3-5 hafif, 6-8 orta, 9-12 şiddetli
PHQ4_HAFIF_USTU  = 5
PHQ4_ORTA_USTU   = 8
ALT_OLCEK_POZ_ESIK = 3     # GAD-2 / PHQ-2 pozitiflik eşiği

# PAM 4x4 grid (Pollak 2011): valence sol→sağ artar, arousal üst→alt azalır.
# Not: pam_arousal'da küçük sayı = yüksek arousal (üst satır).
PAM_VALENCE_ESIK = 3       # >= 3 pozitif valence
PAM_AROUSAL_ESIK = 2       # <= 2 yüksek arousal

NORM_EPSILON = 1e-9        # sıfıra bölünme koruması

BEKLENEN_KOLONLAR = (
    "uid, gun, stress, pam_score, social_level, "
    "phq4_q1, phq4_q2, phq4_q3, phq4_q4, obj_iletisim"
)


def stress_z_skoru_hesapla(
    df: pd.DataFrame,
    uid_kol: str = "uid",
    stress_kol: str = "stress",
) -> pd.DataFrame:
    """Stresi kişi-içi z-skoruna çevirir. Tek veri veya std=0 olanlarda z=0."""
    for kol in (uid_kol, stress_kol):
        if kol not in df.columns:
            raise ValueError(
                f"Eksik kolon: '{kol}'. Beklenen kolonlar: {BEKLENEN_KOLONLAR}"
            )

    df = df.copy()

    def _z(seri: pd.Series) -> pd.Series:
        if len(seri) <= 1:
            return pd.Series(0.0, index=seri.index)
        std = seri.std(ddof=1)
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=seri.index)
        return (seri - seri.mean()) / std

    df["stress_z"] = (
        df.groupby(uid_kol)[stress_kol]
        .transform(_z)
        .fillna(0.0)
        .astype(float)
    )
    return df


def pam_koordinat_quadrant(
    df: pd.DataFrame,
    pam_kol: str = "pam_score",
) -> pd.DataFrame:
    """
    1-16 PAM skorunu valence (1-4), arousal (1-4) ve quadrant'a (Q1-Q4) böler.

    pam=1 üst-sol (Q3 anksiyete), pam=4 üst-sağ (Q1 coşkulu),
    pam=13 alt-sol (Q4 depresyon), pam=16 alt-sağ (Q2 sakin).
    arousal küçükse yüksek arousal (üst satır).
    """
    if pam_kol not in df.columns:
        raise ValueError(
            f"Eksik kolon: '{pam_kol}'. Beklenen kolonlar: {BEKLENEN_KOLONLAR}"
        )

    df = df.copy()
    pam = df[pam_kol].to_numpy()

    df["pam_valence"] = ((pam - 1) % 4) + 1
    df["pam_arousal"] = ((pam - 1) // 4) + 1

    v = df["pam_valence"].to_numpy()
    a = df["pam_arousal"].to_numpy()

    kosullar = [
        (v >= PAM_VALENCE_ESIK) & (a <= PAM_AROUSAL_ESIK),  # Q1 coşkulu
        (v >= PAM_VALENCE_ESIK) & (a >  PAM_AROUSAL_ESIK),  # Q2 sakin
        (v <  PAM_VALENCE_ESIK) & (a <= PAM_AROUSAL_ESIK),  # Q3 anksiyete
        (v <  PAM_VALENCE_ESIK) & (a >  PAM_AROUSAL_ESIK),  # Q4 depresyon
    ]
    secimler = ["Q1", "Q2", "Q3", "Q4"]
    df["pam_quadrant"] = np.select(kosullar, secimler, default="Q?")

    return df


def phq4_sinifla(df: pd.DataFrame) -> pd.DataFrame:
    """PHQ-4'ün 4 sorusundan toplam, alt-ölçek ve risk sınıfı (Kroenke 2009)."""
    soru_kollari = ["phq4_q1", "phq4_q2", "phq4_q3", "phq4_q4"]
    for kol in soru_kollari:
        if kol not in df.columns:
            raise ValueError(
                f"Eksik kolon: '{kol}'. PHQ-4 için 4 soru da gerekli. "
                f"Beklenen kolonlar: {BEKLENEN_KOLONLAR}"
            )

    df = df.copy()
    df["phq4_anksiyete"] = (df["phq4_q1"] + df["phq4_q2"]).astype(int)
    df["phq4_depresyon"] = (df["phq4_q3"] + df["phq4_q4"]).astype(int)
    df["phq4_total"]     = (df["phq4_anksiyete"] + df["phq4_depresyon"]).astype(int)

    total = df["phq4_total"].to_numpy()
    df["phq4_risk"] = np.select(
        [
            total <= PHQ4_NORMAL_USTU,
            total <= PHQ4_HAFIF_USTU,
            total <= PHQ4_ORTA_USTU,
        ],
        [0, 1, 2],
        default=3,
    ).astype(int)

    df["gad2_pozitif"] = df["phq4_anksiyete"] >= ALT_OLCEK_POZ_ESIK
    df["phq2_pozitif"] = df["phq4_depresyon"] >= ALT_OLCEK_POZ_ESIK
    return df


def social_delta_hesapla(
    df: pd.DataFrame,
    sub_kol: str = "social_level",
    obj_kol: str = "obj_iletisim",
) -> pd.DataFrame:
    """
    Sübjektif sosyal his ile objektif iletişim arasındaki fark.
    İkisi de min-max normalize edilip çıkarılır; negatif delta
    kalabalıkta yalnızlık sinyali.
    """
    for kol in (sub_kol, obj_kol):
        if kol not in df.columns:
            raise ValueError(
                f"Eksik kolon: '{kol}'. Beklenen kolonlar: {BEKLENEN_KOLONLAR}"
            )

    df = df.copy()
    df["social_subj_norm"] = _min_max_normalize(df[sub_kol])
    df["social_obj_norm"]  = _min_max_normalize(df[obj_kol])
    df["social_delta"]     = df["social_subj_norm"] - df["social_obj_norm"]
    return df


def _min_max_normalize(seri: pd.Series) -> pd.Series:
    """Min-max normalize (0-1), epsilon ile sıfıra bölünme korumalı."""
    s = seri.astype(float)
    rng = s.max() - s.min()
    return (s - s.min()) / (rng + NORM_EPSILON)


def tum_islemleri_uygula(df: pd.DataFrame) -> pd.DataFrame:
    """Yukarıdaki dört işlemi sırayla uygular."""
    try:
        df = stress_z_skoru_hesapla(df)
        df = pam_koordinat_quadrant(df)
        df = phq4_sinifla(df)
        df = social_delta_hesapla(df)
    except KeyError as hata:
        raise ValueError(
            f"Eksik kolon: {hata}. Beklenen kolonlar: {BEKLENEN_KOLONLAR}"
        ) from hata
    return df
