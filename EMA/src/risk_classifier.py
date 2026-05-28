"""
EMA cevaplarından profil eşleme ve 4-sınıflı risk.

İki paralel yaklaşım: (1) 12 profile Manhattan mesafesiyle en yakını bulma,
(2) kural-tabanlı şeffaf 4-sınıf risk. tahmin_yorumla tek bir EMA örneğini
tek dict çıktıya çevirir.
"""

from __future__ import annotations

import numpy as np

from .ema_processing import (
    PHQ4_HAFIF_USTU,
    PHQ4_NORMAL_USTU,
    PHQ4_ORTA_USTU,
    NORM_EPSILON,
)


# 12 profil — (id, isim, stress_z, quadrant, social_delta,
#              phq4_anksiyete, phq4_depresyon, phq4_total, risk_sinifi_4)
PROFILLER: tuple[tuple[str, str, float, str, float, float, float, int, int], ...] = (
    ("A", "Optimum Denge / Flow",         -0.7, "Q2",  0.05, 0.5, 0.5,  1, 0),
    ("B", "Üretken Coşku",                 0.0, "Q1",  0.45, 1.0, 0.5,  2, 0),
    ("C", "Normal Akademik Stres",         0.3, "Q3",  0.0,  1.5, 0.5,  2, 0),
    ("D", "Hafif Anksiyete",               0.7, "Q3", -0.15, 3.5, 1.0,  4, 1),
    ("E", "Orta Anksiyete + İzolasyon",    1.2, "Q3", -0.35, 5.0, 1.5,  6, 2),
    ("F", "Hafif Depresif Eğilim",         0.3, "Q4", -0.25, 1.0, 3.5,  4, 1),
    ("G", "Orta Depresif + Yalnızlık",     0.5, "Q4", -0.55, 2.0, 5.0,  6, 2),
    ("H", "Karma Yüksek Risk",             1.2, "Q4", -0.75, 5.0, 5.0, 10, 3),
    ("I", "Akut Anksiyete (Kriz Modu 1)",  1.8, "Q3", -0.75, 5.5, 4.0,  9, 3),
    ("J", "Akut Depresif (Kriz Modu 2)",   0.3, "Q4", -0.85, 3.0, 5.5,  8, 3),
    ("K", "Maskeli Burnout",               1.2, "Q1", -0.05, 3.5, 3.5,  7, 2),
    ("L", "Görünmez Kriz (Yalnızlık)",     0.5, "Q4", -0.85, 2.0, 4.0,  6, 2),
)

# Mesafe normalize aralıkları
NORM_STRESS_Z         = 3.0
NORM_SOCIAL_DELTA     = 2.0
NORM_PHQ4_ALT_OLCEK   = 6.0
NORM_PHQ4_TOTAL       = 12.0
QUADRANT_FARK_CEZASI  = 0.5

# 4-sınıf risk meta — (isim, renk)
RISK_META: dict[int, tuple[str, str]] = {
    0: ("İyi Durum",   "yeşil"),
    1: ("Hafif Risk",  "sarı"),
    2: ("Orta Risk",   "turuncu"),
    3: ("Yüksek Risk", "kırmızı"),
}

# Klinik kural eşikleri (stress_z ile)
STRESS_Z_YUKSEK     = 1.5
STRESS_Z_ORTA       = 1.0
STRESS_Z_HAFIF      = 0.5
DELTA_KRITIK        = -0.5

# Cascade için ham stres (1-5, Elo 2003)
STRES_HAM_YUKSEK    = 4   # >= 4
STRES_HAM_ORTA      = 3   # = 3
STRES_HAM_DUSUK     = 2   # <= 2

# GAD-2 / PHQ-2 cut-off (Kroenke 2009)
ALT_OLCEK_POZ_ESIK_HAM = 3   # >= 3 pozitif
ALT_OLCEK_AKUT_ESIK    = 5   # >= 5 akut (Wicke 2022)
ALT_OLCEK_HAFIF_ESIK   = 2   # >= 2 sub-clinical (Löwe 2010)

# Sosyal seviye (1-5)
SOSYAL_IZOLE_ESIK   = 2   # <= 2 izole
SOSYAL_YUKSEK_ESIK  = 4   # >= 4 sosyal

# Tek-örnek stres → kaba z (1-5 → -1.5..+1.5)
TEK_ORNEK_STRESS_ORTA  = 3.0
TEK_ORNEK_STRESS_OLCEK = 1.5

RISK_ACIKLAMA: dict[int, str] = {
    0: "Genel duygusal denge iyi durumda. Belirgin stres veya çökme sinyali yok.",
    1: "Hafif düzeyde stres veya duygu durum belirtisi var; takip önerilir.",
    2: "Orta düzeyde anksiyete/depresyon belirtisi mevcut; profesyonel destek değerlendirilmeli.",
    3: "Yüksek risk sinyali — akut tablo olabilir; öncelikli klinik değerlendirme önerilir.",
}

RISK_ONERI: dict[int, str] = {
    0: "Mevcut günlük rutin korunabilir. Uyku, fiziksel aktivite ve sosyal teması sürdür.",
    1: "Stres yönetimi (nefes, kısa egzersiz) ve düzenli uyku önerilir. EMA takibi sürsün.",
    2: "Üniversite psikolojik danışma birimine başvurmak değerlendirilebilir. Yakın takibe alın.",
    3: "Acil destek hattı ya da klinik psikolog başvurusu önerilir. Tek başına bırakılmamalı.",
}


def en_yakin_profil_bul(
    stress_z: float,
    pam_quadrant: str,
    social_delta: float,
    phq4_anksiyete: int,
    phq4_depresyon: int,
    phq4_total: int,
) -> tuple[str, str, int]:
    """12 profile normalize Manhattan mesafesi; en yakını döner (id, isim, risk).
    Quadrant farklıysa 0.5 ceza eklenir."""
    en_kisa_mesafe = float("inf")
    en_yakin: tuple[str, str, int] = ("?", "Bilinmiyor", -1)

    for (p_id, p_isim, p_sz, p_q, p_d,
         p_anx, p_dep, p_tot, p_risk) in PROFILLER:
        mesafe = (
            abs(stress_z       - p_sz)  / NORM_STRESS_Z       +
            abs(social_delta   - p_d)   / NORM_SOCIAL_DELTA   +
            abs(phq4_anksiyete - p_anx) / NORM_PHQ4_ALT_OLCEK +
            abs(phq4_depresyon - p_dep) / NORM_PHQ4_ALT_OLCEK +
            abs(phq4_total     - p_tot) / NORM_PHQ4_TOTAL
        )
        if pam_quadrant != p_q:
            mesafe += QUADRANT_FARK_CEZASI

        if mesafe < en_kisa_mesafe:
            en_kisa_mesafe = mesafe
            en_yakin = (p_id, p_isim, p_risk)

    return en_yakin


def kural_tabanli_profil_eslemesi(
    stress_ham: int,             # 1-5 ham stres
    pam_quadrant: str,           # "Q1".."Q4"
    social_level: int,           # 1-5 ham sosyal seviye
    phq4_anksiyete: int,         # 0-6
    phq4_depresyon: int,         # 0-6
    phq4_total: int,             # 0-12
) -> tuple[str, str, int, str]:
    """ema.md cascade eşlemesi (Kroenke 2009 + Posner 2008). İlk eşleşen kural
    kazanır (akut kriz → ... → optimum denge); hiçbiri eşleşmezse C."""
    anks_pozitif = phq4_anksiyete >= ALT_OLCEK_POZ_ESIK_HAM
    dep_pozitif  = phq4_depresyon >= ALT_OLCEK_POZ_ESIK_HAM
    anks_akut    = phq4_anksiyete >= ALT_OLCEK_AKUT_ESIK
    dep_akut     = phq4_depresyon >= ALT_OLCEK_AKUT_ESIK

    yuksek_stres = stress_ham >= STRES_HAM_YUKSEK
    orta_stres   = stress_ham == STRES_HAM_ORTA
    dusuk_stres  = stress_ham <= STRES_HAM_DUSUK

    izole  = social_level <= SOSYAL_IZOLE_ESIK
    sosyal = social_level >= SOSYAL_YUKSEK_ESIK

    # 1. Akut depresif kriz (Posner 2008 depresyon zonu)
    if dep_akut and pam_quadrant == "Q4":
        return ("J", "Akut Depresif (Kriz Modu 2)", 3,
                "Yüksek (PHQ-2≥5 ∩ PAM Q4 — Posner 2008 depresyon zonu)")

    # 2. Akut anksiyete kriz (Posner 2008 anksiyete zonu)
    if anks_akut and pam_quadrant == "Q3":
        return ("I", "Akut Anksiyete (Kriz Modu 1)", 3,
                "Yüksek (GAD-2≥5 ∩ PAM Q3 — Posner 2008 anksiyete zonu)")

    # 3. Görünmez kriz — PHQ pozitif ama PAM pozitif yüz (discordance)
    if (anks_pozitif or dep_pozitif) and pam_quadrant in ("Q1", "Q2"):
        return ("L", "Görünmez Kriz (Yalnızlık)", 2,
                "Orta (PAM-PHQ discordance — Posner 2008 valence focus, özgün katkı)")

    # 4. Maskeli burnout — yüksek stres + sub-clinical PHQ + Q2 sakin yüz
    if (yuksek_stres
            and ALT_OLCEK_POZ_ESIK_HAM <= phq4_total <= 5
            and pam_quadrant == "Q2"):
        return ("K", "Maskeli Burnout", 2,
                "Düşük-Orta (Bianchi 2015 burnout-depresyon overlap, özgün katkı)")

    # 5. Karma yüksek risk — her iki tarama pozitif
    if anks_pozitif and dep_pozitif:
        return ("H", "Karma Yüksek Risk", 3,
                "Yüksek (Kroenke 2009: GAD-2 ∩ PHQ-2 ≥3 her iki tarama pozitif)")

    # 6. Orta depresif + yalnızlık
    if dep_pozitif and izole:
        return ("G", "Orta Depresif + Yalnızlık", 2,
                "Yüksek (PHQ-2 cut-off + sosyal izolasyon — Cacioppo 2014)")

    # 7. Orta anksiyete + izolasyon
    if anks_pozitif and izole:
        return ("E", "Orta Anksiyete + İzolasyon", 2,
                "Yüksek (GAD-2 cut-off + sosyal kaçınma — Heinrichs 2003)")

    # 8. Hafif depresif eğilim
    if phq4_depresyon >= ALT_OLCEK_HAFIF_ESIK and not anks_pozitif:
        return ("F", "Hafif Depresif Eğilim", 1,
                "Orta (PHQ-2 ≥2 sub-clinical — Löwe 2010)")

    # 9. Hafif anksiyete
    if phq4_anksiyete >= ALT_OLCEK_HAFIF_ESIK and not dep_pozitif:
        return ("D", "Hafif Anksiyete", 1,
                "Orta (GAD-2 ≥2 sub-clinical — Löwe 2010)")

    # 10. Normal akademik stres
    if phq4_total <= 2 and (orta_stres or yuksek_stres):
        return ("C", "Normal Akademik Stres", 0,
                "Yüksek (PHQ-4 normal ∩ stres ≥3 — DaSilva 2019 'normative student stress')")

    # 11. Üretken coşku
    if phq4_total <= 2 and pam_quadrant == "Q1" and not yuksek_stres:
        return ("B", "Üretken Coşku", 0,
                "Yüksek (PHQ-4 normal ∩ Q1 pozitif yüksek arousal — Watson-Tellegen PA)")

    # 12. Optimum denge / flow
    if (phq4_total <= 2
            and pam_quadrant in ("Q1", "Q2")
            and dusuk_stres
            and sosyal):
        return ("A", "Optimum Denge / Flow", 0,
                "Yüksek (Csikszentmihalyi 1990 flow + Diener flourishing)")

    # Eşleşme yoksa
    return ("C", "Normal Akademik Stres (default)", 0,
            "Düşük (cascade'da kural eşleşmedi)")


def risk_sinifi_4_hesapla(
    stress_z: float,
    pam_quadrant: str,
    social_delta: float,
    phq4_total: int,
) -> tuple[int, str]:
    """Kural-tabanlı 4-sınıf risk; profil eşlemeden bağımsız ikinci kontrol.
    İlk eşleşen kural kazanır."""
    # 1) PHQ-4 toplamı şiddetli (9-12)
    if phq4_total > PHQ4_ORTA_USTU:
        return 3, RISK_ACIKLAMA[3]

    # 2) Akut tablo: yüksek stres + olumsuz quadrant + kritik sosyal kopuş
    if (stress_z > STRESS_Z_YUKSEK
            and pam_quadrant in ("Q3", "Q4")
            and social_delta < DELTA_KRITIK):
        return 3, RISK_ACIKLAMA[3]

    # 3) Orta PHQ-4 (6-8)
    if phq4_total > PHQ4_HAFIF_USTU:
        return 2, RISK_ACIKLAMA[2]

    # 4) Stres yüksek ya da sosyal kopuş kritik
    if stress_z > STRESS_Z_ORTA or social_delta < DELTA_KRITIK:
        return 2, RISK_ACIKLAMA[2]

    # 5) Hafif PHQ-4 (3-5)
    if phq4_total > PHQ4_NORMAL_USTU:
        return 1, RISK_ACIKLAMA[1]

    # 6) Stres hafif yüksek
    if STRESS_Z_HAFIF < stress_z <= STRESS_Z_ORTA:
        return 1, RISK_ACIKLAMA[1]

    # 7) İyi durum
    return 0, RISK_ACIKLAMA[0]


def tahmin_yorumla(ema_tahmin: dict) -> dict:
    """
    Tek bir EMA örneğini (7 cevap) tam çıktı dict'ine çevirir.

    Tek örnek olduğu için kullanıcı geçmişi yok; stress_z gerçek within-person
    z yerine kabaca (stress - 3) / 1.5 ile hesaplanır. Toplu işleme için
    ema_processing.tum_islemleri_uygula kullanılmalı.
    """
    zorunlu = ("uid", "tarih", "stress", "pam_score", "social_level",
               "phq4_q1", "phq4_q2", "phq4_q3", "phq4_q4")
    for kol in zorunlu:
        if kol not in ema_tahmin:
            raise ValueError(
                f"Eksik alan: '{kol}'. tahmin_yorumla için şu alanlar gerekli: "
                f"{', '.join(zorunlu)} (+ opsiyonel obj_iletisim)"
            )

    stress       = float(ema_tahmin["stress"])
    pam_score    = int(ema_tahmin["pam_score"])
    social_level = float(ema_tahmin["social_level"])
    obj_iletisim = ema_tahmin.get("obj_iletisim")

    # Stres z (tek-örnek yaklaşımı)
    stress_z = (stress - TEK_ORNEK_STRESS_ORTA) / TEK_ORNEK_STRESS_OLCEK

    # PAM koordinat + quadrant (arousal=1 üst/yüksek, =4 alt/düşük)
    pam_valence = ((pam_score - 1) % 4) + 1
    pam_arousal = ((pam_score - 1) // 4) + 1
    if pam_valence >= 3 and pam_arousal <= 2:
        pam_quadrant = "Q1"      # coşkulu
    elif pam_valence >= 3 and pam_arousal >= 3:
        pam_quadrant = "Q2"      # sakin
    elif pam_valence <  3 and pam_arousal <= 2:
        pam_quadrant = "Q3"      # anksiyete
    else:
        pam_quadrant = "Q4"      # depresyon

    # PHQ-4
    anx = int(ema_tahmin["phq4_q1"]) + int(ema_tahmin["phq4_q2"])
    dep = int(ema_tahmin["phq4_q3"]) + int(ema_tahmin["phq4_q4"])
    tot = anx + dep
    if tot <= PHQ4_NORMAL_USTU:
        phq4_risk = 0
    elif tot <= PHQ4_HAFIF_USTU:
        phq4_risk = 1
    elif tot <= PHQ4_ORTA_USTU:
        phq4_risk = 2
    else:
        phq4_risk = 3
    gad2_pozitif = anx >= 3
    phq2_pozitif = dep >= 3

    # Social delta — tek örnekte teorik ölçek aralığıyla kaba oran
    if obj_iletisim is None:
        social_delta = 0.0
        s_norm = 0.0
        o_norm = 0.0
    else:
        s_norm = (social_level - 1) / (5 - 1 + NORM_EPSILON)
        o_norm = float(obj_iletisim) / (100.0 + NORM_EPSILON)
        o_norm = float(np.clip(o_norm, 0.0, 1.0))
        social_delta = s_norm - o_norm

    # Profil eşleme + klinik risk
    profil_id, profil_isim, profil_risk = en_yakin_profil_bul(
        stress_z=stress_z,
        pam_quadrant=pam_quadrant,
        social_delta=social_delta,
        phq4_anksiyete=anx,
        phq4_depresyon=dep,
        phq4_total=tot,
    )
    klinik_risk, klinik_aciklama = risk_sinifi_4_hesapla(
        stress_z=stress_z,
        pam_quadrant=pam_quadrant,
        social_delta=social_delta,
        phq4_total=tot,
    )

    # İki karar farklıysa yüksek olanı al (güvenli taraf)
    risk_sinifi = max(profil_risk, klinik_risk)
    risk_isim, risk_renk = RISK_META[risk_sinifi]

    # Sonraki adım önerisi
    if gad2_pozitif and phq2_pozitif:
        sonraki_adim = "PHQ-9 ve GAD-7 tam ölçeklerinin uygulanması önerilir."
    elif gad2_pozitif:
        sonraki_adim = "GAD-7 tam ölçeği uygulanmalı."
    elif phq2_pozitif:
        sonraki_adim = "PHQ-9 tam ölçeği uygulanmalı."
    else:
        sonraki_adim = "Mevcut EMA takibi sürdürülebilir."

    return {
        "ema_tahmin": dict(ema_tahmin),
        "turetilen": {
            "stress_z":        round(float(stress_z), 3),
            "pam_valence":     int(pam_valence),
            "pam_arousal":     int(pam_arousal),
            "pam_quadrant":    pam_quadrant,
            "phq4_anksiyete":  int(anx),
            "phq4_depresyon":  int(dep),
            "phq4_total":      int(tot),
            "phq4_risk":       int(phq4_risk),
            "social_subj_norm": round(float(s_norm), 3),
            "social_obj_norm":  round(float(o_norm), 3),
            "social_delta":     round(float(social_delta), 3),
        },
        "profil_id":   profil_id,
        "profil_isim": profil_isim,
        "risk_sinifi": int(risk_sinifi),
        "risk_isim":   risk_isim,
        "risk_renk":   risk_renk,
        "aciklama":    klinik_aciklama,
        "oneri":       RISK_ONERI[risk_sinifi],
        "klinik_baglami": {
            "gad2_pozitif":  bool(gad2_pozitif),
            "phq2_pozitif":  bool(phq2_pozitif),
            "sonraki_adim":  sonraki_adim,
        },
    }
