"""ema_processing ve risk_classifier'ı sentetik veriyle test eder."""

import json

import pandas as pd

from src.ema_processing import tum_islemleri_uygula
from src.risk_classifier import (
    en_yakin_profil_bul,
    risk_sinifi_4_hesapla,
    tahmin_yorumla,
)


def sentetik_ema_olustur() -> pd.DataFrame:
    """5 farklı profilde sentetik EMA. PAM: arousal=1 üst/yüksek, =4 alt/düşük."""
    return pd.DataFrame([
        # Q1 üretken coşku, düşük stres, sıfır PHQ → B
        {"uid": "u01", "gun": "2026-05-01", "stress": 2, "pam_score": 8,
         "social_level": 4, "phq4_q1": 0, "phq4_q2": 0, "phq4_q3": 0,
         "phq4_q4": 0, "obj_iletisim": 30},
        # Q4 depresif, yüksek PHQ + stres → H ya da J
        {"uid": "u01", "gun": "2026-05-02", "stress": 5, "pam_score": 13,
         "social_level": 1, "phq4_q1": 3, "phq4_q2": 3, "phq4_q3": 1,
         "phq4_q4": 1, "obj_iletisim": 50},
        # Q3 anksiyete, yüksek dep + kritik social_delta → G ya da F
        {"uid": "u02", "gun": "2026-05-01", "stress": 3, "pam_score": 2,
         "social_level": 2, "phq4_q1": 1, "phq4_q2": 1, "phq4_q3": 3,
         "phq4_q4": 3, "obj_iletisim": 5},
        # Q3, total=10, yüksek stres → H ya da J
        {"uid": "u02", "gun": "2026-05-02", "stress": 4, "pam_score": 1,
         "social_level": 1, "phq4_q1": 2, "phq4_q2": 2, "phq4_q3": 3,
         "phq4_q4": 3, "obj_iletisim": 40},
        # Q2 sakin yüz + yüksek stres + orta PHQ → K maskeli burnout
        {"uid": "u03", "gun": "2026-05-01", "stress": 4, "pam_score": 12,
         "social_level": 3, "phq4_q1": 2, "phq4_q2": 2, "phq4_q3": 2,
         "phq4_q4": 1, "obj_iletisim": 25},
    ])


# Beklenen profiller (PAM Pollak 2011 konvansiyonuna göre)
BEKLENEN = {
    ("u01", "2026-05-01"): {"profil": "B",   "not": "Üretken Coşku (Q1 + düşük PHQ)"},
    ("u01", "2026-05-02"): {"profil": "H/J", "not": "Karma Yüksek / Akut Depresif (Q4 + yüksek PHQ)"},
    ("u02", "2026-05-01"): {"profil": "G/F", "not": "Orta/Hafif Depresif (yüksek dep + kritik delta)"},
    ("u02", "2026-05-02"): {"profil": "H/J", "not": "Karma Yüksek / Akut Depresif"},
    ("u03", "2026-05-01"): {"profil": "K",   "not": "Maskeli Burnout"},
}


def main() -> None:
    print("=" * 60)
    print("EMA İŞLEME MODÜLÜ TESTİ")
    print("=" * 60)

    df = sentetik_ema_olustur()
    print(f"\nSentetik veri: {len(df)} satır, {len(df.columns)} kolon")

    df_islenmis = tum_islemleri_uygula(df)
    print(f"\nTüm işlemler uygulandı. Yeni kolonlar:")
    yeni = [k for k in df_islenmis.columns if k not in df.columns]
    for k in yeni:
        print(f"  - {k}")

    print(f"\nTüretilmiş skorlar:")
    gosterilecek = ["uid", "gun", "stress_z", "pam_quadrant",
                    "phq4_anksiyete", "phq4_depresyon", "phq4_total",
                    "phq4_risk", "social_delta"]
    print(df_islenmis[gosterilecek].round(3).to_string(index=False))

    print("\n" + "=" * 60)
    print("RISK CLASSIFIER TESTİ")
    print("=" * 60)

    eslesme_sayaci = {"tam": 0, "kabul": 0, "sapma": 0}
    for _, satir in df_islenmis.iterrows():
        profil = en_yakin_profil_bul(
            stress_z=satir["stress_z"],
            pam_quadrant=satir["pam_quadrant"],
            social_delta=satir["social_delta"],
            phq4_anksiyete=satir["phq4_anksiyete"],
            phq4_depresyon=satir["phq4_depresyon"],
            phq4_total=satir["phq4_total"],
        )
        risk = risk_sinifi_4_hesapla(
            stress_z=satir["stress_z"],
            pam_quadrant=satir["pam_quadrant"],
            social_delta=satir["social_delta"],
            phq4_total=satir["phq4_total"],
        )
        anahtar = (satir["uid"], satir["gun"])
        beklenen = BEKLENEN.get(anahtar, {"profil": "?", "not": "-"})

        if profil[0] == beklenen["profil"]:
            durum = "tam eşleşme"
            eslesme_sayaci["tam"] += 1
        elif profil[0] in beklenen["profil"].split("/"):
            durum = "alternatif kabul"
            eslesme_sayaci["kabul"] += 1
        else:
            durum = f"sapma (beklenen: {beklenen['profil']})"
            eslesme_sayaci["sapma"] += 1

        print(f"\n{satir['uid']} / {satir['gun']}")
        print(f"   Profil      : {profil[0]} - {profil[1]}")
        print(f"   Risk        : Sınıf {risk[0]}")
        print(f"   Beklenen    : {beklenen['profil']} ({beklenen['not']})")
        print(f"   Durum       : {durum}")

    print("\n" + "─" * 60)
    print(f"Eşleşme özeti: tam={eslesme_sayaci['tam']}, "
          f"alt-kabul={eslesme_sayaci['kabul']}, "
          f"sapma={eslesme_sayaci['sapma']}")

    print("\n" + "=" * 60)
    print("FINAL ÇIKTI FORMATI TESTİ (tahmin_yorumla)")
    print("=" * 60)

    ornek_tahmin = {
        "uid": "u01", "tarih": "2026-05-02",
        "stress": 4.2, "pam_score": 13, "social_level": 2.1,
        "phq4_q1": 2, "phq4_q2": 2, "phq4_q3": 1, "phq4_q4": 1,
        "obj_iletisim": 40.0,
    }
    sonuc = tahmin_yorumla(ornek_tahmin)
    print(json.dumps(sonuc, indent=2, ensure_ascii=False))

    print("\nTüm testler tamamlandı.")


if __name__ == "__main__":
    main()
