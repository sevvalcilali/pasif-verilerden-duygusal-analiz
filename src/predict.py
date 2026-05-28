"""
predict_mood() — tek bir kullanıcı-gün için risk tahmini.

7 EMA cevabı + pasif veri alır; cascade (kural), hibrit ML ve pasif ML olmak
üzere 3 sistemden çıktı üretip birleşik bir dict döner (final_risk, renk,
top nedenler, öneri).

    from src.predict import predict_mood
    sonuc = predict_mood(uid=..., gun=..., ema={...}, pasif={...})
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")

# Hibrit, pasif ve forecasting model yolları
HIBRIT_MODEL_PATH      = os.path.join(MODELS, "best_hybrid.pkl")
PASIF_MODEL_PATH       = os.path.join(MODELS, "best_clean_labels.pkl")
FORECASTING_MODEL_PATH = os.path.join(MODELS, "best_forecasting.pkl")

# Risk metaverisi
RISK_META = {
    0: ("İyi Durum",   "yeşil",    "Genel duygusal denge iyi durumda."),
    1: ("Hafif Risk",  "sarı",     "Hafif düzeyde stres veya duygu durum belirtisi."),
    2: ("Orta Risk",   "turuncu",  "Orta düzeyde anksiyete/depresyon belirtisi."),
    3: ("Yüksek Risk", "kırmızı",  "Yüksek risk sinyali — klinik değerlendirme önerilir."),
}

ONERI = {
    0: "Mevcut günlük rutini koru. Uyku, fiziksel aktivite ve sosyal teması sürdür.",
    1: "Stres yönetimi ve düzenli uyku önerilir. EMA takibini sürdür.",
    2: "Üniversite psikolojik danışma birimine başvurmak değerlendirilebilir.",
    3: "Acil destek hattı veya klinik psikolog başvurusu önerilir. Tek başına bırakılmamalı.",
}

# 0. PSİKOLOJİK ANALİZ — Kural Izgarası (EMA puanı × Pasif risk)
# 12 hücrelik karar tablosu: PHQ-4 (4 düzey) × Pasif risk (3 düzey)
# + stres/sosyallik/PAM modifier'ları. Çıktı: kişiselleştirilmiş
# Türkçe yorum + öneri metni. Öğrenilmiş ML değil — klinik karar
# destek (CDSS) tarzı kural-tabanlı entegrasyon.

def _phq4_duzey(total: int) -> str:
    if total <= 2:  return "duşuk"
    if total <= 5:  return "hafif"
    if total <= 8:  return "orta"
    return "yuksek"

def _pasif_duzey(olasilik: float) -> str:
    if olasilik < 0.30: return "duşuk"
    if olasilik < 0.60: return "orta"
    return "yuksek"

def _phq4_metin(d: str) -> str:
    return {"duşuk":"düşük (normal)","hafif":"hafif düzey","orta":"orta düzey","yuksek":"yüksek düzey"}[d]

def _pasif_metin(d: str) -> str:
    return {"duşuk":"düşük","orta":"orta","yuksek":"yüksek"}[d]

def _hucre_yorum(phq4_d: str, pasif_d: str) -> tuple[str, str, str]:
    """12 hücre: (birlesik_yorum, oneri_baslik, renk_etiketi)"""
    K = (phq4_d, pasif_d)
    # PHQ-4 DÜŞÜK satırı
    if K == ("duşuk", "duşuk"):
        return ("Hem klinik tarama hem davranışsal sinyal iyi durumu işaret ediyor. İki sistem aynı yönde — değerlendirmenin güvenilirliği yüksek.",
                "Mevcut günlük rutininizi koruyun — düzenli uyku, fiziksel aktivite ve sosyal teması sürdürmek bu dengeyi destekliyor.",
                "yeşil")
    if K == ("duşuk", "orta"):
        return ("Anketiniz iyi durumu gösteriyor ama telefon davranışınızda hafif farklılaşma var. Sessiz bir değişim başlıyor olabilir.",
                "Son haftalardaki yaşam ritminize göz atın: uyku düzeni, hareket miktarı, sosyal etkileşim. Erken farkındalık koruyucudur.",
                "sarı")
    if K == ("duşuk", "yuksek"):
        return ("Anketiniz iyi durumu işaret etse de telefon davranış örüntünüzde dikkat çekici bir risk sinyali var — iki sistem çelişiyor.",
                "Davranışsal değişiklik anket cevaplarınızın önünde gidiyor olabilir (uyku, sosyallik veya hareket düzeninizde son değişiklik var mı?). Mevcut iyilik halinizi korumak için yaşam tarzı düzeninizi gözden geçirin.",
                "sarı")
    # PHQ-4 HAFİF satırı
    if K == ("hafif", "duşuk"):
        return ("Anketiniz hafif düzey ruhsal yük gösteriyor; davranışsal sinyaliniz ise henüz risk işareti vermiyor. Muhtemelen geçici stres dönemi.",
                "Bu hafta için stres yönetimi ve dinlenme önerilir. Eğer 2 hafta sürerse psikolojik danışma değerlendirilebilir.",
                "sarı")
    if K == ("hafif", "orta"):
        return ("Hem klinik hem davranışsal sinyaller hafif-orta düzey ruhsal yükü işaret ediyor. İki sistem benzer yönde.",
                "EMA takibini sürdürün, uyku ve sosyal teması korumak öncelik. Durum iki haftadan uzun sürerse psikolojik danışma birimine başvurun.",
                "turuncu")
    if K == ("hafif", "yuksek"):
        return ("Anket hafif düzey gösteriyor ama telefon davranışınız belirgin risk sinyali veriyor — davranışsal değişim klinik tabloyu geçmiş olabilir.",
                "Üniversite psikolojik danışma birimine başvurmayı değerlendirin. Bu arada uyku ve sosyallik takibi önemli.",
                "turuncu")
    # PHQ-4 ORTA satırı
    if K == ("orta", "duşuk"):
        return ("Anketiniz orta düzey ruhsal yük gösteriyor; davranışsal sinyaliniz ise stabil. Klinik baskınlığı, davranışsal etki henüz görünmüyor.",
                "Üniversite psikolojik danışma birimine başvurmak değerlendirilebilir. Davranışsal düzenin korunması koruyucu faktör.",
                "turuncu")
    if K == ("orta", "orta"):
        return ("Hem anketiniz hem davranışsal örüntünüz orta düzey risk işaret ediyor. İki sistem aynı yönde — değerlendirmenin güvenilirliği yüksek.",
                "Üniversite psikolojik danışma birimine başvurmanız önerilir. Düzenli uyku, fiziksel aktivite ve sosyal temas destekleyici olur.",
                "turuncu")
    if K == ("orta", "yuksek"):
        return ("Anketiniz orta düzey, davranış örüntünüz yüksek risk gösteriyor. İki sistem birlikte ciddi bir tabloyu işaret ediyor.",
                "Üniversite psikolojik danışma birimine başvurmanız önerilir. Davranışsal göstergelere göre acilen destek alınması önerilir.",
                "kırmızı")
    # PHQ-4 YÜKSEK satırı
    if K == ("yuksek", "duşuk"):
        return ("Anketiniz yüksek düzey ruhsal yük (akut) gösteriyor; davranışsal sinyaliniz şu an stabil. Klinik tablo öncelikli.",
                "Acil destek hattı veya klinik psikolog başvurusu önerilir. Tek başına bırakılmamalıdır.",
                "kırmızı")
    if K == ("yuksek", "orta"):
        return ("Anketiniz yüksek düzey gösteriyor, davranışsal sinyaliniz de risk içeriyor. İki sistem aynı yönde ciddi bir tabloyu işaret ediyor.",
                "Acil destek hattı veya klinik psikolog başvurusu önerilir. Tek başına bırakılmamalıdır.",
                "kırmızı")
    if K == ("yuksek", "yuksek"):
        return ("Hem anketiniz hem davranış örüntünüz şiddetli/akut risk işaret ediyor. İki sistem ortak ve çok yüksek riski gösteriyor.",
                "ACİL destek hattı veya klinik psikolog başvurusu kritik öneme sahiptir. Tek başına bırakılmamalıdır.",
                "kırmızı")
    return ("Değerlendirme sonucu belirsiz.", "EMA takibini sürdürün.", "sarı")

def psikolojik_analiz(ema: dict, cascade_out: dict, pasif_ml_sonuc: Any) -> dict:
    """
    EMA puanları + pasif ML çıktısı + diğer EMA değişkenlerinden
    kural-tabanlı kişiselleştirilmiş psikolojik yorum + öneri üretir.
    """
    phq4 = int(cascade_out["phq4_total"])
    anks = int(cascade_out["phq4_anksiyete"])
    dep  = int(cascade_out["phq4_depresyon"])
    pam_q = cascade_out["pam_quadrant"]
    profil = cascade_out["profil_isim"]
    stress = int(ema["stress"])
    social = int(ema["social_level"])
    pam_s = int(ema["pam_score"])

    pasif_olas = None
    if pasif_ml_sonuc and isinstance(pasif_ml_sonuc, dict) and "olasilik" in pasif_ml_sonuc:
        pasif_olas = float(pasif_ml_sonuc["olasilik"])

    # Pasif sinyal yoksa sadece klinik yorumla
    # Akademik dayanak metni — her cevapta sabit. Kullanıcıya yöntemin
    # bilimsel temelini gösterir, "uydurma yorum" şüphesini önler.
    dayanak = (
        "Yöntem: PHQ-4 klinik tarama (Kroenke ve ark., 2009) × pasif sensing "
        "risk modeli (Dartmouth StudentLife eğitimli; Wang 2014, Saeb 2015 "
        "paradigması) çapraz değerlendirme tablosu. CDSS (Clinical Decision "
        "Support System) yaklaşımı — kural-tabanlı klinik karar destek, "
        "tıbbi tanı değildir."
    )

    if pasif_olas is None:
        ema_yorum = (f"Anketiniz {_phq4_metin(_phq4_duzey(phq4))} ruhsal yük gösteriyor "
                     f"(PHQ-4: {phq4}/12). Profil: {profil}.")
        return {
            "ema_yorum": ema_yorum,
            "pasif_yorum": "Telefon davranış analizi şu an mevcut değil.",
            "birlesik_yorum": "Sadece klinik tarama değerlendirildi (pasif veri yok).",
            "guvenilirlik": "orta — sadece EMA-temelli",
            "oneri": "EMA takibini sürdürün; davranışsal veri biriktikçe değerlendirme zenginleşir.",
            "metin_butun": ema_yorum,
            "renk": "sarı",
            "phq4_duzey": _phq4_duzey(phq4),
            "pasif_duzey": None,
            "dayanak": dayanak,
        }

    phq4_d = _phq4_duzey(phq4)
    pasif_d = _pasif_duzey(pasif_olas)

    # EMA detay metni
    ema_parca = [f"PHQ-4 toplam puanınız {phq4}/12 ({_phq4_metin(phq4_d)})"]
    if anks >= 3:
        ema_parca.append(f"anksiyete alt-skoru {anks} (≥3 pozitif tarama)")
    if dep >= 3:
        ema_parca.append(f"depresyon alt-skoru {dep} (≥3 pozitif tarama)")
    pam_bolge = {"Q1": "yüksek enerji – olumlu", "Q2": "düşük enerji – olumlu (sakin)",
                 "Q3": "yüksek enerji – olumsuz (anksiyeteli)", "Q4": "düşük enerji – olumsuz (durgun)"}.get(pam_q, pam_q)
    ema_parca.append(f"ruh hali bölgeniz {pam_bolge}")
    if stress >= 4:
        ema_parca.append(f"stres seviyeniz yüksek ({stress}/5)")
    if social <= 2:
        ema_parca.append(f"sosyallik puanınız düşük ({social}/5 — izole eğilimi)")
    ema_yorum = "Anket bulgularınız: " + ", ".join(ema_parca) + "."

    # Pasif detay metni
    pasif_yorum = (f"Telefon davranış analizi: pasif risk sinyaliniz "
                   f"%{int(pasif_olas*100)} ({_pasif_metin(pasif_d)} düzey).")

    # 12 hücre yorumu
    birlesik, oneri_ana, renk = _hucre_yorum(phq4_d, pasif_d)

    # Modifier'lar — öneriyi zenginleştir
    oneri_ek = []
    if stress >= 4:
        oneri_ek.append("Stres seviyeniz yüksek — nefes/rahatlama egzersizleri, telefon kullanım sürelerinin kontrolü yardımcı olabilir.")
    if social <= 2:
        oneri_ek.append("Sosyallik puanınız düşük — haftada en az 1-2 sosyal temas hedeflemek faydalı.")
    if pam_q == "Q4" and phq4 >= 6:
        oneri_ek.append("Ruh haliniz durgun-olumsuz bölgede; fiziksel aktivite (günde 20-30 dk yürüyüş) ruh halini destekleyebilir.")
    if pam_q == "Q3" and anks >= 3:
        oneri_ek.append("Anksiyete sinyali baskın — gevşeme teknikleri ve uyku düzenine odaklanmak önerilir.")

    # Güvenilirlik (iki sistemin uyumu)
    if (phq4_d in ("orta","yuksek")) == (pasif_d in ("orta","yuksek")):
        guvenilirlik = "yüksek — klinik tarama ve davranışsal sinyal aynı yönde"
    else:
        guvenilirlik = "orta — klinik tarama ve davranışsal sinyal farklı yönde, daha fazla veri ile kesinleşir"

    oneri_butun = oneri_ana + (" " + " ".join(oneri_ek) if oneri_ek else "")

    metin_butun = f"{ema_yorum} {pasif_yorum} {birlesik} Güvenilirlik: {guvenilirlik}. Öneri: {oneri_butun}"

    return {
        "ema_yorum": ema_yorum,
        "pasif_yorum": pasif_yorum,
        "birlesik_yorum": birlesik,
        "guvenilirlik": guvenilirlik,
        "oneri": oneri_butun,
        "metin_butun": metin_butun,
        "renk": renk,
        "phq4_duzey": phq4_d,
        "pasif_duzey": pasif_d,
        "dayanak": dayanak,
    }

# 1. EMA Cascade — kural-tabanlı (EMA/src/risk_classifier'dan)

def cascade_risk(ema: dict) -> dict:
    """ema.md cascade hiyerarşik kural sistemine göre risk."""

    # PHQ-4 hesapla
    anksiyete = int(ema["phq4_q1"]) + int(ema["phq4_q2"])
    depresyon = int(ema["phq4_q3"]) + int(ema["phq4_q4"])
    total     = anksiyete + depresyon

    # PAM quadrant (Pollak 2011 konvansiyonu — düzeltilmiş)
    pam = int(ema["pam_score"])
    valence = ((pam - 1) % 4) + 1
    arousal = ((pam - 1) // 4) + 1
    # arousal ≤ 2 → yüksek (üst satır), > 2 → düşük (alt satır)
    if valence >= 3 and arousal <= 2:
        quadrant = "Q1"   # Coşkulu
    elif valence >= 3 and arousal > 2:
        quadrant = "Q2"   # Sakin
    elif valence < 3 and arousal <= 2:
        quadrant = "Q3"   # Anksiyete
    else:
        quadrant = "Q4"   # Depresyon

    stress = int(ema["stress"])
    social = int(ema["social_level"])
    anks_pozitif = anksiyete >= 3
    dep_pozitif  = depresyon >= 3
    anks_akut    = anksiyete >= 5
    dep_akut     = depresyon >= 5
    yuksek_stres = stress >= 4
    orta_stres   = stress == 3
    dusuk_stres  = stress <= 2
    izole        = social <= 2
    sosyal_yuksek = social >= 4

    # Cascade öncelik sırası
    if dep_akut and quadrant == "Q4":
        profil = ("J", "Akut Depresif (Kriz Modu 2)", 3)
    elif anks_akut and quadrant == "Q3":
        profil = ("I", "Akut Anksiyete (Kriz Modu 1)", 3)
    elif (anks_pozitif or dep_pozitif) and quadrant in ("Q1", "Q2"):
        profil = ("L", "Görünmez Kriz (Yalnızlık)", 2)
    elif yuksek_stres and 3 <= total <= 5 and quadrant == "Q2":
        profil = ("K", "Maskeli Burnout", 2)
    elif anks_pozitif and dep_pozitif:
        profil = ("H", "Karma Yüksek Risk", 3)
    elif dep_pozitif and izole:
        profil = ("G", "Orta Depresif + Yalnızlık", 2)
    elif anks_pozitif and izole:
        profil = ("E", "Orta Anksiyete + İzolasyon", 2)
    # Hafif tek-yönlü profiller — sadece PHQ-4 puanı değil, PAM quadrantı da
    # aynı yönde sinyal vermeli (EMA içi tutarlılık). "Birkaç gün gergin"
    # ama PAM'da pozitif/sakin quadrantı seçilmişse hafif etiket vermiyoruz.
    elif depresyon >= 2 and quadrant == "Q4" and not anks_pozitif:
        profil = ("F", "Hafif Depresif Eğilim", 1)
    elif anksiyete >= 2 and quadrant == "Q3" and not dep_pozitif:
        profil = ("D", "Hafif Anksiyete", 1)
    elif total <= 2 and (orta_stres or yuksek_stres):
        profil = ("C", "Normal Akademik Stres", 0)
    elif total <= 2 and quadrant == "Q1" and not yuksek_stres:
        profil = ("B", "Üretken Coşku", 0)
    elif total <= 2 and quadrant in ("Q1", "Q2") and dusuk_stres and sosyal_yuksek:
        profil = ("A", "Optimum Denge / Flow", 0)
    else:
        profil = ("C", "Normal Akademik Stres (default)", 0)

    return {
        "risk_sinifi":    profil[2],
        "profil_id":      profil[0],
        "profil_isim":    profil[1],
        "phq4_total":     total,
        "phq4_anksiyete": anksiyete,
        "phq4_depresyon": depresyon,
        "pam_quadrant":   quadrant,
    }

# 2. EMA türetilmiş feature'ları hesapla

def turet_ema_features(ema: dict, obj_iletisim: float = 0.0) -> dict:
    """Master dataset'teki türetilmiş EMA feature'larını hesapla."""

    cascade_out = cascade_risk(ema)
    pam = int(ema["pam_score"])
    valence = ((pam - 1) % 4) + 1
    arousal = ((pam - 1) // 4) + 1

    anksiyete = cascade_out["phq4_anksiyete"]
    depresyon = cascade_out["phq4_depresyon"]
    total = cascade_out["phq4_total"]
    quadrant = cascade_out["pam_quadrant"]

    # PHQ-4 risk sınıfı
    if total <= 2:
        phq4_risk = 0
    elif total <= 5:
        phq4_risk = 1
    elif total <= 8:
        phq4_risk = 2
    else:
        phq4_risk = 3

    # Social delta (basit normalizasyon)
    social_subj_norm = (int(ema["social_level"]) - 1) / 4.0
    # obj_iletisim normalize: 0-100 sınırı kullan
    social_obj_norm = min(obj_iletisim, 100.0) / 100.0
    social_delta = social_subj_norm - social_obj_norm

    return {
        "stress":         int(ema["stress"]),
        "pam_score":      pam,
        "social_level":   int(ema["social_level"]),
        "phq4_q1":        int(ema["phq4_q1"]),
        "phq4_q2":        int(ema["phq4_q2"]),
        "phq4_q3":        int(ema["phq4_q3"]),
        "phq4_q4":        int(ema["phq4_q4"]),
        "stress_z":       0.0,   # tek satır için 0
        "pam_valence":    valence,
        "pam_arousal":    arousal,
        "phq4_anksiyete": anksiyete,
        "phq4_depresyon": depresyon,
        "phq4_total":     total,
        "phq4_risk":      phq4_risk,
        "gad2_pozitif":   1 if anksiyete >= 3 else 0,
        "phq2_pozitif":   1 if depresyon >= 3 else 0,
        "social_subj_norm": social_subj_norm,
        "social_obj_norm":  social_obj_norm,
        "social_delta":     social_delta,
        "obj_iletisim":     obj_iletisim,
        "pam_q_Q1":         1 if quadrant == "Q1" else 0,
        "pam_q_Q2":         1 if quadrant == "Q2" else 0,
        "pam_q_Q3":         1 if quadrant == "Q3" else 0,
        "pam_q_Q4":         1 if quadrant == "Q4" else 0,
    }

# 3. Modelleri yükle (cache)

_MODEL_CACHE: dict[str, Any] = {}

def get_model(path: str) -> dict:
    """Modeli cache'le, tekrar yüklemekten kaçın."""
    if path not in _MODEL_CACHE:
        with open(path, "rb") as f:
            _MODEL_CACHE[path] = pickle.load(f)
    return _MODEL_CACHE[path]

# 4. Pasif veriyi modelin beklediği şekilde hazırla

def pasif_feature_vektoru(pasif: dict, feature_kollar: list[str]) -> pd.DataFrame:
    """
    Verilen pasif dict'ten, model'in beklediği SIRALI feature vektörü üret.
    Eksik feature'lar 0 ile doldurulur.
    """
    row = {f: pasif.get(f, 0.0) for f in feature_kollar}
    return pd.DataFrame([row])

# 5. ANA FONKSİYON — predict_mood()

def predict_mood(
    uid: str,
    gun: str,
    ema: dict,
    pasif: dict | None = None,
    obj_iletisim: float = 0.0,
) -> dict:
    """
    Bir kullanıcının bir günü için risk tahmini.

    Parametreler:
        uid:           Kullanıcı kimliği (örn. "user_001")
        gun:           Tarih (YYYY-MM-DD veya YYYYMMDD)
        ema:           7 EMA cevabı:
                       {stress, pam_score, social_level,
                        phq4_q1, phq4_q2, phq4_q3, phq4_q4}
        pasif:         Pasif sensor verileri (None ise sadece EMA cascade kullanılır)
        obj_iletisim:  Sensing'den iletişim sayısı (call+sms toplamı)

    Returns:
        Risk değerlendirmesi içeren dict.
    """

    # 1. EMA Cascade
    cascade_out = cascade_risk(ema)

    # 2. EMA türetilmiş feature'lar
    ema_turetilen = turet_ema_features(ema, obj_iletisim=obj_iletisim)

    # 3. Hibrit ML tahmin (EMA + Pasif)
    hibrit_ml_sonuc = None
    if pasif is not None:
        try:
            hibrit_data = get_model(HIBRIT_MODEL_PATH)
            hibrit_model = hibrit_data["model"]
            feat_kollar = hibrit_data["feature_kollar"]

            # EMA + Pasif birleştir
            tum_input = {**ema_turetilen, **pasif}
            X = pasif_feature_vektoru(tum_input, feat_kollar)

            prob = hibrit_model.predict_proba(X)[0, 1]
            # Eşik: F1-opt threshold (kayıtlı best_thr_f1)
            esik = float(hibrit_data.get("best_thr_f1", 0.5))
            risk_binary = int(prob >= esik)

            hibrit_ml_sonuc = {
                "risk_binary":     risk_binary,   # 0/1
                "olasilik":        round(float(prob), 4),
                "kullanilan_esik": round(esik, 2),
                "feature_count":   len(feat_kollar),
            }
        except Exception as e:
            hibrit_ml_sonuc = {"hata": str(e)}

    # 4. Pasif-only ML tahmin (sadece davranış)
    pasif_ml_sonuc = None
    if pasif is not None:
        try:
            pasif_data = get_model(PASIF_MODEL_PATH)
            pasif_model = pasif_data["model"]
            feat_kollar = pasif_data["feature_kollar"]

            X = pasif_feature_vektoru(pasif, feat_kollar)
            prob = pasif_model.predict_proba(X)[0, 1]
            # Pasif modelin F1-opt threshold'u
            esik = float(pasif_data.get("best_thr_f1", 0.5))
            risk_binary = int(prob >= esik)

            pasif_ml_sonuc = {
                "risk_binary":     risk_binary,
                "olasilik":        round(float(prob), 4),
                "kullanilan_esik": round(esik, 2),
                "feature_count":   len(feat_kollar),
            }
        except Exception as e:
            pasif_ml_sonuc = {"hata": str(e)}

    # 4b. FORECASTING ML — Bugün EMA+Pasif → YARIN riski
    # Akademik temiz ANA MODEL (F1=0.97, leakage yok). Hibritle aynı feature
    # yapısı (183: EMA türetilen + pasif); fark sadece eğitimde hedefin yarın
    # olması. Bu yüzden inference hibritle aynı, model dosyası farklı.
    forecasting_sonuc = None
    if pasif is not None:
        try:
            fc_data = get_model(FORECASTING_MODEL_PATH)
            fc_model = fc_data["model"]
            feat_kollar = fc_data["feature_kollar"]

            tum_input = {**ema_turetilen, **pasif}
            X = pasif_feature_vektoru(tum_input, feat_kollar)

            prob = fc_model.predict_proba(X)[0, 1]
            esik = float(fc_data.get("best_thr_f1", 0.5))
            risk_binary = int(prob >= esik)

            forecasting_sonuc = {
                "risk_binary":     risk_binary,   # 0/1 → yarın yüksek risk mi
                "olasilik":        round(float(prob), 4),
                "kullanilan_esik": round(esik, 2),
                "feature_count":   len(feat_kollar),
                "auc":             round(float(fc_data.get("auc", 0.0)), 4),
                "hedef":           "yarin",
            }
        except Exception as e:
            forecasting_sonuc = {"hata": str(e)}

    # 5. Birleşik karar — cascade ana, pasif tie-breaker
    # Final risk = klinik validasyonlu cascade sınıfı (PHQ-4/PAM kuralları).
    # Hibrit ML'de data leakage var → final kararı EZMEZ, sadece destekleyici.
    # Ancak pasif ML açıkça düşük risk gösteriyorsa (olasılık < 0.30) cascade'in
    # "Hafif" (1) sinyali 0'a düşürülür — çünkü 7 günlük telefon davranışı
    # tutarlı olumlu sinyal veriyorsa hafif EMA sinyali aşırı-hassas sayılır.
    cascade_risk_sinifi = cascade_out["risk_sinifi"]   # 0-3
    final_risk = cascade_risk_sinifi
    pasif_tie_breaker_aktif = False
    if (
        final_risk == 1
        and pasif_ml_sonuc
        and "olasilik" in pasif_ml_sonuc
        and float(pasif_ml_sonuc["olasilik"]) < 0.30
    ):
        final_risk = 0
        pasif_tie_breaker_aktif = True

    risk_isim, risk_renk, aciklama = RISK_META[final_risk]
    oneri = ONERI[final_risk]

    # 6. Güvenilirlik analizi
    uyum_sayisi = 0
    if hibrit_ml_sonuc and "risk_binary" in hibrit_ml_sonuc:
        ml_yuksek = hibrit_ml_sonuc["risk_binary"] == 1
        cascade_yuksek = cascade_risk_sinifi == 3
        if ml_yuksek == cascade_yuksek:
            uyum_sayisi += 1
    if pasif_ml_sonuc and "risk_binary" in pasif_ml_sonuc:
        ml_yuksek = pasif_ml_sonuc["risk_binary"] == 1
        cascade_yuksek = cascade_risk_sinifi == 3
        if ml_yuksek == cascade_yuksek:
            uyum_sayisi += 1
    if uyum_sayisi == 2:
        guvenilirlik = "yüksek (cascade + hibrit + pasif aynı yönde)"
    elif uyum_sayisi == 1:
        guvenilirlik = "orta (1 sistem aynı, 1 farklı)"
    else:
        guvenilirlik = "düşük (sistemler farklı yönde — manuel kontrol)"

    # 7. Top 5 neden (cascade'den kural-tabanlı)
    top_5_neden = []
    if cascade_out["phq4_total"] >= 9:
        top_5_neden.append(f"PHQ-4 toplam={cascade_out['phq4_total']} (≥9 şiddetli)")
    elif cascade_out["phq4_total"] >= 6:
        top_5_neden.append(f"PHQ-4 toplam={cascade_out['phq4_total']} (6-8 orta)")
    elif cascade_out["phq4_total"] >= 3:
        top_5_neden.append(f"PHQ-4 toplam={cascade_out['phq4_total']} (3-5 hafif)")
    else:
        top_5_neden.append(f"PHQ-4 toplam={cascade_out['phq4_total']} (normal)")

    if cascade_out["phq4_anksiyete"] >= 3:
        top_5_neden.append(f"GAD-2 anksiyete={cascade_out['phq4_anksiyete']} (≥3 pozitif)")
    if cascade_out["phq4_depresyon"] >= 3:
        top_5_neden.append(f"PHQ-2 depresyon={cascade_out['phq4_depresyon']} (≥3 pozitif)")

    if int(ema["stress"]) >= 4:
        top_5_neden.append(f"Stres={ema['stress']} (yüksek)")
    if int(ema["social_level"]) <= 2:
        top_5_neden.append(f"Sosyallik={ema['social_level']} (izole)")

    quadrant_isim = {"Q1": "Coşkulu", "Q2": "Sakin",
                     "Q3": "Anksiyete", "Q4": "Depresif"}
    top_5_neden.append(f"PAM={ema['pam_score']} ({quadrant_isim[cascade_out['pam_quadrant']]} bölge)")

    if pasif_tie_breaker_aktif:
        olz = float(pasif_ml_sonuc["olasilik"])
        top_5_neden.append(
            f"Telefon davranışı tutarlı olumlu (pasif risk olasılığı %{int(olz*100)}) "
            f"→ hafif EMA sinyali iyi durum olarak değerlendirildi"
        )

    top_5_neden = top_5_neden[:5]

    # 8. Final çıktı
    return {
        "kullanici":    uid,
        "gun":          gun,
        "ema_girdi":    ema,
        "cascade": {
            "risk_sinifi":  cascade_risk_sinifi,
            "profil_id":    cascade_out["profil_id"],
            "profil_isim":  cascade_out["profil_isim"],
            "phq4_total":   cascade_out["phq4_total"],
            "pam_quadrant": cascade_out["pam_quadrant"],
        },
        "hibrit_ml":    hibrit_ml_sonuc,
        "pasif_ml":     pasif_ml_sonuc,
        "forecasting":  forecasting_sonuc,
        "psikolojik_analiz": psikolojik_analiz(ema, cascade_out, pasif_ml_sonuc),
        "final_risk":   final_risk,
        "final_isim":   risk_isim,
        "final_renk":   risk_renk,
        "guvenilirlik": guvenilirlik,
        "aciklama":     aciklama,
        "oneri":        oneri,
        "top_5_neden":  top_5_neden,
    }
