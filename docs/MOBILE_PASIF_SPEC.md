# Mobil Pasif Veri Toplama — Teknik Spec

> Telefon (Kotlin) tarafının uyması gereken teknik şartname.
> Telefondan hangi feature, hangi Android API ile toplanır, nasıl gönderilir.

---

## Genel Akış

```
1. App izinleri ister (ADIM 2A — yapıldı)
2. Arka plan servisi saatlik ham veri toplar (ADIM 2B)
3. Günlük: ham veri epoch'lara bölünür (ADIM 2C)
4. EMA + pasif birleştirilip backend'e gönderilir (ADIM 2E)
5. Sonuç ekranda + "Toplanan Veriler" listesinde (ADIM 2D)
```

---

## Backend Endpoint

```
POST http://10.0.2.2:8000/predict/mobile     (emulator)
POST http://192.168.1.4:8000/predict/mobile  (gerçek cihaz, aynı WiFi)
Content-Type: application/json
```

**ÖNEMLİ**: `/predict/quick` değil, **`/predict/mobile`** kullanılacak (yeni endpoint).

---

## İSTEK FORMATI (Kotlin → Backend)

```json
{
  "uid": "telefon_kullanici_001",
  "gun": "2026-05-16",
  "ema": {
    "stress": 4,
    "pam_score": 12,
    "social_level": 2,
    "phq4_q1": 2,
    "phq4_q2": 2,
    "phq4_q3": 3,
    "phq4_q4": 3
  },
  "pasif": {
    "unlock_num_ep_0": 145.0,
    "unlock_num_ep_1": 12.0,
    "act_still_ep_0": 64800.0,
    "loc_dist_ep_0": 8500.0
  }
}
```

**KURAL**: `pasif` içine SADECE toplayabildiğin feature'ları koy.
Toplayamadığını **HİÇ EKLEME** (null/0 değil). Backend medyanla doldurur.

---

## TOPLANACAK FEATURE LİSTESİ (35 feature)

### GRUP 1: Telefon Kullanımı (8 feature)

| Feature | Tip | Birim | Android API | Hesaplama |
|---|---|---|---|---|
| `unlock_num_ep_0` | Double | adet | `UsageStatsManager` | Tüm gün KEYGUARD_HIDDEN event sayısı |
| `unlock_num_ep_1` | Double | adet | `UsageStatsManager` | Saat 00:00-09:00 arası unlock |
| `unlock_num_ep_2` | Double | adet | `UsageStatsManager` | Saat 09:00-18:00 arası unlock |
| `unlock_num_ep_3` | Double | adet | `UsageStatsManager` | Saat 18:00-24:00 arası unlock |
| `unlock_duration_ep_0` | Double | saniye | `UsageStatsManager` | Tüm gün ekran açık toplam süre |
| `unlock_duration_ep_1` | Double | saniye | `UsageStatsManager` | 00-09 ekran süre |
| `unlock_duration_ep_2` | Double | saniye | `UsageStatsManager` | 09-18 ekran süre |
| `unlock_duration_ep_3` | Double | saniye | `UsageStatsManager` | 18-24 ekran süre |

**UsageStats kodu örneği:**
```kotlin
val usm = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
val events = usm.queryEvents(baslangic, bitis)
val event = UsageEvents.Event()
var unlockCount = 0
var screenOnTime = 0L
var lastScreenOn = 0L
while (events.hasNextEvent()) {
    events.getNextEvent(event)
    when (event.eventType) {
        UsageEvents.Event.KEYGUARD_HIDDEN -> {
            unlockCount++
            lastScreenOn = event.timeStamp
        }
        UsageEvents.Event.SCREEN_NON_INTERACTIVE -> {
            if (lastScreenOn > 0) screenOnTime += event.timeStamp - lastScreenOn
        }
    }
}
```

### GRUP 2: Aktivite (15 feature)

| Feature | Tip | Birim | Android API | Hesaplama |
|---|---|---|---|---|
| `act_still_ep_0` | Double | saniye | `ActivityRecognition` | Tüm gün STILL'de geçen toplam süre |
| `act_still_ep_1/2/3` | Double | saniye | `ActivityRecognition` | Epoch bazlı STILL |
| `act_walking_ep_0` | Double | saniye | `ActivityRecognition` | Tüm gün WALKING süre |
| `act_walking_ep_1/2/3` | Double | saniye | `ActivityRecognition` | Epoch bazlı WALKING |
| `act_on_foot_ep_0` | Double | saniye | `ActivityRecognition` | Tüm gün ON_FOOT (yürüme+koşma) |
| `act_on_foot_ep_1/2/3` | Double | saniye | `ActivityRecognition` | Epoch bazlı ON_FOOT |
| `act_running_ep_0` | Double | saniye | `ActivityRecognition` | Tüm gün RUNNING |
| `act_on_bike_ep_0` | Double | saniye | `ActivityRecognition` | Tüm gün ON_BICYCLE |
| `act_in_vehicle_ep_0` | Double | saniye | `ActivityRecognition` | Tüm gün IN_VEHICLE |

**ActivityRecognition kodu örneği:**
```kotlin
// ActivityRecognitionClient ile activity transition'ları dinle
// Her aktivite tipini ve süresini SQLite'e kaydet
// DetectedActivity.STILL, WALKING, RUNNING, ON_BICYCLE, IN_VEHICLE, ON_FOOT
val client = ActivityRecognition.getClient(context)
client.requestActivityUpdates(60000L, pendingIntent)  // 60sn aralık
// BroadcastReceiver'da gelen aktiviteyi süreyle birlikte kaydet
```

### GRUP 3: Konum (5 feature)

| Feature | Tip | Birim | Android API | Hesaplama |
|---|---|---|---|---|
| `loc_dist_ep_0` | Double | metre | `FusedLocationProvider` | Tüm gün konumlar arası toplam mesafe |
| `loc_dist_ep_1/2/3` | Double | metre | `FusedLocationProvider` | Epoch bazlı mesafe |
| `loc_visit_num_ep_0` | Double | adet | `FusedLocationProvider` | Farklı durulan yer sayısı (clustering) |

**Mesafe kodu örneği:**
```kotlin
// Saatlik konum noktaları arası Haversine mesafe topla
fun toplamMesafe(noktalar: List<Location>): Double {
    var total = 0.0
    for (i in 1 until noktalar.size) {
        total += noktalar[i-1].distanceTo(noktalar[i])
    }
    return total
}
```

### GRUP 4: Medya (2 feature)

| Feature | Tip | Birim | Android API | Hesaplama |
|---|---|---|---|---|
| `other_playing_duration_ep_0` | Double | saniye | `UsageStatsManager` | Medya app'lerinde geçen süre |
| `other_playing_num_ep_0` | Double | adet | `UsageStatsManager` | Medya app açma sayısı |

### GRUP 5: GÖNDERME — Backend Hesaplar

Bunları **Kotlin'de hesaplama, GÖNDERME**. Backend ham veriden türetir:
```
sedanter_saat, aktivite_toplam, mobilite_skoru,
gece_telefon_yogunluk, gunduz_gece_telefon_orani
```
*(Eğer göndermezsen backend medyan kullanır — sorun değil)*

---

## EPOCH HESAPLAMA (Kritik!)

Ham saatlik veriyi şu zaman dilimlerine böl:

```
ep_0 = TÜM GÜN       → 00:00 - 23:59 toplamı
ep_1 = GECE          → 00:00 - 09:00 toplamı
ep_2 = GÜNDÜZ        → 09:00 - 18:00 toplamı
ep_3 = AKŞAM         → 18:00 - 24:00 toplamı
```

**Kotlin pseudo-kod:**
```kotlin
fun epochHesapla(saatlikVeri: Map<Int, Double>): Map<String, Double> {
    val ep0 = saatlikVeri.values.sum()                    // tüm gün
    val ep1 = (0..8).sumOf { saatlikVeri[it] ?: 0.0 }     // 00-09
    val ep2 = (9..17).sumOf { saatlikVeri[it] ?: 0.0 }    // 09-18
    val ep3 = (18..23).sumOf { saatlikVeri[it] ?: 0.0 }   // 18-24
    return mapOf("ep_0" to ep0, "ep_1" to ep1, "ep_2" to ep2, "ep_3" to ep3)
}
```

---

## CEVAP FORMATI (Backend → Kotlin)

```json
{
  "kullanici": "telefon_kullanici_001",
  "gun": "2026-05-16",
  "cascade": {
    "risk_sinifi": 3,
    "profil_id": "J",
    "profil_isim": "Akut Depresif (Kriz Modu 2)",
    "phq4_total": 12,
    "pam_quadrant": "Q4"
  },
  "hibrit_ml": {
    "risk_binary": 1,
    "olasilik": 0.9959,
    "kullanilan_esik": 0.11,
    "feature_count": 183
  },
  "pasif_ml": {
    "risk_binary": 0,
    "olasilik": 0.3498,
    "kullanilan_esik": 0.47,
    "feature_count": 159
  },
  "final_risk": 3,
  "final_isim": "Yüksek Risk",
  "final_renk": "kırmızı",
  "guvenilirlik": "orta (1 sistem aynı, 1 farklı)",
  "aciklama": "Yüksek risk sinyali — klinik değerlendirme önerilir.",
  "oneri": "Acil destek hattı veya klinik psikolog başvurusu önerilir.",
  "top_5_neden": [
    "PHQ-4 toplam=12 (≥9 şiddetli)",
    "GAD-2 anksiyete=6 (≥3 pozitif)"
  ],
  "mobil_meta": {
    "telefondan_gelen_feature": 4,
    "medyanla_doldurulan_feature": 155,
    "toplam_pasif_feature": 159,
    "reddedilen_bilinmeyen_feature": [],
    "gelen_feature_listesi": ["act_still_ep_0", "loc_dist_ep_0", ...]
  }
}
```

### `mobil_meta` Ne İşe Yarar?

Kullanıcıya **şeffaflık** için. "Toplanan Veriler" ekranında göster:
```
 Veri Özeti
  Telefondan toplanan: 4 ölçüm
  Tahmini değerle dolduruldu: 155
  Toplam analiz edilen: 159
```

---

## Kotlin Data Class Güncellemesi

```kotlin
// Mevcut TahminIstegi'ye pasif zaten eklenmişti, aynı kalır
data class TahminIstegi(
    val uid: String,
    val gun: String,
    val ema: EMAGirdi,
    val pasif: Map<String, Double> = emptyMap()  // ← buraya toplanan veri
)

// CEVABA mobil_meta ekle
data class MobilMeta(
    @SerializedName("telefondan_gelen_feature")
    val telefondanGelen: Int,
    @SerializedName("medyanla_doldurulan_feature")
    val medyanlaDolduruldu: Int,
    @SerializedName("toplam_pasif_feature")
    val toplamPasif: Int,
    @SerializedName("gelen_feature_listesi")
    val gelenListe: List<String>
)

// TahminCevabi'ye ekle:
@SerializedName("mobil_meta")
val mobilMeta: MobilMeta? = null
```

```kotlin
// API service'e yeni endpoint
@POST("/predict/mobile")
suspend fun mobilTahmin(@Body istek: TahminIstegi): TahminCevabi
```

---

## ÖZET — Kotlin'in Yapacağı

```
1. Saatlik ham veri topla (arka plan servisi):
   - unlock event'leri (UsageStats)
   - aktivite tipleri (ActivityRecognition)
   - konum noktaları (FusedLocation)
   - medya app kullanımı (UsageStats)
   → SQLite'e zaman damgalı kaydet

2. Günlük (EMA doldurunca):
   - O günün ham verisini çek
   - Epoch'lara böl (ep_0/1/2/3)
   - 35 feature'lık pasif Map<String, Double> oluştur
   - Toplayamadıklarını ekleme (backend doldurur)

3. POST /predict/mobile:
   - { uid, gun, ema, pasif } gönder
   - Cevabı al, UI'da göster
   - mobil_meta'yı "Toplanan Veriler" ekranında göster

4. Room'a kaydet (offline geçmiş)
```

---

## Test Edilmiş Backend Davranışı

```
 4 pasif feature gönderildi → 155 medyanla dolduruldu → çalıştı
 Bilinmeyen feature gönderildi → reddedildi, hata vermedi
 HİÇ pasif gönderilmedi → tamamı medyan, yine çalıştı
 feature_count: 183 (24 EMA + 159 pasif) → hibrit model çalışıyor
```

**Backend bu spec'e göre hazır ve test edildi.**
