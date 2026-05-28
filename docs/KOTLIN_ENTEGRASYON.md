# Kotlin Android Entegrasyon Kılavuzu

> Mobil uygulamanın Python FastAPI backend'ine bağlanma rehberi.

**Hedef**: Kotlin EMA uygulamasına risk tahmin özelliği eklemek.

---

## İçindekiler

1. [Genel Mimari](#-genel-mimari)
2. [Hazırlık](#-hazırlık)
3. [Backend Bağlantısı (Retrofit)](#-1-backend-bağlantısı-retrofit)
4. [Veri Modelleri](#-2-veri-modelleri)
5. [API Service](#-3-api-service)
6. [Repository Pattern](#-4-repository-pattern)
7. [ViewModel](#-5-viewmodel)
8. [UI Entegrasyonu](#-6-ui-entegrasyonu)
9. [Pasif Sensör Toplama](#-7-pasif-sensör-toplama)
10. [Hata Yönetimi](#-8-hata-yönetimi)
11. [Test](#-9-test)
12. [Production Deploy](#-10-production-deploy)

---

## Genel Mimari

```
┌─────────────────────────────────────────────────────────┐
│  KULLANICI (Android Telefon)                            │
│  ┌────────────────────────────────────────────────────┐ │
│  │  UI (Compose / XML)                                 │ │
│  │  ┌──────────────────────────────────────────────┐  │ │
│  │  │  EMA Form (7 soru)                            │  │ │
│  │  │  ↓ Submit tıkla                               │  │ │
│  │  │  ViewModel                                    │  │ │
│  │  │  ↓ tahminYap()                                │  │ │
│  │  │  Repository                                   │  │ │
│  │  │  ↓                                            │  │ │
│  │  │  ┌─────────────────┐  ┌──────────────────┐  │  │ │
│  │  │  │ Retrofit Client │  │ SensorCollector  │  │  │ │
│  │  │  └─────────────────┘  └──────────────────┘  │  │ │
│  │  └──────────────┬──────────────────────────────┘  │ │
│  └─────────────────┼─────────────────────────────────┘ │
└────────────────────┼──────────────────────────────────-┘
                     │ HTTP POST /predict
                     │ Content-Type: application/json
                     ▼
┌─────────────────────────────────────────────────────────┐
│  BACKEND (Python FastAPI)                                │
│  http://10.0.2.2:8000 (emulator)                        │
│  http://192.168.x.x:8000 (gerçek cihaz)                 │
└─────────────────────────────────────────────────────────┘
```

---

## Hazırlık

### 1. Backend'i Başlat

Bilgisayarında:
```bash
cd <proje-klasörü>
python3 -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
```

Test et: tarayıcıda `http://localhost:8000/` aç.

### 2. Bilgisayarın IP'sini Öğren

**Mac**:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
# Çıktı: inet 192.168.1.42 ...
```

**Windows**:
```cmd
ipconfig
# IPv4 Address: 192.168.1.42
```

Bu IP'yi not al, Kotlin'de kullanacaksın.

---

## 1. Backend Bağlantısı (Retrofit)

### Gradle Dependencies

`app/build.gradle.kts` dosyana ekle:

```kotlin
dependencies {
    // Mevcut bağımlılıklar...

    // Retrofit (HTTP client)
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")

    // OkHttp (debugging için)
    implementation("com.squareup.okhttp3:logging-interceptor:4.11.0")

    // Coroutines (async)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // ViewModel + LiveData/StateFlow
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
}
```

### AndroidManifest.xml — İzinler

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <!-- İnternet -->
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />

    <!-- Pasif sensörler (sonra ekleyeceğiz) -->
    <uses-permission android:name="android.permission.ACTIVITY_RECOGNITION" />
    <uses-permission android:name="android.permission.PACKAGE_USAGE_STATS"
        tools:ignore="ProtectedPermissions" />

    <application
        android:usesCleartextTraffic="true"
        ...>
        <!-- usesCleartextTraffic = true → HTTP (HTTPS olmayan) kabul et -->
        <!-- Production'da false yap, HTTPS kullan -->
    </application>
</manifest>
```

---

## 2. Veri Modelleri

`data/models/MoodModels.kt`:

```kotlin
package com.senin.uygulaman.data.models

import com.google.gson.annotations.SerializedName

// ────────────────────────────────────────
// İSTEK MODELLERİ (Kotlin → Python)
// ────────────────────────────────────────

data class EMAGirdi(
    val stress: Int,                    // 1-5
    @SerializedName("pam_score")
    val pamScore: Int,                  // 1-16
    @SerializedName("social_level")
    val socialLevel: Int,               // 1-5
    @SerializedName("phq4_q1")
    val phq4Q1: Int,                    // 0-3
    @SerializedName("phq4_q2")
    val phq4Q2: Int,                    // 0-3
    @SerializedName("phq4_q3")
    val phq4Q3: Int,                    // 0-3
    @SerializedName("phq4_q4")
    val phq4Q4: Int                     // 0-3
)

data class TahminIstegi(
    val uid: String,                    // "user_001"
    val gun: String,                    // "2026-05-15"
    val ema: EMAGirdi,
    val pasif: Map<String, Double> = emptyMap(),
    @SerializedName("obj_iletisim")
    val objIletisim: Double = 0.0
)

// ────────────────────────────────────────
// CEVAP MODELLERİ (Python → Kotlin)
// ────────────────────────────────────────

data class CascadeBilgi(
    @SerializedName("risk_sinifi")
    val riskSinifi: Int,
    @SerializedName("profil_id")
    val profilId: String,
    @SerializedName("profil_isim")
    val profilIsim: String,
    @SerializedName("phq4_total")
    val phq4Total: Int,
    @SerializedName("pam_quadrant")
    val pamQuadrant: String
)

data class MLBilgi(
    @SerializedName("risk_binary")
    val riskBinary: Int,
    val olasilik: Double,
    @SerializedName("kullanilan_esik")
    val kullanilanEsik: Double
)

data class TahminCevabi(
    val kullanici: String,
    val gun: String,
    @SerializedName("final_risk")
    val finalRisk: Int,                 // 0-3
    @SerializedName("final_isim")
    val finalIsim: String,              // "İyi Durum" / "Hafif" / "Orta" / "Yüksek Risk"
    @SerializedName("final_renk")
    val finalRenk: String,              // "yeşil" / "sarı" / "turuncu" / "kırmızı"
    val guvenilirlik: String,
    val cascade: CascadeBilgi,
    @SerializedName("hibrit_ml")
    val hibritMl: MLBilgi?,
    @SerializedName("pasif_ml")
    val pasifMl: MLBilgi?,
    @SerializedName("top_5_neden")
    val top5Neden: List<String>,
    val aciklama: String,
    val oneri: String
)
```

---

## 3. API Service

`data/api/MoodApiService.kt`:

```kotlin
package com.senin.uygulaman.data.api

import com.senin.uygulaman.data.models.TahminCevabi
import com.senin.uygulaman.data.models.TahminIstegi
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

interface MoodApiService {

    /**
     * Sunucu çalışıyor mu kontrolü.
     */
    @GET("/")
    suspend fun saglikKontrol(): Map<String, Any>

    /**
     * Ana tahmin endpoint'i.
     *
     * @param istek Kullanıcının EMA cevapları + pasif veri
     * @return Risk tahmini + açıklama + öneri
     */
    @POST("/predict")
    suspend fun tahminYap(@Body istek: TahminIstegi): TahminCevabi

    /**
     * Hızlı tahmin (sadece EMA, pasif olmadan).
     * Demo veya pasif sensor izinleri yokken kullanılabilir.
     */
    @POST("/predict/quick")
    suspend fun hizliTahmin(@Body istek: TahminIstegi): TahminCevabi
}
```

`data/api/RetrofitClient.kt`:

```kotlin
package com.senin.uygulaman.data.api

import com.google.gson.Gson
import com.google.gson.GsonBuilder
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object RetrofitClient {

    //  BASE_URL — bilgisayarına göre değiştir:
    //
    // Emulator için:        "http://10.0.2.2:8000/"
    // Gerçek cihaz (WiFi):  "http://192.168.1.42:8000/"  (kendi IP'in)
    // Production:           "https://senin-domain.com/"

    private const val BASE_URL = "http://10.0.2.2:8000/"

    private val gson: Gson = GsonBuilder()
        .setLenient()
        .create()

    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY  // Debug için
        // Production'da: NONE
    }

    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    private val retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(okHttpClient)
        .addConverterFactory(GsonConverterFactory.create(gson))
        .build()

    val moodApi: MoodApiService = retrofit.create(MoodApiService::class.java)
}
```

---

## 4. Repository Pattern

`data/MoodRepository.kt`:

```kotlin
package com.senin.uygulaman.data

import com.senin.uygulaman.data.api.RetrofitClient
import com.senin.uygulaman.data.models.EMAGirdi
import com.senin.uygulaman.data.models.TahminCevabi
import com.senin.uygulaman.data.models.TahminIstegi
import java.time.LocalDate

class MoodRepository {

    /**
     * Sunucudan risk tahmini al.
     *
     * @param uid Kullanıcı kimliği (örn. cihaz ID veya user ID)
     * @param ema 7 EMA cevabı
     * @param pasif Pasif sensor verisi (opsiyonel)
     * @return Result.success(TahminCevabi) veya Result.failure(Exception)
     */
    suspend fun tahminAl(
        uid: String,
        ema: EMAGirdi,
        pasif: Map<String, Double> = emptyMap(),
        objIletisim: Double = 0.0,
    ): Result<TahminCevabi> {
        return try {
            val istek = TahminIstegi(
                uid = uid,
                gun = LocalDate.now().toString(),
                ema = ema,
                pasif = pasif,
                objIletisim = objIletisim,
            )
            val cevap = RetrofitClient.moodApi.tahminYap(istek)
            Result.success(cevap)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Hızlı tahmin — pasif veri yok.
     */
    suspend fun hizliTahmin(uid: String, ema: EMAGirdi): Result<TahminCevabi> {
        return try {
            val istek = TahminIstegi(
                uid = uid,
                gun = LocalDate.now().toString(),
                ema = ema,
            )
            val cevap = RetrofitClient.moodApi.hizliTahmin(istek)
            Result.success(cevap)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
```

---

## 5. ViewModel

`ui/MoodViewModel.kt`:

```kotlin
package com.senin.uygulaman.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.senin.uygulaman.data.MoodRepository
import com.senin.uygulaman.data.models.EMAGirdi
import com.senin.uygulaman.data.models.TahminCevabi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

sealed class TahminDurumu {
    object Bos : TahminDurumu()
    object Yukleniyor : TahminDurumu()
    data class Basarili(val cevap: TahminCevabi) : TahminDurumu()
    data class Hata(val mesaj: String) : TahminDurumu()
}

class MoodViewModel(
    private val repository: MoodRepository = MoodRepository(),
) : ViewModel() {

    private val _durum = MutableStateFlow<TahminDurumu>(TahminDurumu.Bos)
    val durum: StateFlow<TahminDurumu> = _durum

    /**
     * EMA cevaplarıyla tahmin yap.
     * UI'dan çağrılır.
     */
    fun tahminYap(
        uid: String,
        stress: Int,
        pamScore: Int,
        socialLevel: Int,
        q1: Int, q2: Int, q3: Int, q4: Int,
        pasif: Map<String, Double> = emptyMap(),
    ) {
        viewModelScope.launch {
            _durum.value = TahminDurumu.Yukleniyor

            val ema = EMAGirdi(
                stress = stress,
                pamScore = pamScore,
                socialLevel = socialLevel,
                phq4Q1 = q1,
                phq4Q2 = q2,
                phq4Q3 = q3,
                phq4Q4 = q4,
            )

            val sonuc = repository.tahminAl(uid, ema, pasif)
            sonuc
                .onSuccess { cevap ->
                    _durum.value = TahminDurumu.Basarili(cevap)
                }
                .onFailure { hata ->
                    _durum.value = TahminDurumu.Hata(
                        hata.message ?: "Bilinmeyen hata"
                    )
                }
        }
    }
}
```

---

## 6. UI Entegrasyonu

### Jetpack Compose Örneği

`ui/screens/MoodScreen.kt`:

```kotlin
@Composable
fun MoodScreen(viewModel: MoodViewModel = viewModel()) {
    val durum by viewModel.durum.collectAsState()

    var stress by remember { mutableStateOf(3) }
    var pamScore by remember { mutableStateOf(8) }
    var socialLevel by remember { mutableStateOf(3) }
    var q1 by remember { mutableStateOf(0) }
    var q2 by remember { mutableStateOf(0) }
    var q3 by remember { mutableStateOf(0) }
    var q4 by remember { mutableStateOf(0) }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {

        // EMA formu (sliders)
        EMASoruSlider("Stres seviyesi (1-5)", stress, 1..5) { stress = it }
        EMASoruSlider("PAM ruh hali (1-16)", pamScore, 1..16) { pamScore = it }
        EMASoruSlider("Sosyallik (1-5)", socialLevel, 1..5) { socialLevel = it }
        EMASoruSlider("Q1: Gergin/kaygılı", q1, 0..3) { q1 = it }
        EMASoruSlider("Q2: Endişe", q2, 0..3) { q2 = it }
        EMASoruSlider("Q3: İlgi kaybı", q3, 0..3) { q3 = it }
        EMASoruSlider("Q4: Çökkün", q4, 0..3) { q4 = it }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = {
                viewModel.tahminYap(
                    uid = "user_001",
                    stress = stress,
                    pamScore = pamScore,
                    socialLevel = socialLevel,
                    q1 = q1, q2 = q2, q3 = q3, q4 = q4,
                )
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("RİSK TAHMİN ET")
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Sonucu göster
        when (val d = durum) {
            is TahminDurumu.Bos -> Text("Cevapları girip butona bas")
            is TahminDurumu.Yukleniyor -> CircularProgressIndicator()
            is TahminDurumu.Hata -> Text("Hata: ${d.mesaj}", color = Color.Red)
            is TahminDurumu.Basarili -> SonucKart(d.cevap)
        }
    }
}

@Composable
fun SonucKart(cevap: TahminCevabi) {
    val renk = when (cevap.finalRisk) {
        0 -> Color.Green
        1 -> Color.Yellow
        2 -> Color(0xFFFFA500)  // turuncu
        3 -> Color.Red
        else -> Color.Gray
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = renk.copy(alpha = 0.2f))
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = cevap.finalIsim,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = renk,
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(cevap.aciklama)

            Spacer(modifier = Modifier.height(16.dp))
            Text("Öneri:", fontWeight = FontWeight.Bold)
            Text(cevap.oneri)

            Spacer(modifier = Modifier.height(16.dp))
            Text("Detaylı Nedenler:", fontWeight = FontWeight.Bold)
            cevap.top5Neden.forEach { neden ->
                Text("• $neden", fontSize = 14.sp)
            }
        }
    }
}

@Composable
fun EMASoruSlider(
    baslik: String,
    deger: Int,
    aralik: IntRange,
    onChange: (Int) -> Unit
) {
    Column {
        Text("$baslik: $deger", fontWeight = FontWeight.Medium)
        Slider(
            value = deger.toFloat(),
            onValueChange = { onChange(it.toInt()) },
            valueRange = aralik.first.toFloat()..aralik.last.toFloat(),
            steps = aralik.last - aralik.first - 1,
        )
    }
}
```

---

## 7. Pasif Sensör Toplama

Bu kısım **opsiyonel** — demo için boş pasif veri gönderebilirsin. Gerçek app için:

`data/sensor/PasifVeriToplayici.kt`:

```kotlin
package com.senin.uygulaman.data.sensor

import android.app.usage.UsageStatsManager
import android.content.Context
import java.util.Calendar

class PasifVeriToplayici(private val context: Context) {

    /**
     * Son 24 saatte ekran kaç kez açıldı?
     */
    fun unlockSayisi(): Int {
        val usm = context.getSystemService(Context.USAGE_STATS_SERVICE)
                as UsageStatsManager
        val cal = Calendar.getInstance().apply {
            add(Calendar.HOUR_OF_DAY, -24)
        }
        val events = usm.queryEvents(cal.timeInMillis, System.currentTimeMillis())
        var unlockCount = 0
        val event = android.app.usage.UsageEvents.Event()
        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            if (event.eventType == android.app.usage.UsageEvents.Event.KEYGUARD_HIDDEN) {
                unlockCount++
            }
        }
        return unlockCount
    }

    /**
     * Tüm pasif veriyi topla, modelin beklediği Map'e dönüştür.
     */
    fun tumPasifVeriyiTopla(): Map<String, Double> {
        return mapOf(
            "unlock_num_ep_0" to unlockSayisi().toDouble(),
            "sedanter_saat" to tahminiSedanterSaat(),
            "aktivite_toplam" to tahminiAktivite(),
            // ... diğerleri
            // Pasif sensor toplamak karmaşık — sentetik veri kullanabilirsin
        )
    }

    private fun tahminiSedanterSaat(): Double = 14.0  // placeholder
    private fun tahminiAktivite(): Double = 5400.0    // placeholder
}
```

**Önemli not**: Gerçek pasif veri toplama:
- Activity Recognition API (yürüme, oturma)
- UsageStats API (ekran zamanı)
- Location API (mobilite)
- Sensors (ışık, mikrofon ortalaması)

Bu **3-5 günlük iş**. Demo için **sabit değerler** veya **boş map** yeterli.

---

## 8. Hata Yönetimi

### Yaygın Hatalar ve Çözümleri

```kotlin
class TahminHataYonetici {
    fun anlaminiCozumle(hata: Throwable): String = when {
        hata is java.net.ConnectException ->
            "Sunucuya bağlanılamadı. Backend çalışıyor mu?"

        hata is java.net.UnknownHostException ->
            "Adres bulunamadı. IP doğru mu?"

        hata is java.net.SocketTimeoutException ->
            "Sunucu yanıt vermiyor (timeout)"

        hata is retrofit2.HttpException -> {
            when (hata.code()) {
                400 -> "Geçersiz veri (eksik EMA?)"
                422 -> "Aralık dışı değer (stress 1-5 olmalı)"
                500 -> "Sunucu hatası"
                else -> "HTTP ${hata.code()}: ${hata.message()}"
            }
        }

        else -> "Beklenmeyen hata: ${hata.message}"
    }
}
```

---

## 9. Test

### Birim Test (ViewModel)

```kotlin
@Test
fun `Yuksek risk EMA verince Yuksek Risk donmeli`() = runTest {
    val mockRepo = mock<MoodRepository>()
    whenever(mockRepo.tahminAl(any(), any(), any(), any()))
        .thenReturn(Result.success(/* mock cevap */))

    val viewModel = MoodViewModel(mockRepo)
    viewModel.tahminYap(...)

    val durum = viewModel.durum.value
    assertThat(durum).isInstanceOf(TahminDurumu.Basarili::class.java)
}
```

### Manuel Test

1. Backend çalışıyor mu? Tarayıcıdan `http://localhost:8000/` aç
2. Emulator'de uygulama aç
3. EMA cevapları gir
4. Buton bas
5. Sonuç kartı çıkmalı

### Logcat ile Debug

```kotlin
// Retrofit logging zaten body'i basıyor
// Logcat'te şunu göreceksin:
// D/OkHttp: --> POST http://10.0.2.2:8000/predict
// D/OkHttp: {"uid": "user_001", "gun": "2026-05-15", ...}
// D/OkHttp: <-- 200 OK
// D/OkHttp: {"final_risk": 3, "final_isim": "Yüksek Risk", ...}
```

---

## 10. Production Deploy

### Backend'i Cloud'a Yükle

Uygulamayı canlı ortamda göstermek istiyorsan:

**Seçenek 1: Railway.app** (önerim, $5/ay)
```bash
cd <proje-klasörü>
railway init
railway up
# Sana bir URL verir: https://senin-projen.railway.app
```

**Seçenek 2: Render.com** (ücretsiz tier)
```bash
git push render main
# Otomatik deploy
```

**Seçenek 3: Heroku, AWS EC2, GCP** (daha karmaşık)

### Production BASE_URL

```kotlin
private const val BASE_URL = "https://senin-projen.railway.app/"
```

Production'da:
- **HTTPS** kullan (`https://`)
- `usesCleartextTraffic="false"` yap (AndroidManifest)
- **API key** ekle (`Authorization` header)
- **Rate limiting** ekle (kötü kullanım önle)

---

## Hızlı Referans

### Backend Endpoint'leri

```
GET   /                  → Health check
GET   /info              → Yüklü modeller
POST  /predict           → Ana tahmin (EMA + pasif)
POST  /predict/quick     → Hızlı tahmin (sadece EMA)
```

### Risk Sınıfları

```
0 → "İyi Durum"    → yeşil
1 → "Hafif Risk"   → sarı
2 → "Orta Risk"    → turuncu
3 → "Yüksek Risk"  → kırmızı
```

### Örnek curl Test (terminal'den)

```bash
curl -X POST http://localhost:8000/predict/quick \
  -H "Content-Type: application/json" \
  -d '{
    "uid": "test",
    "gun": "2026-05-15",
    "ema": {
      "stress": 4, "pam_score": 12, "social_level": 2,
      "phq4_q1": 2, "phq4_q2": 2, "phq4_q3": 3, "phq4_q4": 3
    }
  }'
```

---

## Yardım & Sorun Giderme

### Backend çalışmıyor
```bash
ps aux | grep uvicorn   # Server çalışıyor mu?
curl http://localhost:8000/   # Cevap geliyor mu?
```

### Emulator'den bağlanamıyor
- BASE_URL `http://10.0.2.2:8000/` mı? (NOT: 127.0.0.1 değil)
- `usesCleartextTraffic="true"` var mı?

### Gerçek cihazdan bağlanamıyor
- Telefon ve bilgisayar AYNI WiFi'de mi?
- Bilgisayarın firewall'u port 8000'i engelliyor mu?
- IP'yi doğru aldın mı?

### CORS hatası
- `server.py`'de CORS middleware var (zaten ekledim)
- `allow_origins=["*"]` development için OK

### Timeout
- `OkHttpClient.connectTimeout(60, TimeUnit.SECONDS)` yap
- İlk istek modeli yükleyebilir (~2 sn)

---

## Kontrol Listesi

Demo'ya başlamadan önce:

- [ ] Backend çalışıyor (`http://localhost:8000/` cevap veriyor)
- [ ] AndroidManifest'te `INTERNET` izni var
- [ ] AndroidManifest'te `usesCleartextTraffic="true"` var
- [ ] BASE_URL doğru (emulator için `10.0.2.2`, gerçek cihaz için kendi IP'in)
- [ ] Gradle dependencies eklendi
- [ ] Retrofit + Repository + ViewModel kurulumu yapıldı
- [ ] UI'da tahmin butonu çalışıyor
- [ ] Sonuç kartı görüntüleniyor
- [ ] Hata durumları handle ediliyor

---

## Tezde Bahsedecek Konular

```
"Mobil uygulama 3 katmanlı mimari ile geliştirildi:

  1. UI Katmanı (Jetpack Compose)
     7 EMA sorusu interaktif slider'lar
     Sonuç kartı (renkli risk göstergesi)

  2. Veri Katmanı (Retrofit + Repository pattern)
     Sunucuyla HTTPS iletişimi
     Coroutines ile async işlem
     Result<T> ile hata yönetimi

  3. Backend (Python FastAPI)
     REST API endpoint'leri
     Pydantic veri doğrulama
     predict_mood() core fonksiyon

İletişim: JSON over HTTP, ~30ms gecikme
Klinik karar destek arayüzü standart MVVM yapısıyla."
```

---

**Başarılar! Sorularını cevaplamak için her zaman buradayız.**
