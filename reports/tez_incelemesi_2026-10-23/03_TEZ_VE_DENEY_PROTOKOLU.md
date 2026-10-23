# 3. Önerilen tez ve deney protokolü

Bu bölüm uygulanmış yeni deneylerin sonuçlarını değil, mevcut deneyleri koruyarak izlenebilecek araştırma protokolünü tanımlar. Örnek bütçeler danışman, hesap kaynağı ve sonuçların değişkenliğine göre kesinleştirilmelidir.

## 3.1. Önerilen tez çerçevesi

**Türkçe başlık adayı:** _Seyrek Ölçümlü Dağıtım Şebekelerinde Fizik Rehberli Gaussian Process Modelleriyle Gerilim Kestirimi ve Belirsizlik Kalibrasyonu._

**İngilizce başlık adayı:** _Physics-Guided Gaussian Process Voltage Estimation in Sparsely Monitored Distribution Networks: Correlation, Input Uncertainty and Calibration._

Ana kapsam V genliğidir. Tam AC durum kestirimi seçilecekse başlık, hedefler, angle/reference tanımı ve fiziksel residual ölçümleri buna göre genişletilmelidir. Theta hiçbir önerilen ana senaryoda measured input sayılmamalı.

### Araştırma soruları ve falsifiable hipotezler

| Soru | Sınanacak hipotez | Yanlışlanabileceği durum |
|---|---|---|
| RQ1 — Belirsizlik yapısı | Shared kernelde raw bus farkının bir kısmı yalnız target scaling'dir | Normalize PI matrisi rank-1 açıklamasından saparsa önce uygulama/varsayım farkı araştırılır |
| RQ2 — Feature seçiciliği | ARD, noisy/redundant feature'lar altında isotropic modele göre score'u iyileştirir | Etki küçük, restart'a bağımlı veya testte ters olabilir |
| RQ3 — Korelasyon | Joint output covariance, uygun kısmi gözlem düzeninde ölçüm bilgisini daha iyi aktarır | Tam gözlem/noiseless düzende mean kazancı olmayabilir; transfer negatif olabilir |
| RQ4 — Fiziksel önsel | Az label ve farklı sensör maskesinde fiziksel prior daha yararlı olabilir | Yanlış fizik/model bias avantajı yok edebilir |
| RQ5 — Girdi noise'u | Noise-aware yaklaşım raw modelden daha iyi calibrated tahmin verir | Double counting veya yanlış R/prior aralığı gereksiz genişletebilir |
| RQ6 — Ölçümün katkısı | Aynı bus için bilgilendirici yeni ölçüm uncertainty/error'u azaltır | Sabit bütçeli takas, zayıf/yanlış ölçüm veya model misspecification altında kazanım olmayabilir |

Hipotezlerin hepsinin olumlu çıkması gerekmez. Örneğin Corr modelin noise-free tam etiketli düzende üstün gelmemesi, ilgili kernel teorisi ve gözlem düzeniyle açıklanabilirse anlamlı bulgudur.

## 3.2. Tezin minimum ve önerilen kapsamı

**Asgari güçlü çalışma:** mevcut shared ve independent Winter kontrolü + basit güçlü baseline'lar + ARD + doğru uncertainty/calibration değerlendirmesi + birden çok sensor/noise koşulu. Bu bir karşılaştırmalı yöntem tezi olabilir; danışmanın yeni algoritma beklentisi ayrıca belirleyicidir.

**Önerilen ana çalışma:** yukarıdakilere bir joint covariance modeli ve fiziksel-prior varyantını eklemek; az veri, measured role ve mismatch altında etkilerini ayırmak. Physics-only/data-only/hybrid ablation ile fiziksel bilginin katkısını doğrudan göstermek.

**İleri kapsam:** nonlinear AC-constrained GP veya multiarea GNN+GP. Bunların ikisini de aynı tezde zorunlu hedef yapmak önerilmiyor. Önce tamamlanabilir bir ana hikâye ve negatif sonuç halinde savunulabilir yedek plan kurulmalı.

## 3.3. Mevcut pipeline'ı koruyan çalışma düzeni

Mevcut `seasonal_evaluation.py`, noise-free akış ve `winter_independent_evaluation.py` tarihsel referans olarak korunmalı. Yeni karşılaştırmalar ayrı entry point/run type ve ayrı output klasöründe çalışmalı. Bu rapor bunları eklemiyor.

Yeni deneyin veri sözleşmesi:

```text
X: yalnız seçili measured bus'ların P/Q/V bilgisi
Y_eval: temiz non-reference V hedefi
index: benzersiz simbench_step
metadata: gerçek date, timestamp, day, step_in_day, bus identity
prediction: mean + açıkça tanımlanmış variance/covariance
interval: raw veya calibrated olduğu açık
manifest: kaynak/config/data/noise/split/model sürümleri
```

Fiziksel yöntem ek simulation/prior bilgisi kullanıyorsa ayrı `information_budget` alanı bulunmalı. Testte unmeasured P/Q veya theta'yı bilinen değer gibi geçirmek yasak olmalı. Latent/target theta kullanımı ayrıca belirtilmeli; yalnız değerlendirmede kullanılan ground truth modele input sayılmamalı.

Dashboard'a yeni yöntem eklenmesi daha sonraki uygulama aşamasıdır. Gerekli metadata ve ortak CSV sözleşmesi önce sabitlenirse UI'ı baştan yazmaya gerek kalmaz. Scalar length-scale varsayımları ve yeni covariance çıktıları için hedefli adapter gerekir.

## 3.4. Karşılaştırma modelleri ve adalet koşulları

| Kod | Model | Neden gerekli? |
|---|---|---|
| B0 | Measured V'yi doğrudan kullanma | Yalnız measured-target denoise başarısı için referans; unmeasured bus kestiricisi değildir |
| B1 | Eğitimdeki sezon/bus mean'i | Regresyonun çok zayıf baseline'dan gerçekten uzaklaştığını gösterir |
| B2 | Ridge/çoklu lineer regression | LV gerilim ilişkilerinin ne kadarının basit modelle açıklanabildiğini gösterir |
| B3 | Eğitimden joint Gaussian conditioning | Learned covariance ve measurement noise için ucuz güçlü kontrol |
| S0 | Mevcut shared isotropic GPR | Ana tarihsel referans |
| I0 | Mevcut 43 independent GPR | Target-specific kernel kontrolü; ilk aşamada yalnız Winter |
| A0 | Shared feature/group ARD | Input metric seçiciliğini izole eder |
| C0 | Açık tanımlanmış ICM/LMC Corr-GP | Gerçek cross-output covariance ve gözlem düzeni etkisi |
| P0 | Physics prior / mixed prior | Fiziksel önbilginin ek katkısı |
| W0 | Uygun pseudo/prior ile AC WLS veya MAP | Physics-based baseline; bilgi bütçesi açık ise |

Her modeli ilk günden bütün grid'de çalıştırmak gerekli değil. B0/B1/B2/B3 hızlı başlangıç kontrolleri; S0/I0/A0 çekirdek karşılaştırma; C0/P0 önerilen araştırma yönüdür.

WLS, 39 ölçümle otomatik tam gözlenebilir değildir. Ona gerekli pseudo-measurement/zero-injection/prior'ı vermeden çalıştırıp başarısızlığını GPR üstünlüğü diye sunmak adil olmaz. Tersine WLS'ye testteki gerçek unmeasured injection'ları vermek de oracle bilgi avantajıdır. Her yöntem için eğitim verisi, fiziksel bilgi, pseudo data ve online ölçümler tablo halinde verilmelidir.

R01'in birebir yeniden üretimi, tam metin ve gerçek protokol elde edildiğinde ayrı C-paper varyantı olmalı. Kendi ICM uygulamasını yalnız isim benzerliğiyle paper reproduksiyonu diye adlandırmamak gerekir.

## 3.5. Veri ayrımı ve testin korunması

### Hat A — Mevcut benchmark'ı dondur

Mevcut 30 eğitim günü ve 61/62 test gününü değiştirme. Kaydedilmiş model ve sonuçları hash ile arşivle. Bu hat, kod geçmişindeki karşılaştırmaları anlaşılır tutar. Önceden defalarca incelenmiş bu test kümesini tamamen el değmemiş yeni benchmark diye adlandırma.

### Hat B — Yeni yöntem seçimi ve calibration için protokol

Uygulanabilir bir örnek:

- Mevsim başına aynı **30 gün training pool**.
- Bu 30 gün içinde whole-day blocked inner validation; kernel/hyperparameter seçimi burada.
- Seçilen ayarla 30 günün tamamında refit.
- Kalan günlerden **20 gün calibration**, diğer **41 Winter/Autumn, 42 Spring/Summer gün final test**.
- Bütün modeller aynı calibration ve final test günlerini kullanır.

Bu örnekte yıl 120 training + 80 calibration + 166 final test gününe ayrılır. 20 calibration günü bir öneri bütçesidir; %95 day-level quantile için kuyruk çözünürlüğü sınırlıdır ve yüksek kesinlik iddiası vermez. Interval calibration yöntemi ve bağımlılık yapısına göre gerekirse bütçe yeniden tasarlanmalıdır.

Calibration day listesi model fit veya kernel seçimi için kullanılmamalı. Calibration'dan sonra yöntem seçmek de calibration'ı validation'a dönüştürür; son test yine korunmalıdır. Yeni final test alt kümesi kullanılacaksa S0/I0 dahil bütün karşılaştırmalar o aynı alt kümede yapılmalı.

### Hat C — Gerçek genelleme / OOD

İleri zaman bloku, farklı load/PV dağılımı, yanlış fiziksel parametre veya yeni topoloji ayrı deney olarak ayrılmalı. Retrospective gün dağılımı ile causal online değerlendirme aynı tabloda tek skor gibi birleştirilmemeli. Winter'ın Jan/Feb ve December parçaları zaman blokları yapılırken açık ele alınmalı.

Causal görev seçilirse Corr modelinin query zamanından sonraki measured test değerlerine bakmaması gerekir. Offline smoothing/recovery ayrıca yapılabilir; bütün yöntemlere aynı bilgi erişimi verilmelidir.

## 3.6. Paired sensör ve noise deneyleri

Önerilen pilot, mevcut `SENSOR_SEED=42` ve Winter ile başlar. Ana sonuçların tek yerleşime bağlı kalmaması için kaynak elverdiğinde beş önceden belirlenmiş sensor layout ve üç noisy realization kullanılabilir. Bu sayı otomatik istatistiksel yeterlilik garantisi değildir; pilot variability ve compute ölçümüne göre kesinleştirilir.

Her layout'ta bütün modeller:

1. aynı 13 measured bus'ı;
2. aynı 39 feature sırasını;
3. aynı train/calibration/test step'lerini;
4. aynı precomputed noisy input matrisini;
5. aynı temiz hedefleri kullanmalı.

Bu input matrisleri bir kez üretilip hash ile saklanırsa RNG sırasına bağlı yanlış karşılaştırmalar önlenir. Mevcut algoritma aynı günün alt kümesine noise eklediğinde realization değiştirdiğinden önce tam gün noise'u üretilmelidir.

Birden çok layout'ta aynı bus'ı rol değiştirerek incelemek için mümkünse bus/ölçüm/timestep anahtarlı potential-noise alanı ayrıca tanımlanabilir. Mevcut noise algoritmasında sütun pozisyonu da RNG tüketimini etkiler; farklı layout aynı bus'a farklı draw verebilir. Yeni algoritma seçilecekse ayrı schema/version ile uygulanmalı; eski deneyin noise'unun birebir aynısı diye sunulmamalı.

Clean senaryoda noise seed'i tekrarlamak bağımsız tekrar oluşturmaz. Aynı deterministic clean fit'i üç kere çalıştırmak örnek sayısını üçe katlamaz. Ayrı optimizer başlangıçları sensitivity analizi olabilir; sensor placement ve data split tekrarı yerine geçmez.

## 3.7. Measured/unmeasured rolünü doğrudan sınama

### Test 1 — Mevcut grupların betimlenmesi

Her mevsim/model/senaryo için measured ve unmeasured bus sayısı, MAE/RMSE, coverage, mean PI width, eğitim target std ve normalized width. Bu tablo descriptive'dir; nedensel rol etkisi diye adlandırılmaz.

### Test 2 — Aynı bus içinde rol değişimi

Bir target bus farklı 13-sensör layout'larında measured ve unmeasured olsun. Bus düzeyindeki hata/score farkları eşleştirilir; farklı geometry ve diğer sensörlerin etkisi raporlanır. Bütün bus'ların mutlaka iki rolü de aldığı bir tasarım istenirse beş rastgele layout bunu garanti etmez; balanced özel bir layout tasarımı gerekir.

### Test 3 — Sabit kümeye bir ölçüm ekleme

Joint-prior modelde aynı prior ve aynı diğer ölçümler korunur; tek bir V ölçümü eklenince ilgili posterior variance ve çevre bus'ların variance değişimi hesaplanır. Teorik PSD variance reduction burada doğrudan sınanır. Noisy veya biased ölçüm için değişen varsayımlar ayrıca belirtilir.

### Test 4 — Kendi V'sini kaldırma

Mevcut P/Q/V → V görevinde measured bus'ın kendi V girişini kaldıran ayrı ablation, measured-target başarısının ne kadarının doğrudan V erişiminden geldiğini gösterir. Bu yeni görev ana sonuçla karıştırılmamalı; fiziksel kısıt ve noise karşılaştırmaları aynı availability düzenine göre yapılmalıdır.

## 3.8. Training budget ve data-efficiency eğrisi

İlk eğri yalnız Winter'da, aynı training pool içinden **N=25, 96, 288, 960 ve bütün training noktaları** ile çizilebilir. N, gün değil **timestep/snapshot** sayısıdır. Autumn'a taşınırsa bütün training noktaları 2.884; normal üç mevsimde 2.880'dir.

N küçükken ardışık 25 çeyrek saat, mevsimin tamamına yayılan 25 operating point ile aynı zorlukta değildir. Örnekler yalnız training pool'dan, önceden belirlenmiş ve tekrarlanabilir yöntemle seçilmeli. Final test hiçbir budget'ta değişmemeli.

Physics prior için MC simulation sayısı, Corr için hyperparameter fit'te görülen target sample sayısı ve bütün yöntemlerde label sayısı ayrıca raporlanmalı. Binlerce ek physics simulation ile eğitilmiş modeli “yalnız 25 örnek kullandı” diye anlatmamak gerekir. Veri kıtlığı label kıtlığı mı, fizik simulation kıtlığı mı, ikisi mi açık yazılmalıdır.

## 3.9. Birincil ve ikincil metrikler

**Birincil accuracy:** unmeasured-only pooled RMSE ve MAE, ayrıca bus/season dağılımları. **Birincil uncertainty:** coverage ile birlikte Gaussian NLPD veya CRPS/interval score. **İkincil:** measured denoise başarısı, bütün target bus'lar, normalized error/width, runtime ve fiziksel residual.

N voltage hücresi için:

```text
MAE  = mean |μ−y|
RMSE = sqrt(mean (μ−y)²)
Coverage_95 = mean 1[L ≤ y ≤ U]
Mean_width = mean(U−L)
NLPD = mean[ 0.5 log(2πσ²) + (y−μ)²/(2σ²) ]
IntervalScore_α = (U−L)
    + (2/α)(L−y) 1[y<L] + (2/α)(y−U) 1[y>U]
```

`α=0.05` için interval score, aralığı geniş tutmayı ve yanlış daraltmayı birlikte cezalandırır. NLPD ise çok emin olup yanlış olmayı güçlü biçimde cezalandırır. Log-density puanlarının birimi, target scaling ve variance floor bütün yöntemler için açık olmalıdır.

Deterministic measured-copy baseline'ın doğal probability density'si yoktur; keyfî sigma ekleyerek onu olasılıksal modellerle NLPD sıralamasına sokmamak gerekir. Noktasal ve probabilistic baseline'lar uygun metriklerde karşılaştırılır.

### Kalibrasyon görselleri

- Nominal–empirical coverage eğrisi: %50/%80/%90/%95/%99.
- Bus bazında coverage dağılımı ve alt kuyruk; sadece ortalama değil en kötü bus'lar.
- Standardized residual ve PIT histogramı; zamana ve V/yük/PV seviyesine göre koşullu dağılımlar.
- Aynı hedef coverage düzeyinde interval width veya score.
- Raw ve calibrated modelin aynı grafikte, açık etiketlerle karşılaştırması.

Küçük std için sabit evaluation floor uygulanıyorsa kaç hücrede uygulandığı raporlanmalı. Floor, kötü modelin aşırı güvenli hatalarını gizleyecek şekilde teste göre seçilmemeli.

### Fiziksel uygunluk ve karar metrikleri

Tam state veya augmented state tahmin ediliyorsa AC residual ve ölçüm residual'ı, aynı işaret/birim/reference tanımıyla değerlendirilir. V-only model için gerçek açıları gizlice kullanıp “model fiziksel” sonucu çıkarılamaz.

İşletme açısından bir genişletme, önceden seçilmiş voltage-limit aşımı olasılığı ve Brier score/precision–recall değerlendirmesidir. Limitler ilgili uygulamaya göre gerekçelendirilmeli; rare-event sayısı açıklanmalı. Bu, iyi uncertainty'nin hangi kararı iyileştirdiğine somut bağ sağlar.

Joint covariance üreten modellerde isteğe bağlı joint log score veya snapshot-level coverage kullanılabilir. Marginal sigma'lardan cross-bus covariance icat edilmemeli; shared/independent modellerin joint independence varsayımı açık belirtilmelidir.

## 3.10. Güven aralıkları ve istatistiksel karşılaştırma

23.612 timestep × 30 bus hücrelerini bağımsız yüz binlerce örnek sayıp çok dar confidence interval üretmek uygun değildir. Bus'lar fiziksel olarak, zaman noktaları da seri bağımlılığıyla ilişkili olabilir.

Öneri: yöntemler arasında **paired farklar**; sensör yerleşimi dış seviye, noise realization iç seviye ve takvimde bütün gün/hafta bloklarıyla hierarchical/block bootstrap. Aynı örnek seçimi iki yöntem için birlikte uygulanır. Üç noise realization'ın beş layout'ta olması, 15 tamamen bağımsız sensor-layout örneği olduğu anlamına gelmez.

Blok uzunluğu residual autocorrelation ve sensitivity analiziyle gerekçelendirilmeli. Winter'ın yıl içindeki iki parçası veya training günleri çıkarılmış test takvimi, yanlışlıkla kesintisiz tek zaman dizisi gibi birleştirilmemeli. Calendar block sınırları kullanılmalıdır.

Raporlanacaklar: paired mean/median fark, güven aralığı, mutlak p.u. farkı, relative iyileşme ve hesap maliyeti. P-value tek karar ölçütü olmamalı. Çok fazla model/kernel/scenario arasında yalnız en iyi çıkan satırı seçmek selection bias yaratır; birincil karşılaştırmalar önceden belirlenmelidir.

## 3.11. Sistematik robustness matrisi

| Etken | Kontrollü örnekler | Ayrı tutulması gereken varsayım |
|---|---|---|
| Noise seviyesi | Mevcut sigma'nın 0, 1, 2, 5 katı | Training noise ile test noise ayrı belirtilir |
| Bias | Sabit meter offset/gain; sonra drift | Zero-mean random noise ile aynı hata türü değildir |
| Gross error | Seyrek büyük sapma veya heavy tail | Frekans ve büyüklük önceden tanımlanır |
| Ölçüm kaybı | Tek bus kaybı, birkaç kanal kaybı | Imputation/mask/conditioning yöntemi açık olmalı |
| Load/PV dağılımı | Eğitim aralığı dışı ölçek veya bağımlılık | Yeni gerçek PF scenario; temiz cache üstüne yazılmaz |
| Parametre mismatch | Estimator R/X/trafo bilgisinde kontrollü hata | Gerçek ağ sabit; yalnız model bilgisi yanlış |
| Gerçek topoloji değişimi | Enerjili ve geçerli yeni bağlantı düzeni | Yeni gerçek PF çözümü ve ayrı cache gerekir |

Buradaki katsayılar ölçüm standartlarından alınmış kesin saha hata sınıfları değil, önerilen sensitivity seviyeleridir. Uygulama fiziksel dayanakla kesinleştirmelidir.

Radial bir şebekede keyfî bir line'ı açmak adalanma ve çözülemeyen/NaN V üretebilir. Böyle bir durum, geçerli yeniden yapılandırmayla aynı topology-shift deneyi değildir. Scenario önce enerji beslemesi, solver convergence ve fiziksel geçerlilik bakımından kontrol edilmelidir. Başarısız PF'ları sessizce düşürmek yerine neden ve sayıları raporlanmalı.

Gerçek veriyi üreten aynı fiziksel ağ ve load/PV dağılımını estimator'a kusursuz vermek, ideal-model referansıdır. Bu koşuldaki fiziksel avantaj, saha koşullarında aynı ağ ve dağılımın kusursuz bilindiği varsayımına dönüştürülmemeli; parameter/prior mismatch deneyleri bu nedenle gereklidir.

Başlangıçta bütün bu grid'i çalıştırmak yerine iki stres seçilmeli: **test noise mismatch** ve **fiziksel model parameter mismatch**. Bunlar mevcut projenin ana iddialarına doğrudan bağlıdır. Dropout ve ikinci feeder sonraki genişletmedir.

## 3.12. Katkıyı ayıracak ablation deneyleri

| Ablation | Sabit tutulacaklar | Cevaplanan soru |
|---|---|---|
| Isotropic → ARD | Veri, output yapısı, WhiteKernel ve tuning bütçesi | Kazanç input metric'ten mi? |
| Shared → independent | Feature seti, N, noise ve target scope | Ayrı hyperparameter öğrenmek ne kazandırıyor? |
| Diagonal B → learned full/low-rank B | Mean, kernel ailesi ve observation mask | Cross-output covariance etkisi var mı? |
| Learned prior → physics prior → mixed | Online ölçümler, label/MC bütçeleri belgeli | Fiziksel önbilgi faydalı mı? |
| Aynı model raw → calibrated | Mean ve final test sabit | Interval iyileşmesi post-processing'den mi? |
| Kesin input → noise-aware inference | Sensor covariance ve training bilgisi açık | Input uncertainty propagation katkısı ne? |
| Doğru fizik → yanlış fizik | Gerçek test ağı ve z sabit | Modelin fiziksel bilgiye hassasiyeti ne? |

Tam yöntem birden çok bileşen içeriyorsa yalnız S0 ile karşılaştırılmamalı. Örneğin yeni model hem ARD hem full covariance hem calibration kullanıyorsa, iyileşmenin “physics-informed” kısımdan geldiği kendiliğinden söylenemez.

## 3.13. Hesap bütçesi: bilimsel grid ile uygulanabilir çalışma ayrımı

Önce tek bir tam Winter shared GP ve bir independent target için süre/RAM ölçülmeli; ardından 43 modelin toplam maliyeti tahmin edilmeli. Bu rapordaki 48-point örnek fit süreleri 2.880-point fit'e doğrusal taşınamaz.

Örnek olarak beş sensor layout için clean, beş×üç layout/noise koşulu için noisy çalışma, toplam 20 karşılaştırma koşulu verir. Her koşulda I0'nun 43 fit'i, S0/A0 ve yeni yöntemle birlikte yüzlerce exact fit doğurur. Beş training budget, dört mevsim, restart ve ek kernel çarpanlarını aynı anda buna eklemek maliyeti çok büyütür.

Bu yüzden aşamalı grid:

1. **Smoke/pilot:** küçük N, tek layout, Winter; veri sözleşmesi ve numerical doğruluk.
2. **Yöntem seçimi:** tek layout, önceden ayrılmış validation, sınırlı kernel/başlangıç seti.
3. **Core Winter:** seçilmiş yöntemler, aynı tam training bütçesi; birden çok layout/noise.
4. **Data-efficiency:** yalnız az sayıdaki en ilgili yöntem, değişmeyen final test.
5. **Genelleme:** en iyi baseline ve önerilen yöntem için diğer mevsimler ve ikinci feeder.

Bir yöntemi layout tekrarlarında yalnız subset üzerinde çalıştırmak gerekirse, aynı subset'teki paired sonuçları ayrı raporlamak gerekir. 15 koşullu yöntemin ortalamasını tek koşullu I0 sonucu ile eşdeğer karşılaştırma gibi vermemek gerekir.

Sparse/structured GP kullanılırsa approximation ayarları, inducing point sayısı ve training budget'ı kaydedilmeli. Exact referanstan kazanç veya kayıp, küçük sistemde doğrudan ölçülmelidir.

## 3.14. Uygulama ve çalışma paketleri

### WP0 — Bilimsel tanım ve referansın dondurulması

Çıktı: V-only problem tanımı; bilgi erişim tablosu; run/data/code hash; mevcut modellerin same-Winter değerlendirmesi; stress noise tutarlılığı için ayrı düzeltme planı.

Bitiş ölçütü: aynı kaydedilmiş input ve model aynı tahmin/metrikleri üretiyor; theta input değil; annual ve stress realization ilişkisi açık. Kullanıcının mevcut koduna bu rapor aşamasında dokunulmadı.

### WP1 — Baseline ve uncertainty değerlendirmesi

Çıktı: B0/B1/B2/B3/S0/I0; train/inner-validation/calibration/final-test ayrımı; raw/calibrated metrikler.

Bitiş ölçütü: basit modelin rekabetçi olup olmadığı biliniyor; alt gruplarda calibration açık; yeni model için doğru karşılaştırma zemini var.

### WP2 — ARD

Çıktı: shared feature/group ARD; vector length-scale metadata; relevance stability ve feature ablation.

Bitiş ölçütü: sonuç yalnız bir optimizer başlangıcına veya tek test gününe bağlı değil; shared uncertainty'nin rank-1 yapısı doğru yorumlanıyor.

### WP3 — Correlated ve fiziksel prior

Çıktı: açık observation düzeniyle C0; data-only, physics-only ve mixed prior; physical information/MC bütçesi tablosu.

Bitiş ölçütü: B PSD ve boyut/sıra doğru; direct measured conditioning kontrolü geçiyor; tam gözlemde transfer cancellation olasılığı açıklanıyor; yanlış fizik altında davranış ölçülüyor.

### WP4 — Robustness ve dış geçerlilik

Çıktı: iki seçilmiş stress ailesi, ikinci feeder veya farklı profil; paired uncertainty/performance karşılaştırması.

Bitiş ölçütü: yeni yöntemin hangi koşullarda faydalı, hangi koşullarda kırılgan olduğu gösteriliyor; başarısızlıklar saklanmıyor.

### WP5 — Yazım, tekrar üretim ve savunma

Çıktı: kilitli deney konfigürasyonları, ana tablo/figürler, tekrarlanabilir scriptler ve iddia–kanıt eşlemesi.

Yaklaşık 12–16 haftalık bir master çalışma penceresi varsa WP0–1 ilk 2–3 hafta, WP2 sonraki 2 hafta, WP3 yaklaşık 3–4 hafta, WP4 2–3 hafta ve kalan zaman yazım/tekrar üretim için ayrılabilir. Bunlar zaman taahhüdü değildir; öğrencinin altyapısı, compute ve danışman geri bildirimi süreyi değiştirir. Mevcut tezin bir son tarihi bilinmediğinden kesin takvim atanmadı.

## 3.15. Hangi durumda ilerlemeli, hangi durumda sadeleştirmeli?

| Gözlem | Karar |
|---|---|
| Ridge, GPR'ye çok yakın | Nonlinearity katkısı sınırlı olabilir; uncertainty ve OOD'yi ana odağa taşı |
| ARD relevance kararsız | Group-ARD/regularization; nedensel sensör önemi iddiasını daralt |
| Corr mean kazancı yok | Gözlem düzeni, noise ve autokrigeability kontrolü; covariance/eksik kanal avantajını ayrı değerlendir |
| Physics prior nominalde iyi, mismatch'te kötü | Karışım/discrepancy ve model-risk analizi; negatif sonuç önemli |
| Calibration bütün model farkını siliyor | Raw/calibrated ayrımını koru; yeni kernelin pratik faydasını yeniden değerlendir |
| Model yalnız çok büyük N'de iyi | Data-efficiency iddiasını daralt; maliyeti tartış |
| Nonlinear constraint inference kararsız | Core'u linear/ensemble physics-guided modelle tamamla; ileri yöntemi gelecek işe ayır |

Bu kararlar sonuçlara göre hikâye değiştirmek değil, önceden tanımlanmış araştırma sınırlarıdır. Ana testte yalnız olumlu alt küme seçilmemelidir.

## 3.16. Tez bölümleri ve sunum çıktıları

Önerilen tez akışı:

1. Motivasyon, DSSE/V-only problem tanımı ve araştırma soruları.
2. Fiziksel model, ölçümler, gözlenebilirlik ve bilgi varsayımları.
3. GP, ARD, independent/shared/correlated ve uncertainty literatürü.
4. Önerilen fiziksel/joint-prior yöntem ve inference.
5. Veri, split, noise, baselines, calibration ve istatistiksel protokol.
6. Sonuçlar: doğruluk, belirsizlik, measured role, data efficiency ve runtime.
7. Mismatch, dış geçerlilik, başarısızlıklar ve sınırlamalar.
8. Sonuç ve sınırlandırılmış gelecek çalışma.

Ana sunum figürleri: information-flow diyagramı; train/calibration/test takvimi; aynı Winter için accuracy–score tablosu; raw/normalized uncertainty; coverage reliability; aynı bus'ın rol değişimi; data-efficiency eğrisi; mismatch eğrisi; runtime/RAM. Heatmap'ler bu hikâyeyi destekler, ana bilimsel kanıtın yerine geçmez.

## 3.17. Tez seviyesine ulaşmak için kabul kontrolü

- Araştırma sorusu ve başarı/başarısızlık ölçütü açık mı?
- Input/target ve online erişilebilir bilgi doğru tanımlı mı; theta measured input değil mi?
- Model mean'i, covariance'ı ve interval türü matematiksel olarak doğru açıklanıyor mu?
- Güçlü basit ve fiziksel baseline'lar adil bilgi bütçesinde mi?
- Validation/calibration/final-test ayrımı korunuyor mu?
- Measured rolü farklı bus gruplarının doğal değişkenliğinden ayrılıyor mu?
- Nominal coverage, empirical coverage, sharpness ve proper score birlikte var mı?
- Birden fazla sensör/noise koşulu ve bağımlılığa uygun uncertainty değerlendirmesi var mı?
- Fiziksel model hatası ve tek-feeder/sentetik veri sınırı açıklanıyor mu?
- Katkının hangi bileşenden geldiği ablation ile gösteriliyor mu?
- Kaydedilmiş manifest ile sonuçlar tekrar üretilebiliyor mu?
- Özgünlük iddiası en yakın yayınlara göre sınırlı ve kaynaklı mı?

Bu maddelerin çoğu tamamlandığında çalışma, yalnız çalışan bir dashboard olmaktan çıkıp savunulabilir bir araştırma haline gelir. Belirli bir üniversite için tez kabulünü garanti etmek mümkün değildir; burada amaç o akademik değerlendirmeye somut kanıtla girebilecek bir yapı kurmaktır.
