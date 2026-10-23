# 2. Literatür ve bu projeye uygun yöntemler

**23 Ekim 2026 —** Kaynak kodları ve erişim düzeyleri [kaynak dizininde](04_KAYNAKLAR.md). Bu bölümdeki modeller **öneridir**; uygulamaya eklenmediler.

## 2.1. Tez için ana karar

En güçlü yön, çok sayıda GP varyantını yan yana koymak değil, şu soruyu cevaplamaktır:

> Seyrek P/Q/V ölçümleriyle gerilim kestiriminde, girdi seçiciliği, bus'lar arası korelasyon ve fiziksel önbilgi; doğruluk, veri ihtiyacı ve güvenilir belirsizlik bakımından ne kazandırıyor? Bu kazanım sensör yerleşimi, gürültü ve model mismatch altında korunuyor mu?

Bu soru mevcut altyapıyla bağlantılıdır. Ortak modelin normalize varyansının bütün bus'larda aynı olması, independent modelde target-specific kernel elde edilmesi ve measured/unmeasured tartışması, yöntemin neden değiştirileceğini somutlaştırıyor. Ancak çözümün daha dar PI üretmesini değil, **doğru coverage ile daha yararlı tahmin dağılımı** üretmesini hedeflemek gerekir.

Önerilen sıra: önce bilimsel baseline ve tekrar üretilebilirlik; sonra ARD; sonra gerçekten correlated bir model; ardından seçilmiş bir fiziksel önsel yaklaşımı. Calibration bu sıranın sonunda eklenen kozmetik bir düzeltme değil, bütün yöntemlerde değerlendirilecek ayrı bir eksendir.

## 2.2. Corr-GPR: kullanıcının belirttiği makaleyle mevcut modelin farkı

Kullanıcının kastettiği yayın R01: Buechel ve diğerlerinin 2026 _Distribution System State Estimation by Correlated Gaussian Processes_ çalışması. Bibliyografi ve abstract doğrulandı. Tam metin yayıncı girişi istediği için, aşağıdaki ayrıntıların tamamı R01'e atfedilmiyor.

Abstract'tan doğrulanabilenler: V ve phase angle kestirimi; individual ve coupled varyantlar; cross-correlation kullanımı; node başına 25 sample; IEEE 33 ve 69-bus örnekleri; pseudo-measurement kullanmama iddiası. Hangi P/Q/V/angle ölçümlerinin hangi varyantta kullanıldığı, online conditioning düzeni, kernel/noise parametrizasyonu ve “25 sample”ın tam seçim protokolü ancak tam metinle kesinleştirilebilir.

Aynı yöntem ailesinin R03 açık tam metni, Corr-GP'yi ICM/LCM üzerinden açıkça kuruyor. Orada girdinin zaman olduğu çok kanallı sinyal recovery görevi ele alınıyor; bazı kanallar ve zaman aralıkları gözlenirken diğerleri eksik. Bu, mevcut modeldeki bütün eğitim bus'larının aynı operating-point X matrisi üzerinde etiketli olmasıyla aynı veri düzeni değildir.

| Model | Mevcut bilgi paylaşımı | Eksik kalan özellik |
|---|---|---|
| Ortak sklearn GPR | Kernel hiperparametreleri ortak | Öğrenilmiş nonzero cross-output covariance yok |
| 43 independent GPR | Aynı sensör girdileri; ayrı kernel'lar | Joint bus posterior covariance yok |
| ICM / LMC Corr-GP | Task'lar arasında PSD covariance ve ortak latent süreçler | Tasarım ve gözlem düzenine bağlı; otomatik üstünlük sağlamaz |
| Fiziksel joint prior + conditioning | Fizikten türetilmiş state/measurement covariance | Fizik modelinin doğruluğu ve Gaussian yaklaşımı kritik |

### ICM ve LMC'nin matematiksel farkı

ICM'nin genel formu:

```text
Cov[f_i(x), f_j(x′)] = B[i,j] k(x,x′),
B = W Wᵀ + diag(κ),  κ ≥ 0.
```

LMC birden fazla latent covariance bileşeni kullanır:

```text
Cov[f_i(x), f_j(x′)] = Σ_q B_q[i,j] k_q(x,x′).
```

`B`'nin pozitif yarı tanımlı olması gerekir. İkili Pearson korelasyonlarının keyfî biçimde doldurulması, clipping veya mutlak değer almak tek başına geçerli bir kernel sağlamaz. R06 ve D02 PSD parametrizasyonu ve task likelihood tasarımı için iyi başlangıçtır.

### Bu projede nasıl bir Corr deneyi yapılabilir?

İki farklı yol birbirine karıştırılmamalı:

1. **Operating-point multitask regression:** Mevcut 39 boyutlu X aynen korunur; hedefler bus task'ları olur. Training N×43 etiketli yapı üzerinde ICM/LMC kurulur. Bu, mevcut iki GP ile doğrudan bir regression ablation'ıdır. R01'in birebir reproduksiyonu diye adlandırılmamalı.
2. **Observed/unobserved state conditioning:** Fiziksel bus değişkenleri ve/veya zaman kanalları ortak bir prior'ın parçaları olur; test anındaki gerçek P/Q/V ölçümleri doğru observation mask ile posterior'ı günceller. Ölçülmeyen bus hedefleri query edilir. R03'teki missing-channel bakışı bu tasarım için daha yakın bir motivasyondur; R01 reproduksiyonu için tam metin yine gerekir.

R01'in 25 sample/node iddiası, mevcut **30 eğitim günü** ile doğrudan karşılaştırılamaz. Bir mevsimde 30 gün yaklaşık 2.880 timestep demektir. Karşılaştırmada N timestep, bütün hedefler için toplam etiket sayısı, label erişilebilirliği ve eğitim noktası seçimi ayrı raporlanmalı.

### Önemli tuzak: korelasyon ekleyince mean mutlaka iyileşmez

R06'da noiseless, aynı X konumlarında bütün task'ların gözlendiği isotopic ICM için **autokrigeability** açıklanıyor. Basitçe, training covariance `B⊗K` ve query cross-covariance `B⊗k_*` ise ters çarpımda B sadeleşebilir. Mean ağırlıkları output-wise bağımsız kestirimle aynı hale gelebilir.

Bu sonuç her noisy ICM veya LMC için geçerli değildir. Task-specific noise, farklı gözlem maskeleri, eksik hedefler ve birden çok latent bileşen tabloyu değiştirebilir. Fakat mevcut tam etiketli training matrisine yalnızca B eklemenin neden beklenen büyük doğruluk artışını vermeyebileceğini açıklayan doğrudan ilgili bir uyarıdır.

**Tez katkısı adayı:** shared / independent / correlated modellerin, tam ve eksik gözlem düzenlerinde hangi bilgi paylaşımı kazancını sağladığını kontrollü ayırmak. Bu, “Corr daha gelişmiş, o halde daha iyi” varsayımından daha güçlü bir araştırmadır.

## 2.3. ARD: ilk uygulanması en mantıklı kontrollü genişletme

Mevcut isotropic RBF, standardized 39 boyutun tamamına tek length-scale uyguluyor:

```text
k(x,x′) = σ_f² exp[− ||x−x′||² / (2ℓ²)].
```

ARD'de her özelliğin ayrı length-scale'i var:

```text
k_ARD(x,x′) = σ_f² exp[− 1/2 Σ_d (x_d−x′_d)² / ℓ_d²].
```

Böylece gürültülü veya tekrarlı özelliklerin mesafeyi gereksiz büyütmesi azaltılabilir. Bu projede örneklemde sabit kalan P/Q sütunları ve yüksek korelasyonlu feature çiftleri bunun somut motivasyonudur. İlgili temel kaynak R05, özellikle Bölüm 5.1'dir.

Özellikle noise eklenip ardından scaler fit edildiğinde, temiz sinyali sabit sıfır olan bir P/Q sütunundaki küçük ölçüm noise'u standardized alanda birim varyanslı boyuta dönüşüyor. Kontrol örnekleminde dört sütunda bu davranış doğrulandı. ARD'nin faydası için bu özel alt durum izole edilebilir: bütün 39 girdi korunarak ARD ve isotropic modelin, yalnız bu boyutların noise'una hassasiyeti karşılaştırılır.

### Önerilen ARD karşılaştırması

- A0: mevcut isotropic RBF.
- A1: feature-wise ARD RBF; 39 length-scale.
- A2: P/Q/V türüne göre group-ARD; üç length-scale.
- A3: bus'a göre group-ARD; 13 length-scale.
- Yalnız ilk sonuçlar gerekçelendirirse Matérn-3/2 veya Matérn-5/2 ile kernel-family kontrolü.

Group-ARD, hazır RBF'ye üç değer verip 39 boyuta kendiliğinden yayılacağı varsayımıyla kurulamaz. Özellik gruplarına uygun metric/parametre paylaşımı veya active-dimension kernel çarpımları açık tanımlanmalıdır.

Toplam kernel parametreleri, WhiteKernel ve amplitude dahil: mevcut modelde **3**, feature ARD'de **41**, tür-gruplu tasarımda **5**, bus-gruplu tasarımda **15**. 43 bağımsız feature-ARD GP'de 1.763 parametre optimize edilecektir. Bu nedenle ilk ARD deneyi shared Winter model üzerinde yapılmalı; independent-ARD tam grid'i ilk adım olmamalı.

### ARD'den ne beklenebilir, ne beklenemez?

Beklenebilir: noisy gereksiz girdilerin bastırılması, daha uygun input distance, bazı senaryolarda daha iyi RMSE/NLPD ve feature relevance teşhisi. Beklenemez: otomatik fiziksel denge, kendiliğinden calibrated PI veya cross-bus correlation. **Shared ARD, shared modelin normalize posterior std'sinin bütün output'larda aynı olma özelliğini ortadan kaldırmaz.**

Length-scale karşılaştırması ancak eğitim ölçeklemesi sabit ve belgeli ise anlamlıdır. Büyük length-scale “bu özellik bu modelde küçük etki gösterdi” diye yorumlanabilir; sensörün fiziksel olarak gereksiz olduğunun veya nedensel etkisinin ispatı değildir. P/Q/V korelasyonu ve sınırlı örnek nedeniyle sıralama restart'lar arasında değişebilir.

Gerekli değerlendirme: feature ablation, top-k input ile retraining, farklı split/sensor/noise koşullarında relevance ranking stabilitesi ve validation score. `1/ℓ_d` veya `1/ℓ_d²` yalnız tanımlanmış bir ölçekli gösterge olarak verilmelidir; olasılık gibi yorumlanmamalıdır.

R03'teki Matérn-3/2 önerisi bir dakikalık **zaman** sinyali içindir. Buradaki kernel ise 39 boyutlu P/Q/V operating-point uzayında bir fonksiyonu modelliyor. Zaman serisinin pürüzlü görünmesi, bu input–output fonksiyonunun aynı kernel smoothness'ını gerektirdiğini göstermez; kernel seçimi validation ile sınanmalıdır.

### Uygulama alanı

Eski veri/metric pipeline'ı korunabilir. Ayrı model sınıfı veya experiment dosyası içinde kernel, training ve vector-valued metadata tanımlanmalı. `kernel_summary` ve diagnostics içindeki scalar length-scale varsayımları nedeniyle yalnız kernel satırı değiştirilmemeli. Uygulamaya eklenmeden önce eski run'ların dashboard yüklemesinin değişmediği regression kontrolü gerekir.

## 2.4. Physics-informed GP tek bir yöntem değildir

R07–R10 şu aileleri birbirinden ayırmayı gerektiriyor:

| Yaklaşım | Fizik nereye girer? | Bu proje için temel güçlük |
|---|---|---|
| Fiziksel mean + residual GP | `V = m_phys(z) + r_GP(z)` | Unmeasured injection'lar bilinmeden mean nasıl hesaplanacak? |
| Fiziksel covariance/prior | Ensemble PF veya sensitivity'den mean/covariance | Prior mismatch, simulation maliyeti ve Gaussian yaklaşımı |
| Graph/physics-guided kernel | Bus/task covariance B ve input metric | Topolojinin doğru, kernelin PSD ve fiziksel ölçeğin tutarlı olması |
| Lineer constrained GP | Lineer operator kernel'e veya conditional Gaussian'a uygulanır | Gerçek AC denklem nonlinear; lineer sonucun sınırı |
| Nonlinear constraint inference | Likelihood/energy içinde AC residual | Exact GP posterior kaybolur; yaklaşık inference ve yeni uncertainty analizi gerekir |

Bu ayrım terminolojik değildir. Aynı “physics-informed” adının altında çok farklı bilgi erişimi, hesap maliyeti ve garanti bulunabilir. R08'in power-grid örneği stochastic swing denklemlerinden prior çıkarır; mevcut statik LV modelinin yerine bu denklemleri kullanmak fiziksel görevi değiştirir.

## 2.5. Önerilen ana fiziksel yön: joint prior ve gerçek ölçüm conditioning'i

Bu proje için en doğal fiziksel genişletme, ölçülen ve ölçülmeyen durumları aynı prior içinde ele almaktır. Örneğin bir latent augmented state `a`, V ve gerekirse P/Q/angle değişkenlerini kapsasın. Ölçüm modeli, gerçekten ölçülen P/Q/V bileşenlerini seçsin:

```text
a ~ N(m, C),
z = H a + ε,  ε ~ N(0, R),
m_post = m + C Hᵀ (H C Hᵀ + R)⁻¹ (z − Hm),
C_post = C − C Hᵀ (H C Hᵀ + R)⁻¹ H C.
```

Burada H yalnız ölçülen değişkenleri seçiyorsa observation modeli lineerdir; P/Q'nun V/angle ile nonlinear AC ilişkisi prior üretimi içinde bulunur. Geleneksel yalnız `(V,theta)` state ve nonlinear `h(x)` kullanan AC WLS formülasyonu aynı parametrizasyon değildir. Tez hangi state tanımını kullandığını açık yazmalıdır.

Prior için iki kontrollü kaynak kurulabilir:

1. **Veri temelli joint Gaussian kontrol:** mevcut eğitimde bilinen selected P/Q/V ve hedef V'lerin joint mean/covariance'ı. Measured V hem input hem output diye iki bağımsız latent değişken olarak çoğaltılmamalı. Seçili P/Q (26) ve bütün target V (43) için 69 eşsiz bileşenle başlanabilir.
2. **Fizik temelli prior:** eğitim dönemine ait yük/PV dağılımlarından üretilen ayrı AC PF ensemble'ının mean/covariance'ı. Yeni topoloji/sensör maskelerine esneklik için daha geniş augmented state kullanılabilir. Bu ensemble'ın test ground truth'tan üretilmediği ve toplam simulation bütçesi belirtilmeli.

İlk seçenek güçlü ve ucuz bir baseline'dır: fiziksel yaklaşım yalnız shrinkage'lı covariance conditioning'ini mi iyileştiriyor, yoksa fizik gerçekten ek bilgi sağlıyor mu, görülebilir. İkinci seçenek R07–R08'deki prior-from-physics fikrinin statik AC probleme uyarlanmasıdır.

N küçük ve state boyutu büyük olduğunda sample covariance düşük rank olur. Shrinkage/PSD düzenleme ve measurement R gerekir. Bu düzenleme yalnız Cholesky başarılı olsun diye seçilmemeli; held-out validation'da score ile ayarlanmalı. Model bias veya unresolved variation ile numerical jitter aynı parametre sayılmamalı.

### Measured bus etkisini neden daha açık ele alır?

Sabit prior ve sabit diğer ölçümler altında yeni bir bilgilendirici ölçüm eklemek, Gaussian posterior covariance'ı PSD anlamında azaltır. Bu, belirsizlikte sensörün rolünü doğrudan modellemeye yarar. Temiz V ölçümü ideal koşulda ilgili bus'ın V'sini tam belirler; noisy ölçümde ölçüm variance'ı R önem kazanır.

Buna rağmen farklı bus gruplarının raw posterior variance ortalamaları ters sıralanabilir; prior variability farklıdır. Ana sınama aynı bus üzerinde yapılmalı. Sabit bütçeyle bir sensörü taşımak, yalnız yeni sensör eklemekten farklıdır ve bütün covariance için aynı monotonicity sonucunu taşımaz.

### Fizikten prior üretmek, exact AC feasibility garantisi değildir

Her ensemble örneği AC denklemini sağlasa bile nonlinear manifold üzerindeki örneklerin Gaussian conditional mean'i aynı manifold üzerinde olmak zorunda değildir. Dolayısıyla bu yöntem başlangıçta **physics-guided probabilistic voltage estimator** diye adlandırılmalıdır. Exact physics-informed/constraint satisfaction iddiası ancak kullanılan denklemler ve residual analizleriyle sınırlandırılmalıdır.

Bu yaklaşımın olası katkısı, az training label ve değişen sensör maskesinde daha tutarlı posterior elde etmektir. Olası başarısızlıkları: kötü prior, load/PV bağımlılığını yanlış modelleme, parametre/topoloji mismatch ve Gaussian approximation. Bunlar ana deney tasarımının parçası olmalıdır.

## 2.6. Alternatif fiziksel yön: sensitivity veya graph kernel

Bir çalışma noktası çevresinde gerilim değişimleri lineerleştirilebilir:

```text
ΔV ≈ S_P ΔP + S_Q ΔQ = S Δu,
C_V ≈ S C_u Sᵀ.
```

Bu, uygun PSD `C_u` için PSD bir V covariance verir. Fiziksel uzaklık, empedans ve feeder ortak yolları bus covariance'ına girebilir. Cross-covariance blokları da aynı joint modelden türetilebilir. Lineerleştirme noktası test gerçeğinden seçilmemeli; eğitim veya işletmede mevcut bir tahminden gelmelidir.

Daha basit bir graph kernel, simetrik PSD Laplacian için `B=exp(−τL)` veya `B=(L+λI)⁻¹` biçiminde kurulabilir. Bu kernel connectivity bilgisi taşır; tek başına exact AC fizik demek değildir. Bus'lar arasındaki herhangi bir “electrical distance”ı üstel içine koymak her durumda PSD sağlamaz; kernel geçerliliği ayrıca kontrol edilmelidir.

Öğrenilmiş B ile fiziksel B birleştirilebilir:

```text
B_mix = β B_phys + (1−β) B_learned,   0 ≤ β ≤ 1.
```

İki bileşen PSD ise bu karışım da PSD'dir. Ölçek belirsizliği için trace/diagonal normalizasyonu açık seçilmeli. `β=0` öğrenilmiş korelasyon, `β=1` yalnız fizik ve ara değerler için ablation yapılmalı. Topoloji veya R/X yanlış olduğunda learned bileşenin ne kadar yardımcı olduğu, tezde güçlü bir mismatch sorusudur.

Güçlükler: voltage-level/trafo dönüşümleri, reference koşulları, birimlerin tutarlılığı, açık switch'ler ve çalışma noktası değişimi. Mevcut dashboard grafı açık line switch'lerini hesaba katmadığından fiziksel kernel kaynağı olarak doğrudan kullanılmamalı. Bu yöntemin R12/R24 gibi mevcut topology-aware/multitask yayınlara göre ayrışması ayrıca incelenmeli.

## 2.7. Fiziksel mean + residual GP: pratik ama bilgi bütçesi hassas

Başlangıç modeli:

```text
V(z) = m_phys(z) + r_GP(z).
```

`m_phys`, örneğin doğrulanmış bir lineerleştirilmiş dağıtım modeli; residual GP ise sistematik yaklaşım hatasını öğrenir. Sıfır residual mean'in anlamlı bir fiziksel başlangıcı olur ve veri azlığında fayda sağlayabilir [R09].

Ancak 13 bus'ın P/Q/V'sinden bütün feeder için fiziksel mean hesaplamak, unmeasured P/Q'nun nasıl tamamlandığını açıklamayı gerektirir. Eğitimden türetilmiş load/PV prior veya pseudo-measurement kullanılabilir. Bunlar **yeni bilgi varsayımlarıdır** ve açık raporlanmalıdır. Test ground truth'taki unmeasured P/Q, modele girdi veya exact power-flow mean üretimi için verilemez.

Karşılaştırmalar: physics-only mean; GP-only; aynı fiziksel mean + residual GP; ideal/oracle mean yalnız tanı testi olarak. Fiziksel parametrelere kontrollü hata eklenmeli. Residual GP'nin başarısı sadece çok iyi bir oracle mean'den geliyorsa learned katkı ayrılmalıdır.

Bu yön, yaklaşık fiziksel modeli güvenilir kurabiliyorsan güçlü bir yedek tez planıdır. Pseudo-measurement kullanmayan ana araştırma hedefi seçilirse uygunluğu ayrıca değerlendirilmelidir.

## 2.8. Lineer constraint ve nonlinear AC physics-informed GP

Lineer `A f=b` kısıtı altında Gaussian conditioning veya operator-modified kernel, açık matematiksel sonuçlar verebilir [R07,R10]. Fakat AC power flow'da açı farklarının sinüs/kosinüsleri ve gerilim çarpımları vardır. Yalnız GP'nin mean'ine bir AC residual cezası eklemek, standart GaussianProcessRegressor'un aynı closed-form posterior'unu korumaz.

Tam nonlinear yön seçilirse şu tanımlar gerekir:

- State: en az V ve relative angle; gerekiyorsa latent P/Q.
- Observation likelihood: gerçekten ölçülen 13 bus'ın P/Q/V'si ve sensör covariance'ı.
- Fizik: doğru işaret/birim/reference ile nonlinear AC residual.
- Prior: fiziksel veya learned GP, model discrepancy ve gerekirse parameter uncertainty.
- Inference: Laplace/variational/ensemble veya başka açıkça belirtilmiş yaklaşım.
- Uncertainty: optimizer Hessian'ı, variational covariance veya ensemble'ın hangi belirsizliği temsil ettiği.

Fiziksel residual'ların MW, Mvar ve p.u. bileşenleri keyfî ağırlıklarla toplanmamalı. Ölçüm ve model belirsizliğiyle ölçeklenmeli. Gerçek AC feasible point'e projection uygulanırsa mean iyileşebilir; fakat projection öncesi PI'lar aynen tutulamaz. Dönüşüm sonrası uncertainty yeniden değerlendirilmelidir.

**Değerlendirme:** bilimsel potansiyeli yüksek, uygulama ve doğrulama riski de yüksek. İlk aşamada mevcut projeye en az müdahale isteyen yön değildir. Core tez yetişirken zorunlu kapsam yapılması önerilmiyor.

## 2.9. Girdi belirsizliği: bu proje için ARD kadar önemli bir yön

Mevcut deneyin iki temel kolu doğrudan input noise üzerine kurulu. Bu nedenle input uncertainty-aware GP, yalnız daha karmaşık bir kernel denemekten daha doğrudan araştırma değeri taşıyabilir [R11].

Noisy measurement `z=x+ε` iken aranan state dağılımı input'un da belirsizliğini içerir:

```text
E[V | z,D] = E_x[ E[V | x,D] | z ],
Var[V | z,D] = E_x[ Var[V | x,D] | z ]
                + Var_x[ E[V | x,D] | z ].
```

İkinci terim input uncertainty propagation'dır. Sadece modelin `return_std` değerini almak bu integralin yerine geçmez.

### Uygulanabilir basamaklar

1. **Test-input propagation kontrolü:** latent-clean-input modeli için seçilmiş bir `p(x|z)` altında Monte Carlo mean ve total variance hesaplamak. Örnek sayısına göre convergence ve latency ölçülmeli.
2. **Yerel lineer yaklaşım:** GP mean Jacobian'ı `J` ile yaklaşık ek variance `J Σ_x Jᵀ`; R11'in noisy-input training yaklaşımıyla ilişkisi açık kurulmalı.
3. **Training-input uncertainty:** NIGP veya latent-input variational yöntem. Hem training hem test noise için daha tutarlı, fakat daha kapsamlı değişikliktir.
4. **Noise-aware joint conditioning:** önceki fiziksel prior yaklaşımında doğru R ile ölçümleri koşullamak. Bu, genel nonlinear input-GPR integralinin birebir eşdeğeri değildir; farklı bir generative state modelidir.

Önemli sınır: `x ~ N(z,Σ)` dağılımı otomatik olarak gerçek `p(clean x | noisy z)` değildir. Genelde `p(x|z) = p(z|x)p(x)/p(z)` gerekir; heteroscedastic noise ve load/PV prior önemlidir. Locally-flat prior yaklaşımı kullanılırsa varsayım olarak yazılmalıdır.

Ayrıca noisy X üzerinde zaten öğrenilmiş conditional modelin etrafına yeniden noise eklemek aynı ölçüm hatasını iki kez sayabilir. Input-aware yöntemin latent variable ve training objective tanımı net olmadan “MC ekledik, uncertainty düzeldi” sonucuna varılmamalı. Sensor R standardized feature alanına taşınırken scaler ile covariance dönüşümü yapılmalı.

### İlave önemli fikir: ortak input noise, output'lar arasında covariance üretir

Mevcut modelde sabit x verildiğinde output süreçleri bağımsız olsa bile, aynı belirsiz x bütün output mean'lerini etkiler. Vektör mean'in Jacobian'ı `J[j,d]` ile birinci mertebe input kaynaklı covariance `C_input = J Σ_x Jᵀ` olur. Bu matrisin off-diagonal elemanları genel olarak sıfır değildir; ortak sensör hatası farklı bus tahminlerini birlikte etkiler.

Bu mekanizma, öğrenilmiş ICM/LMC prior korelasyonundan farklıdır. Tek bir shared GP'nin farklı bus mean gradient'leri üzerinden hedefe özgü input-noise hassasiyetini ve cross-bus bağımlılığı üretmeyi mümkün kılabilir. Böylece 43 ayrı kernel fit etmeden bus'lar arasındaki uncertainty farkının bir bileşeni modellenebilir. Bu bir üstünlük sonucu değil, mevcut probleme çok doğrudan bağlı bir **araştırma hipotezidir**.

Kontrol, aynı clean-input/latent-input model üzerinde plug-in ve propagated tahminleri karşılaştırmalı; ardından noisy-trained baseline ile ayrı arm olarak değerlendirmelidir. Aynı noise'un iki kez sayılmaması, input posterior varsayımı, scaler chain rule ve mean gradient'lerinin doğruluğu şarttır. Bu yaklaşımın AC feasibility veya calibration garantisi yoktur.

**Katkı adayı:** normal ve bozulmuş sensör koşullarında, ARD'nin feature seçiciliği ile fiziksel noise propagation'ın etkilerini ayrı ölçmek. Bu, mevcut clean/noisy altyapıya doğrudan dayanır.

## 2.10. Calibration: modelden bağımsız, ayrı bir araştırma katmanı

R16, Bayesian belirsizliğin model misspecification nedeniyle kalibre olmayabileceğini; R17–R19 ise interval calibration'ın veri koşullarını açıklıyor.

Basit bir kontrol, yalnız calibration kümesindeki standardized residual ile bir ölçek faktörü tahmin etmektir. Split-conformal türü normalized score:

```text
r_i = |y_i − μ_i| / max(σ_i, ε),
q = finite-sample adjusted (1−α) quantile of calibration scores,
interval = μ(x) ± q σ(x).
```

`q=1.96` sabitlenmez; calibration verisinden gelir. Bu süreç mean'i değiştirmeden aralığı düzeltir. Fakat iyileşmiş calibration'ı yeni kernelin kendi başarısı gibi sunmamak için hem **raw** hem **calibrated** sonuç verilmelidir. Bütün modeller aynı calibration bütçesini almalıdır.

### Zaman serisi ve gruplama koşulları

15 dakikalık bus×timestep hücreleri bağımsız değildir. Tüm hücrelerden quantile üretmek, kendiliğinden bus-conditional veya day-conditional garanti vermez. Klasik finite-sample split-conformal garantisi exchangeability gerektirir; gün blokları seçmek bunu otomatik sağlamaz.

Seçenekler: blok/gün temelli skor, EnbPI'nin gerçekten uygulanmış ensemble/sequential tasarımı, veya online feedback varsa adaptive conformal. Hangi coverage hedefinin kontrol edildiği belirtilmeli: rastgele bir voltage cell, belirli bus, bir snapshot'taki bütün bus'lar veya bütün gün trajectory'si.

Snapshot-level eşzamanlı aralık istenirse calibration score'unu bus maksimumu üzerinden tanımlamak düşünülebilir. Böyle bir kalibrasyonun coverage ve genişlik hedefi, mevcut marginal %95 PI'dan farklıdır. Çok az calibration bloğunda finite-sample quantile kaba veya sonsuz olabilir; nominal %95 için en az 19 exchangeable skor ancak en uç gözleme dayalı sonlu quantile verir. Daha çok bağımlı hücre eklemek bağımsız blok sayısını artırmaz.

### Coverage tek başına yeterli değil

Çok geniş interval her şeyi kapsayabilir. Coverage ile birlikte width, interval score/CRPS ve güven aralığı raporlanmalı. Teste göre q ayarlamak veya aynı günün gerçek hedefleriyle o günün intervalini düzeltmek yapılmamalı. Online calibration ancak geçmişten gelmiş gerçek feedback kullanıyorsa işletme açısından anlamlıdır.

## 2.11. Corr/ARD/physics dışında güçlü araştırma yönleri

### A. Sensör yerleşimi ve bilgi kazancı

Mevcut rastgele tek yerleşime karşı: aynı 13 sensör bütçesinde random, topology/electrical-distance coverage ve covariance tabanlı greedy information gain. Joint Gaussian modelde candidate sensor'ın posterior covariance'a etkisi doğrudan hesaplanabilir; measurement noise ve diğer sensörlerle redundant information hesaba katılmalıdır.

Bu yön, measured/unmeasured soruşturmasını “etiketli iki grubun ortalaması” olmaktan çıkarıp ölçümün marjinal bilgi değerine dönüştürür. Ama GP'yi en iyi gösteren sensör yerleşimini test hedefleriyle seçmek leakage olur. Yerleşim eğitim/validation veya önceden bilinen fizik üzerinden seçilmeli.

### B. Dropout, bias ve asenkron ölçüm

Mevcut 39-sütun scaler, eksik ölçüme veya farklı sensör maskesine hazır değildir. Ölçüm kaybında sıfırla doldurmak doğru genel çözüm değildir. Retraining, açık imputation+mask veya generative conditioning seçenekleri karşılaştırılmalı.

R13 multi-timescale aggregation için doğrudan bir GP örneği; R25 bad data/communication loss bağlamında ilgili DSSE yaklaşımıdır. Bir master tezi için önce tek sensör kaybı ve kalıcı bias yeterince anlamlı olabilir; tam cyberattack tespit sistemi ayrı kapsamdır.

### C. Topoloji/parametre mismatch ve transfer

Fizik bilmek avantaj olduğu kadar yanlış fizik altında risk de yaratır. Sabit doğru-model, R/X/trafo parametre hatası ve ayrı gerçek topoloji değişimi koşulları ayrılmalı. “Gerçek ağ değişti” ile “estimator yanlış parametre kullanıyor” aynı stres değildir [R12].

Yeni topolojide az etiketle adaptation, physics/learned covariance karışımı ve covariance shrinkage bu konuya bağlanabilir. Topoloji-aware GP zaten literatürde var; yenilik belirli az-veri/uncertainty/mismatch sorusuyla sınırlandırılmalı.

### D. Aktif eğitim örneği seçimi

30 düzenli gün yerine aynı PF-label bütçesinde coverage/uncertainty tabanlı seçilmiş operating point'ler denenebilir. Ama candidate pool final testten ayrı olmalı; ölçülemeyen test target'ı acquisition hesabında kullanılmamalı. Whole-day ve point-sampling yöntemlerinde bütçe eşitliği N timestep ve scenario coverage üzerinden tanımlanmalı.

Aktif seçimin toplam AC simulation maliyeti, sadece seçilmiş etikete değil önceden incelenmiş/simüle edilmiş candidate sayısına göre de raporlanmalı. R07'nin active-learning yönü bu konuda başlangıç kaynağıdır.

### E. Multi-fidelity GP

Daha ucuz lineer/DistFlow modelinden çok senaryo ve pahalı AC modelden az senaryo üretip model discrepancy öğrenmek, data efficiency açısından güçlü olabilir [R27]. Ancak aynı case için low/high-fidelity state ve input tanımları eşlenmeli; test ground truth'u low-fidelity input üretmek için kullanılmamalı.

Bu fikir tek başına yeni değildir ve çok-fideliteli mimari, ek simulation ve doğrulama yükü getirir. Core tezden sonra, kaynak varsa değerlendirilecek bir genişletmedir.

### F. Sparse/structured inference

43 task'lı exact multitask modeli naif dense matrise çevirmek yerine Kronecker yapısı, düşük-rank latent covariance veya variational inducing points kullanılabilir [R14]. Speedup'ın karşılığında accuracy ve calibration bozulması ölçülmeli. GPU veya yeni GP kütüphanesi kullanmak kendi başına bilimsel katkı değildir.

### G. GNN+GP ve deep kernel learning

Topology representation ardından GP uncertainty başlığı güçlü görünse de R24 ve R26 bunun zaten yoğun çalışılan bir alan olduğunu gösteriyor. Mevcut 44-bus sistem ve sınırlı çeşitlilik için GNN'nin parametre sayısı, tuning yükü ve açıklanabilirliği gereksiz büyüyebilir. Önce basit graph/sensitivity kernel'in güçlü baseline olması daha mantıklı.

## 2.12. Hangi yönde ilerlemeli?

| Yön | Mevcut sorunla bağı | Uygulama yükü | Tez içindeki önerilen rol |
|---|---|---|---|
| Baseline + doğru calibration değerlendirmesi | Çok yüksek | Düşük/orta | Zorunlu temel |
| Shared ARD / group-ARD | Yüksek | Orta | İlk kontrollü yöntem genişletmesi |
| Joint covariance conditioning | Çok yüksek | Orta | Ana yöntem adayı ve güçlü ucuz baseline |
| ICM/LMC Corr-GP | Çok yüksek | Orta/yüksek | Bilgi paylaşımı karşılaştırması; paper reproduction ayrı |
| Fiziksel mean/covariance | Çok yüksek | Orta/yüksek | Önerilen ana araştırma katkısı |
| Input uncertainty-aware GP | Çok yüksek | Orta/yüksek | Noise çalışmasının güçlü odak alternatifi |
| Sensor placement / dropout | Yüksek | Orta | Ölçüm rolü ve robustness deneyleri |
| Tam nonlinear AC-constrained GP | Yüksek | Çok yüksek | İleri aşama; çekirdek için zorunlu değil |
| GNN/deep kernel | Orta | Yüksek | İlk aşamada önceliksiz |
| Multi-fidelity / aktif öğrenme | Yüksek | Yüksek | Core başarılı olursa genişletme |

Bu puanlar deneysel başarı sonucu değil, mevcut kod ve veri yapısına göre mühendislik/araştırma değerlendirmesidir. ARD'nin veya physics prior'ın üstün geleceği garanti edilmiyor. Negatif sonuç da hangi varsayım altında ve neden başarısız olduğu gösterilirse tez açısından değerlidir.

## 2.13. Özgünlük ve savunulabilir katkı

“GPR'ye ARD eklemek”, “43 ayrı GP fit etmek”, “ICM kullanmak” veya “physics-informed adını eklemek” tek başına yeni yöntem iddiası için yeterli değildir. Buna karşılık şu birleşim tutarlı bir master tezi verebilir:

1. Shared modelin uncertainty alanının rank-1 ölçek yapısını matematiksel olarak açıklamak ve gerçek veride kontrol etmek.
2. Aynı bus üzerinde rol değiştiren sensör deneyleriyle ölçümün marjinal katkısını incelemek.
3. Öğrenilmiş ve fiziksel joint prior'ları aynı input/label bütçesinde, doğru sensor noise ile karşılaştırmak.
4. Raw ve calibrated uncertainty'yi hem nominal hem mismatch koşullarında proper score ile sınamak.
5. Data efficiency ve hesap maliyetini aynı protokol içinde göstermek.

Birinci madde burada türetilen, mevcut sklearn kurulumuna ilişkin bir teşhistir; bütün GP literatüründe yeni bir teorem olarak sunulmamalı. Önerilen sentezin yayın düzeyinde yeni olup olmadığını söylemek için R01 tam metni ve en yakın çalışmaların daha ayrıntılı taraması gerekir. Yüksek lisans tezi için bilimsel katkı, yeni algoritma icadı kadar dikkatli yöntem karşılaştırması ve yeni problem bulguları da olabilir; program ve danışman ölçütleri belirleyicidir.
