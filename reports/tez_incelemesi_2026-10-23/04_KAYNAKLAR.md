# 4. Kaynaklar, okuma kapsamı ve literatür taramasının sınırı

**Erişim tarihi:** 23 Ekim 2026. Kaynaklar, mevcut kodun açığa çıkardığı araştırma sorularına göre tarandı. Kullanıcının belirttiği Corr-GPR başlığı başlangıç noktası olarak alındı; Crossref, OpenAlex, Semantic Scholar, yayıncı ve açık yazar/preprint sürümleriyle bibliyografi kontrol edildi. Ardından Corr-GPR referansları üzerinden ilgili DSSE çalışmaları izlendi. Arama motoru özetleri, özellikle otomatik üretilen özetler, teknik sonuçların kanıtı olarak kullanılmadı.

Bu, kapsamı önceden kayıtlı PRISMA türü bir sistematik tarama değildir. Başlık/özet taraması ile erişilen tam metinlerin yöntem incelemesi ayrı tutulmuştur. Bir makalenin başlığı veya özeti bulunmuş olması tam metninin okunduğu anlamına gelmez. Yeni yöntemlerin özgünlüğü hakkında kesin öncelik iddiası yapılmamıştır.

## Corr-GPR ve DSSE

**[R01] Buechel, M.; Zimmer, M.; Carta, D.; Benigni, A. (2026). _Distribution System State Estimation by Correlated Gaussian Processes._ IEEE Transactions on Instrumentation and Measurement.** [DOI](https://doi.org/10.1109/TIM.2026.3684671) — [kurumsal kayıt](https://publications.rwth-aachen.de/record/1036173).

Erişim: **bibliyografi ve abstract doğrulandı; tam metin okunamadı**. Yayıncı PDF isteği login sayfasına yönlendi; OpenAlex açık tam metin göstermedi. Abstract, V ve açı için individual/coupled varyantlar, cross-correlation, node başına 25 sample ve IEEE 33/69-bus değerlendirmesi bildiriyor. Bu bilgi, tam kernel/likelihood/conditioning düzenini veya ölçüm setini yeniden üretmek için yeterli değildir. Rapor, bu ayrıntıları makaleye atfetmez.

**[R02] Buechel, M.; Zimmer, M.; Carta, D.; Benigni, A. (2025). _Distribution Grid Voltage Estimation by Correlated Gaussian Processes._ IEEE AMPS.** [DOI](https://doi.org/10.1109/AMPS66841.2025.11219969).

Erişim: **bibliyografi ve abstract**. Ölçüm azlığı altında voltage estimation, küçük eğitim seti ve farklı monitoring configurations vurgulanıyor. Yayıncı tam metin için giriş istedi. R01'in doğrudan önceki voltage-estimation çalışması olarak değerlidir; sonuçları mevcut SimBench deneyinin sayılarıyla doğrudan karşılaştırılmadı.

**[R03] Zimmer, M.; Carta, D.; Pesch, T.; Benigni, A. (2024). _Signal Recovery in Power Systems by Correlated Gaussian Processes._ IEEE Open Journal of the Industrial Electronics Society, 5, 692–703.** [DOI](https://doi.org/10.1109/OJIES.2024.3423405) — [açık yayıncı PDF](https://ieeexplore.ieee.org/ielx8/8782706/10417853/10585294.pdf).

Erişim: **tam metin**. Özellikle Bölüm II–III: GP conditioning, `B⊗K` separable kernel, ICM ve LCM; Bölüm IV–V: gerçek LLEC sinyalleri ve missing-interval recovery. Bu yayındaki zaman ekseni ve kısmi gözlem düzeni, mevcut 39 boyutlu operating-point regresyonuyla aynı değildir. R01'in tam metni yerine geçmez; Corr-GP ailesinin açık matematiksel dayanağını sağlar.

**[R04] Primadianto, A.; Lu, C.-N. (2017). _A Review on Distribution System State Estimation._ IEEE Transactions on Power Systems.** [DOI](https://doi.org/10.1109/TPWRS.2016.2632156).

Erişim: **bibliyografi/özet**. DSSE problem tanımı, ölçüm eksikliği ve conventional estimator bağlamını kurmak için kaynak. Tam metin üzerinden belirli algoritma performans değerleri aktarılmadı.

## GP matematiği, korelasyon ve fizik

**[R05] Rasmussen, C. E.; Williams, C. K. I. (2006). _Gaussian Processes for Machine Learning._ MIT Press.** [kitap](https://gaussianprocess.org/gpml/) — [Bölüm 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf) — [Bölüm 4](https://gaussianprocess.org/gpml/chapters/RW4.pdf) — [Bölüm 5](https://gaussianprocess.org/gpml/chapters/RW5.pdf) — [Bölüm 9](https://gaussianprocess.org/gpml/chapters/RW9.pdf).

Erişim: **ilgili bölümlerin tam metni**. Predictive mean/variance, covariance tasarımı, ARD/length-scale, marginal likelihood ve uncertain input tartışmaları için temel kaynak. Normalized multioutput varyans sonucu ayrıca yüklü sklearn implementasyonundan ve gerçek veri kontrolünden türetildi.

**[R06] Álvarez, M. A.; Rosasco, L.; Lawrence, N. D. (2012). _Kernels for Vector-Valued Functions: A Review._ Foundations and Trends in Machine Learning.** [DOI](https://doi.org/10.1561/2200000036) — [açık sürüm](https://arxiv.org/abs/1106.6251).

Erişim: **açık tam metin**. ICM/LMC, covariance matrix, heterotopic/isotopic gözlemler ve noiseless isotopic ICM'deki **autokrigeability / cancellation of inter-task transfer** tartışması bu proje için özellikle önemlidir. “Korelasyon eklemek her durumda mean doğruluğunu artırır” savını desteklemez.

**[R07] Yang, X.; Tartakovsky, G. D.; Tartakovsky, A. M. _Physics-Information-Aided Kriging: Constructing Covariance Functions using Stochastic Simulation Models._** [açık sürüm](https://arxiv.org/abs/1809.03461), incelenen v3: 2021.

Erişim: **tam metin**. Stochastic physics realizations'dan prior mean/covariance, PhIK, multilevel Monte Carlo ve active learning. İlk preprint 2018, incelenen sürüm 2021 tarihlidir; başlık sürümler arasında değişmiştir. Deterministik **lineer** fiziksel constraint sonuçları nonlinear AC denklemine otomatik taşınamaz.

**[R08] Tartakovsky, A. M.; Ma, T.; Barajas-Solano, D. A.; Tipireddy, R. (2023). _Physics-informed Gaussian process regression for states estimation and forecasting in power grids._ International Journal of Forecasting.** [DOI](https://doi.org/10.1016/j.ijforecast.2022.03.007) — [2020 açık preprint](https://arxiv.org/abs/2010.04591).

Erişim: **açık preprint tam metni; journal bibliyografisi doğrulandı**. Bölüm 2.2, stochastic modelden Monte Carlo prior istatistiklerini kuruyor; örnek sistem swing denklemleri, faz açısı, angular speed ve wind mechanical power içeriyor. Bu, 15 dakikalık statik balanced LV voltage-magnitude modelinin doğrudan kopyalanacak çözümü değildir.

**[R09] Cross, E. J. ve diğerleri (2024). _A spectrum of physics-informed Gaussian processes for regression in engineering._ Data-Centric Engineering.** [DOI](https://doi.org/10.1017/dce.2024.2) — [açık 2023 preprint](https://arxiv.org/abs/2309.10656).

Erişim: **açık preprint tam metni; journal metadata doğrulandı**. Fiziksel mean, derived/data-driven covariance birleşimi ve constrained covariance ayrımını kuruyor. “Physics-informed GP”nin tek bir yöntem olmadığını gerekçelendirmek için kullanıldı. Preprint ve journal yazar/sürüm ayrıntıları birebir aynı varsayılmadı.

**[R10] Raissi, M.; Karniadakis, G. E. (2017). _Machine Learning of Linear Differential Equations using Gaussian Processes._** [açık tam metin](https://arxiv.org/abs/1701.02440).

Erişim: **tam metin**. Lineer differential operator'un kernel'e uygulanması ve conservation-law parameter inference. Nonlinear AC güç akışı için exact Gaussian conjugacy iddiasına dayanak olarak kullanılmadı; lineer/nonlinear sınırını açıklamak için kullanıldı.

**[R11] McHutchon, A.; Rasmussen, C. E. (2011). _Gaussian Process Training with Input Noise._ NeurIPS.** [tam metin](https://proceedings.neurips.cc/paper/2011/file/a8e864d04c95572d1aece099af852d0a-Paper.pdf).

Erişim: **tam metin**. Noisy input'ları yerel lineer açılımla gradient'e bağlı output noise'a dönüştürür; test-input belirsizliği de ele alınır. WhiteKernel'in tek başına sensor input uncertainty propagation olmadığına doğrudan ilgili bir yöntem kaynağıdır.

**[R12] Cao, D.; Zhao, J.; Hu, W.; Liao, Q.; Huang, Q.; Chen, Z. (2023). _Topology Change Aware Data-Driven Probabilistic Distribution State Estimation Based on Gaussian Process._ IEEE Transactions on Smart Grid.** [DOI](https://doi.org/10.1109/TSG.2022.3204221).

Erişim: **bibliyografi ve abstract**. Topolojiler arasında transfer sağlayan kernel ve az yeni-topoloji verisiyle adaptation vurgulanıyor. Tam yöntem veya nicel üstünlük rakamları aktarılmadı.

**[R13] Dahale, S.; Natarajan, B. _Recursive Gaussian Process Over Graphs for Integrating Multi-Timescale Measurements in Low-Observable Distribution Systems._ IEEE Transactions on Power Systems, 38(4), 2023; early access/preprint 2022.** [DOI](https://doi.org/10.1109/TPWRS.2022.3204415) — [açık tam metin](https://arxiv.org/abs/2209.01703).

Erişim: **açık tam metin**. Recursive multitask GP ile asenkron/multi-timescale veri uzlaştırması; devamında matrix-completion DSSE. Mevcut synchronous 15-minute pipeline'ın doğrudan eşdeğeri değildir. Sensör dropout/asenkronluk yönünün literatürdeki karşılığını gösterir.

**[R14] Titsias, M. (2009). _Variational Learning of Inducing Variables in Sparse Gaussian Processes._ AISTATS, PMLR 5:567–574.** [yayın](https://proceedings.mlr.press/v5/titsias09a.html) — [tam metin](https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf).

Erişim: **tam metin**. Variational inducing-variable yaklaşımı ve lower bound optimizasyonu. Naif dense multitask GP'nin bellek gereksinimi için uygulanabilir bir alternatif; yaklaşık inference'ın calibration etkisi ayrıca ölçülmelidir.

## Belirsizlik, kalibrasyon ve deney geçerliliği

**[R15] Gneiting, T.; Raftery, A. E. (2007). _Strictly Proper Scoring Rules, Prediction, and Estimation._ Journal of the American Statistical Association.** [DOI](https://doi.org/10.1198/016214506000001437).

Erişim: **bibliyografi doğrulandı**. Proper scoring yaklaşımının temel referansı. Raporun Gaussian NLPD, CRPS ve interval score önerileri bu genel değerlendirme çerçevesindedir; bu yayının tam metni incelendiği iddia edilmez.

**[R16] Kuleshov, V.; Fenner, N.; Ermon, S. (2018). _Accurate Uncertainties for Deep Learning Using Calibrated Regression._ ICML, PMLR 80:2796–2804.** [yayın](https://proceedings.mlr.press/v80/kuleshov18a.html) — [tam metin](https://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf).

Erişim: **tam metin**. Bayesian/probabilistic modelin nominal aralığının gerçek coverage'a eşit olmayabileceğini ve regression calibration yaklaşımını tartışıyor. Model calibration'ının veri bağımlılığı koşulları unutulmamalı.

**[R17] Angelopoulos, A. N.; Bates, S. (2021 preprint). _A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification._** [tam metin](https://arxiv.org/abs/2107.07511).

Erişim: **tam metin**. Split conformal, finite-sample quantile ve marginal coverage. Exchangeability koşulu olmadan mevcut zaman serisi için otomatik %95 garanti çıkarılmadı.

**[R18] Xu, C.; Xie, Y. (2021). _Conformal prediction interval for dynamic time-series._ ICML, PMLR 139:11559–11569.** [yayın](https://proceedings.mlr.press/v139/xu21h.html) — [tam metin](https://proceedings.mlr.press/v139/xu21h/xu21h.pdf).

Erişim: **tam metin**. EnbPI; mixing error koşulları altında yaklaşık marginal coverage. Her residual quantile yöntemi EnbPI diye adlandırılmamalı; bootstrap ensemble ve sequential düzen ayrı gerekliliklerdir.

**[R19] Gibbs, I.; Candès, E. J. (2021). _Adaptive Conformal Inference Under Distribution Shift._** [tam metin](https://arxiv.org/abs/2106.00170).

Erişim: **tam metin**. Değişen dağılımda online interval adaptation ve uzun dönem coverage kontrolü. Future ground truth kullanmadan adaptation yapılması ve online feedback'in gerçekten mevcut olması gerekir.

**[R20] Roberts, D. R. ve diğerleri (2017). _Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure._ Ecography.** [DOI](https://doi.org/10.1111/ecog.02881).

Erişim: **bibliyografi/özet düzeyi**. Bağımlı veride validation tasarımının önemine dayanak. Buradaki “231/246 gün eğitim gününe en fazla bir gün uzaklıkta” bulgusu bu yayından alınmadı; mevcut split'ten hesaplandı.

## Fiziksel veri ve diğer ilgili yönler

**[R21] Meinecke, S. ve diğerleri (2020). _SimBench—A Benchmark Dataset of Electric Power Systems to Compare Innovative Solutions Based on Power Flow Analysis._ Energies 13(12), 3290.** [DOI](https://doi.org/10.3390/en13123290).

Erişim: **bibliyografi; ayrıca projede kullanılan resmi paket ve gerçek takvim/PF çıktıları doğrudan incelendi**. Bu raporda SimBench'in bütün feeder'larının veya gerçek saha başarımının test edildiği iddia edilmez.

**[R22] Thurner, L. ve diğerleri (2018). _pandapower—An Open-Source Python Tool for Convenient Modeling, Analysis, and Optimization of Electric Power Systems._ IEEE Transactions on Power Systems.** [DOI](https://doi.org/10.1109/TPWRS.2018.2829021).

Erişim: **bibliyografi ve kullanılan açık kaynak paket/API**. Fiziksel kontrol pandapower 3.1.2 ile yapıldı. Bu sürüm seçimi kullanıcının denklemlerini değiştirmek için kullanılmadı.

**[R23] Vanin, M.; Van Acker, T.; D'hulst, R.; Van Hertem, D. (2023). _Exact Modeling of Non-Gaussian Measurement Uncertainty in Distribution System State Estimation._ IEEE Transactions on Instrumentation and Measurement.** [DOI](https://doi.org/10.1109/TIM.2023.3287253) — [tam metin](https://arxiv.org/abs/2303.03029).

Erişim: **açık tam metin**. Continuous non-Gaussian uncertainty ve unbalanced three-phase PF kullanan constrained maximum-likelihood SE. Gaussian, balanced mevcut baseline'ın saha genelleme sınırına ilişkin önemli karşılaştırma yönüdür.

**[R24] Hu, J. ve diğerleri (2024). _Robust Multiarea Distribution System State Estimation Based on Structure-Informed Graphic Network and Multitask Gaussian Process._ IEEE Transactions on Industrial Informatics, 20(8), 10599–10612.** [DOI](https://doi.org/10.1109/TII.2024.3394553).

Erişim: **bibliyografi ve abstract**. Graph attention representation, multitask kernel, multiarea/robustness ve interval estimation. GNN+GP fikrinin tek başına özgün katkı sayılmaması gerektiğini gösterir; bu raporda mimarisi yeniden uygulanmadı.

**[R25] Zhang, J.; Liu, Y.; Rouhani, A.; Zhao, J.; Warchol, G. J.; Scott, K. (2024). _A Robust Data-Driven Gaussian Process Regression-Enabled Distribution System State Estimation using Partial Node Data._ IEEE PES General Meeting.** [DOI](https://doi.org/10.1109/PESGM51994.2024.10688998).

Erişim: **bibliyografi ve abstract**. Partial node data, pseudo-measurement inference, gross error/communication loss ve büyük bir gerçek feeder üzerinde değerlendirme bildiriyor. Tezde “sparse measurement + GPR”nin başlı başına yeni olmadığını gösteren doğrudan ilgili çalışma.

**[R26] Hu, J.; Xu, Q. (2025). _A Gaussian Process-Regularized Graphical Learning Method for Distribution System State Estimation Using Extremely Scarce State Variable Labels._ IEEE Transactions on Smart Grid.** [DOI](https://doi.org/10.1109/TSG.2025.3552958).

Erişim: **başlık/bibliyografi doğrulandı**. Tam metin ve sonuçlar incelenmedi. Az etiketli physics/graphical GP alanında ek novelty taraması için aday; bu kaynağa dayanarak yöntem ayrıntısı veya başarım iddiası kurulmadı.

**[R27] Zhang, J.; Zhao, J.; Yang, J.; Zhao, J. (2024 issue). _Deep Multi-Fidelity Bayesian Data Fusion for Probabilistic Distribution System Voltage Estimation With High Penetration of PVs._ IEEE Transactions on Power Systems.** [DOI](https://doi.org/10.1109/TPWRS.2023.3295795).

Erişim: **bibliyografi doğrulandı**. Tam metin okunmadı. Çok-fideliteli voltage-estimation yaklaşımının zaten literatürde bulunduğunu gösteren ek tarama kaynağı; rapordaki özgünlüğe ilişkin sınır bu düzeyde tutuldu.

**[R28] Tan, B.; Su, T.; Weng, Y.; Ye, K.; Pareek, P.; Vorobev, P.; Nguyen, H.; Zhao, J.; Deka, D. (2025 preprint). _Gaussian Processes in Power Systems: Techniques, Applications, and Future Works._** [açık tam metin](https://arxiv.org/abs/2505.15950).

Erişim: **tam metin**. GP'nin power-flow, dynamics, risk, input uncertainty ve scalability bağlamlarını haritalamak için kullanıldı. Preprint olması, otomatik olarak hakemli yayın kabul edildiği anlamına gelmez. İçindeki ilgili DSSE referansları mümkün olduğunda ayrı bibliyografik kaynaklardan kontrol edildi.

## Resmi implementasyon belgeleri

**[D01] scikit-learn Gaussian processes belgeleri.** [resmi belge](https://scikit-learn.org/stable/modules/gaussian_process.html). Erişim: belge ve kurulu **1.8.0** sürümündeki `GaussianProcessRegressor.predict` kaynak kodu. Normalize output std çarpanı ve `WhiteKernel` katkısı gerçek hesapla ayrıca doğrulandı. Canlı stable belge sürümü ile kurulu sürümün birebir aynı olduğu varsayılmadı.

**[D02] GPflow coregionalization örneği.** [resmi belge](https://gpflow.github.io/GPflow/develop/notebooks/advanced/coregionalisation.html). Erişim: tam notebook belgesi. `B=WWᵀ+diag(κ)` PSD parametrizasyonu, output index ve task-specific likelihood kurulumu için uygulama örneği. Bu notebook'u görmek, bu projede GPflow modelinin uygulanıp test edildiği anlamına gelmez.

## Kaynaklardan çıkarılmayan iddialar

- Corr-GPR, ARD, GNN+GP, PhIK ve calibration'ın tek tek yeni icat olduğu söylenmiyor.
- Yayınların farklı feeder/noise/target protokollerindeki RMSE rakamları, kullanıcının modeliyle doğrudan sıralanmıyor.
- PDE veya lineer operator için verilen constraint sonucunun nonlinear AC power flow için geçerli olduğu varsayılmıyor.
- Çalışmaların yalnız abstract'ı erişildiyse kernel, likelihood veya runtime ayrıntıları tamamlanmış gibi gösterilmiyor.
- Bu rapor, 23 Ekim 2026 itibarıyla bulunabilecek bütün ilgili yayınları kapsadığını veya önerilen birleşimin kesin yeni olduğunu iddia etmiyor.
