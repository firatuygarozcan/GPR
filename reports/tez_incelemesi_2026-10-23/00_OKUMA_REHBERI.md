# GPR projesi: tez hazırlık incelemesi

**Tarih:** 23 Ekim 2026. **İncelenen sürüm:** `440f90365c3ed0743139122726f8e737d82000d1`.

Bu çalışma, mevcut projeyi bir yüksek lisans tezine dönüştürmek için hazırlanmış teknik denetim, kaynak incelemesi ve deney önerisidir. Uygulamanın sekiz Python dosyası ve mevcut test dosyası değiştirilmedi. Yeni model uygulanmadı; burada önerilen yöntemlerin mevcut yıllık deneyden daha iyi sonuç verdiği iddia edilmiyor.

## Okuma sırası

1. [Teknik inceleme](01_TEKNIK_INCELEME.md): mevcut modelin gerçekte yaptığı iş, doğrulanan bulgular, bilimsel sınırlamalar ve öncelikler.
2. [Literatür ve yöntem önerileri](02_LITERATUR_VE_YONTEMLER.md): kullanıcının belirttiği Corr-GPR çalışması, ARD, fiziksel önseller, girdi belirsizliği ve diğer araştırma yönleri.
3. [Tez ve deney protokolü](03_TEZ_VE_DENEY_PROTOKOLU.md): önerilen araştırma soruları, karşılaştırmalar, veri ayrımı, metrikler, ablation deneyleri ve aşamalı iş planı.
4. [Kaynaklar ve erişim düzeyi](04_KAYNAKLAR.md): doğrulanan bibliyografiler, açık tam metinler ve yalnızca özetine erişilen yayınlar.
5. [Kanıt dizini](kanitlar/KANIT_DIZINI.md): çalıştırılan kontroller, JSON sonuçları, kaynak dosya özetleri ve doğrulama sınırları.

## Esas değerlendirme

Projede kullanılabilir bir deney altyapısı var: P/Q/V girdileri, temiz hedefler, mevsim bazında bütün gün ayrımı, ayrı noise-free/noisy deneyler, bağımsız Winter deneyi ve ayrıntılı değerlendirme çıktıları. Fakat mevcut haliyle yöntem ve değerlendirme kapsamı, güçlü bir tez iddiasını desteklemek için genişletilmeli. Öncelik, yeni kernel sayısını artırmak yerine neyin neden iyileştiğini ölçebilecek araştırma düzenini kurmaktır.

En uygulanabilir ana yön: **seyrek ölçümlü dağıtım şebekesinde korelasyon ve fiziksel önsel kullanan gerilim kestiriminin, girdi gürültüsü ve sensör yerleşimi altında doğruluk ve belirsizlik bakımından karşılaştırılması.** ARD kontrollü bir ara basamak; kalibrasyon zorunlu değerlendirme bileşeni; tam nonlinear physics-informed GP ise ileri aşama olarak öneriliyor.

## Doğrulamanın kapsamı

- Mevcut testler: **17 test, 22 alt kontrol geçti**. 48 SimBench/pandas `FutureWarning` kaydı var; bunlar optimizer uyarısı değildir.
- Gerçek SimBench takvimi: **35.136 timestep, 366 gün**, DST günleri 92 ve 100 timestep olarak doğrulandı.
- Ayrı bir çalışma dizininde **384 gerçek dengeli AC power-flow adımı** çalıştırıldı. GPR kontrolleri bunun 48 eğitim ve 24 test noktalı alt örnekleminde yapıldı.
- Theta değerleri yapay olarak değiştirilince P/Q/V girdi matrisi ve gerilim hedefleri değişmedi.
- Ortak kernelin hedef ölçeği çıkarılmış standart sapmaları çıktılar arasında aynı; stress/günlük noise farkı ve cache/topoloji dayanıklılığına ilişkin bulgular kontrollü denemelerle doğrulandı.

**Kullanıcının bilgisayarındaki yıllık `outputs` ve ground-truth cache bu ortamda bulunmuyor.** Bu nedenle kullanıcının yıllık MAE/RMSE/coverage değerleri doğrulanmadı; dört mevsim ve 43 bağımsız GPR'nin tam yıllık karşılaştırması yeniden çalıştırılmadı. Örneklem sonuçları yıllık model performansı olarak kullanılmamalı.

## Kaynak erişimi hakkında

Belirtilen **“Distribution System State Estimation by Correlated Gaussian Processes”** makalesinin 2026 bibliyografisi ve özeti doğrulandı; yayıncı tam metin için giriş istedi. Buna karşılık aynı yöntem ailesinin 2024 **“Signal Recovery in Power Systems by Correlated Gaussian Processes”** makalesinin açık tam metni incelendi. Bu iki erişim düzeyi raporda ayrı tutuldu. Diğer açık kaynaklar arasında GPML bölümleri, vektör değerli kernel incelemesi, power-grid PhI-GPR, PhIK, input-noise GP, zaman serisi conformal inference ve sparse GP yayınları bulunuyor.

Bu rapor bir sistematik literatür taraması veya özgünlük garantisi değildir. Belirli sorulara odaklı kaynak taraması ve uygulanabilir araştırma tasarımıdır. Bir yöntemin tez katkısı olabileceği değerlendirmesi, danışmanın ve programın kabul ölçütlerinin yerini tutmaz.
