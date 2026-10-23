# 5. İnceleme kanıtları ve tekrar üretim sınırı

## 5.1. Kaynak sürüm ve değişiklik kontrolü

İncelenen commit: **440f90365c3ed0743139122726f8e737d82000d1**. Uygulamanın sekiz Python dosyası ve mevcut test dosyası, inceleme başında SHA-256 ile kaydedildi. İnceleme sonunda bu dokuz dosyanın byte düzeyindeki hash'leri aynı bulundu.

- `source_hashes_before.json` ve `source_hashes_after.json`: kaynak değişmezliği.
- `code_index.json`: function/class adları ve kaynak satır konumları.
- Yalnız `reports/tez_incelemesi_2026-10-23/` altında rapor, görsel ve kanıt dosyaları eklendi.

İnceleme scriptleri ve paket ortamı `/workspace/thesis-review-2026-10-23/` ve `/workspace/.venvs/gpr-review/` altında, uygulama checkout'ının dışında kuruldu. Uygulama `outputs` klasörü oluşturulmadı. Kullanıcının bilgisayarındaki dosyalara bu oturum erişmedi.

## 5.2. Gerçekten çalıştırılan kontroller

| Kontrol | Sonuç | Kanıt |
|---|---|---|
| Mevcut dashboard testleri | 17 passed, 22 subtests passed | `existing_tests.log` |
| SimBench gerçek takvimi | 35.136 step, 366 gün; DST 92/100 | `scientific_probe.json` |
| Mevsimsel split hesabı | 120 train / 246 test gün; overlap yok | Aynı JSON, `split` |
| Gerçek balanced AC PF | Day 1,4,2,365; toplam 384 step; finite states | Aynı JSON |
| P/Q/V input / theta ayrımı | Theta değişince X ve V hedefleri aynı | `robustness_probe.json` |
| Ortak GPR normalize std | Output farkı yaklaşık 10⁻¹⁶ veya altında | `scientific_probe.json`, `gpr` |
| Elle Cholesky predictive variance | sklearn sonucu ile numerical precision düzeyinde aynı | Aynı JSON |
| PI width / target scale grup oranı | Aynı oran; örnekte 1,0635136934 | Aynı JSON |
| Stress subset noise | Aynı gün/seed, farklı row selection: 39/39 input değişiyor | Aynı JSON |
| Saf-noise feature ölçekleme | Örnekte dört sıfır-sinyal P/Q sütunu noisy scaling sonrası std=1 | `noise_scaling_probe.json` |
| Cache fingerprint | Stub metadata, %10 line-R değişimini ayıramıyor | `robustness_probe.json` |
| Open line switch grafı | Kontrollü kopyada dashboard edge kalıyor, resmi graf düşürüyor | Aynı JSON |

Bu kontrollerde yanlışlığında çalışmayı durduran assertion'lar kullanıldı. Örneklemdeki GP, iki training gününden her dört timestep'te bir alınmış **48 training point** ve iki test gününden her sekiz timestep'te bir alınmış **24 test point** kullanıyor. Bu, mevcut 30-gün training deneyinin yerine geçirilmedi.

`scientific_probe.json` içinde örnek model MAE/RMSE/coverage değerleri teknik kayıt olarak bulunur. **Bunlar yıllık performans tablosu veya yöntem sıralaması için kullanılmamalı.** Burada 43 independent model veya yeni ARD/Corr/physics modeli fit edilmedi.

Cache denemesinde uygulama cache'i değil, dış dizinde oluşturulmuş stub dosyalar kullanıldı. Topoloji denemesinde ağın bir kopyası üzerinde graf davranışı karşılaştırıldı; açılmış line için yeni fiziksel PF senaryosu üretildiği iddia edilmiyor. Mevcut ağda açık switch sayısı sıfırdır.

## 5.3. Test komutu ve ortam

Mevcut suite şu komutla çalıştırıldı:

```bash
cd /workspace/GPR
PYTHONDONTWRITEBYTECODE=1 \
NUMBA_CACHE_DIR=/workspace/.cache/numba \
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
XDG_CACHE_HOME=/workspace/.cache \
MPLCONFIGDIR=/workspace/.cache/matplotlib \
/workspace/.venvs/gpr-review/bin/python -m pytest -q \
  tests/test_dashboard_heatmaps.py
```

Bu komut yolu kullanılan bulut makinesine aittir. Başka bilgisayarda uygun Python ortamı/yol seçilmelidir. Testler kontrollü voltage verisi, gerçek SimBench takvimi ve bazı mock'lar kullanır; bütün yıllık power-flow/GPR run'ını doğrulamaz.

Temel sürümler: Python 3.12.14, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, scikit-learn 1.8.0, pandapower 3.1.2 ve SimBench 1.6.1. Tam paket listesi `requirements_tested.txt` içindedir. Bu liste, orijinal projenin önceden var olan lockfile'ı değildir; inceleme ortamının kaydıdır.

Testlerde 48 SimBench/pandas `FutureWarning` görüldü. Bunlar test başarısızlığı veya GPR optimizer hatası olarak sayılmadı. GPR'nin kendi convergence warning'lerinin uygulamada bastırılması ise teknik raporda ayrı bir bulgudur.

## 5.4. Literatür erişim kaydı

`source_access_manifest.json`, doğrudan fetch edilen kaynakların URL, erişim sonucu, son URL, içerik türü ve SHA-256 bilgisini içerir. Bibliyografik Crossref/OpenAlex aramaları ayrıca dış çalışma dizininde saklandı; makale PDF/HTML tam metinleri bu rapor klasörüne yeniden dağıtılmadı.

HTTP 200 tek başına makale erişimi değildir. Özellikle 2026 ve 2025 Corr makalesi PDF istekleri login HTML'ine yönlendi. Kaynak dizini bu yayınları **abstract/metadata** düzeyinde işaretler. 2024 Corr makalesi ise gerçek açık PDF olarak indirildi ve incelendi.

Akademik kaynaklar için bulut ortamının ağ taslağına kaynak alan adları eklendi; bu oturum için ağ erişimi sağlandı. Uygulama kaynaklarında veya bilimsel konfigürasyonda değişiklik yapılmadı. Ağ taslağının kalıcı yayımlanması ayrı ortam yönetimi işlemidir; bu raporun sonuçları gelecekteki bir cloud oturumunun hazır olduğuna dair garanti değildir.

## 5.5. Önemli kod dayanakları

Aşağıdaki bağlantılar, raporun yazıldığı commit'e sabittir:

| Bulgu | Kaynak |
|---|---|
| Ölçüm ve noise konfigürasyonu | [config.py](https://github.com/firatuygarozcan/GPR/blob/440f90365c3ed0743139122726f8e737d82000d1/seasonal/config.py) |
| Dataset, yalnız P/Q/V input | [dsse.py:1087](https://github.com/firatuygarozcan/GPR/blob/440f90365c3ed0743139122726f8e737d82000d1/seasonal/dsse.py#L1087) |
| Noise formülü ve gün bazında RNG | [dsse.py:1146](https://github.com/firatuygarozcan/GPR/blob/440f90365c3ed0743139122726f8e737d82000d1/seasonal/dsse.py#L1146) |
| Shared model ve warnings | [dsse.py:1218](https://github.com/firatuygarozcan/GPR/blob/440f90365c3ed0743139122726f8e737d82000d1/seasonal/dsse.py#L1218) |
| Cache doğrulaması | [dsse.py:732](https://github.com/firatuygarozcan/GPR/blob/440f90365c3ed0743139122726f8e737d82000d1/seasonal/dsse.py#L732) |
| Independent bus başına fit | [winter_independent_evaluation.py:203](https://github.com/firatuygarozcan/GPR/blob/440f90365c3ed0743139122726f8e737d82000d1/seasonal/winter_independent_evaluation.py#L203) |
| Ana noisy stress alt kümesi | [stress_analysis.py:253](https://github.com/firatuygarozcan/GPR/blob/440f90365c3ed0743139122726f8e737d82000d1/seasonal/stress_analysis.py#L253) |
| Winter noisy stress alt kümesi | [winter_independent_evaluation.py:161](https://github.com/firatuygarozcan/GPR/blob/440f90365c3ed0743139122726f8e737d82000d1/seasonal/winter_independent_evaluation.py#L161) |
| Günlük heatmap doğrulaması | [dashboard.py:862](https://github.com/firatuygarozcan/GPR/blob/440f90365c3ed0743139122726f8e737d82000d1/seasonal/dashboard.py#L862) |
| Otomatik renk limitleri | [dashboard.py:1095](https://github.com/firatuygarozcan/GPR/blob/440f90365c3ed0743139122726f8e737d82000d1/seasonal/dashboard.py#L1095) |
| Fiziksel graf helper'ı | [dashboard.py:1375](https://github.com/firatuygarozcan/GPR/blob/440f90365c3ed0743139122726f8e737d82000d1/seasonal/dashboard.py#L1375) |
| Mevcut regression suite | [test_dashboard_heatmaps.py](https://github.com/firatuygarozcan/GPR/blob/440f90365c3ed0743139122726f8e737d82000d1/tests/test_dashboard_heatmaps.py) |

## 5.6. Bu incelemede yapılmayanlar

- Kullanıcının yıllık cache/CSV/model dosyaları bulunmadığı için onların cell-by-cell audit'i.
- 35.136 fiziksel timestep'in tamamının yeni PF koşusu ve dört 30-gün modelinin full fit'i.
- 43 bağımsız GPR'nin bütün yeni senaryolarda eğitimi.
- Yeni ARD, Corr-GPR, physics-informed veya calibration implementasyonu.
- Gerçek saha, üç-faz, farklı feeder veya yeni topoloji performans ölçümü.
- Yeni tarayıcı üzerinde bütün gerçek sonuçlarla dashboard acceptance testi.
- Uygulama kodu değişikliği veya bu raporların GitHub'a yayımlanması.

Yukarıdaki sınırlar, tamamlanan kaynak/kod incelemesini geçersiz kılmaz; raporun hangi güven düzeyinde kullanılacağını belirler.
