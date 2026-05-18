# **Kısıtlı Video Belleği Koşullarında AMD Polaris Mimarisi İçin Yüksek Performanslı Görsel Çıkarım ve Arayüz Elemanı Tespiti Stratejileri**

Düşük Video Belleği (VRAM) kapasitesine sahip donanımlarda gerçek zamanlı görsel çıkarım (vision inference) süreçlerini optimize etmek, modern derin öğrenme yaklaşımlarının donanım kısıtlarıyla olan geriliminin merkezinde yer almaktadır. Özellikle AMD Radeon RX 460 gibi 2 GB VRAM sınırına sahip ve Polaris 11 mimarisine dayanan grafik işlem birimlerinde (GPU), saniyede 60 kare (FPS) hızında ekran piksellerini analiz etmek, geleneksel derin öğrenme çerçevelerinin (framework) ötesine geçen, donanıma yakın seviyede C tabanlı mühendislik çözümleri gerektirir.1 Ryzen 5 3600 işlemcisinin sunduğu çok çekirdekli performans ile RX 460'ın sınırlı ancak Vulkan destekli hesaplama gücü arasındaki sinerjiyi maksimize etmek için model mimarisi, çıkarım motoru ve bellek yönetim stratejilerinin birbirine tam uyumlu bir ekosistem oluşturması elzemdir.3

## **Donanım Kısıtları ve Polaris Mimarisi Üzerinde Bellek Darboğazları**

AMD RX 460, dördüncü nesil GCN (Graphics Core Next) mimarisini kullanmakta olup 896 veya 1024 akış işlemcisine (stream processor) sahiptir. 128-bit bellek veri yolu üzerinden yaklaşık 112 GB/sn bant genişliği sunan bu kartta, 2 GB VRAM sınırı yalnızca model parametrelerini değil, aynı zamanda çalışma anındaki aktivasyon haritalarını ve sürücü seviyesindeki tampon bellekleri de barındırmak zorundadır.1 Gerçek zamanlı bir çıkarım hattında (inference pipeline), piksellerin DMA-BUF üzerinden doğrudan GPU belleğine aktarılması, veri yolu üzerindeki gecikmeyi minimize etmek için temel bir gerekliliktir.5

Bellek bant genişliği kullanımı (Model Bandwidth Utilization \- MBU), bu tip eski nesil donanımlarda saf işlem gücünden (TFLOPS) daha kritik bir darboğaz oluşturmaktadır. Bir modelin parametre sayısı düşük olsa bile, evrişimli katmanların (convolutional layers) ürettiği ara verilerin büyüklüğü VRAM sınırını hızla aşabilir veya bellek parçalanmasına (fragmentation) yol açabilir.7 Bu nedenle, çıkarım motorunun bellek tahsisat stratejisi, modelin katmanları arasındaki veriyi "sıfır-kopya" (zero-copy) prensibiyle yönetebilmeli ve bellek bloklarını dinamik olarak yeniden kullanabilmelidir.4

| Donanım Bileşeni | Özellik | Çıkarım Sürecine Etkisi |
| :---- | :---- | :---- |
| GPU (RX 460\) | Polaris 11 / GCN 4.0 | Vulkan 1.2+ desteği, FP16 hesaplama hızlandırması |
| VRAM | 2 GB GDDR5 | Katı model boyutu ve aktivasyon haritası sınırı |
| Veri Yolu | PCIe 3.0 x8 | DMA-BUF veri aktarım hızı sınırı |
| İşlemci (Ryzen 5 3600\) | 6C/12T Zen 2 | Ön işleme ve çıkarım sonrası koordinat kod çözme gücü |

## **Optimize Edilmiş Görsel Model Mimari Sınıfları**

2 GB VRAM sınırında 60 FPS hedefi, model mimarisinin hem hesaplama karmaşıklığını (FLOPs) hem de bellek erişim yoğunluğunu minimize etmesini zorunlu kılar. Bu bağlamda, geleneksel çapa tabanlı (anchor-based) dedektörler yerine, parametre verimliliği yüksek olan ve çıkarım sonrası işlem yükünü azaltan modern mimari sınıflarına odaklanılmalıdır.10

### **Çapasız Tek Aşamalı Regresyon Ağları**

Web arayüzü butonları ve iframe alanları gibi belirgin geometrik sınırlara sahip objeleri tespit etmek için en rasyonel yaklaşım, çapasız (anchor-free) tek aşamalı regresyon mimarileridir. Bu sınıftaki modeller, nesne tespitini bir piksel bazlı regresyon problemi olarak ele alarak, önceden tanımlanmış kutu şablonlarına (anchors) olan ihtiyacı ortadan kaldırır.10 Bu durum, bellek kullanımını azaltırken, NMS (Non-Maximum Suppression) gibi çıkarım sonrası adımların işlem yükünü hafifletir. Özellikle arayüz elemanlarının genellikle dikdörtgen ve hizalanmış yapıda olması, bu mimarilerin düşük çözünürlüklü girdilerle (örneğin 416x416) yüksek doğruluk sunmasını sağlar.10

### **Ghost ve Derinlikli Ayrılabilir Evrişim Mekanizmaları**

Bellek bant genişliği kısıtlı olan RX 460 için "Ghost" modülleri içeren mimariler büyük bir avantaj sağlar. Ghost modülleri, bir miktar "intrinsik" özellik haritası ürettikten sonra, bu haritalardan ucuz doğrusal dönüşümlerle (linear transformations) "hayalet" (ghost) haritalar türetir.12 Bu yöntem, parametre sayısını ve dolayısıyla bellekten çekilen ağırlık miktarını yarı yarıya azaltırken ağın temsil kapasitesini korur. Benzer şekilde, derinlikli ayrılabilir evrişimler (depthwise separable convolutions), standart evrişimleri iki aşamaya (filtreleme ve birleştirme) bölerek hesaplama maliyetini dramatik şekilde düşürür.14

### **Hafifletilmiş Görsel Transformatörler (EfficientViTs)**

Tam ölçekli Görsel Transformatörler (ViT), 2 GB VRAM için genellikle çok ağırdır. Ancak, boyutsal olarak tutarlı tasarımlara sahip olan ve dikkat (attention) mekanizmalarını yalnızca ağın derin aşamalarında kullanan hibrit modeller, web arayüzündeki iframe gibi geniş bağlam (context) gerektiren alanların tespitinde etkili olabilir.16 Bu modeller, evrişimli katmanların yerel özellik çıkarma yeteneği ile transformatörlerin küresel ilişki kurma becerisini birleştirir. Polaris donanımı üzerinde bu modellerin çıkarım süreleri, transformatör bloklarının bellek erişim desenleri nedeniyle CNN bazlı rakiplerine göre daha yüksek seyredebilir.18

| Mimari Sınıfı | Avantaj | VRAM Ayak İzi | Beklenen Gecikme (ms) |
| :---- | :---- | :---- | :---- |
| Ghost-CNN | Çok düşük bant genişliği ihtiyacı | \< 20 MB | 6 \- 9 ms |
| Çapasız Hafif CNN | Hızlı çıkarım sonrası işlem | \< 30 MB | 8 \- 12 ms |
| Hibrit ViT-CNN | Yüksek bağlamsal farkındalık | 80 \- 150 MB | 25 \- 45 ms |
| Ters Çevrilmiş Darboğaz CNN | Hassas küçük eleman tespiti | \< 40 MB | 10 \- 15 ms |

## **C-Native Çıkarım Motorlarının Teknik Analizi ve Karşılaştırması**

Python bağımlılığı olmayan bir C projesinde, RX 460 GPU'sunun tüm potansiyelini kullanmak için doğrudan Vulkan API'sine erişen çıkarım motorları tercih edilmelidir. AMD'nin Polaris mimarisi için ROCm desteğinin sınırlı olması, Vulkan tabanlı motorları en kararlı seçenek haline getirir.4

### **NCNN: Düşük Seviyeli Bellek Optimizasyonu**

Tencent tarafından geliştirilen NCNN, tamamen C++ ile yazılmış olup mobil ve uç donanımlar için en optimize edilmiş motorlardan biridir. NCNN'in en büyük avantajı, hiçbir üçüncü taraf bağımlılığına ihtiyaç duymaması ve doğrudan Vulkan komut kuyruklarını (compute queues) yönetebilmesidir.3 NCNN, "blob" yapısı üzerinden bellek yönetimini gerçekleştirir ve katmanlar arasındaki veriyi minimum kopyalama ile aktarır. Winograd evrişim algoritmalarını desteklemesi, Polaris mimarisindeki hesaplama birimlerinin verimliliğini %20-30 oranında artırabilir.20

NCNN ayrıca, GPU üzerinde "sıfır-kopya çıkarım zinciri" (zero-copy inference chaining) kurmaya olanak tanır. DMA-BUF üzerinden gelen piksel verisi bir VkMat nesnesine bağlandıktan sonra, çıkarım süreci boyunca veri GPU belleğinden hiç çıkmadan tüm ağ katmanlarından geçebilir.4

### **MNN: Geometrik Hesaplama ve Arka Uç Soyutlama**

Alibaba'nın MNN motoru, karmaşık operatörleri atomik işlemlere bölen "geometrik hesaplama" yaklaşımıyla öne çıkar.21 MNN, farklı GPU mimarileri için çalışma anında (runtime) otomatik kernel seçimi yaparak en hızlı yolu bulmaya çalışır. RX 460 gibi donanımlarda, MNN'in operatör füzyonu (operator fusion) yetenekleri, bellek erişim sayısını azaltarak performansa katkıda bulunur. Ancak NCNN'e kıyasla ikili (binary) boyutu biraz daha büyüktür ve karmaşık modellerde VRAM yönetimi daha fazla ince ayar gerektirebilir.22

### **ONNX Runtime (C API) ve Vulkan Yürütme Sağlayıcısı**

Microsoft ONNX Runtime, evrensel model desteği sunsa da standart dağıtımları 2 GB VRAM'li sistemler için hantaldır. Ancak, "Minimal Build" seçeneği ile yalnızca gerekli operatörleri içerecek şekilde derlendiğinde, RX 460 üzerinde Vulkan Execution Provider (EP) aracılığıyla yüksek performans sergileyebilir.23 ONNX Runtime'ın en büyük gücü, eğitim aşamasında kullanılan popüler framework'lerden (PyTorch, TensorFlow) modele kayıpsız geçiş imkanı sunmasıdır.25

| Motor | C/C++ API | GPU Arka Ucu | RX 460 Uyumluluğu | Tipik İlk Kurulum Belleği |
| :---- | :---- | :---- | :---- | :---- |
| NCNN | Yerli C++ | Vulkan (Doğrudan) | Mükemmel | \< 50 MB |
| MNN | Yerli C++ | Vulkan / OpenCL | Mükemmel | \~70 MB |
| ONNX Runtime | C API | Vulkan EP | İyi (Kısıtlı Derleme ile) | \> 150 MB |
| LibTorch | C++ API | Vulkan (Deneysel) | Düşük | \> 500 MB |

## **DMA-BUF ve Sıfır-Kopya Piksel Analizi Entegrasyonu**

Saniyede 60 kare hızında 1080p çözünürlüğünde bir ekranı analiz etmek, her saniye yaklaşık 373 MB verinin CPU ve GPU arasında taşınması anlamına gelir. Bu işlem, PCIe veri yolunu ve CPU döngülerini gereksiz yere tüketir. Linux çekirdeğinin sunduğu DMA-BUF altyapısı, ekran kartının tampon belleğindeki piksel verisinin bir dosya tanımlayıcısı (FD) üzerinden çıkarım motoruna paylaşılmasını sağlar.5

### **Vulkan Dış Bellek Aktarımı (VK\_KHR\_external\_memory\_fd)**

DMA-BUF FD, Vulkan çıkarım motoruna VK\_KHR\_external\_memory\_fd uzantısı kullanılarak ithal edilir. Bu süreçte veri kopyalanmaz; bunun yerine çıkarım motoru, ekran kartı üzerindeki mevcut bellek bölgesine doğrudan erişim yetkisi kazanır.27 RX 460 üzerinde bu entegrasyonu sağlarken, senkronizasyon için VkFence veya VkSemaphore yapılarının kullanılması, çıkarım işleminin ekran yenileme döngüsüyle (V-Sync) çakışmasını engellemek için hayatidir.5

### **Asenkron Çıkarım Hattı Tasarımı**

Gerçek zamanlı 60 FPS performansı için tek bir çıkarım döngüsü yeterli değildir. Üçlü tamponlama (triple buffering) mantığıyla çalışan bir hat kurulmalıdır. CPU tarafında Frame ![][image1] için DMA-BUF hazırlanırken, GPU tarafında Frame ![][image2] üzerinde çıkarım yapılmalı ve CPU aynı anda Frame ![][image3] sonuçlarını web arayüzü koordinatlarına dönüştürmelidir.28 NCNN'in Extractor ve VkCompute sınıfları bu asenkron yapıyı destekleyerek, çıkarım sırasında CPU'nun diğer görevleri yürütmesine olanak tanır.4

## **UI Eleman Tespiti: Parametre Sayısı ve Reaksiyon Hızı Dengesi**

Web arayüzü elemanlarını tanımak, doğal sahnelerdeki nesneleri tanımaya göre çok daha kısıtlı bir özellik setine sahiptir. Butonlar ve iframe alanları genellikle yüksek kontrastlı kenarlara, metin bloklarına ve düzenli geometrilere sahiptir. Bu basitlik, modelin "daraltılması" (pruning) ve parametre sayısının azaltılması için rasyonel bir zemin oluşturur.30

### **100ms Altı Reaksiyon İçin Karmaşıklık Üst Sınırı**

Bir kullanıcının gecikmeyi hissetmemesi için uçtan uca reaksiyon süresinin 100ms altında kalması gerekir. Bu süre; ekran yakalama, ön işleme, çıkarım ve çıkarım sonrası adımların toplamıdır. RX 460'ın donanım kapasitesi göz önüne alındığında, 60 FPS hedefi çıkarım aşamasına en fazla 16.6ms süre tanır.16

Yapılan analizler, bu süre zarfında Polaris mimarisi üzerinde **1.5M ile 2.5M arasında parametreye** sahip modellerin en iyi dengeyi sunduğunu göstermektedir.11 5M parametrenin üzerindeki modeller, VRAM bant genişliği sınırlamaları nedeniyle 30ms'nin üzerine çıkarak kare hızının 30 FPS'nin altına düşmesine sebep olur.33

### **Nicemleme (Quantization) ve Polaris Donanım Hızlandırması**

Nicemleme sürecinde, ağırlıkların FP32'den FP16 veya INT8 formatına dönüştürülmesi, bellek ayak izini %50 ila %75 oranında azaltabilir. Ancak, RX 460'ın donanım özellikleri burada belirleyici bir rol oynar. Polaris mimarisi, FP16 (Yarım Hassasiyetli) hesaplamaları FP32'ye göre iki kat hızlı gerçekleştirebilen "Rapid Packed Math" özelliğine sahiptir.36

Aksine, Polaris kartlarında modern Tensor çekirdekleri bulunmadığı için INT8 nicemlemesi genellikle yazılım seviyesinde emüle edilir. Bu durum, model boyutunu küçültse de çıkarım süresini (latency) artırabilir veya FP16'ya göre bir avantaj sağlamayabilir.36 Bu nedenle, RX 460 projesinde **FP16 nicemlemesi** en rasyonel yaklaşımdır.28

### **Model Budama (Pruning) Stratejileri**

Arayüz elemanları gibi kısıtlı bir set için "yapılandırılmış budama" (structured pruning) yapmak oldukça karlıdır. Gereksiz evrişim filtrelerinin kaldırılması, GPU çekirdeklerinin veri beklerken boşa geçirdiği zamanı azaltır. Web butonları gibi düşük doku varyasyonuna sahip nesneler için, modelin ilk katmanlarındaki özellik haritası sayısı dramatik şekilde azaltılabilir, bu da çıkarım süresinde doğrusal bir iyileşme sağlar.28

| Optimizasyon Türü | RX 460 Getirisi | Risk | Öneri |
| :---- | :---- | :---- | :---- |
| FP16 Nicemleme | 1.8x \- 2x Hızlanma | Düşük Doğruluk Kaybı | Şiddetle Tavsiye Edilir |
| INT8 Nicemleme | Bellek Tasarrufu | Latency Artışı (Emülasyon) | Kaçınılmalıdır |
| Yapılandırılmış Budama | Bant Genişliği Tasarrufu | Önemli Katman Kaybı | Kısıtlı Set İçin Uygundur |
| Giriş Boyutu Küçültme | Üstel Hızlanma | Küçük İkon Kaybı | 416x416 Dengelidir |

## **Uygulama Mimarisi ve C Entegrasyon Detayları**

Geliştirilen C projesinde NCNN veya MNN motorunu DMA-BUF ile birleştirmek için izlenmesi gereken düşük seviyeli programlama adımları, donanım kaynaklarının verimli kullanımını garanti eder.

### **Bellek Tahsisatçıları ve Parçalanma Yönetimi**

2 GB VRAM sınırında, grafik sürücüsünün (Mesa/RADV veya AMDVLK) ve çıkarım motorunun bellek kullanımını kontrol altında tutmak gerekir. NCNN, VkBlobAllocator ve VkStagingAllocator sınıfları üzerinden bellek bloklarını yönetir.9 Çıkarım motoruna, modelin en büyük aktivasyon haritasına göre önceden tahsis edilmiş (pre-allocated) bir bellek havuzu sağlamak, çalışma anındaki gecikme dalgalanmalarını (jitter) önler.

### **Çıkarım Sonrası İşlem (Post-processing) Verimliliği**

Modelin çıktısı olan regresyon haritalarını buton koordinatlarına dönüştürmek, GPU yerine CPU üzerinde Ryzen 5 3600'ün SIMD (AVX2) komut setini kullanarak yapılmalıdır. NCNN'den gelen çıktı ncnn::Mat nesnesi, AVX2 ile optimize edilmiş döngülerle taranarak 0.1ms \- 0.5ms gibi sürelerde sonuçlandırılabilir.4 Bu, GPU'nun bir sonraki kare için çıkarıma daha erken başlamasını sağlar.

### **Eğitim Verisi ve Alan Uyarlaması**

Web butonları ve iframe'ler için eğitilecek modelin başarısı, veri setinin çeşitliliğine bağlıdır. Sadece web elementlerine odaklanmak, modelin parametre sayısını azaltırken "aşırı uyum" (overfitting) riskini artırabilir. Modeli önceden eğitilmiş (pre-trained) bir hafif omurga (backbone) üzerine inşa etmek ve sadece web arayüzleri içeren veri setleriyle (örneğin VNIS veri seti) ince ayar (fine-tuning) yapmak, çıkarım hızı ve doğruluk arasındaki en ince çizgidir.30

## **Sonuç ve Stratejik Yol Haritası**

AMD RX 460 ve Ryzen 5 3600 kombinasyonunda, 2 GB VRAM sınırında 60 FPS web arayüzü tespiti yapabilmek için aşağıdaki stratejik kararların uygulanması rasyoneldir:

1. **Mimari Seçimi:** 1.0M \- 2.0M parametre aralığında, "Ghost" modülleri kullanan, çapasız bir tek aşamalı regresör (örneğin NanoDet-Plus veya türevleri) tercih edilmelidir.10  
2. **Çıkarım Motoru:** AMD GPU'larındaki Vulkan performans üstünlüğü ve minimal C API desteği nedeniyle **NCNN** motoru kullanılmalıdır.3  
3. **Veri Akışı:** Gecikmeyi minimize etmek için pikseller DMA-BUF üzerinden VK\_KHR\_external\_memory\_fd ile GPU'ya "sıfır-kopya" aktarılmalıdır.5  
4. **Donanım Optimizasyonu:** Polaris mimarisinin sunduğu FP16 hızlandırmasından yararlanılmalı, ancak özel donanım desteği olmayan INT8 nicemlemesinden kaçınılmalıdır.36  
5. **Giriş Çözünürlüğü:** Arayüz elemanlarının detaylarını korumak ve VRAM'i zorlamamak adına giriş çözünürlüğü **416x416** veya **320x320** olarak sabitlenmelidir.10

Bu yaklaşımlar, modern derin öğrenme tekniklerini eski nesil ama yetenekli donanımlarla buluşturarak, endüstriyel standartlarda tepki süresi ve verimlilik sunan bir görsel analiz sistemi oluşturulmasını mümkün kılmaktadır. Polaris 11 mimarisinin bellek bant genişliği ve Ryzen 3600'ün işlem kapasitesi, doğru mühendislik tercihleriyle 2 GB sınırını aşan bir performans sergileyebilir.

#### **Works cited**

1. Radeon RX 460 \- Price performance comparison \- Video Card Benchmarks, accessed May 11, 2026, [https://www.videocardbenchmark.net/gpu.php?gpu=Radeon+RX+460\&id=3557](https://www.videocardbenchmark.net/gpu.php?gpu=Radeon+RX+460&id=3557)  
2. RX 460 4GB vs. 2GB VRAM Benchmark \- Hardware Unboxed : r/Amd \- Reddit, accessed May 11, 2026, [https://www.reddit.com/r/Amd/comments/4zojk4/rx\_460\_4gb\_vs\_2gb\_vram\_benchmark\_hardware\_unboxed/](https://www.reddit.com/r/Amd/comments/4zojk4/rx_460_4gb_vs_2gb_vram_benchmark_hardware_unboxed/)  
3. PyTorch vs ONNX vs NCNN \- by Nadira Povey \- Medium, accessed May 11, 2026, [https://medium.com/@nadirapovey/pytorch-vs-onnx-vs-ncnn-ee50115b6263](https://medium.com/@nadirapovey/pytorch-vs-onnx-vs-ncnn-ee50115b6263)  
4. FAQ ncnn vulkan \- ncnn documentation \- Read the Docs, accessed May 11, 2026, [https://ncnn.readthedocs.io/en/latest/how-to-use-and-FAQ/FAQ-ncnn-vulkan.html](https://ncnn.readthedocs.io/en/latest/how-to-use-and-FAQ/FAQ-ncnn-vulkan.html)  
5. Buffer Sharing and Synchronization (dma-buf) \- The Linux Kernel documentation, accessed May 11, 2026, [https://docs.kernel.org/driver-api/dma-buf.html](https://docs.kernel.org/driver-api/dma-buf.html)  
6. Buffer Sharing and Synchronization (dma-buf) \- The Linux Kernel Archives, accessed May 11, 2026, [https://www.kernel.org/doc/html/v6.7/driver-api/dma-buf.html](https://www.kernel.org/doc/html/v6.7/driver-api/dma-buf.html)  
7. Compare Inference Latency Across AI Providers, accessed May 11, 2026, [https://www.gmicloud.ai/ja/blog/compare-inference-latency-ai-providers](https://www.gmicloud.ai/ja/blog/compare-inference-latency-ai-providers)  
8. The Hidden Bottlenecks in LLM Inference: Why TFLOPs and VRAM Don't Determine Performance \- virtualizationvelocity, accessed May 11, 2026, [https://www.virtualizationvelocity.com/home/the-hidden-bottlenecks-in-llm-inference](https://www.virtualizationvelocity.com/home/the-hidden-bottlenecks-in-llm-inference)  
9. vulkan notes \- ncnn documentation \- Read the Docs, accessed May 11, 2026, [https://ncnn.readthedocs.io/en/latest/how-to-use-and-FAQ/vulkan-notes.html](https://ncnn.readthedocs.io/en/latest/how-to-use-and-FAQ/vulkan-notes.html)  
10. Wulingtian/nanodet \- GitHub, accessed May 11, 2026, [https://github.com/Wulingtian/nanodet](https://github.com/Wulingtian/nanodet)  
11. NanoDet-Plus Super fast and lightweight anchor-free object detection model. Only 980 KB(int8) / 1.8MB (fp16) and run 97FPS on cellphone · GitHub, accessed May 11, 2026, [https://github.com/RangiLyu/nanodet](https://github.com/RangiLyu/nanodet)  
12. Review — GhostNet: More Features from Cheap Operations | by Sik-Ho Tsang \- Medium, accessed May 11, 2026, [https://sh-tsang.medium.com/review-ghostnet-more-features-from-cheap-operations-1784f3bbc2b](https://sh-tsang.medium.com/review-ghostnet-more-features-from-cheap-operations-1784f3bbc2b)  
13. ghostshiftaddnet: more features from energy-efficient operations \- arXiv, accessed May 11, 2026, [https://arxiv.org/pdf/2109.09495](https://arxiv.org/pdf/2109.09495)  
14. GhostNetV3: Exploring the Training Strategies for Compact Models \- arXiv, accessed May 11, 2026, [https://arxiv.org/html/2404.11202v1](https://arxiv.org/html/2404.11202v1)  
15. RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization \- arXiv, accessed May 11, 2026, [https://arxiv.org/pdf/2211.06088](https://arxiv.org/pdf/2211.06088)  
16. Lightweight Transformer Architectures for Edge Devices in Real-Time Applications \- arXiv, accessed May 11, 2026, [https://arxiv.org/html/2601.03290v1](https://arxiv.org/html/2601.03290v1)  
17. Small Vision-Language Models: A Survey on Compact Architectures and Techniques \- arXiv, accessed May 11, 2026, [https://arxiv.org/html/2503.10665v1](https://arxiv.org/html/2503.10665v1)  
18. Lightweight Vision Transformer Coarse-to-Fine Search via Latency Profiling \- OpenReview, accessed May 11, 2026, [https://openreview.net/pdf?id=sTdd0yCOZ2](https://openreview.net/pdf?id=sTdd0yCOZ2)  
19. AMD GPU Acceleration Technologies Explained: ROCm, HIP, Vulkan, OpenCL & More (2025) \- GitHub Gist, accessed May 11, 2026, [https://gist.github.com/danielrosehill/8793e2028ef4bd08c6ca955a38b40e5b](https://gist.github.com/danielrosehill/8793e2028ef4bd08c6ca955a38b40e5b)  
20. ncnn uses much more memory than other inference frameworks. · Issue \#2750 \- GitHub, accessed May 11, 2026, [https://github.com/Tencent/ncnn/issues/2750](https://github.com/Tencent/ncnn/issues/2750)  
21. MNN: A blazing-fast, lightweight inference engine battle-tested by Alibaba, powering high-performance on-device LLMs and Edge AI. \- GitHub, accessed May 11, 2026, [https://github.com/alibaba/mnn](https://github.com/alibaba/mnn)  
22. MNN: A Universal and Efficient Inference Engine \- MLSys Proceedings, accessed May 11, 2026, [https://proceedings.mlsys.org/paper\_files/paper/2020/file/bc19061f88f16e9ed4a18f0bbd47048a-Paper.pdf](https://proceedings.mlsys.org/paper_files/paper/2020/file/bc19061f88f16e9ed4a18f0bbd47048a-Paper.pdf)  
23. Build ONNX Runtime \- Build with different EPs \- GitHub Pages, accessed May 11, 2026, [https://oliviajain.github.io/onnxruntime/docs/build/eps.html](https://oliviajain.github.io/onnxruntime/docs/build/eps.html)  
24. \[Feature Request\] Add vulkan execution provider · Issue \#21917 · microsoft/onnxruntime, accessed May 11, 2026, [https://github.com/microsoft/onnxruntime/issues/21917](https://github.com/microsoft/onnxruntime/issues/21917)  
25. onnxruntime \- ONNX Runtime, accessed May 11, 2026, [https://onnxruntime.ai/docs/](https://onnxruntime.ai/docs/)  
26. ONNX Runtime GPU | Guides \- Clore.ai, accessed May 11, 2026, [https://docs.clore.ai/guides/gpu-devops/onnx-runtime](https://docs.clore.ai/guides/gpu-devops/onnx-runtime)  
27. External Memory and Synchronization \- Vulkan Documentation, accessed May 11, 2026, [https://docs.vulkan.org/guide/latest/extensions/external.html](https://docs.vulkan.org/guide/latest/extensions/external.html)  
28. What Are Best Practices for Building Low-Latency Vision AI Pipelines for Real-Time Video?, accessed May 11, 2026, [https://getstream.io/blog/low-latency-vision-ai/](https://getstream.io/blog/low-latency-vision-ai/)  
29. 16 contributors, cross-stack improvements: Collabora's work on GStreamer 1.28, accessed May 11, 2026, [https://www.collabora.com/news-and-blog/news-and-events/16-contributors-cross-stack-improvements-collabora-work-gstreamer-128.html](https://www.collabora.com/news-and-blog/news-and-events/16-contributors-cross-stack-improvements-collabora-work-gstreamer-128.html)  
30. UI-UG: A Unified MLLM for UI Understanding and Generation \- arXiv, accessed May 11, 2026, [https://arxiv.org/html/2509.24361v1](https://arxiv.org/html/2509.24361v1)  
31. mujacica/ui-element-privacy \- GitHub, accessed May 11, 2026, [https://github.com/mujacica/ui-element-privacy](https://github.com/mujacica/ui-element-privacy)  
32. Systematic Review of Quantization-Optimized Lightweight Transformer Architectures for Real-Time Fruit Ripeness Detection on Edge Devices \- MDPI, accessed May 11, 2026, [https://www.mdpi.com/2073-431X/15/1/69](https://www.mdpi.com/2073-431X/15/1/69)  
33. Choosing the Right LLM Size vs Latency: The Essential Balance for Optimal AI Performance, accessed May 11, 2026, [https://estha.ai/blog/choosing-the-right-llm-size-vs-latency-the-essential-balance-for-optimal-ai-performance/](https://estha.ai/blog/choosing-the-right-llm-size-vs-latency-the-essential-balance-for-optimal-ai-performance/)  
34. Benchmarking YOLOv8 Variants for Object Detection Efficiency on Jetson Orin NX for Edge Computing Applications \- MDPI, accessed May 11, 2026, [https://www.mdpi.com/2073-431X/15/2/74](https://www.mdpi.com/2073-431X/15/2/74)  
35. YOLOv8 Nano vs YOLOv8 large. Why do we use YOLOv8 nano for Raspberry… | by Elven Kim | Medium, accessed May 11, 2026, [https://medium.com/@elvenkim1/yolov8-nano-vs-yolov8-large-4f21324baa38](https://medium.com/@elvenkim1/yolov8-nano-vs-yolov8-large-4f21324baa38)  
36. FP16 vs. INT8: Speed vs. Efficiency \- YouTube, accessed May 11, 2026, [https://www.youtube.com/shorts/lnDmoab\_leo](https://www.youtube.com/shorts/lnDmoab_leo)  
37. Speech-to-Text Engines GPU Inference Guide: AMD ROCm vs NVIDIA CUDA (2025), accessed May 11, 2026, [https://gist.github.com/danielrosehill/f7ae5e659a02e32d056eb7887203b7a4](https://gist.github.com/danielrosehill/f7ae5e659a02e32d056eb7887203b7a4)  
38. INT8 quantization gives me better accuracy than FP16 \! \[D\] : r/MachineLearning \- Reddit, accessed May 11, 2026, [https://www.reddit.com/r/MachineLearning/comments/1sx35es/int8\_quantization\_gives\_me\_better\_accuracy\_than/](https://www.reddit.com/r/MachineLearning/comments/1sx35es/int8_quantization_gives_me_better_accuracy_than/)  
39. How I Trained YOLOv8 to Detect Mobile UI Elements Using the VNIS Dataset \- Medium, accessed May 11, 2026, [https://medium.com/@eslamelmishtawy/how-i-trained-yolov8-to-detect-mobile-ui-elements-using-the-vnis-dataset-f7f4b582fc09](https://medium.com/@eslamelmishtawy/how-i-trained-yolov8-to-detect-mobile-ui-elements-using-the-vnis-dataset-f7f4b582fc09)  
40. How to Increase Inference Speed for Computer Vision Models \- Roboflow Blog, accessed May 11, 2026, [https://blog.roboflow.com/increase-inference-speed-for-computer-vision/](https://blog.roboflow.com/increase-inference-speed-for-computer-vision/)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAYCAYAAABXysXfAAABkklEQVR4Xu2WvytFYRzGHyUlP5KFMshEKZmEJBkMJovFxOIPIJuFsNhkMCiDUjIYpMjkTpJSBmzuQin5GQMDz3u/53bP+5V7zzn3dFDvpz517/M9t85z3nPPewCHw5GHJh38N6poH12lD/boO930gF7RT3pkjzPs0Q/I/IbO2OOCTNE2HQagnd7SDXpOH+3xz+zTNOSEO+xRhgm6psOAzNJOHYZkBwHLVNALOgops21NhWXar8OAzCHBMgN0iZYhtzrN/gPIMWQehXkkWGaRDnmfJyFlTLksjXTX9z0siZY5odXe50p6R59pjZeNQ/4zUUmsTB1NqWwBsjpmlQybkCdLIYYhvwvqO+TiBcGUedKhZoROq6yevtFryHP+0h6HJq6VKVjGbEZmr9GsQK7eumcxxFXG3Pp5MZtRqQ5JC3K3w5iahSWuMi869DNIT2mJHnhsQco06EFI4ihj3lJeIXuiRQ/kCZa98mna6z/Ao4ue6TACUcvU0kPIO1n2XO8hxUyHXyFqmT9JKy3XocPhcBTFF93AYckXJCQ1AAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAABBElEQVR4XmNgGAUDBqyAeDcQ3wfi/0B8AlUaDHYA8W8GiPwzIG5ElUYFO4H4AQNEsRmqFBgUAfF8dEF0wA3E14E4gQFi0HoUWQiYAsRO6ILowA2IJwExGwPCVerICoDgFANEHi/oBuIAKLuYAWIQyGAYkAfirUh8nOAMEPNB2TxA/AaIPwGxAFQsjQESRniBOBAfQhNrY4C4CuQ6EFgJxAYIaewgCohr0MQkgPgbED8FYl4gvoEqjR3MYYCkJXQwgwHiqsVQTBBcA2IWdEEg0GCAGATCiWhyGMALiM8DMSO6BBSsZoAYJI0uAQM2DJCYgtn4AIjtkBVAgSUQX0IXHAWjAAgA8tMw5+vmET0AAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAYCAYAAABXysXfAAABaUlEQVR4Xu2WvytGURjHv0oWP5KFMsjEaFIkyWAwmU0s/gBWi8Vik8GgDErJYJAiE5OklAGbd6GU/IzFwPd5n3NzzyO3s7iHOp/61H2f7x3u9z33PecFEolEAZ128N9opEN0hT740Xf66T69oh/0yI+r7NJ3aH5D5/z41+iht3SdntNHP/6ZPVqBPnCvH1WZpqt2WCLbCCxTTy/oBLTMlpcqS3TYDkskuMwIXaR1+FqdrvwN5BiaxyK4zAIdc9cz0DJSLqOD7uQ+xyC4zAltctcN9I4+02Y3m4L+ZmISVKaVHprZPHR1ZJWEDejOEhMp82SHlnE6a2Zt9I1eQ/f5Sz8uZBm61YcoKx5KUBk5jOSsschDyeqsOWMjZeTVL0QOo1o7JN3QMuKkyWIgZV7sMM8oPaU1NnBsQsu02yAC8lq+Qs9EjwHoDpZ98xU6mL/B0UfP7LBEWugB9D9Z9qz30GLSIZFIJBJ/j0/iM1TT7caS+AAAAABJRU5ErkJggg==>