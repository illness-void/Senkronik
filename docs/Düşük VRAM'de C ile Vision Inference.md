# **Düşük VRAM Kapasiteli Donanımlarda Gerçek Zamanlı Vision-Inference: Mimari Stratejiler ve C-Native Optimizasyon Raporu**

Bilgisayarlı görü sistemlerinin kısıtlı donanım kaynakları altında, özellikle 2 GB Video Random Access Memory (VRAM) sınırında ve saniyede 60 kare (FPS) hedefiyle çalıştırılması, hem algoritmik tasarım hem de sistem seviyesinde derinlemesine bir optimizasyon gerektirmektedir. AMD Radeon RX 460 gibi Polaris mimarisine dayalı grafik işlem birimleri (GPU), modern tensör çekirdeklerinden yoksun olmaları ve sınırlı bellek bant genişlikleri nedeniyle geleneksel derin öğrenme çerçeveleri için ciddi darboğazlar oluşturmaktadır. Bu rapor, kullanıcı arayüzü (UI) öğelerinin tespiti gibi spesifik bir görev için, Python bağımlılığı olmaksızın doğrudan C projelerine entegre edilebilecek en optimize mimari sınıflarını, çıkarım motorlarını ve bellek yönetimi stratejilerini teknik bir perspektifle analiz etmektedir.

## **I. Mimari Sınıf Analizi: Hafifletilmiş Evrişimli Ağlar ve Distile Edilmiş Transformatörler**

Düşük VRAM kapasitesine sahip bir GPU üzerinde saniyede 60 pikselleri tarayabilen bir sistem kurmak, modelin hem parametre sayısını hem de aktivasyon (activation) haritalarının boyutunu minimize etmeyi zorunlu kılar.1 UI öğeleri, doğal sahnelerdeki nesnelere kıyasla daha keskin kenarlara ve daha basit geometrik formlara sahip olduklarından, bu kısıtlılıklar mimari tasarımda bir avantaja dönüştürülebilir.

### **1\. Evrişimli Mimari Sınıfları ve Operasyonel Verimlilik**

Geleneksel evrişimli sinir ağları (CNN), her bir katmanda yüksek hesaplama maliyeti ve bellek trafiği yaratan standart evrişim işlemlerini kullanmaktadır. 2 GB VRAM sınırında, bu maliyeti düşürmek için "Derinlemesine Ayrılabilir Evrişim" (Depthwise Separable Convolution) mimari sınıfı en rasyonel seçenektir.3 Bu sınıf, standart evrişimi iki ayrı aşamaya böler:

1. **Derinlemesine Evrişim (Depthwise Convolution):** Her bir girdi kanalı için ayrı bir uzamsal filtre uygulanır.  
2. **Noktasal Evrişim (Pointwise Convolution):** Kanallar arasındaki bilgiyi birleştirmek için ![][image1] filtreler kullanılır.

Matematiksel olarak, standart bir evrişimin maliyeti ![][image2] iken, bu ayrılabilir yapı maliyeti ![][image3] seviyesine indirir. Burada ![][image4] çekirdek boyutu, ![][image5] girdi kanalı, ![][image6] çıktı kanalı ve ![][image7] özellik haritası boyutudur. Bu yaklaşım, parametre sayısında ve hesaplama yükünde yaklaşık 8 ile 9 katlık bir azalma sağlamaktadır.5

UI tespiti için bu sınıftaki en gelişmiş yapı "Evrensel Ters Çevrilmiş Darboğaz" (Universal Inverted Bottleneck \- UIB) bloklarıdır. Bu mimari, düşük boyutlu bir özellik haritasını önce genişletip (expansion) ardından tekrar daraltarak (compression) bellek verimliliğini maksimize eder.7 Özellikle AMD RX 460 gibi bellek bant genişliği sınırlı kartlarda, bu mimari sınıfı önbellek (cache) verimliliğini artırarak gecikmeyi (latency) düşürmektedir.9

### **2\. Lineer Dikkat Mekanizmalı Vision Transformerlar (ViTs)**

Vision Transformer mimarileri, küresel bağlamı anlama yetenekleri nedeniyle popülerleşmiş olsa da, standart öz-dikkat (self-attention) mekanizmasının yarattığı ![][image8] bellek karmaşıklığı 2 GB VRAM üzerinde büyük bir engeldir.10 Ancak, "Sandviç Yerleşimi" (Sandwich Layout) kullanan ve lineer dikkat (linear attention) mekanizmalarına sahip distile edilmiş ViT modelleri, 100ms altı reaksiyon süresi için güçlü bir adaydır.12

Bu modeller, ağır "Çok Başlı Öz-Dikkat" (MHSA) katmanlarını beslemek yerine, araya daha verimli "İleri Beslemeli Ağ" (FFN) katmanları yerleştirerek bellek trafiğini optimize eder.12 UI öğeleri arasındaki hiyerarşik ilişkileri (örneğin bir pencere içindeki butonlar) yakalamak için ViT'lerin sunduğu küresel alıcı alan (global receptive field) avantajı, klasik CNN'lerin yerel odaklı yapısına göre daha tutarlı sonuçlar üretebilir.14

### **3\. Çapasız (Anchor-Free) Tespit Başlıkları**

UI öğeleri genellikle eksene paralel dikdörtgenlerden (axis-aligned rectangles) oluştuğu için, geleneksel "çapa kutusu" (anchor-box) tabanlı yöntemlerin getirdiği karmaşık hesaplamalardan kaçınılmalıdır.16 "Çapasız" mimariler, nesneleri bir merkez noktası veya köşe noktası olarak tahmin eder. Bu yaklaşım, NMS (Non-Maximum Suppression) gibi çıkarım sonrası adımların maliyetini düşürerek 60 FPS hedefine ulaşmayı kolaylaştırır.18

## **II. Bellek Ayak İzi ve Çıkarım (Inference) Süreleri Üzerine Kantitatif Analiz**

2 GB VRAM kapasiteli bir AMD RX 460 üzerinde, işletim sistemi ve ekran kompozitörünün tükettiği alan düşüldüğünde, model ve çalışma zamanı (runtime) için yaklaşık 1.2 GB \- 1.5 GB net alan kalmaktadır.20 Bu kısıt altında, modelin parametre sayısı ile reaksiyon hızı arasındaki denge noktası hassas bir şekilde belirlenmelidir.

### **1\. Parametre Sayısı ve Hassasiyet İlişkisi**

Aşağıdaki tablo, 2 GB VRAM sınırında çalışan farklı mimari sınıflarının teorik bellek ayak izlerini ve AMD RX 460 üzerindeki tahmini performanslarını göstermektedir.

| Mimari Sınıfı | Parametre (M) | Hassasiyet | VRAM (MB) | GFLOPS | Tahmini Gecikme (ms) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Ultra-Hafif CNN | 0.5 \- 1.5 | FP16 | 15 \- 30 | 0.1 \- 0.2 | 6 \- 12 ms |
| Optimize CNN (UIB) | 2.5 \- 4.0 | FP16 | 40 \- 80 | 0.3 \- 0.6 | 14 \- 22 ms |
| Hibrit ViT (Lineer) | 2.0 \- 5.0 | FP16 | 60 \- 120 | 0.2 \- 0.5 | 18 \- 28 ms |
| Yoğun CNN (ResNet-like) | 10.0+ | FP32 | 300+ | 2.0+ | 80+ ms |

Tablo 1: 1080p ekran tarama senaryosunda mimari performans kıyaslaması.6

Burada dikkati çeken en önemli veri, "reaksiyon hızı" için üst sınırın 100ms olduğu senaryolarda, modelin 10-12 Milyon parametreye kadar ölçeklenebileceğidir.20 Ancak saniyede 60 kare (16.6ms periyod) hedefi için modelin 2.5M \- 3.5M parametre bandında tutulması teknik bir zorunluluktur.

### **2\. AMD Polaris Mimarisi ve Hassasiyet Seçimi**

AMD RX 460 (Polaris 11), modern GPU'lardaki gibi özel INT8 kuantizasyon birimlerine sahip değildir. Bu nedenle, modelleri INT8 formatına sıkıştırmak CPU tarafında hız kazandırsa da, GPU üzerinde "simülasyon" modunda çalışarak performansı düşürebilir.23 Polaris mimarisi için en rasyonel veri tipi FP16 (Half Precision)'dır. FP16 kullanımı:

* VRAM tüketimini FP32'ye göre %50 azaltır.25  
* Polaris'in 16-bit depolama ve hesaplama yeteneklerini kullanarak bellek bant genişliği darboğazını hafifletir.26  
* Hassasiyet kaybını (accuracy loss) %1'in altında tutarak doğruluğu korur.23

## **III. C-Native Çıkarım Motorları ve Entegrasyon Stratejileri**

Python bağımlılığı olmaksızın doğrudan C veya C++ projelerine bağlanabilen, minimal kütüphane ayak izine sahip motorlar, sistemin toplam reaksiyon süresini belirleyen kritik bileşenlerdir.

### **1\. ncnn (Vulkan Optimize Edilmiş Çıkarım Motoru)**

Tencent tarafından geliştirilen ncnn, özellikle düşük seviyeli GPU hızlandırması gerektiren senaryolar için tasarlanmıştır. AMD GPU'larda Vulkan API üzerinden çalışan ncnn, kısıtlı VRAM ortamlarında en verimli motordur.29

* **Vulkan Hızlandırması:** AMD'nin Linux tarafındaki RADV/ACO sürücüleri ile mükemmel bir uyum sergiler ve GPU kaynaklarını CUDA'dan daha verimli kullanabilir.31  
* **Bellek Yönetimi:** "Packing layout" ve "Winograd convolution" optimizasyonları ile bellek trafiğini minimize eder.33  
* **C API Desteği:** Saf C projeleri için C++ wrapper'ları üzerinden hızlı entegrasyon sunar. Üçüncü taraf bağımlılığı yoktur, statik olarak bağlanabilir.34

### **2\. MNN (Mobile Neural Network)**

Alibaba tarafından geliştirilen MNN, heterojen hesaplama yetenekleriyle öne çıkar. OpenCL ve Vulkan backend'leri AMD Polaris kartlarında stabil bir performans sunar.35

* **Dinamik Planlama:** Çıkarım öncesinde (pre-inference) bellek planlaması yaparak VRAM parçalanmasını engeller.36  
* **Hız Faktörü:** Özellikle derinlemesine ayrılabilir evrişimlerde kernel seviyesinde optimizasyonlar sunarak ncnn ile rekabet eder.37

### **3\. Motor Performans Kıyaslaması (AMD RX 460 Üzerinde)**

| Motor | Birincil API | GPU Backend | Ayak İzi (Library) | Başlatma Süresi |
| :---- | :---- | :---- | :---- | :---- |
| **ncnn** | C++ (C Wrap) | Vulkan | \~15 MB | \< 100 ms |
| **MNN** | C++ | Vulkan / OpenCL | \~20 MB | \~150 ms |
| **ONNX Runtime** | C | DirectML / ROCm | 100+ MB | 500+ ms |
| **LibTorch** | C++ | ROCm (Sınırlı) | 500+ MB | 2+ saniye |

Tablo 2: C-Native motorların RX 460 ortamındaki karşılaştırmalı analizi.29

AMD RX 460 için TensorRT desteği yoktur (NVIDIA özelidir). ONNX Runtime, DirectML üzerinden çalışsa da kütüphane boyutu ve başlatma süresi "minimal" hedefler için hantaldır. Bu nedenle, ncnn veya MNN en rasyonel seçeneklerdir.

## **IV. UI Elementleri İçin Model Daraltma ve Sıkıştırma Rasyonelliği**

UI öğelerinin tespiti için eğitilen bir modelde "budama" (pruning) ve "kuantizasyon" işlemlerinin C tarafındaki getirisi, kullanılan çıkarım motorunun bu optimizasyonları nasıl işlediğine bağlıdır.

### **1\. Yapısal Budama (Structural Pruning)**

UI öğeleri gibi sınırlı bir nesne seti (butonlar, iframe alanları) için modelin kanal sayısını azaltmak en etkili yöntemdir. Ağırlık bazlı seyrek budama (unstructured pruning) C tarafında özel hızlandırıcılar gerektirirken, kanal bazlı yapısal budama doğrudan hesaplama yükünü düşürür.22

Örneğin, modelin her bir katmanındaki filtre sayısının %25 oranında azaltılması, VRAM kullanımını yaklaşık %40, hesaplama süresini ise %45 oranında düşürebilir. UI öğelerinin ayırt edici özellikleri (dik açılar, yüksek kontrastlı kenarlar) düşük kanal sayılarında bile yüksek doğrulukla yakalanabilmektedir.

### **2\. Kuantizasyonun C Tarafındaki Etkisi**

C projelerinde kuantizasyonun en büyük getirisi bellek bant genişliği tasarrufudur. Ancak daha önce belirtildiği gibi, Polaris mimarisinde INT8 yerine FP16 kuantizasyonu tercih edilmelidir. C motorları (ncnn/MNN), FP16 ağırlıklarını doğrudan GPU doku (texture) veya tampon (buffer) birimlerine yükleyerek bellek transfer süresini minimize eder.41

## **V. Sıfır Kopyalama (Zero-Copy) ve DMA-BUF Stratejileri**

Saniyede 60 kare hızında 1080p bir görüntünün (yaklaşık 8.3 MB) her karede CPU üzerinden GPU'ya kopyalanması, veriyolu (PCIe) üzerinde ciddi bir yük oluşturur ve CPU döngülerini tüketir.9 Bu darboğazı aşmak için piksellerin doğrudan GPU belleğinde işlenmesi şarttır.

### **1\. DMA-BUF Entegrasyon Mekanizması**

Linux çekirdeğinin sunduğu DMA-BUF altyapısı, bir dosya tanımlayıcı (fd) aracılığıyla bellek alanlarının paylaşılmasını sağlar. C projesinde Vulkan API'sı kullanılarak bu piksellere şu adımlarla erişilebilir:

1. **Harici Bellek Uzantısı:** VK\_KHR\_external\_memory\_fd uzantısı etkinleştirilmelidir.44  
2. **Bellek İçe Aktarma:** VkImportMemoryFdInfoKHR yapısı ile DMA-BUF dosya tanımlayıcısı Vulkan'ın VkDeviceMemory nesnesine bağlanır.41  
3. **Doğrudan İşleme:** Görüntü verisi CPU belleğine kopyalanmadan, çıkarım motoru tarafından doğrudan girdi tensörü (input tensor) olarak kullanılır.

Bu yöntem, CPU kullanımını minimize ederek Ryzen 5 3600 işlemcinin diğer mantıksal görevlere odaklanmasını sağlar.45

### **2\. Senkronizasyon ve Çerçeve Yönetimi**

60 FPS hedefinde senkronizasyon hataları "tearing" veya "stuttering" etkilerine yol açabilir. VkFence ve VkSemaphore yapıları, piksellerin yazılma işlemi bittiğinde çıkarım motorunun tetiklenmesini garanti altına alır.44 C tarafında bu akışın manuel yönetimi, Python'daki gibi "yüksek seviyeli kütüphane gecikmelerinden" (overhead) kurtulmayı sağlar.

## **VI. Reaksiyon Hızı ve Parametre Sayısı: İnce Çizginin Tanımı**

2 GB VRAM ve 100ms altı reaksiyon süresi hedefinde, "parametre sayısı" ile "hız" arasındaki ilişki doğrusal değildir. Bellek bant genişliği ve GPU kullanım oranı (GPU Utilization) bu denklemin ana belirleyicileridir.1

### **1\. GPU Kullanım Verimliliği**

Modelin tüm katmanlarının VRAM'e sığması (fitting) yeterli değildir; aynı zamanda bu katmanların GPU'yu meşgul tutacak kadar yoğun olması gerekir. Küçük modellerde ( \< 1M parametre), GPU çekirdekleri verinin gelmesini beklerken boşta kalabilir. Bu duruma "bellek kısıtlı" (memory-bound) rejim denir.9

UI tespiti için ideal denge noktası:

* **Model Boyutu:** 2.5 \- 3.5 Milyon Parametre.  
* **Veri Tipi:** FP16.  
* **Girdi Çözünürlüğü:** 640x640 (1080p görüntüden ölçeklendirilmiş).  
* **Tahmini Çıkarım Süresi:** 15 \- 20 ms.

Bu konfigürasyon, sistemin toplam 100ms'lik bütçesinin sadece %20'sini çıkarıma harcamasını sağlayarak geri kalan süreyi görüntü yakalama, post-processing ve eyleme geçme süreçlerine bırakır.22

### **2\. Kısıtlı Set İçin Model Özelleştirme**

Sadece butonlar ve iframe alanlarını tanımak için eğitilen bir modelin, COCO gibi 80 sınıflı genel bir modelden çok daha sığ (shallower) olması rasyoneldir. Modelin derinliğini (depth) azaltıp genişliğini (width) korumak, Polaris mimarisindeki paralel işlem birimlerini daha iyi besler.49

## **VII. Teknik Sonuç ve Uygulama Tavsiyeleri**

AMD Ryzen 5 3600 ve 2 GB VRAM'li RX 460 üzerinde gerçekleştirilecek C tabanlı bir UI tespit projesi için sistem mimarisi aşağıdaki prensipler üzerine inşa edilmelidir.

### **1\. Donanımsal Karar Matrisi**

| Bileşen | Tavsiye Edilen Seçim | Teknik Gerekçe |
| :---- | :---- | :---- |
| **Mimari Sınıfı** | UIB tabanlı Inverted Bottleneck CNN | Polaris bellek bant genişliği ve önbellek yapısıyla uyum.7 |
| **Tespit Yöntemi** | Çapasız (Anchor-Free) | Post-processing yükünü minimize eder.17 |
| **Çıkarım Motoru** | **ncnn** (Statik Link) | Vulkan backend verimliliği ve düşük kütüphane boyutu.29 |
| **Veri Tipi** | FP16 (Half Precision) | Polaris'in doğal hızlandırma sağladığı tek düşük hassasiyet.26 |
| **İletişim** | DMA-BUF (Vulkan External) | PCIe darboğazını aşmak için sıfır-kopyalama zorunluluğu.41 |

### **2\. Uygulama İçin Kritik Eşikler**

* **VRAM Üst Sınırı:** Model ağırlıkları 100 MB'ı, aktivasyon tamponları 200 MB'ı geçmemelidir. Bu, ekran kartının diğer grafik görevleri için yeterli pay bırakır.20  
* **İşlem Sıralaması:** C tarafında vkCmdCopyBufferToImage veya benzeri komutlar yerine, çıkarım motorunun doğrudan bellek adreslerine erişmesi sağlanmalıdır.  
* **Sınıf Daraltma:** Model sadece "buton" ve "iframe" sınıfları için optimize edilmeli; bu, çıktı katmanındaki (head) hesaplama yükünü radikal şekilde düşürecektir.6

Sonuç olarak, Python'un getirdiği yüksek seviyeli soyutlama yükünden kurtulup, Vulkan tabanlı ncnn gibi bir motoru C projesine doğrudan entegre etmek, RX 460 gibi kısıtlı bir donanımda bile saniyede 60 kare pikselleri tarayabilen bir sistemi mümkün kılmaktadır. Mimari olarak "Evrensel Ters Çevrilmiş Darboğaz" yapılarının çapasız tespit başlıklarıyla birleşimi, hem düşük bellek kullanımı hem de 100ms altı reaksiyon süresi için en optimize yolu sunmaktadır.

#### **Works cited**

1. Parameter Count Is the Worst Way to Pick a Model on 8GB VRAM \- DEV Community, accessed May 11, 2026, [https://dev.to/plasmon\_imp/parameter-count-is-the-worst-way-to-pick-a-model-on-8gb-vram-2n18](https://dev.to/plasmon_imp/parameter-count-is-the-worst-way-to-pick-a-model-on-8gb-vram-2n18)  
2. GPU Memory Essentials for AI Performance | NVIDIA Technical Blog, accessed May 11, 2026, [https://developer.nvidia.com/blog/gpu-memory-essentials-for-ai-performance/](https://developer.nvidia.com/blog/gpu-memory-essentials-for-ai-performance/)  
3. Lightweight Deep Learning Models for Face Mask Detection in Real-Time Edge Environments: A Review and Future Research Directions \- MDPI, accessed May 11, 2026, [https://www.mdpi.com/2504-4990/8/4/102](https://www.mdpi.com/2504-4990/8/4/102)  
4. Clinical validation of lightweight CNN architectures for reliable multi-class classification of lung cancer using histopathological imaging techniques \- PMC, accessed May 11, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12910041/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12910041/)  
5. Lightweight Convolutional Neural Networks \- Emergent Mind, accessed May 11, 2026, [https://www.emergentmind.com/topics/lightweight-convolutional-neural-network-cnn](https://www.emergentmind.com/topics/lightweight-convolutional-neural-network-cnn)  
6. Designing Lightweight CNN for Images: Architectural Components and Techniques \- River Publishers, accessed May 11, 2026, [https://www.riverpublishers.com/downloadchapter.php?file=RP\_9788770041010C5.pdf](https://www.riverpublishers.com/downloadchapter.php?file=RP_9788770041010C5.pdf)  
7. MobileNetV4 \- Universal Models for the Mobile Ecosystem \- arXiv, accessed May 11, 2026, [https://arxiv.org/html/2404.10518v1](https://arxiv.org/html/2404.10518v1)  
8. MobileNetV4: Universal Models for the Mobile Ecosystem \- ECVA | European Computer Vision Association, accessed May 11, 2026, [https://www.ecva.net/papers/eccv\_2024/papers\_ECCV/papers/05647.pdf](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05647.pdf)  
9. VRAM bandwidth and its big role in optimization \- Blog \- Procedural Pixels, accessed May 11, 2026, [https://www.proceduralpixels.com/blog/vram-bandwidth-and-its-big-role-in-optimization](https://www.proceduralpixels.com/blog/vram-bandwidth-and-its-big-role-in-optimization)  
10. A Study on Inference Latency for Vision Transformers on Mobile Devices \- arXiv, accessed May 11, 2026, [https://arxiv.org/html/2510.25166v1](https://arxiv.org/html/2510.25166v1)  
11. Lightweight Vision Transformers for Low Energy Edge Inference \- The Laboratory for Computer Architecture, accessed May 11, 2026, [https://lca.ece.utexas.edu/pubs/mlarchsys.pdf](https://lca.ece.utexas.edu/pubs/mlarchsys.pdf)  
12. arXiv:2305.07027v1 \[cs.CV\] 11 May 2023 \- Microsoft, accessed May 11, 2026, [https://www.microsoft.com/en-us/research/wp-content/uploads/2023/06/EfficientViT.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/06/EfficientViT.pdf)  
13. \[2305.07027\] EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention \- arXiv, accessed May 11, 2026, [https://arxiv.org/abs/2305.07027](https://arxiv.org/abs/2305.07027)  
14. Comparing Vision Transformers and Convolutional Neural Networks for Image Classification: A Literature Review \- MDPI, accessed May 11, 2026, [https://www.mdpi.com/2076-3417/13/9/5521](https://www.mdpi.com/2076-3417/13/9/5521)  
15. CNNs vs. Vision Transformers: Which Model Should You Use? \- Roboflow Blog, accessed May 11, 2026, [https://blog.roboflow.com/vision-transformer-vs-cnn-for-detection/](https://blog.roboflow.com/vision-transformer-vs-cnn-for-detection/)  
16. Performance Trade-off of Anchor-Based and Anchor-Free Approaches of Faster R-CNN for Face Detection \- ResearchGate, accessed May 11, 2026, [https://www.researchgate.net/publication/400608869\_Performance\_Trade-off\_of\_Anchor-Based\_and\_Anchor-Free\_Approaches\_of\_Faster\_R-CNN\_for\_Face\_Detection](https://www.researchgate.net/publication/400608869_Performance_Trade-off_of_Anchor-Based_and_Anchor-Free_Approaches_of_Faster_R-CNN_for_Face_Detection)  
17. Anchor-Based vs Anchor-Free Object Detection \- Abhik Sarkar, accessed May 11, 2026, [https://www.abhik.ai/concepts/computer-vision/anchor-based-vs-anchor-free](https://www.abhik.ai/concepts/computer-vision/anchor-based-vs-anchor-free)  
18. Anchor-Free Detection Methods \- Emergent Mind, accessed May 11, 2026, [https://www.emergentmind.com/topics/anchor-free-detection](https://www.emergentmind.com/topics/anchor-free-detection)  
19. YOLO11 Anchor-Free Detection: Benefits \- Ultralytics, accessed May 11, 2026, [https://www.ultralytics.com/blog/benefits-ultralytics-yolo11-being-anchor-free-detector](https://www.ultralytics.com/blog/benefits-ultralytics-yolo11-being-anchor-free-detector)  
20. Florence VLM: The Lightweight Zero-Shot Object Detection Model Ready for Production, accessed May 11, 2026, [https://medium.com/@priyadarshinichavan/florence-vlm-the-lightweight-zero-shot-object-detection-model-ready-for-production-b0938b02b050](https://medium.com/@priyadarshinichavan/florence-vlm-the-lightweight-zero-shot-object-detection-model-ready-for-production-b0938b02b050)  
21. Best GPU Hardware for AI Inference: Full Comparison \- GMI Cloud, accessed May 11, 2026, [https://www.gmicloud.ai/en/blog/best-gpu-hardware-for-ai-inference-full-comparison](https://www.gmicloud.ai/en/blog/best-gpu-hardware-for-ai-inference-full-comparison)  
22. Lightweight Transformer Architectures for Edge Devices in Real-Time Applications \- arXiv, accessed May 11, 2026, [https://arxiv.org/html/2601.03290v1](https://arxiv.org/html/2601.03290v1)  
23. Quantization Tradeoffs: 4-bit vs 8-bit vs FP8 Data \- Digital Applied, accessed May 11, 2026, [https://www.digitalapplied.com/blog/quantization-tradeoffs-4bit-8bit-fp8-performance-data](https://www.digitalapplied.com/blog/quantization-tradeoffs-4bit-8bit-fp8-performance-data)  
24. FP16 vs INT8 vs INT4: When to Use Each for LLM Inference \- GIGAGPU, accessed May 11, 2026, [https://gigagpu.com/fp16-vs-int8-vs-int4-llm-inference/](https://gigagpu.com/fp16-vs-int8-vs-int4-llm-inference/)  
25. FP4 vs FP8 vs FP16 LLM Inference: Quality and Speed Tradeoffs \- iFactory AI, accessed May 11, 2026, [https://ifactoryapp.com/sap-integration/on-prem-ai/fp4-vs-fp8-vs-fp16-llm-inference](https://ifactoryapp.com/sap-integration/on-prem-ai/fp4-vs-fp8-vs-fp16-llm-inference)  
26. AMD Radeon RX 580 Specs \- GPU Database \- TechPowerUp, accessed May 11, 2026, [https://www.techpowerup.com/gpu-specs/radeon-rx-580.c2938](https://www.techpowerup.com/gpu-specs/radeon-rx-580.c2938)  
27. AMD RX 480 Slide Deck Leaked \- Async Compute Polaris 10 Detailed \- Wccftech, accessed May 11, 2026, [https://wccftech.com/amd-rx-480-polaris-10-full-slide-deck-leak/](https://wccftech.com/amd-rx-480-polaris-10-full-slide-deck-leak/)  
28. LLM Quantization: BF16 vs FP8 vs INT4 \- AIMultiple, accessed May 11, 2026, [https://aimultiple.com/llm-quantization](https://aimultiple.com/llm-quantization)  
29. Ultralytics YOLO NCNN Export \- Ultralytics YOLO Docs, accessed May 11, 2026, [https://docs.ultralytics.com/integrations/ncnn/](https://docs.ultralytics.com/integrations/ncnn/)  
30. NVIDIA GeForce vs. AMD Radeon Vulkan Neural Network Performance With NCNN, accessed May 11, 2026, [https://www.phoronix.com/review/realsr-ncnn-vulkan](https://www.phoronix.com/review/realsr-ncnn-vulkan)  
31. More Vulkan NCNN Inference Benchmarks On AMD Radeon vs. NVIDIA GeForce Under Linux \- Phoronix, accessed May 11, 2026, [https://www.phoronix.com/news/NCNN-Vulkan-More-NVIDIA-AMD](https://www.phoronix.com/news/NCNN-Vulkan-More-NVIDIA-AMD)  
32. Quantization : Why is ncnn faster than other frameworks? · Issue \#3267 \- GitHub, accessed May 11, 2026, [https://github.com/Tencent/ncnn/issues/3267](https://github.com/Tencent/ncnn/issues/3267)  
33. ncnn uses much more memory than other inference frameworks. · Issue \#2750 \- GitHub, accessed May 11, 2026, [https://github.com/Tencent/ncnn/issues/2750](https://github.com/Tencent/ncnn/issues/2750)  
34. PyTorch vs ONNX vs NCNN \- by Nadira Povey \- Medium, accessed May 11, 2026, [https://medium.com/@nadirapovey/pytorch-vs-onnx-vs-ncnn-ee50115b6263](https://medium.com/@nadirapovey/pytorch-vs-onnx-vs-ncnn-ee50115b6263)  
35. OpenCL Backend: Zero-copy input using external cl\_mem (DMA buffer) via clMemoryImportARM on Mali · Issue \#4065 · alibaba/MNN \- GitHub, accessed May 11, 2026, [https://github.com/alibaba/MNN/issues/4065](https://github.com/alibaba/MNN/issues/4065)  
36. MNN: A Universal and Efficient Inference Engine \- MLSys Proceedings, accessed May 11, 2026, [https://proceedings.mlsys.org/paper\_files/paper/2020/file/bc19061f88f16e9ed4a18f0bbd47048a-Paper.pdf](https://proceedings.mlsys.org/paper_files/paper/2020/file/bc19061f88f16e9ed4a18f0bbd47048a-Paper.pdf)  
37. MNN speed is awesome : r/LocalLLaMA \- Reddit, accessed May 11, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1nv5x9f/mnn\_speed\_is\_awesome/](https://www.reddit.com/r/LocalLLaMA/comments/1nv5x9f/mnn_speed_is_awesome/)  
38. AMD GPU Acceleration Technologies Explained: ROCm, HIP ..., accessed May 11, 2026, [https://gist.github.com/danielrosehill/8793e2028ef4bd08c6ca955a38b40e5b](https://gist.github.com/danielrosehill/8793e2028ef4bd08c6ca955a38b40e5b)  
39. 11 Best OpenVINO Alternatives for Edge AI and Fast Inference \- Sider AI, accessed May 11, 2026, [https://sider.ai/blog/ai-tools/best-openvino-alternatives-for-edge-ai-and-fast-inference](https://sider.ai/blog/ai-tools/best-openvino-alternatives-for-edge-ai-and-fast-inference)  
40. vinjn/awesome-vulkan \- GitHub, accessed May 11, 2026, [https://github.com/vinjn/awesome-vulkan](https://github.com/vinjn/awesome-vulkan)  
41. DMA buf import into Vulkan \- Stack Overflow, accessed May 11, 2026, [https://stackoverflow.com/questions/78071242/dma-buf-import-into-vulkan](https://stackoverflow.com/questions/78071242/dma-buf-import-into-vulkan)  
42. GitHub \- GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator: Easy to integrate Vulkan memory allocation library, accessed May 11, 2026, [https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)  
43. Zero-copy: Principle and Implementation | by Zhenyuan (Zane) Zhang | Medium, accessed May 11, 2026, [https://medium.com/@kaixin667689/zero-copy-principle-and-implementation-9a5220a62ffd](https://medium.com/@kaixin667689/zero-copy-principle-and-implementation-9a5220a62ffd)  
44. External Memory and Synchronization \- Vulkan Documentation, accessed May 11, 2026, [https://docs.vulkan.org/guide/latest/extensions/external.html](https://docs.vulkan.org/guide/latest/extensions/external.html)  
45. I built a GPU-native image viewer for Linux using raw Vulkan and DMA-BUF zero-copy, accessed May 11, 2026, [https://www.reddit.com/r/Btechtards/comments/1sadoey/i\_built\_a\_gpunative\_image\_viewer\_for\_linux\_using/](https://www.reddit.com/r/Btechtards/comments/1sadoey/i_built_a_gpunative_image_viewer_for_linux_using/)  
46. I found a fundamental incompatibility in wgpu's Vulkan external memory handling, so I rewrote my image viewer in raw Vulkan \- Reddit, accessed May 11, 2026, [https://www.reddit.com/r/vulkan/comments/1sa9s2q/i\_found\_a\_fundamental\_incompatibility\_in\_wgpus/](https://www.reddit.com/r/vulkan/comments/1sa9s2q/i_found_a_fundamental_incompatibility_in_wgpus/)  
47. Any experience with VK\_EXT\_external\_memory\_dma\_buf? : r/vulkan \- Reddit, accessed May 11, 2026, [https://www.reddit.com/r/vulkan/comments/1898bk9/any\_experience\_with\_vk\_ext\_external\_memory\_dma\_buf/](https://www.reddit.com/r/vulkan/comments/1898bk9/any_experience_with_vk_ext_external_memory_dma_buf/)  
48. MobileNetV4 \-- Universal Models for the Mobile Ecosystem | Read Paper on Bytez, accessed May 11, 2026, [https://bytez.com/docs/arxiv/2404.10518/paper](https://bytez.com/docs/arxiv/2404.10518/paper)  
49. Papers Explained 232: MobileNetV4 | by Ritvik Rastogi \- Medium, accessed May 11, 2026, [https://ritvik19.medium.com/papers-explained-232-mobilenetv4-83a526887c30](https://ritvik19.medium.com/papers-explained-232-mobilenetv4-83a526887c30)  
50. Understanding Evaluation parameters for Object Detection Models — Flops, FPS, Latency, Params, Size, Memory, Storage, mAP, AP | by Nikita Malviya | Medium, accessed May 11, 2026, [https://medium.com/@nikitamalviya/evaluation-of-object-detection-models-flops-fps-latency-params-size-memory-storage-map-8dc9c7763cfe](https://medium.com/@nikitamalviya/evaluation-of-object-detection-models-flops-fps-latency-params-size-memory-storage-map-8dc9c7763cfe)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAXCAYAAACS5bYWAAABW0lEQVR4Xu2WvUoDQRSFryikUBGENDZilxSCnSgWoi9gJ4iFRWrfwTKlIFapIqYS/G1UFLWyEAsLOx/Axr8ggo2ewyQ4czO4s5F1m/nga87dgUNyd1iRSMRLDxzTYQ782mEELsBTuKtm/8UgnIU1+OyOfliFd3ADfko+ZSfgI2zAe/jijv28Sz5lbQ4lo7IlWNahRRFO6zCBzMqyzAkc1QMwDG/guB4kkFlZwjf3HA5ZWS88gnNWFkqmZQn/at4kLMyiW3DJeSKcVGX3dBjIjJjCdbjijlLBsq869PGXsn3wGN6K2dduSVV2X4cBsOgOrMBJeCHuDqeBZd906INlD3SYAHd0G1atbErMLdFNYZZt6lBTgB/wTMw3Qijc0U3pPDMPr2C/ypPg3vNH857j9XItpuhXywcxhwas53wswnXpLNpmGa7p0AN3/FLMN0G7w5OYDnxxI5FIpMU3bkZJtJkHzX8AAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAYCAYAAAAGXva8AAABgUlEQVR4Xu2UzSsFYRTGH2VBPks+NhSyVqJIEitRyr9gJWXJQolYsVCyt1GyRJJICGXhs5SFHSUrYYfEczrvvfed485m7tjo/uq3uOc5d9553zkzQJYsf0AT3aXP9Jveut/iAX2jh7TL9cfKCnTRalMvpJv0nXaYLGMe6LUtOiqgOz6yQSY0Qnc5ZwOPfWhPvg2iMgq9YI8NPI6hPfU2iMoO/YA+v3TkQo9XFi02WSRkIRmSPRt4dEIXPDP1SXpPn+g6LXP1VWj/Ba1ytQB90IZxG3gsQnuGbAA9pVnvdyUdpLVe7RcL0Au22cBRQ1+huywwWeLY5caFfjpBc5IdIdzRF+gFLEX0hN5AXxtLK/2kJXSazgTj9MgRyC43TF3utBv6dVpG+ICN0Ss6RdegH5hQ2ukpdGJl0UekPn2iLLZEexN/CGELOoAy0c30i9YFOmJGHodM/YBXu0RwqGKnBXpK5V5tBPr65Hm12Bim59DHMw+d6lK6jdSMNCS7s/xLfgBLx1CXTAc98gAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAABOklEQVR4Xu2TPSwFQRSFT/wVXhAtNSUFFYWSoBKJaESheAWFkqDRCBKJQuu1Co1WopAgNFqlkGipFEQ415l5mZkdOoVkv+TL7sy5M29n9z6g5F/STy/pO/2kD3FcYBKqM+/oQRzHHNILqLglyTyddBWq2UmyLLd0BVrQk2SeWboG1YwlWYE+WqMz0ILRKBVTtJue0VfaGsdFlukcHYA2rcYx2ui0u77R0zjOc0K7aAe06VYcf7+WRjoO5etxXKSZXgXjJ3oUjOehExjb0KbD9fQHRuhuMD6nN+6+QheDzH78hTYFc1k2od7z1Oizu1+ATmK0Q7187Ma/cg0t8GxAR7ROsK7wTLj5pWAuyyC9hz6Cx7ogt9j+OTbfm8zXsc3sCT+gQvs4/hUMQS3T4MZ79NHV+dp9l5WU/BVfKrw/RABl4xEAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAABBElEQVR4XmNgGAUDBqyAeDcQ3wfi/0B8AlUaDHYA8W8GiPwzIG5ElUYFO4H4AQNEsRmqFBgUAfF8dEF0wA3E14E4gQFi0HoUWQiYAsRO6ILowA2IJwExGwPCVerICoDgFANEHi/oBuIAKLuYAWIQyGAYkAfirUh8nOAMEPNB2TxA/AaIPwGxAFQsjQESRniBOBAfQhNrY4C4CuQ6EFgJxAYIaewgCohr0MQkgPgbED8FYl4gvoEqjR3MYYCkJXQwgwHiqsVQTBBcA2IWdEEg0GCAGATCiWhyGMALiM8DMSO6BBSsZoAYJI0uAQM2DJCYgtn4AIjtkBVAgSUQX0IXHAWjAAgA8tMw5+vmET0AAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAYCAYAAADpnJ2CAAABa0lEQVR4Xu2UTytEURjGH7LxNzZ2SD6AUpQksSE2PoClJtlbkC0L2cjGykb5AFJEQigLFkrNYjYTxU7+rJB4Xu+57rlvdxbuXBvdX/2a7nneOe+ce84ZICOjTLroPn2gnzTvnsUj+kyP6aCrT41NaMMWM15Ht+kr7TdZWdzSKzvoaIau9MQGSemErm7JBh6H0JpqGyRhBjrZiA08TqE1HTZIwh59g+5XHFXQVyoNG0z2a6SJHIgDG3gMQJtdmPFlekPvEZ5s8YVOeHURxqCTzdnAYxVaM2UDaINFMzZPh8zYDyvQyXpt4GilT9DV1Zqshr7TYffcRivpJG0PiiwF+gjdJ0s9PaPX0KthkT+DD4R7v+Y+42q/kV8hq9sy4xXQVyL/OhsofZhmaZEu0F1obSx99Bx6MqXhHaKbLo3W6WjwhRLsILy7MmfOy1InuCrj7rmJNoZx+vRA908a/TnT9BJ6f+XVd0fjjP/GFwvmSq/VO83+AAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAYCAYAAACvKj4oAAADK0lEQVR4Xu2XaahNURTHl3keCpmSuUSS+RuPkGT6RHkyFyFkHj5QIimUIZKUzJEpkSES+YIoUUTPF0O+iEKU+P/v2tvdd519yX333Fd6v/r37vmvs9/ZZ5211z5HpJoqpzl0ALoLXYS654ZLSx3oNNTBBirBXmi4+70SegbVzYYz3tTgOFX2QBOtWUkuQVvd707QT2jo76hITegmNCzwUmEKdMaaRaZM9AY7G78r9AZqYvy8NIXWQseg89AN6A60CqoVnOdhFl9Do23A0Ri6Cj0UneAXqFXOGSIboY+i8U+i681yAjpkTccFaLk1Y7Cen0PLRCfuaQk9gK5D9QKfjIdeQDWMb1kHvRK9idhkBogmIfYkponeXH0bcMwSTXK4PhOw1t9CPWzAwcXOya03/mFol/FicK0MgX6IlpRN1GRog/FIX2ibaAK7QB1zwxm6ic7NN6QEs0VPGGUDAZwQJ3fP+OxsC41naSRaAeSo6LXmZMMZdosmIKSFaALZRMqgTZJcg54K0cpLwCf2WXSv+ROcJCf2PvCaOW9C4MUYB213v/uJjnmUDWd4DNU2HnsAz/X6KvE+QK6IJikBy4uD59qAgdnlebcDr4/zbOYtvLkxwTEnw3Ej3DGTfDYbLoiT0HFrMhvvRC+W79F7toieF663gc7rHXgx2DzYTT3suBx3zh2zxBdlwwWxH7pmzYaSffy2PELaiJbxd8m9Gf8E+Tcf7SVyYXBfdGx/0T20V274n9knuqUlYO3zQg1sIIC1zXNWG5+vZX8r0emSHEfKRcfy5p6aWCGwPKNlzs7EC4VrJGS+aJwZsvjGM9YGAtgFB1lTtGK4f3L8QRMrhMvQTmsSlulL6IloKXq4aW6GvkFLJf9GXgEttqaDT5hd1765eLjueIPFeGHmC8o8a3paQzugW6KtmVlnV1oBtQ3Oi8HsHzEe30b4v9jWeQMfJJ4EVgATECa2EDie1xlsA8Vgpugk+blUVXAfZgWmAl/OudVMsoESwgpaYs1iskDiXwClgF/53AnyvYgXBX55sN2PtIGU4XXZPXvaQBpwHz0FtbOBFFkDzbBmNf8LvwAlBaAfTloTAgAAAABJRU5ErkJggg==>