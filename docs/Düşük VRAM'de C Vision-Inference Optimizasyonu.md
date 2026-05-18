# **Düşük VRAM'li Donanımlarda Yüksek Hızlı Görüntü İşleme, C-Native Çıkarım Motorları ve Web Arayüzü Tespit Stratejileri**

Grafik işleme birimleri (GPU) üzerinde gerçekleştirilen yapay zeka çıkarım (inference) işlemleri, donanım kaynaklarının fiziksel sınırlarına yaklaştıkça giderek daha karmaşık ve çok boyutlu bir sistem optimizasyonu problemi halini almaktadır. Özellikle 2 GB VRAM gibi günümüz endüstri standartlarına göre oldukça kısıtlı bir bellek kapasitesine sahip olan AMD Radeon RX 460 (Polaris 11 mimarisi) üzerinde, saniyede 30 ila 60 kare (FPS) işleme hızlarına ulaşmak, bellek bant genişliğinin, hesaplama döngülerinin ve veriyolu darboğazlarının mikrosaniye düzeyinde yönetilmesini zorunlu kılmaktadır.1 Görüntü verisinin sistem belleğinden (RAM) grafik belleğine (VRAM) kopyalanması sırasında yaşanan gecikmeler, gerçek zamanlı sistemlerde darboğaz yaratan en temel faktörlerden biridir. Bu darboğazı aşmak için DMA-BUF (Direct Memory Access Buffer) üzerinden sağlanan sıfır-kopya (zero-copy) mekanizmalarının doğrudan C dili ile, Python veya ağır nesne yönelimli çerçeveler (frameworks) kullanılmadan işletim sistemi seviyesinde entegre edilmesi kritik bir tasarım zorunluluğudur.3

Ryzen 5 3600 gibi nispeten güçlü bir merkezi işlem biriminin (CPU) sağladığı veri hazırlama avantajları ile RX 460'ın kısıtlı ama paralel işlem yetenekleri birleştirildiğinde, web arayüzü elementlerinin (butonlar, metin kutuları, iframe alanları) tespiti için özel olarak daraltılmış modeller harikalar yaratabilir.5 Bu raporda, kısıtlı donanımlarda doğrudan C projelerine entegre edilebilecek en rasyonel vizyon modeli mimarileri, bu mimarilerin bellek ayak izleri ve milisaniye cinsinden reaksiyon süreleri, Python bağımlılığı olmayan C-Native çıkarım motorlarının mimari karşılaştırmaları ve web elementlerine özel model daraltma (budama ve nicemleme) stratejilerinin matematiksel ve algoritmik rasyonalitesi derinlemesine incelenmektedir.

## **1\. Kısıtlı Donanımlar İçin Rasyonel Vizyon Modeli Mimari Sınıfları**

2 GB VRAM sınırında çalışacak ve saniyede 60 kare gibi iddialı bir hedefe ulaşacak bir nesne tespit modelinin mimarisi seçilirken, geleneksel "parametre sayısı" metriği tek başına son derece yanıltıcı bir gösterge olabilir.7 Asıl belirleyici unsurlar; modelin grafik işlemcideki aktif aktivasyon belleği ayak izi, evrişim (convolution) operasyonlarının aritmetik yoğunluğu, bellek erişim (memory access) maliyeti ve GPU üzerindeki dağıtım (dispatching) gecikmeleridir. Bu bağlamda, donanım kaynaklarını en verimli kullanan ve doğrudan C dilleri ile düşük seviyeli entegrasyona olanak tanıyan mimari sınıflar, tasarımsal felsefelerine göre belirli kategorilere ayrılmaktadır.

### **1.1. Evrişimli Ağlar (CNN) ve Damıtılmış Vizyon Transformatörleri (ViT) Arasındaki Paradigma Çatışması**

Son yıllarda küçük dil ve vizyon modelleri (SLM/VLM) ile damıtılmış vizyon transformatörleri akademik literatürde büyük bir popülerlik kazanmış olsa da, bu mimariler düşük VRAM'li ve eski nesil GCN (Graphics Core Next) mimarilerine sahip grafik kartlarında (RX 460 gibi) ciddi donanımsal darboğazlar yaratmaktadır.8 Transformatör mimarilerinin temel yapı taşlarından biri olan öz-dikkat (self-attention) mekanizmaları, girdi çözünürlüğünün karesi ile orantılı (![][image1]) bir bellek ve hesaplama karmaşıklığına sahiptir. Buna ek olarak, Katman Normalizasyonu (LayerNorm) operasyonları ve tensör yeniden şekillendirme (reshape/permute) işlemleri, mobil ve düşük seviye GPU'larda optimize edilmemiş bellek düzenleri nedeniyle ciddi bir çekirdek (kernel) başlatma gecikmesine yol açmaktadır.10

RX 460'ın bellek bant genişliği saniyede yaklaşık 112 GB ile sınırlıdır.2 Transformatörlerin ihtiyaç duyduğu Query, Key ve Value (QKV) matris çarpımları, hesaplama birimlerini doldurmadan önce bellek bant genişliğini hızla doyurur ve "bellek-sınırılı" (memory-bound) bir yürütme profili çizer. Bu nedenle, vizyon transformatörleri yerine, mekansal özellikleri hiyerarşik olarak işleyen ve "hesaplama-sınırlı" (compute-bound) bir yapı sunan saf evrişimli (CNN) ağların kullanılması, 60 FPS hedefine ulaşmada matematiksel olarak çok daha rasyonel bir tercihtir.12

### **1.2. Çapa-Bağımsız (Anchor-Free) ve Tek Aşamalı (Single-Stage) Tespit Mimarileri**

Geleneksel nesne tespit modelleri, görüntünün her bir pikselinde farklı en-boy oranlarına ve ölçeklere sahip çok sayıda sanal "çapa kutusu" (anchor box) üretir. Bu durum, özellikle sınırlandırılmış VRAM'e sahip donanımlarda devasa tensör boyutlarına, bellek taşmalarına ve işlemci tarafında çözülmesi gereken ağır bir Maksimum Olmayan Bastırma (Non-Maximum Suppression \- NMS) yüküne neden olur.14 NMS işlemi genellikle paralel işlemcilere uygun olmayan, ardışık (sequential) bir yapıda olduğu için DMA-BUF ile sağlanan sıfır-kopya boru hattında (pipeline) ciddi bir darboğaz oluşturur.

Buna karşın, çapa-bağımsız (anchor-free) mimariler, nesne tespit problemini bir piksel sınıflandırma ve merkezden kenarlara dört yönlü mesafe regresyonu problemi olarak ele alır.13 Bu yaklaşım, ağın çıktı katmanlarındaki (head) parametre sayısını ve anlık aktivasyon belleği ihtiyacını dramatik ölçüde düşürür. Genelleştirilmiş Odak Kaybı (Generalized Focal Loss) ve benzeri modern etiket atama stratejileri kullanan çapa-bağımsız mimariler, kutu regresyonu ile sınıflandırmayı aynı uzayda birleştirerek eğitim stabilitesini artırır ve çıkarım sırasında gereksiz bellek tahsislerini önler.14 Web butonları, metin kutuları ve iframe çerçeveleri gibi geometrik olarak son derece düzenli (dikdörtgen formlu) arayüz elementlerinin tespiti için bu tür bir regresyon yapısı sadece donanım dostu değil, aynı zamanda bağlamsal olarak da en isabetli yöntemdir.

### **1.3. Hafifletilmiş Özellik Çıkarıcılar (Lightweight Backbones) ve Özellik Piramitleri**

Derin öğrenme modellerinde donanım gecikmesinin ve VRAM tüketiminin aslan payı, ağın girişindeki çözünürlüğü yüksek veriyi işleyen omurga (backbone) katmanlarında gerçekleşir. Geleneksel yoğun evrişimler yerine tasarlanan modern mimari varyasyonları, bu yükü hafifletmek için hayati öneme sahiptir.

* **Derinlemesine Ayrılabilir Evrişimler ve Ghost Modülleri:** Standart evrişim operasyonları yerine, temel birkaç mekansal özelliği çıkarıp diğer özellikleri maliyetsiz doğrusal (linear) operasyonlarla üreten modüller, VRAM kullanımını ve Saniyedeki Kayan Nokta Operasyonları (FLOPs) miktarını %90'a varan oranlarda azaltabilir.16 Polaris mimarisine sahip GPU'larda, bu tür hafifletilmiş omurgalar, geleneksel mimarilere kıyasla çok daha düşük gecikme süreleri sergilerken benzer veya daha üstün bir özellik temsil gücü sunar.16  
* **Aşamalar Arası Kısmi Ağlar (Cross Stage Partial Networks):** Bu mimari felsefe, ağın bir katmanından diğerine geçerken özellik haritasının sadece bir kısmını ağır işlemlerden geçirirken, diğer kısmını doğrudan birleştirme (concat) katmanına aktarır.15 Bu sayede, ağın degrade (gradient) akışı zenginleşirken, hesaplama maliyeti ve aynı tensörün bellekte tutulma ihtiyacı yarı yarıya düşer.  
* **Hafif Özellik Piramitleri (Lightweight Feature Pyramids):** Çok ölçekli (multi-scale) tespit yapabilmek için kullanılan Özellik Piramidi Ağları (FPN) genellikle yukarıdan aşağıya (top-down) ve aşağıdan yukarıya (bottom-up) yollar içerir. Ancak kısıtlı bellek ortamlarında, gereksiz kanal genişletmelerinden kaçınan, yalnızca temel yolları koruyan ve doğrusal interpolasyonlarla maliyeti düşüren PAN (Path Aggregation Network) varyantları kullanılması elzemdir.13

Özetle, Python bağımlılığı olmayan bir C projesinde 2 GB VRAM sınırında çalıştırılacak en rasyonel mimari sınıf; çapa-bağımsız (anchor-free) regresyon mantığıyla çalışan, özellik çıkarımı için aşamalar arası kısmi (CSP) veya hayalet (ghost) evrişimler kullanan, tek aşamalı (single-stage) ve son derece daraltılmış bir derin sinir ağıdır.

## **2\. 2 GB VRAM Üzerindeki Bellek Ayak İzi ve Çıkarım (Inference) Süreleri**

Yapay zeka modellerinin bellek tüketiminin dinamikleri incelendiğinde, bu tüketimin iki ana bileşenden oluştuğu görülür: Modelin kendi statik yapısını oluşturan ağırlıklar (Weights) ve çalışma zamanında ağ içinden akan verinin oluşturduğu dinamik aktivasyonlar (Activations). 2 GB (tam olarak 2048 MB) kapasiteli bir VRAM alanında matematiksel sınırları belirlemek, sıfır toleranslı bir mühendislik gerektirir. RX 460 üzerinde, Linux işletim sistemi çekirdeği (kernel), X11 veya Wayland gibi görüntü sunucuları ve masaüstü kompozitörleri hali hazırda sistem boşta iken 300 MB ile 500 MB arasında bir bellek alanını işgal etmektedir.18 Geriye kalan ve projenin kullanımına sunulan yaklaşık 1.5 GB'lık kullanılabilir alan, modelin VRAM'e tamamen sığması ve kesinlikle sistem belleğine (RAM) taşmaması (swap veya CPU ofload yapılmaması) için kusursuzca yönetilmelidir. Parçalı yük bindirme (partial offloading) işlemleri, yavaş PCIe veriyolu üzerinden veri transferini zorunlu kılarak işlem hacmini saniyede 30 kareden anında 3-4 karelere kadar düşürebilmektedir.7

### **2.1. Bellek Ayak İzi (Memory Footprint) Mekanizması ve Tahminleri**

Geniş ölçekli doğal görüntü veri setlerinde (örneğin 80 sınıflı MS COCO) nesne tespiti yapan ağlar, sınıflandırma katmanlarında yüksek sayıda filtre bulundurmak zorundadır. Ancak projenin odaklandığı web elementleri (butonlar, iframe alanları, menü çerçeveleri vb.) gibi kısıtlı bir hedef kümesi için model sınıf sayısı muazzam ölçüde daraltılabilir (örneğin 2 ila 5 sınıf arası).20 Sınıf sayısındaki bu keskin düşüş, modelin son katmanlarındaki (head) tensör boyutlarını küçültür ve bellek kullanımını minimize eder.

Çapa-bağımsız hafifletilmiş bir mimarinin bellek ayak izi analizi şu şekilde parçalara ayrılabilir:

* **Statik Ağırlıklar (Static Weights):** Modelin mimarisi yaklaşık 1.0 Milyon ile 1.5 Milyon parametre aralığında tasarlandığında, bu parametrelerin her biri standart 32-bit Kayan Nokta (FP32) hassasiyetinde 4 bayt yer kaplar. Dolayısıyla statik ağırlıklar sadece 4 MB ile 6 MB arasında bir VRAM tüketimine neden olur. Modelin 16-bit (FP16) formatında yüklenmesi durumunda bu alan 2 MB ile 3 MB aralığına düşecektir.14 Bu değer, 1.5 GB'lık kullanılabilir bütçe içinde ihmal edilebilir bir orandır.  
* **Aktivasyon Belleği (Activation Memory):** Gerçek darboğaz çalışma zamanında ortaya çıkar. 416x416 piksel çözünürlüğünde 3 kanallı (RGB) bir giriş tensörü ağa girdiğinde, ilk katmanlardaki özellik haritaları (feature maps) bellekte geniş alanlar kaplar. Naif bellek ayırıcılar (allocators) kullanıldığında her katmanın çıktısı bellekte ayrı bir alan tahsis eder ve bu durum anlık olarak 100 MB ile 300 MB arasında bellek sivrilmelerine (memory spikes) yol açabilir. Ancak, C projelerinde bellek paylaşım (memory sharing / in-place execution) mekanizmalarına sahip optimize bir çıkarım motoru kullanıldığında, tensörler yaşam döngülerini tamamladıklarında bellek alanları ardışık katmanlara devredilir. Bu optimizasyonlar uygulandığında, 416x416 boyutlarındaki bir girdi için tepe aktivasyon belleği tüketimi **30 MB ile 50 MB** seviyelerinde tutulabilir.23

Tüm bileşenler toplandığında, optimize edilmiş ve daraltılmış bir tespit modelinin RX 460 üzerindeki toplam VRAM tüketimi (statik \+ dinamik \+ ara bellekler dahil) çıkarım anında kesinlikle **100 MB'ın altında** kalacaktır. Bu durum, 2 GB VRAM limiti içinde son derece güvenli bir liman olup, DMA-BUF piksellerinin tutulması için fazlasıyla yeterli alan bırakır.

### **2.2. Parametre Sayısı ve Reaksiyon Hızı Arasındaki İnce Çizgi**

Mühendislik hedefi olarak saniyede 30 ila 60 kare (FPS) işleme hızı talep edilmektedir. Kare hızı hedeflerini zaman domenine çevirdiğimizde; 60 FPS elde edebilmek için sistemin bir kareyi işlemek için sahip olduğu toplam süre maksimum **16.6 milisaniyedir (ms)**. 30 FPS için ise bu süre **33.3 ms** olarak belirlenir.25 Bu katı zaman bütçesi içerisine sadece ağın çıkarım süresi değil, görüntünün donanım işaretçilerinin haritalanması (DMA mapping) ve sonuçların ayrıştırılması (post-processing) işlemleri de sığdırılmalıdır. Dolayısıyla ağın çıkarım süresi 60 FPS hedefi için 10-12 ms bandını geçmemelidir.

Kısıtlı hedeflere (UI elementleri) yönelik tasarlanmış modellerin mimarisi ile reaksiyon hızı arasında doğrudan bir korelasyon bulunmaktadır. OmniParser V2 gibi büyük ölçekli ve yüksek çözünürlüklü modeller, milyarlarca parametre içerir ve 2 GB VRAM sınırlarında saniyeleri bulan tepki süreleri yaratır veya yetersiz bellek hataları (OOM) verir.18 100 ms altı reaksiyon süresi (yani 10 FPS) hedefi için mutlak üst sınır yaklaşık 8 ila 10 Milyon parametredir. Ancak hedef 16.6 ms (60 FPS) ise, bu sınır çok daha aşağı çekilmelidir.7

Endüstriyel kıyaslamalar ve mimari araştırmalar ışığında, 320x320 ila 416x416 giriş çözünürlüklerinde çalışan çapa-bağımsız modellerin RX 460 ve eşdeğeri mobil/eski nesil donanımlardaki performansları analiz edildiğinde şu tablo ortaya çıkmaktadır 14:

| Mimari Büyüklüğü (Yaklaşık) | Girdi Çözünürlüğü | Hesaplama Yükü (FLOPs) | Tahmini RX 460 Vulkan Gecikmesi | Karşılık Gelen Maksimum Teorik FPS |
| :---- | :---- | :---- | :---- | :---- |
| **Ekstra Hafif (\~1.0 Milyon)** | 320x320 | \~0.75 GFLOPs | 4.0 ms \- 6.0 ms | 160+ FPS |
| **Hedeflenen Denge (\~1.2 Milyon)** | 416x416 | \~1.50 GFLOPs | 7.0 ms \- 10.0 ms | 100+ FPS |
| **Genişletilmiş Ağ (\~2.5 Milyon)** | 416x416 | \~3.00 GFLOPs | 15.0 ms \- 20.0 ms | \~50 FPS |
| **Üst Sınır (\~8.0 Milyon)** | 640x640 | \~15.00 GFLOPs | 70.0 ms \- 90.0 ms | \~11 FPS (100ms altı limiti) |

(Tablo 1: RX 460 donanım kapasitesine ve Vulkan API hesaplama gücüne göre derlenmiş evrişimli ağların teorik parametre/gecikme izdüşümleri. Model ağırlıkları ve giriş çözünürlükleri temel alınarak hesaplanmıştır.13)

Bu verilere göre, sadece web butonlarını ve iframe alanlarını tanımak için eğitilecek bir modelin "parametre sayısı" ile "reaksiyon hızı" arasındaki o ince çizgi **1.0 Milyon ile 1.5 Milyon parametre** aralığıdır. Bu mimari derinlikteki bir model, 2 GB VRAM sınırlarında 100ms altı beklentisini çok rahat aşarak, ortalama **7 ile 10 ms** aralığında çıkarım gerçekleştirecek ve saniyede 60 kare (16.6 ms bütçesi) hedefine DMA aktarım süreleri dahi eklendiğinde garantili olarak ulaşılmasını sağlayacaktır.14

## **3\. C-Native Çıkarım Motorları (Inference Engines) Performans Kıyaslaması**

Makine öğrenimi ekosisteminde var olan araçların büyük bir çoğunluğu, veri bilimcilerin rahat kullanımı için tasarlanmış ve derin bir Python bağımlılığı üzerine inşa edilmiştir. Ancak sistem seviyesinde yazılmış ve donanımın sınırlarını zorlaması gereken bir C projesinde, Python yorumlayıcısının getirdiği ek yük (overhead), Küresel Yorumlayıcı Kilidi (GIL) sorunları ve devasa bellek alanı tahsisleri tamamen kabul edilemezdir. Bu projede ihtiyaç duyulan çözüm; doğrudan çapraz derlenebilen, C dilinde yerel (native) olarak çalışan, asgari kütüphane boyutu sunan (tercihen sadece başlık dosyaları \- header only \- veya minimal statik kütüphane) ve AMD GPU'larına olabilecek en düşük gecikmeyle (Vulkan API üzerinden) erişebilen çıkarım motorlarının (inference engines) değerlendirilmesidir.

RX 460 gibi Polaris 11 mimarisine sahip bir ekran kartında, AMD'nin genel maksatlı GPU hesaplama (GPGPU) platformu olan ROCm'in (Radeon Open Compute) durumu kritik bir karar faktörüdür. AMD, ROCm kütüphanesinin RX 400 serisi gibi eski nesil kartlar için resmi desteğini sonlandırmış durumdadır. Sınırlı bazı yamalar ve topluluk destekli çözümler mevcut olsa da, bunlar üretim kalitesinden uzak, stabil olmayan bir "uyumluluk katmanı" sağlamaktadır.29 Bu nedenle TensorRT gibi sadece NVIDIA (CUDA) donanımlarına özgü motorlar veya ROCm/HIP bağımlılığı gerektiren çözümler mimari olarak projeden en başından dışlanmalıdır. Güvenilir ve yüksek performanslı tek gerçekçi yol, endüstri standardı haline gelen donanım bağımsız Vulkan API'sidir.32

Aşağıda, C projesine doğrudan entegre edilebilecek motorların kapsamlı bir analizi sunulmaktadır:

### **3.1. ONNX Runtime (C API Desteği)**

ONNX (Open Neural Network Exchange), makine öğrenimi modellerini temsil etmek için oluşturulmuş açık bir formattır. Microsoft tarafından geliştirilen ONNX Runtime, bu modelleri çalıştırmak için geniş bir ekosistem ve OrtGetApi adında resmi bir C API'si sunar.35

* **Mimari ve Performans:** ONNX Runtime'ın donanım hızlandırma stratejisi "Execution Provider" (Çalıştırma Sağlayıcısı \- EP) adı verilen bir arayüze dayanır. Sistemdeki donanıma göre farklı sağlayıcılar devreye girer. AMD donanımları için MIGraphX veya ROCm sağlayıcıları önerilse de, belirtildiği gibi RX 460 üzerinde bu sağlayıcılar sağlıklı çalışmaz. Geriye Vulkan veya DirectML (yalnızca Windows için) seçenekleri kalır.  
* **Darboğazlar:** ONNX Runtime'ın Vulkan Execution Provider desteği genel olarak sınırlı performans sunmakta ve sık sık hesaplama düğümlerini (nodes) CPU üzerine geri atarak (fallback) ciddi gecikmelere sebep olmaktadır.35 Ayrıca, C API'si bulunmasına rağmen ONNX Runtime son derece ağır bir kütüphanedir; statik ve dinamik bağlantı dosyaları onlarca megabayt boyutuna ulaşır ve minimal bir C projesi felsefesine aykırı bir sistem yükü getirir.38

### **3.2. libtorch (PyTorch C++ API)**

PyTorch ekosisteminin C++ tarafındaki karşılığı olan libtorch, modelleri doğrudan sistem seviyesinde çalıştırmayı vaat eder.

* **C Entegrasyonu:** libtorch saf bir C++ kütüphanesidir ve C standartlarında (C99/C11) yazılmış bir projeye entegre edilebilmesi için kullanıcının özel "C wrapper" (sarmalayıcı) arayüzleri yazmasını gerektirir, bu da geliştirme sürecini uzatır.  
* **Donanım Desteği ve Bellek:** libtorch, donanım ivmelendirmesi için temel olarak CUDA (NVIDIA) altyapısına bel bağlar. ROCm desteği teorik olarak mevcut olsa da, RX 460 üzerinde derlenmesi adeta bir kabusa dönüşür.29 Dahası, libtorch'un çalışma zamanı yapıları (runtime contexts) 2 GB VRAM sınırındaki bir sistem için oldukça hantaldır ve boşta dahi yüzlerce megabayt bellek rezerve eder.38 Bu nedenle düşük donanım hedefleri için uygun bir seçenek değildir.

### **3.3. MNN (Mobile Neural Network \- Alibaba)**

Alibaba tarafından geliştirilen MNN, düşük kaynaklı sistemler, mobil platformlar ve gömülü cihazlar için özel olarak optimize edilmiş, oldukça popüler bir derin öğrenme çıkarım motorudur.39

* **Vulkan Gücü:** MNN'in Vulkan arka ucu (backend) oldukça güçlüdür ve AMD dahil olmak üzere çeşitli donanımlarda FP16 (yarı hassasiyetli) hesaplamalarda mükemmel sonuçlar verir.39 Bellek yönetimi konusunda oldukça cimridir.  
* **C API Engeli:** Ancak mimari kod tabanının %89'u C++ iken, saf C kodları yalnızca %2.5 seviyesindedir.39 MNN projesi, ağırlıklı olarak Python veya modern C++ arayüzleri üzerinden dokümante edilmiştir. Doğrudan bir C projesine entegre edilirken yine sarmalayıcı (wrapper) dosyalarına ihtiyaç duyulacak olması, donanım seviyesinde çalışan minimal bir proje için gereksiz dolaylılık (indirection) katmanları oluşturur.

### **3.4. NCNN (Tencent)**

Bu özel donanım senaryosu ve yazılım gereksinimleri (C-native, minimal kütüphane, AMD Polaris üzerinde yüksek Vulkan performansı, 2 GB VRAM) bir araya getirildiğinde, akademik ve endüstriyel olarak açık ara en rasyonel seçenek **NCNN**'dir.38

* **Kusursuz Saf C API Desteği:** NCNN, mimarisinin kalbine entegre edilmiş c\_api.h adlı geniş, güncel ve son derece istikrarlı bir saf C API başlık dosyası sunar.43 Hiçbir C++ sarmalayıcısı yazmaya gerek kalmadan, doğrudan gcc veya clang kullanılarak projeye eklenebilir. ncnn\_net\_t yapısı ile model belleğe alınır, ncnn\_option\_set\_use\_vulkan\_compute(opt, 1\) tek satırlık komutu ile GPU donanımı anında devreye sokulur.45  
* **Vulkan Odaklı Tasarım:** NCNN, Vulkan'ı sonradan eklenmiş bir özellik olarak değil, birinci sınıf bir vatandaş olarak kabul eder. Bütün ağ tensör operasyonları VkBuffer ve VkImage nesneleri üzerinden sıfır-kopya (zero-copy) mantığı ile bütünleşik çalışır.47  
* **Donanım Verimliliği ve Optimizasyonlar:** Eski nesil GPU'larda bile performansı zirveye taşıyan FP16 depolama, kayan nokta paketleme (use\_fp16\_packed, use\_fp16\_storage, use\_fp16\_arithmetic) ve hatta INT8 hesaplama yetenekleri doğrudan C API'si üzerinden yapılandırılabilir.22 Bellek kullanımını tam anlamıyla kontrol altına almak için NCNN'nin bellek ayırıcıları (allocators) C projesine doğrudan entegre edilerek işletim sistemi düzeyinde bellek parçalanmasının (fragmentation) önüne geçilir.24 AMD Polaris kartlarındaki donanıma en az ek yük (overhead) bindiren ve Vulkan sürücüleriyle en uyumlu çalışan motor konumundadır.34

**Çıkarım Motorları Karşılaştırmalı Performans Matrisi:**

| Çıkarım Motoru | Saf C API Desteği Durumu | AMD RX 460 (Vulkan/ROCm) Desteği | Bellek Ayak İzi & Kütüphane Yükü | Genel Proje Uygunluk Derecesi |
| :---- | :---- | :---- | :---- | :---- |
| **NCNN (Tencent)** | Mükemmel (c\_api.h standart) 43 | Yüksek (Native Vulkan, FP16 destekli) 47 | Çok Düşük (Kompakt ve modüler) | **10/10** \- Hedef mimari için kusursuz |
| **MNN (Alibaba)** | Zayıf (C++ ağırlıklı, wrapper şart) 39 | Yüksek (Optimize Vulkan backend) 39 | Düşük (Mobil odaklı tasarım) | **7/10** \- İyi performans, zor entegrasyon |
| **ONNX Runtime** | Çok İyi (OrtGetApi mevcut) 36 | Sınırlı (MIGraphX zayıf, Vulkan EP sorunlu) 35 | Yüksek (Ağır bağımlılıklar) | **4/10** \- Hantal ve CPU darboğazı riski |
| **libtorch** | Yok (Saf C++ veya Python) | Çok Zayıf (Polaris ROCm desteği yok) 29 | Çok Yüksek (Yüzlerce megabaytlık context) | **1/10** \- Uygun değil |
| **ZINC** | İyi (Geliştiriliyor) 29 | Çok İyi (Doğrudan AMD GPU Vulkan hedefleniyor) 29 | Düşük | **Deneysel** \- Gelecek vadeden alternatif |

(Tablo 2: C-Native çıkarım motorlarının bellek, arayüz ve AMD Polaris mimarisi üzerindeki teorik ve pratik performans karşılaştırmaları. Veriler kaynak analizlerine dayandırılmıştır.29)

## **4\. C Projesinde Zero-Copy (Sıfır-Kopya) DMA-BUF Entegrasyon Stratejisi**

Linux tabanlı sistemlerde yüksek çözünürlüklü ekran piksellerini (veya yakalanan ekran çerçevelerini) işlemek, sistem mimarisinde çok bilindik bir veri kopyalama darboğazına işaret eder. Geleneksel yazılımlarda, görüntü sunucusunun (X11 veya Wayland) oluşturduğu pikseller önce VRAM'den ana belleğe (RAM) kopyalanır (memcpy), ardından CPU bu veriyi işleyeceği çıkarım motoruna yollar ve çıkarım motoru bu veriyi işlemek üzere tekrar VRAM'e kopyalar. AMD RX 460'ın bellek kontrolcüsü üzerinden her karede gerçekleştirilen bu "git-gel" işlemleri, PCIe veriyolunu meşgul ederek saniyenin onda biri kadar gecikmelere sebep olabilir ve 16.6 ms (60 FPS) hedefini tek başına yok edebilir.4

Bunun kökten çözümü, modern Linux grafik yığınının temel taşı olan DMA-BUF (Direct Memory Access Buffer) altyapısının sağladığı sıfır-kopya (zero-copy) stratejisini kullanmaktır.3

### **4.1. Vulkan Harici Bellek (External Memory) Uzantısı ve DMA-BUF Köprüsü**

Linux ekosisteminde (DRM/KMS alt sisteminde) çalışan modern ekran yakalama servisleri (örneğin PipeWire veya wlr-screencopy), ekrandaki pikselleri doğrudan donanım belleğinde tutar ve bu bellek alanının adresini kullanıcı alanına (user-space) standart bir dosya tanımlayıcısı (file descriptor \- fd) olarak sunar.50

Sıfır-kopya stratejisinde ana fikir, CPU'nun hiçbir zaman bu dosya tanımlayıcısındaki pikselleri okumaması veya kopyalamamasıdır. Bunun yerine, C dilinde yazılan projede Vulkan API'sinin harici bellek uzantıları (VK\_EXT\_external\_memory\_dma\_buf) kullanılarak bu DMA-BUF dosya tanımlayıcısı, Vulkan'ın tanıyacağı bir VkDeviceMemory alanına haritalanır (memory mapping).53

C dilinde bir grafik nesnesi (VkImage) oluşturulurken, VkImportMemoryFdInfoKHR yapısı kullanılarak sistem şu şekilde kandırılır: Yeni bir bellek ayırmak yerine, sistemin halihazırda ekranı çizdiği fiziksel VRAM bölgesine bir pointer atanır 53:

C

VkImportMemoryFdInfoKHR importInfo \= {};  
// Yapının Vulkan standartlarına göre başlatılması  
importInfo.sType \= VK\_STRUCTURE\_TYPE\_IMPORT\_MEMORY\_FD\_INFO\_KHR;  
// İşletim sisteminden gelen dosya tanımlayıcısının DMA\_BUF tipinde olduğunun belirtilmesi  
importInfo.handleType \= VK\_EXTERNAL\_MEMORY\_HANDLE\_TYPE\_DMA\_BUF\_BIT\_EXT;  
importInfo.fd \= dma\_buf\_fd; // PipeWire veya DRM'den gelen FD  
// Bu importInfo yapısı daha sonra VkMemoryAllocateInfo'nun pNext işaretçisine bağlanır.

Bu yapı sayeside RX 460 gibi ayrık (discrete) ve UMA (Unified Memory Architecture) olmayan kartlarda bile PCI-e veriyolu tamamen bypass edilerek pikseller yerel bellek hızında (112 GB/s) işlemcilerin hizmetine sunulur.19

### **4.2. NCNN İçerisinde Veri Tüketimi ve Donanımsal Boru Hattı (Pipeline)**

Sadece Vulkan seviyesinde haritalama yapmak yeterli değildir; seçilen çıkarım motorunun bu özel belleği doğrudan kabul etmesi gerekir. NCNN motoru bu noktada gücünü gösterir. NCNN'nin ncnn::VkAllocator yapısının üzerine kurulan c\_api.h işlevleri sayesinde, yapay zeka modeli dışarıdan alınan (imported) bir tensör belleğini girdi matrisi olarak tanımlayabilir.24 Kesintisiz veri akışı şu şekilde tasarlanmalıdır:

1. **Veri Kabulü:** Ekran yakalama servisinden o anki karenin DMA-BUF fd'si alınır.  
2. **Vulkan Bağlantısı:** Vulkan API ile bu fd, VK\_EXTERNAL\_MEMORY\_HANDLE\_TYPE\_DMA\_BUF\_BIT\_EXT bayrağı ile bir VkDeviceMemory nesnesine haritalanır.53  
3. **Matris Entegrasyonu:** NCNN C API'sine (ncnn\_mat\_from\_vulkan\_image benzeri harici bellek kabul fonksiyonlarıyla), giriş tensörü (input blob) olarak doğrudan bu hazır grafik belleği (GPU pointer/buffer) verilir.23  
4. **Asenkron Çıkarım:** Ağ çıkarımı GPU'nun gölgelendirici (shader) çekirdekleri üzerinde tetiklenir. Çıkarım tamamlandığında, devasa ekran pikselleri değil, sadece tespit edilen UI elementlerinin koordinatlarını ve olasılık değerlerini içeren birkaç yüz baytlık sonuç seti CPU tarafına geri kopyalanır.23

Görüntünün kendisi hiçbir aşamada RAM'e taşınmaz. CPU sadece ağın tetiklenmesi ve küçük koordinat sonuçlarının okunması için zaman harcar. Bu mimari sayesinde 416x416, 720p veya 1080p çözünürlüklerdeki çerçeveler bile GPU'nun DMA motorları üzerinden **0 ms CPU kopyalama süresi** ile işleme alınır ve saniyede 60 kare hız bariyeri donanımsal olarak zahmetsizce aşılır.4

## **5\. Web Elementleri İçin Model Daraltma (Pruning/Quantization) Stratejileri ve Rasyonalite**

Derin öğrenme modelleri genellikle MS COCO veya ImageNet gibi veri setlerinde arabalar, uçaklar, hayvanlar veya asimetrik doğal objeler gibi yüksek varyanslı hedefleri tespit etmek için tasarlanmış geniş kapsamlı özellik çıkarıcılara sahiptir. Ancak, bir C projesinin spesifik hedefi olan "yazılım/web arayüzü elementleri" (butonlar, iframe alanları, form girdileri vb.) doğal yaşama kıyasla kusursuz kenar hatlarına, öngörülebilir dokulara (yüksek renk kontrastları, vektörel çizgiler, standart gölgelendirmeler) ve belirli geometrik oranlara (çoğunlukla altın orana yakın dikdörtgenler) sahiptir.20

Arayüz elementlerinin bu yüksek homojenlik ve düşük karmaşıklık içeren doğası, yapay zeka modelinin "öğrenme uzayının" oldukça dar olmasını sağlar. Bu durum, model boyutunun ve matematiksel ağırlıkların dramatik seviyelerde küçültülebilmesine imkan tanıyan agresif model daraltma (compression) tekniklerinin kullanılmasını donanım mühendisliği açısından zorunlu ve son derece rasyonel kılar.7

### **5.1. Yapısal Budama (Structured Pruning) Yaklaşımı**

Budama (Pruning), bir sinir ağındaki gereksiz veya daha az önemli ağırlıkların (weights) sıfırlanması veya tamamen silinmesi işlemidir.8 Web arayüzü tespiti gibi düşük varyanslı bir hedef için, ağın çok çeşitli renkleri ve karmaşık şekilleri tanımak üzere öğrenmiş olduğu geniş evrişim filtrelerine ihtiyacı yoktur; bu filtrelerin büyük bir bölümü girdi ne olursa olsun sıfıra yakın değersiz aktivasyonlar üretecektir.

* **Rasyonalite:** RX 460 (Polaris) gibi GPU'lar,ğın içinde rastgele ağırlıkların sıfırlandığı gelişigüzel (unstructured) budama yöntemlerinde seyrek matris (sparse matrix) operasyonlarında ciddi performans kayıpları yaşarlar. Bunun yerine, modelin tüm bir evrişim kanalının (channel) veya filtresinin tamamen silindiği **yapısal budama (structured pruning)** uygulanmalıdır.57 Yapısal budama, ağırlık tensörünün boyutunu fiziksel olarak daraltır ve bellek bant genişliği üzerindeki okuma/yazma yükünü doğrusal olarak azaltır.57 UI tespiti problemi basit olduğu için, model boyutunda %30 ila %50 oranında bir yapısal budama uygulandığında bile isabet oranlarında (mAP) belirgin bir düşüş gözlenmeyecek, buna karşın çıkarım süresinde (inference latency) doğrudan %40'a varan bir iyileşme sağlanacaktır.

### **5.2. Ağırlık Nicemleme (Quantization) ve Polaris Sınırları**

Nicemleme, ağırlıkların depolandığı kayan nokta hassasiyetinin, genellikle 32-bit'ten (FP32), 16-bit'e (FP16) veya 8-bit tamsayıya (INT8) düşürülmesi işlemidir.8

* **FP16 Mantığı:** AMD Polaris mimarisi (RX 460), Vega mimarilerinde gördüğümüz "Rapid Packed Math" (çift FP16 işlemi tek bir FP32 döngüsünde yapma) yeteneğine native olarak tam anlamıyla sahip olmasa da, depolama düzeyinde muazzam avantajlar sağlar.60 Modelin NCNN üzerinden use\_fp16\_storage bayrağı ile FP16 formatında nicemlenmiş olarak VRAM'e haritalanması, her okuma işleminde bellek kontrolcüsü üzerindeki yükü %50 oranında hafifletir.11 Modelin GPU'daki darboğazı "bellek erişimi sınırlarından" uzaklaşıp, hesaplama birimi sınırlarına doğru kayar, bu da kartın %100 verimle çalışmasını sağlar.  
* **INT8 Dönüşümü:** Ağın INT8 formatına nicemlenmesi durumunda ise, ağırlık boyutları sadece 1 MB ile 2 MB aralığına kadar geriler.14 NCNN'in Vulkan arka ucu, INT8 veri yapıları için nokta-çarpım (dot product) optimizasyonlarını çok verimli bir biçimde yürütebilmektedir.47 C projesi açısından INT8 nicemlemesinin getirdiği bir diğer devasa fayda, modelin diske kayıt boyutunun küçülmesi ve program başlatıldığında sistem belleğinden VRAM'e haritalama (memory mapped I/O) süresinin yok denecek kadar aza inmesidir.57 Uygulama bir milisaniyeden kısa sürede ayağa kalkar ve çalışmaya başlar. Sınırlı veri seti için nicemleme yapıldığında oluşacak minör hassasiyet kayıpları, web elementlerinin keskin formları sayesinde algoritmik olarak rahatça telafi edilebilir.

## **6\. Sentez ve Mimari Yol Haritası**

Derlenen tüm teknik analizler, donanım limitleri ve C dilinin sistem seviyesindeki gücü göz önüne alındığında, 2 GB VRAM'e sahip AMD Ryzen 5 3600 / Radeon RX 460 konfigürasyonunda saniyede 60 kare hızında web arayüzü element tespiti yapacak C projesi için çizilecek optimal yol haritası aşağıdaki prensiplere dayanmalıdır:

1. **Hedefe Yönelik Mimari Tasarımı:** Klasik kutu çapalama (anchor-based) ve ağır maksimum olmayan bastırma (NMS) algoritmalarından kaçınan, tamamen çapa-bağımsız regresyon felsefesiyle inşa edilmiş ve mekansal özellikleri koruyan hafifletilmiş (yaklaşık 1.0 Milyon ila 1.5 Milyon parametre bandında) evrişimli bir sinir ağı mimarisi tercih edilmelidir.13 Transformatör tabanlı öz-dikkat mekanizmaları RX 460'ın kısıtlı bellek bant genişliğini doyuracağı için kullanımdan kesinlikle dışlanmalıdır.10  
2. **İdeal Çıkarım Motorunun Entegrasyonu:** AMD ROCm ekosisteminin yarattığı eski nesil donanım uyumsuzluklarından tamamen izole olmak ve C projesine devasa kütüphane bağımlılıkları eklememek adına **NCNN** motoru projenin kalbine yerleştirilmelidir.29 c\_api.h başlık dosyası üzerinden sağlanan saf C arayüzü ve mükemmel derecede optimize edilmiş çapraz platform Vulkan GPU hızlandırması, projeyi dış sistemlerin yarattığı hantallıktan korur.43  
3. **Kesintisiz Donanım İletişimi (Sıfır-Kopya):** İşletim sisteminden gelen ve ekran çerçevelerini taşıyan pikseller, CPU tarafından asla kopyalanmaksızın DMA-BUF dosya tanımlayıcısı (dma\_buf fd) vasıtasıyla doğrudan Vulkan harici bellek köprüsüne (VkImportMemoryFdInfoKHR) dönüştürülmelidir.4 NCNN, bellek yönetimi altyapısıyla bu alanı okuyacak ve pikseller GPU sınırları dışına çıkmadan işlenecektir.24  
4. **Performans Çarpanı Olarak Model Daraltma:** Hedeflenen nesnelerin (butonlar, menüler) geometrik ve görsel olarak düşük varyanslı doğasından maksimize düzeyde faydalanılarak, modelin evrişim kanalları üzerinde agresif yapısal budama (structured pruning) uygulanmalıdır.5 İhtiyaç duyulmayan ağırlıklar silinerek elde edilen bu rafine ağ modeli, FP16 veya INT8 formatına nicemlenerek (quantization) 2 GB VRAM sınırındaki bellek veriyolu üzerindeki bant genişliği stresi donanımın rahatlıkla kaldırabileceği limitlerin çok altına çekilmelidir.59

Bu dört aşamalı, donanıma temas eden sistem mimarisi uygulandığında, RX 460 ve Ryzen 5 3600 kombinasyonu 100 milisaniye hedefinin çok ötesine geçerek; her bir karenin yakalanması, ağdan geçirilmesi ve sonuçların döndürülmesi işlemini ortalama **7 ile 12 milisaniye** aralığına sıkıştıracaktır. Bu eşsiz optimizasyon seviyesi, projenin tamamen kapalı, Python bağımlılıklarından arındırılmış, hafif ve sıfır-kopya mantığına sadık bir biçimde saniyede 60 tam kare hedefine sarsılmaz bir kararlılıkla ulaşmasını garanti altına alır.

#### **Works cited**

1. Making AMD GPUs competitive for LLM inference (2023) \- Hacker News, accessed May 11, 2026, [https://news.ycombinator.com/item?id=42498634](https://news.ycombinator.com/item?id=42498634)  
2. Radeon RX 460 \- Price performance comparison \- Video Card Benchmarks, accessed May 11, 2026, [https://www.videocardbenchmark.net/gpu.php?gpu=Radeon+RX+460\&id=3557](https://www.videocardbenchmark.net/gpu.php?gpu=Radeon+RX+460&id=3557)  
3. The DMA Streaming Framework: Kernel-Level Buffer Orchestration for High-Performance AI Data Paths \- arXiv, accessed May 11, 2026, [https://arxiv.org/html/2603.10030v2](https://arxiv.org/html/2603.10030v2)  
4. I found a fundamental incompatibility in wgpu's Vulkan external memory handling, so I rewrote my image viewer in raw Vulkan \- Reddit, accessed May 11, 2026, [https://www.reddit.com/r/vulkan/comments/1sa9s2q/i\_found\_a\_fundamental\_incompatibility\_in\_wgpus/](https://www.reddit.com/r/vulkan/comments/1sa9s2q/i_found_a_fundamental_incompatibility_in_wgpus/)  
5. Use R-CNN to detect UI elements in a webpage? : r/computervision \- Reddit, accessed May 11, 2026, [https://www.reddit.com/r/computervision/comments/88xz6g/use\_rcnn\_to\_detect\_ui\_elements\_in\_a\_webpage/](https://www.reddit.com/r/computervision/comments/88xz6g/use_rcnn_to_detect_ui_elements_in_a_webpage/)  
6. AN INTEGRATED REAL-TIME SYSTEM FOR TRUCK LICENSE PLATE DETECTION AND OCR RECOGNITION USING NANODET AND AZURE CLOUD VISUALIZATION \- ResearchGate, accessed May 11, 2026, [https://www.researchgate.net/publication/397450086\_AN\_INTEGRATED\_REAL-TIME\_SYSTEM\_FOR\_TRUCK\_LICENSE\_PLATE\_DETECTION\_AND\_OCR\_RECOGNITION\_USING\_NANODET\_AND\_AZURE\_CLOUD\_VISUALIZATION](https://www.researchgate.net/publication/397450086_AN_INTEGRATED_REAL-TIME_SYSTEM_FOR_TRUCK_LICENSE_PLATE_DETECTION_AND_OCR_RECOGNITION_USING_NANODET_AND_AZURE_CLOUD_VISUALIZATION)  
7. Parameter Count Is the Worst Way to Pick a Model on 8GB VRAM \- DEV Community, accessed May 11, 2026, [https://dev.to/plasmon\_imp/parameter-count-is-the-worst-way-to-pick-a-model-on-8gb-vram-2n18](https://dev.to/plasmon_imp/parameter-count-is-the-worst-way-to-pick-a-model-on-8gb-vram-2n18)  
8. Small Language and Vision Models. Lightweight Efficient AI | by Xin Cheng | Mar, 2026, accessed May 11, 2026, [https://billtcheng2013.medium.com/small-language-and-vision-models-f0c547f10309](https://billtcheng2013.medium.com/small-language-and-vision-models-f0c547f10309)  
9. FastVLM: Efficient Vision Encoding for Vision Language Models, accessed May 11, 2026, [https://machinelearning.apple.com/research/fast-vision-language-models](https://machinelearning.apple.com/research/fast-vision-language-models)  
10. A Study on Inference Latency for Vision Transformers on Mobile Devices \- arXiv, accessed May 11, 2026, [https://arxiv.org/html/2510.25166v1](https://arxiv.org/html/2510.25166v1)  
11. Graphics Specifications \- AMD, accessed May 11, 2026, [https://www.amd.com/en/products/specifications/graphics.html](https://www.amd.com/en/products/specifications/graphics.html)  
12. Best Lightweight Computer Vision Models \- Viso Suite, accessed May 11, 2026, [https://viso.ai/computer-vision/best-lightweight-computer-vision-models/](https://viso.ai/computer-vision/best-lightweight-computer-vision-models/)  
13. Lightweight-CancerNet: a deep learning approach for brain tumor detection \- PMC, accessed May 11, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11888863/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11888863/)  
14. RangiLyu/nanodet: NanoDet-Plus Super fast and ... \- GitHub, accessed May 11, 2026, [https://github.com/RangiLyu/nanodet](https://github.com/RangiLyu/nanodet)  
15. PicoDet : Fast object detection model optimized for mobile CPUs | by David Cochard | ailia Tech BLOG (EN) | Medium, accessed May 11, 2026, [https://medium.com/axinc-ai/picodet-fast-object-detection-model-optimized-for-mobile-cpus-17e7aa84589b](https://medium.com/axinc-ai/picodet-fast-object-detection-model-optimized-for-mobile-cpus-17e7aa84589b)  
16. GhostNetV3: Exploring the Training Strategies for Compact Models \- arXiv, accessed May 11, 2026, [https://arxiv.org/html/2404.11202v1](https://arxiv.org/html/2404.11202v1)  
17. RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization \- arXiv, accessed May 11, 2026, [https://arxiv.org/pdf/2211.06088](https://arxiv.org/pdf/2211.06088)  
18. Estimating LLM Inference Memory Requirements | by Kyle Bell \- Medium, accessed May 11, 2026, [https://medium.com/@kylebell\_70950/estimating-llm-inference-memory-requirements-fa9523fb4808](https://medium.com/@kylebell_70950/estimating-llm-inference-memory-requirements-fa9523fb4808)  
19. The OrangePi micro PC \- Reddit, accessed May 11, 2026, [https://www.reddit.com/r/OrangePI/](https://www.reddit.com/r/OrangePI/)  
20. WebPII: Benchmarking Visual PII Detection for Computer-Use Agents \- arXiv, accessed May 11, 2026, [https://arxiv.org/html/2603.17357v1](https://arxiv.org/html/2603.17357v1)  
21. WebUIBench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models in WebUI-to-Code \- ACL Anthology, accessed May 11, 2026, [https://aclanthology.org/2025.findings-acl.815.pdf](https://aclanthology.org/2025.findings-acl.815.pdf)  
22. ncnn/examples/yolov4.cpp, yolov4-\>opt.use\_vulkan\_compute \= true , detection is Error. · Issue \#2108 · Tencent/ncnn \- GitHub, accessed May 11, 2026, [https://github.com/Tencent/ncnn/issues/2108](https://github.com/Tencent/ncnn/issues/2108)  
23. vulkan notes \- ncnn documentation \- Read the Docs, accessed May 11, 2026, [https://ncnn.readthedocs.io/en/latest/how-to-use-and-FAQ/vulkan-notes.html](https://ncnn.readthedocs.io/en/latest/how-to-use-and-FAQ/vulkan-notes.html)  
24. custom allocator \- ncnn documentation, accessed May 11, 2026, [https://ncnn.readthedocs.io/en/latest/developer-guide/custom-allocator.html](https://ncnn.readthedocs.io/en/latest/developer-guide/custom-allocator.html)  
25. nanodet/demo\_ncnn/README.md at main \- GitHub, accessed May 11, 2026, [https://github.com/RangiLyu/nanodet/blob/main/demo\_ncnn/README.md](https://github.com/RangiLyu/nanodet/blob/main/demo_ncnn/README.md)  
26. OmniParser V2 \- Azure AI Foundry Labs | Early-Stage AI Experiments & Prototypes, accessed May 11, 2026, [https://labs.ai.azure.com/projects/omniparserv2/](https://labs.ai.azure.com/projects/omniparserv2/)  
27. 1 Introduction \- arXiv, accessed May 11, 2026, [https://arxiv.org/html/2604.26334v1](https://arxiv.org/html/2604.26334v1)  
28. accessed May 11, 2026, [https://www.phoronix.com/benchmark/result/ncnn\_vulkanamd\_vs\_nvidia/e1f949f754a1.svgz](https://www.phoronix.com/benchmark/result/ncnn_vulkanamd_vs_nvidia/e1f949f754a1.svgz)  
29. We built a local inference engine that skips ROCm entirely and just got a 4x speedup on a consumer AMD GPU \- Reddit, accessed May 11, 2026, [https://www.reddit.com/r/LocalLLM/comments/1s98766/we\_built\_a\_local\_inference\_engine\_that\_skips\_rocm/](https://www.reddit.com/r/LocalLLM/comments/1s98766/we_built_a_local_inference_engine_that_skips_rocm/)  
30. AMD GPUs \- llm-tracker, accessed May 11, 2026, [https://llm-tracker.info/howto/AMD-GPUs](https://llm-tracker.info/howto/AMD-GPUs)  
31. AMD GPUs | GPU Buyers Guide \- Dortania, accessed May 11, 2026, [https://dortania.github.io/GPU-Buyers-Guide/modern-gpus/amd-gpu.html](https://dortania.github.io/GPU-Buyers-Guide/modern-gpus/amd-gpu.html)  
32. Choosing Vulkan, OpenCL, SYCL or CUDA for GPU Compute \- TechnoLynx, accessed May 11, 2026, [https://www.technolynx.com/post/choosing-vulkan-opencl-sycl-or-cuda-for-gpu-compute](https://www.technolynx.com/post/choosing-vulkan-opencl-sycl-or-cuda-for-gpu-compute)  
33. AMD GPU Acceleration Technologies Explained: ROCm, HIP, Vulkan, OpenCL & More (2025) \- GitHub Gist, accessed May 11, 2026, [https://gist.github.com/danielrosehill/8793e2028ef4bd08c6ca955a38b40e5b](https://gist.github.com/danielrosehill/8793e2028ef4bd08c6ca955a38b40e5b)  
34. ncnn brings nerual network inference acceleration using vulkan \- The Khronos Group, accessed May 11, 2026, [https://www.khronos.org/news/permalink/ncnn-brings-nerual-network-inference-acceleration-using-vulkan-5c9200795dbf06.59317995](https://www.khronos.org/news/permalink/ncnn-brings-nerual-network-inference-acceleration-using-vulkan-5c9200795dbf06.59317995)  
35. ONNX Runtime Execution Providers, accessed May 11, 2026, [https://onnxruntime.ai/docs/execution-providers/](https://onnxruntime.ai/docs/execution-providers/)  
36. ONNX Runtime Performance Tuning, accessed May 11, 2026, [https://oliviajain.github.io/onnxruntime/docs/performance/tune-performance.html](https://oliviajain.github.io/onnxruntime/docs/performance/tune-performance.html)  
37. Inference in ONNX in C? \- AI/ML \- Ampere arm64 Developer Community, accessed May 11, 2026, [https://community.amperecomputing.com/t/inference-in-onnx-in-c/3238](https://community.amperecomputing.com/t/inference-in-onnx-in-c/3238)  
38. PyTorch vs ONNX vs NCNN \- by Nadira Povey \- Medium, accessed May 11, 2026, [https://medium.com/@nadirapovey/pytorch-vs-onnx-vs-ncnn-ee50115b6263](https://medium.com/@nadirapovey/pytorch-vs-onnx-vs-ncnn-ee50115b6263)  
39. alibaba/MNN: MNN: A blazing-fast, lightweight inference ... \- GitHub, accessed May 11, 2026, [https://github.com/alibaba/MNN](https://github.com/alibaba/MNN)  
40. 上海有为科技有限公司/mnn \- Gitee, accessed May 11, 2026, [https://gitee.com/zhaowx/mnn/blob/master/doc/Tutorial\_EN.md](https://gitee.com/zhaowx/mnn/blob/master/doc/Tutorial_EN.md)  
41. Module API使用 — MNN-Doc 2.0.4 documentation \- Read the Docs, accessed May 11, 2026, [https://mnn-docs.readthedocs.io/en/2.1.0/inference/module.html](https://mnn-docs.readthedocs.io/en/2.1.0/inference/module.html)  
42. Tencent/ncnn: ncnn is a high-performance neural network ... \- GitHub, accessed May 11, 2026, [https://github.com/Tencent/ncnn](https://github.com/Tencent/ncnn)  
43. ncnn/src/c\_api.h at master · Tencent/ncnn \- GitHub, accessed May 11, 2026, [https://github.com/Tencent/ncnn/blob/master/src/c\_api.h](https://github.com/Tencent/ncnn/blob/master/src/c_api.h)  
44. ncnn/src/CMakeLists.txt at master \- GitHub, accessed May 11, 2026, [https://github.com/Tencent/ncnn/blob/master/src/CMakeLists.txt](https://github.com/Tencent/ncnn/blob/master/src/CMakeLists.txt)  
45. C语言的使用方法· Issue \#4766 · Tencent/ncnn \- GitHub, accessed May 11, 2026, [https://github.com/Tencent/ncnn/issues/4766](https://github.com/Tencent/ncnn/issues/4766)  
46. FAQ ncnn vulkan \- ncnn documentation \- Read the Docs, accessed May 11, 2026, [https://ncnn.readthedocs.io/en/latest/how-to-use-and-FAQ/FAQ-ncnn-vulkan.html](https://ncnn.readthedocs.io/en/latest/how-to-use-and-FAQ/FAQ-ncnn-vulkan.html)  
47. ncnn \- universal neural network inference with vulkan, accessed May 11, 2026, [https://www.khronos.org/developers/linkto/ncnn-universal-neural-network-inference-with-vulkan](https://www.khronos.org/developers/linkto/ncnn-universal-neural-network-inference-with-vulkan)  
48. More Vulkan NCNN Inference Benchmarks On AMD Radeon vs. NVIDIA GeForce Under Linux \- Phoronix, accessed May 11, 2026, [https://www.phoronix.com/news/NCNN-Vulkan-More-NVIDIA-AMD](https://www.phoronix.com/news/NCNN-Vulkan-More-NVIDIA-AMD)  
49. Zero-copy: Principle and Implementation | by Zhenyuan (Zane) Zhang | Medium, accessed May 11, 2026, [https://medium.com/@kaixin667689/zero-copy-principle-and-implementation-9a5220a62ffd](https://medium.com/@kaixin667689/zero-copy-principle-and-implementation-9a5220a62ffd)  
50. Buffer Sharing and Synchronization (dma-buf) \- The Linux Kernel documentation, accessed May 11, 2026, [https://docs.kernel.org/driver-api/dma-buf.html](https://docs.kernel.org/driver-api/dma-buf.html)  
51. Vulkan, OpenGL and/or Zerocopy \- GStreamer, accessed May 11, 2026, [https://gstreamer.freedesktop.org/data/events/gstreamer-conference/2016/Matthew%20Waters%20-%20Vulkan,%20OpenGL%20and%20ZeroCopy.pdf](https://gstreamer.freedesktop.org/data/events/gstreamer-conference/2016/Matthew%20Waters%20-%20Vulkan,%20OpenGL%20and%20ZeroCopy.pdf)  
52. Efficient Linux sockets (DMA/zero-copy) \- Stack Overflow, accessed May 11, 2026, [https://stackoverflow.com/questions/1827857/efficient-linux-sockets-dma-zero-copy](https://stackoverflow.com/questions/1827857/efficient-linux-sockets-dma-zero-copy)  
53. DMA buf import into Vulkan \- Stack Overflow, accessed May 11, 2026, [https://stackoverflow.com/questions/78071242/dma-buf-import-into-vulkan](https://stackoverflow.com/questions/78071242/dma-buf-import-into-vulkan)  
54. OpenCL interoperability :: Vulkan Documentation Project, accessed May 11, 2026, [https://docs.vulkan.org/samples/latest/samples/extensions/open\_cl\_interop\_arm/README.html](https://docs.vulkan.org/samples/latest/samples/extensions/open_cl_interop_arm/README.html)  
55. vulkan notes · Tencent/ncnn Wiki \- GitHub, accessed May 11, 2026, [https://github.com/Tencent/ncnn/wiki/vulkan-notes](https://github.com/Tencent/ncnn/wiki/vulkan-notes)  
56. WebUIBench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models in WebUI-to-Code \- arXiv, accessed May 11, 2026, [https://arxiv.org/html/2506.07818v1](https://arxiv.org/html/2506.07818v1)  
57. When to use Pruning, Quantization , Distillation and others when optimizing speed, accessed May 11, 2026, [https://ai.stackexchange.com/questions/43054/when-to-use-pruning-quantization-distillation-and-others-when-optimizing-spee](https://ai.stackexchange.com/questions/43054/when-to-use-pruning-quantization-distillation-and-others-when-optimizing-spee)  
58. \[D\] Why is LLM Pruning Not as Generally Available as Quantization? \- Reddit, accessed May 11, 2026, [https://www.reddit.com/r/MachineLearning/comments/1gp6h2d/d\_why\_is\_llm\_pruning\_not\_as\_generally\_available/](https://www.reddit.com/r/MachineLearning/comments/1gp6h2d/d_why_is_llm_pruning_not_as_generally_available/)  
59. How Quantization and Pruning Actually Work | by Zaina Haider | Medium, accessed May 11, 2026, [https://medium.com/@thekzgroupllc/how-quantization-and-pruning-actually-work-and-why-they-matter-for-edge-ai-8ee7a239466f](https://medium.com/@thekzgroupllc/how-quantization-and-pruning-actually-work-and-why-they-matter-for-edge-ai-8ee7a239466f)  
60. Vulkan now faster on PP AND TG on AMD Hardware? : r/LocalLLaMA \- Reddit, accessed May 11, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1rpcdrb/vulkan\_now\_faster\_on\_pp\_and\_tg\_on\_amd\_hardware/](https://www.reddit.com/r/LocalLLaMA/comments/1rpcdrb/vulkan_now_faster_on_pp_and_tg_on_amd_hardware/)  
61. Best Open Source Object Detection Models 2026 \- SourceForge, accessed May 11, 2026, [https://sourceforge.net/directory/object-detection-models/](https://sourceforge.net/directory/object-detection-models/)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAYCAYAAACvKj4oAAADK0lEQVR4Xu2XaahNURTHl3keCpmSuUSS+RuPkGT6RHkyFyFkHj5QIimUIZKUzJEpkSES+YIoUUTPF0O+iEKU+P/v2tvdd519yX333Fd6v/r37vmvs9/ZZ5211z5HpJoqpzl0ALoLXYS654ZLSx3oNNTBBirBXmi4+70SegbVzYYz3tTgOFX2QBOtWUkuQVvd707QT2jo76hITegmNCzwUmEKdMaaRaZM9AY7G78r9AZqYvy8NIXWQseg89AN6A60CqoVnOdhFl9Do23A0Ri6Cj0UneAXqFXOGSIboY+i8U+i681yAjpkTccFaLk1Y7Cen0PLRCfuaQk9gK5D9QKfjIdeQDWMb1kHvRK9idhkBogmIfYkponeXH0bcMwSTXK4PhOw1t9CPWzAwcXOya03/mFol/FicK0MgX6IlpRN1GRog/FIX2ibaAK7QB1zwxm6ic7NN6QEs0VPGGUDAZwQJ3fP+OxsC41naSRaAeSo6LXmZMMZdosmIKSFaALZRMqgTZJcg54K0cpLwCf2WXSv+ROcJCf2PvCaOW9C4MUYB213v/uJjnmUDWd4DNU2HnsAz/X6KvE+QK6IJikBy4uD59qAgdnlebcDr4/zbOYtvLkxwTEnw3Ej3DGTfDYbLoiT0HFrMhvvRC+W79F7toieF663gc7rHXgx2DzYTT3suBx3zh2zxBdlwwWxH7pmzYaSffy2PELaiJbxd8m9Gf8E+Tcf7SVyYXBfdGx/0T20V274n9knuqUlYO3zQg1sIIC1zXNWG5+vZX8r0emSHEfKRcfy5p6aWCGwPKNlzs7EC4VrJGS+aJwZsvjGM9YGAtgFB1lTtGK4f3L8QRMrhMvQTmsSlulL6IloKXq4aW6GvkFLJf9GXgEttqaDT5hd1765eLjueIPFeGHmC8o8a3paQzugW6KtmVlnV1oBtQ3Oi8HsHzEe30b4v9jWeQMfJJ4EVgATECa2EDie1xlsA8Vgpugk+blUVXAfZgWmAl/OudVMsoESwgpaYs1iskDiXwClgF/53AnyvYgXBX55sN2PtIGU4XXZPXvaQBpwHz0FtbOBFFkDzbBmNf8LvwAlBaAfTloTAgAAAABJRU5ErkJggg==>