# Senkronik

```
Görüntü işleme, tarayıcı kullanımı. Go-go-gadget'ın fiber ağacına tepki olarak farklı
strateji olarak hayata çıkmıştır ve deadline maksimum 7 gündür.

deadline'ın amınakoyduk bulanlar arayanlardır, bas amınakoyayım.
```

> **[Arkose Go-Go-Gadget](https://github.com/react-RE/arkose-go-go-gadget)** — bu notta adı geçen, projenin çıkış sebebi olan repo.

---

## Nedir Bu? · What Is This?

**Senkronik** (Türkçede "eşzamanlı/senkronize" anlamına gelir), bir bilgisayarın
ekranını görüp anlayan, sonra da fareyi oynatarak tepki veren otomatik bir sistemdir.

Şöyle çalışır:

```
  Ekranını izle  →  Gördüğünü anla  →  Fareyi oynat
  (screen_capture)  (Doggystyle / Siege Engine)    (uinput_mouse)
```

**Senkronik** (Turkish for "synchronous") is an automated system that watches your
computer screen, understands what it sees using AI, and moves the mouse in response.

```
  Watch screen  →  Understand with AI  →  Move mouse
  (screen_capture) (Doggystyle / Siege Engine)   (uinput_mouse)
```

Bir oyunda otomatik hedef almak gibi düşünün: önce ekrandaki hedefi bulursun,
sonra nişan alıp ateşlersin. Ama burası oyun değil — gerçek işler için.

---

## 3 Ana Araç · 3 Main Tools

Bu projede 3 tane araç (binary) var:

| Araç · Tool | Ne İşe Yarar? · What It Does |
|---|---|
| `uinput_mouse` | Klavye/fare olmayan bir sisteme **sanal fare** takar. Sanki görünmez bir el fareyi oynatıyormuş gibi. · Injects a **virtual mouse** into Linux — like an invisible hand moving your cursor. |
 | `screen_capture` | Ekranın fotoğrafını çeker, kırpar (crop), küçültür ve **Doggystyle** (Siege Engine) motoruna yollar. · Captures your screen, crops it, converts colors, and sends frames to the **Doggystyle** vision engine. |
| `uinput_click` | İki yöntemle tıklar: önce XTest dener (sanal ekranlar için), olmazsa doğrudan uinput kullanır. · Clicks using two methods: tries XTest first (for virtual screens), falls back to uinput. |

---

## Veri Akışı · Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SENKRONIK                                      │
│                                                                         │
│  ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐  │
│  │  Screen Capture   │     │  Doggystyle       │     │  uinput_mouse  │  │
│  │  (screen_capture) │────▶│  (Siege Engine)    │────▶│  (Virtual Mouse│  │
│  │                   │ DMA │                   │ SHM │                │  │
│  │  PipeWire ile     │─buf │  NanoDet modeli   │     │  500+ Hz       │  │
│  │  1920×1080 alır   │     │  454×1080 analiz  │     │  click/move    │  │
│  │  → 454×1080 crop  │     │  UI elemanı bulur │     │                │  │
│  │  → BGR'a çevirir  │     │  koordinat üretir │     │  kernel-level  │  │
│  └──────────────────┘     └──────────────────┘     └────────────────┘  │
│                               ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  uinput_click (yedek tıklayıcı · backup clicker)               │   │
│  │  XTest → uinput fallback                                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Özet:** Ekran görüntüsü PipeWire ile alınır → DMA-BUF ile sıfır kopya → screen_capture.c
kırpar ve BGR'a çevirir → paylaşımlı bellek (frame_shm) ile Doggystyle (Siege Engine)'a
gönderir → Siege Engine ekrandaki UI öğelerini bulur → koordinatları uinput_mouse'a
iletir → uinput_mouse kernel seviyesinde fareyi oynatır/tıklar.

---

## Derleme · Build

Gereken kütüphaneler: `libpipewire-0.3-dev`, `libspa-0.2-dev`, `libsystemd-dev`,
`libx11-dev`, `libxtst-dev`

```bash
# İkisini birden derle
make

# Sadece uinput_mouse
make uinput_mouse

# Sadece screen_capture (Doggystyle frame_shm gerektirir)
make screen_capture

# uinput_click (manuel, Makefile'da henüz yok)
gcc src/click.c src/uinput_mouse.c -o uinput_click -lpthread -lX11 -lXtst

# Test et
make test

# Temizle
make clean
```

`screen_capture` derlenirken projenin yanındaki `../doggystyle/src/bridge/frame_shm.h`
dosyasını kullanır. Doggystyle yoksa derlenmez.

---

## Çalıştırma · Usage

### uinput_mouse

```bash
# Test modu: 500 Hz fare, ekranda rastgele gezinir, tıklar, Hz raporlar
./uinput_mouse

# Tek tık: (960, 540) noktasına tıkla (1920×1080 ekranda)
./uinput_mouse 960 540

# Özel çözünürlükle tek tık
./uinput_mouse 960 540 1920 1080

# 4K ekranda tıkla
./uinput_mouse 1920 1080 3840 2160
```

### screen_capture

```bash
# İlk çalıştırma: Portal diyalogu açılır, hangi ekranı yakalayacağını seç
./screen_capture

# Sonraki çalıştırmalar: diyalog atlanır (restore token ile)
./screen_capture
```

> **İpucu:** İlk seferde bir ekran seçtikten sonra, `/tmp/senkronik_restore_token`
> dosyasına session kaydedilir. Bir daha diyalog görmezsin — direkt yakalamaya başlar.

### uinput_click

```bash
# XTest dener (Xvfb varsa oraya tıklar), olmazsa uinput kullanır
./uinput_click 960 540
```

---

## Portal Restore Token (Sihirli Anahtar)

`screen_capture` her çalıştığında Portal diyalogu (hangi ekranı yakalayayım?)
göstermek zorunda değildir. İlk seferde seçtiğin ekranın "anahtarını"
`/tmp/senkronik_restore_token` dosyasına kaydeder. Sonraki çalıştırmalarda:

```
→ restore_token var mı? → EVET → SelectSources atla, direkt Start'a geç
→ restore_token yok mu? → HAYIR → Portal diyalogunu göster, seç, kaydet
```

Bu sayede **arkaplanda sessizce çalışabilir** — her açılışta bir ekran seçmen
gerekmez.

---

## Doggystyle Bridge · Köprü

**Doggystyle** (`doggystyle` git repo adı, metaforik kod adı), bu projenin yanında geliştirilen Siege Engine'dir (zig). Gözü
görüp beyni çalıştıran kısım odur. Senkronik ise ona şunları sağlar:

1. **frame_shm** — paylaşımlı bellek bölgesi. `screen_capture` buraya kırpılmış
   BGR kareleri yazar, Doggystyle okur.
2. **uinput_mouse** — Doggystyle'ın karar verdiği koordinatlara tıklama aracı.

Siege Engine, NanoDet-Plus-m-416 modeliyle ekrandaki UI öğelerini (buton, metin
kutusu, simge) tanır ve Senkronik'e "şuraya tıkla" der.

> **Doggystyle reposu:** [github.com/void0x14/doggystyle](https://github.com/void0x14/doggystyle)

---

## Dizin Yapısı · File Reference

```
Senkronik/
├── src/
│   ├── main.c             # uinput_mouse giriş noktası: CLI ve test modu
│   ├── uinput_mouse.c     # Sanal fare sürücüsü (Logitech G Pro taklidi)
│   ├── uinput_mouse.h     # uinput_mouse API'si
│   └── click.c            # uinput_click: XTest + uinput çift yollu tıklayıcı
│
├── screen_capture.c       # PipeWire + DMA-BUF ekran yakalama + Doggystyle köprüsü
│
├── Makefile               # Derleme: uinput_mouse + screen_capture
├── README.md              # Bu dosya
├── LICENSE                # GNU AGPL v3
│
├── build/                 # Derlenmiş .o dosyaları
├── docs/                  # Araştırma notları ve optimizasyon raporları
├── resimler/              # UI elementi PNG'leri (veri artırma için)
│
├── uinput_mouse           # Derlenmiş binary (22 KB)
├── uinput_click           # Derlenmiş binary (22 KB)
├── screen_capture         # Derlenmiş binary (57 KB)
│
└── arastirmalar-ve-cevaplar/
    └── Rasyonel Araştırma - Adım 1.1 (Headless Düzenlemesi)/
        ├── soru.md        # Headless Chrome + DMA-BUF + uinput sorusu
        └── cevap(grok-expert).md  # Detaylı cevap
```

---

## Gereken Kütüphaneler · Dependencies

| Kütüphane | Ne İçin? |
|---|---|
| `libpipewire-0.3-dev` | Ekran yakalama (PipeWire stream) |
| `libspa-0.2-dev` | PipeWire eklentileri |
| `libsystemd-dev` | D-Bus (Portal ile iletişim) |
| `libx11-dev`, `libxtst-dev` | XTest tıklama (uinput_click) |
| Linux kernel `uinput` modülü | Sanal fare (`/dev/uinput`) |
| `input` grup üyeliği veya `sudo` | uinput'a erişim izni |

---

## Kim Ne Zaman Kullanır? · Quick Decision Guide

| Yapmak İstediğin · What You Want | Hangi Araç · Which Tool |
|---|---|
| Fareyi bir noktaya tıkla · Click at a point | `uinput_mouse <x> <y>` |
| Ekranı yakalamaya başla · Start screen capture | `screen_capture` |
| Sanal ekrana (Xvfb) tıkla · Click on virtual screen | `uinput_click <x> <y>` |
| AI'a ekran görüntüsü gönder · Send frame to AI | `screen_capture` (otomatik) |
| Fare Hz testi yap · Test mouse polling rate | `./uinput_mouse` (argümansız) |

---

## Yol Haritası · Roadmap

- [ ] **Kullanıcı Deneyimi** — Kullanıcı deneyimi arttırılacak; araç, yapay zekaların ve yapay sinir ağlarının native şekilde kullanabileceği şekilde dizayn edilecektir.
- [ ] **Kullanım Kılavuzu** — Kullanım kılavuzu, argümanlar ve dokümantasyon oluşturulacak ve genişletilecektir.
- [ ] **Gözlemleme & Takip** — Gerçek zamanlı ve kayıtlı (log) gözlemleme ve takip mekanizmaları eklenecektir.
- [ ] **Performans & Güvenlik** — Performans ve güvenlik sorunları masaya yatırılacak, iyileştirilecek.
- [ ] **Zig Portu** — Mevcut C kod tabanının Zig diline portu yapılacak.
- [ ] **Tam Konfigürasyon** — Kullanıcının istediği şekilde mouse tipleri, Hz değerleri ve tam konfigüre etme desteği eklenecek.
- [ ] **Saf Metal / Sıfır Bağımlılık** — Hiçbir ek kütüphaneye, ek soyutlamaya ve gereksiz yüke gerek kalmadan metal seviyesinde, sıfır bağımlılıkla raw olarak tekrar yazılacaktır.
- [ ] **Çoklu Dil Entegrasyonu** — Hangi dilden projeniz olursa olsun entegre kolaylığı sağlanacaktır.
- [ ] **Dokümantasyon** — Dökümantasyon güçlendirilecek, basitleştirilecek ve daha da anlaşılır hale getirilecektir.
- [ ] **Bağımsız Repo Yapısı** — Senkronik kaynakları ile Siege Engine (Doggystyle) kaynakları bağımsızlaştırılacak. Senkronik kendi başına bağımsız bir proje olarak yürütülecek, ancak Doggystyle reposu ile bağı klasör yapısında korunacak. Böylece her iki proje de ayrı ayrı daha rahat kullanılabilecek.
- [ ] **Native Tarayıcı Çözümü** — Tarayıcı kullanımı üzerine native çözümler üretilecek. Bu repodaki her şey birbiriyle senkronik çalışacak; hedef, en stealth, en lightweight ve en hızlı çalışan tarayıcı kullanım sistemini inşa etmek.

---

## Lisans · License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).
