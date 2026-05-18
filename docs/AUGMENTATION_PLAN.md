# NanoDet-Plus-m-416 Veri Artırma (Data Augmentation) Planı

## 📁 Dizin Yapısı

```
Senkronik/
├── resimler/                          # Kaynak PNG'ler (5 adet, 8-bit RGBA)
│   ├── 2a5bb9cb-8316-4f5e-b3ed-304116bcb0df.png   (504x73)  — "Create Account" butonu
│   ├── 4c36ed6d-99fd-46b7-b7ac-786e35300123.png   (473x75)  — "Audio Puzzle" ikonu
│   ├── 5c8371ce-121d-4c1d-ac33-f72fbe565a58.png   (378x53)  — Diğer UI elementi
│   ├── 9f2c41f9-b6d8-44e7-83e5-731c2320b9ca.png   (330x282) — Diğer UI elementi
│   └── e644800e-eb03-4356-b758-b6f7090f967e.png   (396x124) — Diğer UI elementi
├── dataset/
│   ├── images/
│   │   ├── train/                    # ~160 augment edilmiş görsel
│   │   └── val/                      # ~40 augment edilmiş görsel
│   ├── labels/
│   │   ├── train/                    # YOLO-format etiketler
│   │   └── val/
│   └── meta.yaml                     # Sınıf tanımları
├── tools/
│   └── augment_data.py               # Ana augmentasyon scripti
└── docs/
    └── augmentation_log.md           # Üretim kayıtları
```

## 🎯 NanoDet-Plus-m-416 Gereksinimleri

| Parametre | Değer |
|---|---|
| Giriş çözünürlüğü | 416×416 piksel |
| Label formatı | `class_id x_center y_center width height` (normalize edilmiş 0–1) |
| Kanal sırası | RGB |
| Train/Val split | 80/20 |
| Toplam hedef veri | ≥200 image |

## 🔧 Augmentasyon Teknikleri

### 1. Pano (Canvas) Varyasyonları
- **Arka plan renkleri**: Beyaz (`#FFFFFF`), açık gri (`#F0F0F0`), koyu (`#1A1A2E`), GitHub dark (`#0D1117`)
- **Gradyanlar**: Dikdörtgen üzerinde rastgele lineer gradyan (2 renk arası)
- **Gürültü**: Gaussian noise (σ = 5–20), salt-and-pepper (0.5–2% olasılık)

### 2. Konum Varyasyonları
Kaynak PNG'ler canvas üzerinde rastgele yerleştirilir:
- X konumu: `0` ile `canvas_w - src_w` arasında uniform dağılım
- Y konumu: `0` ile `canvas_h - src_h` arasında uniform dağılım
- **Sınırlama**: PNG'nin merkezi canvas'ın dışına çıkmaz

### 3. Boyut/Zoom Varyasyonları
- **Scale faktörleri**: 0.50, 0.70, 0.80, 1.00, 1.20, 1.50
- Ölçeklenen genişlik/hepsi `int(src_w * scale)` ile hesaplanır
- Minimum boyut: 20px (çok küçültmeyi engelle)

### 4. Renk/Dış Görünüm Varyasyonları
- **Aydınlık mod**: RGB kazancı (±10–30), kontrast (±10–20%)
- **Karanlık mod**: RGB kazancı (±5–15), kontrast artışı (+15–30%)
- **Renk kayması**: HSV'de hue ±10°, saturation ±20%, value ±15%
- **Tersleme**: Yatay ve dikey çevirme (ayrı augmentasyonlar)

### 5. Ekstra Işık/Gürültü Efektleri
- **Gaussian blur**: kernel 1–3, sigma 0.5–1.5
- **Parlaklık (brightness)**: ±20%
- **Sharpness**: ±30%
- **JPEG kalite artefaktları**: Kaydedirken quality=70–95 arası rastgele

## 📊 Label Üretimi

Her augmentasyonda bounding box otomatik hesaplanır:

```python
# Normalize edilmiş koordinatlar
x_center = (x_offset + scaled_w / 2) / canvas_w
y_center = (y_offset + scaled_h / 2) / canvas_h
width    = scaled_w / canvas_w
height   = scaled_h / canvas_h
```

**Sınıf ataması:**
| class_id | Açıklama | Kaynak PNG |
|---|---|---|
| 0 | create_account | `2a5bb9cb-...` |
| 1 | audio_puzzle | `4c36ed6d-...` |
| 2 | ui_element_3 | `5c8371ce-...` |
| 3 | ui_element_4 | `9f2c41f9-...` |
| 4 | ui_element_5 | `e644800e-...` |

## 🐍 Python Script — `tools/augment_data.py`

### Bağımlılıklar (mevcut ortamda yüklü)
```
Pillow==12.2.0
numpy==2.4.4
```

### Akış Diyagramı

```
Kaynak PNG'ler (5 adet)
       │
       ▼
  Arka plan oluştur
  (renk/gradyan/gürültü)
       │
       ▼
  PNG'yi ölçekle (0.5–1.5x)
       │
       ▼
  Canvas'a rastgele yerleştir
       │
       ▼
  Renk augmentasyonu uygula
  (aydınlık/karanlık/shift)
       │
       ▼
  416×416'a resize & crop
       │
       ▼
  Label dosyası yaz (YOLO format)
       │
       ▼
  Train/Val split (80/20)
       │
       ▼
  dataset/images/{train,val}/
  dataset/labels/{train,val}/
```

### Ana Fonksiyonlar

1. **`create_background(width, height, mode)`** → PIL.Image
   - `mode`: "white", "dark", "gray", "gradient", "noisy"

2. **`random_scale(src_img, scale_range=(0.5, 1.5))`** → PIL.Image
   - Uniform rastgele scale faktörü

3. **`random_position(canvas_w, canvas_h, elem_w, elem_h)`** → (x, y)
   - Sınır kontrolü ile rastgele konum

4. **`apply_color_jitter(img, brightness, contrast, hue_shift)`** → PIL.Image
   - Pillow ImageEnhance kullanarak

5. **`add_noise(img, noise_type, intensity)`** → PIL.Image
   - gaussian, salt_pepper

6. **`generate_yolo_label(x, y, w, h, canvas_w, canvas_h, class_id)`** → str
   - Normalize edilmiş string

7. **`augment_all(src_dir, out_dir, count_per_class=40)`**
   - Ana döngü: her sınıf için N augmentasyon üret

### Çalıştırma Komutu

```bash
pip install Pillow numpy scipy  # scipy zaten yüklü
python3 tools/augment_data.py \
    --src resimler/ \
    --out dataset/ \
    --per-class 40 \
    --canvas-size 416 \
    --seed 42
```

### Örnek Çıktı

`dataset/images/train/aug_0001.png` → `dataset/labels/train/aug_0001.txt`
```
# aug_0001.txt içeriği (örnek)
0 0.512 0.488 0.121 0.018
# class_id=0 (create_account), merkez=(0.512, 0.488), boyut=(0.121, 0.018)
```

## 🔗 Zig Tarafında Kullanım

Augmentasyon tamamlandıktan sonra, NanoDet modeli `tools/` dizinine `nano_det_model.bin` olarak konumlandırılacak. Mevcut `tools/audio_payload_probe.zig` pattern'i referans alınarak:

1. `src/tools/nanodet_inference.zig` — TFLite/NCNN model inference
2. `src/tools/nanodet_train.zig` — Varsa training pipeline wrapper

Bu dosyalar henüz geliştirilmeli DEĞİL — önce veri üretimi tamamlanmalı.

## ⚡ Önemli Notlar

1. PNG'ler **alpha kanalına** sahip (RGBA). Canvas üzerine yapıştırılırken alpha blending uygulanır.
2. NanoDet 416×416 giriş bekler — büyük canvas'lar center-crop ile küçültülür.
3. Çok küçük kaynak PNG'ler (örn. 330×282) 0.5x scale'de bile >150px olur → yeterli.
4. **Augmentasyon sırasında hiçbir zaman raw byte offset kullanılmaz** — tüm koordinatlar PIL.Image üzerinden hesaplanır.
5. Her augmentasyon deterministik seed ile üretilir → tekrarlanabilirlik sağlanır.

## 📋 İlerleme Takibi

- [ ] `tools/augment_data.py` yazımı
- [ ] 200+ image/label üretimi
- [ ] Train/val split doğrulanması
- [ ] Label formatının NanoDet ile uyumluluğu testi
- [ ] `docs/augmentation_log.md` güncellemesi

## 🔍 Kaynaklar

- NanoDet-Plus: https://github.com/RangiLyu/nanodet
- YOLO Label Format: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
- Pillow Dokümantasyonu: https://pillow.readthedocs.io/