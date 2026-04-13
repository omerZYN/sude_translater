# sude_translater

**Sude - HotWire Foam Cutter** — DXF profillerinden **Grbl HotWire 6.5 (XYZA)** formatında
4 eksenli hotwire köpük kesim makinesi için G-code üreten Python + Tkinter uygulaması.

## Özellikler

- **DXF import**: LWPOLYLINE, SPLINE, LINE, ARC (kompozit profiller)
- **DXF fidelity**: Düz kanatta ham DXF noktaları aynen korunur, spar çentikleri silinmez
- **Chord-fraction eşleştirmeli profil interpolasyonu** (yalnızca taper için)
- **Taper projeksiyon** (üçgen benzerliği) — dar köpük, geniş carriage aralığı
- **Bağımsız Root/Tip X offset** (sweep) ve **Root Y / Tip Z offset** (bağımsız dikey kaydırma)
- **Kerf + Tel tolerans telafisi** (ısıl erime payı)
- **Saat yönünde (CW) kesim** profilin orta-alt noktasından başlayıp tekrarlanabilir şekilde
- **2D profil önizleme** (motor 0 referans çizgileriyle) + **3D kesim önizleme**
- **Toolpath onay penceresi**: G-code üretilmeden önce her iki carriage'ın tam yolu,
  approach/retract dahil 3D'de gösterilir — onay verilmezse dosya üretilmez
- **Her input için tooltip** — ⓘ ikonuna gelince açıklama çıkar
- **Kalıcı ayarlar**: son girilen değerler `~/.sude_hotwire_settings.json` ile oturumlar arası hatırlanır
- **Makine limitleri**: X/A ≤ 700 mm, Y/Z ≤ 400 mm kontrolü
- **Eksen eşlemesi**: X = root yatay, Y = root dikey, Z = tip dikey, A = tip yatay
- **Spar deliği** ayrı bir DXF olarak yüklenebilir
- **.nc kaydetme**

## Kurulum (Windows)

Arkadaşına en kolay yol `Sude_Windows/` klasöründeki hazır paket:

1. **Python kur**: <https://www.python.org/downloads/> — kurulum sırasında
   **"Add Python to PATH"** kutusunu mutlaka işaretle.
2. `Sude_Windows/` klasörünü indir (veya tüm repo'yu `git clone`).
3. O klasörde **`DERLE.bat`** dosyasına çift tıkla — kütüphaneler otomatik yüklenir ve
   `dist/Sude.exe` oluşur (2-3 dakika).
4. `dist\Sude.exe`'yi çift tıkla, program açılır. EXE tek başına çalışır; istediğin yere
   kopyalayabilirsin.

Detaylı adımlar için `Sude_Windows/BENIOKU.txt` dosyasına bak.

## Kurulum (macOS / Linux / kaynaktan)

```bash
pip install -r requirements.txt
python3 hotwire_cutter.py
```

`.app` paketi için `pyinstaller Sude.spec`.

## Kullanım

1. **Root Profil** → Yükle ile DXF seç (kapalı profil olmalı)
2. Farklı bir uç profili varsa **Tip Profil** → Yükle (düz kanatta boş bırak)
3. Gerekiyorsa **Spar Profil** → Yükle (opsiyonel)
4. Sağdaki parametreleri ayarla — her input'un yanındaki ⓘ ikonuna gel, ne işe
   yaradığını açıklayan tooltip çıkar.
5. **Profil Onizle** / **3D Onizle** ile geometriyi doğrula
6. **G-code Uret** → toolpath onay penceresi açılır; her iki carriage'ın tam yolunu
   incele, sonra **Onayla ve G-code Uret**
7. **G-code Kaydet (.nc)** → Grbl HotWire 6.5 uygulamasında aç ve kes

## Parametreler Özeti

| Parametre | Açıklama |
|---|---|
| Feed Rate | Kesim hızı (mm/min) |
| Tel Mesafesi | İki carriage arasındaki span (mm) |
| Kopuk Genisligi | Köpük bloğunun span yönündeki eni (mm) |
| Sol Bosluk | Sol carriage ile köpük sol yüzü arasındaki mesafe |
| Root/Tip X Offset | Yatay (X / A) kesim başlangıç ötelemesi |
| Root Y / Tip Z Offset | Dikey kesim başlangıç ötelemesi (motor 0 = 0) |
| Kerf Offset | Telin köpükte bıraktığı oluk (tel çapı) |
| Tel Tolerans | Isıl erime payı |
| Guvenli Yukseklik | Approach/retract için ilk kesim noktasının altındaki güvenli mesafe |
| Nokta Sayisi | Taper interpolasyonu için nokta sayısı (düz kanatta kullanılmaz) |

## Koordinat Sistemi

- **X** = Sol (root) carriage yatay, **Y** = Sol carriage dikey
- **A** = Sağ (tip) carriage yatay, **Z** = Sağ carriage dikey
- Her iki carriage için **sıfır noktası motor tarafı** (X=0 ve Y=0 / A=0 ve Z=0)
- Profil otomatik olarak: hücum kenarı → X=0, tabanı → Y=0 düzlemine normalize edilir
- Kullanıcı offset'leri sıfırken profil motor 0'a dayalı durur; offset girildikçe
  bu ofsetler uygulanır

## Lisans

Kişisel kullanım. Dağıtımdan önce yazara danış.
