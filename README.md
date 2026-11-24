# Deteksi & Penghitung Total Jari Tangan (Python + YOLO + Pengolahan Citra)

Program pengolahan citra real-time menggunakan **kamera internal laptop** untuk mendeteksi tangan dan menghitung **total jari yang ditunjukkan** (0–10 jari untuk 2 tangan).

## Fitur Utama
- **Deteksi Objek YOLO (`ultralytics`)**: Menggunakan `yolo11n.pt` secara default (otomatis terunduh saat pertama dijalankan), atau mendukung model YOLO kustom (`.pt`) hasil training tangan/jari.
- **2 Metode Penghitungan Jari**:
  1. **MediaPipe HandLandmarker (`mediapipe`)** *(default)*: Mendeteksi hingga 2 tangan (21 titik landmark per tangan) dan menghitung total jari secara akurat & rotasi-invarian.
  2. **Pengolahan Citra Klasik (`contour`)**: Menggunakan segmentasi warna kulit YCrCb, *Gaussian Blur*, operasi morfologi, *Contour*, *Convex Hull*, dan *Convexity Defects* (lengkap dengan preview mask kulit).
- **Kontrol Keyboard Interaktif**:
  - `Q` / `ESC` : Keluar dari program
  - `M` : Ganti metode (`mediapipe` ↔ `contour`)
  - `Y` : Aktifkan / nonaktifkan overlay deteksi YOLO
  - `S` : Simpan screenshot hasil deteksi ke folder `captures/`

## Cara Menjalankan

Aktifkan virtual environment dan jalankan `main.py`:

```bash
.venv/bin/python main.py
```

### Opsi Tambahan (CLI)
```bash
# Menggunakan metode pengolahan citra klasik (Contour + Convexity Defects)
.venv/bin/python main.py --method contour

# Menggunakan model YOLO kustom (misal hasil training hand/finger detection)
.venv/bin/python main.py --yolo-model path/to/best.pt

# Menggunakan kamera eksternal (index 1)
.venv/bin/python main.py --camera 1

# Menjalankan tanpa overlay YOLO
.venv/bin/python main.py --no-yolo
```

> **Catatan macOS**: Saat pertama kali dijalankan, macOS akan meminta izin akses kamera untuk Terminal / IDE Anda. Pastikan izin tersebut disetujui agar kamera internal laptop dapat digunakan.
