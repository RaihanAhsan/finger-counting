"""
Program Pengolahan Citra: Deteksi Jari Tangan & Penghitung Total Jari
Menggunakan Kamera Internal Laptop, Python, YOLO (Ultralytics), OpenCV, dan MediaPipe.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from finger_counter import ContourFingerCounter, HandResult, MediaPipeFingerCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deteksi & Hitung Total Jari Tangan Real-time dengan Kamera Laptop + YOLO"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Index kamera (0 = kamera internal laptop, default: 0)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolo11n.pt",
        help="Path atau nama model YOLO (.pt), default: yolo11n.pt",
    )
    parser.add_argument(
        "--method",
        choices=["mediapipe", "contour"],
        default="mediapipe",
        help="Metode penghitungan jari: 'mediapipe' (akurat, 2 tangan) atau 'contour' (OpenCV klasik)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold untuk deteksi YOLO (default: 0.40)",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Jalankan tanpa overlay deteksi YOLO",
    )
    return parser.parse_args()


def draw_yolo_detections(
    frame: np.ndarray,
    model: YOLO,
    conf_threshold: float,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Menjalankan inferensi YOLO pada frame dan menggambar bounding box.
    Mengembalikan bounding box 'person' terbesar (jika ada) sebagai ROI opsional.
    """
    results = model.predict(frame, conf=conf_threshold, verbose=False)
    largest_person_box: Optional[Tuple[int, int, int, int]] = None
    largest_area = 0

    for res in results:
        if res.boxes is None:
            continue
        names = res.names
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Jika model COCO standar, fokus pada deteksi 'person' atau 'hand' jika model kustom
            is_coco_model = "person" in names.values() and "hand" not in names.values()
            if is_coco_model and cls_name != "person":
                continue

            color = (255, 165, 0) if cls_name == "person" else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"YOLO {cls_name}: {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

            if cls_name == "person":
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_person_box = (x1, y1, x2, y2)

    return largest_person_box


def draw_hud(
    frame: np.ndarray,
    total_fingers: int,
    hands_info: List[HandResult],
    method: str,
    yolo_enabled: bool,
    yolo_model_name: str,
    fps: float,
) -> None:
    """Menggambar panel informasi (HUD) pada frame."""
    h, w = frame.shape[:2]

    # Panel semi-transparan di kiri atas
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (370, 145), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (12, 12), (370, 145), (0, 220, 180), 2)

    cv2.putText(
        frame,
        f"TOTAL JARI: {total_fingers}",
        (26, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (0, 255, 150),
        3,
    )

    if hands_info:
        detail = " | ".join(f"{h_res.label}: {h_res.finger_count}" for h_res in hands_info)
    else:
        detail = "Arahkan tangan ke kamera..."

    cv2.putText(
        frame,
        detail,
        (26, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 230),
        1,
    )

    yolo_status = f"ON ({Path(yolo_model_name).name})" if yolo_enabled else "OFF"
    cv2.putText(
        frame,
        f"Metode: {method.upper()} | YOLO: {yolo_status}",
        (26, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 220, 255),
        1,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (26, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 255, 180),
        1,
    )

    # Bar petunjuk tombol di bagian bawah
    cv2.rectangle(frame, (0, h - 32), (w, h), (20, 20, 20), -1)
    cv2.putText(
        frame,
        "[Q/ESC] Keluar   [M] Ganti Metode   [Y] Toggle YOLO   [S] Screenshot",
        (16, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
    )


def main() -> None:
    args = parse_args()
    method = args.method
    yolo_enabled = not args.no_yolo

    print(f"[INFO] Memuat model YOLO '{args.yolo_model}'...")
    yolo_model: Optional[YOLO] = YOLO(args.yolo_model) if yolo_enabled else None

    print("[INFO] Menginisialisasi detektor jari MediaPipe & Contour...")
    mp_counter = MediaPipeFingerCounter(max_num_hands=2)
    contour_counter = ContourFingerCounter()

    print(f"[INFO] Membuka kamera internal (index={args.camera})...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Gagal membuka kamera index {args.camera}. "
            "Pastikan izin kamera sudah diberikan ke terminal/IDE di macOS System Settings."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Deteksi Jari Tangan - YOLO + Pengolahan Citra"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    captures_dir = Path("captures")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Gagal membaca frame dari kamera.")
                break

            # Mirror horizontal agar tampilan natural seperti cermin
            frame = cv2.flip(frame, 1)

            if yolo_enabled:
                if yolo_model is None:
                    yolo_model = YOLO(args.yolo_model)
                draw_yolo_detections(frame, yolo_model, args.conf)

            hands_info: List[HandResult] = []
            if method == "mediapipe":
                total_fingers, hands_info = mp_counter.process(frame, draw=True)
            else:
                total_fingers, skin_mask = contour_counter.process(frame, draw=True)
                # Tampilkan preview kecil mask kulit di pojok kanan atas
                mask_bgr = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
                preview = cv2.resize(mask_bgr, (200, 150))
                fh, fw = frame.shape[:2]
                frame[12:162, fw - 212 : fw - 12] = preview
                cv2.rectangle(frame, (fw - 212, 12), (fw - 12, 162), (255, 180, 0), 2)

            now = time.time()
            fps = 1.0 / max(1e-5, now - prev_time)
            prev_time = now

            draw_hud(
                frame=frame,
                total_fingers=total_fingers,
                hands_info=hands_info,
                method=method,
                yolo_enabled=yolo_enabled,
                yolo_model_name=args.yolo_model,
                fps=fps,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # Q atau ESC
                break
            elif key == ord("m"):
                method = "contour" if method == "mediapipe" else "mediapipe"
                print(f"[INFO] Metode diganti ke: {method}")
            elif key == ord("y"):
                yolo_enabled = not yolo_enabled
                print(f"[INFO] YOLO overlay: {'ON' if yolo_enabled else 'OFF'}")
            elif key == ord("s"):
                captures_dir.mkdir(exist_ok=True)
                filename = captures_dir / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(filename), frame)
                print(f"[INFO] Screenshot disimpan: {filename}")
    finally:
        cap.release()
        mp_counter.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
