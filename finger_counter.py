"""
Modul deteksi dan penghitungan jumlah jari tangan menggunakan:
1. MediaPipe HandLandmarker (21 titik landmark tangan)
2. Pengolahan Citra Klasik OpenCV (Segmentasi Kulit YCrCb + Convex Hull & Convexity Defects)
"""

from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Koneksi antar 21 landmark tangan untuk menggambar kerangka tangan di OpenCV
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # Ibu jari (Thumb)
    (0, 5), (5, 6), (6, 7), (7, 8),        # Telunjuk (Index)
    (0, 9), (9, 10), (10, 11), (11, 12),   # Jari tengah (Middle)
    (0, 13), (13, 14), (14, 15), (15, 16), # Jari manis (Ring)
    (0, 17), (17, 18), (18, 19), (19, 20), # Kelingking (Pinky)
    (5, 9), (9, 13), (13, 17),             # Telapak tangan (Palm)
)


@dataclass
class HandResult:
    """Hasil deteksi satu tangan."""
    label: str
    finger_count: int
    fingers_up: List[bool]
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def ensure_hand_landmarker_model(model_path: str | Path = "models/hand_landmarker.task") -> Path:
    """Memastikan file model MediaPipe HandLandmarker tersedia secara lokal."""
    path = Path(model_path)
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Mengunduh model MediaPipe HandLandmarker ke {path} ...")
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, str(path))
    print("[INFO] Model HandLandmarker berhasil diunduh.")
    return path


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class MediaPipeFingerCounter:
    """Penghitung jari berbasis 21 landmark tangan MediaPipe HandLandmarker."""

    def __init__(
        self,
        model_path: str | Path = "models/hand_landmarker.task",
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
    ) -> None:
        resolved_model = ensure_hand_landmarker_model(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(resolved_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, draw: bool = True) -> Tuple[int, List[HandResult]]:
        """Mendeteksi tangan dan menghitung jumlah jari yang terangkat pada frame BGR."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        hands_info: List[HandResult] = []
        total_fingers = 0

        if not result.hand_landmarks:
            return 0, hands_info

        for idx, landmarks in enumerate(result.hand_landmarks):
            pts = [(lm.x * w, lm.y * h) for lm in landmarks]
            fingers_up = self._count_fingers_from_landmarks(pts)
            count = sum(fingers_up)
            total_fingers += count

            xs = [int(p[0]) for p in pts]
            ys = [int(p[1]) for p in pts]
            x1, y1 = max(0, min(xs) - 15), max(0, min(ys) - 15)
            x2, y2 = min(w, max(xs) + 15), min(h, max(ys) + 15)

            label = "Tangan"
            if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                raw_name = result.handedness[idx][0].category_name
                label = "Kanan" if raw_name == "Right" else "Kiri"

            hands_info.append(
                HandResult(
                    label=label,
                    finger_count=count,
                    fingers_up=fingers_up,
                    bbox=(x1, y1, x2, y2),
                )
            )

            if draw:
                self._draw_hand(frame_bgr, pts, (x1, y1, x2, y2), label, count)

        return total_fingers, hands_info

    @staticmethod
    def _count_fingers_from_landmarks(pts: List[Tuple[float, float]]) -> List[bool]:
        """
        Menentukan status masing-masing dari 5 jari (terangkat atau terlipat)
        menggunakan jarak relatif terhadap pergelangan tangan (rotasi-invarian).
        """
        wrist = pts[0]
        pinky_mcp = pts[17]
        index_mcp = pts[5]

        # Ibu jari: bandingkan jarak ujung ibu jari (4) ke pangkal kelingking (17)
        # terhadap sendi IP (3) dan MCP (2) ibu jari.
        thumb_open = (
            _dist(pts[4], pinky_mcp) > _dist(pts[3], pinky_mcp) * 1.08
            and _dist(pts[4], index_mcp) > _dist(pts[3], index_mcp) * 1.05
        )

        # 4 jari lainnya: ujung jari (tip) lebih jauh dari pergelangan daripada sendi tengah (PIP)
        finger_indices = [
            (8, 6, 5),    # Telunjuk: TIP, PIP, MCP
            (12, 10, 9),  # Tengah
            (16, 14, 13), # Manis
            (20, 18, 17), # Kelingking
        ]
        other_fingers = []
        for tip, pip, mcp in finger_indices:
            is_open = (
                _dist(pts[tip], wrist) > _dist(pts[pip], wrist) * 1.05
                and _dist(pts[tip], pts[mcp]) > _dist(pts[pip], pts[mcp])
            )
            other_fingers.append(is_open)

        return [thumb_open] + other_fingers

    @staticmethod
    def _draw_hand(
        frame: np.ndarray,
        pts: List[Tuple[float, float]],
        bbox: Tuple[int, int, int, int],
        label: str,
        count: int,
    ) -> None:
        for start, end in HAND_CONNECTIONS:
            p1 = (int(pts[start][0]), int(pts[start][1]))
            p2 = (int(pts[end][0]), int(pts[end][1]))
            cv2.line(frame, p1, p2, (0, 220, 180), 2)

        for i, (px, py) in enumerate(pts):
            color = (0, 120, 255) if i in (4, 8, 12, 16, 20) else (255, 255, 255)
            radius = 6 if i in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, (int(px), int(py)), radius, color, -1)

        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 180), 2)
        cv2.putText(
            frame,
            f"{label}: {count} jari",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 180),
            2,
        )

    def close(self) -> None:
        self.landmarker.close()


class ContourFingerCounter:
    """
    Penghitung jari menggunakan Pengolahan Citra Klasik OpenCV:
    Segmentasi warna kulit (YCrCb) -> Morfologi -> Kontur -> Convex Hull & Convexity Defects.
    """

    def __init__(self, min_area: int = 4000) -> None:
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process(
        self,
        frame_bgr: np.ndarray,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        draw: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        Menghitung jari pada ROI (atau area tengah frame jika ROI tidak diberikan).
        Mengembalikan (jumlah_jari, mask_kulit).
        """
        h, w = frame_bgr.shape[:2]
        if roi_box is None:
            # Gunakan area ROI default di sisi kanan-tengah agar mudah mengarahkan tangan
            x1, y1 = int(w * 0.50), int(h * 0.15)
            x2, y2 = int(w * 0.95), int(h * 0.85)
        else:
            x1, y1, x2, y2 = roi_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return 0, np.zeros((100, 100), dtype=np.uint8)

        # 1. Segmentasi kulit pada ruang warna YCrCb
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # 2. Filtering & Operasi Morfologi untuk membersihkan noise
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if draw:
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(
                frame_bgr,
                "Area Deteksi (Contour)",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 180, 0),
                2,
            )

        if not contours:
            return 0, mask

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_area:
            return 0, mask

        hull_pts = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull_pts)
        solidity = area / hull_area if hull_area > 0 else 1.0

        hull_idx = cv2.convexHull(largest, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 4:
            return 0, mask

        defects = cv2.convexityDefects(largest, hull_idx)
        if defects is None:
            return 0, mask

        bx, by, bw, bh = cv2.boundingRect(largest)
        valid_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(largest[s][0])
            end = tuple(largest[e][0])
            far = tuple(largest[f][0])

            # Panjang sisi segitiga antar 2 ujung jari dan lembah (defect)
            a = _dist(start, end)
            b = _dist(start, far)
            c = _dist(end, far)
            if b * c == 0:
                continue

            cos_angle = max(-1.0, min(1.0, (b * b + c * c - a * a) / (2 * b * c)))
            angle = math.degrees(math.acos(cos_angle))
            depth = d / 256.0

            # Lembah antar jari yang valid memiliki sudut < 85 derajat dan kedalaman cukup
            if angle < 85.0 and depth > bh * 0.15:
                valid_defects += 1
                if draw:
                    cv2.circle(frame_bgr, (far[0] + x1, far[1] + y1), 6, (0, 0, 255), -1)

        if draw:
            shifted_hull = hull_pts + np.array([[[x1, y1]]])
            cv2.drawContours(frame_bgr, [shifted_hull], -1, (0, 255, 0), 2)

        if valid_defects == 0:
            # Bedakan kepalan tangan (0 jari) vs 1 jari terangkat dari rasio aspek & solidity
            aspect_ratio = bh / max(1, bw)
            finger_count = 1 if (solidity < 0.85 or aspect_ratio > 1.35) else 0
        else:
            finger_count = min(5, valid_defects + 1)

        return finger_count, mask
