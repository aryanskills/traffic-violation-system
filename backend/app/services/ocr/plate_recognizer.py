"""
License Plate Recognition Service — v2.0

Pipeline:
  Vehicle bbox
  → Dedicated YOLO plate detector (if loaded)
  → Fallback: bottom-region crop with white-mask plate localisation
  → Super-resolution upscale (4×)
  → CLAHE + sharpening
  → EasyOCR primary, PaddleOCR secondary (ensemble if both loaded)
  → Indian plate regex validation + common OCR error correction
  → Confidence scoring

No guessed fixed-ratio plate regions as primary method.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.services.detection.vehicle_detector import DetectedObject

logger = get_logger(__name__)

# ─── Indian Plate Patterns ─────────────────────────────────────────────────────
PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$"),     # MH12AB1234
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$"),             # DL01CA9999
    re.compile(r"^\d{2}BH\d{4}[A-Z]{2}$"),                    # 22BH0001AA
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]\d{4}$"),                 # GJ05B4567
]

# Common OCR errors for Indian plates
OCR_CORRECTIONS = {
    "0": "O", "1": "I", "5": "S", "8": "B", "6": "G",
}


def normalize_plate(raw: str) -> str:
    text = re.sub(r"[^A-Z0-9]", "", raw.upper().strip())
    if len(text) >= 2:
        prefix = "".join(OCR_CORRECTIONS.get(c, c) for c in text[:2])
        text = prefix + text[2:]
    return text


def validate_plate(text: str) -> bool:
    cleaned = normalize_plate(text)
    return any(p.match(cleaned) for p in PLATE_PATTERNS)


@dataclass
class PlateResult:
    raw_text: Optional[str]
    normalized_text: Optional[str]
    is_valid_format: bool
    detection_confidence: float
    ocr_confidence: float
    bbox: Optional[Tuple[float, float, float, float]]
    ocr_engine: str = "none"

    @property
    def display_text(self) -> str:
        return self.normalized_text or self.raw_text or "UNKNOWN"


class LicensePlateRecognizer:
    """
    Two-stage plate recognition.
    Stage 1: Plate localisation (YOLO model → white-mask fallback)
    Stage 2: Enhance → EasyOCR / PaddleOCR ensemble → validate
    """

    def __init__(self, plate_model=None, ocr_engine: Optional[str] = None):
        self._plate_model = plate_model
        self._engine_name = ocr_engine or settings.OCR_ENGINE
        self._easy_ocr = None
        self._paddle_ocr = None
        self._ocr_loaded = False
        pass

    def _load_ocr(self):
        loaded = [] 
        # Try EasyOCR
        try:
            import easyocr
            self._easy_ocr = easyocr.Reader(
                ["en"], gpu=(settings.YOLO_DEVICE == "cuda"), verbose=False
            )
            loaded.append("easyocr")
        except Exception as e:
            logger.warning(f"EasyOCR not available: {e}")

        # Try PaddleOCR as secondary
        if self._engine_name == "paddleocr" or (not loaded):
            try:
                from paddleocr import PaddleOCR
                self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                loaded.append("paddleocr")
            except Exception as e:
                logger.warning(f"PaddleOCR not available: {e}")

        if loaded:
            self._ocr_loaded = True
            logger.info(f"OCR engines loaded: {loaded}")
        else:
            logger.error("No OCR engine available")

    # ─── Public ────────────────────────────────────────────────────────────────

    def recognize(self, image: np.ndarray, vehicle: Optional[DetectedObject] = None) -> PlateResult:

        if not self._ocr_loaded:
            logger.info("Loading OCR models on first request...")
            self._load_ocr()

        t0 = time.perf_counter()

            # Stage 1: Locate plate
        plate_crop, bbox, det_conf = self._locate_plate(image, vehicle)

        if plate_crop is None:
                return PlateResult(None, None, False, 0.0, 0.0, None)

        # Stage 2: Enhance
        enhanced = self._enhance(plate_crop)

        # Stage 3: OCR (ensemble)
        raw_text, ocr_conf, engine_used = self._ocr_ensemble(enhanced)

        normalized = normalize_plate(raw_text) if raw_text else None
        valid = validate_plate(raw_text) if raw_text else False

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"Plate OCR: '{normalized}' valid={valid} conf={ocr_conf:.2f} [{elapsed:.0f}ms]")

        return PlateResult(
            raw_text=raw_text,
            normalized_text=normalized,
            is_valid_format=valid,
            detection_confidence=det_conf,
            ocr_confidence=ocr_conf,
            bbox=bbox,
            ocr_engine=engine_used,
        )

    def recognize_batch(self, image, vehicles) -> List[PlateResult]:
        return [self.recognize(image, v) for v in vehicles]

    # ─── Stage 1: Localisation ─────────────────────────────────────────────────

    def _locate_plate(self, image, vehicle):
        # Prefer YOLO plate model
        if self._plate_model is not None:
            result = self._yolo_locate(image, vehicle)
            if result[0] is not None:
                return result

        # Fallback: white-rectangle detector on vehicle ROI
        if vehicle is not None:
            return self._white_rect_locate(image, vehicle)

        return None, None, 0.0

    def _yolo_locate(self, image, vehicle):
        roi, offset = image, (0, 0)
        if vehicle:
            x1,y1,x2,y2 = int(vehicle.x1),int(vehicle.y1),int(vehicle.x2),int(vehicle.y2)
            roi = image[y1:y2, x1:x2]
            offset = (x1, y1)

        best_conf, best_box = 0.0, None
        try:
            results = self._plate_model.predict(roi, verbose=False, conf=0.25)
            for r in results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    conf = float(r.boxes.conf[i])
                    if conf > best_conf:
                        best_conf = conf
                        best_box = r.boxes.xyxy[i].cpu().numpy()
        except Exception as e:
            logger.warning(f"YOLO plate detect failed: {e}")
            return None, None, 0.0

        if best_box is None or best_conf < 0.25:
            return None, None, 0.0

        ox, oy = offset
        x1,y1,x2,y2 = int(best_box[0])+ox, int(best_box[1])+oy, int(best_box[2])+ox, int(best_box[3])+oy
        crop = image[y1:y2, x1:x2]
        return (crop if crop.size > 0 else None), (x1,y1,x2,y2), best_conf

    def _white_rect_locate(self, image, vehicle):
        """
        Find license plate as the most prominent white rectangular region
        in the lower 30% of the vehicle bounding box.
        Uses colour + morphology, NOT a guessed fixed crop.
        """
        h_img, w_img = image.shape[:2]
        vx1,vy1,vx2,vy2 = int(vehicle.x1),int(vehicle.y1),int(vehicle.x2),int(vehicle.y2)
        vh = vy2 - vy1

        # Search only in lower 35% of vehicle (where plate is)
        search_y1 = vy1 + int(vh * 0.65)
        roi = image[search_y1:vy2, vx1:vx2]
        if roi.size == 0:
            return None, None, 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Threshold for bright regions (plate is white/yellow)
        _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / (ch + 1e-6)
            area = cw * ch
            # Typical plate: aspect 2–5, area > 200px²
            if 1.8 < aspect < 6.0 and area > 200 and ch > 8:
                score = area * min(aspect / 3.5, 1.0)
                if score > best_score:
                    best_score = score
                    best = (x + vx1, y + search_y1, x + cw + vx1, y + ch + search_y1)

        if best is None:
            return None, None, 0.0

        bx1, by1, bx2, by2 = best
        crop = image[max(0,by1):min(h_img,by2), max(0,bx1):min(w_img,bx2)]
        return (crop if crop.size > 0 else None), best, 0.55

    # ─── Stage 2: Enhancement ──────────────────────────────────────────────────

    def _enhance(self, plate: np.ndarray) -> np.ndarray:
        """4× upscale → CLAHE → unsharp mask → denoise."""
        target_h = 80
        h, w = plate.shape[:2]
        scale = max(target_h / h, 1.0)
        new_w = max(int(w * scale), 200)
        new_h = max(int(h * scale), target_h)
        up = cv2.resize(plate, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        eq = clahe.apply(gray)

        # Unsharp mask
        blur = cv2.GaussianBlur(eq, (0, 0), 2.0)
        sharp = cv2.addWeighted(eq, 1.6, blur, -0.6, 0)
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)

        denoised = cv2.fastNlMeansDenoising(sharp, h=8)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    # ─── Stage 3: OCR Ensemble ─────────────────────────────────────────────────

    def _ocr_ensemble(self, plate_img) -> Tuple[Optional[str], float, str]:
        """Run available OCR engines and return best result."""
        candidates = []

        if self._easy_ocr:
            try:
                results = self._easy_ocr.readtext(
                    plate_img,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -",
                    detail=1,
                )
                for (_, text, conf) in results:
                    clean = re.sub(r"[^A-Z0-9]", "", text.upper())
                    if len(clean) >= 4:
                        candidates.append((clean, float(conf), "easyocr"))
            except Exception as e:
                logger.debug(f"EasyOCR error: {e}")

        if self._paddle_ocr:
            try:
                result = self._paddle_ocr.ocr(plate_img, cls=True)
                if result and result[0]:
                    for line in result[0]:
                        text, conf = line[1][0], line[1][1]
                        clean = re.sub(r"[^A-Z0-9]", "", text.upper())
                        if len(clean) >= 4:
                            candidates.append((clean, float(conf), "paddleocr"))
            except Exception as e:
                logger.debug(f"PaddleOCR error: {e}")

        if not candidates:
            return None, 0.0, "none"

        # Prefer validated plates, then highest confidence
        validated = [c for c in candidates if validate_plate(c[0])]
        if validated:
            best = max(validated, key=lambda x: x[1])
        else:
            best = max(candidates, key=lambda x: x[1])

        return best[0], best[1], best[2]
