"""
Evidence Image Generation Service — v2.0

Fixes:
- Canvas expansion (info bar / summary panel) now done LAST so all
  bounding box coordinates remain correct throughout drawing.
- Violation bbox coords are offset-corrected after bar is prepended.
- Works even when violations=[] (still draws vehicles + timestamp).
- Robust path handling for Docker volume mounts.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.services.detection.vehicle_detector import DetectedObject, DetectionResult
from app.services.ocr.plate_recognizer import PlateResult
from app.services.violation.violation_engine import ViolationResult

logger = get_logger(__name__)

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
C_GREEN    = (0,   200,   0)
C_RED      = (0,     0, 220)
C_AMBER    = (0,   180, 220)
C_BLUE     = (220,  80,   0)
C_DARK     = (25,   25,  25)
C_WHITE    = (255, 255, 255)
C_YELLOW   = (0,   220, 220)
C_PURPLE   = (200,  60, 200)

SEVERITY_COLORS = {
    "low":      (0, 200, 255),
    "medium":   (0, 165, 255),
    "high":     (0,   0, 255),
    "critical": (0,   0, 180),
}

VEHICLE_COLORS = {
    "car":          C_GREEN,
    "truck":        (255, 128,   0),
    "bus":          (  0, 128, 255),
    "motorcycle":   (255,   0,   0),
    "bicycle":      (  0, 255, 255),
    "auto_rickshaw":(255, 255,   0),
    "pedestrian":   C_PURPLE,
    "unknown":      (128, 128, 128),
}


@dataclass
class EvidencePackage:
    session_id: str
    evidence_image_path: str
    thumbnail_path: str
    generated_at: datetime
    annotation_count: int


class EvidenceGenerator:

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.52
    THICK      = 2
    PAD        = 4
    INFO_H     = 38   # top info bar height in pixels
    LINE_H     = 22   # pixels per line in bottom panel

    def __init__(self):
        Path(settings.EVIDENCE_DIR).mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        image: np.ndarray,
        detection_result: DetectionResult,
        violations: List[ViolationResult],
        plate_results: Optional[List[PlateResult]] = None,
        session_id: Optional[str] = None,
        location_label: Optional[str] = None,
        camera_id: Optional[str] = None,
    ) -> EvidencePackage:
        sid = session_id or str(uuid.uuid4())
        t0  = time.perf_counter()

        # Work on an RGB copy (keeps original untouched)
        canvas = image.copy()

        # ── Step 1: Draw everything on the base image (correct coords) ────────
        self._draw_all_vehicles(canvas, detection_result.objects, violations)
        self._draw_all_violations(canvas, violations)
        if plate_results:
            for plate in plate_results:
                if plate.bbox:
                    self._draw_plate(canvas, plate)
        self._draw_timestamp(canvas)

        # ── Step 2: Prepend info bar (shifts image DOWN by INFO_H px) ─────────
        canvas = self._prepend_info_bar(canvas, location_label, camera_id)

        # ── Step 3: Append summary panel at bottom ────────────────────────────
        if violations:
            canvas = self._append_summary(canvas, violations)

        # ── Step 4: Save ──────────────────────────────────────────────────────
        Path(settings.EVIDENCE_DIR).mkdir(parents=True, exist_ok=True)
        ev_name  = f"evidence_{sid}_{int(time.time())}.jpg"
        ev_path  = Path(settings.EVIDENCE_DIR) / ev_name
        th_path  = Path(settings.EVIDENCE_DIR) / f"thumb_{sid}_{int(time.time())}.jpg"

        success = cv2.imwrite(
            str(ev_path), canvas,
            [cv2.IMWRITE_JPEG_QUALITY, settings.EVIDENCE_JPEG_QUALITY]
        )
        if not success:
            logger.error(f"cv2.imwrite failed for {ev_path}")

        # Thumbnail — 480px wide
        th_w = 480
        th_h = int(canvas.shape[0] * th_w / canvas.shape[1])
        thumb = cv2.resize(canvas, (th_w, th_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(th_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 82])

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Evidence saved: {ev_path.name} "
            f"[{canvas.shape[1]}x{canvas.shape[0]}px, {elapsed:.0f}ms]"
        )

        return EvidencePackage(
            session_id=sid,
            evidence_image_path=str(ev_path),
            thumbnail_path=str(th_path),
            generated_at=datetime.utcnow(),
            annotation_count=len(violations),
        )

    # ── Drawing on base image (before bar prepend) ────────────────────────────

    def _draw_all_vehicles(
        self,
        img: np.ndarray,
        objects: List[DetectedObject],
        violations: List[ViolationResult],
    ):
        """Draw every detected object with colour-coded bbox."""
        violation_boxes = set()
        for v in violations:
            if v.vehicle_bbox:
                # Mark bbox as having a violation (rounded to avoid float drift)
                key = (round(v.vehicle_bbox[0]), round(v.vehicle_bbox[1]))
                violation_boxes.add(key)

        for obj in objects:
            x1, y1, x2, y2 = int(obj.x1), int(obj.y1), int(obj.x2), int(obj.y2)
            vname  = obj.vehicle_type.value
            color  = VEHICLE_COLORS.get(vname, (128, 128, 128))

            # Red border if this vehicle has a violation
            is_viol = (round(obj.x1), round(obj.y1)) in violation_boxes
            draw_color = C_RED if is_viol else color
            thickness  = 3 if is_viol else self.THICK

            cv2.rectangle(img, (x1, y1), (x2, y2), draw_color, thickness)

            label = f"{vname.replace('_',' ').title()} {obj.confidence:.0%}"
            if obj.track_id is not None:
                label = f"#{obj.track_id} {label}"
            self._label(img, label, x1, y1, draw_color)

    def _draw_all_violations(self, img: np.ndarray, violations: List[ViolationResult]):
        """Draw violation badge below each violating vehicle."""
        for v in violations:
            if v.vehicle_bbox is None:
                continue
            x1, y1, x2, y2 = [int(c) for c in v.vehicle_bbox]
            # Extra thick red box
            cv2.rectangle(img, (x1, y1), (x2, y2), C_RED, 3)
            # Small corner marker
            m = 12
            cv2.line(img, (x1, y1), (x1+m, y1), C_RED, 4)
            cv2.line(img, (x1, y1), (x1, y1+m), C_RED, 4)
            cv2.line(img, (x2, y2), (x2-m, y2), C_RED, 4)
            cv2.line(img, (x2, y2), (x2, y2-m), C_RED, 4)

            cat   = v.violation_category.value.replace("_", " ").upper()
            badge = f"[!] {cat} {v.confidence:.0%}"
            sc    = SEVERITY_COLORS.get(v.severity.value, C_RED)
            # Draw badge below the box
            self._label(img, badge, x1, y2 + 20, sc, scale=0.48, thick=1)

    def _draw_plate(self, img: np.ndarray, plate: PlateResult):
        x1, y1, x2, y2 = [int(c) for c in plate.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), C_AMBER, 2)
        txt = plate.display_text
        if plate.ocr_confidence > 0:
            txt += f" {plate.ocr_confidence:.0%}"
        self._label(img, txt, x1, max(y1-4, 14), C_AMBER, scale=0.48)

    def _draw_timestamp(self, img: np.ndarray):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        h, w = img.shape[:2]
        (tw, _), _ = cv2.getTextSize(ts, self.FONT, 0.42, 1)
        # Semi-transparent black background for readability
        cv2.rectangle(img, (w-tw-14, h-22), (w-2, h-2), (0,0,0), cv2.FILLED)
        cv2.putText(img, ts, (w-tw-10, h-7),
                    self.FONT, 0.42, C_WHITE, 1, cv2.LINE_AA)

    # ── Canvas composition helpers ────────────────────────────────────────────

    def _prepend_info_bar(
        self, img: np.ndarray,
        location: Optional[str],
        camera_id: Optional[str],
    ) -> np.ndarray:
        """Prepend a dark info bar above the image."""
        h, w = img.shape[:2]
        bar = np.full((self.INFO_H, w, 3), C_DARK, dtype=np.uint8)

        parts = ["AI TRAFFIC VIOLATION DETECTION SYSTEM"]
        if location:  parts.append(f"| {location}")
        if camera_id: parts.append(f"| CAM: {camera_id}")
        text = "  ".join(parts)

        cv2.putText(bar, text, (10, 26),
                    self.FONT, 0.50, C_YELLOW, 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    def _append_summary(
        self, img: np.ndarray, violations: List[ViolationResult]
    ) -> np.ndarray:
        """Append a dark summary panel below the image."""
        w   = img.shape[1]
        n   = len(violations)
        ph  = max(44, self.LINE_H * (n + 1) + 10)
        panel = np.full((ph, w, 3), C_DARK, dtype=np.uint8)

        cv2.putText(panel,
                    f"VIOLATIONS DETECTED: {n}",
                    (10, 20), self.FONT, 0.52, C_YELLOW, 1, cv2.LINE_AA)

        for i, v in enumerate(violations):
            plate_str = f" | Plate: {v.plate_hint}" if v.plate_hint else ""
            line = (
                f"  [{i+1}] "
                f"{v.violation_category.value.replace('_',' ').upper()} "
                f"| {v.severity.value.upper()} "
                f"| {v.confidence:.0%}"
                f"{plate_str}"
            )
            cv2.putText(panel, line,
                        (10, 20 + self.LINE_H * (i + 1)),
                        self.FONT, 0.42, C_WHITE, 1, cv2.LINE_AA)

        return np.vstack([img, panel])

    # ── Text helper ───────────────────────────────────────────────────────────

    def _label(
        self,
        img: np.ndarray,
        text: str,
        x: int, y: int,
        color: Tuple[int, int, int],
        scale: float = 0.52,
        thick: int = 1,
    ):
        """Draw text with solid dark background box."""
        h, w = img.shape[:2]
        (tw, th), bl = cv2.getTextSize(text, self.FONT, scale, thick)
        # Clamp to image bounds
        x = max(0, min(x, w - tw - self.PAD * 2 - 2))
        y = max(th + self.PAD + 2, min(y, h - bl - 2))
        # Background
        cv2.rectangle(
            img,
            (x, y - th - self.PAD),
            (x + tw + self.PAD * 2, y + bl + self.PAD),
            (0, 0, 0), cv2.FILLED,
        )
        cv2.putText(img, text, (x + self.PAD, y),
                    self.FONT, scale, color, thick, cv2.LINE_AA)
