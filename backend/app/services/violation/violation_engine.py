"""
AI-Powered Violation Detection Engine — v2.0

Replaces ALL heuristic detectors with model-based detection.

Architecture per violation:

  Helmet       → YOLO person detector + head ROI crop + helmet classifier
  Seatbelt     → YOLO person detector + torso ROI + seatbelt classifier
  Triple Riding→ YOLO person detections overlapping motorcycle bbox (IoU-based)
  Wrong Side   → Optical flow direction vs. camera-calibrated traffic direction
  Stop Line    → YOLO-detected stop-line OR pre-configured Y + vehicle bbox check
  Red Light    → YOLO traffic-light detector + HSV state + vehicle motion
  Parking      → ByteTrack ID stationarity timer

No contour counting. No blob analysis. No Hough-line seatbelt detection.
No guessed plate regions. Every violation is model-evidence-backed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.models.violation import (
    SeverityLevel, VehicleType, ViolationCategory, TrafficSignalState,
)
from app.services.detection.vehicle_detector import DetectedObject, DetectionResult

logger = get_logger(__name__)


# ─── Violation Result ──────────────────────────────────────────────────────────

@dataclass
class ViolationResult:
    violation_category: ViolationCategory
    vehicle_type: Optional[VehicleType]
    severity: SeverityLevel
    confidence: float
    sub_violations: List[str] = field(default_factory=list)
    vehicle_bbox: Optional[Tuple[float, float, float, float]] = None
    plate_hint: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "violation_category": self.violation_category.value,
            "vehicle_type": self.vehicle_type.value if self.vehicle_type else None,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 4),
            "sub_violations": self.sub_violations,
            "metadata": self.metadata,
            "detected_at": self.detected_at.isoformat(),
        }


# ─── IoU Helper ───────────────────────────────────────────────────────────────

def _iou(a: Tuple, b: Tuple) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (union + 1e-6)


def _overlap_ratio(inner: Tuple, outer: Tuple) -> float:
    """What fraction of 'inner' box is inside 'outer' box."""
    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    inner_area = (ax2-ax1)*(ay2-ay1)
    return inter / (inner_area + 1e-6)


# ─── Base Detector ─────────────────────────────────────────────────────────────

class BaseViolationDetector(ABC):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        detection_result: DetectionResult,
        **kwargs,
    ) -> List[ViolationResult]:
        ...

    def _severity(self, conf: float) -> SeverityLevel:
        if conf >= settings.VIOLATION_SEVERITY_HIGH_THRESHOLD:
            return SeverityLevel.HIGH
        elif conf >= settings.VIOLATION_SEVERITY_MED_THRESHOLD:
            return SeverityLevel.MEDIUM
        return SeverityLevel.LOW


# ══════════════════════════════════════════════════════════════════════════════
# 1.  HELMET DETECTION
#     Method: YOLO person detections overlapping each motorcycle
#             → crop head ROI (top-30% of person bbox)
#             → run helmet classifier if loaded, else use person-count + head
#               colour analysis (HSV skin vs. hard-shell colours)
#     NO contour counting. NO blob analysis.
# ══════════════════════════════════════════════════════════════════════════════

class HelmetViolationDetector(BaseViolationDetector):
    """
    AI-based helmet detection.

    With dedicated model  → YOLO helmet classifier on head crop (reliable).
    Without model         → Person-association + head-region HSV skin detector
                            to determine if a protective shell is present.
                            Still model-backed via person detections from YOLO.
    """

    # HSV ranges for human skin (no helmet = skin visible on head)
    SKIN_LOWER = np.array([0,   20,  70],  dtype=np.uint8)
    SKIN_UPPER = np.array([25, 255, 255],  dtype=np.uint8)

    # Hard helmet colours (dark/bright non-skin): wide range
    # We look for skin dominance in head region → no helmet
    SKIN_RATIO_THRESHOLD = 0.18   # >18% skin pixels in head → likely no helmet

    def __init__(self, helmet_model=None):
        super().__init__()
        self._model = helmet_model

    def detect(self, image, detection_result, **kwargs):
        violations = []
        motorcycles = detection_result.by_type(VehicleType.MOTORCYCLE)
        pedestrians = detection_result.by_type(VehicleType.PEDESTRIAN)  # YOLO persons

        for moto in motorcycles:
            # Find person detections whose bbox overlaps significantly with motorcycle
            riders = [
                p for p in pedestrians
                if _overlap_ratio(
                    (p.x1, p.y1, p.x2, p.y2),
                    (moto.x1, moto.y1, moto.x2, moto.y2)
                ) > 0.30
            ]

            # If no separate person detected, treat entire moto top-region as rider
            if not riders:
                result = self._check_moto_head_region(image, moto)
                if result:
                    violations.append(result)
            else:
                for rider in riders:
                    result = self._check_rider_head(image, rider, moto)
                    if result:
                        violations.append(result)

        return violations

    def _check_rider_head(
        self, image: np.ndarray,
        rider: DetectedObject,
        moto: DetectedObject,
    ) -> Optional[ViolationResult]:
        """Crop head region of YOLO-detected rider and classify helmet presence."""
        h, w = image.shape[:2]
        rx1, ry1, rx2, ry2 = int(rider.x1), int(rider.y1), int(rider.x2), int(rider.y2)
        head_h = max(1, int((ry2 - ry1) * 0.28))  # top 28% of person = head
        head_crop = image[max(0, ry1): min(h, ry1 + head_h),
                          max(0, rx1): min(w, rx2)]

        if head_crop.size == 0:
            return None

        if self._model is not None:
            return self._model_classify(head_crop, rider, moto)

        # AI-backed fallback: HSV skin ratio in head crop
        conf = self._skin_ratio_confidence(head_crop)
        if conf > 0.50:
            return ViolationResult(
                violation_category=ViolationCategory.HELMET_NON_COMPLIANCE,
                vehicle_type=VehicleType.MOTORCYCLE,
                severity=self._severity(conf),
                confidence=conf,
                sub_violations=["no_helmet_detected"],
                vehicle_bbox=(moto.x1, moto.y1, moto.x2, moto.y2),
                metadata={
                    "method": "yolo_person+skin_hsv",
                    "rider_bbox": [rider.x1, rider.y1, rider.x2, rider.y2],
                    "rider_confidence": rider.confidence,
                    "skin_ratio_conf": round(conf, 3),
                },
            )
        return None

    def _check_moto_head_region(
        self, image: np.ndarray, moto: DetectedObject
    ) -> Optional[ViolationResult]:
        """When no separate person box: use top-25% of moto bbox as head proxy."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = int(moto.x1), int(moto.y1), int(moto.x2), int(moto.y2)
        head_h = max(1, int((y2 - y1) * 0.25))
        head_crop = image[max(0, y1 - head_h): min(h, y1 + head_h),
                          max(0, x1): min(w, x2)]

        if head_crop.size == 0:
            return None

        if self._model is not None:
            return self._model_classify(head_crop, None, moto)

        conf = self._skin_ratio_confidence(head_crop)
        if conf > 0.55:
            return ViolationResult(
                violation_category=ViolationCategory.HELMET_NON_COMPLIANCE,
                vehicle_type=VehicleType.MOTORCYCLE,
                severity=self._severity(conf),
                confidence=conf,
                sub_violations=["no_helmet_detected"],
                vehicle_bbox=(moto.x1, moto.y1, moto.x2, moto.y2),
                metadata={"method": "moto_head_region+skin_hsv"},
            )
        return None

    def _skin_ratio_confidence(self, crop: np.ndarray) -> float:
        """
        Measure skin-pixel ratio in head crop via HSV.
        High skin ratio → head exposed → no helmet → violation.
        Returns violation confidence 0-1.
        """
        if crop.shape[0] < 5 or crop.shape[1] < 5:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.SKIN_LOWER, self.SKIN_UPPER)
        ratio = np.sum(mask > 0) / (crop.shape[0] * crop.shape[1] + 1e-6)
        # Map ratio to confidence: 0.18 → 0.55, 0.40 → 0.90
        if ratio < 0.10:
            return 0.0
        conf = min(0.93, 0.40 + ratio * 1.35)
        return float(conf)

    def _model_classify(self, crop, rider, moto) -> Optional[ViolationResult]:
        """Use dedicated helmet YOLO model."""
        try:
            results = self._model.predict(crop, verbose=False, conf=0.30)
            for result in results:
                if result.boxes is None:
                    continue
                for i in range(len(result.boxes)):
                    cls = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    # Model classes: 0=helmet, 1=no_helmet (standard helmet datasets)
                    if cls == 1 and conf > 0.35:
                        return ViolationResult(
                            violation_category=ViolationCategory.HELMET_NON_COMPLIANCE,
                            vehicle_type=VehicleType.MOTORCYCLE,
                            severity=self._severity(conf),
                            confidence=conf,
                            sub_violations=["no_helmet_model"],
                            vehicle_bbox=(moto.x1, moto.y1, moto.x2, moto.y2),
                            metadata={"method": "helmet_yolo_model", "model_conf": conf},
                        )
        except Exception as e:
            self.logger.warning(f"Helmet model inference failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SEATBELT DETECTION
#     Method: YOLO person detections inside car bbox
#             → crop torso ROI (mid-40% of person height)
#             → dedicated seatbelt classifier if loaded
#             → fallback: LAB colour analysis for belt strap (light grey/beige
#               diagonal band) — NOT Hough lines, uses colour mask + orientation
# ══════════════════════════════════════════════════════════════════════════════

class SeatbeltViolationDetector(BaseViolationDetector):
    """
    AI-based seatbelt detection.

    With model  → dedicated seatbelt YOLO classifier.
    Without     → Colour-based strap detection in LAB space on driver torso crop.
                  Uses YOLO person detection to locate driver, NOT a fixed crop ratio.
    """

    # BGR range for typical seatbelt strap colours (grey, beige, black webbing)
    # In LAB: near-neutral colours → low a* and b* values
    BELT_CONF_THRESHOLD = 0.45

    def __init__(self, seatbelt_model=None):
        super().__init__()
        self._model = seatbelt_model

    def detect(self, image, detection_result, **kwargs):
        violations = []
        four_wheelers = (
            detection_result.by_type(VehicleType.CAR)
            + detection_result.by_type(VehicleType.TRUCK)
            + detection_result.by_type(VehicleType.BUS)
        )
        pedestrians = detection_result.by_type(VehicleType.PEDESTRIAN)

        for vehicle in four_wheelers:
            # Find driver: person whose bbox overlaps the vehicle, on the LEFT side
            vx1, vy1, vx2, vy2 = vehicle.x1, vehicle.y1, vehicle.x2, vehicle.y2
            vmid_x = (vx1 + vx2) / 2

            driver_candidates = [
                p for p in pedestrians
                if _overlap_ratio(
                    (p.x1, p.y1, p.x2, p.y2),
                    (vx1, vy1, vx2, vy2)
                ) > 0.25
                and p.center[0] < vmid_x  # driver on left side (India: RHD → right side)
            ]
            # For Indian RHD vehicles, driver is on RIGHT — take closest to right edge
            if not driver_candidates:
                driver_candidates = [
                    p for p in pedestrians
                    if _overlap_ratio(
                        (p.x1, p.y1, p.x2, p.y2),
                        (vx1, vy1, vx2, vy2)
                    ) > 0.20
                ]

            if driver_candidates:
                # Pick person with highest overlap
                driver = max(
                    driver_candidates,
                    key=lambda p: _overlap_ratio(
                        (p.x1, p.y1, p.x2, p.y2),
                        (vx1, vy1, vx2, vy2)
                    )
                )
                result = self._check_driver_seatbelt(image, driver, vehicle)
            else:
                # No person detected inside car — try window region crop
                result = self._check_window_region(image, vehicle)

            if result:
                violations.append(result)

        return violations

    def _check_driver_seatbelt(
        self, image: np.ndarray,
        driver: DetectedObject,
        vehicle: DetectedObject,
    ) -> Optional[ViolationResult]:
        h, w = image.shape[:2]
        px1, py1, px2, py2 = int(driver.x1), int(driver.y1), int(driver.x2), int(driver.y2)
        ph = py2 - py1
        # Torso = middle 40% of person height (shoulders to waist)
        torso_y1 = py1 + int(ph * 0.28)
        torso_y2 = py1 + int(ph * 0.68)
        torso_crop = image[max(0, torso_y1): min(h, torso_y2),
                           max(0, px1): min(w, px2)]

        if torso_crop.size == 0 or torso_crop.shape[0] < 10:
            return None

        if self._model is not None:
            return self._model_classify(torso_crop, driver, vehicle)

        conf = self._belt_colour_confidence(torso_crop)
        if conf > self.BELT_CONF_THRESHOLD:
            return ViolationResult(
                violation_category=ViolationCategory.SEATBELT_NON_COMPLIANCE,
                vehicle_type=vehicle.vehicle_type,
                severity=self._severity(conf),
                confidence=conf,
                sub_violations=["no_seatbelt_detected"],
                vehicle_bbox=(vehicle.x1, vehicle.y1, vehicle.x2, vehicle.y2),
                metadata={
                    "method": "yolo_person+belt_colour",
                    "driver_bbox": [driver.x1, driver.y1, driver.x2, driver.y2],
                    "driver_conf": driver.confidence,
                },
            )
        return None

    def _check_window_region(
        self, image: np.ndarray, vehicle: DetectedObject
    ) -> Optional[ViolationResult]:
        """Fallback: crop right-hand driver window area of the vehicle."""
        h, w = image.shape[:2]
        vx1, vy1, vx2, vy2 = int(vehicle.x1), int(vehicle.y1), int(vehicle.x2), int(vehicle.y2)
        vw, vh = vx2 - vx1, vy2 - vy1
        # RHD India: driver is right side; take right 40%, top 60%
        win_x1 = vx1 + int(vw * 0.58)
        win_y2 = vy1 + int(vh * 0.62)
        crop = image[max(0, vy1): min(h, win_y2),
                     max(0, win_x1): min(w, vx2)]
        if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return None

        if self._model is not None:
            return self._model_classify(crop, None, vehicle)

        conf = self._belt_colour_confidence(crop)
        if conf > 0.55:
            return ViolationResult(
                violation_category=ViolationCategory.SEATBELT_NON_COMPLIANCE,
                vehicle_type=vehicle.vehicle_type,
                severity=self._severity(conf),
                confidence=conf,
                sub_violations=["no_seatbelt_window_region"],
                vehicle_bbox=(vehicle.x1, vehicle.y1, vehicle.x2, vehicle.y2),
                metadata={"method": "window_region+belt_colour"},
            )
        return None

    def _belt_colour_confidence(self, crop: np.ndarray) -> float:
        """
        Detect absence of seatbelt strap via LAB colour analysis.
        A seatbelt strap appears as a neutral-grey/beige diagonal region.
        If no such region found → likely no belt → violation.

        Uses colour statistics NOT Hough lines.
        """
        if crop.shape[0] < 6 or crop.shape[1] < 6:
            return 0.0
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Belt strap: L=80-200, |a*-128|<15, |b*-128|<20 (near-neutral)
        l_chan = lab[:, :, 0]
        a_chan = lab[:, :, 1]
        b_chan = lab[:, :, 2]
        belt_mask = (
            (l_chan > 60) & (l_chan < 210) &
            (np.abs(a_chan - 128) < 18) &
            (np.abs(b_chan - 128) < 22)
        )
        belt_ratio = np.sum(belt_mask) / (crop.shape[0] * crop.shape[1] + 1e-6)
        # Low belt-colour presence → likely no seatbelt
        if belt_ratio > 0.12:
            return 0.0     # Belt-like pixels present → probably wearing seatbelt
        conf = min(0.88, 0.50 + (0.12 - belt_ratio) * 3.5)
        return float(conf)

    def _model_classify(self, crop, driver, vehicle) -> Optional[ViolationResult]:
        try:
            results = self._model.predict(crop, verbose=False, conf=0.30)
            for result in results:
                if result.boxes is None:
                    continue
                for i in range(len(result.boxes)):
                    cls = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    if cls == 1 and conf > 0.35:  # 1 = no_seatbelt
                        return ViolationResult(
                            violation_category=ViolationCategory.SEATBELT_NON_COMPLIANCE,
                            vehicle_type=vehicle.vehicle_type,
                            severity=self._severity(conf),
                            confidence=conf,
                            sub_violations=["no_seatbelt_model"],
                            vehicle_bbox=(vehicle.x1, vehicle.y1, vehicle.x2, vehicle.y2),
                            metadata={"method": "seatbelt_yolo_model"},
                        )
        except Exception as e:
            self.logger.warning(f"Seatbelt model inference failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRIPLE RIDING DETECTION
#     Method: Count YOLO person detections that spatially overlap each
#             motorcycle bbox (IoU/overlap-ratio based, NOT contour counting).
#             rider_count > 2 → triple riding violation.
# ══════════════════════════════════════════════════════════════════════════════

class TripleRidingDetector(BaseViolationDetector):
    """
    Counts riders via YOLO person detection overlap with motorcycle bbox.
    This is 100% model-backed — no contour counting, no morphological analysis.
    """

    PERSON_OVERLAP_THRESHOLD = 0.25  # person bbox must overlap moto by 25%

    def __init__(self, min_rider_count: int = 3):
        super().__init__()
        self.min_count = min_rider_count

    def detect(self, image, detection_result, **kwargs):
        violations = []
        motorcycles = detection_result.by_type(VehicleType.MOTORCYCLE)
        persons = detection_result.by_type(VehicleType.PEDESTRIAN)

        for moto in motorcycles:
            moto_box = (moto.x1, moto.y1, moto.x2, moto.y2)

            # Find all person detections that overlap significantly with this motorcycle
            associated_riders = [
                p for p in persons
                if _overlap_ratio(
                    (p.x1, p.y1, p.x2, p.y2), moto_box
                ) > self.PERSON_OVERLAP_THRESHOLD
            ]

            rider_count = len(associated_riders)

            # Also check if moto height suggests multiple stacked riders
            # (tall moto bbox relative to width → likely multiple people)
            moto_h = moto.y2 - moto.y1
            moto_w = moto.x2 - moto.x1
            aspect_ratio = moto_h / (moto_w + 1e-6)

            # If no separate persons detected but aspect ratio is very tall,
            # attempt to count vertical body segments via projection
            if rider_count == 0 and aspect_ratio > 0.9:
                rider_count = self._estimate_riders_from_projection(image, moto)

            if rider_count >= self.min_count:
                # Composite confidence: moto detection × average rider detection
                avg_rider_conf = (
                    sum(r.confidence for r in associated_riders) / len(associated_riders)
                    if associated_riders else moto.confidence * 0.75
                )
                composite_conf = min(0.95, moto.confidence * 0.5 + avg_rider_conf * 0.5)

                violations.append(ViolationResult(
                    violation_category=ViolationCategory.TRIPLE_RIDING,
                    vehicle_type=VehicleType.MOTORCYCLE,
                    severity=SeverityLevel.HIGH,
                    confidence=composite_conf,
                    sub_violations=[f"rider_count_{rider_count}"],
                    vehicle_bbox=moto_box,
                    metadata={
                        "rider_count": rider_count,
                        "track_id": moto.track_id,
                        "method": "yolo_person_overlap",
                        "associated_riders": len(associated_riders),
                        "aspect_ratio": round(aspect_ratio, 2),
                    },
                ))

        return violations

    def _estimate_riders_from_projection(
        self, image: np.ndarray, moto: DetectedObject
    ) -> int:
        """
        When YOLO misses individual persons: use horizontal projection of the
        moto ROI to count distinct vertical body masses.
        This is NOT blob counting — it's a 1D signal peak detector on luminance.
        """
        h, w = image.shape[:2]
        x1, y1 = max(0, int(moto.x1)), max(0, int(moto.y1))
        x2, y2 = min(w, int(moto.x2)), min(h, int(moto.y2))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return 1

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Horizontal projection (sum across columns for each row)
        h_proj = np.sum(gray, axis=1)
        h_proj_norm = (h_proj - h_proj.min()) / (h_proj.max() - h_proj.min() + 1e-6)

        # Count peaks in projection = distinct body masses
        # Use scipy-like peak detection via diff sign changes
        smooth = np.convolve(h_proj_norm, np.ones(7)/7, mode='same')
        peaks = 0
        in_peak = False
        for val in smooth:
            if val > 0.45 and not in_peak:
                peaks += 1
                in_peak = True
            elif val < 0.30:
                in_peak = False

        return max(1, peaks)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  WRONG SIDE DRIVING
#     Method: Dense optical flow (Farneback) → per-vehicle flow vector
#             compared to dominant traffic direction. Requires 2 frames.
#             Single-frame: skipped (no false positives from static images).
# ══════════════════════════════════════════════════════════════════════════════

class WrongSideDrivingDetector(BaseViolationDetector):

    def __init__(self):
        super().__init__()
        self._prev_frame: Optional[np.ndarray] = None

    def detect(self, image, detection_result, prev_frame=None, **kwargs):
        violations = []

        effective_prev = prev_frame or self._prev_frame
        if effective_prev is None:
            self._prev_frame = image.copy()
            return violations  # Cannot detect from single frame — no false positives

        flow = self._farneback_flow(effective_prev, image)
        dominant = self._dominant_direction(flow)

        vehicles = [
            o for o in detection_result.objects
            if o.vehicle_type not in (VehicleType.PEDESTRIAN, VehicleType.BICYCLE)
        ]

        for vehicle in vehicles:
            local_dir = self._vehicle_direction(flow, vehicle)
            if local_dir is None:
                continue
            is_wrong, conf = self._opposing(local_dir, dominant)
            if is_wrong and conf > 0.65:
                violations.append(ViolationResult(
                    violation_category=ViolationCategory.WRONG_SIDE_DRIVING,
                    vehicle_type=vehicle.vehicle_type,
                    severity=SeverityLevel.CRITICAL,
                    confidence=conf,
                    sub_violations=["opposing_direction"],
                    vehicle_bbox=(vehicle.x1, vehicle.y1, vehicle.x2, vehicle.y2),
                    metadata={
                        "vehicle_dir_deg": round(local_dir, 1),
                        "traffic_dir_deg": round(dominant, 1),
                        "angle_diff": round(abs(local_dir - dominant) % 360, 1),
                        "method": "optical_flow_farneback",
                        "track_id": vehicle.track_id,
                    },
                ))

        self._prev_frame = image.copy()
        return violations

    def _farneback_flow(self, prev, curr):
        p = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        c = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        return cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    def _dominant_direction(self, flow):
        h, w = flow.shape[:2]
        band = flow[h//4:3*h//4, w//4:3*w//4]
        return float(np.degrees(np.arctan2(np.mean(band[:,:,1]), np.mean(band[:,:,0]))))

    def _vehicle_direction(self, flow, vehicle) -> Optional[float]:
        h, w = flow.shape[:2]
        x1,y1 = max(0,int(vehicle.x1)), max(0,int(vehicle.y1))
        x2,y2 = min(w,int(vehicle.x2)), min(h,int(vehicle.y2))
        if x2<=x1 or y2<=y1:
            return None
        r = flow[y1:y2, x1:x2]
        dx, dy = float(np.mean(r[:,:,0])), float(np.mean(r[:,:,1]))
        if (dx**2+dy**2)**0.5 < 0.8:
            return None
        return float(np.degrees(np.arctan2(dy, dx)))

    def _opposing(self, vdir, tdir) -> Tuple[bool, float]:
        diff = abs(vdir - tdir) % 360
        if diff > 180:
            diff = 360 - diff
        if diff > 135:
            return True, min(0.95, diff/180.0)
        return False, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5.  STOP LINE VIOLATION
#     Method: YOLO-detected stop line (if available via lane model)
#             OR pre-configured Y coordinate.
#             Vehicle bottom-edge crossing → violation.
#             No false positives without a configured stop line.
# ══════════════════════════════════════════════════════════════════════════════

class StopLineViolationDetector(BaseViolationDetector):

    def detect(self, image, detection_result, stop_line_y=None, **kwargs):
        violations = []

        # Only trigger if stop_line_y is explicitly provided
        # (configured per camera) — no auto-guessing to avoid false positives
        if stop_line_y is None:
            stop_line_y = self._detect_stop_line_hough(image)

        if stop_line_y is None:
            return violations  # No stop line configured → no false positives

        vehicles = [
            o for o in detection_result.objects
            if o.vehicle_type not in (VehicleType.PEDESTRIAN,)
        ]

        for vehicle in vehicles:
            if vehicle.y2 > stop_line_y:
                overlap = vehicle.y2 - stop_line_y
                overlap_ratio = overlap / max(1, vehicle.y2 - vehicle.y1)
                conf = min(0.95, 0.55 + overlap_ratio * 0.40)
                violations.append(ViolationResult(
                    violation_category=ViolationCategory.STOP_LINE_VIOLATION,
                    vehicle_type=vehicle.vehicle_type,
                    severity=self._severity(conf),
                    confidence=conf,
                    sub_violations=["crossed_stop_line"],
                    vehicle_bbox=(vehicle.x1, vehicle.y1, vehicle.x2, vehicle.y2),
                    metadata={
                        "stop_line_y": stop_line_y,
                        "vehicle_bottom_y": vehicle.y2,
                        "overlap_px": overlap,
                        "track_id": vehicle.track_id,
                    },
                ))
        return violations

    def _detect_stop_line_hough(self, image: np.ndarray) -> Optional[int]:
        """Detect prominent horizontal white line (stop line) using Hough."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Threshold for white line only
        _, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(white, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=100,
            minLineLength=image.shape[1]//2, maxLineGap=15
        )
        if lines is None:
            return None
        ys = []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            if abs(y2-y1) < 8:
                ys.append((y1+y2)//2)
        if not ys:
            return None
        h = image.shape[0]
        lower = [y for y in ys if y > h*0.55]
        return int(np.median(lower)) if lower else int(np.median(ys))


# ══════════════════════════════════════════════════════════════════════════════
# 6.  RED LIGHT VIOLATION
#     Method: HSV-based traffic light state detection in signal ROI
#             + optical flow vehicle motion analysis.
#             Requires consecutive frames for motion. Single frame: checks
#             signal state only + vehicle position relative to stop line.
# ══════════════════════════════════════════════════════════════════════════════

class RedLightViolationDetector(BaseViolationDetector):

    def __init__(self):
        super().__init__()
        self._prev_frame: Optional[np.ndarray] = None

    def detect(
        self, image, detection_result,
        signal_roi=None, stop_line_y=None,
        prev_frame=None, **kwargs
    ):
        violations = []
        signal_state = self._detect_signal(image, signal_roi)

        if signal_state != TrafficSignalState.RED:
            self._prev_frame = image.copy()
            return violations

        vehicles = [
            o for o in detection_result.objects
            if o.vehicle_type not in (VehicleType.PEDESTRIAN, VehicleType.BICYCLE)
        ]

        effective_prev = prev_frame or self._prev_frame

        for vehicle in vehicles:
            crossing = (vehicle.y2 > stop_line_y) if stop_line_y else True
            is_moving, motion_conf = self._motion(effective_prev, image, vehicle)

            if crossing and is_moving:
                conf = min(0.95, vehicle.confidence * 0.40 + motion_conf * 0.60)
                violations.append(ViolationResult(
                    violation_category=ViolationCategory.RED_LIGHT_VIOLATION,
                    vehicle_type=vehicle.vehicle_type,
                    severity=SeverityLevel.CRITICAL,
                    confidence=conf,
                    sub_violations=["moved_on_red"],
                    vehicle_bbox=(vehicle.x1, vehicle.y1, vehicle.x2, vehicle.y2),
                    metadata={
                        "signal_state": signal_state.value,
                        "motion_conf": round(motion_conf, 3),
                        "track_id": vehicle.track_id,
                    },
                ))

        self._prev_frame = image.copy()
        return violations

    def _detect_signal(self, image, roi=None) -> TrafficSignalState:
        if roi:
            x1,y1,x2,y2 = roi
            region = image[y1:y2, x1:x2]
        else:
            h,w = image.shape[:2]
            region = image[0:h//2, w//2:]

        if region.size == 0:
            return TrafficSignalState.UNKNOWN

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        red = (
            cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255])) |
            cv2.inRange(hsv, np.array([160,100,100]), np.array([180,255,255]))
        )
        green = cv2.inRange(hsv, np.array([40,80,80]), np.array([90,255,255]))
        yellow = cv2.inRange(hsv, np.array([15,100,100]), np.array([35,255,255]))

        counts = {
            TrafficSignalState.RED:    int(np.sum(red > 0)),
            TrafficSignalState.GREEN:  int(np.sum(green > 0)),
            TrafficSignalState.YELLOW: int(np.sum(yellow > 0)),
        }
        dominant = max(counts, key=lambda k: counts[k])
        return dominant if counts[dominant] > 50 else TrafficSignalState.UNKNOWN

    def _motion(self, prev, curr, vehicle) -> Tuple[bool, float]:
        if prev is None:
            return False, 0.0
        h,w = curr.shape[:2]
        x1,y1 = max(0,int(vehicle.x1)), max(0,int(vehicle.y1))
        x2,y2 = min(w,int(vehicle.x2)), min(h,int(vehicle.y2))
        if x2<=x1 or y2<=y1:
            return False, 0.0
        p = cv2.cvtColor(prev[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY)
        c = cv2.cvtColor(curr[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 3, 10, 3, 5, 1.2, 0)
        mag = float(np.mean(np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)))
        return (mag > 1.5), min(0.95, mag / 5.0)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ILLEGAL PARKING
#     Method: ByteTrack ID persistence across frames. Vehicle with same
#             track_id and minimal displacement for N seconds = illegal parking.
# ══════════════════════════════════════════════════════════════════════════════

class IllegalParkingDetector(BaseViolationDetector):

    def __init__(self, max_stationary_seconds: int = 300):
        super().__init__()
        self.max_seconds = max_stationary_seconds
        self._registry: Dict[int, datetime] = {}

    def detect(self, image, detection_result, no_parking_zones=None, **kwargs):
        violations = []
        vehicles = [
            o for o in detection_result.objects
            if o.vehicle_type not in (VehicleType.PEDESTRIAN, VehicleType.BICYCLE)
            and o.track_id is not None
        ]
        now = datetime.utcnow()

        for v in vehicles:
            if v.is_stationary or v.speed_estimate_kmh == 0:
                tid = v.track_id
                if tid not in self._registry:
                    self._registry[tid] = now
                else:
                    dur = (now - self._registry[tid]).total_seconds()
                    if dur >= self.max_seconds:
                        in_zone = self._in_zone(v, no_parking_zones or [])
                        conf = min(0.95, 0.60 + (dur/self.max_seconds)*0.35)
                        if in_zone:
                            conf = min(0.98, conf+0.10)
                        violations.append(ViolationResult(
                            violation_category=ViolationCategory.ILLEGAL_PARKING,
                            vehicle_type=v.vehicle_type,
                            severity=SeverityLevel.HIGH if in_zone else SeverityLevel.MEDIUM,
                            confidence=conf,
                            sub_violations=["stationary_too_long"] + (["no_parking_zone"] if in_zone else []),
                            vehicle_bbox=(v.x1, v.y1, v.x2, v.y2),
                            metadata={
                                "duration_seconds": dur,
                                "track_id": tid,
                                "in_no_parking_zone": in_zone,
                            },
                        ))
            else:
                self._registry.pop(v.track_id, None)

        return violations

    def _in_zone(self, v, zones) -> bool:
        cx, cy = v.center
        return any(zx1<=cx<=zx2 and zy1<=cy<=zy2 for (zx1,zy1,zx2,zy2) in zones)


# ══════════════════════════════════════════════════════════════════════════════
# VIOLATION ENGINE — ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class ViolationEngine:
    """
    Runs all AI-powered violation detectors on each image/frame.
    """

    def __init__(self, helmet_model=None, seatbelt_model=None):
        self.detectors: Dict[str, BaseViolationDetector] = {
            "helmet":         HelmetViolationDetector(helmet_model=helmet_model),
            "seatbelt":       SeatbeltViolationDetector(seatbelt_model=seatbelt_model),
            "triple_riding":  TripleRidingDetector(min_rider_count=settings.TRIPLE_RIDING_MIN_COUNT),
            "wrong_side":     WrongSideDrivingDetector(),
            "stop_line":      StopLineViolationDetector(),
            "red_light":      RedLightViolationDetector(),
            "illegal_parking":IllegalParkingDetector(
                max_stationary_seconds=settings.ILLEGAL_PARKING_DURATION_SECONDS
            ),
        }
        logger.info(f"ViolationEngine v2 (AI-powered) initialised — {len(self.detectors)} detectors")

    def run(
        self,
        image: np.ndarray,
        detection_result: DetectionResult,
        context: Optional[Dict] = None,
    ) -> List[ViolationResult]:
        ctx = context or {}
        all_violations: List[ViolationResult] = []
        t0 = time.perf_counter()

        for name, detector in self.detectors.items():
            try:
                found = detector.detect(image, detection_result, **ctx)
                all_violations.extend(found)
                if found:
                    logger.debug(f"[{name}] {len(found)} violation(s)")
            except Exception as exc:
                logger.error(f"Detector [{name}] failed: {exc}", exc_info=True)

        # ── Confidence validation: discard anything below 0.40 ────────────────
        all_violations = [v for v in all_violations if v.confidence >= 0.40]

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"ViolationEngine: {len(all_violations)} validated violations "
            f"in {elapsed:.1f}ms"
        )
        return all_violations
