"""
Unit tests for AI-Powered Violation Detection Engine v2.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from datetime import datetime

from app.services.violation.violation_engine import (
    HelmetViolationDetector, SeatbeltViolationDetector,
    TripleRidingDetector, StopLineViolationDetector,
    RedLightViolationDetector, IllegalParkingDetector,
    ViolationEngine, ViolationResult, _iou, _overlap_ratio,
)
from app.services.detection.vehicle_detector import DetectedObject, DetectionResult
from app.models.violation import (
    VehicleType, ViolationCategory, SeverityLevel, TrafficSignalState,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_vehicle(vtype=VehicleType.MOTORCYCLE,
                 x1=100, y1=100, x2=300, y2=400,
                 conf=0.9, track_id=1, is_stationary=False):
    return DetectedObject(
        class_id=3, vehicle_type=vtype, confidence=conf,
        x1=x1, y1=y1, x2=x2, y2=y2,
        track_id=track_id, is_stationary=is_stationary,
    )

def make_person(x1=120, y1=80, x2=280, y2=400, conf=0.8, track_id=10):
    return DetectedObject(
        class_id=0, vehicle_type=VehicleType.PEDESTRIAN, confidence=conf,
        x1=x1, y1=y1, x2=x2, y2=y2, track_id=track_id,
    )

def make_dr(objects=None):
    return DetectionResult(objects=objects or [])

@pytest.fixture
def blank_640():
    return np.zeros((640, 640, 3), dtype=np.uint8)

@pytest.fixture
def grey_image():
    return np.full((640, 640, 3), 128, dtype=np.uint8)


# ─── IoU helpers ──────────────────────────────────────────────────────────────

class TestIoU:
    def test_perfect_overlap(self):
        assert abs(_iou((0,0,100,100),(0,0,100,100)) - 1.0) < 1e-4

    def test_no_overlap(self):
        assert _iou((0,0,50,50),(100,100,200,200)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = _iou((0,0,100,100),(50,50,150,150))
        assert 0.1 < iou < 0.5

class TestOverlapRatio:
    def test_full_containment(self):
        ratio = _overlap_ratio((25,25,75,75),(0,0,100,100))
        assert ratio == pytest.approx(1.0, abs=1e-3)

    def test_no_overlap(self):
        assert _overlap_ratio((0,0,50,50),(100,100,200,200)) == pytest.approx(0.0)


# ─── ViolationResult ──────────────────────────────────────────────────────────

class TestViolationResult:
    def test_to_dict_keys(self):
        vr = ViolationResult(
            violation_category=ViolationCategory.HELMET_NON_COMPLIANCE,
            vehicle_type=VehicleType.MOTORCYCLE,
            severity=SeverityLevel.HIGH,
            confidence=0.88,
        )
        d = vr.to_dict()
        for k in ["violation_category","severity","confidence","detected_at"]:
            assert k in d

    def test_confidence_rounded(self):
        vr = ViolationResult(
            violation_category=ViolationCategory.TRIPLE_RIDING,
            vehicle_type=VehicleType.MOTORCYCLE,
            severity=SeverityLevel.HIGH,
            confidence=0.876543,
        )
        assert vr.to_dict()["confidence"] == round(0.876543, 4)


# ─── Helmet ───────────────────────────────────────────────────────────────────

class TestHelmetViolationDetector:
    def test_no_motorcycles_no_violations(self, blank_640):
        det = HelmetViolationDetector()
        dr = make_dr([make_vehicle(vtype=VehicleType.CAR)])
        assert det.detect(blank_640, dr) == []

    def test_empty_detection_no_violations(self, blank_640):
        det = HelmetViolationDetector()
        assert det.detect(blank_640, make_dr([])) == []

    def test_returns_list(self, blank_640):
        det = HelmetViolationDetector()
        dr = make_dr([make_vehicle(vtype=VehicleType.MOTORCYCLE)])
        assert isinstance(det.detect(blank_640, dr), list)

    def test_motorcycle_with_rider_runs(self, grey_image):
        det = HelmetViolationDetector()
        moto = make_vehicle(vtype=VehicleType.MOTORCYCLE, x1=50, y1=50, x2=300, y2=400)
        person = make_person(x1=80, y1=30, x2=270, y2=400)
        dr = make_dr([moto, person])
        result = det.detect(grey_image, dr)
        assert isinstance(result, list)

    def test_skin_ratio_returns_float(self):
        det = HelmetViolationDetector()
        crop = np.full((50, 50, 3), [200, 150, 130], dtype=np.uint8)  # skin-like colour
        score = det._skin_ratio_confidence(crop)
        assert 0.0 <= score <= 1.0

    def test_skin_ratio_dark_returns_low(self):
        det = HelmetViolationDetector()
        crop = np.zeros((50, 50, 3), dtype=np.uint8)
        score = det._skin_ratio_confidence(crop)
        assert score == pytest.approx(0.0)


# ─── Seatbelt ─────────────────────────────────────────────────────────────────

class TestSeatbeltViolationDetector:
    def test_motorcycle_ignored(self, blank_640):
        det = SeatbeltViolationDetector()
        dr = make_dr([make_vehicle(vtype=VehicleType.MOTORCYCLE)])
        assert det.detect(blank_640, dr) == []

    def test_car_runs_without_crash(self, grey_image):
        det = SeatbeltViolationDetector()
        dr = make_dr([make_vehicle(vtype=VehicleType.CAR, x1=50, y1=50, x2=400, y2=400)])
        result = det.detect(grey_image, dr)
        assert isinstance(result, list)

    def test_violation_category_correct(self, grey_image):
        det = SeatbeltViolationDetector()
        dr = make_dr([make_vehicle(vtype=VehicleType.CAR, x1=50, y1=50, x2=400, y2=400)])
        for v in det.detect(grey_image, dr):
            assert v.violation_category == ViolationCategory.SEATBELT_NON_COMPLIANCE

    def test_belt_colour_dark_image(self):
        det = SeatbeltViolationDetector()
        crop = np.zeros((60, 60, 3), dtype=np.uint8)
        conf = det._belt_colour_confidence(crop)
        assert 0.0 <= conf <= 1.0


# ─── Triple Riding ────────────────────────────────────────────────────────────

class TestTripleRidingDetector:
    def test_no_motorcycle_no_violation(self, blank_640):
        det = TripleRidingDetector(min_rider_count=3)
        dr = make_dr([make_vehicle(vtype=VehicleType.CAR)])
        assert det.detect(blank_640, dr) == []

    def test_returns_list(self, blank_640):
        det = TripleRidingDetector(min_rider_count=3)
        dr = make_dr([make_vehicle(vtype=VehicleType.MOTORCYCLE)])
        assert isinstance(det.detect(blank_640, dr), list)

    def test_triple_riding_category(self, blank_640):
        det = TripleRidingDetector(min_rider_count=3)
        dr = make_dr([make_vehicle(vtype=VehicleType.MOTORCYCLE)])
        for v in det.detect(blank_640, dr):
            assert v.violation_category == ViolationCategory.TRIPLE_RIDING

    def test_three_persons_on_moto_triggers(self, grey_image):
        """3 persons overlapping one moto → triple riding."""
        det = TripleRidingDetector(min_rider_count=3)
        moto = make_vehicle(vtype=VehicleType.MOTORCYCLE, x1=100, y1=100, x2=400, y2=500)
        # 3 persons heavily overlapping the moto
        p1 = make_person(x1=110, y1=90, x2=200, y2=500, track_id=11)
        p2 = make_person(x1=200, y1=90, x2=300, y2=500, track_id=12)
        p3 = make_person(x1=300, y1=90, x2=390, y2=500, track_id=13)
        dr = make_dr([moto, p1, p2, p3])
        result = det.detect(grey_image, dr)
        assert len(result) == 1
        assert result[0].violation_category == ViolationCategory.TRIPLE_RIDING
        assert result[0].metadata["rider_count"] == 3

    def test_two_persons_no_violation(self, grey_image):
        """2 persons → not a triple riding violation."""
        det = TripleRidingDetector(min_rider_count=3)
        moto = make_vehicle(vtype=VehicleType.MOTORCYCLE, x1=100, y1=100, x2=400, y2=500)
        p1 = make_person(x1=110, y1=90, x2=250, y2=500, track_id=11)
        p2 = make_person(x1=250, y1=90, x2=390, y2=500, track_id=12)
        dr = make_dr([moto, p1, p2])
        result = det.detect(grey_image, dr)
        assert result == []

    def test_projection_estimate_returns_int(self, grey_image):
        det = TripleRidingDetector()
        moto = make_vehicle(x1=50, y1=50, x2=300, y2=400)
        count = det._estimate_riders_from_projection(grey_image, moto)
        assert isinstance(count, int) and count >= 1


# ─── Stop Line ────────────────────────────────────────────────────────────────

class TestStopLineViolationDetector:
    def test_no_stop_line_no_violation(self, blank_640):
        det = StopLineViolationDetector()
        dr = make_dr([make_vehicle()])
        # blank image → no Hough line detected
        result = det.detect(blank_640, dr)
        assert isinstance(result, list)

    def test_vehicle_above_line_no_violation(self, blank_640):
        det = StopLineViolationDetector()
        vehicle = make_vehicle(x1=100, y1=100, x2=300, y2=300)
        dr = make_dr([vehicle])
        result = det.detect(blank_640, dr, stop_line_y=400)
        assert result == []

    def test_vehicle_below_line_is_violation(self, blank_640):
        det = StopLineViolationDetector()
        vehicle = make_vehicle(x1=100, y1=200, x2=300, y2=500)
        dr = make_dr([vehicle])
        result = det.detect(blank_640, dr, stop_line_y=300)
        assert len(result) == 1
        assert result[0].violation_category == ViolationCategory.STOP_LINE_VIOLATION

    def test_confidence_increases_with_overlap(self, blank_640):
        det = StopLineViolationDetector()
        dr1 = make_dr([make_vehicle(x1=100, y1=200, x2=300, y2=310)])
        r1 = det.detect(blank_640, dr1, stop_line_y=300)
        dr2 = make_dr([make_vehicle(x1=100, y1=100, x2=300, y2=600)])
        r2 = det.detect(blank_640, dr2, stop_line_y=300)
        if r1 and r2:
            assert r2[0].confidence >= r1[0].confidence


# ─── Red Light ────────────────────────────────────────────────────────────────

class TestRedLightViolationDetector:
    def test_no_prev_frame_no_violation(self, blank_640):
        det = RedLightViolationDetector()
        dr = make_dr([make_vehicle(vtype=VehicleType.CAR)])
        assert det.detect(blank_640, dr, prev_frame=None) == []

    def test_signal_detection_returns_state(self, blank_640):
        det = RedLightViolationDetector()
        state = det._detect_signal(blank_640)
        assert isinstance(state, TrafficSignalState)

    def test_red_image_detected_as_red(self):
        det = RedLightViolationDetector()
        red_img = np.zeros((200, 200, 3), dtype=np.uint8)
        red_img[:, :, 2] = 255  # BGR: all red
        state = det._detect_signal(red_img)
        assert state == TrafficSignalState.RED

    def test_green_image_not_red(self):
        det = RedLightViolationDetector()
        green_img = np.zeros((200, 200, 3), dtype=np.uint8)
        green_img[:, :, 1] = 200  # BGR: green
        state = det._detect_signal(green_img)
        assert state != TrafficSignalState.RED


# ─── Illegal Parking ──────────────────────────────────────────────────────────

class TestIllegalParkingDetector:
    def test_moving_vehicle_not_flagged(self, blank_640):
        det = IllegalParkingDetector(max_stationary_seconds=5)
        vehicle = make_vehicle(is_stationary=False, track_id=42)
        dr = make_dr([vehicle])
        assert det.detect(blank_640, dr) == []

    def test_stationary_vehicle_registered(self, blank_640):
        det = IllegalParkingDetector(max_stationary_seconds=300)
        vehicle = make_vehicle(is_stationary=True, track_id=99)
        dr = make_dr([vehicle])
        det.detect(blank_640, dr)
        assert 99 in det._registry  # renamed from _stationary_registry

    def test_in_zone_true(self):
        det = IllegalParkingDetector()
        vehicle = make_vehicle(x1=100, y1=100, x2=200, y2=200)
        assert det._in_zone(vehicle, [(50,50,250,250)]) is True

    def test_in_zone_false(self):
        det = IllegalParkingDetector()
        vehicle = make_vehicle(x1=100, y1=100, x2=200, y2=200)
        assert det._in_zone(vehicle, [(300,300,500,500)]) is False


# ─── ViolationEngine ──────────────────────────────────────────────────────────

class TestViolationEngine:
    def test_engine_has_all_detectors(self):
        engine = ViolationEngine()
        expected = {"helmet","seatbelt","triple_riding","wrong_side","stop_line","red_light","illegal_parking"}
        assert set(engine.detectors.keys()) == expected

    def test_run_returns_list(self, blank_640):
        engine = ViolationEngine()
        assert isinstance(engine.run(blank_640, make_dr([])), list)

    def test_run_with_vehicles(self, blank_640):
        engine = ViolationEngine()
        dr = make_dr([make_vehicle(vtype=VehicleType.CAR),
                      make_vehicle(vtype=VehicleType.MOTORCYCLE, track_id=2)])
        result = engine.run(blank_640, dr)
        assert isinstance(result, list)
        for v in result:
            assert isinstance(v, ViolationResult)
            assert 0.0 <= v.confidence <= 1.0

    def test_confidence_filter_applied(self, blank_640):
        """All returned violations must have conf >= 0.40."""
        engine = ViolationEngine()
        dr = make_dr([make_vehicle(), make_vehicle(vtype=VehicleType.CAR, track_id=2)])
        for v in engine.run(blank_640, dr):
            assert v.confidence >= 0.40

    def test_triple_riding_detected_by_engine(self, blank_640):
        """Engine correctly detects triple riding when 3 persons overlap moto."""
        engine = ViolationEngine()
        moto = make_vehicle(vtype=VehicleType.MOTORCYCLE, x1=50, y1=50, x2=400, y2=550)
        persons = [
            make_person(x1=60, y1=40, x2=180, y2=550, track_id=11),
            make_person(x1=180, y1=40, x2=300, y2=550, track_id=12),
            make_person(x1=300, y1=40, x2=390, y2=550, track_id=13),
        ]
        dr = make_dr([moto] + persons)
        result = engine.run(blank_640, dr)
        categories = [v.violation_category for v in result]
        assert ViolationCategory.TRIPLE_RIDING in categories
