"""
Unit tests for Vehicle Detection Service v2.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.services.detection.vehicle_detector import (
    VehicleDetector, DetectedObject, DetectionResult,
    COCO_TO_VEHICLE, PERSON_CONF_THRESHOLD, VEHICLE_CONF_THRESHOLD,
)
from app.models.violation import VehicleType


@pytest.fixture
def blank_image():
    return np.zeros((640, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_detector():
    with patch("app.services.detection.vehicle_detector.VehicleDetector._load_model"):
        det = VehicleDetector.__new__(VehicleDetector)
        det.conf_threshold = 0.35
        det.iou_threshold = 0.45
        det.device = "cpu"
        det.img_size = 640
        det.enable_tracking = False
        det._model = MagicMock()
        det._loaded = True
        return det


class TestDetectedObject:
    def test_center(self):
        obj = DetectedObject(2, VehicleType.CAR, 0.9, 100, 100, 200, 200)
        assert obj.center == (150.0, 150.0)

    def test_area(self):
        obj = DetectedObject(2, VehicleType.CAR, 0.9, 0, 0, 100, 50)
        assert obj.area == 5000.0

    def test_bbox_normalized(self):
        obj = DetectedObject(2, VehicleType.CAR, 0.85, 10, 20, 110, 120)
        assert obj.bbox_normalized == (10, 20, 110, 120)


class TestDetectionResult:
    def _obj(self, vtype):
        return DetectedObject(0, vtype, 0.8, 0, 0, 50, 50)

    def test_vehicles_excludes_pedestrians(self):
        dr = DetectionResult(objects=[
            self._obj(VehicleType.CAR),
            self._obj(VehicleType.PEDESTRIAN),
            self._obj(VehicleType.MOTORCYCLE),
        ])
        assert len(dr.vehicles) == 2

    def test_pedestrians_filter(self):
        dr = DetectionResult(objects=[
            self._obj(VehicleType.PEDESTRIAN),
            self._obj(VehicleType.CAR),
        ])
        assert len(dr.pedestrians) == 1

    def test_by_type(self):
        dr = DetectionResult(objects=[
            self._obj(VehicleType.CAR),
            self._obj(VehicleType.CAR),
            self._obj(VehicleType.TRUCK),
        ])
        assert len(dr.by_type(VehicleType.CAR)) == 2
        assert len(dr.by_type(VehicleType.TRUCK)) == 1
        assert len(dr.by_type(VehicleType.BUS)) == 0


class TestCOCOMapping:
    def test_all_map_to_vehicle_type(self):
        for cls_id, vtype in COCO_TO_VEHICLE.items():
            assert isinstance(vtype, VehicleType)

    def test_car_mapping(self):
        assert COCO_TO_VEHICLE[2] == VehicleType.CAR

    def test_motorcycle_mapping(self):
        assert COCO_TO_VEHICLE[3] == VehicleType.MOTORCYCLE

    def test_person_is_pedestrian(self):
        assert COCO_TO_VEHICLE[0] == VehicleType.PEDESTRIAN

    def test_person_conf_lower_than_vehicle(self):
        assert PERSON_CONF_THRESHOLD < VEHICLE_CONF_THRESHOLD


class TestVehicleDetector:
    def test_is_loaded(self, mock_detector):
        assert mock_detector.is_loaded is True

    def test_parse_empty_results(self, mock_detector):
        """_parse with no boxes returns empty list."""
        mock_result = MagicMock()
        mock_result.boxes = None
        objects = mock_detector._parse([mock_result], 640, 640)
        assert objects == []

    def test_draw_detections_returns_image(self, mock_detector, blank_image):
        dr = DetectionResult(objects=[
            DetectedObject(2, VehicleType.CAR, 0.8, 10, 10, 200, 200)
        ])
        out = mock_detector.draw_detections(blank_image, dr)
        assert out.shape == blank_image.shape
        assert out is not blank_image

    def test_person_included_in_objects(self):
        """Persons must be in DetectionResult so violation detectors can use them."""
        dr = DetectionResult(objects=[
            DetectedObject(0, VehicleType.PEDESTRIAN, 0.6, 10, 10, 100, 200),
            DetectedObject(3, VehicleType.MOTORCYCLE, 0.8, 50, 50, 300, 400),
        ])
        assert len(dr.pedestrians) == 1
        assert len(dr.by_type(VehicleType.MOTORCYCLE)) == 1

    def test_low_conf_vehicle_filtered(self, mock_detector):
        """Detections below per-class threshold should be filtered."""
        mock_box = MagicMock()
        mock_box.__len__ = lambda s: 1
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].item.return_value = 2  # car
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].item.return_value = 0.10  # below VEHICLE_CONF_THRESHOLD=0.40
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = [10, 10, 100, 100]

        mock_result = MagicMock()
        mock_result.boxes = mock_box
        objects = mock_detector._parse([mock_result], 640, 640)
        assert len(objects) == 0
