"""
Vehicle Detection Service — YOLOv11 + ByteTrack.

Key upgrade: person detections (class 0) are now KEPT in the result
so violation detectors can associate riders/drivers with vehicles via IoU.
Previously persons were silently mixed in — now they are a first-class output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.models.violation import VehicleType

logger = get_logger(__name__)

# COCO → VehicleType. Person (0) is mapped to PEDESTRIAN so it appears in
# DetectionResult.objects and violation detectors can use it.
COCO_TO_VEHICLE: Dict[int, VehicleType] = {
    0:  VehicleType.PEDESTRIAN,   # person (rider / driver / pedestrian)
    1:  VehicleType.BICYCLE,
    2:  VehicleType.CAR,
    3:  VehicleType.MOTORCYCLE,
    5:  VehicleType.BUS,
    7:  VehicleType.TRUCK,
    80: VehicleType.AUTO_RICKSHAW,  # custom fine-tuned class
}

TARGET_CLASSES = set(COCO_TO_VEHICLE.keys())

# Lower confidence threshold so persons on motorcycles aren't missed
PERSON_CONF_THRESHOLD = 0.30   # lower than vehicles
VEHICLE_CONF_THRESHOLD = 0.40


@dataclass
class DetectedObject:
    class_id: int
    vehicle_type: VehicleType
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: Optional[int] = None
    is_stationary: bool = False
    speed_estimate_kmh: Optional[float] = None

    @property
    def bbox_normalized(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class DetectionResult:
    objects: List[DetectedObject] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_shape: Tuple[int, int] = (0, 0)
    model_name: str = ""

    @property
    def vehicles(self) -> List[DetectedObject]:
        return [o for o in self.objects if o.vehicle_type != VehicleType.PEDESTRIAN]

    @property
    def pedestrians(self) -> List[DetectedObject]:
        return [o for o in self.objects if o.vehicle_type == VehicleType.PEDESTRIAN]

    def by_type(self, vtype: VehicleType) -> List[DetectedObject]:
        return [o for o in self.objects if o.vehicle_type == vtype]


class VehicleDetector:
    """
    YOLOv11 vehicle and person detector with ByteTrack tracking.

    Persons are included in output at a lower confidence threshold
    so violation detectors can do rider/driver association via IoU.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
        enable_tracking: bool = False,
    ):
        self.model_path = model_path or settings.YOLO_VEHICLE_MODEL
        self.conf_threshold = confidence_threshold or settings.YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or settings.YOLO_IOU_THRESHOLD
        self.device = device or settings.YOLO_DEVICE
        self.enable_tracking = enable_tracking
        self.img_size = settings.YOLO_IMAGE_SIZE
        self._model = None
        self._loaded = False
        pass

    def _load_model(self):
        try:
            from ultralytics import YOLO
            model_path = Path(settings.MODEL_DIR) / self.model_path
            if not model_path.exists():
                logger.warning(f"Model {model_path} not found. Downloading yolov8n as fallback.")
                self._model = YOLO("yolov8n.pt")
            else:
                self._model = YOLO(str(model_path))
            self._model.to(self.device)
            self._loaded = True
            logger.info(f"Detection model loaded | device={self.device} | conf={self.conf_threshold}")
        except ImportError:
            logger.error("ultralytics not installed")
            raise
        except Exception as exc:
            logger.error(f"Model load failed: {exc}")
            raise

    def detect(self, image: np.ndarray, augment: bool = False) -> DetectionResult:

        if not self._loaded:
            logger.info("Loading YOLO model on first request...")
        self._load_model()

    h, w = image.shape[:2]
    t0 = time.perf_counter()
    try:
            # Use lower confidence to catch persons on vehicles
            if self.enable_tracking:
                raw = self._model.track(
                    source=image, conf=min(self.conf_threshold, PERSON_CONF_THRESHOLD),
                    iou=self.iou_threshold, imgsz=self.img_size,
                    classes=list(TARGET_CLASSES), persist=True, verbose=False,
                )
            else:
                raw = self._model.predict(
                    source=image, conf=min(self.conf_threshold, PERSON_CONF_THRESHOLD),
                    iou=self.iou_threshold, imgsz=self.img_size,
                    classes=list(TARGET_CLASSES), augment=augment, verbose=False,
                )
        except Exception as exc:
            logger.error(f"YOLO inference failed: {exc}")
            raise

        elapsed = (time.perf_counter() - t0) * 1000
        objects = self._parse(raw, w, h)
        logger.debug(f"Detected {len(objects)} objects in {elapsed:.1f}ms")
        return DetectionResult(objects=objects, inference_time_ms=elapsed,
                               image_shape=(h, w), model_name=str(self.model_path))

    def _parse(self, raw_results, image_width, image_height) -> List[DetectedObject]:
        objects: List[DetectedObject] = []
        for result in raw_results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in COCO_TO_VEHICLE:
                    continue
                conf = float(boxes.conf[i].item())
                vtype = COCO_TO_VEHICLE[cls_id]

                # Apply per-class confidence threshold
                min_conf = PERSON_CONF_THRESHOLD if vtype == VehicleType.PEDESTRIAN else VEHICLE_CONF_THRESHOLD
                if conf < min_conf:
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy()
                x1 = float(max(0.0, min(xyxy[0], image_width)))
                y1 = float(max(0.0, min(xyxy[1], image_height)))
                x2 = float(max(0.0, min(xyxy[2], image_width)))
                y2 = float(max(0.0, min(xyxy[3], image_height)))

                track_id = None
                if hasattr(boxes, 'id') and boxes.id is not None:
                    try:
                        track_id = int(boxes.id[i].item())
                    except Exception:
                        pass

                objects.append(DetectedObject(
                    class_id=cls_id, vehicle_type=vtype, confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2, track_id=track_id,
                ))
        return objects

    def draw_detections(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        img = image.copy()
        colors = {
            VehicleType.CAR: (0,200,0), VehicleType.TRUCK: (255,128,0),
            VehicleType.BUS: (0,128,255), VehicleType.MOTORCYCLE: (255,0,0),
            VehicleType.BICYCLE: (0,255,255), VehicleType.AUTO_RICKSHAW: (255,255,0),
            VehicleType.PEDESTRIAN: (200,100,255),
        }
        for obj in result.objects:
            color = colors.get(obj.vehicle_type, (128,128,128))
            x1,y1,x2,y2 = int(obj.x1),int(obj.y1),int(obj.x2),int(obj.y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            label = f"{obj.vehicle_type.value} {obj.confidence:.0%}"
            if obj.track_id:
                label = f"#{obj.track_id} {label}"
            cv2.putText(img,label,(x1,max(y1-4,10)),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
        return img

    @property
    def is_loaded(self) -> bool:
        return self._loaded
