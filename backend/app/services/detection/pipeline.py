"""
Main Detection Pipeline — v2.0

Orchestrates the full end-to-end flow:
  1. Decode & save original
  2. Preprocess (CLAHE, low-light, rain, shadow)
  3. Vehicle detection (YOLOv11 + persons at lower threshold)
  4. Violation detection (AI-backed engine)
  5. License plate OCR (YOLO plate → ensemble OCR)
  6. Evidence image generation (fixed coordinates)
  7. Database persistence
  8. Return structured response
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.violation import (
    DetectionSession, DetectedVehicle, LicensePlate,
    ViolationRecord, ProcessingStatus,
)
from app.preprocessing.image_processor import ImagePreprocessor
from app.schemas.violation import (
    DetectionSessionResponse, VehicleDetection,
    ViolationDetected, BoundingBox,
)
from app.services.analytics.analytics_service import AnalyticsService
from app.services.detection.vehicle_detector import VehicleDetector
from app.services.evidence.evidence_generator import EvidenceGenerator
from app.services.ocr.plate_recognizer import LicensePlateRecognizer
from app.services.violation.violation_engine import ViolationEngine

logger = get_logger(__name__)


class DetectionPipeline:
    """Singleton service — all models loaded once at startup."""

    def __init__(self):
        self.preprocessor     = ImagePreprocessor()
        self.vehicle_detector = None
        self.violation_engine = None
        self.plate_recognizer = None
        self.evidence_generator = EvidenceGenerator()
        self._ready = False

    def warm_up(self):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            self.preprocessor.process(dummy)
            self.vehicle_detector.detect(dummy)
        except Exception as e:
            logger.warning(f"Warm-up partial error (non-fatal): {e}")
        self._ready = True
        logger.info("Detection pipeline warm-up complete ✓")

    # ── Main entry point ─────────────────────────────────────────────────────

    async def process_image(
        self,
        image_bytes: bytes,
        db: AsyncSession,
        original_filename: str = "image.jpg",
        location_label: Optional[str] = None,
        camera_id: Optional[str] = None,
        
    )# Lazy-load models only when first detection request arrives
if self.vehicle_detector is None:
    logger.info("Initializing detection models...")
    self.vehicle_detector = VehicleDetector(enable_tracking=True)

if self.violation_engine is None:
    self.violation_engine = ViolationEngine(
        helmet_model=None,
        seatbelt_model=None
    )

if self.plate_recognizer is None:
    self.plate_recognizer = LicensePlateRecognizer() -> DetectionSessionResponse:

        session_id = uuid.uuid4()
        t_start    = time.perf_counter()

        # Create DB record immediately
        db_session = DetectionSession(
            id=session_id,
            original_filename=original_filename,
            file_path="",
            status=ProcessingStatus.PROCESSING,
            processing_started_at=datetime.utcnow(),
            location_label=location_label,
            camera_id=camera_id,
        )
        db.add(db_session)
        await db.flush()

        evidence_url     = None
        preprocessed_url = None

        try:
            # ── 1. Decode ────────────────────────────────────────────────────
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            original_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if original_image is None:
                raise ValueError("Cannot decode image — unsupported format or corrupted file")

            # Ensure directories exist (important inside Docker)
            Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
            Path(settings.EVIDENCE_DIR).mkdir(parents=True, exist_ok=True)

            orig_path = Path(settings.UPLOAD_DIR) / f"{session_id}_original.jpg"
            cv2.imwrite(str(orig_path), original_image,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            db_session.file_path       = str(orig_path)
            db_session.file_size_bytes = len(image_bytes)

            # ── 2. Preprocess ────────────────────────────────────────────────
            logger.info(f"[{session_id}] Preprocessing")
            prep_result    = self.preprocessor.process(original_image)
            processed_image = prep_result.image

            prep_path = Path(settings.UPLOAD_DIR) / f"{session_id}_preprocessed.jpg"
            cv2.imwrite(str(prep_path), processed_image,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            db_session.preprocessed_image_path = str(prep_path)
            preprocessed_url = f"/api/v1/uploads/{prep_path.name}"

            # ── 3. Vehicle Detection ─────────────────────────────────────────
            logger.info(f"[{session_id}] Vehicle detection")
            detection_result = self.vehicle_detector.detect(processed_image)

            # Count only actual vehicles (not persons) for the summary
            vehicle_count = len(detection_result.vehicles)
            person_count  = len(detection_result.pedestrians)
            db_session.total_vehicles_detected = vehicle_count

            db_vehicles = []
            for obj in detection_result.objects:
                dv = DetectedVehicle(
                    session_id=session_id,
                    track_id=obj.track_id,
                    vehicle_type=obj.vehicle_type,
                    confidence=obj.confidence,
                    bbox_x1=obj.x1, bbox_y1=obj.y1,
                    bbox_x2=obj.x2, bbox_y2=obj.y2,
                    is_stationary=obj.is_stationary,
                )
                db.add(dv)
                db_vehicles.append(dv)
            await db.flush()

            # ── 4. Violation Detection ───────────────────────────────────────
            logger.info(f"[{session_id}] Violation detection "
                        f"(vehicles={vehicle_count}, persons={person_count})")
            violations = self.violation_engine.run(
                image=processed_image,
                detection_result=detection_result,
                context={},
            )
            db_session.total_violations_detected = len(violations)

            # ── 5. License Plate OCR ─────────────────────────────────────────
            logger.info(f"[{session_id}] OCR")
            plate_results = []
            # Only run OCR on actual vehicles (not pedestrians), cap at 8
            for i, vobj in enumerate(detection_result.vehicles[:8]):
                try:
                    plate = self.plate_recognizer.recognize(processed_image, vobj)
                    plate_results.append(plate)
                    if i < len(db_vehicles):
                        db_plate = LicensePlate(
                            vehicle_id=db_vehicles[i].id,
                            raw_text=plate.raw_text,
                            normalized_text=plate.normalized_text,
                            is_valid_format=plate.is_valid_format,
                            detection_confidence=plate.detection_confidence,
                            ocr_confidence=plate.ocr_confidence,
                            bbox_x1=plate.bbox[0] if plate.bbox else None,
                            bbox_y1=plate.bbox[1] if plate.bbox else None,
                            bbox_x2=plate.bbox[2] if plate.bbox else None,
                            bbox_y2=plate.bbox[3] if plate.bbox else None,
                        )
                        db.add(db_plate)
                except Exception as ocr_err:
                    logger.warning(f"OCR failed for vehicle {i}: {ocr_err}")

            # Attach best plate to all violations
            best_plate = max(
                (p for p in plate_results if p.normalized_text and p.is_valid_format),
                key=lambda p: p.ocr_confidence,
                default=None,
            ) or max(
                (p for p in plate_results if p.normalized_text),
                key=lambda p: p.ocr_confidence,
                default=None,
            )
            if best_plate:
                for v in violations:
                    v.plate_hint = best_plate.normalized_text

            # ── 6. Persist Violations ────────────────────────────────────────
            analytics = AnalyticsService(db)
            for v in violations:
                vr = ViolationRecord(
                    session_id=session_id,
                    violation_category=v.violation_category,
                    vehicle_type=v.vehicle_type,
                    severity=v.severity,
                    confidence=v.confidence,
                    sub_violations=v.sub_violations,
                    plate_number=v.plate_hint,
                    location_label=location_label,
                    camera_id=camera_id,
                    extra_data=v.metadata,
                )
                db.add(vr)
                if v.plate_hint:
                    try:
                        await analytics.update_repeat_offender(v.plate_hint, v)
                    except Exception as ae:
                        logger.warning(f"Repeat offender update failed: {ae}")
            await db.flush()

            # ── 7. Evidence Generation ───────────────────────────────────────
            logger.info(f"[{session_id}] Evidence generation")
            try:
                evidence = self.evidence_generator.generate(
                    image=processed_image,
                    detection_result=detection_result,
                    violations=violations,
                    plate_results=plate_results if plate_results else None,
                    session_id=str(session_id),
                    location_label=location_label,
                    camera_id=camera_id,
                )
                # URL uses only the filename — StaticFiles serves from EVIDENCE_DIR
                evidence_url = f"/api/v1/evidence/{Path(evidence.evidence_image_path).name}"
            except Exception as ev_err:
                logger.error(f"Evidence generation failed: {ev_err}", exc_info=True)
                # Non-fatal — continue without evidence image

            # ── 8. Finalise ──────────────────────────────────────────────────
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            db_session.status = ProcessingStatus.COMPLETED
            db_session.processing_completed_at = datetime.utcnow()
            db_session.processing_time_ms = elapsed_ms

            logger.info(
                f"[{session_id}] Done | vehicles={vehicle_count} persons={person_count} "
                f"violations={len(violations)} | {elapsed_ms:.0f}ms"
            )

            return DetectionSessionResponse(
                session_id=session_id,
                status=ProcessingStatus.COMPLETED,
                original_filename=original_filename,
                location_label=location_label,
                camera_id=camera_id,
                processing_time_ms=round(elapsed_ms, 2),
                total_vehicles_detected=vehicle_count,
                total_violations_detected=len(violations),
                vehicles=[
                    VehicleDetection(
                        track_id=obj.track_id,
                        vehicle_type=obj.vehicle_type,
                        confidence=obj.confidence,
                        bbox=BoundingBox(
                            x1=obj.x1, y1=obj.y1, x2=obj.x2, y2=obj.y2,
                            width=obj.x2 - obj.x1, height=obj.y2 - obj.y1,
                        ),
                    )
                    for obj in detection_result.objects
                ],
                violations=[
                    ViolationDetected(
                        violation_category=v.violation_category,
                        vehicle_type=v.vehicle_type,
                        severity=v.severity,
                        confidence=v.confidence,
                        sub_violations=v.sub_violations,
                        plate_number=v.plate_hint,
                        metadata=v.metadata,
                    )
                    for v in violations
                ],
                evidence_image_url=evidence_url,
                preprocessed_image_url=preprocessed_url,
                preprocessing_steps=prep_result.applied_steps,
                preprocessing_quality_score=prep_result.quality_metrics.overall_quality,
                created_at=datetime.utcnow(),
            )

        except Exception as exc:
            logger.error(f"[{session_id}] Pipeline failed: {exc}", exc_info=True)
            db_session.status = ProcessingStatus.FAILED
            db_session.error_message = str(exc)
            raise


# Singleton — loaded once at startup
pipeline = DetectionPipeline()
