"""Health check endpoint."""

import time
from fastapi import APIRouter
from app.core.config import settings
from app.schemas.violation import HealthResponse, ModelStatus

router = APIRouter()
_START = time.time()


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check():
    from app.services.detection.pipeline import pipeline

    models = [
        ModelStatus(
            name="YOLOv11 Vehicle Detector",
            loaded=pipeline.vehicle_detector.is_loaded,
            device=settings.YOLO_DEVICE,
            version="yolov11",
        ),
        ModelStatus(
            name="OCR Engine",
            loaded=pipeline.plate_recognizer._ocr_loaded,
            device=settings.YOLO_DEVICE,
            version=settings.OCR_ENGINE,
        ),
    ]

    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        models=models,
        database="connected",
        uptime_seconds=round(time.time() - _START, 1),
    )
