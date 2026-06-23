"""
Detection API Endpoints.

POST /api/v1/detect  — Upload image, run full pipeline, return results.
GET  /api/v1/sessions/{id} — Get session result by ID.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.models.violation import DetectionSession
from app.schemas.violation import DetectionSessionResponse
from app.services.detection.pipeline import pipeline

logger = get_logger(__name__)
router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@router.post(
    "/detect",
    response_model=DetectionSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze traffic image for violations",
    description=(
        "Upload a traffic surveillance image. The system preprocesses it, "
        "detects vehicles and violations, reads license plates, generates "
        "annotated evidence, and returns structured results."
    ),
)
async def detect_violations(
    file: UploadFile = File(..., description="Traffic image (JPEG/PNG/WEBP)"),
    location_label: Optional[str] = Form(None, description="Location description"),
    camera_id: Optional[str] = Form(None, description="Camera identifier"),
    db: AsyncSession = Depends(get_db),
):
    # ── Validation ────────────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {ALLOWED_TYPES}",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    if len(image_bytes) < 1000:
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupted")

    logger.info(
        f"Detection request | file={file.filename} | "
        f"size={len(image_bytes)//1024}KB | cam={camera_id} | loc={location_label}"
    )

    try:
        result = await pipeline.process_image(
            image_bytes=image_bytes,
            db=db,
            original_filename=file.filename or "upload.jpg",
            location_label=location_label,
            camera_id=camera_id,
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Detection pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Detection processing failed")


@router.get(
    "/sessions/{session_id}",
    response_model=DetectionSessionResponse,
    summary="Retrieve a previous detection session",
)
async def get_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    q = await db.execute(
        select(DetectionSession).where(DetectionSession.id == session_id)
    )
    session = q.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return DetectionSessionResponse(
        session_id=session.id,
        status=session.status,
        original_filename=session.original_filename,
        location_label=session.location_label,
        camera_id=session.camera_id,
        processing_time_ms=session.processing_time_ms,
        total_vehicles_detected=session.total_vehicles_detected or 0,
        total_violations_detected=session.total_violations_detected or 0,
        vehicles=[],
        violations=[],
        created_at=session.created_at,
    )
