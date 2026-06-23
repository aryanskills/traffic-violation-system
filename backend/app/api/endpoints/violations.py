"""
Violations API Endpoints.

GET  /api/v1/violations         — List violations with filters + pagination
GET  /api/v1/violations/{id}    — Get single violation record
GET  /api/v1/violations/plates/{plate} — Lookup by plate number
"""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.violation import (
    ViolationRecord, ViolationCategory, VehicleType, SeverityLevel,
)
from app.schemas.violation import (
    ViolationListResponse, ViolationRecordOut,
)

router = APIRouter()


@router.get(
    "/violations",
    response_model=ViolationListResponse,
    summary="List violations with optional filters",
)
async def list_violations(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    violation_category: Optional[ViolationCategory] = Query(None),
    vehicle_type: Optional[VehicleType] = Query(None),
    severity: Optional[SeverityLevel] = Query(None),
    plate_number: Optional[str] = Query(None, min_length=2),
    camera_id: Optional[str] = Query(None),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    conditions = []
    if start_date:
        conditions.append(ViolationRecord.created_at >= start_date)
    if end_date:
        conditions.append(ViolationRecord.created_at <= end_date)
    if violation_category:
        conditions.append(ViolationRecord.violation_category == violation_category)
    if vehicle_type:
        conditions.append(ViolationRecord.vehicle_type == vehicle_type)
    if severity:
        conditions.append(ViolationRecord.severity == severity)
    if plate_number:
        conditions.append(ViolationRecord.plate_number.ilike(f"%{plate_number}%"))
    if camera_id:
        conditions.append(ViolationRecord.camera_id == camera_id)
    if min_confidence > 0:
        conditions.append(ViolationRecord.confidence >= min_confidence)

    where = and_(*conditions) if conditions else True

    # Count
    count_q = await db.execute(
        select(func.count()).select_from(ViolationRecord).where(where)
    )
    total = count_q.scalar_one() or 0

    # Paginated data
    offset = (page - 1) * page_size
    data_q = await db.execute(
        select(ViolationRecord)
        .where(where)
        .order_by(ViolationRecord.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    records = data_q.scalars().all()

    items = [
        ViolationRecordOut(
            id=r.id,
            session_id=r.session_id,
            violation_category=r.violation_category,
            vehicle_type=r.vehicle_type,
            severity=r.severity,
            confidence=r.confidence,
            plate_number=r.plate_number,
            evidence_image_url=(
                f"/api/v1/evidence/{r.evidence_image_path.split('/')[-1]}"
                if r.evidence_image_path else None
            ),
            location_label=r.location_label,
            camera_id=r.camera_id,
            challan_issued=r.challan_issued,
            metadata=r.extra_data or {},
            created_at=r.created_at,
        )
        for r in records
    ]

    return ViolationListResponse(
        success=True,
        message="OK",
        total=total,
        page=page,
        page_size=page_size,
        items=items,
    )


@router.get(
    "/violations/{violation_id}",
    response_model=ViolationRecordOut,
    summary="Get a single violation by ID",
)
async def get_violation(
    violation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    q = await db.execute(
        select(ViolationRecord).where(ViolationRecord.id == violation_id)
    )
    r = q.scalar_one_or_none()
    if not r:
        raise HTTPException(status_code=404, detail="Violation not found")

    return ViolationRecordOut(
        id=r.id,
        session_id=r.session_id,
        violation_category=r.violation_category,
        vehicle_type=r.vehicle_type,
        severity=r.severity,
        confidence=r.confidence,
        plate_number=r.plate_number,
        evidence_image_url=(
            f"/api/v1/evidence/{r.evidence_image_path.split('/')[-1]}"
            if r.evidence_image_path else None
        ),
        location_label=r.location_label,
        camera_id=r.camera_id,
        challan_issued=r.challan_issued,
        metadata=r.extra_data or {},
        created_at=r.created_at,
    )


@router.get(
    "/violations/plates/{plate_number}",
    response_model=ViolationListResponse,
    summary="Get all violations for a license plate",
)
async def violations_by_plate(
    plate_number: str,
    db: AsyncSession = Depends(get_db),
):
    q = await db.execute(
        select(ViolationRecord)
        .where(ViolationRecord.plate_number.ilike(f"%{plate_number}%"))
        .order_by(ViolationRecord.created_at.desc())
        .limit(100)
    )
    records = q.scalars().all()

    items = [
        ViolationRecordOut(
            id=r.id,
            session_id=r.session_id,
            violation_category=r.violation_category,
            vehicle_type=r.vehicle_type,
            severity=r.severity,
            confidence=r.confidence,
            plate_number=r.plate_number,
            location_label=r.location_label,
            camera_id=r.camera_id,
            challan_issued=r.challan_issued,
            metadata=r.extra_data or {},
            created_at=r.created_at,
        )
        for r in records
    ]
    return ViolationListResponse(
        success=True, message="OK",
        total=len(items), page=1, page_size=len(items), items=items,
    )
