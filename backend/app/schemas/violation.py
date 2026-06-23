"""
Pydantic v2 schemas for API request/response serialization and validation.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from app.models.violation import (
    VehicleType, ViolationCategory, SeverityLevel,
    ProcessingStatus, TrafficSignalState,
)


# ─── Shared / Common ───────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float = 0.0
    height: float = 0.0

    model_config = ConfigDict(from_attributes=True)


class BaseResponse(BaseModel):
    success: bool = True
    message: str = "OK"

    model_config = ConfigDict(from_attributes=True)


# ─── Detection ─────────────────────────────────────────────────────────────────

class DetectionRequest(BaseModel):
    location_label: Optional[str] = Field(None, description="Human-readable location")
    camera_id: Optional[str] = Field(None, description="Camera/sensor identifier")


class VehicleDetection(BaseModel):
    track_id: Optional[int] = None
    vehicle_type: VehicleType
    confidence: float
    bbox: BoundingBox
    is_stationary: bool = False
    speed_estimate_kmh: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class LicensePlateResult(BaseModel):
    raw_text: Optional[str] = None
    normalized_text: Optional[str] = None
    is_valid_format: bool = False
    detection_confidence: Optional[float] = None
    ocr_confidence: Optional[float] = None
    bbox: Optional[BoundingBox] = None

    model_config = ConfigDict(from_attributes=True)


class ViolationDetected(BaseModel):
    violation_category: ViolationCategory
    vehicle_type: Optional[VehicleType] = None
    severity: SeverityLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    sub_violations: List[str] = []
    plate_number: Optional[str] = None
    metadata: Dict[str, Any] = {}

    model_config = ConfigDict(from_attributes=True)


class DetectionSessionResponse(BaseModel):
    session_id: uuid.UUID
    status: ProcessingStatus
    original_filename: str
    location_label: Optional[str] = None
    camera_id: Optional[str] = None
    processing_time_ms: Optional[float] = None
    total_vehicles_detected: int = 0
    total_violations_detected: int = 0
    vehicles: List[VehicleDetection] = []
    violations: List[ViolationDetected] = []
    evidence_image_url: Optional[str] = None
    preprocessed_image_url: Optional[str] = None
    preprocessing_steps: List[str] = []
    preprocessing_quality_score: Optional[float] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ─── Violation Records ─────────────────────────────────────────────────────────

class ViolationRecordOut(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    violation_category: ViolationCategory
    vehicle_type: Optional[VehicleType]
    severity: SeverityLevel
    confidence: float
    plate_number: Optional[str]
    evidence_image_url: Optional[str] = None
    location_label: Optional[str]
    camera_id: Optional[str]
    challan_issued: bool
    metadata: Dict[str, Any] = {}
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ViolationListResponse(BaseResponse):
    total: int
    page: int
    page_size: int
    items: List[ViolationRecordOut]


class ViolationFilterParams(BaseModel):
    """Query parameters for filtering violations."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    violation_category: Optional[ViolationCategory] = None
    vehicle_type: Optional[VehicleType] = None
    plate_number: Optional[str] = None
    camera_id: Optional[str] = None
    severity: Optional[SeverityLevel] = None
    min_confidence: float = 0.0
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=200)


# ─── Analytics ─────────────────────────────────────────────────────────────────

class ViolationCategoryStat(BaseModel):
    category: ViolationCategory
    count: int
    percentage: float


class HourlyDistribution(BaseModel):
    hour: int
    count: int


class DailyTrend(BaseModel):
    date: str  # ISO date string
    count: int


class AnalyticsSummary(BaseModel):
    period_start: datetime
    period_end: datetime
    total_violations: int
    total_sessions: int
    total_vehicles_detected: int
    violations_by_category: List[ViolationCategoryStat]
    violations_by_vehicle_type: Dict[str, int]
    hourly_distribution: List[HourlyDistribution]
    daily_trend: List[DailyTrend]
    avg_confidence: float
    top_offending_plates: List[Dict[str, Any]]
    high_risk_cameras: List[Dict[str, Any]]

    model_config = ConfigDict(from_attributes=True)


# ─── Report Export ─────────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    violation_categories: Optional[List[ViolationCategory]] = None
    camera_ids: Optional[List[str]] = None
    format: str = Field(default="pdf", pattern="^(pdf|csv|xlsx)$")
    include_evidence: bool = False


class ReportResponse(BaseResponse):
    report_id: uuid.UUID
    download_url: str
    format: str
    generated_at: datetime
    record_count: int


# ─── System Health ─────────────────────────────────────────────────────────────

class ModelStatus(BaseModel):
    name: str
    loaded: bool
    device: str
    version: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    models: List[ModelStatus]
    database: str
    uptime_seconds: float
