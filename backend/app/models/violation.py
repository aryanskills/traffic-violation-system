"""
SQLAlchemy ORM models - complete database schema for the traffic violation system.
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey,
    Integer, String, Text, JSON, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


# ─── Enums ─────────────────────────────────────────────────────────────────────

class VehicleType(str, PyEnum):
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    AUTO_RICKSHAW = "auto_rickshaw"
    PEDESTRIAN = "pedestrian"
    UNKNOWN = "unknown"


class ViolationCategory(str, PyEnum):
    HELMET_NON_COMPLIANCE = "helmet_non_compliance"
    SEATBELT_NON_COMPLIANCE = "seatbelt_non_compliance"
    TRIPLE_RIDING = "triple_riding"
    WRONG_SIDE_DRIVING = "wrong_side_driving"
    STOP_LINE_VIOLATION = "stop_line_violation"
    RED_LIGHT_VIOLATION = "red_light_violation"
    ILLEGAL_PARKING = "illegal_parking"


class SeverityLevel(str, PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProcessingStatus(str, PyEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TrafficSignalState(str, PyEnum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


# ─── Mixins ────────────────────────────────────────────────────────────────────

class TimestampMixin:
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class UUIDMixin:
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)


# ─── Detection Session ─────────────────────────────────────────────────────────

class DetectionSession(Base, UUIDMixin, TimestampMixin):
    """Tracks each image/video submitted for analysis."""
    __tablename__ = "detection_sessions"

    # Input
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer)
    media_type = Column(String(50))  # image/jpeg, video/mp4
    location_label = Column(String(255))  # e.g. "Junction 12, MG Road"
    camera_id = Column(String(100))

    # Processing
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False)
    processing_started_at = Column(DateTime(timezone=True))
    processing_completed_at = Column(DateTime(timezone=True))
    processing_time_ms = Column(Float)
    error_message = Column(Text)

    # Results summary
    total_vehicles_detected = Column(Integer, default=0)
    total_violations_detected = Column(Integer, default=0)
    preprocessed_image_path = Column(String(512))

    # Metadata
    raw_detection_payload = Column(JSON)  # Full YOLO output stored for audit

    violations = relationship("ViolationRecord", back_populates="session", cascade="all, delete-orphan")
    vehicles = relationship("DetectedVehicle", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_session_status", "status"),
        Index("ix_session_created", "created_at"),
        Index("ix_session_camera", "camera_id"),
    )


# ─── Detected Vehicle ──────────────────────────────────────────────────────────

class DetectedVehicle(Base, UUIDMixin, TimestampMixin):
    """Individual vehicle detected in a session."""
    __tablename__ = "detected_vehicles"

    session_id = Column(UUID(as_uuid=True), ForeignKey("detection_sessions.id", ondelete="CASCADE"), nullable=False)
    track_id = Column(Integer)  # ByteTrack ID
    vehicle_type = Column(Enum(VehicleType), nullable=False)
    confidence = Column(Float, nullable=False)

    # Bounding box (normalized 0-1)
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)

    # State
    is_stationary = Column(Boolean, default=False)
    stationary_since = Column(DateTime(timezone=True))
    speed_estimate_kmh = Column(Float)

    # Relations
    session = relationship("DetectionSession", back_populates="vehicles")
    license_plate = relationship("LicensePlate", back_populates="vehicle", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_vehicle_session", "session_id"),
        Index("ix_vehicle_type", "vehicle_type"),
    )


# ─── License Plate ─────────────────────────────────────────────────────────────

class LicensePlate(Base, UUIDMixin, TimestampMixin):
    """OCR result for a detected vehicle's license plate."""
    __tablename__ = "license_plates"

    vehicle_id = Column(UUID(as_uuid=True), ForeignKey("detected_vehicles.id", ondelete="CASCADE"), nullable=False)

    # Detection
    raw_text = Column(String(20))       # Raw OCR output
    normalized_text = Column(String(20)) # Cleaned & validated plate number
    is_valid_format = Column(Boolean, default=False)
    detection_confidence = Column(Float)  # YOLO plate detection confidence
    ocr_confidence = Column(Float)        # OCR character confidence (avg)

    # Plate bounding box
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)

    vehicle = relationship("DetectedVehicle", back_populates="license_plate")

    __table_args__ = (
        Index("ix_plate_normalized", "normalized_text"),
    )


# ─── Violation Record ──────────────────────────────────────────────────────────

class ViolationRecord(Base, UUIDMixin, TimestampMixin):
    """Core violation record - one row per detected violation."""
    __tablename__ = "violation_records"

    session_id = Column(UUID(as_uuid=True), ForeignKey("detection_sessions.id", ondelete="CASCADE"), nullable=False)
    vehicle_id = Column(UUID(as_uuid=True), ForeignKey("detected_vehicles.id", ondelete="SET NULL"), nullable=True)

    # Classification
    violation_category = Column(Enum(ViolationCategory), nullable=False)
    vehicle_type = Column(Enum(VehicleType))
    severity = Column(Enum(SeverityLevel), nullable=False)

    # Scoring
    confidence = Column(Float, nullable=False)
    sub_violations = Column(JSON)  # Multi-label: ["no_helmet", "no_seatbelt"]

    # License plate (denormalized for fast lookup)
    plate_number = Column(String(20), index=True)

    # Evidence
    evidence_image_path = Column(String(512))
    evidence_generated_at = Column(DateTime(timezone=True))
    thumbnail_path = Column(String(512))

    # Location
    location_label = Column(String(255))
    camera_id = Column(String(100))

    # Challan / enforcement
    challan_issued = Column(Boolean, default=False)
    challan_id = Column(String(100))

    # Extra metadata per violation type
    extra_data = Column(JSON)  # e.g. {"rider_count": 3, "parking_duration_s": 420}

    session = relationship("DetectionSession", back_populates="violations")

    __table_args__ = (
        Index("ix_violation_category", "violation_category"),
        Index("ix_violation_plate", "plate_number"),
        Index("ix_violation_created", "created_at"),
        Index("ix_violation_camera", "camera_id"),
        Index("ix_violation_severity", "severity"),
    )


# ─── Analytics Snapshot ────────────────────────────────────────────────────────

class DailyAnalyticsSnapshot(Base, UUIDMixin):
    """Pre-aggregated daily analytics for fast dashboard queries."""
    __tablename__ = "daily_analytics_snapshots"

    snapshot_date = Column(DateTime(timezone=True), nullable=False, unique=True)
    camera_id = Column(String(100))

    total_violations = Column(Integer, default=0)
    total_sessions = Column(Integer, default=0)
    total_vehicles = Column(Integer, default=0)

    # Per-category counts (JSON for flexibility)
    violation_counts_by_category = Column(JSON, default=dict)
    violation_counts_by_vehicle = Column(JSON, default=dict)
    violation_counts_by_hour = Column(JSON, default=dict)  # {"0": 5, "1": 3, ...}
    top_offending_plates = Column(JSON, default=list)

    avg_confidence = Column(Float)
    avg_processing_time_ms = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_analytics_date", "snapshot_date"),
    )


# ─── Repeat Offender ──────────────────────────────────────────────────────────

class RepeatOffender(Base, UUIDMixin, TimestampMixin):
    """Tracks vehicles with multiple violations."""
    __tablename__ = "repeat_offenders"

    plate_number = Column(String(20), nullable=False, unique=True, index=True)
    total_violations = Column(Integer, default=1)
    last_violation_at = Column(DateTime(timezone=True))
    last_violation_category = Column(Enum(ViolationCategory))
    risk_score = Column(Float, default=0.0)  # Computed risk
    flagged = Column(Boolean, default=False)
    violation_history = Column(JSON, default=list)
