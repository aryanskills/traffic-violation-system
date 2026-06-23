"""Central configuration management."""

from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        case_sensitive=True, extra="ignore",
    )

    APP_NAME: str = "AI Traffic Violation Detection System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/traffic_violations"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 300

    UPLOAD_DIR: Path = Path("data/uploads")
    EVIDENCE_DIR: Path = Path("data/evidence")
    REPORTS_DIR: Path = Path("data/reports")
    MODEL_DIR: Path = Path("data/models")
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp", "video/mp4"]

    # ─── YOLO Models ────────────────────────────────────────────────────────────
    YOLO_VEHICLE_MODEL: str = "yolov11n.pt"
    YOLO_HELMET_MODEL: str = "yolov11_helmet.pt"
    YOLO_SEATBELT_MODEL: str = "yolov11_seatbelt.pt"
    YOLO_PLATE_MODEL: str = "yolov11_plate.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.35   # Lowered from 0.45 for better recall
    YOLO_IOU_THRESHOLD: float = 0.45
    YOLO_DEVICE: str = "cpu"
    YOLO_IMAGE_SIZE: int = 640

    # ─── Per-class thresholds ────────────────────────────────────────────────────
    PERSON_CONF_THRESHOLD: float = 0.28   # Even lower to catch riders/drivers

    # ─── ByteTrack ───────────────────────────────────────────────────────────────
    TRACKER_MAX_AGE: int = 30
    TRACKER_MIN_HITS: int = 3
    TRACKER_IOU_THRESHOLD: float = 0.3

    # ─── OCR ─────────────────────────────────────────────────────────────────────
    OCR_ENGINE: str = "easyocr"
    OCR_LANGUAGES: List[str] = ["en"]
    OCR_CONFIDENCE_THRESHOLD: float = 0.5
    PLATE_MIN_CONFIDENCE: float = 0.5

    # ─── Violation thresholds ────────────────────────────────────────────────────
    ILLEGAL_PARKING_DURATION_SECONDS: int = 300
    TRIPLE_RIDING_MIN_COUNT: int = 3
    VIOLATION_SEVERITY_HIGH_THRESHOLD: float = 0.80
    VIOLATION_SEVERITY_MED_THRESHOLD: float = 0.55
    VIOLATION_MIN_CONFIDENCE: float = 0.40   # Discard below this

    # ─── Helmet / Seatbelt fallback detector settings ────────────────────────────
    HELMET_SKIN_RATIO_THRESHOLD: float = 0.18
    SEATBELT_BELT_COLOUR_THRESHOLD: float = 0.45

    # ─── Evidence ────────────────────────────────────────────────────────────────
    EVIDENCE_JPEG_QUALITY: int = 95
    BBOX_LINE_THICKNESS: int = 2
    FONT_SCALE: float = 0.6

    # ─── Analytics ───────────────────────────────────────────────────────────────
    ANALYTICS_PAGE_SIZE: int = 50
    REPORT_RETENTION_DAYS: int = 90

    # ─── Celery ──────────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ─── Security ────────────────────────────────────────────────────────────────
    SECRET_KEY: str = "CHANGE-ME-IN-PRODUCTION-USE-RANDOM-64-CHARS"
    API_KEY_HEADER: str = "X-API-Key"
    ENABLE_API_KEY_AUTH: bool = False

    def ensure_directories(self):
        for d in [self.UPLOAD_DIR, self.EVIDENCE_DIR, self.REPORTS_DIR, self.MODEL_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
