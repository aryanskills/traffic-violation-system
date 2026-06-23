"""
AI Traffic Violation Detection System — FastAPI Application Entry Point.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import create_tables
from app.core.logging import configure_logging, get_logger
from app.api.endpoints import detection, violations, analytics, reports, health, evaluation

configure_logging()
logger = get_logger(__name__)

APP_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown."""
    logger.info("=" * 60)
    logger.info(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"  Environment: {settings.ENVIRONMENT}")
    logger.info("=" * 60)

    # Ensure all data directories exist (important inside Docker)
    for d in [settings.UPLOAD_DIR, settings.EVIDENCE_DIR,
              settings.REPORTS_DIR, settings.MODEL_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("Data directories verified.")

    # Initialize DB tables
    logger.info("Initializing database tables…")
    await create_tables()

    # Warm up pipeline (loads YOLO models + OCR)
    logger.info("Warming up detection pipeline…")
    from app.services.detection.pipeline import pipeline
    pipeline.warm_up()

    logger.info("✅  Application ready.")
    yield
    logger.info("Shutting down…")


# ─── App Factory ──────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade AI system for automatic traffic violation detection.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
PREFIX = "/api/v1"
app.include_router(health.router,     prefix=PREFIX, tags=["Health"])
app.include_router(detection.router,  prefix=PREFIX, tags=["Detection"])
app.include_router(violations.router, prefix=PREFIX, tags=["Violations"])
app.include_router(analytics.router,  prefix=PREFIX, tags=["Analytics"])
app.include_router(reports.router,     prefix=PREFIX, tags=["Reports"])
app.include_router(evaluation.router,  prefix=PREFIX, tags=["Evaluation"])


# ─── Static file mounts (after dirs are guaranteed to exist via lifespan) ────
# We mount lazily via startup event so dirs exist before mount
@app.on_event("startup")
async def mount_static():
    for d in [settings.UPLOAD_DIR, settings.EVIDENCE_DIR, settings.REPORTS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
    app.mount("/api/v1/evidence", StaticFiles(directory=str(settings.EVIDENCE_DIR)), name="evidence")
    app.mount("/api/v1/uploads",  StaticFiles(directory=str(settings.UPLOAD_DIR)),   name="uploads")
    app.mount("/api/v1/reports",  StaticFiles(directory=str(settings.REPORTS_DIR)),  name="reports")


# ─── Global Exception Handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal server error", "detail": str(exc)},
    )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "status": "running",
    }
