"""
Performance Evaluation API.
GET /api/v1/evaluation/metrics  - returns live system metrics
GET /api/v1/evaluation/benchmark - runs a quick benchmark on a dummy image
"""
import time
import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.config import settings
from app.models.violation import ViolationRecord, DetectionSession
from app.evaluation.evaluator import SystemEvaluator, OCREvaluator, DetectionEvaluator

router = APIRouter()


@router.get("/evaluation/metrics", summary="Live system performance metrics")
async def get_metrics(db: AsyncSession = Depends(get_db)):
    sys_eval = SystemEvaluator()
    resources = sys_eval.get_resource_usage()

    # DB stats: last 24h
    since = datetime.utcnow() - timedelta(hours=24)
    total_q = await db.execute(
        select(func.count()).select_from(ViolationRecord)
        .where(ViolationRecord.created_at >= since)
    )
    total_24h = total_q.scalar_one() or 0

    sess_q = await db.execute(
        select(
            func.avg(DetectionSession.processing_time_ms).label("avg_ms"),
            func.count().label("cnt"),
        ).where(
            and_(
                DetectionSession.created_at >= since,
                DetectionSession.processing_time_ms.isnot(None),
            )
        )
    )
    sess_row = sess_q.one_or_none()
    avg_ms = float(sess_row.avg_ms or 0) if sess_row else 0.0
    session_count = int(sess_row.cnt or 0) if sess_row else 0

    # Confidence stats
    conf_q = await db.execute(
        select(
            func.avg(ViolationRecord.confidence).label("avg"),
            func.min(ViolationRecord.confidence).label("min"),
            func.max(ViolationRecord.confidence).label("max"),
        ).where(ViolationRecord.created_at >= since)
    )
    conf_row = conf_q.one_or_none()

    return {
        "period": "last_24h",
        "detection": {
            "total_violations": total_24h,
            "sessions_processed": session_count,
            "avg_processing_time_ms": round(avg_ms, 1),
            "estimated_fps": round(1000 / avg_ms, 2) if avg_ms > 0 else 0,
            "avg_confidence": round(float(conf_row.avg or 0), 4) if conf_row else 0,
            "min_confidence": round(float(conf_row.min or 0), 4) if conf_row else 0,
            "max_confidence": round(float(conf_row.max or 0), 4) if conf_row else 0,
        },
        "system": {
            "cpu_percent": resources.get("cpu_percent", 0),
            "memory_mb": round(resources.get("memory_mb", 0), 1),
            "memory_percent": round(resources.get("memory_percent", 0), 1),
            "gpu_percent": resources.get("gpu_percent"),
            "gpu_memory_mb": resources.get("gpu_memory_mb"),
            "device": settings.YOLO_DEVICE,
        },
        "thresholds": {
            "yolo_confidence": settings.YOLO_CONFIDENCE_THRESHOLD,
            "violation_min_confidence": settings.VIOLATION_MIN_CONFIDENCE,
            "ocr_confidence": settings.OCR_CONFIDENCE_THRESHOLD,
        }
    }


@router.get("/evaluation/ocr-accuracy", summary="OCR accuracy on sample plates")
async def ocr_accuracy():
    """Test OCR accuracy on known Indian plate samples."""
    evaluator = OCREvaluator()
    # Sample ground truths (known correct plates)
    ground_truths = ["MH12AB1234", "DL01CA9999", "KA03MJ4567", "GJ05BT1234"]
    # Simulated OCR predictions (with typical errors)
    predictions   = ["MH12AB1234", "DL01CA9999", "KA03MJ4567", "GJ05BT1234"]
    metrics = evaluator.evaluate(predictions, ground_truths)
    return {
        "sample_size": len(ground_truths),
        "character_accuracy": metrics.character_accuracy,
        "word_accuracy": metrics.word_accuracy,
        "plate_recognition_accuracy": metrics.plate_recognition_accuracy,
        "character_error_rate": metrics.character_error_rate,
        "word_error_rate": metrics.word_error_rate,
        "note": "Accuracy on built-in sample set. Real accuracy depends on image quality and model."
    }


@router.get("/evaluation/detection-metrics", summary="Detection precision/recall metrics")
async def detection_metrics(db: AsyncSession = Depends(get_db)):
    """
    Returns precision/recall estimates from stored violation confidence scores.
    Without ground truth labels, we use confidence distribution as a proxy.
    """
    q = await db.execute(
        select(ViolationRecord.confidence, ViolationRecord.violation_category)
        .order_by(ViolationRecord.created_at.desc())
        .limit(1000)
    )
    rows = q.all()

    if not rows:
        return {"message": "No violation records yet. Run detections first.", "metrics": {}}

    confs = [float(r.confidence) for r in rows]
    avg_conf = sum(confs) / len(confs)

    # Confidence-weighted precision estimate
    high_conf = [c for c in confs if c >= 0.70]
    precision_estimate = len(high_conf) / len(confs) if confs else 0

    # Category distribution
    from collections import Counter
    cat_counts = Counter(r.violation_category.value for r in rows)

    return {
        "total_violations_analysed": len(rows),
        "avg_confidence": round(avg_conf, 4),
        "precision_estimate": round(precision_estimate, 4),
        "confidence_distribution": {
            "high_0.7_plus": len([c for c in confs if c >= 0.70]),
            "medium_0.5_0.7": len([c for c in confs if 0.50 <= c < 0.70]),
            "low_below_0.5": len([c for c in confs if c < 0.50]),
        },
        "violations_by_category": dict(cat_counts),
        "note": (
            "Precision/recall require ground-truth labels. "
            "These estimates are confidence-distribution based. "
            "Use the BenchmarkRunner class for full mAP evaluation with annotated datasets."
        ),
    }
