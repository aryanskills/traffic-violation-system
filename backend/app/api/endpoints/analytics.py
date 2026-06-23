"""Analytics API Endpoints."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.violation import AnalyticsSummary
from app.services.analytics.analytics_service import AnalyticsService

router = APIRouter()


@router.get(
    "/analytics/summary",
    response_model=AnalyticsSummary,
    summary="Dashboard analytics summary",
)
async def get_analytics_summary(
    start_date: Optional[datetime] = Query(None, description="Start of period (default: 30 days ago)"),
    end_date: Optional[datetime] = Query(None, description="End of period (default: now)"),
    camera_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    svc = AnalyticsService(db)
    return await svc.get_summary(
        start_date=start_date or datetime.utcnow() - timedelta(days=30),
        end_date=end_date or datetime.utcnow(),
        camera_id=camera_id,
    )


@router.get("/analytics/daily", summary="Daily violation counts for charting")
async def daily_violations(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    svc = AnalyticsService(db)
    summary = await svc.get_summary(
        start_date=datetime.utcnow() - timedelta(days=days),
        end_date=datetime.utcnow(),
    )
    return {"success": True, "data": summary.daily_trend}


@router.get("/analytics/hourly", summary="Hourly violation distribution")
async def hourly_violations(
    days: int = Query(default=7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    svc = AnalyticsService(db)
    summary = await svc.get_summary(
        start_date=datetime.utcnow() - timedelta(days=days),
        end_date=datetime.utcnow(),
    )
    return {"success": True, "data": summary.hourly_distribution}


@router.get("/analytics/top-plates", summary="Top repeat offending plates")
async def top_plates(
    days: int = Query(default=30),
    db: AsyncSession = Depends(get_db),
):
    svc = AnalyticsService(db)
    summary = await svc.get_summary(
        start_date=datetime.utcnow() - timedelta(days=days),
        end_date=datetime.utcnow(),
    )
    return {"success": True, "data": summary.top_offending_plates}
