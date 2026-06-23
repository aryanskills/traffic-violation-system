"""Report Export API Endpoints."""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.violation import ViolationCategory
from app.services.analytics.analytics_service import AnalyticsService

router = APIRouter()


@router.get("/reports/export/csv", summary="Export violations as CSV")
async def export_csv(
    start_date: Optional[datetime] = Query(default=None),
    end_date: Optional[datetime] = Query(default=None),
    violation_category: Optional[ViolationCategory] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    svc = AnalyticsService(db)
    filters = {}
    if violation_category:
        filters["violation_category"] = violation_category

    csv_bytes = await svc.export_csv(
        start_date=start_date or datetime.utcnow() - timedelta(days=30),
        end_date=end_date or datetime.utcnow(),
        filters=filters or None,
    )
    filename = f"violations_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/reports/export/excel", summary="Export violations as Excel")
async def export_excel(
    start_date: Optional[datetime] = Query(default=None),
    end_date: Optional[datetime] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    svc = AnalyticsService(db)
    try:
        xlsx_bytes = await svc.export_excel(
            start_date=start_date or datetime.utcnow() - timedelta(days=30),
            end_date=end_date or datetime.utcnow(),
        )
    except ImportError:
        raise HTTPException(status_code=501, detail="openpyxl not installed on server")

    filename = f"violations_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/reports/export/pdf", summary="Export summary report as PDF")
async def export_pdf(
    start_date: Optional[datetime] = Query(default=None),
    end_date: Optional[datetime] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    svc = AnalyticsService(db)
    try:
        pdf_bytes = await svc.export_pdf(
            start_date=start_date or datetime.utcnow() - timedelta(days=30),
            end_date=end_date or datetime.utcnow(),
        )
    except ImportError:
        raise HTTPException(status_code=501, detail="reportlab not installed on server")

    filename = f"violation_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
