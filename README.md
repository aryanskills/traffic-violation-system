# 🚦 AI Traffic Violation Detection System

> **Production-grade, hackathon-ready** automatic traffic violation detection using YOLOv11, ByteTrack, EasyOCR, FastAPI, PostgreSQL, and React.

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Folder Structure](#folder-structure)
4. [Violation Types](#violation-types)
5. [Technology Stack](#technology-stack)
6. [Database Schema](#database-schema)
7. [API Reference](#api-reference)
8. [Data Flow](#data-flow)
9. [Setup & Installation](#setup--installation)
10. [Running the System](#running-the-system)
11. [Model Selection Rationale](#model-selection-rationale)
12. [Performance Evaluation](#performance-evaluation)
13. [Scalability Plan](#scalability-plan)
14. [Hackathon MVP Roadmap](#hackathon-mvp-roadmap)
15. [Demo Flow](#demo-flow)

---

## System Overview

The **AI Traffic Violation Detection System** analyzes traffic surveillance images and automatically detects, classifies, and records traffic violations in real time.

| Capability | Detail |
|---|---|
| Vehicle detection | 7 classes (car, truck, bus, motorcycle, bicycle, auto-rickshaw, pedestrian) |
| Violation detection | 7 types (helmet, seatbelt, triple riding, wrong side, stop line, red light, parking) |
| Tracking | ByteTrack multi-object tracking (ID persistence across frames) |
| OCR | Indian license plate recognition (EasyOCR / PaddleOCR) |
| Image preprocessing | Low light · deblur · rain · shadow · normalization |
| Evidence generation | Annotated JPEG with bounding boxes, labels, timestamps |
| Analytics | Daily/weekly/monthly trends, heatmaps, repeat offenders |
| Export | PDF · CSV · Excel reports |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         React Dashboard (Port 3000)                │
│   Upload Page │ Dashboard │ Violations Table │ Reports Export      │
└────────────────────┬───────────────────────────────────────────────┘
                     │  HTTP / REST
┌────────────────────▼───────────────────────────────────────────────┐
│                    FastAPI Backend (Port 8000)                      │
│  /detect  │  /violations  │  /analytics  │  /reports               │
└────────────────────┬───────────────────────────────────────────────┘
                     │
         ┌───────────▼────────────────────────────────┐
         │         Detection Pipeline                  │
         │                                             │
         │  1. ImagePreprocessor (OpenCV)              │
         │     ├── Low Light (CLAHE)                   │
         │     ├── Deblur (unsharp masking)            │
         │     ├── Rain removal (guided filter)        │
         │     └── Shadow removal (HSV normalization)  │
         │                                             │
         │  2. VehicleDetector (YOLOv11 + ByteTrack)   │
         │     └── 7 vehicle / road-user classes       │
         │                                             │
         │  3. ViolationEngine                         │
         │     ├── HelmetViolationDetector             │
         │     ├── SeatbeltViolationDetector           │
         │     ├── TripleRidingDetector                │
         │     ├── WrongSideDrivingDetector            │
         │     ├── StopLineViolationDetector           │
         │     ├── RedLightViolationDetector           │
         │     └── IllegalParkingDetector              │
         │                                             │
         │  4. LicensePlateRecognizer (EasyOCR)        │
         │     ├── YOLO plate detection                │
         │     ├── Plate crop + enhancement            │
         │     └── OCR + Indian format validation      │
         │                                             │
         │  5. EvidenceGenerator (OpenCV)              │
         │     └── Annotated JPEG + thumbnail          │
         └───────────┬────────────────────────────────┘
                     │
         ┌───────────▼────────────────────────────────┐
         │       PostgreSQL Database                   │
         │  detection_sessions │ detected_vehicles     │
         │  license_plates     │ violation_records     │
         │  daily_analytics    │ repeat_offenders      │
         └────────────────────────────────────────────┘
```

---

## Folder Structure

```
project/
├── backend/
│   ├── app/
│   │   ├── main.py                          # FastAPI app + lifespan
│   │   ├── core/
│   │   │   ├── config.py                    # All settings (pydantic-settings)
│   │   │   ├── database.py                  # SQLAlchemy async engine + sessions
│   │   │   └── logging.py                   # Structured logging
│   │   ├── models/
│   │   │   └── violation.py                 # SQLAlchemy ORM models (full schema)
│   │   ├── schemas/
│   │   │   └── violation.py                 # Pydantic v2 request/response schemas
│   │   ├── preprocessing/
│   │   │   └── image_processor.py           # Full preprocessing pipeline
│   │   ├── services/
│   │   │   ├── detection/
│   │   │   │   ├── vehicle_detector.py      # YOLOv11 + ByteTrack detector
│   │   │   │   └── pipeline.py              # End-to-end orchestrator
│   │   │   ├── violation/
│   │   │   │   └── violation_engine.py      # All 7 violation detectors + engine
│   │   │   ├── ocr/
│   │   │   │   └── plate_recognizer.py      # 2-stage plate OCR pipeline
│   │   │   ├── evidence/
│   │   │   │   └── evidence_generator.py    # Annotated evidence image generator
│   │   │   └── analytics/
│   │   │       └── analytics_service.py     # Analytics queries + PDF/CSV/Excel export
│   │   ├── api/
│   │   │   └── endpoints/
│   │   │       ├── health.py                # GET /health
│   │   │       ├── detection.py             # POST /detect, GET /sessions/{id}
│   │   │       ├── violations.py            # GET /violations, /violations/{id}
│   │   │       ├── analytics.py             # GET /analytics/summary, /daily, /hourly
│   │   │       └── reports.py              # GET /reports/export/{format}
│   │   ├── evaluation/
│   │   │   └── evaluator.py                 # mAP, OCR metrics, system benchmarks
│   │   └── utils/
│   ├── tests/
│   │   ├── unit/
│   │   │   ├── test_vehicle_detector.py
│   │   │   ├── test_violation_engine.py
│   │   │   ├── test_ocr.py
│   │   │   └── test_evaluator.py
│   │   └── integration/
│   ├── requirements.txt
│   └── pytest.ini
├── frontend/
│   ├── src/
│   │   ├── App.tsx                          # Router + sidebar layout
│   │   ├── main.tsx                         # React entry point
│   │   ├── index.css                        # Tailwind base + custom classes
│   │   ├── types/api.ts                     # TypeScript interfaces
│   │   ├── services/api.ts                  # Axios API layer
│   │   └── pages/
│   │       ├── DashboardPage.tsx            # Analytics charts (Recharts)
│   │       ├── UploadPage.tsx               # Image upload + result display
│   │       ├── ViolationsPage.tsx           # Searchable violations table
│   │       └── ReportsPage.tsx              # PDF/CSV/Excel export
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Violation Types

| # | Violation | Detection Method | Severity |
|---|---|---|---|
| 1 | **Helmet Non-Compliance** | Crop rider ROI → head region analysis / helmet YOLO model | High |
| 2 | **Seatbelt Non-Compliance** | Driver window crop → Hough diagonal strap line detection | High |
| 3 | **Triple Riding** | Motorcycle ROI → blob count ≥ 3 | High |
| 4 | **Wrong Side Driving** | Farneback optical flow → vehicle direction vs. traffic flow | Critical |
| 5 | **Stop Line Violation** | Hough horizontal line → vehicle bottom below line | Medium–High |
| 6 | **Red Light Violation** | HSV signal detection (RED) + optical flow movement | Critical |
| 7 | **Illegal Parking** | ByteTrack ID + stationary duration timer (default 5 min) | Medium–High |

---

## Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Object Detection** | YOLOv11 (Ultralytics) | Best-in-class speed/accuracy tradeoff; supports custom classes; integrated ByteTrack |
| **Multi-object Tracking** | ByteTrack (via Ultralytics) | SOTA tracker; handles occlusion; maintains IDs across frames with minimal overhead |
| **Optical Flow** | OpenCV Farneback | Dense flow for wrong-side and red-light motion analysis; no extra model needed |
| **OCR** | EasyOCR | Works well on Indian plates; GPU+CPU support; easy integration |
| **Image Preprocessing** | OpenCV + NumPy | CLAHE, guided filter, morphological transforms — all battle-tested |
| **Backend** | FastAPI + SQLAlchemy 2.0 | Async-native; auto-generated OpenAPI docs; type-safe |
| **Database** | PostgreSQL + asyncpg | ACID; excellent JSON support; rich indexing for analytics queries |
| **Frontend** | React + Vite + Tailwind | Fast build; minimal config; Recharts for analytics visualization |
| **Containerization** | Docker Compose | One-command full stack deployment |
| **Report Export** | ReportLab (PDF) + openpyxl (Excel) | Production-proven Python libraries |

### Why YOLOv11 over alternatives?

- **vs. YOLOv8**: YOLOv11 achieves higher mAP with fewer parameters (~22% reduction). Faster on CPU.
- **vs. DETR / RT-DETR**: YOLOv11 is 3–5× faster at inference; DETR needs more VRAM.
- **vs. EfficientDet**: YOLOv11 has much better Python ecosystem and Ultralytics support.
- **ByteTrack integration**: Built into Ultralytics, enabling `.track()` with one line.

### Why EasyOCR over PaddleOCR?

- Simpler pip install; no C++ PaddlePaddle dependency.
- Good accuracy on English alphanumeric (Indian plates).
- GPU-optional; works well CPU-only for hackathon.
- PaddleOCR is available as a drop-in swap via `OCR_ENGINE=paddleocr`.

---

## Database Schema

```sql
-- Core tables

detection_sessions
  id UUID PK, original_filename, file_path, status, camera_id, location_label,
  processing_time_ms, total_vehicles_detected, total_violations_detected,
  created_at, updated_at

detected_vehicles
  id UUID PK, session_id FK, track_id, vehicle_type, confidence,
  bbox_x1/y1/x2/y2, is_stationary, stationary_since, speed_estimate_kmh

license_plates
  id UUID PK, vehicle_id FK, raw_text, normalized_text, is_valid_format,
  detection_confidence, ocr_confidence, bbox_x1/y1/x2/y2

violation_records
  id UUID PK, session_id FK, vehicle_id FK,
  violation_category, vehicle_type, severity, confidence,
  sub_violations (JSON), plate_number (indexed),
  evidence_image_path, location_label, camera_id,
  challan_issued, metadata (JSON), created_at

daily_analytics_snapshots
  id UUID PK, snapshot_date (unique), camera_id,
  total_violations, total_sessions,
  violation_counts_by_category (JSON),
  violation_counts_by_hour (JSON),
  top_offending_plates (JSON)

repeat_offenders
  id UUID PK, plate_number (unique, indexed),
  total_violations, last_violation_at, risk_score, flagged
```

**Key indexes:**
- `violation_records(plate_number)` — fast plate lookup
- `violation_records(created_at)` — date range queries
- `violation_records(camera_id)` — per-camera analytics
- `violation_records(violation_category)` — category filtering

---

## API Reference

### Detection

```
POST   /api/v1/detect
  Form: file (image), location_label?, camera_id?
  → DetectionSessionResponse

GET    /api/v1/sessions/{session_id}
  → DetectionSessionResponse
```

### Violations

```
GET    /api/v1/violations
  Query: start_date, end_date, violation_category, vehicle_type,
         severity, plate_number, camera_id, min_confidence, page, page_size
  → ViolationListResponse

GET    /api/v1/violations/{id}
  → ViolationRecordOut

GET    /api/v1/violations/plates/{plate_number}
  → ViolationListResponse
```

### Analytics

```
GET    /api/v1/analytics/summary?start_date&end_date&camera_id
  → AnalyticsSummary (violations by category, hourly, daily trend, top plates)

GET    /api/v1/analytics/daily?days=30
GET    /api/v1/analytics/hourly?days=7
GET    /api/v1/analytics/top-plates?days=30
```

### Reports

```
GET    /api/v1/reports/export/csv?start_date&end_date
GET    /api/v1/reports/export/excel?start_date&end_date
GET    /api/v1/reports/export/pdf?start_date&end_date
  → file download (Content-Disposition: attachment)
```

### System

```
GET    /api/v1/health
  → { status, version, environment, models, database, uptime_seconds }
```

Full interactive docs: `http://localhost:8000/docs`

---

## Data Flow

```
User uploads image
        │
        ▼
POST /api/v1/detect
        │
        ▼
┌─────────────────────────────────────┐
│  DetectionPipeline.process_image()  │
│                                     │
│  1. Decode bytes → np.ndarray       │
│  2. ImagePreprocessor.process()     │
│     ├── Low light CLAHE             │
│     ├── Unsharp mask deblur         │
│     ├── Guided filter rain removal  │
│     ├── HSV shadow normalization    │
│     └── Resize + denoise pad        │
│                                     │
│  3. VehicleDetector.detect()        │
│     └── YOLOv11 → DetectionResult  │
│                                     │
│  4. ViolationEngine.run()           │
│     ├── Helmet detector             │
│     ├── Seatbelt detector           │
│     ├── Triple riding detector      │
│     ├── Wrong side detector         │
│     ├── Stop line detector          │
│     ├── Red light detector          │
│     └── Parking detector            │
│     → List[ViolationResult]         │
│                                     │
│  5. LicensePlateRecognizer.recognize│
│     ├── YOLO plate detect / heuristic│
│     ├── Enhance crop (CLAHE + thresh)│
│     └── EasyOCR → normalized plate  │
│                                     │
│  6. EvidenceGenerator.generate()    │
│     └── Annotated JPEG saved        │
│                                     │
│  7. DB: session + vehicles +        │
│         violations + plates saved   │
│                                     │
│  8. Return DetectionSessionResponse │
└─────────────────────────────────────┘
        │
        ▼
React frontend displays:
  - Evidence image with annotations
  - Violation list (category, severity, confidence)
  - Vehicle list
  - Processing time
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 15+ (or Docker)
- Git

### Option A — Docker (Recommended for Hackathon Demo)

```bash
git clone <repo-url>
cd project

# Copy and configure environment
cp .env.example .env
# Edit .env if needed (defaults work for local demo)

# Build and start all services
docker compose up --build

# Access:
#   Frontend:  http://localhost:3000
#   API docs:  http://localhost:8000/docs
#   Backend:   http://localhost:8000
```

### Option B — Manual Setup

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up PostgreSQL database
createdb traffic_violations

# Copy env file
cp ../.env.example ../.env
# Edit DATABASE_URL in .env to match your PostgreSQL credentials

# Run the backend
uvicorn app.main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000
```

### Run Tests

```bash
cd backend
pytest tests/unit/ -v

# With coverage report
pytest tests/unit/ -v --cov=app --cov-report=html
# Open htmlcov/index.html
```

---

## Running the System

### Adding YOLO Models

The system downloads `yolov8n.pt` as a fallback demo model. For production:

1. **Vehicle detection**: Use `yolov11n.pt` or a fine-tuned model from Ultralytics Hub.
2. **Helmet detection**: Fine-tune YOLOv11 on a helmet dataset (IDD, HelmetNet).
3. **Plate detection**: Fine-tune YOLOv11 on an Indian license plate dataset.

Place model files in `data/models/` and update `.env`:
```
YOLO_VEHICLE_MODEL=yolov11n.pt
YOLO_HELMET_MODEL=yolov11_helmet.pt
YOLO_PLATE_MODEL=yolov11_plate.pt
```

### GPU Acceleration

```bash
# In .env
YOLO_DEVICE=cuda   # NVIDIA GPU
# or
YOLO_DEVICE=mps    # Apple Silicon
```

---

## Performance Evaluation

The system includes a complete evaluation framework (`backend/app/evaluation/evaluator.py`):

### Detection Metrics
| Metric | Description |
|---|---|
| mAP@50 | Mean Average Precision at IoU 0.50 |
| mAP@50-95 | Mean AP across IoU 0.50–0.95 (COCO standard) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | Harmonic mean of precision and recall |

### OCR Metrics
| Metric | Description |
|---|---|
| Character Accuracy | 1 - Character Error Rate |
| Word Accuracy | Exact plate match rate |
| CER | Levenshtein distance / GT length |

### System Metrics
| Metric | Target |
|---|---|
| Inference time (CPU, YOLOv11n) | < 150ms/image |
| End-to-end pipeline | < 500ms/image |
| FPS (video stream) | 5–10 FPS on CPU, 30+ FPS on GPU |
| API response time | < 2 seconds |

Run benchmark:
```python
from app.evaluation.evaluator import BenchmarkRunner
runner = BenchmarkRunner()
report = runner.run_full_benchmark(pipeline, test_images, annotations, "data/benchmarks/report.json")
```

---

## Scalability Plan

### Horizontal Scaling

```
Load Balancer (Nginx)
    │
    ├── Backend Instance 1  (uvicorn, 4 workers)
    ├── Backend Instance 2  (uvicorn, 4 workers)
    └── Backend Instance N

Shared:
    PostgreSQL (primary + read replicas)
    Redis (cluster)
    Shared volume / S3 (evidence images)
```

### Async Processing (Production)

For high-throughput deployments, use Celery workers:

```
API receives image → queues Celery task → returns task_id
Worker pool → runs pipeline → saves results → Redis publish
Client polls /sessions/{id} → gets completed result
```

### Performance Targets

| Scenario | Configuration | Throughput |
|---|---|---|
| Single server (demo) | 1× CPU, 8GB RAM | ~2 images/sec |
| Production (small) | 4× CPU servers + 1 GPU | ~20 images/sec |
| Production (large) | GPU cluster + async queue | 100+ images/sec |

---

## Hackathon MVP Roadmap

### ✅ Phase 1 — Architecture & Core (Day 1)
- [x] Full system architecture design
- [x] Database schema
- [x] Docker Compose setup
- [x] FastAPI skeleton + health endpoint

### ✅ Phase 2 — Detection Engine (Day 1–2)
- [x] Image preprocessing pipeline
- [x] YOLOv11 vehicle detector
- [x] All 7 violation detectors
- [x] ByteTrack integration

### ✅ Phase 3 — OCR + Evidence (Day 2)
- [x] License plate detection + EasyOCR
- [x] Indian plate format validation
- [x] Evidence image generation
- [x] Database persistence

### ✅ Phase 4 — API + Frontend (Day 2–3)
- [x] All REST API endpoints
- [x] React dashboard with Recharts
- [x] Upload + detect page
- [x] Violations table with filters
- [x] Report export (PDF/CSV/Excel)

### ✅ Phase 5 — Tests + Evaluation (Day 3)
- [x] Unit tests (vehicle detector, violation engine, OCR, evaluator)
- [x] Evaluation framework (mAP, CER, system benchmarks)
- [x] Full Docker deployment

### 🔮 Future Enhancements
- [ ] Real-time video stream processing (RTSP)
- [ ] Mobile app (React Native) for field officers
- [ ] Challan generation + RTO integration API
- [ ] Face recognition for offender database
- [ ] Edge deployment (Jetson Nano, Raspberry Pi 5)
- [ ] Fine-tuned models for Indian roads (IDD dataset)

---

## Demo Flow

The complete hackathon demonstration follows these steps:

```
1.  Open http://localhost:3000 → Dashboard
    └── See live analytics: violations chart, hourly distribution, top plates

2.  Navigate to "Detect" tab
    └── Upload a traffic image (drag & drop)
    └── Optionally enter location + camera ID
    └── Click "Run Detection"

3.  System processes in < 2 seconds:
    ├── Preprocessing (CLAHE, deblur, rain, shadow)
    ├── YOLOv11 vehicle detection
    ├── 7 violation checks
    ├── License plate OCR
    └── Evidence image generation

4.  Results displayed:
    ├── Annotated evidence image with bounding boxes
    ├── List of detected violations (category, severity, confidence)
    ├── Detected vehicles list
    └── Processing time

5.  Navigate to "Violations" tab
    └── Searchable table of all violations
    └── Filter by category, severity, plate number, camera

6.  Navigate to "Reports" tab
    └── Set date range
    └── Export PDF / CSV / Excel

7.  Return to Dashboard
    └── Updated charts reflect new detections
    └── Repeat offender plate appears in top-plates widget
```

---

## Contributing

```bash
# Install dev dependencies
pip install -r requirements.txt
npm install  # in frontend/

# Linting
ruff check backend/
eslint frontend/src/

# Run all tests
cd backend && pytest tests/ -v
```

---

## License

MIT License — free for hackathon, academic, and commercial use.

---

## Team / Credits

Built as a **National-Level Hackathon** submission demonstrating production-grade AI integration for smart traffic enforcement.

**Stack:** YOLOv11 · ByteTrack · EasyOCR · FastAPI · PostgreSQL · React · Tailwind · Docker
