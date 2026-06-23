export type VehicleType =
  | 'car' | 'truck' | 'bus' | 'motorcycle' | 'bicycle'
  | 'auto_rickshaw' | 'pedestrian' | 'unknown'

export type ViolationCategory =
  | 'helmet_non_compliance'
  | 'seatbelt_non_compliance'
  | 'triple_riding'
  | 'wrong_side_driving'
  | 'stop_line_violation'
  | 'red_light_violation'
  | 'illegal_parking'

export type SeverityLevel  = 'low' | 'medium' | 'high' | 'critical'
export type ProcessingStatus = 'pending' | 'processing' | 'completed' | 'failed'

export interface BoundingBox {
  x1: number; y1: number; x2: number; y2: number
  width: number; height: number
}

export interface VehicleDetection {
  track_id?: number
  vehicle_type: VehicleType
  confidence: number
  bbox: BoundingBox
  is_stationary: boolean
  speed_estimate_kmh?: number
}

export interface ViolationDetected {
  violation_category: ViolationCategory
  vehicle_type?: VehicleType
  severity: SeverityLevel
  confidence: number
  sub_violations: string[]
  plate_number?: string
  metadata: Record<string, unknown>
}

export interface DetectionSession {
  session_id: string
  status: ProcessingStatus
  original_filename: string
  location_label?: string
  camera_id?: string
  processing_time_ms?: number
  total_vehicles_detected: number
  total_violations_detected: number
  vehicles: VehicleDetection[]
  violations: ViolationDetected[]
  evidence_image_url?: string
  preprocessed_image_url?: string
  preprocessing_steps?: string[]
  preprocessing_quality_score?: number
  created_at: string
}

export interface ViolationRecord {
  id: string
  session_id: string
  violation_category: ViolationCategory
  vehicle_type?: VehicleType
  severity: SeverityLevel
  confidence: number
  plate_number?: string
  evidence_image_url?: string
  location_label?: string
  camera_id?: string
  challan_issued: boolean
  metadata: Record<string, unknown>
  created_at: string
}

export interface ViolationListResponse {
  success: boolean
  total: number
  page: number
  page_size: number
  items: ViolationRecord[]
}

export interface CategoryStat {
  category: ViolationCategory
  count: number
  percentage: number
}

export interface HourlyDistribution { hour: number; count: number }
export interface DailyTrend         { date: string; count: number }

export interface AnalyticsSummary {
  period_start: string
  period_end: string
  total_violations: number
  total_sessions: number
  total_vehicles_detected: number
  violations_by_category: CategoryStat[]
  violations_by_vehicle_type: Record<string, number>
  hourly_distribution: HourlyDistribution[]
  daily_trend: DailyTrend[]
  avg_confidence: number
  top_offending_plates: Array<{ plate: string; count: number }>
  high_risk_cameras: Array<{ camera_id: string; count: number }>
}

export interface SystemMetrics {
  period: string
  detection: {
    total_violations: number
    sessions_processed: number
    avg_processing_time_ms: number
    estimated_fps: number
    avg_confidence: number
    min_confidence: number
    max_confidence: number
  }
  system: {
    cpu_percent: number
    memory_mb: number
    memory_percent: number
    gpu_percent: number | null
    gpu_memory_mb: number | null
    device: string
  }
  thresholds: {
    yolo_confidence: number
    violation_min_confidence: number
    ocr_confidence: number
  }
}
