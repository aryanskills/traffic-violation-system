import axios from 'axios'
import type {
  DetectionSession, ViolationListResponse,
  AnalyticsSummary,
} from '../types/api'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 120_000,
})

// ── Detection ─────────────────────────────────────────────────────────────────

export async function detectViolations(
  file: File,
  locationLabel?: string,
  cameraId?: string,
): Promise<DetectionSession> {
  const form = new FormData()
  form.append('file', file)
  if (locationLabel) form.append('location_label', locationLabel)
  if (cameraId)      form.append('camera_id', cameraId)
  const { data } = await api.post<DetectionSession>('/detect', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function getSession(sessionId: string): Promise<DetectionSession> {
  const { data } = await api.get<DetectionSession>(`/sessions/${sessionId}`)
  return data
}

// ── Violations ────────────────────────────────────────────────────────────────

export async function listViolations(params: {
  start_date?: string
  end_date?: string
  violation_category?: string
  vehicle_type?: string
  severity?: string
  plate_number?: string
  camera_id?: string
  min_confidence?: number
  page?: number
  page_size?: number
}): Promise<ViolationListResponse> {
  const { data } = await api.get<ViolationListResponse>('/violations', { params })
  return data
}

export async function getViolationsByPlate(plate: string): Promise<ViolationListResponse> {
  const { data } = await api.get<ViolationListResponse>(`/violations/plates/${plate}`)
  return data
}

// ── Analytics ─────────────────────────────────────────────────────────────────

export async function getAnalyticsSummary(params?: {
  start_date?: string
  end_date?: string
  camera_id?: string
}): Promise<AnalyticsSummary> {
  const { data } = await api.get<AnalyticsSummary>('/analytics/summary', { params })
  return data
}

// ── Evaluation ────────────────────────────────────────────────────────────────

export async function getSystemMetrics() {
  const { data } = await api.get('/evaluation/metrics')
  return data
}

export async function getDetectionMetrics() {
  const { data } = await api.get('/evaluation/detection-metrics')
  return data
}

export async function getOcrMetrics() {
  const { data } = await api.get('/evaluation/ocr-accuracy')
  return data
}

// ── Reports ───────────────────────────────────────────────────────────────────

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a   = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

export async function exportCSV(params?: Record<string, string>) {
  const { data } = await api.get('/reports/export/csv', { params, responseType: 'blob' })
  downloadBlob(data, `violations_${Date.now()}.csv`)
}

export async function exportExcel(params?: Record<string, string>) {
  const { data } = await api.get('/reports/export/excel', { params, responseType: 'blob' })
  downloadBlob(data, `violations_${Date.now()}.xlsx`)
}

export async function exportPDF(params?: Record<string, string>) {
  const { data } = await api.get('/reports/export/pdf', { params, responseType: 'blob' })
  downloadBlob(data, `violation_report_${Date.now()}.pdf`)
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function getHealth() {
  const { data } = await api.get('/health')
  return data
}
