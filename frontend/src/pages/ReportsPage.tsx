import React, { useState, useEffect } from 'react'
import { FileText, Download, Loader2, BarChart3, TrendingUp, AlertTriangle, RefreshCw } from 'lucide-react'
import toast from 'react-hot-toast'
import { exportCSV, exportExcel, exportPDF } from '../services/api'

type Format = 'pdf' | 'csv' | 'xlsx'

const FORMATS: { key: Format; label: string; desc: string; color: string; icon: string }[] = [
  { key: 'pdf',  label: 'PDF Report',   desc: 'Summary with charts, category breakdown & violation table', color: 'border-red-800 hover:border-red-500',   icon: '📄' },
  { key: 'csv',  label: 'CSV Export',   desc: 'Raw records: all columns, all violations, easily imported', color: 'border-green-800 hover:border-green-500', icon: '📊' },
  { key: 'xlsx', label: 'Excel Export', desc: 'Formatted workbook with styled headers & data rows',        color: 'border-blue-800 hover:border-blue-500',  icon: '📋' },
]

interface PerfMetrics {
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
    device: string
  }
  thresholds: {
    yolo_confidence: number
    violation_min_confidence: number
    ocr_confidence: number
  }
}

export default function ReportsPage() {
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate]     = useState('')
  const [loading, setLoading]     = useState<Format | null>(null)
  const [metrics, setMetrics]     = useState<PerfMetrics | null>(null)
  const [metricsLoading, setMetricsLoading] = useState(true)

  const loadMetrics = async () => {
    setMetricsLoading(true)
    try {
      const r = await fetch('/api/v1/evaluation/metrics')
      if (r.ok) setMetrics(await r.json())
    } catch { /* non-fatal */ }
    finally { setMetricsLoading(false) }
  }

  useEffect(() => { loadMetrics() }, [])

  const handleExport = async (format: Format) => {
    setLoading(format)
    const params: Record<string,string> = {}
    if (startDate) params.start_date = new Date(startDate).toISOString()
    if (endDate)   params.end_date   = new Date(endDate).toISOString()
    try {
      if (format === 'csv')  await exportCSV(params)
      if (format === 'xlsx') await exportExcel(params)
      if (format === 'pdf')  await exportPDF(params)
      toast.success(`${format.toUpperCase()} downloaded`)
    } catch { toast.error(`${format.toUpperCase()} export failed`) }
    finally { setLoading(null) }
  }

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <h1 className="text-xl font-bold text-white">Reports & Performance Evaluation</h1>

      {/* Performance evaluation panel */}
      <div className="card space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
            <BarChart3 size={15} className="text-blue-400"/> System Performance Metrics
          </h2>
          <button onClick={loadMetrics}
            className="p-1.5 bg-gray-800 rounded hover:bg-gray-700 transition-colors">
            <RefreshCw size={13} className={metricsLoading ? 'animate-spin text-blue-400' : 'text-gray-400'} />
          </button>
        </div>

        {metricsLoading ? (
          <div className="text-center text-gray-500 text-sm py-4">Loading metrics…</div>
        ) : metrics ? (
          <>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              {[
                { label: 'Avg Processing Time', value: `${metrics.detection.avg_processing_time_ms.toFixed(0)}ms`, color: 'text-cyan-400', icon: '⚡' },
                { label: 'Estimated FPS',        value: `${metrics.detection.estimated_fps.toFixed(1)}`, color: 'text-green-400', icon: '🎬' },
                { label: 'Avg Confidence',       value: `${(metrics.detection.avg_confidence*100).toFixed(1)}%`, color: 'text-purple-400', icon: '🎯' },
                { label: '24h Violations',       value: metrics.detection.total_violations, color: 'text-orange-400', icon: '⚠️' },
              ].map(m => (
                <div key={m.label} className="bg-gray-800 rounded-xl p-3">
                  <p className="text-xs text-gray-500">{m.icon} {m.label}</p>
                  <p className={`text-xl font-black mt-1 ${m.color}`}>{m.value}</p>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Detection confidence range */}
              <div>
                <p className="text-xs text-gray-400 mb-2 font-medium">Confidence Range (24h)</p>
                <div className="space-y-2">
                  {[
                    { label: 'Min', value: metrics.detection.min_confidence, color: 'bg-red-500' },
                    { label: 'Avg', value: metrics.detection.avg_confidence, color: 'bg-yellow-500' },
                    { label: 'Max', value: metrics.detection.max_confidence, color: 'bg-green-500' },
                  ].map(c => (
                    <div key={c.label} className="flex items-center gap-2">
                      <span className="text-xs text-gray-500 w-8">{c.label}</span>
                      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div className={`h-full rounded-full ${c.color}`} style={{ width: `${c.value*100}%` }} />
                      </div>
                      <span className="text-xs text-gray-300 font-mono w-10 text-right">{(c.value*100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* System resources */}
              <div>
                <p className="text-xs text-gray-400 mb-2 font-medium">System Resources</p>
                <div className="space-y-2 text-xs">
                  {[
                    { label: 'CPU Usage', value: `${metrics.system.cpu_percent.toFixed(1)}%`, bar: metrics.system.cpu_percent/100 },
                    { label: 'Memory',    value: `${metrics.system.memory_mb.toFixed(0)} MB (${metrics.system.memory_percent.toFixed(1)}%)`, bar: metrics.system.memory_percent/100 },
                    { label: 'Device',    value: metrics.system.device.toUpperCase(), bar: 0 },
                  ].map(r => (
                    <div key={r.label} className="flex items-center gap-2">
                      <span className="text-gray-500 w-20 flex-shrink-0">{r.label}</span>
                      {r.bar > 0 ? (
                        <>
                          <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500 rounded-full" style={{ width: `${r.bar*100}%` }} />
                          </div>
                          <span className="text-gray-300 font-mono w-24 text-right">{r.value}</span>
                        </>
                      ) : (
                        <span className="text-gray-300 font-mono">{r.value}</span>
                      )}
                    </div>
                  ))}
                  {metrics.system.gpu_percent !== null && (
                    <div className="flex items-center gap-2">
                      <span className="text-gray-500 w-20">GPU Usage</span>
                      <span className="text-gray-300 font-mono">{metrics.system.gpu_percent}%</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Active thresholds */}
            <div className="bg-gray-800/50 rounded-lg p-3 text-xs">
              <p className="text-gray-400 mb-2 font-medium">Active Detection Thresholds</p>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { k: 'YOLO Confidence', v: metrics.thresholds.yolo_confidence },
                  { k: 'Min Violation Conf', v: metrics.thresholds.violation_min_confidence },
                  { k: 'OCR Confidence', v: metrics.thresholds.ocr_confidence },
                ].map(t => (
                  <div key={t.k} className="text-center bg-gray-800 rounded p-2">
                    <p className="text-gray-500">{t.k}</p>
                    <p className="text-blue-400 font-bold mt-0.5">{(t.v*100).toFixed(0)}%</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Accuracy note */}
            <div className="bg-blue-950/30 border border-blue-900/50 rounded-lg p-3 text-xs text-blue-300">
              <p className="font-medium mb-1">📊 Precision / Recall / mAP</p>
              <p className="text-blue-400/80">
                Full mAP@50 and mAP@50-95 evaluation requires annotated ground-truth datasets.
                The BenchmarkRunner in <code className="font-mono">app/evaluation/evaluator.py</code> computes
                these when you provide labelled test images. See <code className="font-mono">GET /api/v1/evaluation/detection-metrics</code> for
                confidence-distribution based estimates on your recorded violations.
              </p>
            </div>
          </>
        ) : (
          <p className="text-gray-500 text-xs">Could not load metrics. Ensure backend is running.</p>
        )}
      </div>

      {/* Report export */}
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <TrendingUp size={15} className="text-green-400"/> Export Reports
        </h2>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-gray-400 block mb-1">From Date</label>
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">To Date</label>
            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>
        <p className="text-xs text-gray-600">Leave empty to export the last 30 days of data.</p>

        <div className="space-y-2">
          {FORMATS.map(f => (
            <button key={f.key} onClick={() => handleExport(f.key)}
              disabled={!!loading}
              className={`w-full card border-2 text-left flex items-center gap-4 transition-all disabled:opacity-50 ${f.color}`}>
              <span className="text-2xl flex-shrink-0">{f.icon}</span>
              <div className="flex-1 min-w-0">
                <p className="font-semibold text-white text-sm">{f.label}</p>
                <p className="text-xs text-gray-400 mt-0.5">{f.desc}</p>
              </div>
              {loading === f.key
                ? <Loader2 size={18} className="animate-spin text-gray-400 flex-shrink-0"/>
                : <Download size={16} className="text-gray-500 flex-shrink-0"/>
              }
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
