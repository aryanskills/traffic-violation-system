import React, { useEffect, useState } from 'react'
import { BarChart3, RefreshCw, Activity, Target, Cpu, Clock } from 'lucide-react'

export default function EvaluationPage() {
  const [metrics, setMetrics] = useState<any>(null)
  const [detMetrics, setDetMetrics] = useState<any>(null)
  const [ocrMetrics, setOcrMetrics] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  const load = async () => {
    setLoading(true)
    try {
      const [m, d, o] = await Promise.allSettled([
        fetch('/api/v1/evaluation/metrics').then(r => r.json()),
        fetch('/api/v1/evaluation/detection-metrics').then(r => r.json()),
        fetch('/api/v1/evaluation/ocr-accuracy').then(r => r.json()),
      ])
      if (m.status === 'fulfilled') setMetrics(m.value)
      if (d.status === 'fulfilled') setDetMetrics(d.value)
      if (o.status === 'fulfilled') setOcrMetrics(o.value)
    } finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  return (
    <div className="p-6 space-y-5 max-w-4xl mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">Performance Evaluation</h1>
          <p className="text-gray-500 text-xs mt-1">
            Detection accuracy, OCR metrics, system performance and scalability
          </p>
        </div>
        <button onClick={load}
          className="flex items-center gap-2 btn-secondary text-sm">
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''}/> Refresh
        </button>
      </div>

      {loading ? (
        <div className="text-center text-gray-500 py-12">Loading evaluation data…</div>
      ) : (
        <>
          {/* System Performance */}
          {metrics && (
            <div className="card space-y-4">
              <h2 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                <Cpu size={14} className="text-cyan-400"/> Computational Performance
              </h2>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                {[
                  { label: 'Avg Inference Time', value: `${metrics.detection.avg_processing_time_ms?.toFixed(0) ?? '—'}ms`, desc: 'End-to-end pipeline', color: 'text-cyan-400' },
                  { label: 'Throughput (FPS)',   value: `${metrics.detection.estimated_fps?.toFixed(2) ?? '—'}`, desc: 'Frames per second', color: 'text-green-400' },
                  { label: 'Sessions (24h)',     value: metrics.detection.sessions_processed ?? '—', desc: 'Images processed', color: 'text-blue-400' },
                  { label: 'Violations (24h)',   value: metrics.detection.total_violations ?? '—', desc: 'Detected', color: 'text-orange-400' },
                ].map(m => (
                  <div key={m.label} className="bg-gray-800/60 rounded-xl p-3">
                    <p className="text-xs text-gray-500">{m.label}</p>
                    <p className={`text-2xl font-black mt-1 ${m.color}`}>{m.value}</p>
                    <p className="text-xs text-gray-600 mt-0.5">{m.desc}</p>
                  </div>
                ))}
              </div>

              {/* Resource utilization */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-gray-400 mb-2 font-medium">Resource Utilization</p>
                  <div className="space-y-2.5">
                    {[
                      { label: 'CPU', value: metrics.system.cpu_percent, unit: '%', color: 'bg-blue-500' },
                      { label: 'RAM', value: metrics.system.memory_percent, unit: `% (${metrics.system.memory_mb?.toFixed(0)}MB)`, color: 'bg-purple-500' },
                      ...(metrics.system.gpu_percent !== null ? [
                        { label: 'GPU', value: metrics.system.gpu_percent, unit: '%', color: 'bg-green-500' }
                      ] : []),
                    ].map(r => (
                      <div key={r.label}>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-gray-400">{r.label}</span>
                          <span className="text-gray-300">{r.value?.toFixed(1)}{r.unit}</span>
                        </div>
                        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${r.color}`} style={{ width: `${Math.min(r.value,100)}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-xs text-gray-400 mb-2 font-medium">Scalability</p>
                  <div className="space-y-1.5 text-xs">
                    {[
                      { k: 'Deployment', v: 'Docker containerised — horizontal scaling ready' },
                      { k: 'API Workers', v: '2–4 uvicorn workers per container' },
                      { k: 'DB Connection Pool', v: '10 base + 20 overflow' },
                      { k: 'Async Architecture', v: 'FastAPI + asyncpg (non-blocking)' },
                      { k: 'Cache Layer', v: 'Redis for session & analytics cache' },
                      { k: 'Task Queue', v: 'Celery + Redis (async heavy jobs)' },
                      { k: 'GPU Support', v: 'CUDA / MPS via YOLO_DEVICE env var' },
                    ].map(r => (
                      <div key={r.k} className="flex gap-2 justify-between">
                        <span className="text-gray-500 flex-shrink-0">{r.k}</span>
                        <span className="text-gray-300 text-right">{r.v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Detection Metrics */}
          {detMetrics && (
            <div className="card space-y-4">
              <h2 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                <Target size={14} className="text-purple-400"/> Violation Detection Metrics
              </h2>
              {detMetrics.message ? (
                <p className="text-gray-500 text-sm">{detMetrics.message}</p>
              ) : (
                <>
                  <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                    {[
                      { label: 'Records Analysed', value: detMetrics.total_violations_analysed, color: 'text-gray-300' },
                      { label: 'Avg Confidence',   value: `${(detMetrics.avg_confidence*100).toFixed(1)}%`, color: 'text-blue-400' },
                      { label: 'Precision Est.',   value: `${(detMetrics.precision_estimate*100).toFixed(1)}%`, color: 'text-green-400' },
                    ].map(m => (
                      <div key={m.label} className="bg-gray-800/60 rounded-xl p-3">
                        <p className="text-xs text-gray-500">{m.label}</p>
                        <p className={`text-xl font-black mt-1 ${m.color}`}>{m.value}</p>
                      </div>
                    ))}
                  </div>

                  <div>
                    <p className="text-xs text-gray-400 mb-2 font-medium">Confidence Distribution</p>
                    <div className="grid grid-cols-3 gap-2 text-center">
                      {[
                        { label: 'High (≥70%)', value: detMetrics.confidence_distribution?.['high_0.7_plus'] ?? 0, color: 'text-green-400 bg-green-950' },
                        { label: 'Medium (50–70%)', value: detMetrics.confidence_distribution?.['medium_0.5_0.7'] ?? 0, color: 'text-yellow-400 bg-yellow-950' },
                        { label: 'Low (<50%)', value: detMetrics.confidence_distribution?.['low_below_0.5'] ?? 0, color: 'text-red-400 bg-red-950' },
                      ].map(b => (
                        <div key={b.label} className={`rounded-lg p-3 ${b.color.split(' ')[1]}`}>
                          <p className={`text-xl font-black ${b.color.split(' ')[0]}`}>{b.value}</p>
                          <p className="text-xs opacity-70 mt-1">{b.label}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {detMetrics.violations_by_category && (
                    <div>
                      <p className="text-xs text-gray-400 mb-2 font-medium">Violations by Category</p>
                      <div className="space-y-1.5">
                        {Object.entries(detMetrics.violations_by_category as Record<string,number>)
                          .sort(([,a],[,b]) => b-a)
                          .map(([cat, count]) => {
                            const max = Math.max(...Object.values(detMetrics.violations_by_category as Record<string,number>))
                            return (
                              <div key={cat} className="flex items-center gap-2">
                                <span className="text-xs text-gray-400 w-44 truncate capitalize">{cat.replace(/_/g,' ')}</span>
                                <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                                  <div className="h-full bg-blue-500 rounded-full" style={{ width: `${(count/max)*100}%` }} />
                                </div>
                                <span className="text-xs text-gray-300 w-8 text-right font-mono">{count}</span>
                              </div>
                            )
                          })}
                      </div>
                    </div>
                  )}

                  <div className="bg-gray-800/40 rounded-lg p-3 text-xs text-gray-400">
                    ℹ️ {detMetrics.note}
                  </div>
                </>
              )}
            </div>
          )}

          {/* OCR Metrics */}
          {ocrMetrics && (
            <div className="card space-y-4">
              <h2 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                <Activity size={14} className="text-amber-400"/> License Plate OCR Metrics
              </h2>
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                {[
                  { label: 'Plate Recognition Acc.', value: `${(ocrMetrics.plate_recognition_accuracy*100).toFixed(1)}%`, color: 'text-green-400' },
                  { label: 'Character Accuracy',     value: `${(ocrMetrics.character_accuracy*100).toFixed(1)}%`, color: 'text-blue-400' },
                  { label: 'Word Accuracy',          value: `${(ocrMetrics.word_accuracy*100).toFixed(1)}%`, color: 'text-purple-400' },
                  { label: 'Character Error Rate',   value: `${(ocrMetrics.character_error_rate*100).toFixed(1)}%`, color: 'text-red-400' },
                  { label: 'Word Error Rate',        value: `${(ocrMetrics.word_error_rate*100).toFixed(1)}%`, color: 'text-orange-400' },
                  { label: 'Sample Size',            value: ocrMetrics.sample_size, color: 'text-gray-300' },
                ].map(m => (
                  <div key={m.label} className="bg-gray-800/60 rounded-xl p-3">
                    <p className="text-xs text-gray-500">{m.label}</p>
                    <p className={`text-xl font-black mt-1 ${m.color}`}>{m.value}</p>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-500">📝 {ocrMetrics.note}</p>
            </div>
          )}

          {/* mAP explanation */}
          <div className="card border border-blue-900/30">
            <h2 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
              <BarChart3 size={14} className="text-blue-400"/> mAP / Precision / Recall Evaluation
            </h2>
            <div className="space-y-2 text-xs text-gray-400">
              <p>Full <strong className="text-gray-300">mAP@50</strong> and <strong className="text-gray-300">mAP@50-95</strong> evaluation requires annotated ground-truth bounding boxes.
              The <code className="text-blue-300 font-mono">BenchmarkRunner</code> class in <code className="text-blue-300 font-mono">backend/app/evaluation/evaluator.py</code> supports this.</p>
              <div className="bg-gray-800 rounded-lg p-3 font-mono text-xs text-green-300 mt-2">
                {`from app.evaluation.evaluator import BenchmarkRunner\nrunner = BenchmarkRunner()\nreport = runner.run_full_benchmark(\n  pipeline, test_images, annotations,\n  "data/benchmarks/report.json"\n)`}
              </div>
              <p>The report JSON contains: mAP@50, mAP@50-95, precision, recall, F1, CER, WER, FPS, CPU/GPU usage.</p>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
