import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  Upload, Loader2, CheckCircle2, AlertTriangle, Car,
  Eye, Shield, Cpu, Clock, MapPin, Hash
} from 'lucide-react'
import toast from 'react-hot-toast'
import { detectViolations } from '../services/api'
import type { DetectionSession, ViolationDetected, VehicleDetection } from '../types/api'

const SEVERITY_COLORS: Record<string, string> = {
  critical: 'bg-red-950 border-red-700 text-red-300',
  high:     'bg-red-900 border-red-700 text-red-300',
  medium:   'bg-yellow-900 border-yellow-700 text-yellow-300',
  low:      'bg-blue-900 border-blue-700 text-blue-300',
}

const VEHICLE_ICONS: Record<string, string> = {
  car: '🚗', truck: '🚛', bus: '🚌', motorcycle: '🏍️',
  bicycle: '🚲', auto_rickshaw: '🛺', pedestrian: '🚶', unknown: '🚘',
}

const VIOLATION_ICONS: Record<string, string> = {
  helmet_non_compliance: '⛑️',
  seatbelt_non_compliance: '🔒',
  triple_riding: '👥',
  wrong_side_driving: '⚠️',
  stop_line_violation: '🛑',
  red_light_violation: '🔴',
  illegal_parking: '🅿️',
}

function ViolationCard({ v, index }: { v: ViolationDetected; index: number }) {
  const cat = v.violation_category.replace(/_/g, ' ')
  const icon = VIOLATION_ICONS[v.violation_category] || '⚠️'
  return (
    <div className={`border rounded-xl px-4 py-3 text-sm ${SEVERITY_COLORS[v.severity] || 'bg-gray-800 text-gray-300 border-gray-700'}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-start gap-2">
          <span className="text-lg mt-0.5">{icon}</span>
          <div>
            <p className="font-bold uppercase tracking-wide text-xs">{cat}</p>
            <p className="text-xs opacity-70 mt-0.5 capitalize">
              {v.vehicle_type?.replace(/_/g, ' ') || 'Unknown vehicle'}
              {v.plate_number && (
                <span className="ml-2 font-mono bg-black/30 px-1 rounded">{v.plate_number}</span>
              )}
            </p>
            {v.sub_violations?.length > 0 && (
              <p className="text-xs opacity-50 mt-0.5">{v.sub_violations.join(' · ')}</p>
            )}
          </div>
        </div>
        <div className="text-right flex-shrink-0">
          <p className="text-xl font-black">{(v.confidence * 100).toFixed(0)}%</p>
          <p className="text-xs opacity-60 capitalize font-medium">{v.severity}</p>
        </div>
      </div>
    </div>
  )
}

function VehicleRow({ v }: { v: VehicleDetection }) {
  const icon = VEHICLE_ICONS[v.vehicle_type] || '🚘'
  const label = v.vehicle_type.replace(/_/g, ' ')
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-800 last:border-0">
      <div className="flex items-center gap-2">
        <span>{icon}</span>
        <span className="text-sm text-gray-300 capitalize">{label}</span>
        {v.track_id != null && (
          <span className="text-xs text-gray-600 font-mono">#{v.track_id}</span>
        )}
        {v.is_stationary && (
          <span className="text-xs bg-yellow-900 text-yellow-300 px-1.5 py-0.5 rounded">stationary</span>
        )}
      </div>
      <div className="flex items-center gap-3 text-xs text-gray-400">
        <span className="font-mono">
          {Math.round(v.bbox.x1)},{Math.round(v.bbox.y1)} → {Math.round(v.bbox.x2)},{Math.round(v.bbox.y2)}
        </span>
        <span className="font-bold text-gray-300">{(v.confidence * 100).toFixed(0)}%</span>
      </div>
    </div>
  )
}

function PreprocessingBadge({ step }: { step: string }) {
  const labels: Record<string, string> = {
    low_light_clahe_gamma: '💡 Low-light enhanced',
    deblur_unsharp: '🔍 Deblurred',
    rain_removal: '🌧️ Rain removed',
    shadow_normalisation: '🌑 Shadow removed',
    clahe: '🎨 CLAHE contrast',
    resize_pad: '📐 Normalised',
    upscale_low_res: '⬆️ Upscaled',
  }
  return (
    <span className="text-xs bg-gray-800 text-gray-300 px-2 py-1 rounded-full border border-gray-700">
      {labels[step] || step}
    </span>
  )
}

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [location, setLocation] = useState('')
  const [cameraId, setCameraId] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<DetectionSession | null>(null)
  const [showPreprocessed, setShowPreprocessed] = useState(false)
  const [activeTab, setActiveTab] = useState<'evidence'|'vehicles'|'preprocessing'>('evidence')

  const onDrop = useCallback((accepted: File[]) => {
    const f = accepted[0]
    if (!f) return
    setFile(f); setResult(null); setActiveTab('evidence')
    const reader = new FileReader()
    reader.onload = e => setPreview(e.target?.result as string)
    reader.readAsDataURL(f)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [], 'image/webp': [] },
    maxFiles: 1, maxSize: 50 * 1024 * 1024,
  })

  const handleDetect = async () => {
    if (!file) return
    setLoading(true)
    try {
      const res = await detectViolations(file, location || undefined, cameraId || undefined)
      setResult(res)
      setActiveTab('evidence')
      if (res.total_violations_detected > 0) {
        toast.error(`${res.total_violations_detected} violation(s) detected!`)
      } else {
        toast.success('No violations detected.')
      }
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Detection failed')
    } finally {
      setLoading(false)
    }
  }

  const preprocessing = result?.preprocessing_steps

  return (
    <div className="p-6 space-y-6 max-w-6xl mx-auto">
      <div>
        <h1 className="text-xl font-bold text-white">Detect Violations</h1>
        <p className="text-gray-400 text-sm mt-1">
          Upload a traffic image — the system preprocesses, detects, classifies violations, reads plates, and generates annotated evidence.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Left: Upload ── */}
        <div className="space-y-4">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
              isDragActive ? 'border-blue-500 bg-blue-950/20' : 'border-gray-700 hover:border-gray-500'
            }`}
          >
            <input {...getInputProps()} />
            {preview ? (
              <img src={preview} alt="Preview" className="max-h-52 mx-auto rounded-lg object-contain" />
            ) : (
              <div className="space-y-2 py-4">
                <Upload size={36} className="mx-auto text-gray-500" />
                <p className="text-gray-400 text-sm">
                  {isDragActive ? 'Drop image here…' : 'Drag & drop or click to upload'}
                </p>
                <p className="text-gray-600 text-xs">JPEG · PNG · WEBP · Max 50MB</p>
              </div>
            )}
          </div>

          {file && (
            <p className="text-xs text-gray-500 text-center">
              {file.name} — {(file.size / 1024).toFixed(0)} KB
            </p>
          )}

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-gray-400 flex items-center gap-1 mb-1">
                <MapPin size={11}/> Location
              </label>
              <input type="text" value={location} onChange={e => setLocation(e.target.value)}
                placeholder="e.g. MG Road Junction"
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400 flex items-center gap-1 mb-1">
                <Hash size={11}/> Camera ID
              </label>
              <input type="text" value={cameraId} onChange={e => setCameraId(e.target.value)}
                placeholder="e.g. CAM-007"
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          <button onClick={handleDetect} disabled={!file || loading}
            className="w-full btn-primary flex items-center justify-center gap-2 py-3 text-base font-semibold disabled:opacity-40">
            {loading
              ? <><Loader2 size={18} className="animate-spin" /> Analyzing image…</>
              : <><Cpu size={18} /> Run AI Detection</>
            }
          </button>

          {/* Pipeline steps legend */}
          <div className="card text-xs text-gray-500 space-y-1">
            <p className="text-gray-400 font-medium mb-2">Detection pipeline:</p>
            {[
              '1. Image quality assessment + enhancement',
              '2. YOLOv8/v11 vehicle & person detection',
              '3. AI violation analysis (IoU-based)',
              '4. License plate OCR (EasyOCR ensemble)',
              '5. Annotated evidence generation',
            ].map((s, i) => (
              <p key={i}>{s}</p>
            ))}
          </div>
        </div>

        {/* ── Right: Results ── */}
        <div className="space-y-4">
          {!result && !loading && (
            <div className="card flex flex-col items-center justify-center py-16 text-gray-600 gap-3">
              <Eye size={40} className="opacity-30" />
              <p className="text-sm">Results appear here after detection</p>
            </div>
          )}

          {result && (
            <>
              {/* Stat row */}
              <div className="grid grid-cols-4 gap-2">
                {[
                  { label: 'Vehicles', value: result.total_vehicles_detected, color: 'text-blue-400' },
                  { label: 'Violations', value: result.total_violations_detected,
                    color: result.total_violations_detected > 0 ? 'text-red-400' : 'text-green-400' },
                  { label: 'Persons', value: result.vehicles.filter(v=>v.vehicle_type==='pedestrian').length, color: 'text-purple-400' },
                  { label: 'Time',
                    value: result.processing_time_ms ? `${result.processing_time_ms.toFixed(0)}ms` : '—',
                    color: 'text-cyan-400' },
                ].map(s => (
                  <div key={s.label} className="card text-center py-3">
                    <p className={`text-xl font-black ${s.color}`}>{s.value}</p>
                    <p className="text-xs text-gray-500 mt-0.5">{s.label}</p>
                  </div>
                ))}
              </div>

              {/* Tab bar */}
              <div className="flex gap-1 bg-gray-900 rounded-lg p-1 border border-gray-800">
                {([
                  { key: 'evidence', label: 'Evidence', icon: Eye },
                  { key: 'vehicles', label: `Vehicles (${result.vehicles.length})`, icon: Car },
                  { key: 'preprocessing', label: 'Preprocessing', icon: Shield },
                ] as const).map(t => (
                  <button key={t.key} onClick={() => setActiveTab(t.key)}
                    className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-medium transition-colors ${
                      activeTab === t.key
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}>
                    <t.icon size={13}/>{t.label}
                  </button>
                ))}
              </div>

              {/* Tab: Evidence */}
              {activeTab === 'evidence' && (
                <div className="space-y-3">
                  {result.evidence_image_url ? (
                    <div className="card p-0 overflow-hidden rounded-xl border border-gray-700">
                      <img
                        src={result.evidence_image_url}
                        alt="Annotated evidence"
                        className="w-full object-contain"
                        style={{ maxHeight: '320px' }}
                        onError={e => {
                          (e.target as HTMLImageElement).style.display = 'none'
                        }}
                      />
                      <div className="flex items-center justify-between px-4 py-2 bg-gray-900 border-t border-gray-800">
                        <span className="text-xs text-gray-400">
                          Annotated evidence image with bounding boxes
                        </span>
                        <a href={result.evidence_image_url} download
                          className="text-xs text-blue-400 hover:underline font-medium">
                          ⬇ Download
                        </a>
                      </div>
                    </div>
                  ) : (
                    <div className="card text-center text-gray-500 text-sm py-6">
                      Evidence image not yet available
                    </div>
                  )}

                  {result.violations.length > 0 ? (
                    <div className="space-y-2">
                      <h3 className="text-sm font-semibold text-red-400 flex items-center gap-1.5">
                        <AlertTriangle size={14}/> Violations Detected
                      </h3>
                      {result.violations.map((v, i) => (
                        <ViolationCard key={i} v={v} index={i} />
                      ))}
                    </div>
                  ) : (
                    <div className="card flex items-center gap-3 text-green-400 border-green-900">
                      <CheckCircle2 size={22}/>
                      <div>
                        <p className="font-semibold text-sm">No violations detected</p>
                        <p className="text-xs text-green-600">All checks passed for this image</p>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Tab: Vehicles */}
              {activeTab === 'vehicles' && (
                <div className="card space-y-1">
                  <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
                    <Car size={14}/> All Detected Objects
                  </h3>
                  {result.vehicles.length === 0 ? (
                    <p className="text-gray-500 text-sm text-center py-4">No objects detected</p>
                  ) : (
                    result.vehicles.map((v, i) => <VehicleRow key={i} v={v} />)
                  )}
                  <div className="pt-2 mt-2 border-t border-gray-800">
                    <p className="text-xs text-gray-600">
                      Bounding box coordinates shown in pixels on 640×640 preprocessed image.
                      Persons are detected at lower confidence threshold to enable rider/driver association.
                    </p>
                  </div>
                </div>
              )}

              {/* Tab: Preprocessing */}
              {activeTab === 'preprocessing' && (
                <div className="space-y-3">
                  <div className="card">
                    <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
                      <Shield size={14}/> Image Preprocessing Applied
                    </h3>
                    {preprocessing && preprocessing.length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {preprocessing.map((s, i) => <PreprocessingBadge key={i} step={s} />)}
                      </div>
                    ) : (
                      <div className="space-y-2 text-xs text-gray-400">
                        <p>The following preprocessing steps run on every image:</p>
                        <div className="grid grid-cols-2 gap-2 mt-2">
                          {[
                            { icon: '💡', label: 'Low-light CLAHE', desc: 'Boosts dark images via LAB colour space' },
                            { icon: '🌧️', label: 'Rain removal', desc: 'Guided filter suppresses streaks' },
                            { icon: '🌑', label: 'Shadow removal', desc: 'HSV normalisation flattens shadows' },
                            { icon: '🔍', label: 'Deblurring', desc: 'Unsharp mask for motion/focus blur' },
                            { icon: '🎨', label: 'CLAHE contrast', desc: 'Applied to all images for consistency' },
                            { icon: '📐', label: 'Letterbox resize', desc: 'Padded to 640×640 for YOLO' },
                          ].map((item, i) => (
                            <div key={i} className="bg-gray-800 rounded-lg p-2">
                              <p className="font-medium text-gray-300">{item.icon} {item.label}</p>
                              <p className="text-gray-500 text-xs mt-0.5">{item.desc}</p>
                            </div>
                          ))}
                        </div>
                        <p className="text-gray-600 mt-2">
                          Steps are applied selectively based on quality assessment (brightness, blur score, rain detection).
                        </p>
                      </div>
                    )}
                  </div>

                  {result.preprocessed_image_url && (
                    <div className="card p-0 overflow-hidden rounded-xl border border-gray-700">
                      <div className="px-3 py-2 bg-gray-900 border-b border-gray-800 flex items-center justify-between">
                        <span className="text-xs text-gray-400 font-medium">Preprocessed image (fed to YOLO)</span>
                        <a href={result.preprocessed_image_url} download
                          className="text-xs text-blue-400 hover:underline">⬇ Download</a>
                      </div>
                      <img
                        src={result.preprocessed_image_url}
                        alt="Preprocessed"
                        className="w-full object-contain max-h-48"
                      />
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
