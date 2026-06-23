import React, { useEffect, useState, useCallback } from 'react'
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import {
  AlertTriangle, Car, Camera, TrendingUp, RefreshCw,
  Shield, Zap, Activity, Clock, MapPin,
} from 'lucide-react'
import { getAnalyticsSummary } from '../services/api'
import type { AnalyticsSummary } from '../types/api'

const VIOLATION_COLORS: Record<string, string> = {
  helmet_non_compliance:    '#ef4444',
  seatbelt_non_compliance:  '#f97316',
  triple_riding:            '#eab308',
  wrong_side_driving:       '#8b5cf6',
  stop_line_violation:      '#06b6d4',
  red_light_violation:      '#dc2626',
  illegal_parking:          '#6b7280',
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

function StatCard({ label, value, icon: Icon, color, sub }: {
  label: string; value: string | number; icon: React.FC<any>; color: string; sub?: string
}) {
  return (
    <div className="card flex items-center gap-4">
      <div className={`p-3 rounded-xl ${color} flex-shrink-0`}>
        <Icon size={20} className="text-white" />
      </div>
      <div className="min-w-0">
        <p className="text-gray-400 text-xs truncate">{label}</p>
        <p className="text-2xl font-black text-white">{value}</p>
        {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}

const TOOLTIP_STYLE = {
  contentStyle: { background: '#111827', border: '1px solid #374151', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: '#e5e7eb' },
}

type Period = '7d' | '30d' | '90d'

export default function DashboardPage() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [period, setPeriod] = useState<Period>('30d')
  const [sysMetrics, setSysMetrics] = useState<any>(null)

  const DAYS: Record<Period, number> = { '7d': 7, '30d': 30, '90d': 90 }

  const load = useCallback(async () => {
    setLoading(true); setError(null)
    const end = new Date()
    const start = new Date(end.getTime() - DAYS[period] * 86400_000)
    try {
      const [sumData, sysData] = await Promise.allSettled([
        getAnalyticsSummary({
          start_date: start.toISOString(),
          end_date: end.toISOString(),
        }),
        fetch('/api/v1/evaluation/metrics').then(r => r.json()),
      ])
      if (sumData.status === 'fulfilled') setSummary(sumData.value)
      else setError('Failed to load analytics')
      if (sysData.status === 'fulfilled') setSysMetrics(sysData.value)
    } finally {
      setLoading(false)
    }
  }, [period])

  useEffect(() => { load() }, [load])

  if (loading) return (
    <div className="flex items-center justify-center h-full gap-3 text-gray-400">
      <RefreshCw size={18} className="animate-spin" />
      <span className="text-sm">Loading analytics…</span>
    </div>
  )

  if (error || !summary) return (
    <div className="flex flex-col items-center justify-center h-full gap-3 text-gray-500">
      <AlertTriangle size={32} className="text-red-500 opacity-50" />
      <p className="text-sm">{error || 'No data available yet. Upload and analyse images to see statistics.'}</p>
      <button onClick={load} className="btn-secondary text-xs px-3 py-1.5">Retry</button>
    </div>
  )

  const pieData = summary.violations_by_category.map(c => ({
    name: c.category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    value: c.count,
    pct: c.percentage,
    color: VIOLATION_COLORS[c.category] || '#6b7280',
    icon: VIOLATION_ICONS[c.category] || '⚠️',
  }))

  const vehicleData = Object.entries(summary.violations_by_vehicle_type)
    .map(([type, count]) => ({ type: type.replace(/_/g, ' '), count }))
    .sort((a, b) => b.count - a.count)

  return (
    <div className="p-6 space-y-5 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">Analytics Dashboard</h1>
          <p className="text-gray-500 text-xs mt-0.5">
            {new Date(summary.period_start).toLocaleDateString()} – {new Date(summary.period_end).toLocaleDateString()}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {(['7d','30d','90d'] as Period[]).map(p => (
            <button key={p} onClick={() => setPeriod(p)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                period === p ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}>
              {p}
            </button>
          ))}
          <button onClick={load}
            className="p-1.5 bg-gray-800 rounded-lg text-gray-400 hover:text-white transition-colors">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard label="Total Violations"    value={summary.total_violations}
          icon={AlertTriangle} color="bg-red-700"
          sub={`${period} period`} />
        <StatCard label="Sessions Processed"  value={summary.total_sessions}
          icon={Camera} color="bg-blue-700"
          sub="images analysed" />
        <StatCard label="Vehicles Detected"   value={summary.total_vehicles_detected}
          icon={Car} color="bg-green-700"
          sub="across all sessions" />
        <StatCard label="Avg Confidence"      value={`${(summary.avg_confidence * 100).toFixed(1)}%`}
          icon={Shield} color="bg-purple-700"
          sub="violation score" />
      </div>

      {/* System metrics row */}
      {sysMetrics && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <div className="card py-3">
            <p className="text-xs text-gray-500 flex items-center gap-1"><Zap size={11}/> Avg Processing</p>
            <p className="text-lg font-bold text-cyan-400 mt-1">
              {sysMetrics.detection?.avg_processing_time_ms?.toFixed(0) || '—'}ms
            </p>
          </div>
          <div className="card py-3">
            <p className="text-xs text-gray-500 flex items-center gap-1"><Activity size={11}/> Throughput</p>
            <p className="text-lg font-bold text-cyan-400 mt-1">
              {sysMetrics.detection?.estimated_fps?.toFixed(1) || '—'} FPS
            </p>
          </div>
          <div className="card py-3">
            <p className="text-xs text-gray-500 flex items-center gap-1"><Clock size={11}/> 24h Violations</p>
            <p className="text-lg font-bold text-orange-400 mt-1">
              {sysMetrics.detection?.total_violations ?? '—'}
            </p>
          </div>
          <div className="card py-3">
            <p className="text-xs text-gray-500 flex items-center gap-1"><Zap size={11}/> Device</p>
            <p className="text-lg font-bold text-gray-300 mt-1 uppercase">
              {sysMetrics.system?.device || 'cpu'}
            </p>
          </div>
        </div>
      )}

      {/* Charts row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Daily Trend */}
        <div className="card lg:col-span-2">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">Daily Violations Trend</h2>
          {summary.daily_trend.length === 0 ? (
            <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
              No data for selected period
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={summary.daily_trend}>
                <defs>
                  <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="date" tick={{ fill:'#6b7280', fontSize:10 }}
                  tickFormatter={d => d.slice(5)} />
                <YAxis tick={{ fill:'#6b7280', fontSize:10 }} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Area type="monotone" dataKey="count" stroke="#3b82f6"
                  fill="url(#grad)" strokeWidth={2} name="Violations" />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Violation Category Pie */}
        <div className="card">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">By Category</h2>
          {pieData.length === 0 ? (
            <div className="h-40 flex items-center justify-center text-gray-600 text-xs">
              No violations recorded yet
            </div>
          ) : (
            <>
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%"
                    innerRadius={42} outerRadius={70}
                    dataKey="value" paddingAngle={2}>
                    {pieData.map((e, i) => <Cell key={i} fill={e.color} />)}
                  </Pie>
                  <Tooltip {...TOOLTIP_STYLE}
                    formatter={(val: any, name: any, props: any) =>
                      [`${val} (${props.payload.pct}%)`, name]} />
                </PieChart>
              </ResponsiveContainer>
              <div className="space-y-1 mt-1">
                {pieData.map(d => (
                  <div key={d.name} className="flex items-center gap-2 text-xs">
                    <div className="w-2 h-2 rounded-sm flex-shrink-0" style={{background:d.color}} />
                    <span className="text-gray-400 truncate flex-1">{d.icon} {d.name}</span>
                    <span className="text-gray-300 font-semibold">{d.value}</span>
                    <span className="text-gray-600">({d.pct}%)</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Charts row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Hourly */}
        <div className="card">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">Peak Violation Hours</h2>
          {summary.hourly_distribution.length === 0 ? (
            <div className="h-40 flex items-center justify-center text-gray-600 text-xs">No data</div>
          ) : (
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={summary.hourly_distribution} barSize={8}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="hour" tick={{ fill:'#6b7280', fontSize:9 }}
                  tickFormatter={h => `${h}h`} />
                <YAxis tick={{ fill:'#6b7280', fontSize:9 }} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Bar dataKey="count" fill="#3b82f6" radius={[2,2,0,0]} name="Violations" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Vehicle type breakdown */}
        <div className="card">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">Violations by Vehicle Type</h2>
          {vehicleData.length === 0 ? (
            <div className="h-40 flex items-center justify-center text-gray-600 text-xs">No data</div>
          ) : (
            <div className="space-y-2">
              {vehicleData.slice(0, 6).map(v => {
                const max = vehicleData[0].count
                return (
                  <div key={v.type}>
                    <div className="flex justify-between text-xs mb-0.5">
                      <span className="text-gray-400 capitalize">{v.type}</span>
                      <span className="text-gray-300 font-medium">{v.count}</span>
                    </div>
                    <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500 rounded-full"
                        style={{ width: `${(v.count / max) * 100}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Repeat offenders + high-risk cameras */}
        <div className="card space-y-4">
          <div>
            <h2 className="text-sm font-semibold text-gray-300 mb-2">🔁 Repeat Offenders</h2>
            {summary.top_offending_plates.length === 0 ? (
              <p className="text-gray-600 text-xs">None found</p>
            ) : (
              <div className="space-y-1.5">
                {summary.top_offending_plates.slice(0, 5).map((p, i) => (
                  <div key={p.plate} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-gray-600 text-xs w-3">{i+1}</span>
                      <span className="font-mono text-blue-300 text-xs">{p.plate}</span>
                    </div>
                    <span className={`text-xs px-1.5 py-0.5 rounded-full font-semibold ${
                      p.count >= 5 ? 'bg-red-900 text-red-300'
                      : p.count >= 3 ? 'bg-yellow-900 text-yellow-300'
                      : 'bg-gray-800 text-gray-400'}`}>
                      {p.count}×
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="border-t border-gray-800 pt-3">
            <h2 className="text-sm font-semibold text-gray-300 mb-2">📍 High-Risk Cameras</h2>
            {summary.high_risk_cameras.length === 0 ? (
              <p className="text-gray-600 text-xs">No camera data</p>
            ) : (
              <div className="space-y-1.5">
                {summary.high_risk_cameras.slice(0, 4).map(c => (
                  <div key={c.camera_id} className="flex items-center justify-between">
                    <span className="text-gray-400 text-xs flex items-center gap-1">
                      <MapPin size={10}/> {c.camera_id}
                    </span>
                    <span className="text-xs text-orange-400 font-medium">{c.count} violations</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
