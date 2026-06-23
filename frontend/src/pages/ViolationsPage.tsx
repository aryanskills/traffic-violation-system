import React, { useEffect, useState } from 'react'
import {
  Search, Filter, ChevronLeft, ChevronRight,
  Eye, Download, ExternalLink, AlertTriangle, Clock,
} from 'lucide-react'
import { listViolations } from '../services/api'
import type { ViolationRecord, ViolationCategory, SeverityLevel } from '../types/api'

const CATEGORIES: ViolationCategory[] = [
  'helmet_non_compliance','seatbelt_non_compliance','triple_riding',
  'wrong_side_driving','stop_line_violation','red_light_violation','illegal_parking',
]

const SEV_BADGE: Record<SeverityLevel, string> = {
  critical: 'bg-red-950 text-red-300 border border-red-800',
  high:     'bg-red-900 text-red-300 border border-red-800',
  medium:   'bg-yellow-900 text-yellow-300 border border-yellow-800',
  low:      'bg-blue-900 text-blue-300 border border-blue-800',
}

const CAT_ICONS: Record<string, string> = {
  helmet_non_compliance: '⛑️',
  seatbelt_non_compliance: '🔒',
  triple_riding: '👥',
  wrong_side_driving: '⚠️',
  stop_line_violation: '🛑',
  red_light_violation: '🔴',
  illegal_parking: '🅿️',
}

function ConfBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color = pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-400 font-mono">{pct}%</span>
    </div>
  )
}

export default function ViolationsPage() {
  const [items, setItems] = useState<ViolationRecord[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [plate, setPlate] = useState('')
  const [category, setCategory] = useState('')
  const [severity, setSeverity] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [selected, setSelected] = useState<ViolationRecord | null>(null)
  const PAGE_SIZE = 25

  const fetchData = async (p = page) => {
    setLoading(true)
    try {
      const res = await listViolations({
        plate_number: plate || undefined,
        violation_category: category || undefined,
        severity: severity || undefined,
        start_date: startDate ? new Date(startDate).toISOString() : undefined,
        end_date: endDate ? new Date(endDate).toISOString() : undefined,
        page: p, page_size: PAGE_SIZE,
      })
      setItems(res.items); setTotal(res.total)
    } finally { setLoading(false) }
  }

  useEffect(() => { fetchData() }, [page])

  const search = () => { setPage(1); fetchData(1) }
  const clear = () => {
    setPlate(''); setCategory(''); setSeverity(''); setStartDate(''); setEndDate('')
    setPage(1); setTimeout(() => fetchData(1), 0)
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* Main list */}
      <div className={`flex flex-col flex-1 p-6 space-y-4 overflow-auto ${selected ? 'lg:pr-0' : ''}`}>
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold text-white">Violations</h1>
          <span className="text-xs text-gray-500">{total} records</span>
        </div>

        {/* Filter row */}
        <div className="card flex flex-wrap gap-3 items-end">
          <div>
            <label className="text-xs text-gray-400 block mb-1">License Plate</label>
            <div className="relative">
              <Search size={13} className="absolute left-2.5 top-2.5 text-gray-500" />
              <input type="text" value={plate} onChange={e => setPlate(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && search()}
                placeholder="e.g. MH12AB"
                className="pl-8 pr-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-600 w-36 focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Category</label>
            <select value={category} onChange={e => setCategory(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none w-44">
              <option value="">All Categories</option>
              {CATEGORIES.map(c => (
                <option key={c} value={c}>
                  {CAT_ICONS[c]} {c.replace(/_/g,' ').replace(/\b\w/g,l=>l.toUpperCase())}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Severity</label>
            <select value={severity} onChange={e => setSeverity(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none w-28">
              <option value="">All</option>
              {['critical','high','medium','low'].map(s => (
                <option key={s} value={s}>{s.charAt(0).toUpperCase()+s.slice(1)}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">From</label>
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none" />
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">To</label>
            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none" />
          </div>
          <button onClick={search} className="btn-primary flex items-center gap-1.5 text-sm">
            <Filter size={13}/> Filter
          </button>
          <button onClick={clear} className="btn-secondary text-sm">Clear</button>
        </div>

        {/* Table */}
        <div className="card p-0 overflow-hidden flex-1">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800 bg-gray-950">
                  {['Category','Vehicle','Severity','Confidence','Plate #','Camera','Location','Timestamp','Evidence'].map(h => (
                    <th key={h} className="text-left px-3 py-3 text-xs text-gray-500 font-medium whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60">
                {loading ? (
                  <tr><td colSpan={9} className="text-center py-12 text-gray-500 text-sm">Loading…</td></tr>
                ) : items.length === 0 ? (
                  <tr><td colSpan={9} className="text-center py-12 text-gray-600 text-sm">
                    <AlertTriangle size={24} className="mx-auto mb-2 opacity-30" />
                    No violations found. Adjust filters or upload images.
                  </td></tr>
                ) : items.map(r => (
                  <tr key={r.id}
                    onClick={() => setSelected(r === selected ? null : r)}
                    className={`cursor-pointer transition-colors ${
                      selected?.id === r.id ? 'bg-blue-950/40' : 'hover:bg-gray-800/30'
                    }`}>
                    <td className="px-3 py-2.5 whitespace-nowrap">
                      <span className="text-gray-200 text-xs">
                        {CAT_ICONS[r.violation_category]} {r.violation_category.replace(/_/g,' ')}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-gray-400 text-xs capitalize whitespace-nowrap">
                      {r.vehicle_type?.replace(/_/g,' ') || '—'}
                    </td>
                    <td className="px-3 py-2.5 whitespace-nowrap">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${SEV_BADGE[r.severity]}`}>
                        {r.severity}
                      </span>
                    </td>
                    <td className="px-3 py-2.5">
                      <ConfBar value={r.confidence} />
                    </td>
                    <td className="px-3 py-2.5 whitespace-nowrap">
                      {r.plate_number
                        ? <span className="font-mono text-blue-300 text-xs bg-blue-950/40 px-1.5 py-0.5 rounded">{r.plate_number}</span>
                        : <span className="text-gray-600 text-xs">—</span>}
                    </td>
                    <td className="px-3 py-2.5 text-gray-500 text-xs">{r.camera_id || '—'}</td>
                    <td className="px-3 py-2.5 text-gray-500 text-xs max-w-28 truncate">{r.location_label || '—'}</td>
                    <td className="px-3 py-2.5 text-gray-500 text-xs whitespace-nowrap">
                      {new Date(r.created_at).toLocaleString()}
                    </td>
                    <td className="px-3 py-2.5">
                      {r.evidence_image_url ? (
                        <div className="flex gap-1.5">
                          <button onClick={e => { e.stopPropagation(); setSelected(r) }}
                            className="p-1 bg-gray-700 rounded hover:bg-blue-700 transition-colors">
                            <Eye size={12} />
                          </button>
                          <a href={r.evidence_image_url} download onClick={e => e.stopPropagation()}
                            className="p-1 bg-gray-700 rounded hover:bg-green-700 transition-colors">
                            <Download size={12} />
                          </a>
                        </div>
                      ) : (
                        <span className="text-gray-700 text-xs">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between px-4 py-3 border-t border-gray-800 bg-gray-950/50">
            <span className="text-xs text-gray-500">
              Showing {Math.min((page-1)*PAGE_SIZE+1, total)}–{Math.min(page*PAGE_SIZE, total)} of {total}
            </span>
            <div className="flex gap-2 items-center">
              <button onClick={() => setPage(p => Math.max(1,p-1))} disabled={page===1}
                className="p-1.5 rounded bg-gray-800 hover:bg-gray-700 disabled:opacity-30 transition-colors">
                <ChevronLeft size={14}/>
              </button>
              <span className="text-xs text-gray-400 px-1">Page {page}</span>
              <button onClick={() => setPage(p => p+1)} disabled={items.length < PAGE_SIZE}
                className="p-1.5 rounded bg-gray-800 hover:bg-gray-700 disabled:opacity-30 transition-colors">
                <ChevronRight size={14}/>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Detail panel */}
      {selected && (
        <div className="w-80 border-l border-gray-800 bg-gray-950 flex flex-col overflow-auto flex-shrink-0">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
            <h3 className="text-sm font-semibold text-white">Violation Detail</h3>
            <button onClick={() => setSelected(null)} className="text-gray-500 hover:text-white text-xs">✕</button>
          </div>

          {/* Evidence image */}
          {selected.evidence_image_url ? (
            <div className="p-3">
              <img src={selected.evidence_image_url} alt="Evidence"
                className="w-full rounded-lg object-contain border border-gray-800"
                style={{ maxHeight: 200 }} />
              <a href={selected.evidence_image_url} download
                className="mt-1 flex items-center justify-center gap-1.5 text-xs text-blue-400 hover:underline">
                <Download size={11}/> Download Evidence Image
              </a>
            </div>
          ) : (
            <div className="p-3 text-center text-gray-600 text-xs border-b border-gray-800">
              No evidence image
            </div>
          )}

          {/* Metadata */}
          <div className="p-4 space-y-3 border-t border-gray-800">
            {[
              { label: 'Category', value: `${CAT_ICONS[selected.violation_category]} ${selected.violation_category.replace(/_/g,' ')}` },
              { label: 'Vehicle Type', value: selected.vehicle_type?.replace(/_/g,' ') || '—' },
              { label: 'Severity', value: selected.severity.toUpperCase() },
              { label: 'Confidence', value: `${(selected.confidence*100).toFixed(1)}%` },
              { label: 'License Plate', value: selected.plate_number || 'Not detected' },
              { label: 'Camera ID', value: selected.camera_id || '—' },
              { label: 'Location', value: selected.location_label || '—' },
              { label: 'Challan', value: selected.challan_issued ? '✓ Issued' : 'Pending' },
              { label: 'Timestamp', value: new Date(selected.created_at).toLocaleString() },
              { label: 'Session ID', value: selected.session_id.slice(0,8)+'…' },
            ].map(({ label, value }) => (
              <div key={label} className="flex justify-between gap-2">
                <span className="text-xs text-gray-500 flex-shrink-0">{label}</span>
                <span className="text-xs text-gray-200 text-right">{value}</span>
              </div>
            ))}

            {/* Metadata JSON */}
            {selected.metadata && Object.keys(selected.metadata).length > 0 && (
              <div className="pt-2 border-t border-gray-800">
                <p className="text-xs text-gray-500 mb-1.5">Detection Metadata</p>
                <div className="bg-gray-900 rounded-lg p-2 space-y-1">
                  {Object.entries(selected.metadata).map(([k,v]) => (
                    <div key={k} className="flex justify-between gap-1">
                      <span className="text-xs text-gray-600 font-mono">{k}</span>
                      <span className="text-xs text-gray-300 font-mono text-right">
                        {typeof v === 'number' ? v.toFixed?.(3) ?? v : String(v)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
