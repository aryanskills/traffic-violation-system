import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import { Camera, BarChart3, AlertTriangle, FileText, Activity, TrendingUp } from 'lucide-react'
import DashboardPage from './pages/DashboardPage'
import UploadPage from './pages/UploadPage'
import ViolationsPage from './pages/ViolationsPage'
import ReportsPage from './pages/ReportsPage'
import EvaluationPage from './pages/EvaluationPage'

const NAV = [
  { to: '/',            icon: BarChart3,     label: 'Dashboard'   },
  { to: '/upload',      icon: Camera,        label: 'Detect'      },
  { to: '/violations',  icon: AlertTriangle, label: 'Violations'  },
  { to: '/reports',     icon: FileText,      label: 'Reports'     },
  { to: '/evaluation',  icon: TrendingUp,    label: 'Evaluation'  },
]

export default function App() {
  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col flex-shrink-0">
        <div className="px-5 py-5 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <Activity className="text-blue-400 flex-shrink-0" size={20} />
            <span className="font-bold text-sm leading-tight text-white">
              AI Traffic<br />Violation System
            </span>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-0.5">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white font-medium'
                    : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="px-4 py-4 border-t border-gray-800 space-y-1">
          <p className="text-xs text-gray-600">v2.0.0 · AI Engine</p>
          <p className="text-xs text-gray-700">YOLOv8 · EasyOCR</p>
          <div className="flex items-center gap-1.5 mt-2">
            <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs text-gray-500">System Online</span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto min-h-screen">
        <Routes>
          <Route path="/"           element={<DashboardPage />} />
          <Route path="/upload"     element={<UploadPage />} />
          <Route path="/violations" element={<ViolationsPage />} />
          <Route path="/reports"    element={<ReportsPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
        </Routes>
      </main>
    </div>
  )
}
