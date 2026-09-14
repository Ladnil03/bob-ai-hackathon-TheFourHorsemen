import { useState, lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, NavLink, Outlet, Link } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { Menu, X, Upload, Trophy, LayoutDashboard, BarChart2, BrainCircuit, ArrowLeft } from 'lucide-react';
import Preloader from './components/Preloader';
import BrandLogo from './components/BrandLogo';
import './index.css';

const LandingPage = lazy(() => import('./pages/LandingPage'));
const UploadPage = lazy(() => import('./pages/UploadPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const AnalysisPage = lazy(() => import('./pages/AnalysisPage'));
const PredictPage = lazy(() => import('./pages/PredictPage'));
const BestModelPage = lazy(() => import('./pages/BestModelPage'));

const navItems = [
  { to: '/upload', icon: Upload, label: 'Upload & Load' },
  { to: '/best-model', icon: Trophy, label: 'SOTA Model' },
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/analysis', icon: BarChart2, label: 'Analysis' },
  { to: '/predict', icon: BrainCircuit, label: 'Predict' },
];

function Sidebar({ open, setOpen }) {
  return (
    <>
      {open && (
        <div className="fixed inset-0 bg-black/20 z-40 lg:hidden" onClick={() => setOpen(false)} />
      )}
      <aside className={`fixed top-0 left-0 z-40 h-full w-60 bg-card border-r border-border transform transition-transform duration-200 pt-14
        ${open ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0 lg:static lg:pt-0`}>
        <button onClick={() => setOpen(false)} className="lg:hidden absolute top-4 right-3 p-1.5 text-muted-foreground hover:bg-muted rounded">
          <X className="w-4 h-4" />
        </button>
        <nav className="flex flex-col gap-0.5 p-3 h-full overflow-y-auto">
          <Link to="/" className="flex items-center gap-2 px-3 py-2 mb-3 text-xs text-muted-foreground hover:text-foreground transition-colors">
            <ArrowLeft className="w-3.5 h-3.5" />
            Back to Landing
          </Link>
          <div className="text-[10px] font-bold uppercase tracking-[0.15em] text-muted-foreground mb-3 px-3">Command Center</div>
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-2.5 px-3 py-2 rounded text-sm font-medium transition-all
                ${isActive
                  ? 'bg-primary/10 text-foreground font-semibold'
                  : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'
                }`
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
    </>
  );
}

function DashboardLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar open={sidebarOpen} setOpen={setSidebarOpen} />
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Dashboard top bar */}
        <header className="flex items-center justify-between px-4 md:px-6 py-3 border-b border-border bg-card/50 shrink-0">
          <div className="flex items-center gap-3">
            <button onClick={() => setSidebarOpen(true)} className="lg:hidden p-1.5 text-muted-foreground hover:bg-muted rounded">
              <Menu className="w-5 h-5" />
            </button>
            <BrandLogo className="w-6 h-6" />
            <span className="text-sm font-semibold text-foreground tracking-tight">SemiYield AI</span>
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium ml-2 hidden sm:inline">Command Center</span>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8">
          <Suspense fallback={
            <div className="w-full h-64 flex flex-col gap-3 items-center justify-center">
              <div className="w-6 h-6 border-2 border-foreground/20 border-t-foreground rounded-full animate-spin" />
              <p className="text-xs text-muted-foreground">Loading module…</p>
            </div>
          }>
            <div className="max-w-7xl mx-auto">
              <Outlet />
            </div>
          </Suspense>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Preloader>
      <BrowserRouter>
        <Toaster position="top-right" />
        <Routes>
          {/* Landing page: owns its own nav, full-page scroll */}
          <Route path="/" element={
            <Suspense fallback={<div className="h-screen w-screen bg-[#070a0f]" />}>
              <LandingPage />
            </Suspense>
          } />
          {/* Dashboard: separate layout with sidebar */}
          <Route element={<DashboardLayout />}>
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/best-model" element={<BestModelPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/analysis" element={<AnalysisPage />} />
            <Route path="/predict" element={<PredictPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </Preloader>
  );
}
