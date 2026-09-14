import { useState, lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, NavLink, Outlet } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { Menu, X, Upload, Trophy, LayoutDashboard, BarChart2, BrainCircuit } from 'lucide-react';
import Preloader from './components/Preloader';
import GlobalNav from './components/GlobalNav';
import './index.css';

// Lazy loaded pages for performance
const LandingPage = lazy(() => import('./pages/LandingPage'));
const UploadPage = lazy(() => import('./pages/UploadPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const AnalysisPage = lazy(() => import('./pages/AnalysisPage'));
const PredictPage = lazy(() => import('./pages/PredictPage'));
const BestModelPage = lazy(() => import('./pages/BestModelPage'));

const navItems = [
  { to: '/upload', icon: Upload, label: 'Upload & Load' },
  { to: '/best-model', icon: Trophy, label: 'SOTA' },
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
      <aside className={`fixed top-0 left-0 z-40 h-full w-64 bg-card border-r border-border shadow-sm transform transition-transform duration-200 ease-in-out pt-16
        ${open ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0 lg:static lg:shadow-none lg:pt-0`}>
        {/* Mobile close button */}
        <button onClick={() => setOpen(false)} className="lg:hidden absolute top-4 right-4 p-2 text-muted-foreground hover:bg-muted rounded-md">
          <X className="w-5 h-5" />
        </button>

        <nav className="flex flex-col gap-1 p-4 h-full overflow-y-auto">
          <div className="text-xs font-bold uppercase tracking-widest text-muted-foreground mb-4 px-2 mt-4 lg:mt-0">Menu</div>
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-all duration-150
                ${isActive
                  ? 'bg-muted text-foreground font-semibold border-l-2 border-foreground'
                  : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground border-l-2 border-transparent'
                }`
              }
            >
              <Icon className="w-5 h-5" />
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
    </>
  );
}

// Layout wrapper for dashboard pages
function DashboardLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  return (
    <div className="flex flex-1 overflow-hidden bg-background">
      <Sidebar open={sidebarOpen} setOpen={setSidebarOpen} />
      <div className="flex-1 flex flex-col overflow-hidden relative">
        <button 
          onClick={() => setSidebarOpen(true)} 
          className="lg:hidden absolute top-4 left-4 z-30 p-2 bg-card border border-border shadow-sm rounded-md"
        >
          <Menu className="w-5 h-5" />
        </button>
        <main className="flex-1 overflow-y-auto p-4 md:p-8 bg-background">
          <Suspense fallback={
            <div className="w-full h-full flex flex-col gap-4 items-center justify-center p-12">
              <div className="w-8 h-8 border-2 border-foreground border-t-transparent rounded-full animate-spin"></div>
              <p className="text-sm font-medium text-muted-foreground animate-pulse">Loading dashboard module...</p>
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
        <div className="flex flex-col h-screen overflow-hidden">
          <GlobalNav />
          <Toaster position="top-right" />
          <Routes>
            <Route path="/" element={
              <Suspense fallback={<div className="h-screen w-screen bg-background" />}>
                <LandingPage />
              </Suspense>
            } />
            <Route element={<DashboardLayout />}>
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/best-model" element={<BestModelPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/analysis" element={<AnalysisPage />} />
              <Route path="/predict" element={<PredictPage />} />
            </Route>
          </Routes>
        </div>
      </BrowserRouter>
    </Preloader>
  );
}
