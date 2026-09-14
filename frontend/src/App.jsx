import { useState } from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { UploadCloud, LayoutDashboard, BarChart3, Target, Info, Menu, X, Cpu, Award } from 'lucide-react';
import UploadPage from './pages/UploadPage';
import DashboardPage from './pages/DashboardPage';
import AnalysisPage from './pages/AnalysisPage';
import PredictPage from './pages/PredictPage';
import BestModelPage from './pages/BestModelPage';
import './index.css';

const navItems = [
  { to: '/', icon: UploadCloud, label: 'Upload & Load' },
  { to: '/best-model', icon: Award, label: 'Best Model (SOTA)' },
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/analysis', icon: BarChart3, label: 'Analysis' },
  { to: '/predict', icon: Target, label: 'Predict' },
];

function Sidebar({ open, setOpen }) {
  return (
    <>
      {open && (
        <div className="fixed inset-0 bg-black/20 z-40 lg:hidden" onClick={() => setOpen(false)} />
      )}
      <aside className={`fixed top-0 left-0 z-50 h-full w-64 bg-white border-r border-border shadow-lg transform transition-transform duration-200 ease-in-out
        ${open ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0 lg:static lg:shadow-none`}>
        <div className="flex items-center gap-3 px-6 py-5 border-b border-border">
          <div className="w-9 h-9 rounded-lg gradient-primary flex items-center justify-center">
            <Cpu className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-base text-foreground">SemiYield AI</h1>
            <p className="text-xs text-muted-foreground">Yield Optimization</p>
          </div>
        </div>
        <nav className="flex flex-col gap-1 p-3 mt-2">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-150
                ${isActive
                  ? 'bg-primary text-primary-foreground shadow-md shadow-primary/25'
                  : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
                }`
              }
            >
              <Icon className="w-4.5 h-4.5" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-border">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Info className="w-3.5 h-3.5" />
            <span>Team Thefentasticfour</span>
          </div>
        </div>
      </aside>
    </>
  );
}

function Header({ setOpen }) {
  return (
    <header className="sticky top-0 z-30 flex items-center gap-4 px-6 py-3 bg-white/80 backdrop-blur-md border-b border-border">
      <button onClick={() => setOpen(o => !o)} className="lg:hidden p-2 hover:bg-secondary rounded-lg">
        <Menu className="w-5 h-5" />
      </button>
      <h2 className="text-sm font-semibold text-foreground">Semiconductor Yield Optimization Platform</h2>
    </header>
  );
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <BrowserRouter>
      <Toaster position="top-right" />
      <div className="flex h-screen overflow-hidden">
        <Sidebar open={sidebarOpen} setOpen={setSidebarOpen} />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header setOpen={setSidebarOpen} />
          <main className="flex-1 overflow-y-auto p-6 bg-background">
            <Routes>
              <Route path="/" element={<UploadPage />} />
              <Route path="/best-model" element={<BestModelPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/analysis" element={<AnalysisPage />} />
              <Route path="/predict" element={<PredictPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}
