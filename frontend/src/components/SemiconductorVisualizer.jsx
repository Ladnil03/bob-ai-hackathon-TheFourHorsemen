import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Cpu, BrainCircuit, LineChart, CheckCircle2, AlertTriangle, Zap, ShieldCheck, ArrowUpRight, Gauge, Radio, RefreshCw, Layers } from 'lucide-react';

export default function SemiconductorVisualizer({ section = 0 }) {
  return (
    <div className="w-full h-full flex items-center justify-center p-4 lg:p-8 select-none">
      <div className="w-full max-w-xl bg-white border border-slate-200/90 rounded-2xl shadow-xl shadow-slate-200/60 p-6 md:p-8 relative overflow-hidden transition-all duration-500">
        
        {/* Subtle background grid pattern */}
        <div 
          className="absolute inset-0 opacity-[0.03] pointer-events-none"
          style={{
            backgroundImage: `radial-gradient(#2563eb 1px, transparent 1px)`,
            backgroundSize: '16px 16px'
          }}
        />

        {/* Section Title Header */}
        <div className="flex items-center justify-between border-b border-slate-100 pb-4 mb-6">
          <div className="flex items-center gap-2.5">
            <div className="w-2.5 h-2.5 rounded-full bg-blue-600 animate-pulse" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-600">
              {section === 0 && "300mm Wafer Die Blueprint"}
              {section === 1 && "Live Fab Sensor Telemetry"}
              {section === 2 && "Spatial Yield Risk Heatmap"}
              {section === 3 && "Root Cause Feature Graph"}
              {section === 4 && "Closed-Loop Recipe Control"}
            </span>
          </div>
          <div className="flex items-center gap-1.5 bg-blue-50 border border-blue-100 rounded-full px-3 py-1">
            <Zap className="w-3.5 h-3.5 text-blue-600" />
            <span className="text-[11px] font-bold text-blue-700">Sub-ms Inference</span>
          </div>
        </div>

        {/* Dynamic Display based on active scroll section */}
        <AnimatePresence mode="wait">
          {section === 0 && <HeroWaferDiagram key="hero" />}
          {section === 1 && <SenseSensorDiagram key="sense" />}
          {section === 2 && <PredictHeatmapDiagram key="predict" />}
          {section === 3 && <ExplainGraphDiagram key="explain" />}
          {section === 4 && <OptimizeRecipeDiagram key="optimize" />}
        </AnimatePresence>

        {/* Bottom Telemetry Bar */}
        <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500 font-medium">
          <div className="flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <span>FAB 4 — 300mm Line</span>
          </div>
          <div className="font-mono text-slate-700 bg-slate-100 px-2.5 py-0.5 rounded text-[11px] font-semibold">
            CatBoost + ONNX Engine
          </div>
        </div>

      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────
   SECTION 0: Hero Wafer Vector SVG & Inspection Blueprint
   ───────────────────────────────────────────────────────────── */
function HeroWaferDiagram() {
  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ duration: 0.4 }}
      className="flex flex-col items-center gap-6"
    >
      <div className="relative w-64 h-64 md:w-72 md:h-72 flex items-center justify-center">
        {/* SVG Silicon Wafer */}
        <svg viewBox="0 0 200 200" className="w-full h-full drop-shadow-md">
          {/* Outer Notch Wafer Edge */}
          <circle cx="100" cy="100" r="92" fill="#F8FAFC" stroke="#CBD5E1" strokeWidth="3" />
          <circle cx="100" cy="100" r="88" fill="#EFF6FF" stroke="#93C5FD" strokeWidth="1.5" strokeDasharray="4 4" />
          
          {/* Silicon Grid Dies */}
          <g fill="#DBEAFE" stroke="#3B82F6" strokeWidth="0.8" opacity="0.85">
            {[-3, -2, -1, 0, 1, 2, 3].map((x) =>
              [-3, -2, -1, 0, 1, 2, 3].map((y) => {
                const dist = Math.sqrt(x * x + y * y);
                if (dist > 3.2) return null;
                const isWarning = (x === 2 && y === -1) || (x === -2 && y === 2);
                const isFail = (x === 3 && y === 0);
                return (
                  <rect
                    key={`${x}-${y}`}
                    x={90 + x * 20}
                    y={90 + y * 20}
                    width="18"
                    height="18"
                    rx="2"
                    fill={isFail ? '#FEE2E2' : isWarning ? '#FEF3C7' : '#DBEAFE'}
                    stroke={isFail ? '#EF4444' : isWarning ? '#F59E0B' : '#2563EB'}
                    strokeWidth={isFail || isWarning ? 1.5 : 0.8}
                  />
                );
              })
            )}
          </g>

          {/* Notch at bottom */}
          <path d="M 94 191 A 6 6 0 0 0 106 191 Z" fill="#CBD5E1" />

          {/* Laser Scanner Beam */}
          <motion.line
            x1="10" y1="50" x2="190" y2="50"
            stroke="#2563EB" strokeWidth="2" strokeDasharray="6 3"
            animate={{ y1: [30, 170, 30], y2: [30, 170, 30] }}
            transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
          />
        </svg>

        {/* Floating Die Inspection Badge */}
        <div className="absolute -top-2 -right-2 bg-emerald-500 text-white font-mono text-[11px] font-bold px-3 py-1 rounded-full shadow-md flex items-center gap-1.5">
          <CheckCircle2 className="w-3.5 h-3.5" />
          Yield 94.8%
        </div>

        <div className="absolute -bottom-2 -left-2 bg-slate-900 text-white font-mono text-[10px] px-3 py-1.5 rounded-lg shadow-md">
          Die Count: 1,420 / Wafer
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 w-full text-center">
        <div className="bg-slate-50 border border-slate-200/80 p-2.5 rounded-xl">
          <p className="text-[10px] uppercase font-bold text-slate-400">Pass Dies</p>
          <p className="text-sm font-extrabold text-slate-800 font-mono">1,346</p>
        </div>
        <div className="bg-amber-50 border border-amber-200/80 p-2.5 rounded-xl">
          <p className="text-[10px] uppercase font-bold text-amber-600">Marginal</p>
          <p className="text-sm font-extrabold text-amber-700 font-mono">48</p>
        </div>
        <div className="bg-red-50 border border-red-200/80 p-2.5 rounded-xl">
          <p className="text-[10px] uppercase font-bold text-red-500">Fail Risk</p>
          <p className="text-sm font-extrabold text-red-600 font-mono">26</p>
        </div>
      </div>
    </motion.div>
  );
}

/* ─────────────────────────────────────────────────────────────
   SECTION 1: Sense — High-Tech Oscilloscope Fab Telemetry Suite
   ───────────────────────────────────────────────────────────── */
function SenseSensorDiagram() {
  const [activeTab, setActiveTab] = useState('etch');
  const [liveMetrics, setLiveMetrics] = useState({
    rfPower: 100.2,
    pressure: 35.1,
    gasFlow: 70.4,
  });

  // Real-time subtle telemetry jitter simulation
  useEffect(() => {
    const timer = setInterval(() => {
      setLiveMetrics({
        rfPower: +(100.0 + Math.random() * 0.4).toFixed(1),
        pressure: +(35.0 + Math.random() * 0.3).toFixed(1),
        gasFlow: +(69.8 + Math.random() * 0.9).toFixed(1),
      });
    }, 1200);
    return () => clearInterval(timer);
  }, []);

  const sensors = [
    { 
      id: 'rf', 
      name: 'RF Plasma Power', 
      val: `${liveMetrics.rfPower} kW`, 
      target: '100.0 kW',
      sig: '+2.1σ', 
      color: '#2563eb', 
      status: 'Drift Alert',
      alert: true,
      points: [25, 18, 30, 12, 35, 15, 28, 10, 32, 20, 28, 14, 30] 
    },
    { 
      id: 'press', 
      name: 'Chamber Pressure', 
      val: `${liveMetrics.pressure} Torr`, 
      target: '35.0 Torr',
      sig: '+0.3σ', 
      color: '#059669', 
      status: 'Optimal',
      alert: false,
      points: [20, 22, 19, 21, 20, 18, 21, 20, 19, 22, 20, 21, 20] 
    },
    { 
      id: 'gas', 
      name: 'CF₄ Gas Flow Rate', 
      val: `${liveMetrics.gasFlow} sccm`, 
      target: '68.0 sccm',
      sig: '+1.7σ', 
      color: '#d97706', 
      status: 'High Flow',
      alert: true,
      points: [15, 25, 12, 30, 10, 28, 14, 32, 16, 28, 20, 30, 25] 
    },
  ];

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.4 }}
      className="space-y-4"
    >
      {/* Category Tabs */}
      <div className="flex items-center justify-between bg-slate-100 p-1 rounded-xl text-xs font-semibold text-slate-600">
        {[
          { id: 'etch', label: 'Etch Chamber 04' },
          { id: 'thermal', label: 'Thermal Zone' },
          { id: 'gas', label: 'Mass Flow' },
        ].map(t => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`flex-1 py-1.5 rounded-lg transition-all text-center ${
              activeTab === t.id ? 'bg-white text-blue-600 shadow-xs font-bold' : 'hover:text-slate-900'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Sensor Signals List with Oscilloscope Waveform Display */}
      <div className="space-y-3">
        {sensors.map((s) => {
          // Construct SVG Polyline Points (scaled safely inside 240x40 box)
          const svgPoints = s.points.map((p, idx) => `${idx * 20},${p}`).join(' ');
          
          return (
            <div key={s.id} className="bg-white border border-slate-200/90 rounded-xl p-3.5 shadow-xs hover:border-blue-200 transition-colors">
              <div className="flex justify-between items-center mb-2">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full animate-pulse" style={{ backgroundColor: s.color }} />
                  <span className="text-xs font-bold text-slate-800">{s.name}</span>
                </div>
                <div className="flex items-center gap-2 font-mono text-xs">
                  <span className="font-extrabold text-slate-900">{s.val}</span>
                  <span className={`text-[10px] px-2 py-0.5 rounded font-bold ${
                    s.alert ? 'bg-amber-50 text-amber-700 border border-amber-200' : 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  }`}>
                    {s.sig}
                  </span>
                </div>
              </div>

              {/* Oscilloscope Waveform Box */}
              <div className="h-12 w-full bg-slate-900 rounded-lg p-1.5 relative overflow-hidden flex items-center shadow-inner">
                {/* Background Oscilloscope Grid Lines */}
                <div className="absolute inset-0 opacity-20 pointer-events-none flex flex-col justify-between p-1">
                  <div className="border-b border-blue-400 border-dashed w-full" />
                  <div className="border-b border-blue-400 border-dashed w-full" />
                  <div className="border-b border-blue-400 border-dashed w-full" />
                </div>

                {/* Sweeping Laser Line */}
                <motion.div 
                  className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 shadow-[0_0_8px_#22d3ee] z-10"
                  animate={{ left: ['0%', '100%'] }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                />

                {/* SVG Waveform Line */}
                <svg viewBox="0 0 240 40" className="w-full h-full relative z-0">
                  <defs>
                    <linearGradient id={`grad-${s.id}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={s.color} stopOpacity="0.4" />
                      <stop offset="100%" stopColor={s.color} stopOpacity="0.0" />
                    </linearGradient>
                  </defs>
                  
                  {/* Waveform Area Fill */}
                  <polygon points={`0,40 ${svgPoints} 240,40`} fill={`url(#grad-${s.id})`} />
                  
                  {/* Waveform Stroke */}
                  <polyline
                    fill="none"
                    stroke={s.color === '#2563eb' ? '#60a5fa' : s.color === '#059669' ? '#34d399' : '#fbbf24'}
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={svgPoints}
                  />
                </svg>
              </div>
            </div>
          );
        })}
      </div>

      {/* Sensor Health Status Summary */}
      <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3 flex items-center justify-between text-xs font-semibold text-slate-700">
        <div className="flex items-center gap-1.5">
          <Radio className="w-4 h-4 text-blue-600 animate-pulse" />
          <span>Active Sensor Channels: <strong className="text-slate-900">590 / 590</strong></span>
        </div>
        <span className="font-mono text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded text-[11px] font-bold">
          100% Signal Fidelity
        </span>
      </div>
    </motion.div>
  );
}

/* ─────────────────────────────────────────────────────────────
   SECTION 2: Predict Failure Heatmap & Spatial Analysis
   ───────────────────────────────────────────────────────────── */
function PredictHeatmapDiagram() {
  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ duration: 0.4 }}
      className="space-y-5"
    >
      <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-4">
        <div className="flex justify-between items-center mb-3">
          <span className="text-xs font-bold text-slate-700">Die Yield Risk Map</span>
          <span className="text-[11px] font-mono text-red-600 bg-red-50 border border-red-100 px-2 py-0.5 rounded font-bold">
            High Risk Identified (Die #42)
          </span>
        </div>

        {/* Wafer Spatial Grid Matrix */}
        <div className="grid grid-cols-7 gap-1.5 p-2 bg-white rounded-lg border border-slate-150">
          {Array.from({ length: 35 }).map((_, i) => {
            const isCritical = i === 18 || i === 19 || i === 25;
            const isWarning = i === 11 || i === 12 || i === 24;
            return (
              <motion.div
                key={i}
                whileHover={{ scale: 1.1 }}
                className={`h-7 rounded flex items-center justify-center text-[10px] font-mono font-bold transition-colors ${
                  isCritical 
                    ? 'bg-red-500 text-white shadow-sm' 
                    : isWarning 
                      ? 'bg-amber-400 text-slate-900' 
                      : 'bg-blue-100 text-blue-800 hover:bg-blue-200'
                }`}
              >
                {i + 1}
              </motion.div>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3">
          <p className="text-[10px] uppercase font-bold text-slate-400 mb-1">Failure Risk</p>
          <p className="text-xl font-extrabold text-red-600 font-mono">87.4%</p>
        </div>
        <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3">
          <p className="text-[10px] uppercase font-bold text-slate-400 mb-1">Model Confidence</p>
          <p className="text-xl font-extrabold text-blue-600 font-mono">91.8%</p>
        </div>
      </div>
    </motion.div>
  );
}

/* ─────────────────────────────────────────────────────────────
   SECTION 3: Explain Root-Cause Diagnostic Diagram
   ───────────────────────────────────────────────────────────── */
function ExplainGraphDiagram() {
  const causes = [
    { title: 'RF Power Drift', weight: '34%', desc: 'Plasma density fluctuation during etch' },
    { title: 'Chamber Temp Gradient', weight: '23%', desc: 'Outer edge thermal dissipation' },
    { title: 'CF₄ Gas Flow Offset', weight: '18%', desc: 'Mass flow controller calibration' },
  ];

  return (
    <motion.div 
      initial={{ opacity: 0, x: 10 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -10 }}
      transition={{ duration: 0.4 }}
      className="space-y-4"
    >
      <div className="bg-blue-50/70 border border-blue-200/80 rounded-xl p-3.5 flex items-start gap-3">
        <BrainCircuit className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
        <div>
          <h4 className="text-xs font-bold text-blue-900">Groq 120B MoE Root-Cause Report</h4>
          <p className="text-[11px] text-blue-700/90 leading-relaxed mt-0.5">
            Primary failure driven by RF Power drift causing micro-trenching on outer wafer perimeter.
          </p>
        </div>
      </div>

      <div className="space-y-2.5">
        {causes.map((c, i) => (
          <div key={c.title} className="bg-slate-50 border border-slate-200/80 rounded-xl p-3">
            <div className="flex justify-between items-center text-xs font-bold mb-1">
              <span className="text-slate-800">{c.title}</span>
              <span className="font-mono text-red-600">{c.weight} Contribution</span>
            </div>
            <p className="text-[11px] text-slate-500 mb-2">{c.desc}</p>
            <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
              <motion.div 
                className="h-full bg-red-500 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: c.weight }}
                transition={{ duration: 0.8, delay: i * 0.1 }}
              />
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

/* ─────────────────────────────────────────────────────────────
   SECTION 4: Optimize Recipe Adjustment Panel
   ───────────────────────────────────────────────────────────── */
function OptimizeRecipeDiagram() {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.4 }}
      className="space-y-4"
    >
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3">
          <p className="text-[10px] font-bold uppercase text-slate-400 mb-2">Original Recipe</p>
          <div className="space-y-1.5 text-xs font-mono">
            <div className="flex justify-between text-slate-600"><span>RF Power</span><span>100.0 kW</span></div>
            <div className="flex justify-between text-slate-600"><span>CF₄ Flow</span><span>70.0 sccm</span></div>
            <div className="flex justify-between text-slate-600"><span>Pressure</span><span>35.0 Torr</span></div>
          </div>
        </div>

        <div className="bg-emerald-50/70 border border-emerald-200 rounded-xl p-3">
          <p className="text-[10px] font-bold uppercase text-emerald-700 mb-2">AI Optimized</p>
          <div className="space-y-1.5 text-xs font-mono font-bold">
            <div className="flex justify-between text-emerald-800"><span>RF Power</span><span>98.2 kW</span></div>
            <div className="flex justify-between text-emerald-800"><span>CF₄ Flow</span><span>68.7 sccm</span></div>
            <div className="flex justify-between text-emerald-800"><span>Pressure</span><span>34.2 Torr</span></div>
          </div>
        </div>
      </div>

      <div className="bg-emerald-600 text-white rounded-xl p-4 flex items-center justify-between shadow-md">
        <div>
          <p className="text-[10px] uppercase font-bold text-emerald-100">Predicted Yield Impact</p>
          <p className="text-lg font-extrabold font-mono mt-0.5">91.4% → 94.1% (+2.7%)</p>
        </div>
        <div className="bg-white/20 hover:bg-white/30 text-white text-xs font-bold px-3 py-2 rounded-lg transition-colors flex items-center gap-1 cursor-pointer">
          Apply Setpoints <ArrowUpRight className="w-3.5 h-3.5" />
        </div>
      </div>
    </motion.div>
  );
}
