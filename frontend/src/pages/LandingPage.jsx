import React, { useRef, useState, useEffect, useMemo } from 'react';
import { motion, useScroll } from 'framer-motion';
import { Link } from 'react-router-dom';
import { ArrowRight, Code, Activity, Cpu, BrainCircuit, LineChart, BarChart2, ChevronRight } from 'lucide-react';
import BrandLogo from '@/components/BrandLogo';
import SemiconductorVisualizer from '@/components/SemiconductorVisualizer';

/* ─────────────────────────────────────────────────────────────
   LIGHT THEME LANDING PAGE
   No 3D Models on right side — Powered by interactive SVG telemetry.
   ───────────────────────────────────────────────────────────── */

export default function LandingPage() {
  const scrollRef = useRef(null);
  const { scrollYProgress } = useScroll({ target: scrollRef, offset: ['start start', 'end end'] });
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    return scrollYProgress.on('change', (v) => setProgress(v));
  }, [scrollYProgress]);

  const section = useMemo(() => {
    if (progress < 0.2) return 0;
    if (progress < 0.4) return 1;
    if (progress < 0.6) return 2;
    if (progress < 0.8) return 3;
    return 4;
  }, [progress]);

  return (
    <div className="bg-slate-50 text-slate-900 min-h-screen selection:bg-blue-100 selection:text-blue-900">

      {/* ─── NAVIGATION (Fixed Light Header) ─── */}
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 md:px-10 py-3.5 bg-white/90 backdrop-blur-md border-b border-slate-200/80 shadow-xs">
        <Link to="/" className="flex items-center gap-2.5">
          <BrandLogo className="w-7 h-7" />
          <span className="text-base font-bold tracking-tight text-slate-900">SemiYield AI</span>
        </Link>
        
        <div className="hidden md:flex items-center gap-8 text-xs font-semibold tracking-wider uppercase text-slate-600">
          <a href="#product" className="hover:text-blue-600 transition-colors">Product</a>
          <a href="#intelligence" className="hover:text-blue-600 transition-colors">Intelligence</a>
          <a href="#technology" className="hover:text-blue-600 transition-colors">Technology</a>
        </div>

        <div className="flex items-center gap-3">
          <a 
            href="https://github.com/Ladnil03/bob-ai-hackathon-Thefentasticfour" 
            target="_blank" 
            rel="noreferrer" 
            className="p-2 text-slate-400 hover:text-slate-700 transition-colors rounded-lg hover:bg-slate-100"
            title="GitHub Repository"
          >
            <Code className="w-4 h-4" />
          </a>
          <Link 
            to="/upload" 
            className="hidden sm:flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-xl text-xs font-bold tracking-wide uppercase transition-all shadow-sm shadow-blue-500/20 hover:scale-[1.02]"
          >
            Command Center
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </nav>

      {/* ─── SCROLL STORY ─── */}
      <div ref={scrollRef} id="product" style={{ height: '500vh' }} className="relative">
        <div className="sticky top-0 h-screen w-full flex overflow-hidden">

          {/* LEFT: Interactive Story Panels */}
          <div className="w-full md:w-[48%] lg:w-[45%] h-full flex items-center relative z-10 px-6 md:px-12 lg:px-16 pt-16">
            <div className="relative w-full max-w-lg">

              {/* SECTION 0: HERO */}
              <SectionPanel visible={section === 0}>
                <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-blue-50 border border-blue-200 text-blue-700 rounded-full text-xs font-bold uppercase tracking-wider mb-6">
                  AI-Powered Yield Intelligence
                </span>
                <h1 className="text-[clamp(2.2rem,4.5vw,3.6rem)] font-extrabold leading-[1.1] tracking-tight text-slate-900 mb-6">
                  Predict Yield.<br />
                  <span className="text-blue-600">Explain Failure.</span><br />
                  Optimize Process.
                </h1>
                <p className="text-slate-600 text-base leading-relaxed mb-8 max-w-md">
                  Predict wafer failures before they become expensive yield losses. Closed-loop AI for modern 300mm semiconductor fabrication lines.
                </p>
                <div className="flex items-center gap-4">
                  <Link to="/upload" className="inline-flex items-center gap-2 bg-blue-600 text-white hover:bg-blue-700 px-6 py-3 rounded-xl text-sm font-bold shadow-md shadow-blue-500/20 transition-all hover:scale-[1.02]">
                    Explore Command Center
                    <ArrowRight className="w-4 h-4" />
                  </Link>
                  <a href="#intelligence" className="inline-flex items-center gap-1.5 text-slate-600 hover:text-slate-900 text-sm font-semibold px-4 py-3">
                    Learn How It Works ↓
                  </a>
                </div>
              </SectionPanel>

              {/* SECTION 1: SENSE */}
              <SectionPanel visible={section === 1}>
                <SectionLabel>01 — Sense</SectionLabel>
                <h2 className="text-[clamp(1.8rem,3.5vw,2.6rem)] font-extrabold leading-tight tracking-tight text-slate-900 mb-4">
                  See the Process.
                </h2>
                <p className="text-slate-600 text-sm leading-relaxed mb-6 max-w-sm">
                  Continuous process signals from 590+ sensors reveal subtle deviations before they become yield losses.
                </p>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: 'RF Power',    val: '100.2 kW', sig: '+2.1σ', alert: true },
                    { label: 'Pressure',    val: '35.1 Torr', sig: '+0.3σ', alert: false },
                    { label: 'Temperature', val: '312.8°C', sig: '-0.8σ', alert: false },
                    { label: 'CF₄ Flow',    val: '70.4 sccm', sig: '+1.7σ', alert: true },
                  ].map(s => (
                    <div key={s.label} className="bg-white border border-slate-200/90 rounded-xl p-3.5 shadow-xs">
                      <p className="text-[10px] uppercase font-bold tracking-wider text-slate-400 mb-1">{s.label}</p>
                      <p className="text-base font-mono font-extrabold text-slate-900">{s.val}</p>
                      <p className={`text-[11px] font-mono font-semibold mt-1 ${s.alert ? 'text-amber-600' : 'text-slate-500'}`}>{s.sig}</p>
                    </div>
                  ))}
                </div>
              </SectionPanel>

              {/* SECTION 2: PREDICT */}
              <SectionPanel visible={section === 2}>
                <SectionLabel>02 — Predict</SectionLabel>
                <h2 className="text-[clamp(1.8rem,3.5vw,2.6rem)] font-extrabold leading-tight tracking-tight text-slate-900 mb-4">
                  Know What Will Fail.
                </h2>
                <p className="text-slate-600 text-sm leading-relaxed mb-6 max-w-sm">
                  Sub-millisecond ONNX inference identifies at-risk dies before they reach electrical testing.
                </p>
                <div className="space-y-3">
                  <MetricBlock label="Failure Risk" value="87.4%" color="text-red-600 bg-red-50 border-red-200" />
                  <MetricBlock label="Model Confidence" value="91.8%" color="text-blue-600 bg-blue-50 border-blue-200" />
                  <MetricBlock label="Inference Latency" value="<1.2 ms" color="text-emerald-700 bg-emerald-50 border-emerald-200" />
                </div>
              </SectionPanel>

              {/* SECTION 3: EXPLAIN */}
              <SectionPanel visible={section === 3}>
                <SectionLabel>03 — Explain</SectionLabel>
                <h2 className="text-[clamp(1.8rem,3.5vw,2.6rem)] font-extrabold leading-tight tracking-tight text-slate-900 mb-4">
                  Know Why.
                </h2>
                <p className="text-slate-600 text-sm leading-relaxed mb-6 max-w-sm">
                  Root cause analysis powered by 120B parameter MoE LLM. From anomaly to actionable insight.
                </p>
                <div className="space-y-3">
                  {[
                    { cause: 'RF Power Drift', pct: 34 },
                    { cause: 'Chamber Temperature', pct: 23 },
                    { cause: 'CF₄ Flow Rate', pct: 18 },
                    { cause: 'Etch Rate Deviation', pct: 12 },
                  ].map(c => (
                    <div key={c.cause} className="bg-white border border-slate-200/90 rounded-xl p-3 shadow-xs">
                      <div className="flex justify-between text-xs font-bold mb-1.5">
                        <span className="text-slate-700">{c.cause}</span>
                        <span className="font-mono text-slate-900">{c.pct}%</span>
                      </div>
                      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                        <motion.div
                          className="h-full bg-blue-600 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${c.pct}%` }}
                          transition={{ duration: 0.8 }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </SectionPanel>

              {/* SECTION 4: OPTIMIZE */}
              <SectionPanel visible={section === 4}>
                <SectionLabel>04 — Optimize</SectionLabel>
                <h2 className="text-[clamp(1.8rem,3.5vw,2.6rem)] font-extrabold leading-tight tracking-tight text-slate-900 mb-4">
                  Turn Insight Into Action.
                </h2>
                <p className="text-slate-600 text-sm leading-relaxed mb-5 max-w-sm">
                  Reinforcement learning continuously optimizes recipe setpoints based on production yield feedback.
                </p>
                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="bg-white border border-slate-200/90 rounded-xl p-3 shadow-xs">
                    <p className="text-[10px] uppercase font-bold text-slate-400 mb-2">Current</p>
                    <RecipeRow label="RF Power" value="100 kW" />
                    <RecipeRow label="CF₄ Flow" value="70 sccm" />
                    <RecipeRow label="Pressure" value="35 Torr" />
                  </div>
                  <div className="bg-emerald-50/80 border border-emerald-200/90 rounded-xl p-3 shadow-xs">
                    <p className="text-[10px] uppercase font-bold text-emerald-700 mb-2">AI Optimized</p>
                    <RecipeRow label="RF Power" value="98.2 kW" highlight />
                    <RecipeRow label="CF₄ Flow" value="68.7 sccm" highlight />
                    <RecipeRow label="Pressure" value="34.2 Torr" highlight />
                  </div>
                </div>
                <div className="bg-emerald-600 text-white rounded-xl px-4 py-3.5 flex items-center justify-between shadow-sm">
                  <span className="text-xs text-emerald-100 uppercase tracking-wider font-bold">Expected Yield</span>
                  <span className="font-mono text-base font-extrabold">91.4% → 94.1%</span>
                </div>
              </SectionPanel>

            </div>
          </div>

          {/* RIGHT: High-tech SVG Semiconductor Visualizer (No 3D Models) */}
          <div className="hidden md:block md:w-[52%] lg:w-[55%] h-full relative">
            <SemiconductorVisualizer section={section} />

            {/* Section Progress Navigation Bar */}
            <div className="absolute right-6 top-1/2 -translate-y-1/2 flex flex-col gap-3 z-20">
              {['Hero', 'Sense', 'Predict', 'Explain', 'Optimize'].map((name, i) => (
                <div key={name} className="flex items-center gap-2.5 group cursor-pointer" title={name}>
                  <span className={`text-[10px] font-bold uppercase tracking-wider transition-all duration-300 ${section === i ? 'text-blue-600 opacity-100 translate-x-0' : 'text-slate-400 opacity-0 group-hover:opacity-100 -translate-x-1'}`}>
                    {name}
                  </span>
                  <div className={`w-2.5 rounded-full transition-all duration-300 ${section === i ? 'h-7 bg-blue-600 shadow-sm shadow-blue-500/50' : 'h-2.5 bg-slate-300 group-hover:bg-slate-400'}`} />
                </div>
              ))}
            </div>
          </div>

          {/* Mobile: Fallback Visualizer */}
          <div className="md:hidden absolute bottom-0 left-0 right-0 h-[35vh] opacity-60 pointer-events-none">
            <SemiconductorVisualizer section={section} />
          </div>

        </div>
      </div>

      {/* ─── INTELLIGENCE LOOP ─── */}
      <section id="intelligence" className="py-24 px-6 md:px-12 bg-white border-t border-slate-200/80">
        <div className="max-w-4xl mx-auto text-center">
          <span className="inline-block px-3 py-1 bg-blue-50 border border-blue-200 text-blue-700 rounded-full text-xs font-bold uppercase tracking-wider mb-4">
            Closed-Loop Architecture
          </span>
          <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight text-slate-900 mb-12">
            Yield Intelligence Engine
          </h2>

          <div className="flex flex-wrap items-center justify-center gap-3 md:gap-4">
            {[
              { name: 'Sense',    icon: Activity },
              { name: 'Predict',  icon: Cpu },
              { name: 'Explain',  icon: BrainCircuit },
              { name: 'Optimize', icon: LineChart },
              { name: 'Learn',    icon: BarChart2 },
            ].map((step, i, arr) => (
              <React.Fragment key={step.name}>
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.08 }}
                  viewport={{ once: true }}
                  className="flex flex-col items-center gap-2 px-5 py-4 bg-slate-50 border border-slate-200/80 rounded-xl min-w-[96px] shadow-xs hover:border-blue-300 transition-colors"
                >
                  <step.icon className="w-5 h-5 text-blue-600" />
                  <span className="text-xs uppercase tracking-wider text-slate-800 font-bold">{step.name}</span>
                </motion.div>
                {i < arr.length - 1 && (
                  <ChevronRight className="w-4 h-4 text-slate-300 shrink-0" />
                )}
              </React.Fragment>
            ))}
            <ChevronRight className="w-4 h-4 text-slate-300 shrink-0" />
            <span className="text-xs uppercase tracking-wider text-blue-600 font-bold bg-blue-50 border border-blue-200 px-3 py-1.5 rounded-lg">
              Continuous Loop
            </span>
          </div>
        </div>
      </section>

      {/* ─── TECHNOLOGY ─── */}
      <section id="technology" className="py-24 px-6 md:px-12 bg-slate-50 border-t border-slate-200/80">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <span className="inline-block px-3 py-1 bg-blue-50 border border-blue-200 text-blue-700 rounded-full text-xs font-bold uppercase tracking-wider mb-4">
              Technology Stack
            </span>
            <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight text-slate-900">
              Built for Production Semiconductor Fabs
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[
              { title: 'ML Prediction', desc: 'CatBoost ensemble with ONNX Runtime. Sub-millisecond inference on 590 sensor features with 5-fold cross-validation.', label: 'Inference' },
              { title: 'Root-Cause Intelligence', desc: 'Groq GPT-OSS 120B MoE engine generates phased action roadmaps, financial impact estimates, and recipe offsets.', label: 'Prescriptive AI' },
              { title: 'RL Recipe Optimization', desc: 'Contextual Bandit with Thompson Sampling over Beta-Bernoulli priors. 8 operational regimes, 25 action combinations.', label: 'Reinforcement Learning' },
              { title: 'Model Operations', desc: 'Embedded MLflow tracking with zero external dependencies. Model versioning, staging, and production promotion.', label: 'MLOps' },
            ].map((t, i) => (
              <motion.div
                key={t.title}
                initial={{ opacity: 0, y: 15 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.08 }}
                viewport={{ once: true }}
                className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs hover:shadow-md hover:border-slate-300 transition-all"
              >
                <span className="text-[10px] uppercase tracking-wider text-blue-600 font-bold bg-blue-50 border border-blue-100 px-2.5 py-1 rounded-md">{t.label}</span>
                <h3 className="text-lg font-extrabold mt-3.5 mb-2 text-slate-900">{t.title}</h3>
                <p className="text-sm text-slate-600 leading-relaxed">{t.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── FINAL CTA ─── */}
      <section className="py-24 px-6 md:px-12 bg-white border-t border-slate-200/80 text-center">
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-2xl mx-auto"
        >
          <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight text-slate-900 leading-tight mb-4">
            The Next Wafer Should Be Better.
          </h2>
          <p className="text-slate-600 text-base mb-8 max-w-md mx-auto">
            Move from reactive yield analysis to predictive semiconductor manufacturing intelligence.
          </p>
          <Link to="/upload" className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-7 py-3.5 rounded-xl text-sm font-bold shadow-lg shadow-blue-500/25 transition-all hover:scale-[1.02]">
            Launch Command Center
            <ArrowRight className="w-4 h-4" />
          </Link>
        </motion.div>
      </section>

      {/* ─── FOOTER ─── */}
      <footer className="py-8 px-6 bg-slate-900 text-slate-400 border-t border-slate-800 flex items-center justify-between text-xs font-medium">
        <div className="flex items-center gap-3">
          <BrandLogo className="w-5 h-5" />
          <span>Thefentasticfour — IBM Bob AI Hackathon</span>
        </div>
        <a 
          href="https://github.com/Ladnil03/bob-ai-hackathon-Thefentasticfour" 
          target="_blank" 
          rel="noreferrer" 
          className="hover:text-white transition-colors flex items-center gap-1.5"
        >
          <Code className="w-4 h-4" />
          <span>GitHub</span>
        </a>
      </footer>

    </div>
  );
}

/* ─── Helper Components ─── */

function SectionPanel({ visible, children }) {
  return (
    <div
      className="absolute inset-0 flex items-center transition-all duration-500"
      style={{ opacity: visible ? 1 : 0, transform: visible ? 'translateY(0)' : 'translateY(12px)', pointerEvents: visible ? 'auto' : 'none' }}
    >
      <div className="w-full">{children}</div>
    </div>
  );
}

function SectionLabel({ children }) {
  return <p className="text-xs font-mono uppercase tracking-widest text-blue-600 font-bold mb-3">{children}</p>;
}

function MetricBlock({ label, value, color }) {
  return (
    <div className={`border rounded-xl px-4 py-3 flex items-center justify-between shadow-xs ${color}`}>
      <span className="text-xs uppercase font-bold tracking-wider">{label}</span>
      <span className="font-mono text-lg font-extrabold">{value}</span>
    </div>
  );
}

function RecipeRow({ label, value, highlight = false }) {
  return (
    <div className="flex justify-between items-center py-1 border-b border-slate-100 last:border-0">
      <span className="text-xs text-slate-500">{label}</span>
      <span className={`text-xs font-mono font-bold ${highlight ? 'text-emerald-700' : 'text-slate-800'}`}>{value}</span>
    </div>
  );
}