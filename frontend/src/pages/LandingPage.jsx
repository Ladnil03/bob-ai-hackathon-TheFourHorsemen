import React, { lazy, Suspense } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Sparkles, Code, ArrowRight, Cpu, BrainCircuit, Database, LineChart } from 'lucide-react';
import PipelineVisualizer from '@/components/PipelineVisualizer';

// Lazy load the heavy 3D scene
const Hero3DScene = lazy(() => import('@/components/Hero3DScene'));

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] } }
};

export default function LandingPage() {
  return (
    <div className="w-full overflow-x-hidden overflow-y-auto">

      {/* ─── HERO: Dark background so the 3D wafer is visible ─── */}
      <section className="relative w-full min-h-[92vh] flex flex-col items-center justify-center px-6 bg-[#0a0f1a] text-white overflow-hidden">
        
        {/* 3D Canvas fills the entire hero */}
        <div className="absolute inset-0 z-0 opacity-90">
          <Suspense fallback={
            <div className="w-full h-full flex items-center justify-center">
              <div className="w-10 h-10 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            </div>
          }>
            <Hero3DScene />
          </Suspense>
        </div>

        {/* Hero content on top */}
        <motion.div 
          className="relative z-10 flex flex-col items-center text-center max-w-4xl gap-5"
          initial="hidden"
          animate="visible"
          variants={fadeUp}
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 backdrop-blur-md border border-white/10 text-[11px] font-bold uppercase tracking-[0.2em] text-blue-300">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Thefentasticfour — IBM Bob AI Hackathon</span>
          </div>

          <h1 className="text-4xl sm:text-5xl md:text-7xl font-extrabold tracking-tight leading-[1.08] mt-4">
            Semiconductor Yield
            <br />
            <span className="text-blue-400">Optimization</span>
          </h1>
          
          <p className="text-lg md:text-xl text-white/60 max-w-2xl font-light leading-relaxed">
            Closed-loop AI that predicts wafer failure in sub-milliseconds, prescribes root causes with a 120B MoE LLM, and continuously tunes recipes via Reinforcement Learning.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4 mt-6">
            <a 
              href="https://github.com/Ladnil03/bob-ai-hackathon-Thefentasticfour" 
              target="_blank" 
              rel="noreferrer"
              className="flex items-center gap-2 px-5 py-2.5 bg-white/10 backdrop-blur-md text-white border border-white/15 hover:bg-white/20 transition-colors rounded-full text-sm font-medium"
            >
              <Code className="w-4 h-4" />
              View on GitHub
            </a>
            <Link 
              to="/upload" 
              className="group flex items-center gap-2 px-5 py-2.5 bg-blue-500 text-white hover:bg-blue-400 transition-colors rounded-full text-sm font-medium shadow-lg shadow-blue-500/30"
            >
              Open Dashboard
              <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
            </Link>
          </div>
        </motion.div>

        {/* Bottom fade from dark hero into light content */}
        <div className="absolute bottom-0 left-0 right-0 h-24 bg-[linear-gradient(to_top,var(--color-background),transparent)] z-10" />
      </section>

      {/* ─── PIPELINE SECTION: Light background ─── */}
      <section className="relative py-20 px-6 bg-background">
        <motion.div
          className="max-w-5xl mx-auto"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-80px" }}
          variants={fadeUp}
        >
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground">
              How the Pipeline Works
            </h2>
            <p className="text-muted-foreground mt-3 max-w-xl mx-auto">
              From raw sensor data to optimized recipes — fully automated, fully adaptive.
            </p>
          </div>

          <PipelineVisualizer />
        </motion.div>
      </section>

      {/* ─── STATS SECTION: Key numbers ─── */}
      <section className="py-20 px-6 border-t border-border bg-muted/30">
        <motion.div
          className="max-w-5xl mx-auto"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-80px" }}
          variants={fadeUp}
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { value: '<1.2ms', label: 'Inference Latency', icon: Cpu },
              { value: '590+', label: 'Sensor Channels', icon: Database },
              { value: '120B', label: 'MoE LLM Parameters', icon: BrainCircuit },
              { value: '$50–100M', label: 'Monthly Yield Loss at Risk', icon: LineChart },
            ].map(({ value, label, icon: Icon }) => (
              <div key={label} className="flex flex-col items-center text-center p-6 bg-background rounded-2xl border border-border shadow-sm">
                <Icon className="w-8 h-8 text-primary mb-3" />
                <span className="text-2xl md:text-3xl font-bold text-foreground tracking-tight">{value}</span>
                <span className="text-xs text-muted-foreground mt-1.5 font-medium uppercase tracking-wider">{label}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* ─── FOOTER ─── */}
      <footer className="py-10 px-6 border-t border-border bg-background text-center">
        <a 
          href="https://github.com/Ladnil03/bob-ai-hackathon-Thefentasticfour" 
          target="_blank" 
          rel="noreferrer"
          className="inline-block text-muted-foreground hover:text-foreground transition-colors mb-3"
        >
          <Code className="w-5 h-5" />
        </a>
        <p className="text-sm text-muted-foreground">
          Built for the IBM Bob AI Hackathon by Thefentasticfour.
        </p>
      </footer>
    </div>
  );
}
