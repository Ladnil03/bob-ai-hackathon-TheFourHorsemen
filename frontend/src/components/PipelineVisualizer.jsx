import React from 'react';
import { motion } from 'framer-motion';
import { Cpu, BrainCircuit, Activity, LineChart, Database, ArrowRight } from 'lucide-react';

export default function PipelineVisualizer() {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.6 } }
  };

  const line = {
    hidden: { scaleX: 0, opacity: 0 },
    show: { scaleX: 1, opacity: 1, transition: { duration: 0.8, ease: "easeInOut" } }
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-card rounded-2xl border border-border shadow-sm">
      <h3 className="text-xl font-bold tracking-tight mb-8 text-center text-foreground">
        Closed-Loop AI Pipeline
      </h3>
      
      <motion.div 
        variants={container}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-50px" }}
        className="flex flex-col lg:flex-row items-center justify-between gap-4 lg:gap-0 relative"
      >
        
        {/* Node 1: Ingestion */}
        <motion.div variants={item} className="z-10 flex flex-col items-center text-center max-w-[180px]">
          <div className="w-16 h-16 bg-muted rounded-xl border border-border flex items-center justify-center mb-4 shadow-sm relative">
            <Database className="w-7 h-7 text-foreground" />
            <motion.div 
              animate={{ opacity: [0, 1, 0] }}
              transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
              className="absolute -right-2 top-1/2 w-2 h-2 rounded-full bg-primary"
            />
          </div>
          <h4 className="font-semibold text-sm">Sensor Ingestion</h4>
          <p className="text-[11px] text-muted-foreground mt-1">590+ streaming fab sensors</p>
        </motion.div>

        {/* Connection Line 1 */}
        <div className="hidden lg:block flex-1 h-[2px] bg-muted relative mx-2">
          <motion.div variants={line} className="absolute inset-0 bg-primary/50 origin-left" />
          <motion.div 
            animate={{ x: ["0%", "100%"] }} 
            transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
            className="absolute top-1/2 -translate-y-1/2 left-0 w-4 h-4 text-primary"
          >
            <ArrowRight className="w-4 h-4" />
          </motion.div>
        </div>

        {/* Node 2: Predictor */}
        <motion.div variants={item} className="z-10 flex flex-col items-center text-center max-w-[180px]">
          <div className="w-16 h-16 bg-muted rounded-xl border border-border flex items-center justify-center mb-4 shadow-sm">
            <Cpu className="w-7 h-7 text-foreground" />
          </div>
          <h4 className="font-semibold text-sm">ONNX Classifier</h4>
          <p className="text-[11px] text-muted-foreground mt-1">Sub-millisecond failure prediction</p>
        </motion.div>

        {/* Connection Line 2 */}
        <div className="hidden lg:block flex-1 h-[2px] bg-muted relative mx-2">
          <motion.div variants={line} className="absolute inset-0 bg-primary/50 origin-left" />
          <motion.div 
            animate={{ x: ["0%", "100%"] }} 
            transition={{ repeat: Infinity, duration: 1.5, ease: "linear", delay: 0.5 }}
            className="absolute top-1/2 -translate-y-1/2 left-0 w-4 h-4 text-primary"
          >
            <ArrowRight className="w-4 h-4" />
          </motion.div>
        </div>

        {/* Node 3: LLM Copilot */}
        <motion.div variants={item} className="z-10 flex flex-col items-center text-center max-w-[180px]">
          <div className="w-16 h-16 bg-muted rounded-xl border border-border flex items-center justify-center mb-4 shadow-sm relative">
            <BrainCircuit className="w-7 h-7 text-foreground" />
            <motion.div 
              animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
              transition={{ repeat: Infinity, duration: 2 }}
              className="absolute -inset-1 rounded-xl border border-primary/30"
            />
          </div>
          <h4 className="font-semibold text-sm">Groq 120B Copilot</h4>
          <p className="text-[11px] text-muted-foreground mt-1">Prescriptive root cause analysis</p>
        </motion.div>

        {/* Connection Line 3 */}
        <div className="hidden lg:block flex-1 h-[2px] bg-muted relative mx-2">
          <motion.div variants={line} className="absolute inset-0 bg-primary/50 origin-left" />
          <motion.div 
            animate={{ x: ["0%", "100%"] }} 
            transition={{ repeat: Infinity, duration: 1.5, ease: "linear", delay: 1 }}
            className="absolute top-1/2 -translate-y-1/2 left-0 w-4 h-4 text-primary"
          >
            <ArrowRight className="w-4 h-4" />
          </motion.div>
        </div>

        {/* Node 4: RL Feedback */}
        <motion.div variants={item} className="z-10 flex flex-col items-center text-center max-w-[180px]">
          <div className="w-16 h-16 bg-muted rounded-xl border border-border flex items-center justify-center mb-4 shadow-sm">
            <LineChart className="w-7 h-7 text-foreground" />
          </div>
          <h4 className="font-semibold text-sm">Contextual Bandit</h4>
          <p className="text-[11px] text-muted-foreground mt-1">Live recipe optimization</p>
        </motion.div>

      </motion.div>
    </div>
  );
}
