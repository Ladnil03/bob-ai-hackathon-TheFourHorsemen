import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu } from 'lucide-react';

const Preloader = ({ children }) => {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate initial loading sequence for assets/data
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1200); // 1.2s pre-loader

    return () => clearTimeout(timer);
  }, []);

  return (
    <>
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 1 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-white"
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5 }}
              className="flex flex-col items-center"
            >
              <div className="relative mb-6">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                  className="absolute inset-0 rounded-full border-t-2 border-primary border-opacity-50"
                  style={{ width: '64px', height: '64px', top: '-12px', left: '-12px' }}
                />
                <Cpu size={40} className="text-primary" />
              </div>
              
              <h2 className="text-2xl font-bold tracking-tight text-foreground">
                Thefentasticfour
              </h2>
              <p className="mt-2 text-sm text-muted-foreground uppercase tracking-widest">
                Initializing AI Copilot...
              </p>
              
              <div className="mt-8 h-1 w-48 overflow-hidden rounded-full bg-secondary">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: '100%' }}
                  transition={{ duration: 1.0, ease: "easeInOut" }}
                  className="h-full bg-primary"
                />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
      {!loading && children}
    </>
  );
};

export default Preloader;
