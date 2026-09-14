import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ArrowRight, Code } from 'lucide-react';
import BrandLogo from './BrandLogo';

export default function GlobalNav() {
  const location = useLocation();
  const isDashboard = location.pathname !== '/';

  return (
    <nav className="sticky top-0 z-50 w-full flex items-center justify-between px-6 py-4 bg-white/90 backdrop-blur-xl border-b border-slate-200 shadow-xs">
      <div className="flex items-center gap-3">
        <BrandLogo className="w-8 h-8" />
        <Link to="/" className="font-bold text-lg tracking-tight text-slate-900 hover:opacity-80 transition-opacity">
          SemiYield AI
        </Link>
      </div>

      <div className="hidden md:flex items-center gap-6 text-sm font-medium text-muted-foreground">
        <a href="#product-preview" className="hover:text-foreground transition-colors">Product</a>
        <a href="#predict" className="hover:text-foreground transition-colors">Intelligence</a>
        <a href="#optimize" className="hover:text-foreground transition-colors">Optimization</a>
        <a href="#technology" className="hover:text-foreground transition-colors">Technology</a>
        <a href="#intelligence-loop" className="hover:text-foreground transition-colors">Demo</a>
      </div>

      <div className="flex items-center gap-2">
        <a 
          href="https://github.com/Ladnil03/bob-ai-hackathon-Thefentasticfour" 
          target="_blank" 
          rel="noreferrer" 
          className="text-muted-foreground hover:text-foreground transition-colors flex items-center gap-2 font-medium text-sm"
        >
          <Code className="w-5 h-5" />
          <span className="hidden sm:inline">GitHub</span>
        </a>
        
        { !isDashboard && (
          <Link 
            to="/upload" 
            className="group flex items-center gap-2 px-4 py-2 bg-foreground text-background rounded-full font-medium text-sm transition-all hover:scale-105 hover:shadow-md"
          >
            OPEN COMMAND CENTER →
            <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
          </Link>
        ) }
      </div>
    </nav>
  );
}