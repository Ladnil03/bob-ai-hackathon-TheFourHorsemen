import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ArrowRight, Code } from 'lucide-react';
import logo from '../assets/logo.png';

export default function GlobalNav() {
  const location = useLocation();
  const isDashboard = location.pathname !== '/';

  return (
    <nav className="sticky top-0 z-50 w-full flex items-center justify-between px-6 py-4 bg-background/80 backdrop-blur-xl border-b border-border shadow-sm">
      <div className="flex items-center gap-3">
        <img src={logo} alt="Logo" className="w-9 h-9 rounded-lg object-contain" />
        <Link to="/" className="font-bold text-lg tracking-tight text-foreground hover:opacity-80 transition-opacity">
          SemiYield AI
        </Link>
      </div>

      <div className="flex items-center gap-6">
        {!isDashboard && (
          <div className="hidden md:flex items-center gap-6 text-sm font-medium text-muted-foreground mr-4">
            <a href="#problem" className="hover:text-foreground transition-colors">Problem</a>
            <a href="#solution" className="hover:text-foreground transition-colors">Solution</a>
            <a href="#how-it-works" className="hover:text-foreground transition-colors">How it works</a>
          </div>
        )}

        <a 
          href="https://github.com/Ladnil03/bob-ai-hackathon-Thefentasticfour" 
          target="_blank" 
          rel="noreferrer" 
          className="text-muted-foreground hover:text-foreground transition-colors flex items-center gap-2 font-medium text-sm"
        >
          <Code className="w-5 h-5" />
          <span className="hidden sm:inline">GitHub</span>
        </a>

        {!isDashboard && (
          <Link 
            to="/upload" 
            className="group flex items-center gap-2 bg-foreground text-background px-4 py-2 rounded-full font-medium text-sm transition-all hover:scale-105 hover:shadow-md"
          >
            <span>Dashboard</span>
            <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
          </Link>
        )}
      </div>
    </nav>
  );
}
