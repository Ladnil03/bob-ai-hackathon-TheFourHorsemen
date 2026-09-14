import React from 'react';

export default function IconPlaceholder({ name, className = "" }) {
  // A simple placeholder box that renders the name inside it (or just an empty box if small)
  return (
    <div 
      className={`inline-flex items-center justify-center bg-muted border border-border/50 text-muted-foreground rounded text-[10px] uppercase font-bold overflow-hidden ${className}`}
      title={`Icon Placeholder: ${name}`}
    >
      <span className="truncate px-1 opacity-50">{name.slice(0, 3)}</span>
    </div>
  );
}
