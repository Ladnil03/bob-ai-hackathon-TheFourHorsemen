import React from 'react';

export default function BrandLogo({ className = "w-7 h-7" }) {
  return (
    <svg
      viewBox="0 0 40 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`shrink-0 ${className}`}
    >
      {/* Outer Chip Frame */}
      <rect x="5" y="5" width="30" height="30" rx="6" fill="#2563EB" fillOpacity="0.1" stroke="#2563EB" strokeWidth="2" />
      
      {/* Silicon Die Substrate Grid */}
      <rect x="11" y="11" width="18" height="18" rx="3" fill="#1E40AF" fillOpacity="0.85" stroke="#3B82F6" strokeWidth="1.5" />
      
      {/* Wafer Die Quad Grid */}
      <rect x="13.5" y="13.5" width="5.5" height="5.5" rx="1" fill="#60A5FA" />
      <rect x="21" y="13.5" width="5.5" height="5.5" rx="1" fill="#3B82F6" />
      <rect x="13.5" y="21" width="5.5" height="5.5" rx="1" fill="#3B82F6" />
      <rect x="21" y="21" width="5.5" height="5.5" rx="1" fill="#10B981" />
      
      {/* Chip Pins / Leads */}
      <path d="M12 2V5 M20 2V5 M28 2V5" stroke="#2563EB" strokeWidth="1.75" strokeLinecap="round" />
      <path d="M12 35V38 M20 35V38 M28 35V38" stroke="#2563EB" strokeWidth="1.75" strokeLinecap="round" />
      <path d="M2 12H5 M2 20H5 M2 28H5" stroke="#2563EB" strokeWidth="1.75" strokeLinecap="round" />
      <path d="M35 12H38 M35 20H38 M35 28H38" stroke="#2563EB" strokeWidth="1.75" strokeLinecap="round" />
      
      {/* Center Core Node */}
      <circle cx="20" cy="20" r="1.5" fill="#FFFFFF" />
    </svg>
  );
}
