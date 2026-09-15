import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

/* ═══════════════════════════════════════════════════════════════════════
   SEMICONDUCTOR 3D — DETAILED CHIP PACKAGE WITH DATA FLOW
   
   Multi-layer IC package that explodes apart on scroll, then when
   fully assembled shows live data flow pulses through pins & traces.
   
   Scroll behavior:
     progress 0.0  → Fully assembled + active data flow through pins
     progress 0.1+ → Layers begin separating from top down
     progress 0.5  → Maximum explosion, all layers visible
     progress 0.9  → Layers reassembled
     progress 1.0  → Fully assembled + active data flow (green accent)
   ═══════════════════════════════════════════════════════════════════════ */

export default function Semiconductor3D({ progress = 0 }) {
  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ position: [4.5, 3.8, 5.5], fov: 38 }}
        dpr={[1, 1.5]}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        style={{ background: 'transparent' }}
        shadows
      >
        <SceneLighting progress={progress} />
        <ChipPackage progress={progress} />
        <OrbitControls 
          enableZoom={false} 
          enablePan={false} 
          enableRotate={false}
          autoRotate 
          autoRotateSpeed={0.3}
          maxPolarAngle={Math.PI / 2.8}
          minPolarAngle={Math.PI / 3.5}
          maxAzimuthAngle={Math.PI / 4}
          minAzimuthAngle={-Math.PI / 4}
        />
      </Canvas>
    </div>
  );
}

/* ─── Scene Lighting ─── */
function SceneLighting({ progress }) {
  const assembled = 1 - Math.sin(progress * Math.PI);
  return (
    <>
      <ambientLight intensity={0.45} />
      <directionalLight 
        position={[8, 12, 6]} intensity={1.4} castShadow
        shadow-mapSize-width={1024} shadow-mapSize-height={1024}
        color="#f8fafc"
      />
      <directionalLight position={[-5, 6, -4]} intensity={0.5} color="#bfdbfe" />
      {/* Die glow when assembled */}
      <pointLight 
        position={[0, 1.2, 0]} 
        intensity={assembled * 1.2} 
        color="#3b82f6" 
        distance={5} 
      />
      <pointLight position={[0, -1, 0]} intensity={0.2} color="#fbbf24" distance={6} />
    </>
  );
}

/* ═══ Main Chip Package ═══ */
function ChipPackage({ progress }) {
  const groupRef = useRef();
  
  // Explode: bell curve peaking at 0.5
  const explode = Math.sin(progress * Math.PI);
  // Assembled factor: 1 when progress is near 0 or 1
  const assembled = Math.max(0, 1 - explode * 2.5);
  
  // Section color
  const accent = useMemo(() => {
    if (progress < 0.2) return new THREE.Color('#3b82f6');
    if (progress < 0.4) return new THREE.Color('#06b6d4');
    if (progress < 0.6) return new THREE.Color('#ef4444');
    if (progress < 0.8) return new THREE.Color('#f59e0b');
    return new THREE.Color('#10b981');
  }, [progress]);

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.x = THREE.MathUtils.lerp(
        groupRef.current.rotation.x, 0.25 + progress * 0.12, 0.04
      );
    }
  });

  return (
    <group ref={groupRef} position={[0, 0, 0]}>
      
      {/* ── Layer 0: PCB Motherboard ── */}
      <ExplodingLayer y={0} explode={explode} offset={-3.2} floatAmp={0.8}>
        <PCBBoard assembled={assembled} />
      </ExplodingLayer>

      {/* ── Layer 1: BGA Solder Balls ── */}
      <ExplodingLayer y={0.22} explode={explode} offset={-2.4} floatAmp={1.0}>
        <SolderBallArray />
      </ExplodingLayer>

      {/* ── Layer 2: Package Substrate ── */}
      <ExplodingLayer y={0.45} explode={explode} offset={-1.5} floatAmp={1.2}>
        <PackageSubstrate />
      </ExplodingLayer>

      {/* ── Layer 3: Die Attach ── */}
      <ExplodingLayer y={0.66} explode={explode} offset={-0.7} floatAmp={1.1}>
        <DieAttach />
      </ExplodingLayer>

      {/* ── Layer 4: Silicon Die ── */}
      <ExplodingLayer y={0.82} explode={explode} offset={0.1} floatAmp={1.3}>
        <SiliconDie accent={accent} explode={explode} assembled={assembled} />
      </ExplodingLayer>

      {/* ── Layer 5: Bond Wires ── */}
      <ExplodingLayer y={0.92} explode={explode} offset={0.7} floatAmp={1.0}>
        <BondWires assembled={assembled} />
      </ExplodingLayer>

      {/* ── Layer 6: Mold Compound ── */}
      <ExplodingLayer y={1.12} explode={explode} offset={1.5} floatAmp={0.9}>
        <MoldCompound />
      </ExplodingLayer>

      {/* ── Layer 7: Thermal Paste ── */}
      <ExplodingLayer y={1.32} explode={explode} offset={2.1} floatAmp={1.1}>
        <ThermalPaste />
      </ExplodingLayer>

      {/* ── Layer 8: IHS (Heat Spreader) with CPU branding ── */}
      <ExplodingLayer y={1.48} explode={explode} offset={2.9} floatAmp={0.7}>
        <HeatSpreader assembled={assembled} />
      </ExplodingLayer>

      {/* ── Data Flow Particles (visible when assembled) ── */}
      <DataFlowParticles assembled={assembled} accent={accent} />

      {/* ── Pin Data Pulses (visible when assembled) ── */}
      <PinDataPulses assembled={assembled} accent={accent} />

      {/* ── Floating ambient particles ── */}
      <AmbientParticles accent={accent} explode={explode} />

      {/* ── Layer labels when exploded ── */}
      <LayerLabels explode={explode} />
    </group>
  );
}

/* ─── Exploding Layer ─── */
function ExplodingLayer({ children, y, explode, offset, floatAmp = 1 }) {
  const ref = useRef();
  const targetY = y + explode * offset;

  useFrame((state) => {
    if (ref.current) {
      ref.current.position.y = THREE.MathUtils.lerp(ref.current.position.y, targetY, 0.06);
      ref.current.position.y += Math.sin(state.clock.elapsedTime * floatAmp + offset) * 0.006 * explode;
    }
  });

  return <group ref={ref} position={[0, y, 0]}>{children}</group>;
}


/* ═══════════════════════════════════════════════════════
   PCB MOTHERBOARD — Green FR4 with traces, vias, caps
   ═══════════════════════════════════════════════════════ */
function PCBBoard({ assembled }) {
  const vias = useMemo(() => {
    const v = [];
    for (let x = -1.4; x <= 1.4; x += 0.4) {
      for (let z = -1.4; z <= 1.4; z += 0.4) {
        if (Math.random() > 0.25) v.push([x + (Math.random() - 0.5) * 0.08, z + (Math.random() - 0.5) * 0.08]);
      }
    }
    return v;
  }, []);

  // Surface mount components (caps, resistors)
  const smds = useMemo(() => [
    [-1.2, 1.0, 0.12, 0.06], [1.0, 1.2, 0.06, 0.12], [-0.8, -1.3, 0.12, 0.06],
    [1.3, -0.6, 0.06, 0.14], [-1.4, 0.3, 0.1, 0.06], [0.5, 1.4, 0.06, 0.1],
    [-0.3, -1.5, 0.12, 0.06], [1.5, 0.8, 0.06, 0.12], [-1.1, -0.8, 0.1, 0.06],
  ], []);

  return (
    <group>
      {/* FR4 Board */}
      <mesh castShadow receiveShadow>
        <boxGeometry args={[3.8, 0.12, 3.8]} />
        <meshStandardMaterial color="#15803d" roughness={0.75} metalness={0.05} />
      </mesh>
      {/* Solder mask */}
      <mesh position={[0, 0.061, 0]}>
        <boxGeometry args={[3.78, 0.002, 3.78]} />
        <meshStandardMaterial color="#166534" roughness={0.9} />
      </mesh>
      {/* Main copper traces (H + V) */}
      {Array.from({ length: 12 }).map((_, i) => {
        const t = (i / 11) * 3.2 - 1.6;
        const isMain = Math.abs(t) < 0.5;
        return (
          <group key={`tr-${i}`}>
            <mesh position={[t, 0.063, 0]}>
              <boxGeometry args={[isMain ? 0.05 : 0.03, 0.004, 3.6]} />
              <meshStandardMaterial color={isMain ? '#d4a056' : '#b8860b'} metalness={0.85} roughness={0.25} />
            </mesh>
            <mesh position={[0, 0.063, t]}>
              <boxGeometry args={[3.6, 0.004, isMain ? 0.05 : 0.03]} />
              <meshStandardMaterial color={isMain ? '#d4a056' : '#b8860b'} metalness={0.85} roughness={0.25} />
            </mesh>
          </group>
        );
      })}
      {/* Via holes */}
      {vias.map(([x, z], i) => (
        <mesh key={`via-${i}`} position={[x, 0.064, z]}>
          <cylinderGeometry args={[0.025, 0.025, 0.012, 8]} />
          <meshStandardMaterial color="#c9a84c" metalness={0.9} roughness={0.2} />
        </mesh>
      ))}
      {/* Surface mount components */}
      {smds.map(([x, z, w, h], i) => (
        <group key={`smd-${i}`}>
          <mesh position={[x, 0.075, z]}>
            <boxGeometry args={[w, 0.04, h]} />
            <meshStandardMaterial color={i % 3 === 0 ? '#1e293b' : '#44403c'} roughness={0.7} />
          </mesh>
          {/* Solder pads */}
          <mesh position={[x - w * 0.35, 0.065, z]}>
            <boxGeometry args={[w * 0.3, 0.008, h * 0.8]} />
            <meshStandardMaterial color="#c9a84c" metalness={0.9} roughness={0.15} />
          </mesh>
          <mesh position={[x + w * 0.35, 0.065, z]}>
            <boxGeometry args={[w * 0.3, 0.008, h * 0.8]} />
            <meshStandardMaterial color="#c9a84c" metalness={0.9} roughness={0.15} />
          </mesh>
        </group>
      ))}
      {/* Edge gold fingers */}
      {Array.from({ length: 18 }).map((_, i) => {
        const t = (i / 17) * 3.2 - 1.6;
        return (
          <group key={`ef-${i}`}>
            <mesh position={[t, 0.04, 1.92]}><boxGeometry args={[0.1, 0.03, 0.06]} /><meshStandardMaterial color="#daa520" metalness={0.92} roughness={0.12} /></mesh>
            <mesh position={[t, 0.04, -1.92]}><boxGeometry args={[0.1, 0.03, 0.06]} /><meshStandardMaterial color="#daa520" metalness={0.92} roughness={0.12} /></mesh>
          </group>
        );
      })}
      {/* Mounting holes */}
      {[[-1.7, -1.7], [-1.7, 1.7], [1.7, -1.7], [1.7, 1.7]].map(([x, z], i) => (
        <mesh key={`mh-${i}`} position={[x, 0.064, z]} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[0.08, 0.025, 8, 16]} />
          <meshStandardMaterial color="#c9a84c" metalness={0.85} roughness={0.2} />
        </mesh>
      ))}
    </group>
  );
}


/* ═══════════════════════════════════
   BGA SOLDER BALL ARRAY
   ═══════════════════════════════════ */
function SolderBallArray() {
  const balls = useMemo(() => {
    const items = [];
    const count = 14;
    const spacing = 0.19;
    const half = (count - 1) * spacing / 2;
    for (let x = 0; x < count; x++) {
      for (let z = 0; z < count; z++) {
        const cx = x * spacing - half;
        const cz = z * spacing - half;
        if (Math.abs(cx) < 0.25 && Math.abs(cz) < 0.25) continue;
        items.push([cx, cz]);
      }
    }
    return items;
  }, []);

  return (
    <group>
      {balls.map(([x, z], i) => (
        <mesh key={i} position={[x, -0.04, z]} castShadow>
          <sphereGeometry args={[0.04, 10, 10]} />
          <meshStandardMaterial color="#9ca3af" metalness={0.85} roughness={0.25} />
        </mesh>
      ))}
      <mesh position={[0, -0.035, 0]}>
        <cylinderGeometry args={[0.22, 0.22, 0.015, 16]} />
        <meshStandardMaterial color="#d4a056" metalness={0.9} roughness={0.15} />
      </mesh>
    </group>
  );
}


/* ═══════════════════════════════════
   PACKAGE SUBSTRATE
   ═══════════════════════════════════ */
function PackageSubstrate() {
  const routing = useMemo(() => {
    const traces = [];
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 14) {
      traces.push({ angle, inner: 0.28, outer: 1.15 });
    }
    return traces;
  }, []);

  return (
    <group>
      <mesh castShadow receiveShadow>
        <boxGeometry args={[2.9, 0.15, 2.9]} />
        <meshStandardMaterial color="#1a1a2e" roughness={0.8} metalness={0.1} />
      </mesh>
      <mesh position={[0, 0.076, 0]}>
        <boxGeometry args={[2.88, 0.002, 2.88]} />
        <meshStandardMaterial color="#2d1b0e" roughness={0.85} metalness={0.15} />
      </mesh>
      {routing.map((r, i) => {
        const dx = Math.cos(r.angle);
        const dz = Math.sin(r.angle);
        const len = r.outer - r.inner;
        const mx = (r.inner + r.outer) / 2 * dx;
        const mz = (r.inner + r.outer) / 2 * dz;
        return (
          <mesh key={i} position={[mx, 0.078, mz]} rotation={[0, -r.angle, 0]}>
            <boxGeometry args={[len, 0.003, 0.018]} />
            <meshStandardMaterial color="#c9a84c" metalness={0.9} roughness={0.2} />
          </mesh>
        );
      })}
      {/* Bond pads ring */}
      {Array.from({ length: 36 }).map((_, i) => {
        const angle = (i / 36) * Math.PI * 2;
        const r = 0.9;
        return (
          <mesh key={`bp-${i}`} position={[Math.cos(angle) * r, 0.079, Math.sin(angle) * r]}>
            <boxGeometry args={[0.055, 0.004, 0.055]} />
            <meshStandardMaterial color="#daa520" metalness={0.92} roughness={0.12} />
          </mesh>
        );
      })}
    </group>
  );
}


/* ═══════════════════════════════════
   DIE ATTACH
   ═══════════════════════════════════ */
function DieAttach() {
  return (
    <mesh castShadow>
      <boxGeometry args={[1.5, 0.04, 1.5]} />
      <meshStandardMaterial color="#78350f" roughness={0.7} metalness={0.1} transparent opacity={0.85} />
    </mesh>
  );
}


/* ═══════════════════════════════════════════════════════
   SILICON DIE — With detailed macro blocks & power grid
   ═══════════════════════════════════════════════════════ */
function SiliconDie({ accent, explode, assembled }) {
  const dieRef = useRef();

  // Pulsing glow when assembled
  useFrame((state) => {
    if (dieRef.current) {
      const pulse = assembled * (0.1 + Math.sin(state.clock.elapsedTime * 2) * 0.05);
      dieRef.current.emissiveIntensity = pulse;
    }
  });

  const macroCells = useMemo(() => [
    { x: -0.38, z: -0.38, w: 0.3, h: 0.3, label: 'ALU Core 0' },
    { x: 0.02, z: -0.38, w: 0.3, h: 0.3, label: 'ALU Core 1' },
    { x: 0.42, z: -0.38, w: 0.18, h: 0.3, label: 'L1$' },
    { x: -0.38, z: 0.02, w: 0.42, h: 0.22, label: 'L2 Unified Cache' },
    { x: 0.12, z: 0.02, w: 0.48, h: 0.12, label: 'Memory Controller' },
    { x: 0.12, z: 0.18, w: 0.48, h: 0.12, label: 'PCIe / I/O PHY' },
    { x: -0.38, z: 0.32, w: 0.22, h: 0.18, label: 'PLL Array' },
    { x: -0.1, z: 0.32, w: 0.18, h: 0.18, label: 'PMU' },
    { x: 0.14, z: 0.32, w: 0.14, h: 0.18, label: 'JTAG' },
    { x: 0.34, z: 0.32, w: 0.26, h: 0.18, label: 'SerDes' },
  ], []);

  return (
    <group>
      {/* Silicon substrate */}
      <mesh castShadow receiveShadow>
        <boxGeometry args={[1.35, 0.08, 1.35]} />
        <meshStandardMaterial color="#0f172a" roughness={0.4} metalness={0.3} />
      </mesh>
      
      {/* Active die surface */}
      <mesh position={[0, 0.041, 0]}>
        <boxGeometry args={[1.33, 0.002, 1.33]} />
        <meshStandardMaterial 
          ref={dieRef}
          color="#1e3a5f" roughness={0.35} metalness={0.5}
          emissive={accent}
          emissiveIntensity={0.08}
        />
      </mesh>

      {/* I/O Pad Ring — gold pads around perimeter */}
      {Array.from({ length: 28 }).map((_, i) => {
        const side = Math.floor(i / 7);
        const idx = i % 7;
        const t = (idx / 6) * 1.1 - 0.55;
        const positions = [
          [t, 0.043, -0.65], [t, 0.043, 0.65],
          [-0.65, 0.043, t], [0.65, 0.043, t],
        ];
        return (
          <mesh key={`pad-${i}`} position={positions[side]}>
            <boxGeometry args={[0.045, 0.004, 0.045]} />
            <meshStandardMaterial 
              color="#fbbf24" metalness={0.95} roughness={0.1} 
              emissive="#fbbf24" emissiveIntensity={assembled * 0.4}
            />
          </mesh>
        );
      })}

      {/* Macro cell functional blocks */}
      {macroCells.map((cell, ci) => (
        <group key={ci}>
          <mesh position={[cell.x + cell.w / 2, 0.044, cell.z + cell.h / 2]}>
            <boxGeometry args={[cell.w - 0.02, 0.003, cell.h - 0.02]} />
            <meshStandardMaterial 
              color={accent} transparent opacity={0.15 + explode * 0.25}
              emissive={accent} emissiveIntensity={0.25 + assembled * 0.3}
            />
          </mesh>
          {/* Cell border */}
          <lineSegments position={[cell.x + cell.w / 2, 0.046, cell.z + cell.h / 2]}>
            <edgesGeometry args={[new THREE.BoxGeometry(cell.w - 0.02, 0.001, cell.h - 0.02)]} />
            <lineBasicMaterial color={accent} transparent opacity={0.4 + explode * 0.3} />
          </lineSegments>
          {/* Internal transistor grid pattern */}
          {Array.from({ length: 3 }).map((_, gi) => {
            const lx = cell.x + 0.03 + (gi / 2) * (cell.w - 0.06);
            return (
              <mesh key={`tg-${ci}-${gi}`} position={[lx, 0.045, cell.z + cell.h / 2]}>
                <boxGeometry args={[0.003, 0.001, cell.h - 0.04]} />
                <meshStandardMaterial color={accent} transparent opacity={0.2} />
              </mesh>
            );
          })}
        </group>
      ))}

      {/* Power distribution grid */}
      {Array.from({ length: 14 }).map((_, i) => {
        const t = (i / 13) * 1.2 - 0.6;
        return (
          <group key={`pgrid-${i}`}>
            <mesh position={[t, 0.0435, 0]}>
              <boxGeometry args={[0.004, 0.001, 1.25]} />
              <meshStandardMaterial color="#94a3b8" transparent opacity={0.2} metalness={0.9} />
            </mesh>
            <mesh position={[0, 0.0435, t]}>
              <boxGeometry args={[1.25, 0.001, 0.004]} />
              <meshStandardMaterial color="#94a3b8" transparent opacity={0.2} metalness={0.9} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}


/* ═══════════════════════════════════
   BOND WIRES — Gold arcs from die to substrate
   ═══════════════════════════════════ */
function BondWires({ assembled }) {
  const wires = useMemo(() => {
    const items = [];
    for (let i = 0; i < 28; i++) {
      const angle = (i / 28) * Math.PI * 2;
      const innerR = 0.58;
      const outerR = 1.05;
      const start = new THREE.Vector3(Math.cos(angle) * innerR, 0, Math.sin(angle) * innerR);
      const end = new THREE.Vector3(Math.cos(angle) * outerR, -0.05, Math.sin(angle) * outerR);
      const mid = new THREE.Vector3(
        (start.x + end.x) / 2,
        0.22 + Math.random() * 0.08,
        (start.z + end.z) / 2
      );
      items.push(new THREE.QuadraticBezierCurve3(start, mid, end));
    }
    return items;
  }, []);

  return (
    <group>
      {wires.map((curve, i) => (
        <mesh key={i}>
          <tubeGeometry args={[curve, 14, 0.007, 6, false]} />
          <meshStandardMaterial 
            color="#fbbf24" metalness={0.95} roughness={0.1}
            emissive="#fbbf24" emissiveIntensity={0.05 + assembled * 0.25}
          />
        </mesh>
      ))}
    </group>
  );
}


/* ═══════════════════════════════════
   MOLD COMPOUND
   ═══════════════════════════════════ */
function MoldCompound() {
  return (
    <group>
      <mesh castShadow receiveShadow>
        <boxGeometry args={[2.5, 0.28, 2.5]} />
        <meshStandardMaterial color="#1c1917" roughness={0.85} metalness={0.05} transparent opacity={0.65} />
      </mesh>
      {/* Laser marking area */}
      <mesh position={[0, 0.141, 0]}>
        <boxGeometry args={[1.2, 0.002, 0.25]} />
        <meshStandardMaterial color="#292524" roughness={0.9} />
      </mesh>
      {/* Pin 1 dot */}
      <mesh position={[-0.95, 0.142, -0.95]}>
        <cylinderGeometry args={[0.055, 0.055, 0.003, 12]} />
        <meshStandardMaterial color="#44403c" roughness={0.8} />
      </mesh>
    </group>
  );
}


/* ═══════════════════════════════════
   THERMAL PASTE
   ═══════════════════════════════════ */
function ThermalPaste() {
  return (
    <mesh>
      <boxGeometry args={[2.1, 0.025, 2.1]} />
      <meshStandardMaterial color="#6b7280" roughness={0.95} metalness={0.1} transparent opacity={0.55} />
    </mesh>
  );
}


/* ═══════════════════════════════════════════════════════
   IHS WITH CPU BRANDING — Shows label when assembled
   ═══════════════════════════════════════════════════════ */
function HeatSpreader({ assembled }) {
  const brandRef = useRef();
  
  useFrame((state) => {
    if (brandRef.current) {
      brandRef.current.material.opacity = assembled * 0.85;
    }
  });

  return (
    <group>
      {/* IHS body */}
      <mesh castShadow receiveShadow>
        <boxGeometry args={[2.9, 0.2, 2.9]} />
        <meshStandardMaterial color="#a8a29e" metalness={0.92} roughness={0.18} />
      </mesh>
      {/* Top surface branding area */}
      <mesh position={[0, 0.101, 0]}>
        <boxGeometry args={[1.8, 0.002, 0.9]} />
        <meshStandardMaterial color="#78716c" metalness={0.88} roughness={0.25} />
      </mesh>
      {/* CPU Text Label — visible when assembled */}
      <Text
        ref={brandRef}
        position={[0, 0.108, -0.08]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.22}
        color="#57534e"
        anchorX="center"
        anchorY="middle"
        font={undefined}
        material-transparent
        material-opacity={assembled * 0.85}
      >
        SemiYield AI
      </Text>
      <Text
        position={[0, 0.108, 0.18]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.1}
        color="#78716c"
        anchorX="center"
        anchorY="middle"
        font={undefined}
        material-transparent
        material-opacity={assembled * 0.7}
      >
        590-Sensor Yield Processor
      </Text>
      {/* Edge wireframe */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(2.9, 0.2, 2.9)]} />
        <lineBasicMaterial color="#d6d3d1" transparent opacity={0.4} />
      </lineSegments>
      {/* Retention clip notches */}
      {[[-1.47, 0], [1.47, 0], [0, -1.47], [0, 1.47]].map(([x, z], i) => (
        <mesh key={i} position={[x, 0, z]}>
          <boxGeometry args={[x === 0 ? 0.3 : 0.06, 0.22, z === 0 ? 0.3 : 0.06]} />
          <meshStandardMaterial color="#78716c" metalness={0.9} roughness={0.2} />
        </mesh>
      ))}
    </group>
  );
}


/* ═══════════════════════════════════════════════════════════════
   DATA FLOW PARTICLES — Travel through pins when chip assembled
   Simulates electrical signals flowing in/out of the package
   ═══════════════════════════════════════════════════════════════ */
function DataFlowParticles({ assembled, accent }) {
  const groupRef = useRef();
  const particleCount = 40;
  
  // Create particles that flow along pin traces
  const particles = useMemo(() => {
    const items = [];
    for (let i = 0; i < particleCount; i++) {
      const edge = Math.floor(Math.random() * 4);
      const t = Math.random() * 3.2 - 1.6;
      let startX, startZ, endX, endZ;
      
      // Flow from PCB edge toward center
      switch(edge) {
        case 0: startX = t; startZ = 1.9; endX = t * 0.2; endZ = 0; break;
        case 1: startX = t; startZ = -1.9; endX = t * 0.2; endZ = 0; break;
        case 2: startX = 1.9; startZ = t; endX = 0; endZ = t * 0.2; break;
        default: startX = -1.9; startZ = t; endX = 0; endZ = t * 0.2; break;
      }
      
      items.push({
        startX, startZ, endX, endZ,
        speed: 0.3 + Math.random() * 0.6,
        phase: Math.random() * Math.PI * 2,
        y: 0.1 + Math.random() * 0.15,
      });
    }
    return items;
  }, []);

  useFrame((state) => {
    if (!groupRef.current || assembled < 0.3) return;
    const t = state.clock.elapsedTime;
    
    groupRef.current.children.forEach((child, i) => {
      if (i >= particles.length) return;
      const p = particles[i];
      const progress = ((t * p.speed + p.phase) % 1);
      
      child.position.x = THREE.MathUtils.lerp(p.startX, p.endX, progress);
      child.position.z = THREE.MathUtils.lerp(p.startZ, p.endZ, progress);
      child.position.y = p.y + Math.sin(progress * Math.PI) * 0.3;
      child.material.opacity = assembled * Math.sin(progress * Math.PI) * 0.8;
      child.scale.setScalar(0.8 + Math.sin(progress * Math.PI) * 0.5);
    });
  });

  if (assembled < 0.2) return null;

  return (
    <group ref={groupRef}>
      {particles.map((_, i) => (
        <mesh key={i} position={[0, 0, 0]}>
          <sphereGeometry args={[0.025, 6, 6]} />
          <meshBasicMaterial 
            color={accent} transparent opacity={0}
          />
        </mesh>
      ))}
    </group>
  );
}


/* ═══════════════════════════════════════════════════════════════
   PIN DATA PULSES — Glowing rings that pulse outward from center
   when chip is active (assembled)
   ═══════════════════════════════════════════════════════════════ */
function PinDataPulses({ assembled, accent }) {
  const ring1 = useRef();
  const ring2 = useRef();
  const ring3 = useRef();

  useFrame((state) => {
    if (assembled < 0.5) return;
    const t = state.clock.elapsedTime;
    
    [ring1, ring2, ring3].forEach((ref, i) => {
      if (!ref.current) return;
      const phase = (t * 0.4 + i * 0.33) % 1;
      const scale = 0.5 + phase * 3;
      ref.current.scale.set(scale, 1, scale);
      ref.current.material.opacity = assembled * (1 - phase) * 0.3;
    });
  });

  if (assembled < 0.4) return null;

  return (
    <group position={[0, 0.15, 0]} rotation={[Math.PI / 2, 0, 0]}>
      <mesh ref={ring1}>
        <torusGeometry args={[0.5, 0.015, 8, 32]} />
        <meshBasicMaterial color={accent} transparent opacity={0} side={THREE.DoubleSide} />
      </mesh>
      <mesh ref={ring2}>
        <torusGeometry args={[0.5, 0.012, 8, 32]} />
        <meshBasicMaterial color={accent} transparent opacity={0} side={THREE.DoubleSide} />
      </mesh>
      <mesh ref={ring3}>
        <torusGeometry args={[0.5, 0.01, 8, 32]} />
        <meshBasicMaterial color={accent} transparent opacity={0} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}


/* ═══════════════════════════════════════════════════════
   LAYER LABELS — Visible when exploded
   ═══════════════════════════════════════════════════════ */
function LayerLabels({ explode }) {
  if (explode < 0.3) return null;
  
  const fade = Math.min(1, (explode - 0.3) * 3);

  const labels = [
    { y: -3.2, text: 'PCB Motherboard', color: '#22c55e' },
    { y: -2.2, text: 'BGA Solder Array', color: '#94a3b8' },
    { y: -1.3, text: 'Package Substrate', color: '#c9a84c' },
    { y: -0.5, text: 'Die Attach Film', color: '#92400e' },
    { y: 0.1, text: 'Silicon Die', color: '#3b82f6' },
    { y: 0.9, text: 'Bond Wires (Au)', color: '#eab308' },
    { y: 1.6, text: 'Mold Compound', color: '#78716c' },
    { y: 2.2, text: 'Thermal Interface', color: '#6b7280' },
    { y: 3.0, text: 'Heat Spreader (IHS)', color: '#a8a29e' },
  ];
  
  return (
    <group>
      {labels.map((l, i) => {
        const yPos = l.y * explode + 0.8;
        return (
          <group key={i} position={[0, yPos, 0]}>
            {/* Connector line from chip edge to label */}
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position" count={2}
                  array={new Float32Array([1.6, 0, 0, 2.3, 0, 0])}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color={l.color} transparent opacity={fade * 0.6} />
            </line>
            {/* Dot at connection point */}
            <mesh position={[1.6, 0, 0]}>
              <sphereGeometry args={[0.03, 8, 8]} />
              <meshBasicMaterial color={l.color} transparent opacity={fade * 0.8} />
            </mesh>
            {/* Text label */}
            <Text
              position={[2.4, 0, 0]}
              fontSize={0.12}
              color={l.color}
              anchorX="left"
              anchorY="middle"
              material-transparent
              material-opacity={fade * 0.9}
            >
              {l.text}
            </Text>
          </group>
        );
      })}
    </group>
  );
}


/* ═══════════════════════════════════════════════════════
   AMBIENT PARTICLES
   ═══════════════════════════════════════════════════════ */
function AmbientParticles({ accent, explode }) {
  const ref = useRef();
  const count = 70;
  
  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = 2.8 + Math.random() * 2;
      pos[i * 3] = Math.cos(angle) * r;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 5;
      pos[i * 3 + 2] = Math.sin(angle) * r;
    }
    return pos;
  }, []);

  useFrame((state) => {
    if (!ref.current) return;
    const arr = ref.current.geometry.attributes.position.array;
    const t = state.clock.elapsedTime;
    for (let i = 0; i < count; i++) {
      arr[i * 3 + 1] += Math.sin(t * 0.3 + i * 0.5) * 0.002;
    }
    ref.current.geometry.attributes.position.needsUpdate = true;
    ref.current.material.opacity = 0.1 + explode * 0.35;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={count} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial size={0.03} color={accent} transparent opacity={0.15} sizeAttenuation />
    </points>
  );
}
