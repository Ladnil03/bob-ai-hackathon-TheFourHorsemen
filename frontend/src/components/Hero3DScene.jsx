import React, { useRef, useMemo, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

/* ─── Die grid generation ─── */
function useDieGrid(radius, dieSize, gap) {
  return useMemo(() => {
    const spacing = dieSize + gap;
    const dies = [];
    const edge = 0.35;

    for (let x = -radius; x < radius; x += spacing) {
      for (let z = -radius; z < radius; z += spacing) {
        const cx = x + spacing / 2;
        const cz = z + spacing / 2;
        const dist = Math.sqrt(cx * cx + cz * cz);
        if (dist > radius - edge) continue;

        // Notch at bottom
        if (cz > radius - 0.6 && Math.abs(cx) < 0.35) continue;

        // Deterministic risk assignment (clustered around center-right)
        const seed = Math.sin(cx * 5.17 + cz * 3.89) * 0.5 + 0.5;
        const clusterDist = Math.sqrt((cx - 0.8) ** 2 + (cz + 0.4) ** 2);
        let risk = 0;
        if (clusterDist < 1.5 && seed > 0.55) risk = 0.4 + seed * 0.3;
        if (clusterDist < 0.8 && seed > 0.5)  risk = 0.75 + seed * 0.25;

        dies.push({ x: cx, z: cz, dist, risk, idx: dies.length });
      }
    }
    return dies;
  }, [radius, dieSize, gap]);
}

/* ─── Semiconductor Wafer ─── */
function SemiconductorWafer({ progress }) {
  const groupRef = useRef();
  const dieRef = useRef();

  const WAFER_R = 3.2;
  const DIE_SIZE = 0.19;
  const DIE_GAP = 0.025;
  const DIE_H = 0.018;
  const WAFER_H = 0.14;

  const dies = useDieGrid(WAFER_R, DIE_SIZE, DIE_GAP);

  const dummy  = useMemo(() => new THREE.Object3D(), []);
  const cA     = useMemo(() => new THREE.Color(), []);
  const cB     = useMemo(() => new THREE.Color(), []);
  const cOut   = useMemo(() => new THREE.Color(), []);

  // Pre-allocate color constants
  const COL = useMemo(() => ({
    healthy:    new THREE.Color('#2c2c38'),
    healthyLit: new THREE.Color('#333340'),
    amber:      new THREE.Color('#b45309'),
    red:        new THREE.Color('#dc2626'),
    green:      new THREE.Color('#16a34a'),
    greenDark:  new THREE.Color('#1a3328'),
  }), []);

  useFrame((state, delta) => {
    if (!groupRef.current || !dieRef.current) return;

    // Slow continuous rotation
    groupRef.current.rotation.y += delta * 0.06;

    const t = state.clock.elapsedTime;

    dies.forEach((die, i) => {
      // Slight height shimmer on surface
      const shimmer = Math.sin(die.x * 4 + t * 0.5) * Math.cos(die.z * 4 + t * 0.3) * 0.003;
      dummy.position.set(die.x, WAFER_H / 2 + DIE_H / 2 + shimmer, die.z);
      dummy.scale.setScalar(1);
      dummy.updateMatrix();
      dieRef.current.setMatrixAt(i, dummy.matrix);

      // ─── Die color logic based on scroll progress ───
      const p = progress;

      if (p < 0.25) {
        // HERO: all healthy, subtle
        cOut.copy(COL.healthy);
      } else if (p < 0.45) {
        // SENSE: mostly healthy, slight warm-up on at-risk
        const st = (p - 0.25) / 0.2;
        if (die.risk > 0.7) {
          cOut.copy(COL.healthy).lerp(COL.amber, st * 0.3);
        } else {
          cOut.copy(COL.healthy);
        }
      } else if (p < 0.65) {
        // PREDICT: risk becomes visible
        const st = (p - 0.45) / 0.2;
        if (die.risk > 0.7) {
          cOut.copy(COL.amber).lerp(COL.red, st);
        } else if (die.risk > 0.35) {
          cOut.copy(COL.healthy).lerp(COL.amber, st);
        } else {
          cOut.copy(COL.healthy);
        }
      } else if (p < 0.8) {
        // EXPLAIN: full risk visible
        if (die.risk > 0.7) cOut.copy(COL.red);
        else if (die.risk > 0.35) cOut.copy(COL.amber);
        else cOut.copy(COL.healthy);
      } else {
        // OPTIMIZE: recovering to healthy
        const st = (p - 0.8) / 0.2;
        if (die.risk > 0.7) {
          cOut.copy(COL.red).lerp(COL.green, st);
        } else if (die.risk > 0.35) {
          cOut.copy(COL.amber).lerp(COL.green, st);
        } else {
          cOut.copy(COL.healthy).lerp(COL.greenDark, st * 0.5);
        }
      }

      dieRef.current.setColorAt(i, cOut);
    });

    dieRef.current.instanceMatrix.needsUpdate = true;
    if (dieRef.current.instanceColor) dieRef.current.instanceColor.needsUpdate = true;
  });

  return (
    <group ref={groupRef} rotation={[-0.45, 0, 0.05]} position={[0, -0.3, 0]}>
      {/* Wafer body */}
      <mesh castShadow receiveShadow>
        <cylinderGeometry args={[WAFER_R, WAFER_R, WAFER_H, 64]} />
        <meshPhysicalMaterial
          color="#18182a"
          metalness={0.65}
          roughness={0.28}
          clearcoat={0.25}
          clearcoatRoughness={0.5}
        />
      </mesh>

      {/* Wafer bevel / edge ring */}
      <mesh position={[0, 0, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[WAFER_R, 0.04, 12, 64]} />
        <meshStandardMaterial color="#2a2a3d" metalness={0.85} roughness={0.15} />
      </mesh>

      {/* Dies */}
      <instancedMesh ref={dieRef} args={[null, null, dies.length]} castShadow>
        <boxGeometry args={[DIE_SIZE, DIE_H, DIE_SIZE]} />
        <meshStandardMaterial
          color="#ffffff"
          metalness={0.35}
          roughness={0.65}
        />
      </instancedMesh>
    </group>
  );
}

/* ─── Exported Canvas ─── */
export default function Hero3DScene({ progress = 0 }) {
  return (
    <Canvas
      gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
      shadows
      style={{ width: '100%', height: '100%' }}
    >
      <PerspectiveCamera makeDefault position={[4.5, 3.5, 4.5]} fov={35} />

      {/* Studio lighting */}
      <ambientLight intensity={0.08} />
      <directionalLight
        position={[6, 10, 4]}
        intensity={1.8}
        color="#ffffff"
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <pointLight position={[-5, 3, -4]} intensity={0.35} color="#4488cc" />
      <pointLight position={[3, 1, -5]}  intensity={0.2}  color="#6699ff" />
      <spotLight
        position={[0, 8, 0]}
        intensity={0.6}
        angle={0.35}
        penumbra={0.7}
        color="#ffffff"
      />

      <SemiconductorWafer progress={progress} />
    </Canvas>
  );
}