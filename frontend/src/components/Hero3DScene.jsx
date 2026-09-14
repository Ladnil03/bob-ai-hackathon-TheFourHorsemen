import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

function WaferGrid() {
  const meshRef = useRef();
  
  const GRID_SIZE = 36;
  const SPACING = 0.45;
  const RADIUS = 7.5;
  
  const { positions, totalInstances } = useMemo(() => {
    const pos = [];
    const center = GRID_SIZE * SPACING / 2;
    
    for (let x = 0; x < GRID_SIZE; x++) {
      for (let y = 0; y < GRID_SIZE; y++) {
        const px = x * SPACING - center;
        const py = y * SPACING - center;
        if (Math.sqrt(px * px + py * py) <= RADIUS) {
          pos.push(new THREE.Vector3(px, py, 0));
        }
      }
    }
    return { positions: pos, totalInstances: pos.length };
  }, []);

  const dummy = useMemo(() => new THREE.Object3D(), []);
  const colorObj = useMemo(() => new THREE.Color(), []);

  useFrame((state) => {
    if (!meshRef.current) return;
    const time = state.clock.getElapsedTime();
    
    positions.forEach((pos, i) => {
      const wave = Math.sin(pos.x * 0.4 + time * 0.6) * Math.cos(pos.y * 0.4 + time * 0.4);
      const zOffset = wave * 0.15;
      
      dummy.position.set(pos.x, pos.y, zOffset);
      const scale = 0.85 + wave * 0.15;
      dummy.scale.set(scale, scale, 0.6);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
      
      // Electric blue glow: brighter chips on wave peaks
      const intensity = (wave + 1) / 2; // 0..1
      colorObj.setRGB(
        0.02 + 0.08 * intensity,  // very subtle red
        0.06 + 0.25 * intensity,  // blue-ish green channel
        0.15 + 0.85 * intensity   // strong blue
      );
      meshRef.current.setColorAt(i, colorObj);
    });
    
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
    
    // Slow rotation
    meshRef.current.rotation.z -= 0.0008;
    meshRef.current.rotation.x = -Math.PI / 3.5;
    meshRef.current.position.y = -2;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, totalInstances]}>
      <boxGeometry args={[0.35, 0.35, 0.12]} />
      <meshStandardMaterial 
        color="#0a1628" 
        emissive="#2563eb"
        emissiveIntensity={0.6}
        roughness={0.15} 
        metalness={0.9}
        toneMapped={false}
      />
    </instancedMesh>
  );
}

export default function Hero3DScene() {
  return (
    <Canvas
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
      gl={{ antialias: true, alpha: true }}
    >
      <PerspectiveCamera makeDefault position={[0, 3, 14]} />
      <ambientLight intensity={0.15} />
      <directionalLight position={[5, 10, 5]} intensity={1.5} color="#ffffff" />
      <pointLight position={[-3, 2, 6]} intensity={4} color="#3b82f6" />
      <pointLight position={[3, -2, 4]} intensity={2} color="#60a5fa" />
      <WaferGrid />
    </Canvas>
  );
}
