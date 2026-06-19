import { useRef, useEffect, useState, useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Sky, PointerLockControls, Html, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { NPCSummary, Settlement } from '../../api.js';

// ─── Types ─────────────────────────────────────────────────────────────────

interface SceneProps {
  npcs: NPCSummary[];
  settlement: Settlement | null;
  isMobile: boolean;
  onLockChange: (locked: boolean) => void;
  onSelectNPC: (npc: NPCSummary | null) => void;
  onNearbyNPCs: (npcs: NPCSummary[]) => void;
}

interface BuildingDef {
  pos: [number, number, number];
  w: number;
  d: number;
  h: number;
  type: 'inn' | 'smithy' | 'market' | 'house';
}

// ─── Deterministic helpers ─────────────────────────────────────────────────

function hash(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193);
  return h >>> 0;
}

function frac(s: string): number { return (hash(s) % 10000) / 10000; }

export function npcBasePos(id: string): [number, number, number] {
  const angle = frac(id) * Math.PI * 2;
  const radius = 6 + frac(id + '_r') * 16;
  return [Math.cos(angle) * radius, 0, Math.sin(angle) * radius];
}

// ─── Terrain ───────────────────────────────────────────────────────────────

function wave(x: number, z: number): number {
  return (
    Math.sin(x * 0.05) * Math.cos(z * 0.05) * 4 +
    Math.sin(x * 0.11 + 1.3) * Math.cos(z * 0.09 + 0.7) * 2.5 +
    Math.sin(x * 0.02 + 2.1) * Math.cos(z * 0.03 + 1.4) * 9
  );
}

function Terrain() {
  const geo = useMemo(() => {
    const g = new THREE.PlaneGeometry(400, 400, 80, 80);
    const pos = g.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i);
      const r = Math.sqrt(x * x + y * y);
      const flatness = THREE.MathUtils.smoothstep(r, 22, 50);
      pos.setZ(i, wave(x, y) * flatness);
    }
    g.rotateX(-Math.PI / 2);
    g.computeVertexNormals();
    return g;
  }, []);

  return (
    <mesh geometry={geo} receiveShadow>
      <meshLambertMaterial color="#4a7c59" />
    </mesh>
  );
}

// ─── Settlement buildings ──────────────────────────────────────────────────

function generateBuildings(population: number): BuildingDef[] {
  const count = Math.max(8, Math.min(32, Math.floor(population * 1.5)));
  const types: BuildingDef['type'][] = ['inn', 'smithy', 'market', 'house'];
  return Array.from({ length: count }, (_, i) => {
    const ring = Math.floor(i / 6);
    const angle = (i % 6) / 6 * Math.PI * 2 + ring * 0.52 + frac(`ba${i}`) * 0.3;
    const radius = 11 + ring * 9 + frac(`br${i}`) * 4;
    return {
      pos: [
        Math.cos(angle) * radius + (frac(`bx${i}`) - 0.5) * 2,
        0,
        Math.sin(angle) * radius + (frac(`bz${i}`) - 0.5) * 2,
      ],
      w: 3 + frac(`bw${i}`) * 3,
      d: 3 + frac(`bd${i}`) * 3,
      h: 2.5 + frac(`bh${i}`) * 4,
      type: types[i % 4],
    };
  });
}

const BUILDING_COLORS = { inn: '#9e8070', smithy: '#5a5a62', market: '#9e9060', house: '#9e8e78' };
const ROOF_COLORS = { inn: '#6b3a1e', smithy: '#3a3a42', market: '#7a6030', house: '#5c3a1e' };

function Building({ def }: { def: BuildingDef }) {
  return (
    <group position={def.pos}>
      <mesh position={[0, def.h / 2, 0]} castShadow receiveShadow>
        <boxGeometry args={[def.w, def.h, def.d]} />
        <meshLambertMaterial color={BUILDING_COLORS[def.type]} />
      </mesh>
      <mesh position={[0, def.h + def.h * 0.18, 0]} rotation={[0, Math.PI / 4, 0]} castShadow>
        <coneGeometry args={[Math.max(def.w, def.d) * 0.75, def.h * 0.45, 4]} />
        <meshLambertMaterial color={ROOF_COLORS[def.type]} />
      </mesh>
    </group>
  );
}

// ─── NPC figure ────────────────────────────────────────────────────────────

const PEOPLE_HUE: Record<string, string> = {
  valari: '#8b7bb5', threnosi: '#7b9b8b', aurath: '#bb8b5b', default: '#a87b5b',
};

function NPCFigure({ npc, playerPos, onSelect }: {
  npc: NPCSummary;
  playerPos: React.MutableRefObject<THREE.Vector3>;
  onSelect: (n: NPCSummary) => void;
}) {
  const groupRef = useRef<THREE.Group>(null!);
  const base = useMemo(() => npcBasePos(npc.id), [npc.id]);
  const target = useRef<[number, number]>([base[0], base[2]]);
  const timer = useRef(frac(npc.id + '_t') * 5);
  const [showLabel, setShowLabel] = useState(false);
  const showRef = useRef(false);
  const color = PEOPLE_HUE[npc.people?.toLowerCase()] ?? PEOPLE_HUE.default;

  useFrame((_, dt) => {
    const g = groupRef.current;
    if (!g) return;

    // Wander toward target
    timer.current -= dt;
    if (timer.current <= 0) {
      timer.current = 3 + Math.random() * 5;
      const a = Math.random() * Math.PI * 2;
      const r = 2 + Math.random() * 5;
      target.current = [base[0] + Math.cos(a) * r, base[2] + Math.sin(a) * r];
    }
    const dx = target.current[0] - g.position.x;
    const dz = target.current[1] - g.position.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    if (dist > 0.15) {
      const spd = 0.9 * dt;
      g.position.x += (dx / dist) * spd;
      g.position.z += (dz / dist) * spd;
      g.rotation.y = Math.atan2(dx, dz);
    }

    // Show label when player is nearby
    const pd = playerPos.current.distanceTo(g.position);
    const near = pd < 14;
    if (near !== showRef.current) { showRef.current = near; setShowLabel(near); }
  });

  return (
    <group ref={groupRef} position={base}>
      {/* Body */}
      <mesh position={[0, 0.65, 0]} castShadow>
        <capsuleGeometry args={[0.28, 0.7, 4, 8]} />
        <meshLambertMaterial color={color} />
      </mesh>
      {/* Head */}
      <mesh position={[0, 1.44, 0]} castShadow>
        <sphereGeometry args={[0.22, 8, 8]} />
        <meshLambertMaterial color="#f0c8a0" />
      </mesh>
      {/* Invisible click target (unlocked mode / mobile) */}
      <mesh position={[0, 0.9, 0]} onClick={(e) => { e.stopPropagation(); onSelect(npc); }}>
        <capsuleGeometry args={[0.5, 1.2, 4, 8]} />
        <meshBasicMaterial visible={false} />
      </mesh>
      {showLabel && (
        <Html position={[0, 2.15, 0]} center distanceFactor={10}>
          <div className="npc-label-3d" onClick={() => onSelect(npc)}>
            <span className="npc-label-name">{npc.name}</span>
            {npc.jobId && <span className="npc-label-job">{npc.jobId}</span>}
          </div>
        </Html>
      )}
    </group>
  );
}

// ─── Player controls (desktop first-person) ────────────────────────────────

function PlayerControls({ npcs, playerPos, onLockChange, onSelectNPC, onNearbyNPCs }: {
  npcs: NPCSummary[];
  playerPos: React.MutableRefObject<THREE.Vector3>;
  onLockChange: (l: boolean) => void;
  onSelectNPC: (n: NPCSummary | null) => void;
  onNearbyNPCs: (n: NPCSummary[]) => void;
}) {
  const ref = useRef<any>(null);
  const keys = useRef(new Set<string>());
  const { camera } = useThree();
  const hudClock = useRef(0);

  useEffect(() => {
    camera.position.set(0, 1.7, 30);
  }, [camera]);

  useEffect(() => {
    function onDown(e: KeyboardEvent) {
      keys.current.add(e.code);
      // E — talk to nearest NPC
      if (e.code === 'KeyE' && ref.current?.isLocked) {
        let best: NPCSummary | null = null;
        let bestDist = 5.5;
        for (const npc of npcs) {
          const bp = npcBasePos(npc.id);
          const d = camera.position.distanceTo(new THREE.Vector3(bp[0], 0, bp[2]));
          if (d < bestDist) { bestDist = d; best = npc; }
        }
        if (best) {
          ref.current?.unlock();
          onSelectNPC(best);
        }
      }
    }
    function onUp(e: KeyboardEvent) { keys.current.delete(e.code); }
    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup', onUp);
    return () => { window.removeEventListener('keydown', onDown); window.removeEventListener('keyup', onUp); };
  }, [camera, npcs, onSelectNPC]);

  useFrame((_, dt) => {
    const ctrl = ref.current;
    if (!ctrl?.isLocked) return;

    const speed = 9 * dt;
    const fwd = new THREE.Vector3();
    camera.getWorldDirection(fwd);
    fwd.y = 0; fwd.normalize();
    const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0));

    if (keys.current.has('KeyW') || keys.current.has('ArrowUp'))    camera.position.addScaledVector(fwd, speed);
    if (keys.current.has('KeyS') || keys.current.has('ArrowDown'))  camera.position.addScaledVector(fwd, -speed);
    if (keys.current.has('KeyA') || keys.current.has('ArrowLeft'))  camera.position.addScaledVector(right, -speed);
    if (keys.current.has('KeyD') || keys.current.has('ArrowRight')) camera.position.addScaledVector(right, speed);

    camera.position.x = THREE.MathUtils.clamp(camera.position.x, -185, 185);
    camera.position.z = THREE.MathUtils.clamp(camera.position.z, -185, 185);
    camera.position.y = 1.7;

    playerPos.current.copy(camera.position);

    // Update HUD nearby list at 2 Hz
    hudClock.current += dt;
    if (hudClock.current > 0.5) {
      hudClock.current = 0;
      const nearby = npcs.filter(npc => {
        const bp = npcBasePos(npc.id);
        return camera.position.distanceTo(new THREE.Vector3(bp[0], 0, bp[2])) < 6;
      });
      onNearbyNPCs(nearby);
    }
  });

  return (
    <PointerLockControls
      ref={ref}
      onLock={() => onLockChange(true)}
      onUnlock={() => onLockChange(false)}
    />
  );
}

// ─── Town well (center landmark) ──────────────────────────────────────────

function TownWell() {
  return (
    <group>
      <mesh position={[0, 0.5, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[1.4, 1.4, 1, 14]} />
        <meshLambertMaterial color="#888" />
      </mesh>
      <mesh position={[0, 1.25, 0]} castShadow>
        <boxGeometry args={[3, 0.15, 0.15]} />
        <meshLambertMaterial color="#5c3a1e" />
      </mesh>
    </group>
  );
}

// ─── Main exported scene ───────────────────────────────────────────────────

export function World3DScene({ npcs, settlement, isMobile, onLockChange, onSelectNPC, onNearbyNPCs }: SceneProps) {
  const playerPos = useRef(new THREE.Vector3(0, 1.7, 30));
  const { camera } = useThree();
  const buildings = useMemo(() => generateBuildings(settlement?.population ?? 24), [settlement?.population]);

  // Keep playerPos ref in sync for mobile (no PlayerControls)
  useFrame(() => { if (isMobile) playerPos.current.copy(camera.position); });

  return (
    <>
      <Sky sunPosition={[80, 30, -80]} turbidity={6} rayleigh={1.5} />
      <ambientLight intensity={0.55} />
      <directionalLight
        position={[80, 80, 40]}
        intensity={1.2}
        castShadow={!isMobile}
        shadow-mapSize={[isMobile ? 512 : 1024, isMobile ? 512 : 1024]}
        shadow-camera-far={200}
        shadow-camera-left={-80}
        shadow-camera-right={80}
        shadow-camera-top={80}
        shadow-camera-bottom={-80}
      />

      <Terrain />
      <TownWell />

      {buildings.map((b, i) => <Building key={i} def={b} />)}

      {npcs.map(npc => (
        <NPCFigure
          key={npc.id}
          npc={npc}
          playerPos={playerPos}
          onSelect={onSelectNPC}
        />
      ))}

      {isMobile
        ? <OrbitControls
            target={[0, 1, 0]}
            minDistance={5}
            maxDistance={100}
            maxPolarAngle={Math.PI / 2.05}
          />
        : <PlayerControls
            npcs={npcs}
            playerPos={playerPos}
            onLockChange={onLockChange}
            onSelectNPC={onSelectNPC}
            onNearbyNPCs={onNearbyNPCs}
          />
      }
    </>
  );
}
