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

function getTerrainY(x: number, z: number): number {
  const r = Math.sqrt(x * x + z * z);
  const flatness = THREE.MathUtils.smoothstep(r, 22, 50);
  return wave(x, z) * flatness;
}

function Terrain() {
  const geo = useMemo(() => {
    const g = new THREE.PlaneGeometry(400, 400, 80, 80);
    const pos = g.attributes.position as THREE.BufferAttribute;
    const colors: number[] = [];
    const grassColor = new THREE.Color('#4a7c59');
    const dirtColor = new THREE.Color('#7a6040');
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i);
      const r = Math.sqrt(x * x + y * y);
      const flatness = THREE.MathUtils.smoothstep(r, 22, 50);
      pos.setZ(i, wave(x, y) * flatness);
      // Blend dirt near center, grass further out
      const t = THREE.MathUtils.smoothstep(r, 6, 28);
      const c = dirtColor.clone().lerp(grassColor, t);
      colors.push(c.r, c.g, c.b);
    }
    g.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    g.rotateX(-Math.PI / 2);
    g.computeVertexNormals();
    return g;
  }, []);

  return (
    <mesh geometry={geo} receiveShadow>
      <meshStandardMaterial vertexColors roughness={0.95} metalness={0} />
    </mesh>
  );
}

// ─── Atmospheric fog ───────────────────────────────────────────────────────

function SceneFog() {
  const { scene } = useThree();
  useEffect(() => {
    scene.fog = new THREE.FogExp2('#b8d4f0', 0.008);
    return () => { scene.fog = null; };
  }, [scene]);
  return null;
}

// ─── Stone path at town center ─────────────────────────────────────────────

function TownPlaza() {
  return (
    <group>
      {/* Cobblestone circle */}
      <mesh position={[0, 0.025, 0]} receiveShadow rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[13, 32]} />
        <meshStandardMaterial color="#7a7060" roughness={1} />
      </mesh>
      {/* Paths radiating outward */}
      {[0, Math.PI / 2, Math.PI, Math.PI * 1.5].map((a, i) => (
        <mesh key={i} position={[Math.cos(a) * 18, 0.02, Math.sin(a) * 18]}
          rotation={[-Math.PI / 2, 0, a]} receiveShadow>
          <planeGeometry args={[3, 12]} />
          <meshStandardMaterial color="#7a7060" roughness={1} />
        </mesh>
      ))}
    </group>
  );
}

// ─── Lamp posts ────────────────────────────────────────────────────────────

function LampPost({ pos }: { pos: [number, number, number] }) {
  return (
    <group position={pos}>
      <mesh position={[0, 2.1, 0]} castShadow>
        <cylinderGeometry args={[0.05, 0.08, 4.2, 6]} />
        <meshLambertMaterial color="#3a3a3a" />
      </mesh>
      {/* Arm */}
      <mesh position={[0.4, 4.2, 0]} rotation={[0, 0, Math.PI / 6]}>
        <cylinderGeometry args={[0.04, 0.04, 0.9, 5]} />
        <meshLambertMaterial color="#3a3a3a" />
      </mesh>
      {/* Globe */}
      <mesh position={[0.72, 4.45, 0]}>
        <sphereGeometry args={[0.22, 8, 8]} />
        <meshBasicMaterial color="#fff8c0" />
      </mesh>
      <pointLight position={[0.72, 4.45, 0]} intensity={1.5} distance={11} color="#ffd060" decay={2} />
    </group>
  );
}

const LAMP_POSITIONS: [number, number, number][] = [
  [9, 0, 0], [-9, 0, 0], [0, 0, 9], [0, 0, -9],
  [6.5, 0, 6.5], [-6.5, 0, 6.5], [6.5, 0, -6.5], [-6.5, 0, -6.5],
];

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
  const frontZ = def.d / 2 + 0.02;
  const winY = def.h * 0.62;
  const winW = Math.min(0.9, def.w * 0.26);
  const winH = def.h * 0.26;
  const doorW = def.w * 0.22;
  const doorH = def.h * 0.40;
  return (
    <group position={def.pos}>
      {/* Body */}
      <mesh position={[0, def.h / 2, 0]} castShadow receiveShadow>
        <boxGeometry args={[def.w, def.h, def.d]} />
        <meshStandardMaterial color={BUILDING_COLORS[def.type]} roughness={0.9} metalness={0} />
      </mesh>
      {/* Windows */}
      <mesh position={[def.w * 0.28, winY, frontZ]}>
        <planeGeometry args={[winW, winH]} />
        <meshBasicMaterial color="#9cc8e8" />
      </mesh>
      <mesh position={[-def.w * 0.28, winY, frontZ]}>
        <planeGeometry args={[winW, winH]} />
        <meshBasicMaterial color="#9cc8e8" />
      </mesh>
      {/* Window frames */}
      <mesh position={[def.w * 0.28, winY, frontZ + 0.01]}>
        <planeGeometry args={[winW + 0.1, winH + 0.1]} />
        <meshBasicMaterial color="#3a2010" wireframe />
      </mesh>
      <mesh position={[-def.w * 0.28, winY, frontZ + 0.01]}>
        <planeGeometry args={[winW + 0.1, winH + 0.1]} />
        <meshBasicMaterial color="#3a2010" wireframe />
      </mesh>
      {/* Door */}
      <mesh position={[0, doorH / 2, frontZ]}>
        <planeGeometry args={[doorW, doorH]} />
        <meshBasicMaterial color="#2a1508" />
      </mesh>
      {/* Door arch */}
      <mesh position={[0, doorH, frontZ]}>
        <torusGeometry args={[doorW / 2, 0.07, 5, 8, Math.PI]} />
        <meshLambertMaterial color="#2a1508" />
      </mesh>
      {/* Chimney */}
      <mesh position={[def.w * 0.3, def.h * 1.1, def.d * 0.25]} castShadow>
        <boxGeometry args={[0.4, def.h * 0.35, 0.4]} />
        <meshStandardMaterial color="#5a4030" roughness={1} />
      </mesh>
      {/* Roof */}
      <mesh position={[0, def.h + def.h * 0.18, 0]} rotation={[0, Math.PI / 4, 0]} castShadow>
        <coneGeometry args={[Math.max(def.w, def.d) * 0.75, def.h * 0.45, 4]} />
        <meshStandardMaterial color={ROOF_COLORS[def.type]} roughness={0.75} />
      </mesh>
    </group>
  );
}

// ─── Trees ─────────────────────────────────────────────────────────────────

function Tree({ pos, seed }: { pos: [number, number, number]; seed: string }) {
  const h = 4.0 + frac(seed + 'h') * 3.5;
  const r = 1.1 + frac(seed + 'r') * 0.9;
  const leafColor = frac(seed + 'lc') > 0.4 ? '#28542a' : '#3a6b30';
  const leafColor2 = frac(seed + 'lc2') > 0.5 ? '#1e4020' : '#2e5a22';
  return (
    <group position={pos}>
      {/* Trunk */}
      <mesh position={[0, h * 0.28, 0]} castShadow>
        <cylinderGeometry args={[0.13, 0.22, h * 0.56, 6]} />
        <meshLambertMaterial color="#4a2e10" />
      </mesh>
      {/* Bottom foliage cone */}
      <mesh position={[0, h * 0.62, 0]} castShadow>
        <coneGeometry args={[r, h * 0.62, 7]} />
        <meshLambertMaterial color={leafColor} />
      </mesh>
      {/* Mid cone */}
      <mesh position={[0, h * 0.80, 0]} castShadow>
        <coneGeometry args={[r * 0.72, h * 0.48, 7]} />
        <meshLambertMaterial color={leafColor2} />
      </mesh>
      {/* Top cone */}
      <mesh position={[0, h * 0.94, 0]} castShadow>
        <coneGeometry args={[r * 0.42, h * 0.32, 7]} />
        <meshLambertMaterial color={leafColor} />
      </mesh>
    </group>
  );
}

function Trees() {
  const trees = useMemo(() => {
    const list: Array<{ pos: [number, number, number]; seed: string }> = [];
    for (let i = 0; i < 75; i++) {
      const angle = frac(`ta${i}`) * Math.PI * 2;
      const radius = 48 + frac(`tr${i}`) * 130;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = getTerrainY(x, z);
      list.push({ pos: [x, y, z], seed: `tree${i}` });
    }
    // Small grove near settlement edge
    for (let i = 0; i < 18; i++) {
      const angle = frac(`tg${i}`) * Math.PI * 2;
      const radius = 36 + frac(`tgr${i}`) * 10;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = getTerrainY(x, z);
      list.push({ pos: [x, y, z], seed: `grove${i}` });
    }
    return list;
  }, []);

  return (
    <>
      {trees.map(({ pos, seed }) => <Tree key={seed} pos={pos} seed={seed} />)}
    </>
  );
}

// ─── NPC figure ────────────────────────────────────────────────────────────

const PEOPLE_HUE: Record<string, string> = {
  valari: '#7a6ab0', threnosi: '#6a8f80', aurath: '#b07840', default: '#906840',
};

const CLOAK_HUE: Record<string, string> = {
  valari: '#5a4a90', threnosi: '#4a7060', aurath: '#904a20', default: '#705030',
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
  const skinColor = PEOPLE_HUE[npc.people?.toLowerCase()] ?? PEOPLE_HUE.default;
  const cloakColor = CLOAK_HUE[npc.people?.toLowerCase()] ?? CLOAK_HUE.default;

  useFrame((_, dt) => {
    const g = groupRef.current;
    if (!g) return;

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

    const pd = playerPos.current.distanceTo(g.position);
    const near = pd < 14;
    if (near !== showRef.current) { showRef.current = near; setShowLabel(near); }
  });

  return (
    <group ref={groupRef} position={base}>
      {/* Cloak / body */}
      <mesh position={[0, 0.68, 0]} castShadow>
        <capsuleGeometry args={[0.30, 0.78, 4, 8]} />
        <meshStandardMaterial color={cloakColor} roughness={0.9} />
      </mesh>
      {/* Inner body */}
      <mesh position={[0, 0.65, 0]} castShadow>
        <capsuleGeometry args={[0.22, 0.65, 4, 8]} />
        <meshStandardMaterial color={skinColor} roughness={0.85} />
      </mesh>
      {/* Head */}
      <mesh position={[0, 1.52, 0]} castShadow>
        <sphereGeometry args={[0.24, 8, 8]} />
        <meshStandardMaterial color="#f0c8a0" roughness={0.7} />
      </mesh>
      {/* Hat */}
      <mesh position={[0, 1.72, 0]}>
        <coneGeometry args={[0.28, 0.3, 6]} />
        <meshLambertMaterial color={cloakColor} />
      </mesh>
      {/* Click target */}
      <mesh position={[0, 0.9, 0]} onClick={(e) => { e.stopPropagation(); onSelect(npc); }}>
        <capsuleGeometry args={[0.5, 1.2, 4, 8]} />
        <meshBasicMaterial visible={false} />
      </mesh>
      {showLabel && (
        <Html position={[0, 2.2, 0]} center distanceFactor={10}>
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
      {/* Stone base */}
      <mesh position={[0, 0.5, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[1.4, 1.5, 1, 14]} />
        <meshStandardMaterial color="#808080" roughness={0.95} />
      </mesh>
      {/* Water surface */}
      <mesh position={[0, 0.98, 0]}>
        <cylinderGeometry args={[1.25, 1.25, 0.06, 14]} />
        <meshStandardMaterial color="#3870a0" roughness={0.1} metalness={0.3} />
      </mesh>
      {/* Wooden crossbeam */}
      <mesh position={[0, 1.55, 0]} castShadow>
        <boxGeometry args={[3.2, 0.16, 0.16]} />
        <meshLambertMaterial color="#5c3a1e" />
      </mesh>
      {/* Support posts */}
      {[-1.5, 1.5].map((x, i) => (
        <mesh key={i} position={[x, 1.25, 0]} castShadow>
          <cylinderGeometry args={[0.1, 0.1, 1.6, 6]} />
          <meshLambertMaterial color="#5c3a1e" />
        </mesh>
      ))}
      {/* Rope & bucket */}
      <mesh position={[0, 1.35, 0]}>
        <cylinderGeometry args={[0.04, 0.04, 0.6, 5]} />
        <meshLambertMaterial color="#a89060" />
      </mesh>
      <mesh position={[0, 0.95, 0]}>
        <cylinderGeometry args={[0.2, 0.15, 0.35, 6]} />
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

  useFrame(() => { if (isMobile) playerPos.current.copy(camera.position); });

  return (
    <>
      <SceneFog />
      <Sky sunPosition={[100, 20, -60]} turbidity={3} rayleigh={1.2} mieCoefficient={0.004} mieDirectionalG={0.8} />
      <hemisphereLight args={['#b0d0f8', '#4a5c30', 0.7]} />
      <ambientLight intensity={0.25} />
      <directionalLight
        position={[80, 80, 40]}
        intensity={1.4}
        castShadow={!isMobile}
        shadow-mapSize={[isMobile ? 512 : 2048, isMobile ? 512 : 2048]}
        shadow-camera-far={200}
        shadow-camera-left={-80}
        shadow-camera-right={80}
        shadow-camera-top={80}
        shadow-camera-bottom={-80}
        shadow-bias={-0.0005}
      />

      <Terrain />
      <TownPlaza />
      <TownWell />
      <Trees />

      {LAMP_POSITIONS.map((pos, i) => <LampPost key={i} pos={pos} />)}
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
