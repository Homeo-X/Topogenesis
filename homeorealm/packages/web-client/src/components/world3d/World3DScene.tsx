import { useRef, useEffect, useState, useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Sky, PointerLockControls, Html, OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import type { EnvironmentState, NPCSummary, Settlement } from '../../api.js';

// ─── Types ─────────────────────────────────────────────────────────────────

interface SceneProps {
  npcs: NPCSummary[];
  settlement: Settlement | null;
  environment?: EnvironmentState | null;
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

type Vec3 = [number, number, number];
type Collider = { x: number; z: number; radius: number };

const MEDIEVAL_ASSET_BASE = '/assets/medieval-fair';
const MEDIEVAL_MODELS = {
  barrel: `${MEDIEVAL_ASSET_BASE}/Barrel.glb`,
  boothFood: `${MEDIEVAL_ASSET_BASE}/Booth_Food01.glb`,
  cart: `${MEDIEVAL_ASSET_BASE}/Cart.glb`,
  entranceBoard: `${MEDIEVAL_ASSET_BASE}/EntranceBoard.glb`,
  flagsLine: `${MEDIEVAL_ASSET_BASE}/Fair_Flags_Line.glb`,
  lamp: `${MEDIEVAL_ASSET_BASE}/Lamp.glb`,
};

const FALLBACK_ENVIRONMENT: EnvironmentState = {
  regionId: 'crown_valley',
  season: 'spring',
  weather: 'mist',
  physics: {
    airTemperatureC: 16,
    humidity: 0.62,
    rainfallMm: 0,
    windSpeedMps: 2.4,
    windDirectionDeg: 80,
    airPressureKpa: 101.1,
    soilMoisture: 0.55,
    groundwater: 0.55,
    evaporationRate: 0.15,
    turbulence: 0.25,
  },
  chemistry: {
    oxygenRatio: 0.209,
    carbonDioxidePpm: 420,
    soilPH: 6.8,
    mineralSaturation: 0.42,
    dissolvedIron: 0.28,
    organicMatter: 0.52,
    fermentation: 0.12,
    corrosion: 0.16,
  },
  signals: [],
};

function envOrFallback(environment?: EnvironmentState | null): EnvironmentState {
  return environment ?? FALLBACK_ENVIRONMENT;
}

function windVector(environment: EnvironmentState): THREE.Vector2 {
  const a = THREE.MathUtils.degToRad(environment.physics.windDirectionDeg);
  return new THREE.Vector2(Math.cos(a), Math.sin(a));
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

function SceneFog({ environment }: { environment: EnvironmentState }) {
  const { scene } = useThree();
  useEffect(() => {
    const humidityDensity = environment.physics.humidity * 0.008;
    const rainDensity = environment.physics.rainfallMm > 2 ? Math.min(0.011, environment.physics.rainfallMm * 0.00045) : 0;
    const chemistryHaze = environment.chemistry.carbonDioxidePpm > 520 ? 0.002 : 0;
    const density = 0.004 + humidityDensity + rainDensity + chemistryHaze;
    const color = environment.weather === 'dry_heat'
      ? '#d8c7a5'
      : environment.weather === 'storm' || environment.weather === 'rain'
        ? '#9fb1bd'
        : '#b8d4f0';
    scene.fog = new THREE.FogExp2(color, density);
    return () => { scene.fog = null; };
  }, [environment, scene]);
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

function createColliders(buildings: BuildingDef[]): Collider[] {
  const buildingColliders = buildings.map((building) => ({
    x: building.pos[0],
    z: building.pos[2],
    radius: Math.sqrt(building.w * building.w + building.d * building.d) * 0.5 + 0.65,
  }));
  return [
    ...buildingColliders,
    { x: 0, z: 0, radius: 2.3 },
    { x: 18, z: 6, radius: 2.7 },
    { x: 12.5, z: 17.2, radius: 2.0 },
    { x: 9.5, z: 17.5, radius: 1.7 },
    { x: -6.8, z: 12.7, radius: 1.6 },
    { x: -31, z: -24, radius: 3.1 },
  ];
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

function Tree({ pos, seed, environment }: { pos: [number, number, number]; seed: string; environment: EnvironmentState }) {
  const groupRef = useRef<THREE.Group>(null);
  const h = 4.0 + frac(seed + 'h') * 3.5;
  const r = 1.1 + frac(seed + 'r') * 0.9;
  const leafColor = frac(seed + 'lc') > 0.4 ? '#28542a' : '#3a6b30';
  const leafColor2 = frac(seed + 'lc2') > 0.5 ? '#1e4020' : '#2e5a22';
  const wind = environment.physics.windSpeedMps;
  const turbulence = environment.physics.turbulence;

  useFrame(({ clock }) => {
    const group = groupRef.current;
    if (!group) return;
    const phase = frac(seed + 'phase') * Math.PI * 2;
    const sway = Math.min(0.12, wind * 0.012 + turbulence * 0.035);
    group.rotation.z = Math.sin(clock.elapsedTime * (0.9 + turbulence) + phase) * sway;
    group.rotation.x = Math.cos(clock.elapsedTime * 0.65 + phase) * sway * 0.45;
  });

  return (
    <group ref={groupRef} position={pos}>
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

function Trees({ environment }: { environment: EnvironmentState }) {
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
      {trees.map(({ pos, seed }) => <Tree key={seed} pos={pos} seed={seed} environment={environment} />)}
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

function resolveCollision(position: THREE.Vector3, colliders: Collider[], playerRadius: number): THREE.Vector3 {
  const resolved = position.clone();
  for (const collider of colliders) {
    const dx = resolved.x - collider.x;
    const dz = resolved.z - collider.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    const minDist = collider.radius + playerRadius;
    if (dist > 0.0001 && dist < minDist) {
      const push = minDist - dist;
      resolved.x += (dx / dist) * push;
      resolved.z += (dz / dist) * push;
    }
  }
  return resolved;
}

function PlayerControls({ npcs, playerPos, colliders, environment, onLockChange, onSelectNPC, onNearbyNPCs }: {
  npcs: NPCSummary[];
  playerPos: React.MutableRefObject<THREE.Vector3>;
  colliders: Collider[];
  environment: EnvironmentState;
  onLockChange: (l: boolean) => void;
  onSelectNPC: (n: NPCSummary | null) => void;
  onNearbyNPCs: (n: NPCSummary[]) => void;
}) {
  const ref = useRef<any>(null);
  const keys = useRef(new Set<string>());
  const velocity = useRef(new THREE.Vector3());
  const { camera } = useThree();
  const hudClock = useRef(0);

  useEffect(() => {
    camera.position.set(0, 1.7, 12);
    camera.lookAt(0, 1.2, 0);
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
    if (!ctrl?.isLocked) {
      velocity.current.multiplyScalar(0.8);
      return;
    }

    const fwd = new THREE.Vector3();
    camera.getWorldDirection(fwd);
    fwd.y = 0; fwd.normalize();
    const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0));
    const input = new THREE.Vector3();

    if (keys.current.has('KeyW') || keys.current.has('ArrowUp')) input.add(fwd);
    if (keys.current.has('KeyS') || keys.current.has('ArrowDown')) input.addScaledVector(fwd, -1);
    if (keys.current.has('KeyA') || keys.current.has('ArrowLeft')) input.addScaledVector(right, -1);
    if (keys.current.has('KeyD') || keys.current.has('ArrowRight')) input.add(right);

    const wetGroundSlowdown = environment.physics.soilMoisture > 0.65 ? 1.2 : 0;
    const windDrag = environment.physics.windSpeedMps > 7 ? 0.6 : 0;
    const maxSpeed = 7.2 - wetGroundSlowdown - windDrag;
    const desired = input.lengthSq() > 0 ? input.normalize().multiplyScalar(maxSpeed) : new THREE.Vector3();
    velocity.current.lerp(desired, 1 - Math.exp(-dt * 10));
    if (desired.lengthSq() === 0) velocity.current.multiplyScalar(Math.exp(-dt * 7));

    let next = camera.position.clone().addScaledVector(velocity.current, dt);
    next.x = THREE.MathUtils.clamp(next.x, -185, 185);
    next.z = THREE.MathUtils.clamp(next.z, -185, 185);
    next = resolveCollision(next, colliders, 0.55);
    next.y = 1.7 + getTerrainY(next.x, next.z);
    camera.position.copy(next);

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

function GLBProp({
  url,
  position,
  rotation = [0, 0, 0],
  scale = 1,
}: {
  url: string;
  position: Vec3;
  rotation?: Vec3;
  scale?: number | Vec3;
}) {
  const gltf = useGLTF(url) as { scene: THREE.Group };
  const clone = useMemo(() => {
    const scene = gltf.scene.clone(true);
    scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if ((mesh as any).isMesh) {
        mesh.castShadow = true;
        mesh.receiveShadow = true;
      }
    });
    return scene;
  }, [gltf.scene]);

  return <primitive object={clone} position={position} rotation={rotation} scale={scale} />;
}

const FESTIVAL_MODEL_PLACEMENTS: Array<{
  key: string;
  url: string;
  position: Vec3;
  rotation?: Vec3;
  scale?: number | Vec3;
}> = [
  { key: 'food-booth', url: MEDIEVAL_MODELS.boothFood, position: [18, 0, 6], rotation: [0, -Math.PI / 2, 0], scale: 0.78 },
  { key: 'cart', url: MEDIEVAL_MODELS.cart, position: [12.5, 0, 17.2], rotation: [0, Math.PI * 0.8, 0], scale: 0.86 },
  { key: 'entry-board', url: MEDIEVAL_MODELS.entranceBoard, position: [9.5, 0, 17.5], rotation: [0, -Math.PI * 0.72, 0], scale: 0.62 },
  { key: 'flags', url: MEDIEVAL_MODELS.flagsLine, position: [-14, 0, 17], rotation: [0, Math.PI * 0.08, 0], scale: 0.62 },
  { key: 'lamp-market', url: MEDIEVAL_MODELS.lamp, position: [15.5, 0, 11.4], rotation: [0, -Math.PI / 3, 0], scale: 1 },
  { key: 'barrel-a', url: MEDIEVAL_MODELS.barrel, position: [15.4, 0, 5.5], rotation: [0, Math.PI / 8, 0], scale: 0.82 },
  { key: 'barrel-b', url: MEDIEVAL_MODELS.barrel, position: [16.3, 0, 4.8], rotation: [0, Math.PI / 5, 0], scale: 0.7 },
];

function HearthwellShrine() {
  const glowRef = useRef<THREE.Group>(null);

  useFrame(({ clock }) => {
    const glow = glowRef.current;
    if (!glow) return;
    const pulse = 1 + Math.sin(clock.elapsedTime * 2.4) * 0.06;
    glow.rotation.y = clock.elapsedTime * 0.22;
    glow.scale.setScalar(pulse);
  });

  return (
    <group>
      {Array.from({ length: 14 }, (_, i) => {
        const a = (i / 14) * Math.PI * 2;
        const r = 2.35 + (i % 2) * 0.18;
        return (
          <mesh
            key={`hearth-stone-${i}`}
            position={[Math.cos(a) * r, 0.18, Math.sin(a) * r]}
            rotation={[0, -a, 0]}
            castShadow
            receiveShadow
          >
            <boxGeometry args={[0.46, 0.34, 0.9]} />
            <meshStandardMaterial color={i % 3 === 0 ? '#7d7466' : '#5f6358'} roughness={0.96} />
          </mesh>
        );
      })}

      <group ref={glowRef}>
        <mesh position={[0, 1.85, 0]} castShadow>
          <octahedronGeometry args={[0.62, 0]} />
          <meshStandardMaterial color="#ffe2a0" emissive="#d58a2a" emissiveIntensity={1.7} roughness={0.35} />
        </mesh>
        <mesh position={[0, 1.85, 0]}>
          <sphereGeometry args={[1.25, 18, 12]} />
          <meshBasicMaterial color="#ffcf74" transparent opacity={0.13} depthWrite={false} />
        </mesh>
        {Array.from({ length: 18 }, (_, i) => {
          const a = (i / 18) * Math.PI * 2;
          const r = 0.85 + frac(`mote-${i}`) * 0.95;
          return (
            <mesh key={`hearth-mote-${i}`} position={[Math.cos(a) * r, 1.05 + frac(`my-${i}`) * 1.35, Math.sin(a) * r]}>
              <sphereGeometry args={[0.045, 6, 6]} />
              <meshBasicMaterial color="#ffd36e" transparent opacity={0.72} />
            </mesh>
          );
        })}
      </group>
      <pointLight position={[0, 2.1, 0]} intensity={2.1} distance={17} color="#ffc76b" decay={2} />
    </group>
  );
}

function MarketGoods({ position, seed }: { position: Vec3; seed: string }) {
  const colors = ['#caa85b', '#b65f42', '#6f9c62', '#9f7448'];
  return (
    <group position={position}>
      {Array.from({ length: 7 }, (_, i) => (
        <mesh key={`${seed}-goods-${i}`} position={[(frac(`${seed}-x-${i}`) - 0.5) * 1.2, 0.18, (frac(`${seed}-z-${i}`) - 0.5) * 0.8]} castShadow>
          {i % 2 === 0 ? <boxGeometry args={[0.28, 0.22, 0.28]} /> : <sphereGeometry args={[0.16, 8, 6]} />}
          <meshStandardMaterial color={colors[i % colors.length]} roughness={0.9} />
        </mesh>
      ))}
    </group>
  );
}

function FestivalMarket() {
  return (
    <group>
      <mesh position={[9.8, 0.035, 10.4]} rotation={[-Math.PI / 2, 0, Math.PI / 4]} receiveShadow>
        <planeGeometry args={[7.5, 2.8]} />
        <meshStandardMaterial color="#9c6b46" roughness={0.92} />
      </mesh>
      {FESTIVAL_MODEL_PLACEMENTS.map((asset) => (
        <GLBProp
          key={asset.key}
          url={asset.url}
          position={asset.position}
          rotation={asset.rotation}
          scale={asset.scale}
        />
      ))}
      <MarketGoods position={[17.6, 0, 8.3]} seed="market-a" />
      <MarketGoods position={[11.4, 0, 15.3]} seed="market-b" />
      <pointLight position={[15.5, 3.4, 11.4]} intensity={1.1} distance={12} color="#ffd38a" decay={2} />
    </group>
  );
}

function NoticeBoard() {
  return (
    <group position={[-6.8, 0, 12.7]} rotation={[0, 0.34, 0]}>
      <mesh position={[-0.9, 0.9, 0]} castShadow>
        <cylinderGeometry args={[0.08, 0.1, 1.8, 6]} />
        <meshLambertMaterial color="#4d2f16" />
      </mesh>
      <mesh position={[0.9, 0.9, 0]} castShadow>
        <cylinderGeometry args={[0.08, 0.1, 1.8, 6]} />
        <meshLambertMaterial color="#4d2f16" />
      </mesh>
      <mesh position={[0, 1.35, 0]} castShadow receiveShadow>
        <boxGeometry args={[2.4, 1.15, 0.12]} />
        <meshStandardMaterial color="#6b4423" roughness={0.95} />
      </mesh>
      {[-0.62, 0, 0.58].map((x, i) => (
        <mesh key={`notice-paper-${i}`} position={[x, 1.38 - (i % 2) * 0.18, 0.07]}>
          <planeGeometry args={[0.5, 0.42]} />
          <meshBasicMaterial color={i === 1 ? '#ecd7a6' : '#d9c497'} />
        </mesh>
      ))}
      <mesh position={[0, 2.02, 0]} rotation={[0, 0, Math.PI / 4]} castShadow>
        <boxGeometry args={[1.9, 0.12, 0.16]} />
        <meshLambertMaterial color="#4d2f16" />
      </mesh>
    </group>
  );
}

function WheatClump({ pos, seed }: { pos: Vec3; seed: string }) {
  return (
    <group position={pos}>
      {Array.from({ length: 5 }, (_, i) => {
        const x = (frac(`${seed}-x-${i}`) - 0.5) * 0.45;
        const z = (frac(`${seed}-z-${i}`) - 0.5) * 0.45;
        const h = 0.62 + frac(`${seed}-h-${i}`) * 0.42;
        return (
          <group key={`${seed}-stem-${i}`} position={[x, 0, z]} rotation={[0, 0, (frac(`${seed}-tilt-${i}`) - 0.5) * 0.2]}>
            <mesh position={[0, h / 2, 0]} castShadow>
              <cylinderGeometry args={[0.012, 0.018, h, 5]} />
              <meshLambertMaterial color="#b49542" />
            </mesh>
            <mesh position={[0, h + 0.08, 0]} castShadow>
              <coneGeometry args={[0.07, 0.18, 5]} />
              <meshLambertMaterial color="#d0b65f" />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

function CropFields() {
  const clumps = useMemo(() => {
    const list: Array<{ pos: Vec3; seed: string }> = [];
    for (let row = 0; row < 6; row++) {
      for (let col = 0; col < 8; col++) {
        const x = -35 + col * 1.55 + frac(`field-x-${row}-${col}`) * 0.35;
        const z = 18 + row * 1.35 + frac(`field-z-${row}-${col}`) * 0.35;
        list.push({ pos: [x, getTerrainY(x, z), z], seed: `field-${row}-${col}` });
      }
    }
    return list;
  }, []);

  return (
    <group>
      <mesh position={[-29.6, 0.03, 22.4]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[15.5, 10]} />
        <meshStandardMaterial color="#6f6a39" roughness={1} />
      </mesh>
      {clumps.map((clump) => <WheatClump key={clump.seed} pos={clump.pos} seed={clump.seed} />)}
      <mesh position={[-20.8, 0.08, 23.2]} rotation={[-Math.PI / 2, 0, 0.2]} receiveShadow>
        <planeGeometry args={[2.4, 8]} />
        <meshStandardMaterial color="#836245" roughness={1} />
      </mesh>
    </group>
  );
}

function DungeonRupture() {
  const ringRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (!ringRef.current) return;
    ringRef.current.rotation.z = clock.elapsedTime * 0.25;
  });

  return (
    <group position={[-31, getTerrainY(-31, -24), -24]} rotation={[0, 0.58, 0]}>
      {[0, 1, 2, 3].map((i) => (
        <mesh key={`rupture-crack-${i}`} position={[0, 0.04, 0]} rotation={[0, (i / 4) * Math.PI + frac(`crack-${i}`) * 0.25, 0]} receiveShadow>
          <boxGeometry args={[0.22, 0.05, 3.8 + frac(`crack-l-${i}`) * 2.4]} />
          <meshStandardMaterial color="#120f19" emissive="#1b0c31" emissiveIntensity={0.6} roughness={1} />
        </mesh>
      ))}
      <mesh ref={ringRef} position={[0, 1.55, 0]}>
        <ringGeometry args={[0.88, 1.18, 28]} />
        <meshBasicMaterial color="#8d75ff" transparent opacity={0.68} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, 1.55, 0]}>
        <circleGeometry args={[0.78, 28]} />
        <meshBasicMaterial color="#100b1f" transparent opacity={0.78} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, 0.62, 0]} castShadow>
        <octahedronGeometry args={[0.36, 0]} />
        <meshStandardMaterial color="#d6d2ff" emissive="#6854d8" emissiveIntensity={1.4} roughness={0.45} />
      </mesh>
      <pointLight position={[0, 1.3, 0]} intensity={1.9} distance={14} color="#7d64ff" decay={2} />
    </group>
  );
}

function RainSystem({ environment }: { environment: EnvironmentState }) {
  const pointsRef = useRef<THREE.Points>(null);
  const rainIntensity = Math.min(1, environment.physics.rainfallMm / 18);
  const wind = useMemo(() => windVector(environment), [environment]);
  const geometry = useMemo(() => {
    const count = 900;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[i * 3] = (frac(`rain-x-${i}`) - 0.5) * 90;
      positions[i * 3 + 1] = 4 + frac(`rain-y-${i}`) * 28;
      positions[i * 3 + 2] = (frac(`rain-z-${i}`) - 0.5) * 90;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return g;
  }, []);

  useFrame((_, dt) => {
    if (rainIntensity <= 0.03 || !pointsRef.current) return;
    const attr = pointsRef.current.geometry.getAttribute('position') as THREE.BufferAttribute;
    const fall = (10 + rainIntensity * 18) * dt;
    const drift = environment.physics.windSpeedMps * 0.22 * dt;
    for (let i = 0; i < attr.count; i++) {
      let y = attr.getY(i) - fall;
      let x = attr.getX(i) + wind.x * drift;
      let z = attr.getZ(i) + wind.y * drift;
      if (y < 0.15) {
        y = 18 + frac(`rain-reset-y-${i}`) * 14;
        x = (frac(`rain-reset-x-${i}`) - 0.5) * 90;
        z = (frac(`rain-reset-z-${i}`) - 0.5) * 90;
      }
      attr.setXYZ(i, x, y, z);
    }
    attr.needsUpdate = true;
  });

  if (rainIntensity <= 0.03) return null;

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial color="#9fc3d8" size={0.055 + rainIntensity * 0.035} transparent opacity={0.25 + rainIntensity * 0.45} depthWrite={false} />
    </points>
  );
}

function Puddles({ environment }: { environment: EnvironmentState }) {
  const wetness = Math.max(environment.physics.soilMoisture, Math.min(1, environment.physics.rainfallMm / 24));
  if (wetness < 0.45) return null;

  const puddles = [
    [-4.8, 6.7, 1.5, 0.55],
    [4.2, -5.5, 1.1, 0.38],
    [10.5, 8.4, 0.9, 0.48],
    [-11.5, -3.4, 1.2, 0.3],
    [6.8, 16.2, 1.5, 0.42],
  ];

  return (
    <group>
      {puddles.map(([x, z, radius, squash], i) => (
        <mesh key={`puddle-${i}`} position={[x, 0.045, z]} rotation={[-Math.PI / 2, 0, frac(`puddle-r-${i}`) * Math.PI]} receiveShadow>
          <circleGeometry args={[radius, 18]} />
          <meshStandardMaterial
            color="#4f7890"
            roughness={0.08}
            metalness={0.1}
            transparent
            opacity={0.12 + wetness * 0.18 * squash}
          />
        </mesh>
      ))}
    </group>
  );
}

function ChemistryMotes({ environment }: { environment: EnvironmentState }) {
  const activity = Math.max(environment.chemistry.fermentation, environment.chemistry.corrosion) - 0.35;
  const groupRef = useRef<THREE.Group>(null);

  useFrame(({ clock }) => {
    if (!groupRef.current) return;
    groupRef.current.rotation.y = clock.elapsedTime * 0.08;
    groupRef.current.position.y = Math.sin(clock.elapsedTime * 0.7) * 0.05;
  });

  if (activity <= 0) return null;

  const color = environment.chemistry.corrosion > environment.chemistry.fermentation ? '#b06a3a' : '#8fcf8c';
  const opacity = Math.min(0.45, 0.12 + activity * 0.45);

  return (
    <group ref={groupRef}>
      {Array.from({ length: 42 }, (_, i) => {
        const angle = frac(`chem-a-${i}`) * Math.PI * 2;
        const radius = 3.5 + frac(`chem-r-${i}`) * 12;
        return (
          <mesh key={`chem-${i}`} position={[Math.cos(angle) * radius + 11, 0.8 + frac(`chem-y-${i}`) * 2.8, Math.sin(angle) * radius + 7]}>
            <sphereGeometry args={[0.035 + frac(`chem-s-${i}`) * 0.035, 6, 6]} />
            <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
          </mesh>
        );
      })}
    </group>
  );
}

function EnvironmentEffects({ environment }: { environment: EnvironmentState }) {
  return (
    <>
      <RainSystem environment={environment} />
      <Puddles environment={environment} />
      <ChemistryMotes environment={environment} />
    </>
  );
}

function SettlementDressing() {
  return (
    <>
      <FestivalMarket />
      <NoticeBoard />
      <CropFields />
      <DungeonRupture />
    </>
  );
}

export function World3DScene({ npcs, settlement, environment: environmentInput, isMobile, onLockChange, onSelectNPC, onNearbyNPCs }: SceneProps) {
  const playerPos = useRef(new THREE.Vector3(0, 1.7, 12));
  const { camera } = useThree();
  const environment = envOrFallback(environmentInput);
  const buildings = useMemo(() => generateBuildings(settlement?.population ?? 24), [settlement?.population]);
  const colliders = useMemo(() => createColliders(buildings), [buildings]);
  const skyTurbidity = 2.5 + environment.physics.humidity * 3 + Math.min(2, environment.physics.rainfallMm / 8);
  const sunHeight = environment.weather === 'storm' || environment.weather === 'rain' ? 8 : 20;

  useFrame(() => { if (isMobile) playerPos.current.copy(camera.position); });

  return (
    <>
      <SceneFog environment={environment} />
      <Sky sunPosition={[100, sunHeight, -60]} turbidity={skyTurbidity} rayleigh={1.2} mieCoefficient={0.004 + environment.physics.humidity * 0.006} mieDirectionalG={0.8} />
      <hemisphereLight args={['#b0d0f8', '#4a5c30', 0.7]} />
      <ambientLight intensity={environment.weather === 'storm' ? 0.18 : 0.25} />
      <directionalLight
        position={[80, 80, 40]}
        intensity={environment.weather === 'storm' ? 0.85 : 1.4}
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
      <HearthwellShrine />
      <SettlementDressing />
      <EnvironmentEffects environment={environment} />
      <Trees environment={environment} />

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
            colliders={colliders}
            environment={environment}
            onLockChange={onLockChange}
            onSelectNPC={onSelectNPC}
            onNearbyNPCs={onNearbyNPCs}
          />
      }
    </>
  );
}
