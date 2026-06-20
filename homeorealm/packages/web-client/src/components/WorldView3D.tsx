import { Canvas } from '@react-three/fiber';
import { Suspense, useState, useEffect, useRef } from 'react';
import { api } from '../api.js';
import type { EnvironmentState, NPCSummary, Settlement } from '../api.js';
import { PlayerHUD } from './PlayerHUD.js';
import { World3DScene } from './world3d/World3DScene.js';

// ─── Virtual Joystick ──────────────────────────────────────────────────────

function VirtualJoystick({ axisRef }: { axisRef: { current: { x: number; y: number } } }) {
  const baseRef = useRef<HTMLDivElement>(null);
  const thumbRef = useRef<HTMLDivElement>(null);
  const activeId = useRef<number | null>(null);
  const center = useRef({ x: 0, y: 0 });
  const RADIUS = 48;

  useEffect(() => {
    const base = baseRef.current;
    if (!base) return;

    function move(cx: number, cy: number) {
      const dx = cx - center.current.x;
      const dy = cy - center.current.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const clamped = Math.min(dist, RADIUS);
      const nx = dist > 0 ? (dx / dist) * clamped : 0;
      const ny = dist > 0 ? (dy / dist) * clamped : 0;
      axisRef.current = { x: nx / RADIUS, y: ny / RADIUS };
      if (thumbRef.current) {
        thumbRef.current.style.transform = `translate(calc(-50% + ${nx}px), calc(-50% + ${ny}px))`;
      }
    }

    function onStart(e: TouchEvent) {
      if (activeId.current !== null) return;
      e.preventDefault();
      const t = e.changedTouches[0];
      activeId.current = t.identifier;
      const rect = base!.getBoundingClientRect();
      center.current = { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
      move(t.clientX, t.clientY);
    }

    function onMove(e: TouchEvent) {
      for (let i = 0; i < e.changedTouches.length; i++) {
        const t = e.changedTouches[i];
        if (t.identifier === activeId.current) { e.preventDefault(); move(t.clientX, t.clientY); }
      }
    }

    function onEnd(e: TouchEvent) {
      for (let i = 0; i < e.changedTouches.length; i++) {
        if (e.changedTouches[i].identifier === activeId.current) {
          activeId.current = null;
          axisRef.current = { x: 0, y: 0 };
          if (thumbRef.current) thumbRef.current.style.transform = 'translate(-50%,-50%)';
        }
      }
    }

    base.addEventListener('touchstart', onStart, { passive: false });
    window.addEventListener('touchmove', onMove, { passive: false });
    window.addEventListener('touchend', onEnd);
    return () => {
      base.removeEventListener('touchstart', onStart);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('touchend', onEnd);
    };
  }, [axisRef]);

  return (
    <div ref={baseRef} className="joystick-base">
      <div ref={thumbRef} className="joystick-thumb" />
    </div>
  );
}

// ─── Touch look zone (right half → camera yaw) ────────────────────────────

function TouchLookZone({ yawRef }: { yawRef: { current: number } }) {
  const ref = useRef<HTMLDivElement>(null);
  const activeId = useRef<number | null>(null);
  const lastX = useRef(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    function onStart(e: TouchEvent) {
      if (activeId.current !== null) return;
      e.preventDefault();
      const t = e.changedTouches[0];
      activeId.current = t.identifier;
      lastX.current = t.clientX;
    }

    function onMove(e: TouchEvent) {
      for (let i = 0; i < e.changedTouches.length; i++) {
        const t = e.changedTouches[i];
        if (t.identifier === activeId.current) {
          e.preventDefault();
          yawRef.current += (t.clientX - lastX.current) * 0.008;
          lastX.current = t.clientX;
        }
      }
    }

    function onEnd(e: TouchEvent) {
      for (let i = 0; i < e.changedTouches.length; i++) {
        if (e.changedTouches[i].identifier === activeId.current) activeId.current = null;
      }
    }

    el.addEventListener('touchstart', onStart, { passive: false });
    el.addEventListener('touchmove', onMove, { passive: false });
    el.addEventListener('touchend', onEnd);
    return () => {
      el.removeEventListener('touchstart', onStart);
      el.removeEventListener('touchmove', onMove);
      el.removeEventListener('touchend', onEnd);
    };
  }, [yawRef]);

  return <div ref={ref} className="touch-look-zone" />;
}

// ─── Action buttons (Talk / Sprint) ───────────────────────────────────────

function ActionButtons({
  sprintRef,
  talkRef,
}: {
  sprintRef: { current: boolean };
  talkRef: { current: boolean };
}) {
  return (
    <div className="action-buttons">
      <button
        className="action-btn action-sprint"
        onTouchStart={(e) => { e.preventDefault(); sprintRef.current = true; }}
        onTouchEnd={(e) => { e.preventDefault(); sprintRef.current = false; }}
        onMouseDown={() => { sprintRef.current = true; }}
        onMouseUp={() => { sprintRef.current = false; }}
      >
        <span>B</span>
        <span className="action-label">Sprint</span>
      </button>
      <button
        className="action-btn action-talk"
        onTouchStart={(e) => { e.preventDefault(); talkRef.current = true; }}
        onMouseDown={() => { talkRef.current = true; }}
      >
        <span>A</span>
        <span className="action-label">Talk</span>
      </button>
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────

export function WorldView3D() {
  const [locked, setLocked] = useState(false);
  const [day, setDay] = useState(0);
  const [npcs, setNpcs] = useState<NPCSummary[]>([]);
  const [settlement, setSettlement] = useState<Settlement | null>(null);
  const [environment, setEnvironment] = useState<EnvironmentState | null>(null);
  const [selectedNPC, setSelectedNPC] = useState<NPCSummary | null>(null);
  const [nearbyNPCs, setNearbyNPCs] = useState<NPCSummary[]>([]);
  const isMobile = /Mobi|Android/i.test(navigator.userAgent);

  const joystickAxisRef = useRef({ x: 0, y: 0 });
  const cameraYawRef = useRef(0);
  const sprintRef = useRef(false);
  const talkRef = useRef(false);

  useEffect(() => {
    api.getWorld().then((world) => { setDay(world.day); setEnvironment(world.environment); }).catch(() => {});
    api.getNPCs().then(setNpcs).catch(() => {});
    api.getSettlements().then((settlements) => { if (settlements[0]) setSettlement(settlements[0]); }).catch(() => {});
  }, []);

  function handleSelectNPC(npc: NPCSummary | null) {
    setSelectedNPC(npc);
    if (npc) setLocked(false);
  }

  function refreshWorld() {
    api.getWorld().then((world) => { setDay(world.day); setEnvironment(world.environment); }).catch(() => {});
    api.getNPCs().then(setNpcs).catch(() => {});
    api.getSettlements().then((settlements) => { if (settlements[0]) setSettlement(settlements[0]); }).catch(() => {});
  }

  return (
    <div className="world-3d-wrap">
      {/* Portrait-mode rotate hint */}
      {isMobile && (
        <div className="portrait-warning">
          <div className="portrait-icon">⟳</div>
          <div className="portrait-title">Rotate Device</div>
          <p className="portrait-sub">HomeoRealm plays best in landscape mode</p>
        </div>
      )}

      <Canvas
        shadows={!isMobile}
        camera={{ fov: isMobile ? 65 : 75, near: 0.1, far: 1000, position: [0, 1.7, 12] }}
        gl={{ antialias: !isMobile, powerPreference: 'high-performance' }}
      >
        <Suspense fallback={null}>
          <World3DScene
            npcs={npcs}
            settlement={settlement}
            environment={environment}
            isMobile={isMobile}
            onLockChange={setLocked}
            onSelectNPC={handleSelectNPC}
            onNearbyNPCs={setNearbyNPCs}
            joystickAxisRef={joystickAxisRef}
            cameraYawRef={cameraYawRef}
            sprintRef={sprintRef}
            talkRef={talkRef}
          />
        </Suspense>
      </Canvas>

      {/* HUD overlay */}
      <div className="hud-3d">
        {!isMobile && <PlayerHUD nearbyNPCs={nearbyNPCs} onWorldChanged={refreshWorld} />}
        {environment && !isMobile && <PhysicsChemistryHUD environment={environment} />}

        {/* Desktop: pre-lock prompt */}
        {!isMobile && !locked && !selectedNPC && (
          <div className="enter-3d-prompt">
            <div className="enter-3d-title">{settlement?.name ?? 'Auralis'}</div>
            <div className="enter-3d-day">Day {day}</div>
            <p className="enter-3d-sub">Click anywhere to enter first-person view</p>
            <p className="enter-3d-hint">WASD / Arrow keys — move · Mouse — look · E — talk · ESC — exit</p>
          </div>
        )}

        {/* Desktop: in-game HUD */}
        {!isMobile && locked && (
          <>
            <div className="crosshair-3d">
              <span className="ch-h" /><span className="ch-v" />
            </div>
            <div className="hud-3d-top">
              <span className="hud-3d-settlement">{settlement?.name ?? 'Auralis'}</span>
              <span className="hud-3d-day">Day {day}</span>
            </div>
            {nearbyNPCs.length > 0 && (
              <div className="hud-3d-nearby">
                <div className="nearby-title">Nearby residents</div>
                {nearbyNPCs.slice(0, 4).map((npc) => (
                  <div key={npc.id} className="nearby-npc-row">
                    <span className="nearby-name">{npc.name}</span>
                    <span className="nearby-job">{npc.jobId ?? '-'}</span>
                  </div>
                ))}
                <div className="interact-prompt">Press E to talk</div>
              </div>
            )}
            <div className="hud-3d-esc">ESC to exit</div>
          </>
        )}

        {/* Mobile: top status bar */}
        {isMobile && (
          <div className="hud-3d-top hud-3d-top--mobile">
            <span className="hud-3d-settlement">{settlement?.name ?? 'Auralis'}</span>
            <span className="hud-3d-day">Day {day}</span>
            {environment && (
              <span className="hud-3d-weather">{environment.weather.replace('_', ' ')}</span>
            )}
          </div>
        )}

        {/* NPC inspect panel — both platforms */}
        {selectedNPC && (
          <div className="npc-inspect-3d">
            <button className="npc-inspect-close" onClick={() => setSelectedNPC(null)}>✕</button>
            <div className="npc-inspect-name">{selectedNPC.name}</div>
            <div className="npc-inspect-sub">{selectedNPC.people} / {selectedNPC.jobId ?? 'No occupation'}</div>
            <div className="npc-inspect-bars">
              <div className="inspect-stat">
                <span>Viability</span>
                <div className="inspect-bar-track">
                  <div className="inspect-bar-fill" style={{ width: `${selectedNPC.viability * 100}%`, background: viabilityColor(selectedNPC.viability) }} />
                </div>
                <span>{(selectedNPC.viability * 100).toFixed(0)}%</span>
              </div>
              <div className="inspect-stat">
                <span>Health</span>
                <div className="inspect-bar-track">
                  <div className="inspect-bar-fill" style={{ width: `${selectedNPC.health * 100}%`, background: '#6c8ccc' }} />
                </div>
                <span>{(selectedNPC.health * 100).toFixed(0)}%</span>
              </div>
              <div className="inspect-stat">
                <span>Wealth</span>
                <div className="inspect-bar-track">
                  <div className="inspect-bar-fill" style={{ width: `${Math.min(selectedNPC.wealth / 200, 1) * 100}%`, background: '#c8a84b' }} />
                </div>
                <span>{selectedNPC.wealth.toFixed(0)}</span>
              </div>
            </div>
            <div className="npc-inspect-action">"{selectedNPC.currentAction}"</div>
            <button className="enter-3d-btn" onClick={() => setSelectedNPC(null)} style={{ marginTop: 12 }}>
              Close
            </button>
          </div>
        )}
      </div>

      {/* MOBA overlay — mobile only, shown when no NPC panel is open */}
      {isMobile && !selectedNPC && (
        <div className="moba-overlay">
          <VirtualJoystick axisRef={joystickAxisRef} />
          <TouchLookZone yawRef={cameraYawRef} />
          <ActionButtons sprintRef={sprintRef} talkRef={talkRef} />
          {nearbyNPCs.length > 0 && (
            <div className="moba-nearby">
              <div className="nearby-title">Nearby</div>
              {nearbyNPCs.slice(0, 3).map(npc => (
                <div key={npc.id} className="nearby-npc-row">
                  <span className="nearby-name">{npc.name}</span>
                </div>
              ))}
              <div className="interact-prompt">A = Talk</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function viabilityColor(value: number): string {
  if (value > 0.6) return '#5c8c6c';
  if (value > 0.3) return '#c8a84b';
  return '#c85c5c';
}

function PhysicsChemistryHUD({ environment }: { environment: EnvironmentState }) {
  const p = environment.physics;
  const c = environment.chemistry;
  return (
    <div className="physics-chem-hud">
      <div className="physics-chem-title">{environment.weather.replace('_', ' ')}</div>
      <div className="physics-chem-grid">
        <span>Temp</span><strong>{p.airTemperatureC.toFixed(1)}C</strong>
        <span>Wind</span><strong>{p.windSpeedMps.toFixed(1)} m/s</strong>
        <span>Rain</span><strong>{p.rainfallMm.toFixed(1)} mm</strong>
        <span>pH</span><strong>{c.soilPH.toFixed(2)}</strong>
        <span>CO2</span><strong>{c.carbonDioxidePpm} ppm</strong>
        <span>Corrosion</span><strong>{Math.round(c.corrosion * 100)}%</strong>
      </div>
    </div>
  );
}
