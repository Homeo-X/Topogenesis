import { Canvas } from '@react-three/fiber';
import { Suspense, useState, useEffect } from 'react';
import { api } from '../api.js';
import type { NPCSummary, Settlement } from '../api.js';
import { World3DScene } from './world3d/World3DScene.js';

export function WorldView3D() {
  const [locked, setLocked] = useState(false);
  const [day, setDay] = useState(0);
  const [npcs, setNpcs] = useState<NPCSummary[]>([]);
  const [settlement, setSettlement] = useState<Settlement | null>(null);
  const [selectedNPC, setSelectedNPC] = useState<NPCSummary | null>(null);
  const [nearbyNPCs, setNearbyNPCs] = useState<NPCSummary[]>([]);
  const isMobile = /Mobi|Android/i.test(navigator.userAgent);

  useEffect(() => {
    api.getWorld().then(w => setDay(w.day)).catch(() => {});
    api.getNPCs().then(setNpcs).catch(() => {});
    api.getSettlements().then(s => { if (s[0]) setSettlement(s[0]); }).catch(() => {});
  }, []);

  function handleSelectNPC(npc: NPCSummary | null) {
    setSelectedNPC(npc);
    if (npc) setLocked(false);
  }

  return (
    <div className="world-3d-wrap">
      <Canvas
        shadows={!isMobile}
        camera={{ fov: 75, near: 0.1, far: 1000, position: [0, 1.7, 30] }}
        gl={{ antialias: !isMobile, powerPreference: 'high-performance' }}
      >
        <Suspense fallback={null}>
          <World3DScene
            npcs={npcs}
            settlement={settlement}
            isMobile={isMobile}
            onLockChange={setLocked}
            onSelectNPC={handleSelectNPC}
            onNearbyNPCs={setNearbyNPCs}
          />
        </Suspense>
      </Canvas>

      {/* HTML overlay — pointer-events:none lets clicks reach canvas by default */}
      <div className="hud-3d">

        {/* Click-to-enter prompt */}
        {!locked && !selectedNPC && (
          <div className="enter-3d-prompt">
            <div className="enter-3d-title">{settlement?.name ?? 'Auralis'}</div>
            <div className="enter-3d-day">Day {day}</div>
            {isMobile
              ? <p className="enter-3d-hint">Drag to look · Pinch to zoom</p>
              : <>
                  <p className="enter-3d-sub">Click anywhere to enter first-person view</p>
                  <p className="enter-3d-hint">WASD / Arrow keys — move &nbsp;·&nbsp; Mouse — look &nbsp;·&nbsp; E — talk to NPC &nbsp;·&nbsp; ESC — exit</p>
                </>
            }
          </div>
        )}

        {/* First-person HUD */}
        {locked && (
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
                {nearbyNPCs.slice(0, 4).map(n => (
                  <div key={n.id} className="nearby-npc-row">
                    <span className="nearby-name">{n.name}</span>
                    <span className="nearby-job">{n.jobId ?? '—'}</span>
                  </div>
                ))}
                <div className="interact-prompt">Press E to talk</div>
              </div>
            )}
            <div className="hud-3d-esc">ESC to exit</div>
          </>
        )}

        {/* NPC inspector panel */}
        {selectedNPC && (
          <div className="npc-inspect-3d">
            <button className="npc-inspect-close" onClick={() => setSelectedNPC(null)}>✕</button>
            <div className="npc-inspect-name">{selectedNPC.name}</div>
            <div className="npc-inspect-sub">{selectedNPC.people} · {selectedNPC.jobId ?? 'No occupation'}</div>
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
    </div>
  );
}

function viabilityColor(v: number): string {
  if (v > 0.6) return '#5c8c6c';
  if (v > 0.3) return '#c8a84b';
  return '#c85c5c';
}
