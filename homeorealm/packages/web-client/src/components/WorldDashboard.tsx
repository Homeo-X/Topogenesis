import { useState, useEffect } from 'react';
import { api, type WorldSummary } from '../api.js';

type Props = { onTick?: () => void };

export function WorldDashboard({ onTick }: Props) {
  const [world, setWorld] = useState<WorldSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [seed, setSeed] = useState('42');

  async function load() {
    try { setWorld(await api.getWorld()); } catch {}
  }

  useEffect(() => { load(); }, []);

  async function tick() {
    setLoading(true);
    try { await api.tick(); await load(); onTick?.(); } finally { setLoading(false); }
  }

  async function run7() {
    setLoading(true);
    try { await api.runDays(7); await load(); onTick?.(); } finally { setLoading(false); }
  }

  async function reset() {
    setLoading(true);
    try { await api.resetWorld(parseInt(seed) || 42); await load(); onTick?.(); } finally { setLoading(false); }
  }

  if (!world) return <div className="loading">Connecting to Hearthwell...</div>;

  const s = world.settlements[0];
  const r = s?.resources;

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>World Chronicle — Day {world.day}</h2>
        <div className="controls">
          <button onClick={tick} disabled={loading}>Advance 1 Day</button>
          <button onClick={run7} disabled={loading}>Run 7 Days</button>
          <input value={seed} onChange={e => setSeed(e.target.value)} placeholder="seed" style={{ width: 70 }} />
          <button onClick={reset} disabled={loading}>Reset World</button>
        </div>
      </div>

      <div className="card-grid">
        <StatCard label="Settlement" value={s?.name ?? '—'} />
        <StatCard label="Population" value={String(world.totalNPCs)} />
        <StatCard label="Avg Viability" value={world.avgViability} />
        <StatCard label="Active Quests" value={String(world.activeQuests)} />
        <StatCard label="Dungeon Rooms" value={String(world.dungeonRooms)} />
      </div>

      <EnvironmentPanel world={world} />

      {r && (
        <div className="resource-panel">
          <h3>Settlement Resources — {s?.name}</h3>
          <div className="resource-grid">
            <ResourceBar label="Food" value={r.food} max={200} color="#c8a84b" />
            <ResourceBar label="Wood" value={r.wood} max={150} color="#7a8c5c" />
            <ResourceBar label="Ore" value={r.ore} max={100} color="#8c7a6c" />
            <ResourceBar label="Cloth" value={r.cloth} max={80} color="#6c7a8c" />
            <ResourceBar label="Medicine" value={r.medicine} max={60} color="#5c8c6c" />
            <ResourceBar label="Tools" value={r.tools} max={80} color="#8c6c5c" />
            <ResourceBar label="Coin" value={r.coin} max={1000} color="#c8b84b" />
          </div>
          <div className="stability-grid">
            <StabilityBar label="⬡ Hearthwell" value={r.heartwellStability} />
            <StabilityBar label="♦ Public Morale" value={r.publicMorale} />
            <StabilityBar label="⚔ Security" value={r.security} />
          </div>
        </div>
      )}
    </div>
  );
}

function EnvironmentPanel({ world }: { world: WorldSummary }) {
  const env = world.environment;
  const p = env.physics;
  const c = env.chemistry;
  return (
    <div className="resource-panel environment-panel">
      <h3>Physics & Chemistry Layer</h3>
      <div className="environment-grid">
        <EnvMetric label="Weather" value={env.weather.replace('_', ' ')} />
        <EnvMetric label="Season" value={env.season} />
        <EnvMetric label="Temp" value={`${p.airTemperatureC.toFixed(1)} C`} />
        <EnvMetric label="Humidity" value={`${Math.round(p.humidity * 100)}%`} />
        <EnvMetric label="Rainfall" value={`${p.rainfallMm.toFixed(1)} mm`} />
        <EnvMetric label="Wind" value={`${p.windSpeedMps.toFixed(1)} m/s`} />
        <EnvMetric label="Soil pH" value={c.soilPH.toFixed(2)} />
        <EnvMetric label="CO2" value={`${c.carbonDioxidePpm} ppm`} />
        <EnvMetric label="Oxygen" value={`${(c.oxygenRatio * 100).toFixed(1)}%`} />
        <EnvMetric label="Fermentation" value={`${Math.round(c.fermentation * 100)}%`} />
        <EnvMetric label="Corrosion" value={`${Math.round(c.corrosion * 100)}%`} />
        <EnvMetric label="Minerals" value={`${Math.round(c.mineralSaturation * 100)}%`} />
      </div>
      {env.signals.length > 0 && (
        <div className="environment-signals">
          {env.signals.map((signal) => <span key={signal}>{signal}</span>)}
        </div>
      )}
    </div>
  );
}

function EnvMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="env-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

function ResourceBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="resource-bar">
      <span className="res-label">{label}</span>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="res-value">{value.toFixed(1)}</span>
    </div>
  );
}

function StabilityBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  const color = pct > 60 ? '#5c8c6c' : pct > 30 ? '#c8a84b' : '#c85c5c';
  return (
    <div className="stability-bar">
      <span>{label}</span>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span>{pct}%</span>
    </div>
  );
}
