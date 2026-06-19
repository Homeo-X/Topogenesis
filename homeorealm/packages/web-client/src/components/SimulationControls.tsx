import { useState, useEffect } from 'react';
import { api, type WorldEvent } from '../api.js';

export function SimulationControls() {
  const [day, setDay] = useState(0);
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<WorldEvent[]>([]);
  const [tickCount, setTickCount] = useState(1);
  const [message, setMessage] = useState('');

  useEffect(() => {
    api.getWorld().then(w => setDay(w.day)).catch(() => {});
    api.getEvents().then(setEvents).catch(() => {});
  }, []);

  async function tick() {
    setRunning(true);
    setMessage('');
    try {
      const result = await api.tick();
      setDay(result.day);
      const evs = await api.getEvents();
      setEvents(evs);
      setMessage(`Day ${result.day} complete.`);
    } catch {
      setMessage('Tick failed — is the server running?');
    }
    setRunning(false);
  }

  async function runDays() {
    setRunning(true);
    setMessage('');
    try {
      const result = await api.runDays(tickCount);
      setDay(result.day);
      const evs = await api.getEvents();
      setEvents(evs);
      setMessage(`Advanced to day ${result.day}.`);
    } catch {
      setMessage('Run failed — is the server running?');
    }
    setRunning(false);
  }

  async function reset() {
    setRunning(true);
    try {
      await api.resetWorld(12345);
      const w = await api.getWorld();
      setDay(w.day);
      setEvents([]);
      setMessage('World reset to day 0 (seed: 12345).');
    } catch {
      setMessage('Reset failed.');
    }
    setRunning(false);
  }

  const salientEvents = events.filter(e => e.salience >= 0.6).slice(-30).reverse();

  return (
    <div className="sim-controls-panel">
      <div className="sim-header">
        <h2>Simulation Controls</h2>
        <div className="day-badge">Day {day}</div>
      </div>

      <div className="control-row">
        <button className="btn-primary" onClick={tick} disabled={running}>
          {running ? 'Simulating...' : 'Tick +1 Day'}
        </button>
        <div className="run-group">
          <input
            type="number"
            min={1}
            max={365}
            value={tickCount}
            onChange={e => setTickCount(Math.max(1, parseInt(e.target.value) || 1))}
            className="tick-input"
          />
          <button className="btn-secondary" onClick={runDays} disabled={running}>
            Run {tickCount} Days
          </button>
        </div>
        <button className="btn-danger" onClick={reset} disabled={running}>
          Reset World
        </button>
      </div>

      {message && <div className="sim-message">{message}</div>}

      <div className="event-log">
        <h3>Event Log — High Salience ({salientEvents.length})</h3>
        {salientEvents.length === 0 && (
          <div className="empty">No significant events yet. Advance the simulation to see history.</div>
        )}
        <div className="event-list">
          {salientEvents.map(e => (
            <div key={e.id} className={`event-row event-${e.type.split('_')[0]}`}>
              <span className="event-day">Day {e.day}</span>
              <span className="event-type">{e.type.replace(/_/g, ' ')}</span>
              <span className="event-actor">{e.actorId ?? '—'}</span>
              <span className="event-salience">{(e.salience * 100).toFixed(0)}%</span>
              {e.payload?.summary && (
                <span className="event-summary">{String(e.payload.summary)}</span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
