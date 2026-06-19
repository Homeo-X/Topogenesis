import { useState, useEffect } from 'react';
import { api, type DungeonRoom } from '../api.js';

export function DungeonView() {
  const [rooms, setRooms] = useState<DungeonRoom[]>([]);

  useEffect(() => { api.getDungeons().then(setRooms).catch(() => {}); }, []);

  return (
    <div className="dungeon-panel">
      <h2>Dungeon Topogenesis — Active Rooms ({rooms.length})</h2>
      {rooms.length === 0 && (
        <div className="empty">No dungeon rooms generated yet. Settlement pressures drive dungeon formation — advance the simulation.</div>
      )}
      <div className="dungeon-grid">
        {rooms.map(r => (
          <div key={r.id} className={`dungeon-card difficulty-${Math.floor(r.difficulty * 3)}`}>
            <div className="dungeon-header">
              <h4>{r.type.replace(/_/g, ' ')}</h4>
              <span className="difficulty-badge">Difficulty {Math.round(r.difficulty * 100)}%</span>
            </div>
            <div className="dungeon-pressure">Origin: <em>{r.pressureSource.replace(/_/g, ' ')}</em></div>
            <p>{r.description}</p>
            <div className="dungeon-tags">{r.tags.map(t => <span key={t} className="tag">{t}</span>)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
