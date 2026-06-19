import { useState, useEffect } from 'react';
import { api, type Household } from '../api.js';

export function HouseholdPanel() {
  const [households, setHouseholds] = useState<Household[]>([]);
  const [selected, setSelected] = useState<Household | null>(null);

  useEffect(() => {
    api.getHouseholds().then(setHouseholds).catch(() => {});
  }, []);

  async function selectHousehold(id: string) {
    try {
      const hh = await api.getHousehold(id);
      setSelected(hh);
    } catch {}
  }

  return (
    <div className="household-panel">
      <h2>Households ({households.length})</h2>
      <div className="household-grid">
        {households.map(h => (
          <div key={h.id} className={`household-card ${h.foodStores < 2 ? 'household-warning' : ''}`} onClick={() => selectHousehold(h.id)}>
            <div className="hh-name">{h.name}</div>
            <div className="hh-members">{h.memberIds.length} members</div>
            <div className="hh-food">Food: {h.foodStores.toFixed(1)}</div>
            <div className="hh-wealth">Wealth: {h.wealth.toFixed(0)}</div>
            {h.foodStores < 2 && <div className="hh-alert">⚠ Food shortage</div>}
          </div>
        ))}
      </div>

      {selected && (
        <div className="quest-modal" onClick={() => setSelected(null)}>
          <div className="quest-detail" onClick={e => e.stopPropagation()}>
            <h3>{selected.name}</h3>
            <div>Food Stores: {selected.foodStores.toFixed(2)}</div>
            <div>Wealth: {selected.wealth.toFixed(0)} coin</div>
            <div>Home Quality: {(selected.homeQuality * 100).toFixed(0)}%</div>
            <div>Reputation: {(selected.reputation * 100).toFixed(0)}%</div>
            <h4>Members</h4>
            {(selected.members ?? []).map(m => (
              <div key={m.id}>{m.name} — {m.jobId ?? 'no job'}</div>
            ))}
            <button onClick={() => setSelected(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}
