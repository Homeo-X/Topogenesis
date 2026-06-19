import { useState, useEffect } from 'react';
import { api, type Settlement } from '../api.js';

export function SettlementEconomy() {
  const [settlements, setSettlements] = useState<Settlement[]>([]);

  useEffect(() => { api.getSettlements().then(setSettlements).catch(() => {}); }, []);

  return (
    <div className="economy-panel">
      <h2>Settlement Economy</h2>
      {settlements.map(s => (
        <div key={s.id} className="settlement-card">
          <h3>{s.name} <span className="settlement-region">— {s.regionId.replace(/_/g, ' ')}</span></h3>
          <div className="econ-meta">Population: {s.population}</div>
          {s.pressures && s.pressures.length > 0 && (
            <div className="pressures">
              {s.pressures.map(p => <span key={p} className="pressure-tag">{p.replace(/_/g, ' ')}</span>)}
            </div>
          )}
          <div className="resource-table">
            {Object.entries(s.resources).map(([k, v]) => {
              if (typeof v !== 'number') return null;
              return (
                <div key={k} className="resource-row">
                  <span className="res-name">{k.replace(/([A-Z])/g, ' $1').toLowerCase()}</span>
                  <span className="res-amount">{(v as number).toFixed(1)}</span>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
