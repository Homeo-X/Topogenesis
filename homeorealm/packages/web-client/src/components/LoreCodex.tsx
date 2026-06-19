import { useState, useEffect } from 'react';
import { api, type Region, type Faction, type People, type Asset } from '../api.js';

type Tab = 'regions' | 'factions' | 'peoples' | 'assets';

export function LoreCodex() {
  const [tab, setTab] = useState<Tab>('regions');
  const [regions, setRegions] = useState<Region[]>([]);
  const [factions, setFactions] = useState<Faction[]>([]);
  const [peoples, setPeoples] = useState<People[]>([]);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [assetFilter, setAssetFilter] = useState('');

  useEffect(() => {
    api.getRegions().then(setRegions).catch(() => {});
    api.getFactions().then(setFactions).catch(() => {});
    api.getPeoples().then(setPeoples).catch(() => {});
    api.getAssets().then(setAssets).catch(() => {});
  }, []);

  return (
    <div className="lore-codex">
      <h2>Lore Codex — Auralis</h2>
      <div className="tabs">
        {(['regions', 'factions', 'peoples', 'assets'] as Tab[]).map(t => (
          <button key={t} className={tab === t ? 'tab active' : 'tab'} onClick={() => setTab(t)}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'regions' && (
        <div className="codex-entries">
          {regions.map(r => (
            <div key={r.id} className="codex-card">
              <h3>{r.name}</h3>
              <div className="codex-sub">Climate: {r.climate} · Resources: {r.mainResources.join(', ')}</div>
              <p>{r.description}</p>
            </div>
          ))}
        </div>
      )}

      {tab === 'factions' && (
        <div className="codex-entries">
          {factions.map(f => (
            <div key={f.id} className="codex-card">
              <h3>{f.name}</h3>
              <div className="codex-sub">Function: {f.function}</div>
              <p>{f.description}</p>
            </div>
          ))}
        </div>
      )}

      {tab === 'peoples' && (
        <div className="codex-entries">
          {peoples.map(p => (
            <div key={p.id} className="codex-card">
              <h3>{p.name}</h3>
              <div className="codex-sub">Traits: {p.culturalTraits.join(', ')}</div>
              <p>{p.description}</p>
              <p className="codex-asset"><em>{p.assetTheme}</em></p>
            </div>
          ))}
        </div>
      )}

      {tab === 'assets' && (
        <div>
          <input value={assetFilter} onChange={e => setAssetFilter(e.target.value)} placeholder="Filter assets..." className="filter-input" />
          <div className="codex-entries">
            {assets
              .filter(a => !assetFilter || a.name.toLowerCase().includes(assetFilter.toLowerCase()) || a.category.includes(assetFilter))
              .map(a => (
                <div key={a.id} className="codex-card asset-card">
                  <div className="asset-header">
                    <h4>{a.name}</h4>
                    <span className={`priority-badge ${a.productionPriority}`}>{a.productionPriority}</span>
                    <span className="asset-category">{a.category}</span>
                  </div>
                  <p>{a.description}</p>
                  <div className="asset-meta">
                    <span>Use: {a.gameplayUse}</span>
                    <span>Keywords: {a.visualKeywords.join(', ')}</span>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
