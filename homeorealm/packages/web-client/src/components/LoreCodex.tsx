import { useMemo, useState, useEffect } from 'react';
import { api, type Region, type Faction, type People, type Asset } from '../api.js';

type Tab = 'regions' | 'factions' | 'peoples' | 'assets';

const TAB_LABELS: Record<Tab, string> = {
  regions: 'Regions',
  factions: 'Factions',
  peoples: 'Peoples',
  assets: 'Assets',
};

function titleCase(value: string): string {
  return value
    .split(/[_-]/g)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

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

  const filteredAssets = useMemo(() => {
    const query = assetFilter.trim().toLowerCase();
    if (!query) return assets;
    return assets.filter((asset) =>
      [
        asset.name,
        asset.category,
        asset.region,
        asset.productionPriority,
        asset.sourceCollection,
        asset.sourceLicense,
        ...asset.visualKeywords,
      ]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(query))
    );
  }, [assetFilter, assets]);

  const sourcedAssets = useMemo(() => assets.filter((asset) => asset.sourceUrl || asset.localAssetUrl).slice(0, 6), [assets]);

  return (
    <div className="lore-codex">
      <section className="codex-hero">
        <div className="codex-hero-copy">
          <span className="codex-eyebrow">Auralis field record</span>
          <h2>Lore Codex</h2>
          <p>
            Auralis is built around Hearthwells: warm civic anchors that keep families,
            markets, oaths, and dangerous subframe doors from unraveling into myth.
          </p>
        </div>
        <div className="codex-hero-stats" aria-label="Codex totals">
          <div><strong>{regions.length}</strong><span>Zones</span></div>
          <div><strong>{factions.length}</strong><span>Orders</span></div>
          <div><strong>{assets.length}</strong><span>Manifest</span></div>
        </div>
      </section>

      {sourcedAssets.length > 0 && (
        <section className="codex-source-strip">
          <div>
            <span className="codex-eyebrow">Live art pass</span>
            <strong>Imported CC0 medieval fair models now dress the 3D town.</strong>
          </div>
          <div className="source-pill-row">
            {sourcedAssets.map((asset) => (
              <span key={asset.id} className="source-pill">{asset.name}</span>
            ))}
          </div>
        </section>
      )}

      <div className="tabs">
        {(['regions', 'factions', 'peoples', 'assets'] as Tab[]).map((value) => (
          <button key={value} className={tab === value ? 'tab active' : 'tab'} onClick={() => setTab(value)}>
            {TAB_LABELS[value]}
          </button>
        ))}
      </div>

      {tab === 'regions' && (
        <div className="codex-entries">
          {regions.map((region) => (
            <article key={region.id} className="codex-card codex-card-featured">
              <div className="codex-card-kicker">{region.climate}</div>
              <h3>{region.name}</h3>
              <p>{region.description}</p>
              <div className="codex-tag-row">
                {region.mainResources.map((resource) => <span key={resource} className="codex-chip">{titleCase(resource)}</span>)}
              </div>
            </article>
          ))}
        </div>
      )}

      {tab === 'factions' && (
        <div className="codex-entries">
          {factions.map((faction) => (
            <article key={faction.id} className="codex-card">
              <div className="codex-card-kicker">{faction.function}</div>
              <h3>{faction.name}</h3>
              <p>{faction.description}</p>
              {faction.allegiances && (
                <div className="codex-tag-row">
                  {faction.allegiances.map((allegiance) => <span key={allegiance} className="codex-chip">{titleCase(allegiance)}</span>)}
                </div>
              )}
            </article>
          ))}
        </div>
      )}

      {tab === 'peoples' && (
        <div className="codex-entries">
          {peoples.map((people) => (
            <article key={people.id} className="codex-card">
              <div className="codex-card-kicker">{titleCase(people.region)}</div>
              <h3>{people.name}</h3>
              <p>{people.description}</p>
              <div className="codex-tag-row">
                {people.culturalTraits.map((trait) => <span key={trait} className="codex-chip">{titleCase(trait)}</span>)}
              </div>
              <p className="codex-asset">{people.assetTheme}</p>
            </article>
          ))}
        </div>
      )}

      {tab === 'assets' && (
        <div>
          <input
            value={assetFilter}
            onChange={(event) => setAssetFilter(event.target.value)}
            placeholder="Filter by name, region, license, source, or keyword..."
            className="filter-input"
          />
          <div className="codex-entries">
            {filteredAssets.map((asset) => (
              <article key={asset.id} className="codex-card asset-card">
                <div className="asset-header">
                  <h4>{asset.name}</h4>
                  <span className={`priority-badge ${asset.productionPriority}`}>{asset.productionPriority}</span>
                  <span className="asset-category">{asset.category}</span>
                </div>
                <p>{asset.description}</p>
                <div className="asset-meta">
                  <span>Use: {asset.gameplayUse}</span>
                  <span>Keywords: {asset.visualKeywords.join(', ')}</span>
                  {asset.localAssetUrl && <span>Local: {asset.localAssetUrl}</span>}
                  {asset.sourceLicense && <span>License: {asset.sourceLicense}</span>}
                  {asset.sourceCollection && <span>Source: {asset.sourceCollection}</span>}
                  {asset.sourceUrl && (
                    <a href={asset.sourceUrl} target="_blank" rel="noreferrer">
                      Open source asset
                    </a>
                  )}
                </div>
              </article>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
