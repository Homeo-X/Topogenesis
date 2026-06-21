import { useEffect, useMemo, useState } from 'react';
import { api, type DungeonRoom, type EnvironmentState, type NPCSummary, type PlayerActionInput, type PlayerState, type Quest, type Region, type Settlement, type WorldSummary } from '../api.js';

type MapMode = 'world' | 'region' | 'area';
type Overlay = 'terrain' | 'climate' | 'resources' | 'travel' | 'pressure';
type FocusKind = 'region' | 'resource' | 'settlement' | 'district' | 'resident' | 'quest' | 'dungeon' | 'player';

type MapFocus = {
  id: string;
  kind: FocusKind;
  title: string;
  body: string;
  tags: string[];
};

type RegionShape = {
  id: string;
  label: string;
  path: string;
  labelPos: [number, number];
  color: string;
  highColor: string;
  elevation: string;
  features: string[];
  resources: Array<{ label: string; x: number; y: number; kind: string }>;
  roads: string[];
  rivers: string[];
};

const REGION_SHAPES: RegionShape[] = [
  {
    id: 'quiet_north',
    label: 'Quiet North',
    path: 'M230 62 C330 24 482 30 610 52 C704 68 780 96 824 150 C782 190 695 194 610 180 C522 166 440 176 348 166 C270 158 214 128 230 62 Z',
    labelPos: [535, 108],
    color: '#d8e5ea',
    highColor: '#f3f7f8',
    elevation: 'glacial plateau, ancestor-stone ridges',
    features: ['archive vaults', 'ice roads', 'topogenic crystals'],
    resources: [
      { label: 'Crystal scree', x: 590, y: 96, kind: 'ore' },
      { label: 'Cold herbs', x: 420, y: 135, kind: 'medicine' },
    ],
    roads: ['M390 156 C462 132 544 124 664 156'],
    rivers: ['M512 164 C500 210 478 246 448 286'],
  },
  {
    id: 'highroot',
    label: 'Highroot Canopy',
    path: 'M124 150 C200 95 318 120 374 185 C332 244 332 306 260 350 C180 344 102 304 72 236 C58 198 82 170 124 150 Z',
    labelPos: [222, 232],
    color: '#4f7d56',
    highColor: '#82a764',
    elevation: 'vertical forest terraces',
    features: ['canopy lifts', 'rain gardens', 'rare woods'],
    resources: [
      { label: 'Canopy fruit', x: 190, y: 205, kind: 'food' },
      { label: 'Rare timber', x: 255, y: 288, kind: 'wood' },
    ],
    roads: ['M246 326 C268 276 284 228 344 186'],
    rivers: ['M116 224 C164 238 226 250 300 236'],
  },
  {
    id: 'mireglass',
    label: 'Mireglass',
    path: 'M96 328 C164 302 248 332 302 390 C344 434 322 512 238 548 C154 574 76 520 62 440 C56 392 66 348 96 328 Z',
    labelPos: [202, 431],
    color: '#557c75',
    highColor: '#91b1a2',
    elevation: 'reed lowlands and reflecting pools',
    features: ['memory pools', 'reed houses', 'funeral lantern roads'],
    resources: [
      { label: 'Glass reeds', x: 178, y: 398, kind: 'medicine' },
      { label: 'Pool herbs', x: 230, y: 486, kind: 'medicine' },
    ],
    roads: ['M276 402 C230 424 184 448 118 496'],
    rivers: ['M260 360 C220 402 194 448 178 530', 'M96 380 C154 392 206 418 284 460'],
  },
  {
    id: 'crown_valley',
    label: 'Crown Valley',
    path: 'M344 214 C428 172 552 178 636 232 C690 268 674 356 606 408 C532 466 414 452 350 398 C288 344 274 260 344 214 Z',
    labelPos: [488, 310],
    color: '#8aa35c',
    highColor: '#d1b866',
    elevation: 'river valley, farms, civic market roads',
    features: ['Hearthwell', 'festival grounds', 'wheat terraces'],
    resources: [
      { label: 'Wheat fields', x: 420, y: 348, kind: 'food' },
      { label: 'Hearthwell', x: 500, y: 306, kind: 'hearth' },
      { label: 'Market road', x: 566, y: 338, kind: 'trade' },
    ],
    roads: ['M316 382 C398 332 470 312 616 270', 'M422 210 C470 282 512 338 594 424'],
    rivers: ['M460 182 C438 248 438 310 392 390 C368 430 336 464 286 500'],
  },
  {
    id: 'ash_ring',
    label: 'Ash-Ring',
    path: 'M660 238 C752 202 870 248 916 342 C944 398 898 470 820 500 C734 532 642 494 608 426 C568 348 594 270 660 238 Z',
    labelPos: [778, 372],
    color: '#7e5d55',
    highColor: '#bd6c43',
    elevation: 'volcanic marches and basalt passes',
    features: ['forge vents', 'watch forts', 'lava-glass gullies'],
    resources: [
      { label: 'Ash ore', x: 742, y: 328, kind: 'ore' },
      { label: 'Basalt forge', x: 820, y: 406, kind: 'tools' },
    ],
    roads: ['M618 424 C690 396 760 370 878 356'],
    rivers: ['M694 244 C730 312 770 358 842 442'],
  },
  {
    id: 'saffron_coast',
    label: 'Saffron Coast',
    path: 'M348 438 C456 492 606 472 724 504 C792 522 858 548 902 586 L312 586 C288 542 292 482 348 438 Z',
    labelPos: [596, 536],
    color: '#c8a45f',
    highColor: '#5f9dad',
    elevation: 'salt coast, docks, tidal caves',
    features: ['harbors', 'fish markets', 'sail shrines'],
    resources: [
      { label: 'Salt pans', x: 610, y: 542, kind: 'coin' },
      { label: 'Fishing docks', x: 748, y: 562, kind: 'food' },
    ],
    roads: ['M366 452 C454 510 562 526 734 546'],
    rivers: ['M448 440 C486 492 510 530 540 584'],
  },
  {
    id: 'subframe_labyrinth',
    label: 'Subframe',
    path: 'M430 392 C482 366 556 382 602 436 C574 480 516 500 452 476 C418 452 406 420 430 392 Z',
    labelPos: [510, 434],
    color: '#2e2740',
    highColor: '#7254af',
    elevation: 'unstable layer below Hearthwells',
    features: ['pressure rooms', 'glyph corridors', 'memory rupture'],
    resources: [
      { label: 'Rift', x: 512, y: 430, kind: 'dungeon' },
    ],
    roads: [],
    rivers: [],
  },
];

const AREA_LAYERS = [
  { id: 'plaza', label: 'Hearthwell Plaza', x: 500, y: 318, w: 150, h: 95, color: '#8c8270' },
  { id: 'market', label: 'Market Row', x: 648, y: 280, w: 150, h: 90, color: '#b99b54' },
  { id: 'fields', label: 'Wheat Terraces', x: 248, y: 332, w: 230, h: 150, color: '#c8ad55' },
  { id: 'homes', label: 'Household Ring', x: 330, y: 190, w: 260, h: 120, color: '#9b7960' },
  { id: 'grove', label: 'Highroot Grove', x: 182, y: 130, w: 220, h: 130, color: '#4e7650' },
  { id: 'watch', label: 'Hearthwarden Edge', x: 724, y: 178, w: 170, h: 120, color: '#6e737a' },
  { id: 'rupture', label: 'Subframe Rupture', x: 238, y: 110, w: 110, h: 76, color: '#40305f' },
];

const PLAYER_AREA_POS: Record<PlayerState['location'], [number, number]> = {
  town: [506, 318],
  market: [674, 318],
  wilds: [250, 168],
  dungeon: [250, 168],
  home: [330, 190],
};

const MAP_ACTIONS: Array<{ label: string; action: PlayerActionInput; tone?: 'primary' | 'danger' }> = [
  { label: 'Town', action: { type: 'travel', location: 'town' } },
  { label: 'Market', action: { type: 'travel', location: 'market' } },
  { label: 'Wilds', action: { type: 'travel', location: 'wilds' } },
  { label: 'Gather', action: { type: 'gather' }, tone: 'primary' },
  { label: 'Trade', action: { type: 'trade' } },
  { label: 'Delve', action: { type: 'delve_dungeon' }, tone: 'danger' },
  { label: 'Rest', action: { type: 'rest' } },
];

function titleCase(value: string): string {
  return value
    .split(/[_-]/g)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function regionById(id: string): RegionShape {
  return REGION_SHAPES.find((region) => region.id === id) ?? REGION_SHAPES[3]!;
}

function makeFocus(kind: FocusKind, id: string, title: string, body: string, tags: string[] = []): MapFocus {
  return { kind, id, title, body, tags };
}

export function MapAtlas() {
  const [mode, setMode] = useState<MapMode>('world');
  const [overlay, setOverlay] = useState<Overlay>('terrain');
  const [selectedRegionId, setSelectedRegionId] = useState('crown_valley');
  const [focus, setFocus] = useState<MapFocus>(() => makeFocus('region', 'crown_valley', 'Crown Valley', 'The starting heartland: farms, marriage houses, market roads, and a watchful Hearthwell.', ['safe start', 'bread roads']));
  const [world, setWorld] = useState<WorldSummary | null>(null);
  const [regions, setRegions] = useState<Region[]>([]);
  const [settlements, setSettlements] = useState<Settlement[]>([]);
  const [npcs, setNpcs] = useState<NPCSummary[]>([]);
  const [quests, setQuests] = useState<Quest[]>([]);
  const [dungeons, setDungeons] = useState<DungeonRoom[]>([]);
  const [player, setPlayer] = useState<PlayerState | null>(null);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);

  async function loadAtlas(resetFocus = false) {
    const [nextWorld, nextRegions, nextSettlements, nextNpcs, nextQuests, nextDungeons, nextPlayer] = await Promise.all([
      api.getWorld(),
      api.getRegions(),
      api.getSettlements(),
      api.getNPCs(),
      api.getQuests(),
      api.getDungeons(),
      api.getPlayer(),
    ]);

    setWorld(nextWorld);
    setRegions(nextRegions);
    setSettlements(nextSettlements);
    setNpcs(nextNpcs);
    setQuests(nextQuests);
    setDungeons(nextDungeons);
    setPlayer(nextPlayer);

    const homeRegionId = nextSettlements[0]?.regionId ?? 'crown_valley';
    setSelectedRegionId((current) => current || homeRegionId);
    if (resetFocus) {
      const shape = regionById(homeRegionId);
      const lore = nextRegions.find((region) => region.id === homeRegionId);
      setFocus(makeFocus('region', shape.id, shape.label, lore?.description ?? shape.elevation, shape.features));
    }
  }

  useEffect(() => { loadAtlas(true).catch(() => {}); }, []);

  async function runMapAction(action: PlayerActionInput) {
    setBusy(true);
    setMessage('');
    try {
      const result = await api.playerAction(action);
      setPlayer(result.player);
      setMessage(result.message);
      await loadAtlas(false);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const selectedShape = regionById(selectedRegionId);
  const selectedRegion = regions.find((region) => region.id === selectedRegionId);
  const activeQuests = quests.filter((quest) => quest.isActive);
  const settlement = settlements[0];
  const environment = world?.environment;
  const route = useMemo(() => computeRouteReadiness(mode, overlay, environment, player, selectedShape), [environment, mode, overlay, player, selectedShape]);

  const stats = useMemo(() => ({
    residents: npcs.length,
    activeQuests: activeQuests.length,
    rooms: dungeons.length,
    weather: environment?.weather ? titleCase(environment.weather) : 'Unknown',
  }), [activeQuests.length, dungeons.length, environment?.weather, npcs.length]);

  return (
    <div className="map-atlas">
      <div className="map-toolbar">
        <div className="map-tabs">
          {(['world', 'region', 'area'] as MapMode[]).map((value) => (
            <button key={value} className={mode === value ? 'map-tab active' : 'map-tab'} onClick={() => setMode(value)}>
              {titleCase(value)} Map
            </button>
          ))}
        </div>
        <div className="map-controls">
          <select value={selectedRegionId} onChange={(event) => { setSelectedRegionId(event.target.value); setMode('region'); }} aria-label="Region">
            {REGION_SHAPES.filter((region) => region.id !== 'subframe_labyrinth').map((region) => (
              <option key={region.id} value={region.id}>{region.label}</option>
            ))}
          </select>
          <select value={overlay} onChange={(event) => setOverlay(event.target.value as Overlay)} aria-label="Overlay">
            <option value="terrain">Terrain</option>
            <option value="climate">Climate</option>
            <option value="resources">Resources</option>
            <option value="travel">Travel</option>
            <option value="pressure">Pressure</option>
          </select>
        </div>
      </div>

      <div className="map-shell">
        <section className="map-canvas-panel">
          {mode === 'world' && <WorldMap overlay={overlay} selectedRegionId={selectedRegionId} onSelectRegion={(id) => {
            const shape = regionById(id);
            const lore = regions.find((region) => region.id === id);
            setSelectedRegionId(id);
            setFocus(makeFocus('region', id, shape.label, lore?.description ?? shape.elevation, shape.features));
            setMode('region');
          }} onFocus={setFocus} settlements={settlements} environment={environment} focus={focus} />}
          {mode === 'region' && <RegionMap shape={selectedShape} region={selectedRegion} overlay={overlay} settlement={settlement} quests={activeQuests} dungeons={dungeons} environment={environment} focus={focus} onFocus={setFocus} />}
          {mode === 'area' && <AreaMap overlay={overlay} settlement={settlement} npcs={npcs} quests={activeQuests} dungeons={dungeons} environment={environment} player={player} focus={focus} onFocus={setFocus} />}
        </section>

        <aside className="map-side-panel">
          <h2>{mode === 'world' ? 'Auralis World' : mode === 'region' ? selectedShape.label : settlement?.name ?? 'Vennholt'}</h2>
          <div className="map-stat-grid">
            <MapStat label="Residents" value={String(stats.residents)} />
            <MapStat label="Quests" value={String(stats.activeQuests)} />
            <MapStat label="Dungeons" value={String(stats.rooms)} />
            <MapStat label="Weather" value={stats.weather} />
          </div>
          <div className="map-detail-block">
            <h3>{mode === 'area' ? 'Area Notes' : 'Region Notes'}</h3>
            <p>{mode === 'world' ? 'Surface regions are stitched by trade roads, rivers, seasonal passes, and the hidden subframe pressure layer below every Hearthwell.' : selectedRegion?.description ?? selectedShape.elevation}</p>
            <div className="map-chip-row">
              {selectedShape.features.map((feature) => <span key={feature}>{feature}</span>)}
            </div>
          </div>
          {environment && (
            <div className="map-detail-block">
              <h3>Environment</h3>
              <div className="map-kv">
                <span>Temperature</span><strong>{environment.physics.airTemperatureC.toFixed(1)} C</strong>
                <span>Rainfall</span><strong>{environment.physics.rainfallMm.toFixed(1)} mm</strong>
                <span>Soil pH</span><strong>{environment.chemistry.soilPH.toFixed(2)}</strong>
                <span>Corrosion</span><strong>{Math.round(environment.chemistry.corrosion * 100)}%</strong>
              </div>
            </div>
          )}
          <div className="map-detail-block">
            <h3>Selected</h3>
            <div className="map-focus-card">
              <div className="map-focus-kind">{titleCase(focus.kind)}</div>
              <strong>{focus.title}</strong>
              <p>{focus.body}</p>
              {focus.tags.length > 0 && (
                <div className="map-chip-row compact">
                  {focus.tags.map((tag) => <span key={tag}>{tag}</span>)}
                </div>
              )}
            </div>
          </div>
          <div className="map-detail-block">
            <h3>Expedition Readiness</h3>
            <div className="map-route-card">
              <div className="map-route-head">
                <span>{route.label}</span>
                <strong>{Math.round(route.readiness * 100)}%</strong>
              </div>
              <div className="map-route-meter">
                <div className="map-route-fill" style={{ width: `${route.readiness * 100}%`, background: route.color }} />
              </div>
              <p>{route.note}</p>
              {player && <div className="map-kv compact"><span>Position</span><strong>{titleCase(player.location)}</strong><span>Stamina</span><strong>{Math.round(player.stamina * 100)}%</strong></div>}
            </div>
          </div>
          <div className="map-detail-block">
            <h3>Map Actions</h3>
            <div className="map-action-grid">
              {MAP_ACTIONS.map(({ label, action, tone }) => (
                <button key={`${action.type}-${action.location ?? 'self'}`} className={tone ? `map-action ${tone}` : 'map-action'} disabled={busy} onClick={() => runMapAction(action)}>
                  {label}
                </button>
              ))}
            </div>
            {message && <div className="map-message">{message}</div>}
          </div>
          <div className="map-legend">
            <LegendItem color="#7c5a31" label="Roads" />
            <LegendItem color="#5f9dad" label="Water" />
            <LegendItem color="#d1b866" label="Food" />
            <LegendItem color="#8b8d93" label="Ore / tools" />
            <LegendItem color="#7a6eff" label="Dungeon pressure" />
          </div>
        </aside>
      </div>
    </div>
  );
}

function WorldMap({
  overlay,
  selectedRegionId,
  onSelectRegion,
  settlements,
  environment,
  focus,
  onFocus,
}: {
  overlay: Overlay;
  selectedRegionId: string;
  onSelectRegion: (id: string) => void;
  settlements: Settlement[];
  environment?: EnvironmentState;
  focus: MapFocus;
  onFocus: (focus: MapFocus) => void;
}) {
  return (
    <svg className="atlas-svg" viewBox="0 0 1000 620" preserveAspectRatio="xMidYMin meet" role="img" aria-label="Auralis world map">
      <defs>
        <pattern id="mapGrid" width="24" height="24" patternUnits="userSpaceOnUse">
          <path d="M24 0H0V24" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
        </pattern>
        <filter id="mapShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="8" stdDeviation="6" floodColor="#000" floodOpacity="0.28" />
        </filter>
      </defs>
      <rect width="1000" height="620" fill="#2f5f74" />
      <path d="M0 514 C164 492 264 518 376 492 C474 468 586 464 702 500 C818 536 914 520 1000 486 L1000 620 L0 620 Z" fill="#244d65" opacity="0.8" />
      <rect width="1000" height="620" fill="url(#mapGrid)" opacity="0.38" />
      <path d="M90 110 C245 8 620 8 838 122 C978 196 972 448 880 560 C700 612 340 604 126 546 C10 420 8 204 90 110 Z" fill="#55634f" filter="url(#mapShadow)" />

      {REGION_SHAPES.map((region) => {
        const selected = region.id === selectedRegionId;
        return (
          <g key={region.id} className="map-region" onClick={() => region.id !== 'subframe_labyrinth' && onSelectRegion(region.id)}>
            <path d={region.path} fill={overlay === 'climate' ? region.highColor : region.color} stroke={selected ? '#f3e7b0' : '#1c1c22'} strokeWidth={selected ? 4 : 2} opacity={region.id === 'subframe_labyrinth' ? 0.78 : 0.96} />
            {overlay === 'terrain' && <TerrainLines seed={region.id} region={region} />}
            {(overlay === 'travel' || overlay === 'terrain') && region.roads.map((road, index) => <path key={`${region.id}-road-${index}`} d={road} fill="none" stroke="#7c5a31" strokeWidth="4" strokeLinecap="round" strokeDasharray={region.id === 'subframe_labyrinth' ? '8 8' : undefined} />)}
            {region.rivers.map((river, index) => <path key={`${region.id}-river-${index}`} d={river} fill="none" stroke="#65a9c2" strokeWidth="3" strokeLinecap="round" opacity="0.85" />)}
            <text x={region.labelPos[0]} y={region.labelPos[1]} className="map-label" textAnchor="middle">{region.label}</text>
          </g>
        );
      })}

      {overlay === 'resources' && REGION_SHAPES.flatMap((region) => region.resources.map((resource) => (
        <MapMarker key={`${region.id}-${resource.label}`} x={resource.x} y={resource.y} kind={resource.kind} label={resource.label} selected={focus.id === `${region.id}-${resource.label}`} onSelect={() => onFocus(makeFocus('resource', `${region.id}-${resource.label}`, resource.label, `${resource.label} anchors ${region.label}'s ${resource.kind} economy.`, [region.label, resource.kind]))} />
      )))}

      {settlements.map((settlement, index) => (
        <MapMarker key={settlement.id} x={500 + index * 24} y={306 + index * 12} kind="settlement" label={settlement.name} selected={focus.id === settlement.id} onSelect={() => onFocus(makeFocus('settlement', settlement.id, settlement.name, `Population ${settlement.population}. Heartwell stability ${(settlement.resources.heartwellStability * 100).toFixed(0)}%, morale ${(settlement.resources.publicMorale * 100).toFixed(0)}%.`, ['settlement', settlement.regionId]))} />
      ))}

      {environment && overlay === 'pressure' && (
        <g opacity="0.8">
          <circle cx="500" cy="306" r={68 + environment.chemistry.corrosion * 36} fill="none" stroke="#b06a3a" strokeWidth="3" strokeDasharray="9 8" />
          <circle cx="500" cy="306" r={94 + environment.physics.rainfallMm} fill="none" stroke="#65a9c2" strokeWidth="2" opacity="0.65" />
        </g>
      )}
    </svg>
  );
}

function RegionMap({
  shape,
  region,
  overlay,
  settlement,
  quests,
  dungeons,
  environment,
  focus,
  onFocus,
}: {
  shape: RegionShape;
  region?: Region;
  overlay: Overlay;
  settlement?: Settlement;
  quests: Quest[];
  dungeons: DungeonRoom[];
  environment?: EnvironmentState;
  focus: MapFocus;
  onFocus: (focus: MapFocus) => void;
}) {
  const resourceNodes = shape.resources;
  return (
    <svg className="atlas-svg" viewBox="0 0 1000 620" preserveAspectRatio="xMidYMin meet" role="img" aria-label={`${shape.label} region map`}>
      <defs>
        <radialGradient id="regionWash" cx="50%" cy="48%" r="70%">
          <stop offset="0%" stopColor={shape.highColor} stopOpacity="0.78" />
          <stop offset="100%" stopColor={shape.color} stopOpacity="1" />
        </radialGradient>
      </defs>
      <rect width="1000" height="620" fill="#1a1c23" />
      <path d="M88 84 C246 24 770 20 906 118 C972 220 936 486 790 568 C572 604 252 590 92 506 C38 358 34 206 88 84 Z" fill="url(#regionWash)" stroke="#0f1016" strokeWidth="5" />
      {Array.from({ length: 14 }, (_, i) => {
        const y = 110 + i * 34;
        const drift = Math.sin(i * 0.8) * 28;
        return <path key={`contour-${i}`} d={`M126 ${y} C284 ${y - 42 + drift} 438 ${y + 38 - drift} 592 ${y - 8} C724 ${y - 46} 806 ${y + 28} 880 ${y}`} fill="none" stroke="rgba(255,255,255,0.16)" strokeWidth="1.4" />;
      })}
      {shape.rivers.map((river, index) => <path key={`region-river-${index}`} d={river} fill="none" stroke="#74b7cf" strokeWidth="8" strokeLinecap="round" opacity="0.82" />)}
      {shape.roads.map((road, index) => <path key={`region-road-${index}`} d={road} fill="none" stroke="#76532e" strokeWidth="7" strokeLinecap="round" strokeDasharray={overlay === 'pressure' ? '14 8' : undefined} />)}

      {overlay === 'climate' && environment && (
        <g opacity="0.58">
          <circle cx="500" cy="306" r={130 + environment.physics.humidity * 80} fill="#73a9bd" opacity="0.22" />
          <circle cx="650" cy="218" r={80 + environment.physics.windSpeedMps * 8} fill="#d8c479" opacity="0.18" />
        </g>
      )}

      {resourceNodes.map((resource) => <MapMarker key={resource.label} x={resource.x + 18} y={resource.y + 20} kind={resource.kind} label={resource.label} selected={focus.id === `${shape.id}-${resource.label}`} onSelect={() => onFocus(makeFocus('resource', `${shape.id}-${resource.label}`, resource.label, `${resource.label} can support ${shape.label}'s ${resource.kind} supply chain when the roads stay open.`, [resource.kind, shape.label]))} />)}
      {settlement && shape.id === settlement.regionId && <MapMarker x={506} y={318} kind="settlement" label={settlement.name} selected={focus.id === settlement.id} onSelect={() => onFocus(makeFocus('settlement', settlement.id, settlement.name, `Home settlement in ${shape.label}; population ${settlement.population}.`, ['home', 'Hearthwell']))} />}
      {quests.slice(0, 5).map((quest, index) => <MapMarker key={quest.id} x={390 + index * 42} y={370 + (index % 2) * 40} kind="quest" label={quest.title} selected={focus.id === quest.id} onSelect={() => onFocus(makeFocus('quest', quest.id, quest.title, quest.description, [`urgency ${Math.round(quest.urgency * 100)}%`, `difficulty ${Math.round(quest.difficulty * 100)}%`, ...quest.tags.slice(0, 2)]))} />)}
      {dungeons.slice(0, 4).map((room, index) => <MapMarker key={room.id} x={610 + index * 34} y={420 - (index % 2) * 32} kind="dungeon" label={titleCase(room.type)} selected={focus.id === room.id} onSelect={() => onFocus(makeFocus('dungeon', room.id, titleCase(room.type), room.description, [`difficulty ${Math.round(room.difficulty * 100)}%`, room.pressureSource, ...room.tags.slice(0, 2)]))} />)}
      <text x="500" y="72" className="map-title" textAnchor="middle">{shape.label}</text>
      <text x="500" y="98" className="map-subtitle" textAnchor="middle">{region?.climate ?? shape.elevation}</text>
    </svg>
  );
}

function AreaMap({
  overlay,
  settlement,
  npcs,
  quests,
  dungeons,
  environment,
  player,
  focus,
  onFocus,
}: {
  overlay: Overlay;
  settlement?: Settlement;
  npcs: NPCSummary[];
  quests: Quest[];
  dungeons: DungeonRoom[];
  environment?: EnvironmentState;
  player: PlayerState | null;
  focus: MapFocus;
  onFocus: (focus: MapFocus) => void;
}) {
  const npcClusters = useMemo(() => {
    const jobs = ['farmer', 'merchant', 'guard', 'healer', 'crafter', 'adventurer'];
    return jobs.map((job, index) => ({
      job,
      count: npcs.filter((npc) => npc.jobId === job).length,
      x: [315, 674, 772, 520, 608, 250][index]!,
      y: [420, 318, 220, 292, 390, 168][index]!,
    }));
  }, [npcs]);

  return (
    <svg className="atlas-svg" viewBox="0 0 1000 620" preserveAspectRatio="xMidYMin meet" role="img" aria-label={`${settlement?.name ?? 'Vennholt'} area map`}>
      <rect width="1000" height="620" fill="#293b31" />
      <path d="M0 448 C200 412 346 430 510 408 C650 390 790 394 1000 356 L1000 620 L0 620 Z" fill="#526742" />
      <path d="M0 130 C126 96 246 118 342 178 C444 242 562 202 676 150 C792 96 888 116 1000 162 L1000 0 L0 0 Z" fill="#445f4c" />
      <path d="M455 0 C430 126 456 250 404 364 C374 430 324 498 268 620" fill="none" stroke="#6bb3ce" strokeWidth="18" opacity="0.7" />
      <path d="M94 508 C270 420 384 368 512 318 C654 262 746 244 912 212" fill="none" stroke="#76532e" strokeWidth="18" strokeLinecap="round" />
      <path d="M512 318 C536 390 640 432 818 526" fill="none" stroke="#76532e" strokeWidth="12" strokeLinecap="round" opacity="0.88" />
      <path d="M512 318 C452 252 378 210 260 170" fill="none" stroke="#76532e" strokeWidth="10" strokeLinecap="round" opacity="0.82" />

      {AREA_LAYERS.map((area) => (
        <g key={area.id} className="map-district" onClick={() => onFocus(makeFocus('district', area.id, area.label, districtDescription(area.id), [area.id, settlement?.name ?? 'Vennholt']))}>
          <ellipse cx={area.x} cy={area.y} rx={area.w / 2} ry={area.h / 2} fill={area.color} opacity={overlay === 'pressure' && area.id === 'rupture' ? 0.92 : 0.72} stroke="rgba(255,255,255,0.16)" strokeWidth="2" />
          <text x={area.x} y={area.y} className="map-area-label" textAnchor="middle">{area.label}</text>
        </g>
      ))}

      {environment && overlay === 'climate' && (
        <g>
          <circle cx="402" cy="364" r={80 + environment.physics.soilMoisture * 60} fill="#5f9dad" opacity="0.22" />
          <path d={`M120 100 C300 ${120 + environment.physics.windSpeedMps * 4} 620 ${80 - environment.physics.windSpeedMps * 2} 900 132`} fill="none" stroke="#d8e5ea" strokeWidth="5" opacity="0.34" strokeDasharray="20 14" />
        </g>
      )}

      {overlay === 'terrain' && Array.from({ length: 11 }, (_, i) => <path key={`area-contour-${i}`} d={`M90 ${120 + i * 36} C246 ${84 + i * 34} 390 ${156 + i * 18} 530 ${126 + i * 33} C702 ${88 + i * 31} 814 ${156 + i * 24} 930 ${120 + i * 36}`} fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="1.2" />)}

      {overlay === 'resources' && (
        <>
          <MapMarker x={315} y={420} kind="food" label="Wheat" selected={focus.id === 'wheat'} onSelect={() => onFocus(makeFocus('resource', 'wheat', 'Wheat Terraces', 'Primary food reserve. Gather and delivery quests usually depend on this slope.', ['food', 'fields']))} />
          <MapMarker x={518} y={318} kind="hearth" label="Hearthwell" selected={focus.id === 'hearthwell'} onSelect={() => onFocus(makeFocus('resource', 'hearthwell', 'Hearthwell', 'Social gravity and metaphysical stability source for the settlement.', ['stability', 'oaths']))} />
          <MapMarker x={674} y={318} kind="trade" label="Market" selected={focus.id === 'market-resource'} onSelect={() => onFocus(makeFocus('resource', 'market-resource', 'Market Row', 'Coin, trade, rumor flow, and food exchange concentrate here.', ['trade', 'coin']))} />
          <MapMarker x={250} y={168} kind="dungeon" label="Rupture" selected={focus.id === 'rupture-resource'} onSelect={() => onFocus(makeFocus('dungeon', 'rupture-resource', 'Subframe Rupture', 'A pressure leak below the civic layer. Delves reduce fear but cost health and stamina.', ['pressure', 'dungeon']))} />
        </>
      )}

      {npcClusters.map((cluster) => cluster.count > 0 && <MapMarker key={cluster.job} x={cluster.x} y={cluster.y} kind="resident" label={`${titleCase(cluster.job)} x${cluster.count}`} selected={focus.id === `residents-${cluster.job}`} onSelect={() => onFocus(makeFocus('resident', `residents-${cluster.job}`, `${titleCase(cluster.job)} cluster`, `${cluster.count} residents are currently represented in this profession group.`, [cluster.job, 'residents']))} />)}
      {quests.slice(0, 4).map((quest, index) => <MapMarker key={quest.id} x={578 + index * 36} y={250 + (index % 2) * 34} kind="quest" label={quest.title} selected={focus.id === quest.id} onSelect={() => onFocus(makeFocus('quest', quest.id, quest.title, quest.description, [`urgency ${Math.round(quest.urgency * 100)}%`, `difficulty ${Math.round(quest.difficulty * 100)}%`, ...quest.tags.slice(0, 2)]))} />)}
      {dungeons.slice(0, 3).map((room, index) => <MapMarker key={room.id} x={250 + index * 28} y={168 + index * 24} kind="dungeon" label={titleCase(room.type)} selected={focus.id === room.id} onSelect={() => onFocus(makeFocus('dungeon', room.id, titleCase(room.type), room.description, [`difficulty ${Math.round(room.difficulty * 100)}%`, room.pressureSource]))} />)}
      {player && <MapMarker x={PLAYER_AREA_POS[player.location][0]} y={PLAYER_AREA_POS[player.location][1] - 34} kind="player" label="You" selected={focus.kind === 'player'} onSelect={() => onFocus(makeFocus('player', player.id, player.name, `Currently in ${titleCase(player.location)} with ${Math.round(player.stamina * 100)}% stamina and ${Math.round(player.health * 100)}% health.`, [`level ${player.level}`, `${player.wealth.toFixed(0)} coin`]))} />}

      <text x="500" y="44" className="map-title" textAnchor="middle">{settlement?.name ?? 'Vennholt'} Area</text>
      <text x="500" y="70" className="map-subtitle" textAnchor="middle">roads, districts, residents, resources, and active pressures</text>
    </svg>
  );
}

function TerrainLines({ region, seed }: { region: RegionShape; seed: string }) {
  return (
    <g opacity="0.45">
      {Array.from({ length: 4 }, (_, i) => {
        const y = region.labelPos[1] - 42 + i * 22;
        const x = region.labelPos[0];
        const spread = 56 + i * 24 + seed.length * 2;
        return <ellipse key={`${seed}-${i}`} cx={x} cy={y} rx={spread} ry={18 + i * 7} fill="none" stroke="rgba(255,255,255,0.22)" strokeWidth="1.2" />;
      })}
    </g>
  );
}

function MapMarker({ x, y, kind, label, selected = false, onSelect }: { x: number; y: number; kind: string; label: string; selected?: boolean; onSelect?: () => void }) {
  const color = markerColor(kind);
  return (
    <g className={onSelect ? selected ? 'map-marker interactive selected' : 'map-marker interactive' : 'map-marker'} transform={`translate(${x} ${y})`} onClick={(event) => { event.stopPropagation(); onSelect?.(); }}>
      {kind === 'player' && <circle r="20" fill="none" stroke={color} strokeWidth="2" opacity="0.45" className="map-player-pulse" />}
      {onSelect && <circle r="18" fill="transparent" />}
      <circle r={selected ? 13 : 10} fill={color} stroke={selected ? '#f3e7b0' : '#101016'} strokeWidth="3" />
      <circle r="4" fill="#fff" opacity="0.8" />
      <text x="15" y="4" className="map-marker-label">{label}</text>
    </g>
  );
}

function markerColor(kind: string): string {
  if (kind === 'food') return '#d1b866';
  if (kind === 'wood') return '#4f7d56';
  if (kind === 'ore' || kind === 'tools') return '#8b8d93';
  if (kind === 'medicine') return '#5c8c6c';
  if (kind === 'dungeon') return '#7a6eff';
  if (kind === 'trade' || kind === 'coin') return '#c89c4b';
  if (kind === 'hearth' || kind === 'settlement') return '#ffcf74';
  if (kind === 'quest') return '#c85c5c';
  if (kind === 'player') return '#ffffff';
  return '#4fa8c8';
}

function districtDescription(id: string): string {
  if (id === 'plaza') return 'Public oath space, civic announcements, and the visible mouth of the Hearthwell.';
  if (id === 'market') return 'Trade, gossip, food exchange, and coin pressure gather here.';
  if (id === 'fields') return 'Food production terrace; poor weather and soil chemistry show up here first.';
  if (id === 'homes') return 'Household ring where family needs, reputation, and rest cycles are centered.';
  if (id === 'grove') return 'Wild edge for gathering, herbs, and encounters beyond town certainty.';
  if (id === 'watch') return 'Guard perimeter and early warning line against dungeon pressure.';
  return 'A subframe leak where memory, fear, and physical pressure distort the civic map.';
}

function computeRouteReadiness(mode: MapMode, overlay: Overlay, environment?: EnvironmentState, player?: PlayerState | null, selectedShape?: RegionShape) {
  const stamina = player?.stamina ?? 0.65;
  const health = player?.health ?? 0.8;
  const rain = environment?.physics.rainfallMm ?? 0;
  const wind = environment?.physics.windSpeedMps ?? 0;
  const corrosion = environment?.chemistry.corrosion ?? 0;
  const modeRisk = mode === 'area' ? 0.08 : mode === 'region' ? 0.18 : 0.28;
  const overlayRisk = overlay === 'pressure' ? 0.18 : overlay === 'climate' ? 0.1 : 0.04;
  const weatherRisk = Math.min(0.28, rain / 28 + wind / 80 + corrosion * 0.22);
  const readiness = Math.max(0.08, Math.min(1, stamina * 0.58 + health * 0.32 + 0.22 - modeRisk - overlayRisk - weatherRisk));
  const label = readiness > 0.68 ? 'Clear Route' : readiness > 0.38 ? 'Cautious Route' : 'Danger Route';
  const color = readiness > 0.68 ? '#5c8c6c' : readiness > 0.38 ? '#c8a84b' : '#c85c5c';
  const place = mode === 'world' ? 'regional crossing' : mode === 'region' ? selectedShape?.label ?? 'region' : 'local district';
  const note = readiness > 0.68
    ? `Good conditions for a ${place} move. Use the map actions to travel, gather, trade, or scout.`
    : readiness > 0.38
      ? `The ${place} is workable, but weather and stamina can turn small errands into pressure.`
      : `Poor expedition window. Rest or stay near town before pushing deeper.`;
  return { readiness, label, color, note };
}

function MapStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="map-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div className="legend-item">
      <span style={{ background: color }} />
      {label}
    </div>
  );
}
