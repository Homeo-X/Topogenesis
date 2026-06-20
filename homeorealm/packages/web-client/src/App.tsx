import { lazy, Suspense, useEffect, useState } from 'react';
import { Layout } from './components/Layout.js';
import { api } from './api.js';

const WorldDashboard  = lazy(() => import('./components/WorldDashboard.js').then(m => ({ default: m.WorldDashboard })));
const NPCList         = lazy(() => import('./components/NPCProfile.js').then(m => ({ default: m.NPCList })));
const QuestBoard      = lazy(() => import('./components/QuestBoard.js').then(m => ({ default: m.QuestBoard })));
const SettlementEconomy = lazy(() => import('./components/SettlementEconomy.js').then(m => ({ default: m.SettlementEconomy })));
const HouseholdPanel  = lazy(() => import('./components/HouseholdPanel.js').then(m => ({ default: m.HouseholdPanel })));
const SimulationControls = lazy(() => import('./components/SimulationControls.js').then(m => ({ default: m.SimulationControls })));
const LoreCodex       = lazy(() => import('./components/LoreCodex.js').then(m => ({ default: m.LoreCodex })));
const DungeonView     = lazy(() => import('./components/DungeonView.js').then(m => ({ default: m.DungeonView })));
const WorldView3D     = lazy(() => import('./components/WorldView3D.js').then(m => ({ default: m.WorldView3D })));
const MapAtlas        = lazy(() => import('./components/MapAtlas.js').then(m => ({ default: m.MapAtlas })));

function LoadingFallback() {
  return <div className="suspense-loading">Loading...</div>;
}

export function App() {
  const [view, setView] = useState('world3d');
  const [worldDay, setWorldDay] = useState(0);
  const [autoRun, setAutoRun] = useState(false);
  const [speedMs, setSpeedMs] = useState(3000);
  const [refreshKey, setRefreshKey] = useState(0);
  const [clockBusy, setClockBusy] = useState(false);

  useEffect(() => {
    api.getWorld().then(w => setWorldDay(w.day)).catch(() => {});
  }, []);

  useEffect(() => {
    if (!autoRun) return;

    let cancelled = false;
    const intervalId = window.setInterval(async () => {
      if (cancelled) return;
      setClockBusy(true);
      try {
        const result = await api.tick();
        if (cancelled) return;
        setWorldDay(result.day);
        setRefreshKey(k => k + 1);
      } catch {
        if (!cancelled) setAutoRun(false);
      } finally {
        if (!cancelled) setClockBusy(false);
      }
    }, speedMs);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [autoRun, speedMs]);

  function handleManualRefresh() {
    api.getWorld().then(w => {
      setWorldDay(w.day);
      setRefreshKey(k => k + 1);
    }).catch(() => {});
  }

  return (
    <Layout
      activeView={view}
      onNavigate={setView}
      worldDay={worldDay}
      autoRun={autoRun}
      speedMs={speedMs}
      clockBusy={clockBusy}
      onToggleAutoRun={() => setAutoRun(r => !r)}
      onSpeedChange={setSpeedMs}
    >
      <Suspense fallback={<LoadingFallback />}>
        {view === 'dashboard'   && <WorldDashboard key={`dashboard-${refreshKey}`} onTick={handleManualRefresh} />}
        {view === 'npcs'        && <NPCList key={`npcs-${refreshKey}`} />}
        {view === 'quests'      && <QuestBoard key={`quests-${refreshKey}`} />}
        {view === 'settlement'  && <SettlementEconomy key={`settlement-${refreshKey}`} />}
        {view === 'households'  && <HouseholdPanel key={`households-${refreshKey}`} />}
        {view === 'events'      && <SimulationControls key={`events-${refreshKey}`} onTick={handleManualRefresh} />}
        {view === 'lore'        && <LoreCodex />}
        {view === 'dungeons'    && <DungeonView key={`dungeons-${refreshKey}`} />}
        {view === 'world3d'     && <WorldView3D key={`world3d-${refreshKey}`} />}
        {view === 'maps'        && <MapAtlas key={`maps-${refreshKey}`} />}
      </Suspense>
    </Layout>
  );
}
