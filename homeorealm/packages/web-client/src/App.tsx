import { lazy, Suspense, useState } from 'react';
import { Layout } from './components/Layout.js';

const WorldDashboard  = lazy(() => import('./components/WorldDashboard.js').then(m => ({ default: m.WorldDashboard })));
const NPCList         = lazy(() => import('./components/NPCProfile.js').then(m => ({ default: m.NPCList })));
const QuestBoard      = lazy(() => import('./components/QuestBoard.js').then(m => ({ default: m.QuestBoard })));
const SettlementEconomy = lazy(() => import('./components/SettlementEconomy.js').then(m => ({ default: m.SettlementEconomy })));
const HouseholdPanel  = lazy(() => import('./components/HouseholdPanel.js').then(m => ({ default: m.HouseholdPanel })));
const SimulationControls = lazy(() => import('./components/SimulationControls.js').then(m => ({ default: m.SimulationControls })));
const LoreCodex       = lazy(() => import('./components/LoreCodex.js').then(m => ({ default: m.LoreCodex })));
const DungeonView     = lazy(() => import('./components/DungeonView.js').then(m => ({ default: m.DungeonView })));
const WorldView3D     = lazy(() => import('./components/WorldView3D.js').then(m => ({ default: m.WorldView3D })));

function LoadingFallback() {
  return <div className="suspense-loading">Loading...</div>;
}

export function App() {
  const [view, setView] = useState('dashboard');

  return (
    <Layout activeView={view} onNavigate={setView}>
      <Suspense fallback={<LoadingFallback />}>
        {view === 'dashboard'   && <WorldDashboard />}
        {view === 'npcs'        && <NPCList />}
        {view === 'quests'      && <QuestBoard />}
        {view === 'settlement'  && <SettlementEconomy />}
        {view === 'households'  && <HouseholdPanel />}
        {view === 'events'      && <SimulationControls />}
        {view === 'lore'        && <LoreCodex />}
        {view === 'dungeons'    && <DungeonView />}
        {view === 'world3d'     && <WorldView3D />}
      </Suspense>
    </Layout>
  );
}
