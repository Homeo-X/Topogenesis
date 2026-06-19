import { useState } from 'react';
import { Layout } from './components/Layout.js';
import { WorldDashboard } from './components/WorldDashboard.js';
import { NPCList } from './components/NPCProfile.js';
import { QuestBoard } from './components/QuestBoard.js';
import { SettlementEconomy } from './components/SettlementEconomy.js';
import { HouseholdPanel } from './components/HouseholdPanel.js';
import { LoreCodex } from './components/LoreCodex.js';
import { SimulationControls } from './components/SimulationControls.js';
import { DungeonView } from './components/DungeonView.js';

export function App() {
  const [view, setView] = useState('dashboard');

  return (
    <Layout activeView={view} onNavigate={setView}>
      {view === 'dashboard' && <WorldDashboard />}
      {view === 'npcs' && <NPCList />}
      {view === 'quests' && <QuestBoard />}
      {view === 'settlement' && <SettlementEconomy />}
      {view === 'households' && <HouseholdPanel />}
      {view === 'events' && <SimulationControls />}
      {view === 'lore' && <LoreCodex />}
      {view === 'dungeons' && <DungeonView />}
    </Layout>
  );
}
