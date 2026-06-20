import { useEffect, useState } from 'react';
import { api, type NPCSummary, type PlayerActionInput, type PlayerState, type Quest } from '../api.js';

type Props = {
  nearbyNPCs?: NPCSummary[];
  onWorldChanged?: () => void;
};

const ACTIONS: { label: string; type: PlayerActionInput['type']; location?: PlayerActionInput['location'] }[] = [
  { label: 'Town', type: 'travel', location: 'town' },
  { label: 'Market', type: 'travel', location: 'market' },
  { label: 'Wilds', type: 'travel', location: 'wilds' },
  { label: 'Rest', type: 'rest' },
  { label: 'Gather', type: 'gather' },
  { label: 'Train', type: 'train' },
  { label: 'Trade', type: 'trade' },
  { label: 'Dungeon', type: 'delve_dungeon' },
];

export function PlayerHUD({ nearbyNPCs = [], onWorldChanged }: Props) {
  const [player, setPlayer] = useState<PlayerState | null>(null);
  const [quests, setQuests] = useState<Quest[]>([]);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);

  async function load() {
    const [nextPlayer, nextQuests] = await Promise.all([api.getPlayer(), api.getQuests()]);
    setPlayer(nextPlayer);
    setQuests(nextQuests);
  }

  useEffect(() => { load().catch(() => {}); }, []);

  async function runAction(action: PlayerActionInput) {
    setBusy(true);
    setMessage('');
    try {
      const result = await api.playerAction(action);
      setPlayer(result.player);
      setMessage(result.message);
      setQuests(await api.getQuests());
      onWorldChanged?.();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  if (!player) return null;

  const acceptedQuestIds = new Set(player.questLog.filter(q => q.status === 'accepted').map(q => q.questId));
  const activeQuest = quests.find(q => acceptedQuestIds.has(q.id));
  const openQuest = quests.find(q => !q.acceptedByPlayerId);
  const activeQuestReady = activeQuest?.objectives.every(o => o.completed) ?? false;
  const rep = Math.round((player.reputation[player.settlementId] ?? 0) * 100);
  const nextNpc = nearbyNPCs[0];

  return (
    <div className="player-hud">
      <div className="player-hud-main">
        <div>
          <div className="player-name">{player.name}</div>
          <div className="player-sub">Lv {player.level} {player.location} - {player.wealth.toFixed(0)} coin - Rep {rep}%</div>
        </div>
        <div className="player-bars">
          <HudBar label="HP" value={player.health} color="#c85c5c" />
          <HudBar label="STA" value={player.stamina} color="#5c8c6c" />
          <HudBar label="XP" value={(player.experience % 100) / 100} color="#7a6eff" />
        </div>
      </div>

      <div className="player-actions">
        {ACTIONS.map(action => (
          <button
            key={`${action.type}-${action.location ?? 'self'}`}
            disabled={busy}
            onClick={() => runAction({ type: action.type, location: action.location })}
          >
            {action.label}
          </button>
        ))}
        <button disabled={busy || !nextNpc} onClick={() => nextNpc && runAction({ type: 'talk', npcId: nextNpc.id })}>
          Talk
        </button>
      </div>

      <div className="player-quest-strip">
        {activeQuest ? (
          <>
            <span>{activeQuest.title} ({activeQuest.objectives.filter(o => o.completed).length}/{activeQuest.objectives.length})</span>
            <button disabled={busy || !activeQuestReady} onClick={() => runAction({ type: 'complete_quest', questId: activeQuest.id })}>Complete</button>
          </>
        ) : openQuest ? (
          <>
            <span>{openQuest.title}</span>
            <button disabled={busy} onClick={() => runAction({ type: 'accept_quest', questId: openQuest.id })}>Accept</button>
          </>
        ) : (
          <span>No active quest. Let the world run or aid the town.</span>
        )}
      </div>

      <div className="player-inventory">
        {player.inventory.length === 0 ? <span>Inventory empty</span> : player.inventory.map(item => (
          <span key={item.itemId}>{item.itemId.replace(/_/g, ' ')} x{item.quantity}</span>
        ))}
      </div>

      {message && <div className="player-message">{message}</div>}
      <div className="player-log">
        {player.actionLog.slice(0, 3).map((entry, index) => <span key={`${index}-${entry}`}>{entry}</span>)}
      </div>
    </div>
  );
}

function HudBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div className="hud-stat-bar">
      <span>{label}</span>
      <div className="hud-stat-track"><div className="hud-stat-fill" style={{ width: `${pct}%`, background: color }} /></div>
      <span>{pct}%</span>
    </div>
  );
}
