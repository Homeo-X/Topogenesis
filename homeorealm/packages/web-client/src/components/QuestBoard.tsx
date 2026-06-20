import { useState, useEffect } from 'react';
import { api, type PlayerState, type Quest } from '../api.js';

export function QuestBoard() {
  const [quests, setQuests] = useState<Quest[]>([]);
  const [selected, setSelected] = useState<Quest | null>(null);
  const [player, setPlayer] = useState<PlayerState | null>(null);
  const [message, setMessage] = useState('');
  const [busyQuestId, setBusyQuestId] = useState<string | null>(null);

  async function load() {
    const [nextQuests, nextPlayer] = await Promise.all([api.getQuests(), api.getPlayer()]);
    setQuests(nextQuests);
    setPlayer(nextPlayer);
    if (selected) setSelected(nextQuests.find(q => q.id === selected.id) ?? null);
  }

  useEffect(() => { load().catch(() => {}); }, []);

  async function runQuestAction(quest: Quest, type: 'accept_quest' | 'complete_quest') {
    setBusyQuestId(quest.id);
    setMessage('');
    try {
      const result = await api.playerAction({ type, questId: quest.id });
      setPlayer(result.player);
      setMessage(result.message);
      const nextQuests = await api.getQuests();
      setQuests(nextQuests);
      setSelected(nextQuests.find(q => q.id === quest.id) ?? null);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyQuestId(null);
    }
  }

  const urgentQuests = quests.filter(q => q.urgency >= 0.7);
  const otherQuests = quests.filter(q => q.urgency < 0.7);
  const activeRecords = new Set(player?.questLog.filter(q => q.status === 'accepted').map(q => q.questId) ?? []);
  const completedCount = player?.questLog.filter(q => q.status === 'completed').length ?? 0;
  const settlementReputation = player ? player.reputation[player.settlementId] ?? 0 : 0;

  return (
    <div className="quest-board">
      <h2>Town Board - Active Quests ({quests.length})</h2>
      {player && (
        <div className="player-summary">
          <div>
            <strong>{player.name}</strong>
            <span>Coin {player.wealth.toFixed(0)}</span>
            <span>Reputation {Math.round(settlementReputation * 100)}%</span>
            <span>Completed {completedCount}</span>
          </div>
          {message && <p>{message}</p>}
        </div>
      )}
      {quests.length === 0 && <div className="empty">The board is bare. The settlement seems stable for now.</div>}

      {urgentQuests.length > 0 && <h3 className="urgent-header">Urgent</h3>}
      {urgentQuests.map(q => (
        <QuestCard
          key={q.id}
          quest={q}
          accepted={activeRecords.has(q.id)}
          busy={busyQuestId === q.id}
          onClick={() => setSelected(q)}
          onAccept={() => runQuestAction(q, 'accept_quest')}
          onComplete={() => runQuestAction(q, 'complete_quest')}
        />
      ))}

      {otherQuests.length > 0 && <h3 className="normal-header">Current Notices</h3>}
      {otherQuests.map(q => (
        <QuestCard
          key={q.id}
          quest={q}
          accepted={activeRecords.has(q.id)}
          busy={busyQuestId === q.id}
          onClick={() => setSelected(q)}
          onAccept={() => runQuestAction(q, 'accept_quest')}
          onComplete={() => runQuestAction(q, 'complete_quest')}
        />
      ))}

      {selected && (
        <div className="quest-modal" onClick={() => setSelected(null)}>
          <div className="quest-detail" onClick={e => e.stopPropagation()}>
            <h3>{selected.title}</h3>
            <div className="quest-cause">Origin: <em>{selected.cause.replace(/_/g, ' ')}</em></div>
            <p>{selected.description}</p>
            <div className="quest-stats">
              <span>Urgency {Math.round(selected.urgency * 100)}%</span>
              <span>Difficulty {Math.round(selected.difficulty * 100)}%</span>
              <span>Day {selected.generatedOnDay}</span>
            </div>
            <h4>Objectives</h4>
            {selected.objectives.map(o => (
              <div key={o.description} className="objective">{o.completed ? 'Done' : 'Open'} - {o.description}</div>
            ))}
            <div className="quest-tags">{selected.tags.map(t => <span key={t} className="tag">{t}</span>)}</div>
            <QuestActions
              quest={selected}
              accepted={activeRecords.has(selected.id)}
              busy={busyQuestId === selected.id}
              onAccept={() => runQuestAction(selected, 'accept_quest')}
              onComplete={() => runQuestAction(selected, 'complete_quest')}
            />
            <button onClick={() => setSelected(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}

function QuestCard({
  quest,
  accepted,
  busy,
  onClick,
  onAccept,
  onComplete,
}: {
  quest: Quest;
  accepted: boolean;
  busy: boolean;
  onClick: () => void;
  onAccept: () => void;
  onComplete: () => void;
}) {
  const urgencyClass = quest.urgency >= 0.8 ? 'quest-critical' : quest.urgency >= 0.6 ? 'quest-high' : 'quest-normal';
  return (
    <div className={`quest-card ${urgencyClass}`} onClick={onClick}>
      <div className="quest-title">{quest.title}</div>
      <div className="quest-preview">{quest.description.slice(0, 80)}...</div>
      <div className="quest-meta">
        <span>Day {quest.generatedOnDay}</span>
        <span>Urgency {Math.round(quest.urgency * 100)}%</span>
        <span>{quest.objectives.filter(o => o.completed).length}/{quest.objectives.length} objectives</span>
        {accepted && <span className="tag">accepted</span>}
        {quest.tags.slice(0, 3).map(t => <span key={t} className="tag">{t}</span>)}
      </div>
      <QuestActions
        quest={quest}
        accepted={accepted}
        busy={busy}
        onAccept={onAccept}
        onComplete={onComplete}
      />
    </div>
  );
}

function QuestActions({
  quest,
  accepted,
  busy,
  onAccept,
  onComplete,
}: {
  quest: Quest;
  accepted: boolean;
  busy: boolean;
  onAccept: () => void;
  onComplete: () => void;
}) {
  return (
    <div className="quest-actions" onClick={e => e.stopPropagation()}>
      {!accepted && (
        <button className="btn-secondary" disabled={busy || !!quest.acceptedByPlayerId} onClick={onAccept}>
          {busy ? 'Working...' : 'Accept'}
        </button>
      )}
      {accepted && (
        <button className="btn-primary" disabled={busy || !quest.objectives.every(o => o.completed)} onClick={onComplete}>
          {busy ? 'Working...' : 'Complete'}
        </button>
      )}
    </div>
  );
}
