import { useState, useEffect } from 'react';
import { api, type Quest } from '../api.js';

export function QuestBoard() {
  const [quests, setQuests] = useState<Quest[]>([]);
  const [selected, setSelected] = useState<Quest | null>(null);

  useEffect(() => { api.getQuests().then(setQuests).catch(() => {}); }, []);

  const urgentQuests = quests.filter(q => q.urgency >= 0.7);
  const otherQuests = quests.filter(q => q.urgency < 0.7);

  return (
    <div className="quest-board">
      <h2>Town Board — Active Quests ({quests.length})</h2>
      {quests.length === 0 && <div className="empty">The board is bare — the settlement seems stable... for now.</div>}

      {urgentQuests.length > 0 && <h3 className="urgent-header">⚠ Urgent</h3>}
      {urgentQuests.map(q => <QuestCard key={q.id} quest={q} onClick={() => setSelected(q)} />)}

      {otherQuests.length > 0 && <h3 className="normal-header">Current Notices</h3>}
      {otherQuests.map(q => <QuestCard key={q.id} quest={q} onClick={() => setSelected(q)} />)}

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
              <div key={o.description} className="objective">{o.completed ? '✓' : '○'} {o.description}</div>
            ))}
            <div className="quest-tags">{selected.tags.map(t => <span key={t} className="tag">{t}</span>)}</div>
            <button onClick={() => setSelected(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}

function QuestCard({ quest, onClick }: { quest: Quest; onClick: () => void }) {
  const urgencyClass = quest.urgency >= 0.8 ? 'quest-critical' : quest.urgency >= 0.6 ? 'quest-high' : 'quest-normal';
  return (
    <div className={`quest-card ${urgencyClass}`} onClick={onClick}>
      <div className="quest-title">{quest.title}</div>
      <div className="quest-preview">{quest.description.slice(0, 80)}...</div>
      <div className="quest-meta">
        <span>Day {quest.generatedOnDay}</span>
        <span>Urgency {Math.round(quest.urgency * 100)}%</span>
        {quest.tags.slice(0, 3).map(t => <span key={t} className="tag">{t}</span>)}
      </div>
    </div>
  );
}
