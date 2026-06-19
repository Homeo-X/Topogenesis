import { useState, useEffect, useMemo, useCallback } from 'react';
import { api, type NPCSummary, type NPCFull, type Relationship, type Memory } from '../api.js';
import { RelationshipGraph } from './RelationshipGraph.js';

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(() => typeof window !== 'undefined' && window.innerWidth <= 640);
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 640px)');
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);
  return isMobile;
}

export function NPCList() {
  const [npcs, setNpcs] = useState<NPCSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState('');
  const isMobile = useIsMobile();

  useEffect(() => { api.getNPCs().then(setNpcs).catch(() => {}); }, []);

  const filtered = useMemo(() =>
    npcs.filter(n =>
      !filter ||
      n.name.toLowerCase().includes(filter.toLowerCase()) ||
      (n.jobId ?? '').toLowerCase().includes(filter.toLowerCase()),
    ),
    [npcs, filter],
  );

  const handleSelect = useCallback((id: string) => setSelected(id), []);

  return (
    <div className={`npc-panel${isMobile && selected ? ' mobile-detail-open' : ''}`}>
      <div className="npc-list-col">
        <h2>Residents ({npcs.length})</h2>
        <input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Filter by name or job..." className="filter-input" />
        <div className="npc-list">
          {filtered.map(n => (
            <div
              key={n.id}
              className={`npc-row ${selected === n.id ? 'selected' : ''}`}
              onClick={() => handleSelect(n.id)}
            >
              <span className="npc-name">{n.name}</span>
              <span className="npc-job">{n.jobId ?? '—'}</span>
              <ViabilityPip value={n.viability} />
            </div>
          ))}
        </div>
      </div>
      <div className="npc-detail-col">
        {isMobile && selected && (
          <button className="mobile-back-btn" onClick={() => setSelected(null)}>← Back to residents</button>
        )}
        {selected
          ? <NPCDetail id={selected} />
          : <div className="placeholder">Select a resident to view their biography.</div>
        }
      </div>
    </div>
  );
}

function NPCDetail({ id }: { id: string }) {
  const [npc, setNpc] = useState<NPCFull | null>(null);
  const [rels, setRels] = useState<Relationship[]>([]);
  const [mems, setMems] = useState<Memory[]>([]);
  const [tab, setTab] = useState<'overview' | 'relationships' | 'memories' | 'frame'>('overview');

  useEffect(() => {
    setNpc(null);
    api.getNPC(id).then(setNpc).catch(() => {});
    api.getNPCRelationships(id).then(setRels).catch(() => {});
    api.getNPCMemories(id).then(setMems).catch(() => {});
  }, [id]);

  if (!npc) return <div className="loading">Loading biography...</div>;

  return (
    <div className="npc-detail">
      <h3>{npc.name} <span className="npc-people">({npc.people})</span></h3>
      <div className="npc-meta">
        <span>Job: {npc.jobId ?? '—'}</span>
        <span>Viability: {(npc.viability * 100).toFixed(0)}%</span>
        <span>Health: {(npc.health * 100).toFixed(0)}%</span>
        <span>Wealth: {npc.wealth.toFixed(0)} coin</span>
      </div>
      {npc.currentIntention && (
        <div className="intention">Current intention: <em>{npc.currentIntention.action}</em> — "{npc.currentIntention.motivation}"</div>
      )}

      <div className="tabs">
        {(['overview', 'relationships', 'memories', 'frame'] as const).map(t => (
          <button key={t} className={tab === t ? 'tab active' : 'tab'} onClick={() => setTab(t)}>{capitalize(t)}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <div className="overview-grid">
          <div>
            <h4>Needs</h4>
            {Object.entries(npc.needs).map(([k, v]) => <NeedBar key={k} label={k} value={v as number} />)}
          </div>
          <div>
            <h4>Affect</h4>
            {Object.entries(npc.affect).map(([k, v]) => (
              <div key={k} className="affect-row">
                <span>{k}</span><span>{(v as number).toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div>
            <h4>Skills</h4>
            {Object.entries(npc.skills).filter(([, v]) => (v as number) > 0.05).map(([k, v]) => (
              <NeedBar key={k} label={k} value={v as number} />
            ))}
          </div>
        </div>
      )}

      {tab === 'relationships' && (
        <div>
          <RelationshipGraph npcId={id} npcName={npc.name} relationships={rels} />
          <div className="rel-list">
            {rels.length === 0 && <div className="empty">No relationships yet.</div>}
            {rels.map(r => (
              <div key={r.targetId} className="rel-row">
                <span className="rel-name">{r.targetName}</span>
                <span className="rel-type">{r.kinshipType}</span>
                <span>Affection {(r.affection * 100).toFixed(0)}%</span>
                <span>Trust {(r.trust * 100).toFixed(0)}%</span>
                {r.conflict > 0.3 && <span className="conflict-flag">⚠ Conflict</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === 'memories' && (
        <div className="mem-list">
          {mems.length === 0 && <div className="empty">No memories yet.</div>}
          {mems.slice(0, 15).map(m => (
            <div key={m.id} className={`mem-row ${m.emotionalValence > 0.3 ? 'positive' : m.emotionalValence < -0.3 ? 'negative' : ''}`}>
              <div className="mem-summary">{m.summary}</div>
              <div className="mem-meta">Day {m.day} · Salience {(m.salience * 100).toFixed(0)}% · {m.eventType}</div>
            </div>
          ))}
        </div>
      )}

      {tab === 'frame' && (
        <div className="frame-panel">
          <h4>Frame Distinctions (Topogenesis)</h4>
          {npc.frame.distinctions.length === 0 && <div className="empty">No distinctions formed yet.</div>}
          {npc.frame.distinctions.map(d => (
            <div key={d.id} className="distinction-row">
              <div className="dist-label">{d.label}</div>
              <div className="dist-meta">Domain: {d.domain} · Confidence: {(d.confidence * 100).toFixed(0)}% · Formed Day {d.createdOnDay}</div>
            </div>
          ))}
          {npc.frame.valuedTags.length > 0 && <div><strong>Valued:</strong> {npc.frame.valuedTags.join(', ')}</div>}
          {npc.frame.tabooTags.length > 0 && <div><strong>Taboo:</strong> {npc.frame.tabooTags.join(', ')}</div>}
        </div>
      )}
    </div>
  );
}

function NeedBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  const color = pct > 60 ? '#5c8c6c' : pct > 30 ? '#c8a84b' : '#c85c5c';
  return (
    <div className="need-bar">
      <span className="need-label">{label}</span>
      <div className="bar-track sm">
        <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="need-val">{pct}%</span>
    </div>
  );
}

function ViabilityPip({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct > 60 ? '#5c8c6c' : pct > 30 ? '#c8a84b' : '#c85c5c';
  return <span className="viability-pip" style={{ color }}>{pct}%</span>;
}

function capitalize(s: string): string { return s.charAt(0).toUpperCase() + s.slice(1); }
