import { useState, useEffect, type ReactNode } from 'react';

type NavItem = { label: string; view: string };

const ALL_NAV: NavItem[] = [
  { label: 'Dashboard',  view: 'dashboard' },
  { label: '3D World',   view: 'world3d' },
  { label: 'Maps',       view: 'maps' },
  { label: 'NPCs',       view: 'npcs' },
  { label: 'Quests',     view: 'quests' },
  { label: 'Settlement', view: 'settlement' },
  { label: 'Households', view: 'households' },
  { label: 'Events',     view: 'events' },
  { label: 'Lore Codex', view: 'lore' },
  { label: 'Dungeons',   view: 'dungeons' },
];

// First 4 items appear in the mobile bottom tab bar
const PRIMARY_TABS = ALL_NAV.slice(0, 4);
// Everything else goes in the More drawer
const SECONDARY_ITEMS = ALL_NAV.slice(4);

const SPEED_OPTIONS = [
  { label: 'Calm',   ms: 5000 },
  { label: 'Normal', ms: 3000 },
  { label: 'Fast',   ms: 1200 },
];

type Props = {
  activeView: string;
  onNavigate: (v: string) => void;
  children: ReactNode;
  worldDay: number;
  autoRun: boolean;
  speedMs: number;
  clockBusy: boolean;
  onToggleAutoRun: () => void;
  onSpeedChange: (ms: number) => void;
};

export function Layout({
  activeView, onNavigate, children,
  worldDay, autoRun, speedMs, clockBusy,
  onToggleAutoRun, onSpeedChange,
}: Props) {
  const [moreOpen, setMoreOpen] = useState(false);
  const isSecondaryActive = SECONDARY_ITEMS.some(t => t.view === activeView);

  function handleNav(v: string) {
    onNavigate(v);
    setMoreOpen(false);
  }

  useEffect(() => {
    if (!moreOpen) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') setMoreOpen(false); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [moreOpen]);

  useEffect(() => {
    document.body.style.overflow = moreOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [moreOpen]);

  return (
    <div className="layout">
      <header className="top-bar">
        <div className="logo">⬡ HomeoRealm</div>

        {/* Desktop navigation */}
        <nav className="desktop-nav">
          {ALL_NAV.map(item => (
            <button
              key={item.view}
              className={activeView === item.view ? 'nav-btn active' : 'nav-btn'}
              onClick={() => handleNav(item.view)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        {/* World clock — visible on desktop + compact on mobile */}
        <div className="world-clock" aria-label="World clock">
          <span className="clock-day">Day {worldDay}</span>
          <button
            className={autoRun ? 'clock-toggle running' : 'clock-toggle'}
            onClick={onToggleAutoRun}
          >
            {autoRun ? 'Pause' : 'Auto'}
          </button>
          <select
            className="clock-speed"
            value={speedMs}
            onChange={e => onSpeedChange(Number(e.target.value))}
            aria-label="Simulation speed"
          >
            {SPEED_OPTIONS.map(opt => (
              <option key={opt.ms} value={opt.ms}>{opt.label}</option>
            ))}
          </select>
          {clockBusy && <span className="clock-pulse" aria-label="Advancing" />}
        </div>
      </header>

      <main className="content">{children}</main>

      {/* ── Mobile bottom tab bar ── */}
      <nav className="bottom-nav" aria-label="Main navigation">
        {PRIMARY_TABS.map(item => (
          <button
            key={item.view}
            className={activeView === item.view ? 'bottom-tab active' : 'bottom-tab'}
            onClick={() => handleNav(item.view)}
          >
            <span className="bottom-tab-icon"><TabIcon view={item.view} /></span>
            <span className="bottom-tab-label">{item.label}</span>
          </button>
        ))}
        <button
          className={isSecondaryActive || moreOpen ? 'bottom-tab active' : 'bottom-tab'}
          onClick={() => setMoreOpen(o => !o)}
          aria-expanded={moreOpen}
          aria-haspopup="dialog"
        >
          <span className="bottom-tab-icon">≡</span>
          <span className="bottom-tab-label">More</span>
        </button>
      </nav>

      {/* ── Slide-up More drawer ── */}
      {moreOpen && (
        <div className="more-overlay" onClick={() => setMoreOpen(false)} role="dialog" aria-modal>
          <div className="more-sheet" onClick={e => e.stopPropagation()}>
            <div className="more-sheet-handle" aria-hidden />
            <div className="more-sheet-title">More</div>
            <div className="more-sheet-grid">
              {SECONDARY_ITEMS.map(item => (
                <button
                  key={item.view}
                  className={activeView === item.view ? 'more-item active' : 'more-item'}
                  onClick={() => handleNav(item.view)}
                >
                  <span className="more-item-icon"><MoreIcon view={item.view} /></span>
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function TabIcon({ view }: { view: string }) {
  const m: Record<string, string> = {
    dashboard: '⬡', world3d: '◎', maps: '⊕', npcs: '◑', quests: '◈',
  };
  return <>{m[view] ?? '·'}</>;
}

function MoreIcon({ view }: { view: string }) {
  const m: Record<string, string> = {
    settlement: '⊞', households: '⌂', events: '⊙', lore: '◉', dungeons: '⟁',
  };
  return <>{m[view] ?? '·'}</>;
}
