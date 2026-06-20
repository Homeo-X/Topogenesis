import { useState, useEffect, type ReactNode } from 'react';

type NavItem = { label: string; view: string };

const NAV_ITEMS: NavItem[] = [
  { label: 'Dashboard', view: 'dashboard' },
  { label: '3D World', view: 'world3d' },
  { label: 'Maps', view: 'maps' },
  { label: 'NPCs', view: 'npcs' },
  { label: 'Quests', view: 'quests' },
  { label: 'Settlement', view: 'settlement' },
  { label: 'Households', view: 'households' },
  { label: 'Events', view: 'events' },
  { label: 'Lore Codex', view: 'lore' },
  { label: 'Dungeons', view: 'dungeons' },
];

const SPEED_OPTIONS = [
  { label: 'Calm', ms: 5000 },
  { label: 'Normal', ms: 3000 },
  { label: 'Fast', ms: 1200 },
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
  activeView,
  onNavigate,
  children,
  worldDay,
  autoRun,
  speedMs,
  clockBusy,
  onToggleAutoRun,
  onSpeedChange,
}: Props) {
  const [menuOpen, setMenuOpen] = useState(false);

  function handleNav(v: string) {
    onNavigate(v);
    setMenuOpen(false);
  }

  useEffect(() => {
    if (!menuOpen) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') setMenuOpen(false); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [menuOpen]);

  return (
    <div className="layout">
      <header className="top-bar">
        <div className="logo">HomeoRealm</div>
        <nav className={menuOpen ? 'open' : ''} onClick={e => { if (e.target === e.currentTarget) setMenuOpen(false); }}>
          {NAV_ITEMS.map(item => (
            <button
              key={item.view}
              className={activeView === item.view ? 'nav-btn active' : 'nav-btn'}
              onClick={() => handleNav(item.view)}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <div className="world-clock" aria-label="World clock">
          <span className="clock-day">Day {worldDay}</span>
          <button className={autoRun ? 'clock-toggle running' : 'clock-toggle'} onClick={onToggleAutoRun}>
            {autoRun ? 'Pause' : 'Auto'}
          </button>
          <select
            className="clock-speed"
            value={speedMs}
            onChange={e => onSpeedChange(Number(e.target.value))}
            aria-label="Simulation speed"
          >
            {SPEED_OPTIONS.map(option => (
              <option key={option.ms} value={option.ms}>{option.label}</option>
            ))}
          </select>
          {clockBusy && <span className="clock-pulse" aria-label="Advancing" />}
        </div>
        <button
          className="hamburger"
          onClick={() => setMenuOpen(o => !o)}
          aria-label={menuOpen ? 'Close menu' : 'Open menu'}
          aria-expanded={menuOpen}
        >
          {menuOpen ? 'Close' : 'Menu'}
        </button>
      </header>
      <main className="content">{children}</main>
    </div>
  );
}
