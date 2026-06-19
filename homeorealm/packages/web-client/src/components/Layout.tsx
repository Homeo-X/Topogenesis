import { useState, useEffect, type ReactNode } from 'react';

type NavItem = { label: string; view: string };

const NAV_ITEMS: NavItem[] = [
  { label: 'Dashboard', view: 'dashboard' },
  { label: 'NPCs', view: 'npcs' },
  { label: 'Quests', view: 'quests' },
  { label: 'Settlement', view: 'settlement' },
  { label: 'Households', view: 'households' },
  { label: 'Events', view: 'events' },
  { label: 'Lore Codex', view: 'lore' },
  { label: 'Dungeons', view: 'dungeons' },
];

type Props = { activeView: string; onNavigate: (v: string) => void; children: ReactNode };

export function Layout({ activeView, onNavigate, children }: Props) {
  const [menuOpen, setMenuOpen] = useState(false);

  function handleNav(v: string) {
    onNavigate(v);
    setMenuOpen(false);
  }

  // Close drawer on Escape
  useEffect(() => {
    if (!menuOpen) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') setMenuOpen(false); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [menuOpen]);

  return (
    <div className="layout">
      <header className="top-bar">
        <div className="logo">⬡ HomeoRealm</div>
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
        <button
          className="hamburger"
          onClick={() => setMenuOpen(o => !o)}
          aria-label={menuOpen ? 'Close menu' : 'Open menu'}
          aria-expanded={menuOpen}
        >
          {menuOpen ? '✕' : '☰'}
        </button>
      </header>
      <main className="content">{children}</main>
    </div>
  );
}
