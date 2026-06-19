import type { ReactNode } from 'react';

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
  return (
    <div className="layout">
      <header className="top-bar">
        <div className="logo">⬡ HomeoRealm Online</div>
        <nav>
          {NAV_ITEMS.map(item => (
            <button
              key={item.view}
              className={activeView === item.view ? 'nav-btn active' : 'nav-btn'}
              onClick={() => onNavigate(item.view)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="content">{children}</main>
    </div>
  );
}
