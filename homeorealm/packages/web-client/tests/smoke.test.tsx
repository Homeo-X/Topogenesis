import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QuestBoard } from '../src/components/QuestBoard.js';
import { LoreCodex } from '../src/components/LoreCodex.js';
import { HouseholdPanel } from '../src/components/HouseholdPanel.js';

vi.mock('../src/api.js', () => ({
  api: {
    getQuests: vi.fn().mockResolvedValue([]),
    getRegions: vi.fn().mockResolvedValue([]),
    getFactions: vi.fn().mockResolvedValue([]),
    getPeoples: vi.fn().mockResolvedValue([]),
    getAssets: vi.fn().mockResolvedValue([]),
    getHouseholds: vi.fn().mockResolvedValue([]),
    getNPCs: vi.fn().mockResolvedValue([]),
    getWorld: vi.fn().mockResolvedValue({ day: 0, settlements: [], totalNPCs: 0, avgViability: '0%', activeQuests: 0, dungeonRooms: 0 }),
    getEvents: vi.fn().mockResolvedValue([]),
  },
}));

describe('QuestBoard smoke', () => {
  it('renders with empty state', async () => {
    render(<QuestBoard />);
    expect(screen.getByText(/Town Board/)).toBeTruthy();
    expect(screen.getByText(/The board is bare/)).toBeTruthy();
  });
});

describe('LoreCodex smoke', () => {
  it('renders tabs', async () => {
    render(<LoreCodex />);
    expect(screen.getByText(/Lore Codex/)).toBeTruthy();
    expect(screen.getByText('Regions')).toBeTruthy();
    expect(screen.getByText('Factions')).toBeTruthy();
    expect(screen.getByText('Peoples')).toBeTruthy();
    expect(screen.getByText('Assets')).toBeTruthy();
  });
});

describe('HouseholdPanel smoke', () => {
  it('renders with empty households', async () => {
    render(<HouseholdPanel />);
    expect(screen.getByText(/Households/)).toBeTruthy();
  });
});
