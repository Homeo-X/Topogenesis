import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QuestBoard } from '../src/components/QuestBoard.js';
import { LoreCodex } from '../src/components/LoreCodex.js';
import { HouseholdPanel } from '../src/components/HouseholdPanel.js';
import { MapAtlas } from '../src/components/MapAtlas.js';

const environment = vi.hoisted(() => ({
  regionId: 'crown_valley',
  season: 'spring',
  weather: 'mist',
  physics: {
    airTemperatureC: 16.4,
    humidity: 0.62,
    rainfallMm: 1.8,
    windSpeedMps: 2.4,
    windDirectionDeg: 80,
    airPressureKpa: 101.1,
    soilMoisture: 0.55,
    groundwater: 0.55,
    evaporationRate: 0.15,
    turbulence: 0.25,
  },
  chemistry: {
    oxygenRatio: 0.209,
    carbonDioxidePpm: 420,
    soilPH: 6.8,
    mineralSaturation: 0.42,
    dissolvedIron: 0.28,
    organicMatter: 0.52,
    fermentation: 0.12,
    corrosion: 0.16,
  },
  signals: [],
}));

vi.mock('../src/api.js', () => ({
  api: {
    getQuests: vi.fn().mockResolvedValue([]),
    getRegions: vi.fn().mockResolvedValue([]),
    getFactions: vi.fn().mockResolvedValue([]),
    getPeoples: vi.fn().mockResolvedValue([]),
    getAssets: vi.fn().mockResolvedValue([]),
    getHouseholds: vi.fn().mockResolvedValue([]),
    getNPCs: vi.fn().mockResolvedValue([]),
    getSettlements: vi.fn().mockResolvedValue([{
      id: 'set_1',
      name: 'Vennholt',
      regionId: 'crown_valley',
      population: 25,
      resources: {
        food: 100,
        wood: 50,
        ore: 20,
        cloth: 20,
        medicine: 10,
        tools: 8,
        coin: 100,
        heartwellStability: 0.8,
        publicMorale: 0.7,
        security: 0.6,
      },
    }]),
    getDungeons: vi.fn().mockResolvedValue([]),
    getWorld: vi.fn().mockResolvedValue({ day: 0, settlements: [], totalNPCs: 0, avgViability: '0%', activeQuests: 0, dungeonRooms: 0, environment }),
    getEvents: vi.fn().mockResolvedValue([]),
    getPlayer: vi.fn().mockResolvedValue({
      id: 'player_1',
      name: 'Wayfarer',
      settlementId: 'set_1',
      location: 'town',
      health: 1,
      stamina: 1,
      level: 1,
      experience: 0,
      skills: {},
      wealth: 20,
      reputation: { set_1: 0.1 },
      inventory: [],
      questLog: [],
      actionLog: [],
    }),
    playerAction: vi.fn().mockResolvedValue({ message: 'ok', player: {} }),
  },
}));

describe('QuestBoard smoke', () => {
  it('renders with empty state', async () => {
    render(<QuestBoard />);
    expect(await screen.findByText(/Town Board/)).toBeTruthy();
    expect(await screen.findByText(/The board is bare/)).toBeTruthy();
  });
});

describe('LoreCodex smoke', () => {
  it('renders tabs', async () => {
    render(<LoreCodex />);
    expect(await screen.findByText(/Lore Codex/)).toBeTruthy();
    expect(screen.getByText('Regions')).toBeTruthy();
    expect(screen.getByText('Factions')).toBeTruthy();
    expect(screen.getByText('Peoples')).toBeTruthy();
    expect(screen.getByText('Assets')).toBeTruthy();
  });
});

describe('HouseholdPanel smoke', () => {
  it('renders with empty households', async () => {
    render(<HouseholdPanel />);
    expect(await screen.findByText(/Households/)).toBeTruthy();
  });
});

describe('MapAtlas smoke', () => {
  it('renders player-aware map controls', async () => {
    render(<MapAtlas />);
    expect(await screen.findByText(/Auralis World/)).toBeTruthy();
    expect(screen.getByText('World Map')).toBeTruthy();
    expect(screen.getByText('Expedition Readiness')).toBeTruthy();
    expect(screen.getByText('Map Actions')).toBeTruthy();
    expect(screen.getByText('Gather')).toBeTruthy();
  });
});
