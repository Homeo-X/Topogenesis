/** Dungeon topogenesis — rooms generated from settlement pressure vectors. */

import { nextId } from '../ids.js';
import type { DungeonRoom, SettlementResources } from '../types.js';
import type { SeededRng } from '../rng.js';
import { getSettlementPressures } from '../settlement/economy.js';

type RoomTemplate = {
  type: DungeonRoom['type'];
  pressureSource: string;
  descriptions: string[];
  tags: string[];
  baseDifficulty: number;
};

const ROOM_TEMPLATES: RoomTemplate[] = [
  {
    type: 'hunger', pressureSource: 'food_shortage',
    descriptions: [
      'A hall of phantom feasts — tables laden with food that dissolves at touch.',
      'The Famished Gallery: endless rows of empty larder shelves, faint cries from the walls.',
      'A rotting granary where grain turns to ash as it is gathered.',
    ],
    tags: ['food', 'shortage', 'hunger', 'phantom'],
    baseDifficulty: 0.4,
  },
  {
    type: 'hunger', pressureSource: 'critical_food_shortage',
    descriptions: [
      'The Gnawing Vault — a chamber where Famine Sprites patrol, draining vitality from the unprepared.',
    ],
    tags: ['food', 'critical', 'combat', 'hunger'],
    baseDifficulty: 0.7,
  },
  {
    type: 'oath', pressureSource: 'relationship_conflict',
    descriptions: [
      'The Chamber of Broken Vows: witnesses made of crystallized memory line the walls, reciting old promises.',
      'An oath-binding chamber where two NPC echoes re-enact their dispute and must be reconciled.',
    ],
    tags: ['social', 'conflict', 'oath', 'memory'],
    baseDifficulty: 0.5,
  },
  {
    type: 'boundary', pressureSource: 'security_crisis',
    descriptions: [
      'Shifting Border Halls: the walls move, mimicking a settlement with no stable defenses.',
      'A monster nest rooted in the settlement\'s undefended shadow.',
    ],
    tags: ['security', 'combat', 'boundary', 'monster'],
    baseDifficulty: 0.6,
  },
  {
    type: 'kin', pressureSource: 'household_food_shortage',
    descriptions: [
      'The Ancestor Echo Hall: household spirits call out with unfulfilled obligations.',
    ],
    tags: ['household', 'kinship', 'memory', 'ritual'],
    baseDifficulty: 0.4,
  },
  {
    type: 'knowledge', pressureSource: 'low_morale',
    descriptions: [
      'The Contradictory Archive: maps that lead nowhere, books that rewrite themselves, scholars who argue without resolution.',
    ],
    tags: ['knowledge', 'morale', 'confusion', 'investigation'],
    baseDifficulty: 0.5,
  },
  {
    type: 'hearthwell', pressureSource: 'hearthwell_instability',
    descriptions: [
      'The Distortion Nave: a room where physical laws bend, frame distinctions fail, and memory becomes unreliable.',
      'The Rupture Chamber — core of the Subframe Labyrinth, source of the Hearthwell\'s illness.',
    ],
    tags: ['hearthwell', 'ritual', 'distortion', 'frame'],
    baseDifficulty: 0.8,
  },
];

/** Generate dungeon rooms from settlement pressure vector. */
export function generateDungeonRooms(
  settlementId: string,
  resources: SettlementResources,
  population: number,
  day: number,
  rng: SeededRng,
): DungeonRoom[] {
  const pressures = getSettlementPressures(resources, population);
  if (pressures.length === 0) return [];

  const rooms: DungeonRoom[] = [];
  const usedPressures = new Set<string>();

  for (const pressure of pressures) {
    if (usedPressures.has(pressure)) continue;
    const matching = ROOM_TEMPLATES.filter(t => t.pressureSource === pressure);
    if (matching.length === 0) continue;

    const template = rng.nextChoice(matching);
    const description = rng.nextChoice(template.descriptions);
    const difficulty = Math.min(1, template.baseDifficulty + (rng.next() - 0.5) * 0.2);

    rooms.push({
      id: nextId('room'),
      settlementId,
      type: template.type,
      pressureSource: pressure,
      description,
      difficulty,
      tags: template.tags,
      generatedOnDay: day,
    });
    usedPressures.add(pressure);
  }

  return rooms;
}

/** Refresh dungeon rooms when pressures change. */
export function refreshDungeonRooms(
  existingRooms: DungeonRoom[],
  settlementId: string,
  resources: SettlementResources,
  population: number,
  day: number,
  rng: SeededRng,
): DungeonRoom[] {
  // Keep rooms less than 10 days old if pressure still exists
  const currentPressures = new Set(getSettlementPressures(resources, population));
  const surviving = existingRooms.filter(
    r => r.settlementId === settlementId && currentPressures.has(r.pressureSource) && day - r.generatedOnDay < 10,
  );
  const newPressures = [...currentPressures].filter(p => !surviving.some(r => r.pressureSource === p));

  if (newPressures.length === 0) return surviving;

  const newRooms = generateDungeonRooms(settlementId, resources, population, day, rng);
  return [...surviving, ...newRooms.filter(r => newPressures.includes(r.pressureSource))];
}
