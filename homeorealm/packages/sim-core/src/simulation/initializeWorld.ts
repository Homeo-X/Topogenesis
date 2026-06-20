/** World initialization from content data and seed. */

import type { WorldState, NPCState, SettlementState, HouseholdState } from '../types.js';
import { createNPC, addRelationship } from '../npc/npcState.js';
import { createHousehold } from '../settlement/households.js';
import { createInitialResources } from '../settlement/resources.js';
import { createEventStore } from '../events.js';
import { createRng } from '../rng.js';
import { nextId, resetIdCounter } from '../ids.js';
import type { NPCArchetype } from '../npc/npcState.js';
import { getJobsForSettlement } from '../settlement/jobs.js';
import { createPlayer } from '../player/playerState.js';
import { createInitialEnvironment } from '../environment/environment.js';

export const DEMO_ARCHETYPES: NPCArchetype[] = [
  {
    id: 'auran_farmer', people: 'Aurans', possibleJobs: ['farmer', 'gatherer'],
    traitBias: { industrious: 0.7, communal: 0.6, traditional: 0.5 },
    skillBias: { farming: 0.5, gathering: 0.3, cooking: 0.4 },
    namePool: ['Aldric', 'Maren', 'Solen', 'Willa', 'Dafton', 'Liora', 'Cress', 'Edwin'],
  },
  {
    id: 'auran_crafter', people: 'Aurans', possibleJobs: ['crafter', 'merchant'],
    traitBias: { precise: 0.7, patient: 0.6 },
    skillBias: { crafting: 0.55, trade: 0.35 },
    namePool: ['Bram', 'Sela', 'Torvin', 'Nara', 'Holt', 'Fara'],
  },
  {
    id: 'auran_healer', people: 'Aurans', possibleJobs: ['healer', 'scholar'],
    traitBias: { compassionate: 0.8, studious: 0.6 },
    skillBias: { medicine: 0.6, scholarship: 0.4 },
    namePool: ['Mirith', 'Galen', 'Calyx', 'Sera'],
  },
  {
    id: 'auran_guard', people: 'Aurans', possibleJobs: ['guard', 'adventurer'],
    traitBias: { disciplined: 0.7, loyal: 0.6, brave: 0.5 },
    skillBias: { combat: 0.55, exploration: 0.35 },
    namePool: ['Tor', 'Vex', 'Lyra', 'Denn', 'Kaev', 'Risa'],
  },
  {
    id: 'mireborn_gatherer', people: 'Mireborn', possibleJobs: ['gatherer', 'healer'],
    traitBias: { perceptive: 0.7, melancholic: 0.4 },
    skillBias: { gathering: 0.6, medicine: 0.4 },
    namePool: ['Ossil', 'Veyne', 'Mura', 'Thal', 'Sheva'],
  },
  {
    id: 'auran_civic', people: 'Aurans', possibleJobs: ['civic_officer', 'merchant'],
    traitBias: { diplomatic: 0.7, pragmatic: 0.5 },
    skillBias: { governance: 0.55, trade: 0.45 },
    namePool: ['Aldren', 'Pira', 'Cyron', 'Vell'],
  },
];

const JOBS_POOL = ['farmer', 'gatherer', 'guard', 'crafter', 'healer', 'merchant', 'civic_officer', 'adventurer'];

/** Initialize a world with a seeded demo settlement and NPC population. */
export function initializeWorld(seed: number, npcCount: number = 25): { world: WorldState; eventStore: ReturnType<typeof createEventStore> } {
  resetIdCounter();
  const rng = createRng(seed);
  const eventStore = createEventStore();

  const settlementId = nextId('set');
  const resources = createInitialResources('crown_valley');

  const npcs: Record<string, NPCState> = {};
  const households: Record<string, HouseholdState> = {};

  // Create households (roughly 1 per 3-4 NPCs)
  const hhCount = Math.ceil(npcCount / 3);
  const hhIds: string[] = [];
  for (let i = 0; i < hhCount; i++) {
    const hhName = rng.nextChoice(['Venn', 'Holt', 'Marren', 'Ashbel', 'Cressa', 'Dunmore', 'Fairwick', 'Greymont', 'Ironside']);
    const hh = createHousehold(settlementId, [], `${hhName} Household`);
    households[hh.id] = hh;
    hhIds.push(hh.id);
  }

  const availableJobs = getJobsForSettlement(npcCount);

  // Create NPCs
  for (let i = 0; i < npcCount; i++) {
    const archetype = rng.nextChoice(DEMO_ARCHETYPES);
    const hhId = hhIds[i % hhIds.length]!;
    const jobIndex = i % availableJobs.length;
    const jobId = availableJobs[jobIndex] ?? 'farmer';

    const npc = createNPC(archetype, settlementId, hhId, jobId, 0);
    // Vary starting needs slightly
    npc.needs.hunger = 0.5 + rng.next() * 0.4;
    npc.needs.rest = 0.5 + rng.next() * 0.4;
    npc.needs.safety = 0.6 + rng.next() * 0.3;
    npc.health = 0.7 + rng.next() * 0.3;
    npc.wealth = 5 + rng.nextInt(0, 30);

    npcs[npc.id] = npc;
    households[hhId] = {
      ...households[hhId]!,
      memberIds: [...households[hhId]!.memberIds, npc.id],
    };
  }

  // Seed some starting relationships (household members know each other)
  const npcList = Object.values(npcs);
  for (const hh of Object.values(households)) {
    const members = hh.memberIds.map(id => npcs[id]).filter(Boolean) as NPCState[];
    for (let i = 0; i < members.length; i++) {
      for (let j = i + 1; j < members.length; j++) {
        const a = members[i]!;
        const b = members[j]!;
        npcs[a.id] = addRelationship(npcs[a.id]!, b.id, 'kin');
        npcs[b.id] = addRelationship(npcs[b.id]!, a.id, 'kin');
      }
    }
  }

  // A few cross-household acquaintances
  for (let i = 0; i < Math.min(10, npcList.length); i++) {
    const a = rng.nextChoice(npcList);
    const b = rng.nextChoice(npcList);
    if (a.id !== b.id && !a.relationships[b.id]) {
      npcs[a.id] = addRelationship(npcs[a.id]!, b.id, 'acquaintance');
      npcs[b.id] = addRelationship(npcs[b.id]!, a.id, 'acquaintance');
    }
  }

  const settlement: SettlementState = {
    id: settlementId,
    name: 'Vennholt',
    regionId: 'crown_valley',
    factionId: 'hearthwardens',
    population: npcCount,
    resources,
    npcIds: Object.keys(npcs),
    householdIds: hhIds,
    day: 0,
  };

  const world: WorldState = {
    seed,
    day: 0,
    tick: 0,
    settlements: { [settlementId]: settlement },
    npcs,
    households,
    quests: {},
    dungeonRooms: [],
    environment: createInitialEnvironment(seed, settlement.regionId),
    player: createPlayer(settlementId),
  };

  return { world, eventStore };
}
