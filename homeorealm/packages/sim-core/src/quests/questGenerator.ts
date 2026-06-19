/** Emergent quest generator — quests arise from actual simulation pressure. */

import { nextId } from '../ids.js';
import type { EmergentQuest, QuestObjective, QuestReward, QuestConsequence, NPCState, SettlementState, HouseholdState, WorldState } from '../types.js';
import type { SeededRng } from '../rng.js';
import { computeViability } from '../npc/needs.js';
import { getSettlementPressures } from '../settlement/economy.js';

export type QuestTemplate = {
  id: string;
  cause: string;
  titleTemplate: string;
  descriptionTemplate: string;
  tags: string[];
  urgencyBase: number;
  difficultyBase: number;
};

const QUEST_TEMPLATES: QuestTemplate[] = [
  {
    id: 'food_shortage_gather',
    cause: 'food_shortage',
    titleTemplate: 'The Empty Granary',
    descriptionTemplate: 'The settlement\'s food stores are critically low. Gather or deliver food before hunger weakens the community.',
    tags: ['food', 'gathering', 'survival'],
    urgencyBase: 0.8, difficultyBase: 0.3,
  },
  {
    id: 'food_shortage_trade',
    cause: 'food_shortage',
    titleTemplate: 'Trade for Provisions',
    descriptionTemplate: 'Secure a trade agreement to import food from a neighboring region.',
    tags: ['food', 'trade', 'diplomacy'],
    urgencyBase: 0.7, difficultyBase: 0.5,
  },
  {
    id: 'npc_hunger_crisis',
    cause: 'npc_hunger',
    titleTemplate: 'A Neighbor Starves',
    descriptionTemplate: 'A resident is in danger from lack of food. Bring provisions before their health fails.',
    tags: ['food', 'personal', 'compassion'],
    urgencyBase: 0.9, difficultyBase: 0.2,
  },
  {
    id: 'medicine_shortage',
    cause: 'medicine_shortage',
    titleTemplate: 'Healer\'s Lament',
    descriptionTemplate: 'Medicine stocks are dangerously low. Gather herbs or bring medicine to the settlement.',
    tags: ['medicine', 'gathering', 'health'],
    urgencyBase: 0.7, difficultyBase: 0.4,
  },
  {
    id: 'security_crisis',
    cause: 'security_crisis',
    titleTemplate: 'Thin Walls',
    descriptionTemplate: 'The settlement\'s defenses are failing. Patrol the perimeter and recruit more guards.',
    tags: ['security', 'combat', 'protection'],
    urgencyBase: 0.8, difficultyBase: 0.6,
  },
  {
    id: 'hearthwell_instability',
    cause: 'hearthwell_instability',
    titleTemplate: 'The Well Sickens',
    descriptionTemplate: 'The Hearthwell\'s stability is wavering. Investigate the cause and perform the stabilization rites.',
    tags: ['hearthwell', 'ritual', 'investigation'],
    urgencyBase: 0.85, difficultyBase: 0.7,
  },
  {
    id: 'low_morale',
    cause: 'low_morale',
    titleTemplate: 'Lift the Spirits',
    descriptionTemplate: 'Public morale is suffering. Organize a festival or community event to restore hope.',
    tags: ['morale', 'festival', 'community'],
    urgencyBase: 0.5, difficultyBase: 0.3,
  },
  {
    id: 'household_food_shortage',
    cause: 'household_food_shortage',
    titleTemplate: 'The {household} Household Hungers',
    descriptionTemplate: 'A household is running out of food. Help them before members fall ill.',
    tags: ['food', 'household', 'social'],
    urgencyBase: 0.75, difficultyBase: 0.25,
  },
  {
    id: 'npc_viability_crisis',
    cause: 'npc_viability_crisis',
    titleTemplate: '{npc_name} Calls for Help',
    descriptionTemplate: 'A community member is struggling to maintain their wellbeing and has asked for assistance.',
    tags: ['viability', 'social', 'compassion'],
    urgencyBase: 0.7, difficultyBase: 0.3,
  },
  {
    id: 'relationship_conflict',
    cause: 'relationship_conflict',
    titleTemplate: 'A Rift in the Community',
    descriptionTemplate: 'Two residents are in serious conflict. Mediate before it tears apart their shared work.',
    tags: ['social', 'mediation', 'conflict'],
    urgencyBase: 0.5, difficultyBase: 0.5,
  },
  {
    id: 'tools_shortage',
    cause: 'tools_shortage',
    titleTemplate: 'The Smith\'s Call',
    descriptionTemplate: 'Tool shortages are hampering all production. Craft or source tools urgently.',
    tags: ['crafting', 'tools', 'economy'],
    urgencyBase: 0.65, difficultyBase: 0.4,
  },
  {
    id: 'dungeon_threat',
    cause: 'hearthwell_instability',
    titleTemplate: 'Subframe Rupture',
    descriptionTemplate: 'A dungeon passage has opened beneath the settlement. Descend and stabilize the frame before monsters emerge.',
    tags: ['dungeon', 'combat', 'hearthwell', 'exploration'],
    urgencyBase: 0.9, difficultyBase: 0.8,
  },
];

/** Maximum active quests per settlement — prevents spam. */
const MAX_QUESTS_PER_SETTLEMENT = 6;

/** Maps granular pressure names to the quest cause key used in templates. */
const PRESSURE_TO_CAUSE: Record<string, string> = {
  critical_food_shortage: 'food_shortage',
  food_shortage: 'food_shortage',
  medicine_shortage: 'medicine_shortage',
  tools_shortage: 'tools_shortage',
  security_crisis: 'security_crisis',
  low_morale: 'low_morale',
  hearthwell_instability: 'hearthwell_instability',
};

/** Generate quests from current world state. Returns new quests only (no duplicates by cause). */
export function generateEmergentQuests(
  world: WorldState,
  settlementId: string,
  rng: SeededRng,
): EmergentQuest[] {
  const settlement = world.settlements[settlementId];
  if (!settlement) return [];

  const currentDay = world.day;
  const pressures = getSettlementPressures(settlement.resources, settlement.population);
  const existingCauses = new Set(
    Object.values(world.quests)
      .filter(q => q.settlementId === settlementId && q.isActive)
      .map(q => q.cause),
  );

  // Don't generate more if already at max
  const activeCount = Object.values(world.quests).filter(q => q.settlementId === settlementId && q.isActive).length;
  if (activeCount >= MAX_QUESTS_PER_SETTLEMENT) return [];

  const newQuests: EmergentQuest[] = [];
  const settlementNpcs = settlement.npcIds.map(id => world.npcs[id]).filter(Boolean) as NPCState[];

  // Settlement-level pressure quests
  for (const pressure of pressures) {
    const cause = PRESSURE_TO_CAUSE[pressure] ?? pressure;
    if (existingCauses.has(cause)) continue;
    const templates = QUEST_TEMPLATES.filter(t => t.cause === cause);
    if (templates.length === 0) continue;
    const template = rng.nextChoice(templates);
    const quest = buildQuest(template, settlement, undefined, undefined, currentDay, rng);
    quest.cause = cause;
    newQuests.push(quest);
    existingCauses.add(cause);
  }

  // NPC-level viability quests
  const criticalNpcs = settlementNpcs.filter(n => computeViability(n) < 0.3 && n.isAlive);
  for (const npc of criticalNpcs.slice(0, 2)) {
    const cause = 'npc_viability_crisis';
    if (existingCauses.has(cause + npc.id)) continue;
    const template = QUEST_TEMPLATES.find(t => t.id === 'npc_viability_crisis')!;
    const quest = buildQuest(template, settlement, npc, undefined, currentDay, rng);
    quest.cause = cause + npc.id;
    quest.issuerNpcId = npc.id;
    quest.title = quest.title.replace('{npc_name}', npc.name);
    quest.description = quest.description;
    newQuests.push(quest);
    existingCauses.add(cause + npc.id);
  }

  // Hunger crisis quests from individual NPCs
  const hungryNpcs = settlementNpcs.filter(n => n.needs.hunger < 0.25 && n.isAlive);
  if (hungryNpcs.length > 0 && !existingCauses.has('npc_hunger')) {
    const npc = rng.nextChoice(hungryNpcs);
    const template = QUEST_TEMPLATES.find(t => t.id === 'npc_hunger_crisis')!;
    const quest = buildQuest(template, settlement, npc, undefined, currentDay, rng);
    quest.issuerNpcId = npc.id;
    newQuests.push(quest);
    existingCauses.add('npc_hunger');
  }

  // Relationship conflict quests
  const conflictedNpcs = settlementNpcs.filter(n =>
    Object.values(n.relationships).some(r => r.conflict > 0.6) && n.isAlive,
  );
  if (conflictedNpcs.length >= 2 && !existingCauses.has('relationship_conflict')) {
    const template = QUEST_TEMPLATES.find(t => t.id === 'relationship_conflict')!;
    const quest = buildQuest(template, settlement, conflictedNpcs[0], undefined, currentDay, rng);
    newQuests.push(quest);
    existingCauses.add('relationship_conflict');
  }

  // Household food shortage quests
  const hungrHouseholds = settlement.householdIds
    .map(id => world.households[id])
    .filter(Boolean)
    .filter(h => h!.foodStores < 1.5) as HouseholdState[];
  if (hungrHouseholds.length > 0 && !existingCauses.has('household_food_shortage')) {
    const hh = hungrHouseholds[0]!;
    const template = QUEST_TEMPLATES.find(t => t.id === 'household_food_shortage')!;
    const quest = buildQuest(template, settlement, undefined, hh, currentDay, rng);
    quest.title = quest.title.replace('{household}', hh.name);
    newQuests.push(quest);
    existingCauses.add('household_food_shortage');
  }

  return newQuests.slice(0, MAX_QUESTS_PER_SETTLEMENT - activeCount);
}

function buildQuest(
  template: QuestTemplate,
  settlement: SettlementState,
  npc: NPCState | undefined,
  household: HouseholdState | undefined,
  day: number,
  rng: SeededRng,
): EmergentQuest {
  const urgency = Math.min(1, template.urgencyBase + rng.next() * 0.1 - 0.05);
  const difficulty = Math.min(1, template.difficultyBase + rng.next() * 0.15 - 0.075);

  const objectives: QuestObjective[] = buildObjectives(template, rng);
  const rewards: QuestReward[] = buildRewards(urgency, difficulty, rng);
  const consequences: QuestConsequence[] = buildConsequences(template, settlement, rng);

  return {
    id: nextId('q'),
    title: template.titleTemplate,
    description: template.descriptionTemplate,
    issuerNpcId: npc?.id,
    settlementId: settlement.id,
    cause: template.cause,
    urgency,
    difficulty,
    objectives,
    rewards,
    consequences,
    expiresOnDay: day + Math.floor(5 + (1 - urgency) * 15),
    tags: template.tags,
    generatedOnDay: day,
    isActive: true,
  };
}

function buildObjectives(template: QuestTemplate, rng: SeededRng): QuestObjective[] {
  const obj: QuestObjective = {
    id: nextId('obj'),
    description: '',
    type: 'gather',
    completed: false,
  };

  if (template.tags.includes('food')) {
    obj.type = 'gather';
    obj.resourceId = 'food';
    obj.quantity = 10 + rng.nextInt(5, 20);
    obj.description = `Gather ${obj.quantity} units of food for the settlement.`;
  } else if (template.tags.includes('crafting')) {
    obj.type = 'craft';
    obj.resourceId = 'tools';
    obj.quantity = 5 + rng.nextInt(2, 8);
    obj.description = `Craft ${obj.quantity} tools for settlement use.`;
  } else if (template.tags.includes('combat') || template.tags.includes('security')) {
    obj.type = 'guard';
    obj.description = 'Patrol the settlement perimeter for 3 days.';
  } else if (template.tags.includes('mediation')) {
    obj.type = 'mediate';
    obj.description = 'Speak to both parties and help them reach an agreement.';
  } else if (template.tags.includes('dungeon')) {
    obj.type = 'explore';
    obj.description = 'Enter the Subframe Labyrinth and seal the rupture.';
  } else if (template.tags.includes('investigation')) {
    obj.type = 'investigate';
    obj.description = 'Examine the Hearthwell and identify the source of instability.';
  } else if (template.tags.includes('festival')) {
    obj.type = 'deliver';
    obj.description = 'Organize a community gathering with food and music.';
  } else if (template.tags.includes('medicine')) {
    obj.type = 'gather';
    obj.resourceId = 'medicine';
    obj.quantity = 8 + rng.nextInt(3, 12);
    obj.description = `Gather ${obj.quantity} units of medicinal herbs.`;
  } else {
    obj.type = 'deliver';
    obj.description = 'Complete the task as requested.';
  }

  return [obj];
}

function buildRewards(urgency: number, difficulty: number, rng: SeededRng): QuestReward[] {
  const coinReward = Math.floor(10 + urgency * 30 + difficulty * 20 + rng.next() * 10);
  const rewards: QuestReward[] = [
    { type: 'coin', value: coinReward },
    { type: 'reputation', value: 0.05 + urgency * 0.1 },
  ];
  if (difficulty > 0.5) {
    rewards.push({ type: 'skill', value: 0.05 });
  }
  return rewards;
}

function buildConsequences(
  template: QuestTemplate,
  settlement: SettlementState,
  rng: SeededRng,
): QuestConsequence[] {
  return [
    {
      type: 'settlement_change',
      targetId: settlement.id,
      magnitude: 0.1 + template.urgencyBase * 0.2,
      description: `Addressing ${template.cause} improves settlement stability.`,
    },
  ];
}
