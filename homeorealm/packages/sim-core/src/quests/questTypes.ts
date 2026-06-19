/** Quest lifecycle utilities. */

import type { EmergentQuest } from '../types.js';

export function expireQuests(quests: Record<string, EmergentQuest>, currentDay: number): Record<string, EmergentQuest> {
  const updated: Record<string, EmergentQuest> = {};
  for (const [id, quest] of Object.entries(quests)) {
    if (quest.expiresOnDay !== undefined && currentDay >= quest.expiresOnDay && quest.isActive) {
      updated[id] = { ...quest, isActive: false };
    } else {
      updated[id] = quest;
    }
  }
  return updated;
}

export function getActiveQuests(quests: Record<string, EmergentQuest>): EmergentQuest[] {
  return Object.values(quests).filter(q => q.isActive);
}

export function completeQuest(quest: EmergentQuest): EmergentQuest {
  return { ...quest, isActive: false, objectives: quest.objectives.map(o => ({ ...o, completed: true })) };
}
