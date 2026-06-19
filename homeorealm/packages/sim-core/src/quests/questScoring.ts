/** Quest relevance scoring for player UI ordering. */

import type { EmergentQuest } from '../types.js';

export function scoreQuestRelevance(quest: EmergentQuest): number {
  return quest.urgency * 2 + quest.difficulty * 0.5;
}

export function sortQuestsByPriority(quests: EmergentQuest[]): EmergentQuest[] {
  return [...quests].sort((a, b) => scoreQuestRelevance(b) - scoreQuestRelevance(a));
}
