/** Settlement governance: morale, stability, and civic decisions. */

import type { SettlementResources } from '../types.js';

export type GovernanceAction = 'rationing' | 'festival_decree' | 'security_alert' | 'hearthwell_maintenance' | 'trade_mission';

export function applyGovernanceAction(
  resources: SettlementResources,
  action: GovernanceAction,
): SettlementResources {
  const updated = { ...resources };
  switch (action) {
    case 'rationing':
      updated.publicMorale = Math.max(0, updated.publicMorale - 0.1);
      break;
    case 'festival_decree':
      updated.publicMorale = Math.min(1, updated.publicMorale + 0.15);
      updated.food = Math.max(0, updated.food - 5);
      break;
    case 'security_alert':
      updated.security = Math.min(1, updated.security + 0.1);
      updated.publicMorale = Math.max(0, updated.publicMorale - 0.05);
      break;
    case 'hearthwell_maintenance':
      updated.heartwellStability = Math.min(1, updated.heartwellStability + 0.1);
      break;
    case 'trade_mission':
      updated.coin = Math.min(1000, updated.coin + 15);
      break;
  }
  return updated;
}
