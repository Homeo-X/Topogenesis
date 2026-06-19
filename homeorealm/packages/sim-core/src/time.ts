/**
 * Deterministic world calendar for HomeoRealm Online.
 * Season cycle: Spring → Summer → Autumn → Winter (each 30 days)
 * Day phases: dawn, morning, midday, afternoon, evening, night
 */

export type Season = 'spring' | 'summer' | 'autumn' | 'winter';
export type DayPhase = 'dawn' | 'morning' | 'midday' | 'afternoon' | 'evening' | 'night';

export type WorldClock = {
  day: number;       // absolute day count from 0
  year: number;      // 0-based
  dayOfYear: number; // 0–119 (120-day year, 4 seasons × 30 days)
  season: Season;
  dayOfSeason: number; // 0–29
  phase: DayPhase;
  tick: number;      // current tick within day (0–5)
};

const DAYS_PER_SEASON = 30;
const SEASONS_PER_YEAR = 4;
const DAYS_PER_YEAR = DAYS_PER_SEASON * SEASONS_PER_YEAR;
const PHASES: DayPhase[] = ['dawn', 'morning', 'midday', 'afternoon', 'evening', 'night'];
const SEASONS: Season[] = ['spring', 'summer', 'autumn', 'winter'];

export function computeClock(day: number, tick = 0): WorldClock {
  const year = Math.floor(day / DAYS_PER_YEAR);
  const dayOfYear = day % DAYS_PER_YEAR;
  const seasonIndex = Math.floor(dayOfYear / DAYS_PER_SEASON);
  const season = SEASONS[seasonIndex] as Season;
  const dayOfSeason = dayOfYear % DAYS_PER_SEASON;
  const phase = PHASES[tick % PHASES.length] as DayPhase;
  return { day, year, dayOfYear, season, dayOfSeason, phase, tick };
}

export type SeasonalModifiers = {
  farmingYield: number;
  travelRisk: number;
  illnessRisk: number;
  festivalProbability: number;
  monsterActivity: number;
  resourceRegenRate: number;
};

const SEASONAL_MODIFIERS: Record<Season, SeasonalModifiers> = {
  spring: { farmingYield: 1.2, travelRisk: 0.8, illnessRisk: 0.9, festivalProbability: 0.3, monsterActivity: 0.8, resourceRegenRate: 1.3 },
  summer: { farmingYield: 1.5, travelRisk: 0.6, illnessRisk: 0.7, festivalProbability: 0.4, monsterActivity: 1.0, resourceRegenRate: 1.1 },
  autumn: { farmingYield: 1.4, travelRisk: 1.0, illnessRisk: 1.1, festivalProbability: 0.5, monsterActivity: 1.2, resourceRegenRate: 0.9 },
  winter: { farmingYield: 0.4, travelRisk: 1.5, illnessRisk: 1.5, festivalProbability: 0.2, monsterActivity: 0.7, resourceRegenRate: 0.5 },
};

export function getSeasonalModifiers(season: Season): SeasonalModifiers {
  return SEASONAL_MODIFIERS[season];
}

export function isHarvestDay(clock: WorldClock): boolean {
  return clock.season === 'autumn' && clock.dayOfSeason === 14;
}

export function isFestivalDay(clock: WorldClock): boolean {
  // Major festivals on day 0 of each season, plus mid-season
  return clock.dayOfSeason === 0 || clock.dayOfSeason === 15;
}

export function clockToString(clock: WorldClock): string {
  return `Year ${clock.year + 1}, ${capitalize(clock.season)} Day ${clock.dayOfSeason + 1} (${capitalize(clock.phase)})`;
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
