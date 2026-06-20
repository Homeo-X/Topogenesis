import type { EnvironmentState, SettlementResources, SettlementState } from '../types.js';
import { computeClock, type Season } from '../time.js';
import { createRng } from '../rng.js';

export type EnvironmentUpdate = {
  environment: EnvironmentState;
  resourceDeltas: Partial<SettlementResources>;
  events: Array<{
    type: string;
    payload: Record<string, unknown>;
    tags: string[];
    salience: number;
  }>;
};

const SEASON_BASE_TEMP: Record<Season, number> = {
  spring: 16,
  summer: 25,
  autumn: 13,
  winter: 4,
};

function clamp(value: number, min = 0, max = 1): number {
  return Math.max(min, Math.min(max, value));
}

function clampPH(value: number): number {
  return Math.max(4.5, Math.min(8.5, value));
}

function seasonWave(dayOfSeason: number): number {
  return Math.sin((dayOfSeason / 30) * Math.PI * 2);
}

function idealTemperatureFactor(tempC: number, low: number, high: number): number {
  if (tempC <= low || tempC >= high) return 0;
  const midpoint = (low + high) / 2;
  const span = (high - low) / 2;
  return clamp(1 - Math.abs(tempC - midpoint) / span);
}

export function createInitialEnvironment(seed: number, regionId = 'crown_valley'): EnvironmentState {
  const rng = createRng((seed ^ 0x9e3779b9) >>> 0);
  return {
    regionId,
    season: 'spring',
    weather: 'mist',
    physics: {
      airTemperatureC: 15 + rng.next() * 3,
      humidity: 0.62,
      rainfallMm: 1.8,
      windSpeedMps: 2.4,
      windDirectionDeg: rng.nextInt(0, 359),
      airPressureKpa: 101.1,
      soilMoisture: 0.58,
      groundwater: 0.55,
      evaporationRate: 0.18,
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
    signals: ['settled morning mist'],
  };
}

export function ensureEnvironment(
  environment: EnvironmentState | undefined,
  seed: number,
  regionId = 'crown_valley',
): EnvironmentState {
  return environment ?? createInitialEnvironment(seed, regionId);
}

export function advanceEnvironment(
  previous: EnvironmentState,
  settlement: SettlementState,
  day: number,
  seed: number,
): EnvironmentUpdate {
  const clock = computeClock(day);
  const rng = createRng((seed ^ hashString(`${settlement.id}:${day}:environment`)) >>> 0);
  const wave = seasonWave(clock.dayOfSeason);
  const physics = previous.physics;
  const chemistry = previous.chemistry;
  const stability = settlement.resources.heartwellStability;
  const populationPressure = clamp(settlement.population / 80, 0, 1);

  const baseTemp = SEASON_BASE_TEMP[clock.season] + wave * 3.5;
  const airTemperatureC = baseTemp + (rng.next() - 0.5) * 4 + (stability - 0.5) * 2;
  const pressureDrift = (rng.next() - 0.5) * 1.4 - physics.rainfallMm * 0.01;
  const airPressureKpa = Math.max(96.5, Math.min(104.5, physics.airPressureKpa * 0.72 + (101.2 + pressureDrift) * 0.28));
  const windSpeedMps = Math.max(0.4, physics.windSpeedMps * 0.55 + (1.5 + rng.next() * 5.2 + Math.abs(101.3 - airPressureKpa) * 0.9) * 0.45);
  const windDirectionDeg = (physics.windDirectionDeg * 0.72 + rng.nextInt(0, 359) * 0.28 + 360) % 360;
  const turbulence = clamp(windSpeedMps / 9 + rng.next() * 0.25);

  const seasonalHumidity = clock.season === 'winter' ? 0.48 : clock.season === 'summer' ? 0.52 : 0.62;
  const humidity = clamp(physics.humidity * 0.62 + (seasonalHumidity + rng.next() * 0.28 - airTemperatureC * 0.006) * 0.38);
  const stormLift = Math.max(0, 100.7 - airPressureKpa) * 0.18;
  const rainPotential = humidity + stormLift + (clock.season === 'spring' ? 0.08 : 0) - (clock.season === 'winter' ? 0.06 : 0);
  const rainfallMm = rainPotential > 0.72 ? (rainPotential - 0.68) * 34 + rng.next() * 5 : Math.max(0, (humidity - 0.64) * 4);
  const evaporationRate = clamp((airTemperatureC - 2) / 36 * (1 - humidity) + windSpeedMps * 0.012);
  const soilMoisture = clamp(physics.soilMoisture * 0.72 + (physics.soilMoisture + rainfallMm / 60 - evaporationRate * 0.55) * 0.28);
  const groundwater = clamp(physics.groundwater * 0.9 + (soilMoisture + rainfallMm / 100) * 0.1);

  const organicMatter = clamp(chemistry.organicMatter * 0.96 + (soilMoisture * 0.05) + (settlement.resources.food / 800) - 0.02);
  const fermentationTemp = idealTemperatureFactor(airTemperatureC, 8, 34);
  const fermentation = clamp(chemistry.fermentation * 0.65 + organicMatter * humidity * fermentationTemp * 0.26 + populationPressure * 0.04);
  const carbonDioxidePpm = Math.round(410 + populationPressure * 45 + fermentation * 160 + organicMatter * 45);
  const oxygenRatio = clamp(0.209 - fermentation * 0.006 + windSpeedMps * 0.0004, 0.198, 0.214);
  const soilPH = clampPH(chemistry.soilPH * 0.88 + (6.9 - fermentation * 0.55 + settlement.resources.medicine / 600) * 0.12);
  const mineralSaturation = clamp(chemistry.mineralSaturation * 0.83 + (groundwater * 0.32 + settlement.resources.ore / 400) * 0.17);
  const dissolvedIron = clamp(chemistry.dissolvedIron * 0.82 + (mineralSaturation * soilMoisture * (soilPH < 6.4 ? 1.2 : 0.8)) * 0.18);
  const corrosion = clamp(chemistry.corrosion * 0.62 + (humidity * 0.28 + rainfallMm / 70 + dissolvedIron * 0.12 + Math.max(0, 6.4 - soilPH) * 0.05) * 0.38);

  const weather = chooseWeather(clock.season, airTemperatureC, humidity, rainfallMm, windSpeedMps);
  const resourceDeltas = computeResourceDeltas({
    airTemperatureC,
    soilMoisture,
    rainfallMm,
    corrosion,
    mineralSaturation,
    fermentation,
    soilPH,
    oxygenRatio,
  });
  const signals = buildSignals(weather, resourceDeltas, { soilPH, corrosion, fermentation, mineralSaturation });

  const environment: EnvironmentState = {
    regionId: settlement.regionId,
    season: clock.season,
    weather,
    physics: {
      airTemperatureC,
      humidity,
      rainfallMm,
      windSpeedMps,
      windDirectionDeg,
      airPressureKpa,
      soilMoisture,
      groundwater,
      evaporationRate,
      turbulence,
    },
    chemistry: {
      oxygenRatio,
      carbonDioxidePpm,
      soilPH,
      mineralSaturation,
      dissolvedIron,
      organicMatter,
      fermentation,
      corrosion,
    },
    signals,
  };

  return {
    environment,
    resourceDeltas,
    events: buildEvents(environment),
  };
}

function chooseWeather(
  season: Season,
  airTemperatureC: number,
  humidity: number,
  rainfallMm: number,
  windSpeedMps: number,
): EnvironmentState['weather'] {
  if (season === 'winter' && airTemperatureC < 2) return 'frost';
  if (rainfallMm > 12 && windSpeedMps > 6.5) return 'storm';
  if (rainfallMm > 2.5) return 'rain';
  if (airTemperatureC > 29 && humidity < 0.42) return 'dry_heat';
  if (humidity > 0.76) return 'mist';
  return 'clear';
}

function computeResourceDeltas(input: {
  airTemperatureC: number;
  soilMoisture: number;
  rainfallMm: number;
  corrosion: number;
  mineralSaturation: number;
  fermentation: number;
  soilPH: number;
  oxygenRatio: number;
}): Partial<SettlementResources> {
  const growingTemp = idealTemperatureFactor(input.airTemperatureC, 4, 32);
  const pHHealth = clamp(1 - Math.abs(input.soilPH - 6.8) / 1.8);
  const moistureGain = (input.soilMoisture - 0.48) * 2.4;
  const floodPenalty = input.rainfallMm > 22 ? -1.3 : 0;
  const droughtPenalty = input.soilMoisture < 0.28 ? -1.2 : 0;

  return {
    food: moistureGain * growingTemp * pHHealth + floodPenalty + droughtPenalty,
    wood: (input.soilMoisture - 0.4) * 0.35,
    medicine: (input.soilMoisture * pHHealth - 0.35) * 0.45,
    tools: -input.corrosion * 0.18,
    ore: input.mineralSaturation * 0.12,
    publicMorale: input.fermentation > 0.5 ? -0.002 : 0.001,
    heartwellStability: input.oxygenRatio > 0.204 && pHHealth > 0.65 ? 0.002 : -0.002,
  };
}

function buildSignals(
  weather: EnvironmentState['weather'],
  deltas: Partial<SettlementResources>,
  chemistry: { soilPH: number; corrosion: number; fermentation: number; mineralSaturation: number },
): string[] {
  const signals = [weather.replace('_', ' ')];
  if ((deltas.food ?? 0) > 0.5) signals.push('crops respond to soil moisture');
  if ((deltas.food ?? 0) < -0.5) signals.push('crop stress');
  if (chemistry.corrosion > 0.55) signals.push('ironwork corrosion risk');
  if (chemistry.fermentation > 0.5) signals.push('cellars ferment strongly');
  if (chemistry.soilPH < 6.1) signals.push('soil acidity rising');
  if (chemistry.mineralSaturation > 0.65) signals.push('mineral-rich groundwater');
  return signals;
}

function buildEvents(environment: EnvironmentState): EnvironmentUpdate['events'] {
  const events: EnvironmentUpdate['events'] = [];
  if (environment.weather === 'storm' || environment.weather === 'rain') {
    events.push({
      type: 'environment_weather',
      payload: {
        weather: environment.weather,
        rainfallMm: Number(environment.physics.rainfallMm.toFixed(2)),
        windSpeedMps: Number(environment.physics.windSpeedMps.toFixed(2)),
      },
      tags: ['environment', 'weather', environment.weather],
      salience: environment.weather === 'storm' ? 0.72 : 0.45,
    });
  }
  if (environment.chemistry.corrosion > 0.6) {
    events.push({
      type: 'environment_chemistry',
      payload: {
        process: 'corrosion',
        corrosion: Number(environment.chemistry.corrosion.toFixed(3)),
        soilPH: Number(environment.chemistry.soilPH.toFixed(2)),
      },
      tags: ['environment', 'chemistry', 'corrosion'],
      salience: 0.58,
    });
  }
  if (environment.chemistry.fermentation > 0.62) {
    events.push({
      type: 'environment_chemistry',
      payload: {
        process: 'fermentation',
        fermentation: Number(environment.chemistry.fermentation.toFixed(3)),
        carbonDioxidePpm: environment.chemistry.carbonDioxidePpm,
      },
      tags: ['environment', 'chemistry', 'fermentation'],
      salience: 0.5,
    });
  }
  return events;
}

function hashString(value: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < value.length; i++) h = Math.imul(h ^ value.charCodeAt(i), 0x01000193);
  return h >>> 0;
}
