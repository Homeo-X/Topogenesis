/**
 * Deterministic seeded PRNG using a mulberry32 algorithm.
 * All simulation randomness MUST flow through a SeededRng instance
 * so that runs are reproducible given the same seed.
 */

export type SeededRng = {
  next(): number;           // [0, 1)
  nextInt(min: number, max: number): number;  // [min, max]
  nextBool(p?: number): boolean;
  nextChoice<T>(arr: readonly T[]): T;
  nextWeighted<T>(items: readonly T[], weights: readonly number[]): T;
  fork(salt: string): SeededRng;
  seed: number;
};

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(str: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = (Math.imul(h, 0x01000193)) >>> 0;
  }
  return h;
}

export function createRng(seed: number): SeededRng {
  const gen = mulberry32(seed);

  const rng: SeededRng = {
    seed,
    next() { return gen(); },
    nextInt(min, max) {
      return Math.floor(rng.next() * (max - min + 1)) + min;
    },
    nextBool(p = 0.5) { return rng.next() < p; },
    nextChoice<T>(arr: readonly T[]): T {
      if (arr.length === 0) throw new Error('Empty array in nextChoice');
      return arr[Math.floor(rng.next() * arr.length)] as T;
    },
    nextWeighted<T>(items: readonly T[], weights: readonly number[]): T {
      const total = weights.reduce((a, b) => a + b, 0);
      let r = rng.next() * total;
      for (let i = 0; i < items.length; i++) {
        r -= weights[i] as number;
        if (r <= 0) return items[i] as T;
      }
      return items[items.length - 1] as T;
    },
    fork(salt: string): SeededRng {
      const newSeed = (hashString(salt) ^ (seed * 1664525 + 1013904223)) >>> 0;
      return createRng(newSeed);
    },
  };
  return rng;
}
