/** Deterministic ID generation from a counter + prefix. */

let _counter = 0;

export function resetIdCounter(): void { _counter = 0; }

export function nextId(prefix: string): string {
  return `${prefix}_${(++_counter).toString(36).padStart(6, '0')}`;
}

export function seededId(prefix: string, seed: number, salt: string): string {
  const hash = ((seed * 1664525 + 1013904223) ^ hashStr(salt)) >>> 0;
  return `${prefix}_${hash.toString(36)}`;
}

function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (Math.imul(h, 31) + s.charCodeAt(i)) >>> 0;
  }
  return h;
}
