export function readFileSync(_path: string, _encoding?: string): string {
  throw new Error('fs.readFileSync is not available in the browser');
}
export function existsSync(_path: string): boolean { return false; }
export function readdirSync(_path: string): string[] { return []; }
