export function join(...parts: string[]): string {
  return parts.join('/').replace(/\/+/g, '/');
}
export function dirname(p: string): string {
  return p.substring(0, p.lastIndexOf('/')) || '.';
}
export function resolve(...parts: string[]): string {
  return join(...parts);
}
export function basename(p: string, ext?: string): string {
  const base = p.split('/').pop() ?? '';
  return ext && base.endsWith(ext) ? base.slice(0, -ext.length) : base;
}
export const sep = '/';
