/*
 * Dependency-free app-icon generator for Forge System OS.
 *
 * Renders the brand motif — three stacked upward chevrons in an ember→cyan
 * gradient (forge heat cooling into "system" blue, also reading as "level up")
 * on a deep-navy field — and encodes PNGs by hand using only Node's built-in
 * zlib. Produces every asset app.json references. Re-run with: node scripts/make-icons.js
 */
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");

// ---- palette ---------------------------------------------------------------
const NAVY_TOP = [11, 18, 32]; // #0B1220
const NAVY_BOT = [17, 26, 46]; // #111A2E
const AMBER = [245, 158, 11]; // #F59E0B
const SKY = [56, 189, 248]; // #38BDF8

const lerp = (a, b, t) => a + (b - a) * t;
const clamp01 = (t) => (t < 0 ? 0 : t > 1 ? 1 : t);
const mix = (c1, c2, t) => [lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)];
const smoothstep = (edge0, edge1, x) => {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
};

// distance from point P to segment AB, in pixels
function distToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 === 0 ? 0 : ((px - ax) * dx + (py - ay) * dy) / len2;
  t = clamp01(t);
  const cx = ax + t * dx;
  const cy = ay + t * dy;
  return Math.hypot(px - cx, py - cy);
}

// Build the three chevrons' segments for a canvas of size S, scaled by `scale`
// (fraction of S the motif occupies). Returns {segments, top, bottom, stroke}.
function buildMotif(S, scale) {
  const cx = S / 2;
  const hw = 0.26 * S * scale; // horizontal reach of each arm
  const dv = 0.2 * S * scale; // vertical drop of each arm
  const gap = 0.165 * S * scale; // vertical spacing between apexes
  const stroke = 0.085 * S * scale; // line thickness
  const motifH = 2 * gap + dv;
  const topApexY = S / 2 - motifH / 2; // vertically center the stack
  const segments = [];
  for (let k = 0; k < 3; k++) {
    const apexY = topApexY + k * gap;
    const apex = [cx, apexY];
    const left = [cx - hw, apexY + dv];
    const right = [cx + hw, apexY + dv];
    segments.push([apex, left], [apex, right]);
  }
  return { segments, top: topApexY, bottom: topApexY + 2 * gap + dv, stroke };
}

// Render an RGBA Uint8 buffer.
function render(S, { background }) {
  const motif = buildMotif(S, 1);
  const half = motif.stroke / 2;
  const buf = Buffer.alloc(S * S * 4);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      // nearest distance to any chevron stroke
      let d = Infinity;
      for (const [a, b] of motif.segments) {
        const dd = distToSegment(x + 0.5, y + 0.5, a[0], a[1], b[0], b[1]);
        if (dd < d) d = dd;
      }
      const aa = S * 0.0016; // ~1.6px feather at 1024
      const cov = smoothstep(half + aa, half - aa, d);

      // chevron gradient color by vertical position within the motif
      const gt = clamp01((y - motif.top) / (motif.bottom - motif.top));
      const chev = mix(AMBER, SKY, gt);

      let r, g, bl, al;
      if (background) {
        const bg = mix(NAVY_TOP, NAVY_BOT, y / (S - 1));
        r = lerp(bg[0], chev[0], cov);
        g = lerp(bg[1], chev[1], cov);
        bl = lerp(bg[2], chev[2], cov);
        al = 255;
      } else {
        r = chev[0];
        g = chev[1];
        bl = chev[2];
        al = Math.round(cov * 255);
      }
      const i = (y * S + x) * 4;
      buf[i] = Math.round(r);
      buf[i + 1] = Math.round(g);
      buf[i + 2] = Math.round(bl);
      buf[i + 3] = al;
    }
  }
  return buf;
}

// ---- minimal PNG encoder ---------------------------------------------------
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, "ascii");
  const crcBuf = Buffer.alloc(4);
  crcBuf.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([len, typeBuf, data, crcBuf]);
}
function encodePng(S, rgba) {
  const sig = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(S, 0);
  ihdr.writeUInt32BE(S, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // color type RGBA
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;
  // raw scanlines with filter byte 0
  const raw = Buffer.alloc(S * (S * 4 + 1));
  for (let y = 0; y < S; y++) {
    raw[y * (S * 4 + 1)] = 0;
    rgba.copy(raw, y * (S * 4 + 1) + 1, y * S * 4, (y + 1) * S * 4);
  }
  const idat = zlib.deflateSync(raw, { level: 9 });
  return Buffer.concat([sig, chunk("IHDR", ihdr), chunk("IDAT", idat), chunk("IEND", Buffer.alloc(0))]);
}

// ---- emit ------------------------------------------------------------------
const outDir = path.join(__dirname, "..", "assets");
fs.mkdirSync(outDir, { recursive: true });
const write = (name, S, opts) => {
  const png = encodePng(S, render(S, opts));
  fs.writeFileSync(path.join(outDir, name), png);
  console.log(`wrote assets/${name} (${S}x${S}, ${png.length} bytes)`);
};

write("icon.png", 1024, { background: true }); // iOS / general app icon
write("adaptive-icon.png", 1024, { background: false }); // Android foreground (bg color via app.json)
write("splash-icon.png", 1024, { background: false }); // splash logo (bg color via app.json)
write("favicon.png", 64, { background: true }); // web
console.log("done");
