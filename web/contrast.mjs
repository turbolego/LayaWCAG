// WCAG 2.x contrast-ratio helpers for LayaWCAG (pure JS — runs in Node and the browser).
// Used to build "computed-style hints" so the model can ground its 1.4.3 / 1.4.6 /
// 1.4.11 prediction in the *actual* foreground/background pair instead of guessing
// from markup. No model, no DOM: pass resolved rgb values (from getComputedStyle
// in the browser) or hex strings.
//
// Reference: WCAG 2.x "Relative Luminance" + contrast-ratio definition.
//   L = 0.2126*R + 0.7152*G + 0.0722*B   (R,G,B are linearized sRGB in [0,1])
//   ratio = (L_lighter + 0.05) / (L_darker + 0.05)

// Parse "#rgb", "#rrggbb" (optionally with alpha) into {r,g,b,a} in 0..255.
export function parseHex(hex) {
  let h = String(hex).trim().replace(/^#/, "");
  let a = 1;
  // Support #rrggbbaa
  if (h.length === 8) { a = parseInt(h.slice(6, 8), 16) / 255; h = h.slice(0, 6); }
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  if (h.length !== 6) throw new Error(`bad hex color: ${hex}`);
  const n = parseInt(h, 16);
  if (Number.isNaN(n)) throw new Error(`bad hex color: ${hex}`);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255, a };
}

// Accept "#rrggbb" | {r,g,b(,a)} | "rgb(1,2,3)" / "rgba(1,2,3,0.5)" → normalize.
export function toRgb(color) {
  if (color == null) throw new Error("color is null");
  if (typeof color === "string") {
    const s = color.trim();
    let m = s.match(/^rgba?\(([^)]+)\)/i);
    if (m) {
      const p = m[1].split(",").map((x) => parseFloat(x));
      return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 };
    }
    if (s[0] === "#") return parseHex(s);
    throw new Error(`unrecognized color: ${color}`);
  }
  if (typeof color === "object") return { r: color.r, g: color.g, b: color.b, a: color.a ?? 1 };
  throw new Error("unrecognized color type");
}

// Linearize a single 0..255 sRGB channel to [0,1].
function linearize(c) {
  const s = c / 255;
  return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
}

// Relative luminance of {r,g,b}.
export function relLuminance(rgb) {
  const { r, g, b } = toRgb(rgb);
  return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b);
}

// WCAG contrast ratio between two colors (order-independent). ~1.0 to 21.0.
export function contrastRatio(fg, bg) {
  const L1 = relLuminance(fg);
  const L2 = relLuminance(bg);
  const [hi, lo] = L1 >= L2 ? [L1, L2] : [L2, L1];
  return (hi + 0.05) / (lo + 0.05);
}

// WCAG "large text" = >= 18pt (24px) regular, or >= 14pt (18.66px) bold.
export function isLargeText(fontSizePx, fontWeight = 400) {
  if (fontSizePx >= 24) return true;
  if (fontSizePx >= 18.66 && fontWeight >= 700) return true;
  return false;
}

// Verdicts against the WCAG thresholds.
//   AA : 4.5:1 normal, 3:1 large
//   AAA: 7:1 normal,   4.5:1 large
export function contrastVerdict(fg, bg, { fontSizePx = 16, fontWeight = 400 } = {}) {
  const large = isLargeText(fontSizePx, fontWeight);
  const ratio = contrastRatio(fg, bg);
  const aaThreshold = large ? 3.0 : 4.5;
  const aaaThreshold = large ? 4.5 : 7.0;
  return {
    ratio,
    large,
    aa: ratio >= aaThreshold,
    aaa: ratio >= aaaThreshold,
    // The level at which the pair passes, or "fail".
    level: ratio >= aaaThreshold ? "AAA" : ratio >= aaThreshold ? "AA" : "fail",
  };
}

// Alpha-composite fg over a possibly-transparent bg chain, returning the
// effective opaque pair. `bg` may be {r,g,b,a}; if a < 1 it is composited over
// `pageBg` (opaque). Returns {fg, bg, ratio, verdict}.
export function resolvedContrast(fg, bg, pageBg = "#ffffff") {
  const f = toRgb(fg);
  const b = toRgb(bg);
  let effBg = { r: toRgb(pageBg).r, g: toRgb(pageBg).g, b: toRgb(pageBg).b, a: 1 };
  if (b.a < 1) {
    const a = b.a;
    effBg = {
      r: b.r * a + effBg.r * (1 - a),
      g: b.g * a + effBg.g * (1 - a),
      b: b.b * a + effBg.b * (1 - a),
      a: 1,
    };
  } else {
    effBg = { r: b.r, g: b.g, b: b.b, a: 1 };
  }
  const verdict = contrastVerdict(f, effBg);
  return { fg: f, bg: effBg, ...verdict };
}

// One hint line to prepend to a section's model input. `tag` is the element
// tag, `fg`/`bg` resolved colors. e.g.
//   [contrast] <h1> fg=#111 bg=#fff 15.9:1 OK(AA)
export function formatContrastHint(tag, fg, bg, opts = {}) {
  const v = contrastVerdict(fg, bg, opts);
  const tag2 = (t) => { const c = toRgb(t); return "#" + [c.r, c.g, c.b].map((x) => Math.round(x).toString(16).padStart(2, "0")).join(""); };
  const mark = v.aaa ? "OK(AAA)" : v.aa ? "OK(AA)" : "FAIL";
  return `[contrast] <${tag}> fg=${tag2(fg)} bg=${tag2(bg)} ${v.ratio.toFixed(1)}:1 ${mark}${v.large ? " large" : ""}`;
}

// Build a compact multi-line hint block for a set of text elements.
// `items`: [{ tag, fg, bg, fontSizePx, fontWeight }]
export function buildContrastHints(items) {
  return items
    .map((it) => formatContrastHint(it.tag, it.fg, it.bg, { fontSizePx: it.fontSizePx, fontWeight: it.fontWeight }))
    .join("\n");
}

// Prepend a contrast-hint block to the model input. This is the glue the demo
// uses: collect resolved fg/bg pairs from a rendered DOM (getComputedStyle on
// the live page, or inside a sandboxed same-origin iframe for pasted HTML),
// build the hint block, and inject it so the model grounds 1.4.3/1.4.6/1.4.11
// in the real ratio instead of guessing. If `items` is empty/absent the input
// is returned unchanged, so callers can opt in safely.
export function injectContrastHints(modelInput, items) {
  if (!items || !items.length) return modelInput;
  return buildContrastHints(items) + "\n" + modelInput;
}
