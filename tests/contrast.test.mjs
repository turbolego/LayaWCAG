// Headless tests for web/contrast.mjs — the WCAG contrast-ratio + hint helpers.
//   node tests/contrast.test.mjs
import assert from "node:assert/strict";
import {
  parseHex, relLuminance, contrastRatio, isLargeText,
  contrastVerdict, resolvedContrast, formatContrastHint, buildContrastHints, injectContrastHints,
} from "../web/contrast.mjs";

let n = 0;
function t(name, fn) { n++; try { fn(); console.log(`  ok ${n} - ${name}`); } catch (e) { console.error(`  FAIL ${n} - ${name}\n    ${e.message}`); process.exitCode = 1; } }

// --- parseHex ---
t("parseHex #000 -> black", () => { const c = parseHex("#000"); assert.equal(c.r, 0); assert.equal(c.g, 0); assert.equal(c.b, 0); });
t("parseHex #ffffff -> white", () => { const c = parseHex("#ffffff"); assert.equal(c.r, 255); assert.equal(c.b, 255); });
t("parseHex #123 -> #112233", () => { const c = parseHex("#123"); assert.equal(c.r, 0x11); assert.equal(c.g, 0x22); assert.equal(c.b, 0x33); });
t("parseHex throws on garbage", () => assert.throws(() => parseHex("#zzz")));

// --- relLuminance ---
t("luminance black=0", () => assert.equal(relLuminance("#000000"), 0));
t("luminance white=1", () => assert.equal(relLuminance("#ffffff"), 1));
t("luminance red-only", () => { const L = relLuminance("#ff0000"); assert.ok(L > 0.2 && L < 0.25, `got ${L}`); });

// --- contrastRatio (anchor values) ---
t("ratio black/white = 21:1", () => { const r = contrastRatio("#000000", "#ffffff"); assert.ok(Math.abs(r - 21) < 1e-6, `got ${r}`); });
t("ratio same color = 1:1", () => assert.equal(contrastRatio("#123456", "#123456"), 1));
t("ratio symmetric", () => assert.equal(contrastRatio("#fff", "#000"), contrastRatio("#000", "#fff")));
t("ratio #767676/white ~4.54 (AA pass)", () => { const r = contrastRatio("#767676", "#ffffff"); assert.ok(r >= 4.5 && r < 4.6, `got ${r}`); });
t("ratio #aaa/white fails AA normal", () => { const r = contrastRatio("#aaaaaa", "#ffffff"); assert.ok(r < 4.5, `got ${r}`); });

// --- isLargeText ---
t("24px is large", () => assert.equal(isLargeText(24), true));
t("16px is not large", () => assert.equal(isLargeText(16), false));
t("18.66px bold is large", () => assert.equal(isLargeText(18.66, 700), true));
t("18.66px regular is not large", () => assert.equal(isLargeText(18.66, 400), false));

// --- contrastVerdict ---
t("verdict black/white AAA", () => { const v = contrastVerdict("#000", "#fff"); assert.equal(v.aaa, true); assert.equal(v.level, "AAA"); });
t("verdict #aaa/white fails AA normal", () => { const v = contrastVerdict("#aaa", "#fff"); assert.equal(v.aa, false); assert.equal(v.level, "fail"); });
t("verdict #767676/white: AA normal, AAA when large", () => {
  const normal = contrastVerdict("#767676", "#ffffff");
  assert.equal(normal.aa, true);      // 4.54 >= 4.5
  assert.equal(normal.aaa, false);    // 4.54 < 7.0
  assert.equal(normal.level, "AA");
  const large = contrastVerdict("#767676", "#ffffff", { fontSizePx: 24 });
  assert.equal(large.aaa, true);      // 4.54 >= 4.5 (large-text AAA)
  assert.equal(large.level, "AAA");
});

// --- resolvedContrast (alpha compositing) ---
t("opaque bg is pass-through", () => { const r = resolvedContrast("#000", "#ffffff", "#fff"); assert.ok(Math.abs(r.ratio - 21) < 1e-6); });
t("semi-transparent bg composites over pageBg", () => {
  // fg black over a 50% gray on white page -> effective bg is #808080-ish
  const r = resolvedContrast("#000", { r: 128, g: 128, b: 128, a: 0.5 }, "#ffffff");
  // effective bg luminance is between 0.2126*... and 1; ratio must be < 21
  assert.ok(r.ratio < 21 && r.ratio > 3, `got ${r.ratio}`);
});

// --- formatContrastHint ---
t("hint format includes ratio + verdict", () => {
  const h = formatContrastHint("h1", "#111111", "#ffffff");
  assert.ok(h.startsWith("[contrast] <h1>"), h);
  assert.ok(/:1 OK\(A/.test(h) || /:1 FAIL/.test(h), h);
  assert.match(h, /fg=#111111 bg=#ffffff/);
});
t("hint marks low-contrast FAIL", () => {
  const h = formatContrastHint("p", "#aaaaaa", "#ffffff");
  assert.match(h, /FAIL/);
});
t("hint marks large", () => {
  const h = formatContrastHint("h2", "#111111", "#ffffff", { fontSizePx: 32 });
  assert.match(h, /large/);
});

// --- buildContrastHints ---
t("buildContrastHints joins lines", () => {
  const out = buildContrastHints([
    { tag: "h1", fg: "#111", bg: "#fff" },
    { tag: "p", fg: "#aaa", bg: "#fff" },
  ]);
  const lines = out.split("\n");
  assert.equal(lines.length, 2);
  assert.match(lines[0], /\[contrast\] <h1>/);
  assert.match(lines[1], /\[contrast\] <p>/);
});

// --- injectContrastHints ---
t("injectContrastHints prepends hint block", () => {
  const out = injectContrastHints("<h1>T</h1>", [{ tag: "h1", fg: "#111", bg: "#fff" }]);
  assert.ok(out.startsWith("[contrast] <h1>"), out);
  assert.ok(out.endsWith("<h1>T</h1>"), out);
  assert.ok(out.includes("\n<h1>T</h1>"), out);
});
t("injectContrastHints is a no-op when no items", () => {
  assert.equal(injectContrastHints("<p>x</p>", []), "<p>x</p>");
  assert.equal(injectContrastHints("<p>x</p>", undefined), "<p>x</p>");
  assert.equal(injectContrastHints("<p>x</p>", null), "<p>x</p>");
});

console.log(`\n${n} tests ${process.exitCode ? "FAILED" : "passed"}`);
