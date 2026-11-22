// Continuous-learning demo: the Laya WCAG decision model (WASM) + an LLM teacher.
// The loop (same shape as laya.Escalator, https://laya.readthedocs.io/en/latest/vegaml.html):
//   1. WASM decision model scores each candidate criterion -> p(violated)
//   2. criteria the model is not confident about (abstain) or that are out of its trained
//      distribution (unbound) are escalated to a real LLM teacher — AI Horde's free worker
//      network (often Qwen models, e.g. Qwen 3.8), a Qwen in this browser via
//      Transformers.js, or your own OpenAI-compatible endpoint
//   3. the teacher's verdict is learned in-browser: a small per-criterion delta head
//      (logistic update over example-signature features, Laya's continuous learner,
//      simplified to fit in a browser page) is re-fit after every escalation
//   4. re-score: the same criteria now resolve without the teacher -> LLM calls decay
//
// Model loading mirrors app.js: ONNX Runtime Web (WASM) + stitched external-data parts.
import * as ort from "./vendor/ort.min.mjs";
import { Tokenizer } from "./vendor/tokenizers.min.mjs";
import { buildWcagSequence, collateOne, loadCriteria } from "./laya-wcag-core.js";

const $ = (id) => document.getElementById(id);
const MODEL_DIR = "./model/";
const KEY_W = "laya-escalator-w";
const KEY_MEM = "laya-escalator-mem";
const KEY_TEA = "laya-escalator-teach";

// ---------- WCAG 2.2 examples (markup only, same constraint as the model) ----------
// cands: criteria the example is about. truth: ground truth used by the OFFLINE dev-teacher
// (1 = violation present, 0 = no violation). test: the strict one-liner sent to a live LLM.
const EXAMPLES = [
  {
    id: "img-no-alt", name: "Image without alt text",
    html: '<img src="chart.png" width="200">\n<p>Q3 results</p>',
    cands: [
      { id: "1.1.1", truth: 1, test: "Every <img> must have a non-empty alt attribute (text alternative)." },
      { id: "4.1.2", truth: 0, test: "Does any control/component lack an accessible name, role or state?" },
      { id: "1.3.1", truth: 0, test: "Is information, structure or relationships conveyed programmatically, and is heading/label order broken?" },
      { id: "1.4.5", truth: 0, test: "Is this an image of text that is the only way the content can be obtained (e.g. a text-only screenshot)?" },
    ],
  },
  {
    id: "input-no-label", name: "Form input without label",
    html: '<form><input type="email" name="email" placeholder="you@example.com"><button>Go</button></form>',
    cands: [
      { id: "3.3.2", truth: 1, test: "Does every input field have a persistent accessible label (<label for>, aria-label or similar)? A placeholder alone is not a label." },
      { id: "4.1.2", truth: 0, test: "Do all controls (inputs, buttons) expose an accessible name? The button's visible text counts as a name." },
      { id: "1.3.1", truth: 0, test: "Is the input purpose conveyed by name/role only, with no missing structure or relationship information?" },
      { id: "3.3.1", truth: 0, test: "Can the input purpose be identified programmatically? type=email helps here." },
    ],
  },
  {
    id: "heading-skip", name: "Heading level skipped",
    html: '<h1>Page title</h1>\n<h4>First section</h4>\n<p>Body text</p>',
    cands: [
      { id: "1.3.1", truth: 1, test: "Do heading levels follow without skipping (no h1 directly followed by h4)?" },
      { id: "2.4.6", truth: 1, test: "Do the headings describe the sections and form a proper hierarchy?" },
      { id: "2.4.1", truth: 0, test: "Is there a mechanism to bypass repeated blocks? Headings alone are not a bypass mechanism, and this fragment has none." },
    ],
  },
  {
    id: "iframe-no-title", name: "Iframe without title",
    html: '<iframe src="https://example.com/widget" width="600" height="400"></iframe>',
    cands: [
      { id: "4.1.2", truth: 1, test: "Does the embedded component (iframe) have an accessible name (title attribute)?" },
      { id: "1.1.1", truth: 0, test: "Is there non-text content lacking a text alternative? For frames the name comes via title, not alt, so this specific failure is not 1.1.1." },
      { id: "2.4.1", truth: 0, test: "Is there a mechanism to bypass repeated blocks? This fragment has no repeated navigation, so it does not fail." },
    ],
  },
  {
    id: "icon-link", name: "Icon-only link, no text",
    html: '<a href="/next"><img src="arrow.gif"></a>',
    cands: [
      { id: "1.1.1", truth: 1, test: "Every <img> must have a non-empty alt attribute. (Icon images inside links need alt.)" },
      { id: "2.4.4", truth: 1, test: "Is the purpose of this link identifiable from its text/name? An icon with no alt and no visible text is not identifiable." },
      { id: "4.1.2", truth: 0, test: "Does the link expose an accessible name, role and state? Judge component state only; the missing name is covered by 1.1.1 here." },
    ],
  },
  {
    id: "div-button", name: "role=button without name",
    html: '<div role="button" class="btn"></div>',
    cands: [
      { id: "4.1.2", truth: 1, test: "The element declares role=button. Does it expose an accessible name (text content, aria-label or aria-labelledby)?" },
      { id: "2.4.4", truth: 1, test: "Is the purpose of this button identifiable to the user? An empty div with no name is not identifiable." },
      { id: "1.3.1", truth: 0, test: "Is any structure or relationship information missing? The role is present; keyboard handling is not tested here." },
    ],
  },
  {
    id: "no-lang", name: "Page without lang attribute",
    html: '<html>\n<head><title>Report</title></head>\n<body>\n<main><h1>Q3</h1><p>Numbers</p></main>\n</body>\n</html>',
    cands: [
      { id: "3.1.1", truth: 1, test: "Does <html> declare a lang attribute so the page language is programmatically identifiable?" },
      { id: "1.3.1", truth: 0, test: "Is information, structure or relationships conveyed with correct markup?" },
      { id: "1.4.3", truth: 0, test: "Is there any text whose color contrast fails 4.5:1? No colors are present in this fragment, so it does not fail." },
    ],
  },
  {
    id: "clean", name: "Clean semantic snippet",
    html: '<main aria-label="Main">\n  <h1>Contact us</h1>\n  <label for="email">Email</label>\n  <input id="email" type="email" name="email">\n  <img src="logo.png" alt="Company logo">\n  <button type="submit">Send</button>\n</main>',
    cands: [
      { id: "1.1.1", truth: 0, test: "Every <img> must have a non-empty alt attribute." },
      { id: "3.3.2", truth: 0, test: "Does every input field have a persistent accessible label?" },
      { id: "4.1.2", truth: 0, test: "Do all controls expose an accessible name, role and state?" },
      { id: "1.4.3", truth: 0, test: "Is there text whose color contrast fails 4.5:1? No colors present, so it does not fail." },
    ],
  },
];

// Example-signature features the in-browser learner reasons over (binary, regex-based).
const F_NAMES = ["img without alt", "input without label", "heading skip", "iframe without title",
  "icon-only link", "role=button without name", "<html> without lang", "has form", "has input", "has label"];
function features(html) {
  const imgs = [...html.matchAll(/<img\b[^>]*\/?>/gi)];
  const imgNoAlt = imgs.some((m) => !/alt\s*=/i.test(m[0]));
  const iframes = [...html.matchAll(/<iframe\b[^>]*>/gi)];
  const iframeNoTitle = iframes.some((m) => !/title\s*=/i.test(m[0]));
  const inputs = (html.match(/<input\b/gi) || []).length;
  const labels = (html.match(/<label\b/gi) || []).length;
  const inputNoLabel = inputs > 0 && labels === 0;
  const hlevels = [...html.matchAll(/<h([1-6])\b/gi)].map((m) => +m[1]);
  let skip = false; for (let i = 1; i < hlevels.length; i++) if (hlevels[i] - hlevels[i - 1] > 1) skip = true;
  const linkIconOnly = [...html.matchAll(/<a\b[^>]*>[\s\S]*?<\/a>/gi)].some((m) => /^\s*<img/i.test(m[0]) && !/alt\s*=/i.test(m[0]));
  const roleBtnNoName = /role\s*=\s*["']button["']/i.test(html) && !/aria-label/i.test(html);
  const htmlTag = html.match(/<html\b[^>]*>/i);
  const noLang = htmlTag ? !/lang\s*=/i.test(htmlTag[0]) : false;
  return [imgNoAlt, inputNoLabel, skip, iframeNoTitle, linkIconOnly, roleBtnNoName, noLang,
    html.match(/<form\b/i) != null, inputs > 0, labels > 0].map(Boolean);
}

// ---------- The in-browser learner: a per-criterion delta head over example features ----------
// Laya's Learner re-trains its decision engine from confirmed answers on-device; this page
// keeps the frozen ONNX encoder and learns a small logistic delta per criterion instead.
// Same idea, sized for a browser tab: it generalizes across examples (the "iframe without
// title" pattern learned on example A resolves example B without another LLM call).
class DeltaLearner {
  constructor(nCrit, nFeat) {
    this.nCrit = nCrit; this.nFeat = nFeat;
    this.W = new Float64Array(nCrit * (nFeat + 1));
    this.mem = [];
    this.load();
  }
  deltaFor(c, phi) {
    const base = this.W.subarray(c * (this.nFeat + 1), (c + 1) * (this.nFeat + 1));
    let d = base[this.nFeat];
    for (let f = 0; f < this.nFeat; f++) d += base[f] * phi[f];
    return d;
  }
  push(phi, c, label) {
    this.mem.push({ phi, c, label });
    if (this.mem.length > 256) this.mem.shift();
  }
  hasMemory(c) { return this.mem.some((r) => r.c === c); }
  fit(c) {
    const rows = this.mem.filter((r) => r.c === c);
    if (!rows.length) return;
    const nF = this.nFeat, lr = 0.8, l2 = 0.05, iters = 300;
    const w = new Float64Array(nF + 1);
    for (let it = 0; it < iters; it++) {
      const g = new Float64Array(nF + 1);
      for (const r of rows) {
        let z = w[nF];
        for (let f = 0; f < nF; f++) z += w[f] * r.phi[f];
        const p = 1 / (1 + Math.exp(-z));
        const err = p - r.label;
        for (let f = 0; f < nF; f++) g[f] += err * r.phi[f];
        g[nF] += err;
      }
      const m = rows.length;
      for (let f = 0; f <= nF; f++) {
        g[f] /= m;
        if (f > 0) g[f] += l2 * w[f];
        w[f] -= lr * g[f];
      }
    }
    for (let f = 0; f <= nF; f++) w[f] = Math.max(-6, Math.min(6, w[f])); // keep deltas bounded
    this.W.set(w, c * (nF + 1));
    this.save();
  }
  size() { let n = 0; for (let c = 0; c < this.nCrit; c++) for (let f = 0; f <= this.nFeat; f++) if (Math.abs(this.W[c * (this.nFeat + 1) + f]) > 1e-4) { n++; break; } return n; }
  save() {
    try {
      const active = [];
      for (let c = 0; c < this.nCrit; c++) {
        let on = false;
        for (let f = 0; f <= this.nFeat; f++) if (Math.abs(this.W[c * (this.nFeat + 1) + f]) > 1e-4) { on = true; break; }
        if (on) active.push([c, Array.from(this.W.subarray(c * (this.nFeat + 1), (c + 1) * (this.nFeat + 1)))]);
      }
      localStorage.setItem(KEY_W, JSON.stringify({ active, mem: this.mem.slice(-128) }));
    } catch { /* private mode: learning just won't persist */ }
  }
  load() {
    try {
      const raw = localStorage.getItem(KEY_W);
      if (!raw) return;
      const d = JSON.parse(raw);
      for (const [c, w] of d.active || []) this.W.set(w, c * (this.nFeat + 1));
      this.mem = (d.mem || []).slice(-256);
    } catch { /* corrupt state: start fresh */ }
  }
  reset() { this.W = new Float64Array(this.nCrit * (this.nFeat + 1)); this.mem = []; localStorage.removeItem(KEY_W); }
}

// ---------- Model loading (same pattern as app.js) ----------
async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}. Has the model been built?`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; onProgress?.(got, total); }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}
async function fetchParts(entry, onProgress) {
  let cache = null;
  try { cache = await caches.open("laya-escalator-" + entry.sha256.slice(0, 16)); } catch { /* no cache in private mode */ }
  const out = new Uint8Array(entry.size); let off = 0, fromCache = 0;
  for (const part of entry.parts) {
    const url = MODEL_DIR + part;
    let bytes = null;
    try { const hit = cache && await cache.match(url); if (hit) { bytes = new Uint8Array(await hit.arrayBuffer()); fromCache++; } } catch { /* fall through */ }
    if (!bytes) {
      bytes = await fetchBytes(url, (got) => onProgress(off + got, entry.size));
      try { if (cache) await cache.put(url, new Response(bytes)); } catch { /* quota: fine */ }
    }
    if (off + bytes.length > entry.size) throw new Error("weights are larger than the manifest says");
    out.set(bytes, off); off += bytes.length; onProgress(off, entry.size);
  }
  if (off !== entry.size) throw new Error(`weights incomplete: got ${off} of ${entry.size} bytes`);
  fetchParts.lastFromCache = fromCache;
  return out;
}
function specialIds(tok) {
  const ids = { cls: tok.token_to_id("[CLS]"), sep: tok.token_to_id("[SEP]") };
  ids.pad = tok["pad_token_id"] != null ? tok["pad_token_id"] : tok.token_to_id("[PAD]");
  return ids;
}
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

let ortOk = false;
let tokenizer = null, sp = null, criteria = null, session = null;
let learner = null;
const critIndex = {};

async function loadModel() {
  $("eloadBtn").disabled = true; $("runBtn").disabled = true; $("autoBtn").disabled = true;
  $("ebadges").innerHTML = "";
  try { await loadModelInner(); } catch (e) {
    console.error(e); setStatus("Could not load the model: " + (e?.message || e), "warn");
  } finally { setProgress(null); $("eloadBtn").disabled = false; }
}
async function loadModelInner() {
  const want = $("backend").value === "webgpu" ? "webgpu" : "wasm";
  ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

  setStatus("Loading tokenizer, config and criteria list…"); setProgress(0);
  const [tj, tc, crit, manifest] = await Promise.all(
    ["tokenizer.json", "tokenizer_config.json", "wcag_criteria.json", "manifest.json"].map((f) =>
      fetch(MODEL_DIR + f).then((r) => { if (!r.ok) throw new Error(f + ": HTTP " + r.status); return r.json(); })));
  tokenizer = new Tokenizer(tj, tc);
  sp = specialIds(tokenizer);
  criteria = crit;
  criteria.forEach((c, i) => (critIndex[c.criterion] = i));
  const v = manifest.variants.wcag;
  if (!v) throw new Error("manifest.json has no \"wcag\" variant");

  const t0 = performance.now();
  const progress = (name) => (got, total) => {
    setProgress(total ? got / total : null);
    setStatus(`Downloading ${name}: ${(got / 1048576).toFixed(0)}${total ? " / " + (total / 1048576).toFixed(0) : ""} MB — one-time, cached afterwards`);
  };
  const graph = await fetchBytes(MODEL_DIR + v.onnx, progress("model graph"));
  const data = await fetchParts(v.data, progress("weights"));
  setStatus("Creating ONNX session (WASM)…");
  session = await ort.InferenceSession.create(graph, {
    executionProviders: want === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
    graphOptimizationLevel: "all",
    // ORT-web mounts preloaded external data under the *location string inside the ONNX graph* (v.data.name).
    externalData: [{ path: v.data.name, data }],
  });
  // Warm up: first WASM compile + a forward pass.
  await rawLogits("<p>warm up</p>");
  const ms = performance.now() - t0;
  setProgress(null); setStatus(`Ready — ${data.length / 1048576 | 0} MB in ${(ms / 1000).toFixed(1)} s (first load caches the parts in your browser)`);
  badge(`WASM · ${data.length / 1048576 | 0} MB · ${(ms / 1000).toFixed(1)} s`);
  badge("cross-origin isolated: " + (self.crossOriginIsolated ? "yes" : "no"));
  // Out-of-bounds probe: which criteria does this model actually respond to?
  await probeBounds();
  $("runBtn").disabled = false; $("autoBtn").disabled = false;
  // Test/inspection hooks (harmless in production): module scope isn't visible to page scripts.
  window.__esc = {
    EXAMPLES, features, rawLogits, BOUNDS_MIN,
    criteria: () => criteria, bounds: () => bounds, learner: () => learner,
  };
  window.escReady = true; // the session is free for inference now
}

// Raw forward pass -> 63 logits (sigmoid is NOT applied; the learner works on logits).
async function rawLogits(text) {
  const seq = buildWcagSequence(tokenizer, sp, text, 512);
  const b = collateOne(seq, sp.pad);
  const feeds = {
    input_ids: new ort.Tensor("int64", b.ids, [1, b.L]),
    attention_mask: new ort.Tensor("int64", b.att, [1, b.L]),
  };
  const out = await session.run(feeds);
  return out.criterion_scores.data; // Float32Array(63)
}

// ---------- Out-of-bounds probe ----------
// A criterion the model was never trained on does not respond to anything: its p(violated)
// stays flat across very different examples. The response range (max p − min p) over the
// bundled examples is the detector — trained criteria move, untrained ones don't.
let bounds = null; // Float32Array(63) of response range (max p - min p) across the examples
const BOUNDS_MIN = 0.10; // trained criteria range ≥ 0.15; untrained stay within ~0.02 (measured)
async function probeBounds() {
  const hi = new Float32Array(criteria.length).fill(-Infinity);
  const lo = new Float32Array(criteria.length).fill(Infinity);
  for (const ex of EXAMPLES) {
    const logit = await rawLogits(ex.html);
    for (let i = 0; i < logit.length; i++) {
      const p = sigmoid(logit[i]);
      if (p > hi[i]) hi[i] = p;
      if (p < lo[i]) lo[i] = p;
    }
  }
  bounds = hi;
  for (let i = 0; i < bounds.length; i++) bounds[i] = hi[i] - lo[i];
  let responsive = 0; for (let i = 0; i < bounds.length; i++) if (bounds[i] >= BOUNDS_MIN) responsive++;
  badge(`${responsive} of 63 criteria respond to the bundled examples (in-bounds)`);
  setStatus(`Ready. ${responsive} criteria respond to the examples; the other ${criteria.length - responsive} are unbound — the model abstains there by design and the teacher answers.`);
}

// ---------- Teachers ----------
// Four teacher backends:
//  horde   - AI Horde: a REAL Qwen-family model on the free public worker network (no API key).
//            submit POST /generate/text/async, poll GET /generate/text/status/<id>.
//  wasm    - a REAL Qwen loaded in this browser via Transformers.js (ONNX over WASM/WebGPU).
//  live    - your own OpenAI-compatible endpoint (key stays in your browser's localStorage).
//  offline - deterministic ground truth for the bundled examples. NOT a real model: dev/testing only.
function teacherCfg() {
  let c = { mode: "horde", base: "", key: "", model: "", hordeModel: "", wasmModel: "" };
  try { c = { ...c, ...JSON.parse(localStorage.getItem(KEY_TEA) || "{}") }; } catch { /* fresh */ }
  return c;
}
function saveTeacherCfg(c) { localStorage.setItem(KEY_TEA, JSON.stringify(c)); }

// Shared strict yes/no prompt (used by every real-teacher mode).
function buildVerdictPrompt(ex, cd) {
  const sys = "You are a senior WCAG 2.2 accessibility auditor answering yes/no verdicts for an automated pipeline. Be strict and literal: judge only what the fragment shows; missing required markup is a violation. Answer with exactly one word: YES (the criterion IS violated) or NO (it is not violated, or it does not apply to this fragment).";
  const user = `HTML fragment:\n\`\`\`html\n${ex.html}\n\`\`\`\n\nCriterion ${cd.id} — ${cd.test}\nQuestion: Does this fragment violate this criterion?`;
  return { sys, user };
}

// ---- AI Horde (default, real) ----
const HORDE_API = "https://aihorde.net/api/v2";
const HORDE_HD = { "Content-Type": "application/json", apikey: "0000000000", "Client-Agent": "laya-wcag-escalator-demo/1.0" };
let hordeModels = [];
async function loadHordeModels() {
  const r = await fetch(`${HORDE_API}/status/models?type=text`, { headers: { ...HORDE_HD } });
  if (!r.ok) throw new Error(`AI Horde models: HTTP ${r.status}`);
  hordeModels = (await r.json()).filter((m) => m.type === "text");
  return hordeModels;
}
// Default: prefer a real Qwen worker — lowest queue ETA first, then most workers,
// then a Qwen 3.8-class name when tied. (?? not ||: eta 0 is a real "empty queue".)
function pickDefaultHordeModel() {
  const rank = (m) => (m.eta ?? 1e9) * 1000 - (m.count || 0); // ETA dominates, workers break ties
  const qw = hordeModels.filter((m) => /qwen/i.test(m.name)).sort((a, b) => rank(a) - rank(b));
  if (qw.length) return qw[0].name;
  const all = [...hordeModels].sort((a, b) => rank(a) - rank(b));
  return all[0] ? all[0].name : "";
}
const sleep = (ms, signal) => new Promise((res, rej) => {
  const t = setTimeout(res, ms);
  if (signal) signal.addEventListener("abort", () => { clearTimeout(t); rej(new DOMException("stopped", "AbortError")); }, { once: true });
});
async function askHorde(cfg, ex, cd, signal, onProgress) {
  const model = cfg.hordeModel || (hordeModels.length ? pickDefaultHordeModel() : "Qwen/Qwen3.8-Flash-Next");
  const { sys, user } = buildVerdictPrompt(ex, cd);
  const submit = await fetch(`${HORDE_API}/generate/text/async`, {
    method: "POST", headers: { ...HORDE_HD }, signal,
    body: JSON.stringify({
      prompt: sys + "\n\n" + user, models: [model], context_fit: "reject",
      params: { max_context_length: 2048, max_length: 16, temperature: 0.0, top_p: 0.9, rep_pen: 1.05 },
    }),
  });
  const sj = await submit.json().catch(() => null);
  if (!submit.ok) throw new Error(`AI Horde submit HTTP ${submit.status}${sj?.message ? ": " + sj.message : ""}`);
  if (!sj || !sj.id) throw new Error("AI Horde accepted no job ID (worker queue may be full — try another model or later).");
  onProgress?.(`AI Horde · ${model} — queued (job ${sj.id.slice(0, 8)}…)`);
  const deadline = Date.now() + 10 * 60 * 1000; // free queues move slowly; cap per escalation
  while (Date.now() < deadline) {
    await sleep(4000, signal);
    const sr = await fetch(`${HORDE_API}/generate/text/status/${encodeURIComponent(sj.id)}`, { headers: { apikey: HORDE_HD.apikey }, signal });
    const s = await sr.json().catch(() => null);
    if (!sr.ok) throw new Error(`AI Horde status HTTP ${sr.status}`);
    if (s.faulted) throw new Error("AI Horde faulted this generation — try another model.");
    if (s.done) {
      const text = (s.generations?.[0]?.text || "").trim();
      const label = parseVerdict(text);
      if (label == null) throw new Error("model answer was not a clear YES/NO: " + text.slice(0, 80));
      return label;
    }
    onProgress?.(`AI Horde · ${model} — queue position ${s.queue_position ?? "?"} (${Math.round((Date.now() - (deadline - 10 * 60 * 1000)) / 1000)}s waiting…)`);
  }
  throw new Error("AI Horde queue did not clear in 10 min — stop and retry, or pick a faster model.");
}

// ---- Qwen in this browser (Transformers.js, real local model) ----
let wasmQwen = null; // { pipe, repo }
const WASM_PRESETS = ["onnx-community/Qwen2.5-0.5B", "onnx-community/Qwen2.5-1.5B"];
async function loadWasmQwen(repo, onProgress) {
  if (wasmQwen && wasmQwen.repo === repo) return wasmQwen;
  const mod = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.1/dist/transformers.min.js");
  const { pipeline, AutoTokenizer, env } = mod;
  if (navigator.gpu) { try { env.backends.onnx.webgpu = { enabled: true }; } catch { /* fall back to WASM */ } }
  onProgress?.(`Loading ${repo} tokenizer…`);
  const tokenizer = await AutoTokenizer.from_pretrained(repo);
  onProgress?.(`Loading ${repo} model weights…`);
  const pipe = await pipeline("text-generation", repo, { progress_callback: onProgress });
  onProgress?.("Running warm-up pass…");
  await pipe(tokenizer.apply_chat_template("hi", { add_generation_prompt: false }), { max_new_tokens: 1, do_sample: false });
  wasmQwen = { pipe, tokenizer, repo };
  return wasmQwen;
}
async function askWasm(ex, cd) {
  if (!wasmQwen) throw new Error("In-browser Qwen not loaded yet — press “Load Qwen in browser” first.");
  const { sys, user } = buildVerdictPrompt(ex, cd);
  const prompt = wasmQwen.tokenizer.apply_chat_template([
    { role: "system", content: sys }, { role: "user", content: user },
  ], { add_generation_prompt: true });
  const out = await wasmQwen.pipe(prompt, { max_new_tokens: 16, do_sample: false });
  const text = (out[0]?.generated_text || "").trim();
  const label = parseVerdict(text);
  if (label == null) throw new Error("model answer was not a clear YES/NO: " + text.slice(0, 80));
  return label;
}

// ---- Custom OpenAI-compatible endpoint (your key, your provider) ----
async function askLiveLLM(cfg, ex, cd, signal) {
  const base = cfg.base.replace(/\/+$/, "");
  if (!base) throw new Error("Live mode needs a base URL (e.g. https://openrouter.ai/api/v1).");
  const { sys, user } = buildVerdictPrompt(ex, cd);
  const doCall = async () => {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 60_000);
    const onAbort = () => ctl.abort();
    if (signal) signal.addEventListener("abort", onAbort, { once: true });
    try {
      const r = await fetch(base + "/chat/completions", {
        method: "POST",
        signal: ctl.signal,
        headers: { "Content-Type": "application/json", ...(cfg.key ? { Authorization: "Bearer " + cfg.key } : {}) },
        body: JSON.stringify({ model: cfg.model, messages: [{ role: "system", content: sys }, { role: "user", content: user }], temperature: 0, max_tokens: 8 }),
      });
      if (!r.ok) throw new Error(`LLM HTTP ${r.status}`);
      const d = await r.json();
      return (d.choices?.[0]?.message?.content || "").trim();
    } finally { clearTimeout(timer); if (signal) signal.removeEventListener("abort", onAbort); }
  };
  for (let attempt = 0; attempt < 2; attempt++) {
    const text = await doCall();
    const label = parseVerdict(text);
    if (label != null) return label;
  }
  throw new Error("LLM answer was not a clear YES/NO: " + (text || "(no text)"))
}

async function askTeacher(ex, cd, signal, onProgress) {
  const cfg = teacherCfg();
  if (cfg.mode === "horde") {
    const label = await askHorde(cfg, ex, cd, signal, onProgress);
    return { label, source: "AI Horde · real model " + (cfg.hordeModel || "auto") };
  }
  if (cfg.mode === "wasm") {
    const label = await askWasm(ex, cd);
    return { label, source: "in-browser Qwen (Transformers.js, " + (wasmQwen ? wasmQwen.repo : "not loaded") + ")" };
  }
  if (cfg.mode === "live") {
    const label = await askLiveLLM(cfg, ex, cd, signal);
    return { label, source: "live LLM · " + cfg.model };
  }
  // offline: deterministic, NOT a real model
  await sleep(300 + Math.random() * 250);
  return { label: cd.truth, source: "offline deterministic table (NOT a real LLM — dev testing only)" };
}

function parseVerdict(text) {
  const t = (text || "").toUpperCase().trim();
  if (!t) return null;
  const iYes = t.indexOf("YES"), iNo = t.search(/\bNO\b/);
  if (iYes === -1 && iNo === -1) return /VIOLAT|TRUE|FAIL|\b1\b/.test(t) ? 1 : /NO VIOL|FALSE|PASS|\b0\b/.test(t) ? 0 : null;
  if (iYes === -1) return 0;
  if (iNo === -1) return 1;
  return iYes < iNo ? 1 : 0;
}

// ---------- State ----------
let stats = { calls: 0, modelResolved: 0, teacherResolved: 0, rounds: [] }; // rounds: [{calls, abstainPct}]
let running = false;
let runAbort = null; // AbortController for the active run (stops long Horde queue waits)

// Score one example: model pass -> escalate the uncertain/unbound -> learn -> re-score.
// Public entry (Run button): guarded so it can't interrupt an auto-run or itself.
// While a run is in flight the button becomes “Stop” (clicking it aborts the active round).
async function runExample(idx) {
  if (!session) return;
  if (running) { runAbort?.abort(); return; }
  runAbort = new AbortController();
  running = true; setRunButtons(true);
  try { await runExampleInner(idx); }
  catch (e) { if (e?.name !== "AbortError") addLog("Run stopped: " + e.message, "warn"); else addLog("Run stopped (you clicked Stop).", "warn"); }
  finally { running = false; runAbort = null; setRunButtons(false); }
}
function setRunButtons(isRunning) {
  $("runBtn").disabled = false; $("autoBtn").disabled = false;
  $("runBtn").textContent = isRunning ? "Stop (aborts long LLM waits)" : "Run (Ctrl+Enter)";
  $("autoBtn").textContent = isRunning ? "Stop" : "Auto-run all";
}
async function runExampleInner(idx) {
  const ex = EXAMPLES[idx];
  $("exhtml").textContent = ex.html;
  const phi = features(ex.html);
  const base = await rawLogits(ex.html);

  // Model pass: classify each candidate criterion.
  const tau = parseFloat($("tau").value);
  const rows = ex.cands.map((cd) => {
    const i = critIndex[cd.id];
    const p0 = sigmoid(base[i] + learner.deltaFor(i, phi));
    // Unbound until the base model responds to it, OR the learner has memory for it (learning pulls it into bounds).
    const inBounds = bounds[i] >= BOUNDS_MIN || learner.hasMemory(i);
    const status = !inBounds ? "unbound" : (Math.max(p0, 1 - p0) >= tau ? "model" : "abstain");
    return { cd, i, p0, p1: p0, status, inBounds, learned: false };
  });
  const esc = rows.filter((r) => r.status !== "model");

  // Escalation: ask the teacher for every uncertain / unbound criterion.
  let calls = 0, failed = [];
  for (const r of esc) {
    if (runAbort?.signal.aborted) { failed.push(r); break; }
    addLog(`${ex.name} · ${r.cd.id} ${criteria[r.i].description}`, `${r.status} — asking the teacher…`, false);
    try {
      const t = await askTeacher(ex, r.cd, runAbort?.signal, (msg) => { $("kNote").textContent = msg; });
      calls++;
      learner.push(phi, r.i, t.label);
      r.label = t.label;
      r.learned = true;
      addLog(`${ex.name} · ${r.cd.id}`, `→ teacher said ${t.label ? "VIOLATION" : "no violation"} · ${t.source}`);
    } catch (e) {
      if (e?.name === "AbortError") { failed.push(r); break; }
      calls++; failed.push(r);
      addLog(`${ex.name} · ${r.cd.id}`, `escalation failed: ${e.message}`, true);
      r.status = "failed";
    }
  }
  for (const r of esc) if (r.learned) learner.fit(r.i);

  // Re-score with the learned deltas.
  for (const r of rows) {
    r.p1 = sigmoid(base[r.i] + learner.deltaFor(r.i, phi));
    if (r.learned) r.status = r.inBounds ? "abstain→learned" : "unbound→learned";
  }

  stats.calls += calls;
  const nModel = rows.filter((r) => r.status === "model").length;
  stats.modelResolved += nModel;
  stats.teacherResolved += rows.filter((r) => String(r.status).includes("learned")).length;
  $("kCalls").textContent = stats.calls;
  $("kCallsLast").textContent = calls;
  $("kModel").textContent = stats.modelResolved;
  $("kTeacher").textContent = stats.teacherResolved;
  const nEsc = esc.length;
  $("kAbstain").textContent = rows.length ? Math.round((nEsc / rows.length) * 100) + "%" : "0%";
  if (esc.length) $("kNote").textContent = `${calls} teacher call${calls === 1 ? "" : "s"} on this example${failed.length ? ` (${failed.length} failed/stopped)` : ""} — the verdicts are learned, so the next run of this example should need fewer.`;
  else $("kNote").textContent = "0 teacher calls — the model resolved every criterion on its own (no LLM needed).";

  renderVerdicts(ex, rows);
  return { total: rows.length, esc: esc.length, failed: failed.length };
}

// Auto-run: every example in order, one "round". Records the round for the decay chart.
async function autoRun() {
  if (!session) return;
  if (running) { runAbort?.abort(); return; } // second click = stop
  runAbort = new AbortController();
  running = true; setRunButtons(true);
  try {
    const calls0 = stats.calls;
    let escNow = 0, totNow = 0;
    for (let i = 0; i < EXAMPLES.length; i++) {
      $("exsel").value = String(i);
      $("kNote").textContent = `Running example ${i + 1}/${EXAMPLES.length} — ${EXAMPLES[i].name}…`;
      const res = await runExampleInner(i);
      escNow += res.esc; totNow += res.total;
    }
    const calls = stats.calls - calls0;
    stats.rounds.push({ calls, abstainPct: totNow ? Math.round((escNow / totNow) * 100) : 0 });
    if (stats.rounds.length > 12) stats.rounds.shift();
    drawChart();
    addLog(`Round ${stats.rounds.length}: ${calls} teacher call${calls === 1 ? "" : "s"} · ${stats.rounds[stats.rounds.length - 1].abstainPct}% of candidate criteria needed the teacher this pass`, false, true);
  } catch (e) {
    if (e?.name === "AbortError") addLog("Run stopped (queued LLM calls aborted).", "warn");
    else addLog("Run stopped: " + e.message, "warn");
  } finally {
    running = false; runAbort = null; setRunButtons(false);
  }
}

async function resetLearning() {
  learner.reset();
  stats = { calls: 0, modelResolved: 0, teacherResolved: 0, rounds: [] };
  ["kCalls", "kCallsLast", "kModel", "kTeacher"].forEach((id) => ($(id).textContent = "0"));
  $("kAbstain").textContent = "–"; $("kNote").textContent = "Learning reset — the next auto-run restarts the decay curve.";
  $("log").innerHTML = ""; $("chart").width = $("chart").width; drawChart();
  addLog("Learning reset (delta head zeroed, memory cleared).");
}

// ---------- UI ----------
function setStatus(msg, cls = "") { const s = $("estatus"); s.textContent = msg; s.className = "status " + cls; }
function setProgress(f) { const p = $("eprog"); p.style.display = f == null ? "none" : ""; if (f != null) $("eprogBar").style.width = Math.round(f * 100) + "%"; }
function badge(txt, ok = false) { const b = document.createElement("span"); b.className = "badge" + (ok ? " ok" : ""); b.textContent = txt; $("ebadges").appendChild(b); }
function addLog(text, kind = "info", head = false) {
  const li = document.createElement("li");
  li.className = kind + (head ? " head" : "");
  li.textContent = text;
  $("log").prepend(li);
  while ($("log").children.length > 80) $("log").lastChild.remove();
}

function renderVerdicts(ex, rows) {
  const tbl = $("verdicts");
  const crit = {}; criteria.forEach((c, i) => (crit[i] = c));
  const rowsHtml = rows.map((r) => {
    const c = crit[r.i];
    const verdict =
      r.status === "model" ? (r.p1 >= 0.5 ? "violation" : "no violation")
      : r.label != null ? (r.label ? "violation" : "no violation")
      : "—";
    const verdictCls = verdict === "violation" ? "v" : verdict === "no violation" ? "o" : "";
    const after = r.learned ? `<span class="pmove">${fmtPct(r.p0)} → <b>${fmtPct(r.p1)}</b></span>` : `<span class="pmove">${fmtPct(r.p1)}</span>`;
    return `<tr>
      <td class="cid">${r.cd.id}</td>
      <td class="cdesc">${esc(c.description)}<div class="ctest">${esc(r.cd.test)}</div></td>
      <td><span class="pbar"><span style="width:${Math.round(r.p0 * 100)}%"></span></span><span class="p">${fmtPct(r.p0)}</span></td>
      <td>${statusChip(r)}</td>
      <td>${after}</td>
      <td><span class="verdict ${verdictCls}">${verdict}</span></td>
    </tr>`;
  }).join("");
  tbl.innerHTML = `<table><thead><tr><th>Criterion</th><th>Strict test</th><th>p(violated) · model</th><th>Status</th><th>p(violated) · after learning</th><th>Verdict</th></tr></thead><tbody>${rowsHtml}</tbody></table>`;
}
function statusChip(r) {
  const map = {
    model: ["knows", "The model answered — no LLM call"],
    abstain: ["abstain", "Confidence below the line → escalated to the LLM"],
    unbound: ["unbound", "Out of the model's trained distribution → escalated to the LLM"],
    "abstain→learned": ["learned", "Abstained → LLM answer learned, re-scored"],
    "unbound→learned": ["learned", "Unbound → LLM answer learned, re-scored"],
    failed: ["failed", "Escalation failed"],
  };
  const [label, tip] = map[r.status] || [r.status, ""];
  return `<span class="chip ${label}" title="${esc(tip)}">${esc(label)}</span>`;
}
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const fmtPct = (x) => (x * 100).toFixed(0) + "%";

// Two-series decay chart: bars = teacher calls per auto-run round, line = % criteria still uncertain.
function drawChart() {
  const cv = $("chart");
  const ctx = cv.getContext("2d");
  const W = cv.width = cv.clientWidth || 600, H = cv.height = 200;
  ctx.clearRect(0, 0, W, H);
  const padL = 36, padB = 24, padT = 14, padR = 36;
  const rounds = stats.rounds;
  if (!rounds.length) {
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--ink2");
    ctx.font = "13px sans-serif";
    ctx.fillText("Run “Auto-run all” — each pass is a round. Watch the bars drop.", padL, H / 2);
    return;
  }
  const maxCalls = Math.max(1, ...rounds.map((r) => r.calls));
  const bw = (W - padL - padR) / rounds.length;
  ctx.font = "11px monospace";
  // bars: calls
  rounds.forEach((r, i) => {
    const h = (H - padT - padB) * (r.calls / maxCalls);
    const x = padL + i * bw + bw * 0.25;
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--bar-top") || "#2a5bd7";
    ctx.fillRect(x, H - padB - h, bw * 0.5, h);
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--ink");
    ctx.fillText(String(r.calls), x + bw * 0.15, H - padB - h - 4);
  });
  // line: abstain %
  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue("--warn") || "#b4570a";
  ctx.fillStyle = ctx.strokeStyle;
  ctx.lineWidth = 2;
  ctx.beginPath();
  rounds.forEach((r, i) => {
    const x = padL + i * bw + bw * 0.5;
    const y = padT + (H - padT - padB) * (1 - r.abstainPct / 100);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  rounds.forEach((r, i) => {
    const x = padL + i * bw + bw * 0.5;
    const y = padT + (H - padT - padB) * (1 - r.abstainPct / 100);
    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 7); ctx.fill();
    ctx.fillText(r.abstainPct + "%", x + 6, y - 6);
  });
  // x labels
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--ink2");
  rounds.forEach((_, i) => ctx.fillText("R" + (i + 1), padL + i * bw + bw * 0.4, H - 8));
  ctx.fillText("calls", padL - 26, padT);
  ctx.fillText("uncertain", W - padR + 4, padT);
}

// ---------- Wiring ----------
function initTeacherUI() {
  const cfg = teacherCfg();
  $("tmode").value = ["horde", "wasm", "live", "offline"].includes(cfg.mode) ? cfg.mode : "horde";
  $("tbase").value = cfg.base || "";
  $("tkey").value = cfg.key || "";
  $("tmodel").value = cfg.model || "";
  $("hmodel").value = cfg.hordeModel || "";
  $("wmodel").value = cfg.wasmModel || WASM_PRESETS[0];
  const persist = () => {
    const c = {
      mode: $("tmode").value,
      base: $("tbase").value, key: $("tkey").value, model: $("tmodel").value,
      hordeModel: $("hmodel").value, wasmModel: $("wmodel").value,
    };
    saveTeacherCfg(c);
    applyTeacherUI();
    addLog(
      c.mode === "horde" ? `Teacher: AI Horde (real model) — ${c.hordeModel || "auto (fastest Qwen)"}`
      : c.mode === "wasm" ? `Teacher: in-browser Qwen (${c.wasmModel || WASM_PRESETS[0]})`
      : c.mode === "live" ? `Teacher: live LLM @ ${c.base} (${c.model})`
      : "Teacher: OFFLINE deterministic table (no real model — dev testing only)");
  };
  $("tmode").addEventListener("change", persist);
  ["tbase", "tkey", "tmodel", "hmodel", "wmodel"].forEach((id) => $(id).addEventListener("change", persist));
  $("hrefresh").addEventListener("click", async () => {
    $("hstatus").textContent = "Loading AI Horde model list…";
    try {
      await loadHordeModels();
      fillHordeSelect();
      const d = teacherCfg(); d.hordeModel = $("hmodel").value || pickDefaultHordeModel(); $("hmodel").value = d.hordeModel;
      $("hstatus").textContent = `${hordeModels.length} real text models available · default: ${$("hmodel").value || pickDefaultHordeModel()}`;
      persist();
    } catch (e) { $("hstatus").textContent = "Could not reach AI Horde: " + e.message; addLog("Horde model list failed: " + e.message, "warn"); }
  });
  $("wload").addEventListener("click", async () => {
    const repo = $("wmodel").value || WASM_PRESETS[0];
    const cfg = teacherCfg(); cfg.wasmModel = repo; saveTeacherCfg(cfg);
    try {
      $("wstatus").textContent = `Loading ${repo} in your browser (one-time, ~100–400 MB)…`;
      await loadWasmQwen(repo, (ev) => {
        if (typeof ev === "string") $("wstatus").textContent = ev;
        else if (ev && ev.status === "progress" && ev.total) $("wstatus").textContent = `Downloading ${ev.file}: ${Math.round((ev.loaded / ev.total) * 100)}%`;
      });
      $("wstatus").textContent = `In-browser Qwen ready: ${repo} (WebGPU ${navigator.gpu ? "if available" : "unavailable"}, else WASM) — real local inference, nothing leaves your machine.`;
      addLog(`In-browser Qwen loaded: ${repo}`);
    } catch (e) {
      $("wstatus").textContent = "Could not load the in-browser model: " + e.message + " — try another preset or switch to AI Horde.";
      addLog("WASM Qwen load failed: " + e.message, "warn");
    }
  });
  applyTeacherUI();
}
function applyTeacherUI() {
  const m = $("tmode").value;
  $("hordeFields").style.display = m === "horde" ? "" : "none";
  $("wasmFields").style.display = m === "wasm" ? "" : "none";
  $("liveFields").style.display = m === "live" ? "" : "none";
  $("offlineNote").style.display = m === "offline" ? "" : "none";
}
function fillHordeSelect() {
  const sel = $("hmodel");
  const prev = sel.value;
  sel.innerHTML = "";
  const auto = document.createElement("option");
  auto.value = ""; auto.textContent = "Auto — fastest Qwen worker (lowest queue ETA)";
  sel.appendChild(auto);
  for (const m of hordeModels) {
    const o = document.createElement("option");
    o.value = m.name;
    o.textContent = `${m.name} — ETA ${m.eta != null ? m.eta + "s" : "?"} · ${m.queued} queued · ${m.count} worker${m.count === 1 ? "" : "s"}`;
    sel.appendChild(o);
  }
  if (hordeModels.length && !sel.selectedOptions?.[0]?.value) sel.value = prev && hordeModels.some((m) => m.name === prev) ? prev : pickDefaultHordeModel();
}
// Preload the Horde model list + pick a sensible default on page open (no key, free, real models).
initHordePreflight();
async function initHordePreflight() {
  const cfg = teacherCfg();
  if (cfg.mode === "offline") return;
  try {
    await loadHordeModels();
    fillHordeSelect();
    if (!cfg.hordeModel && !$("hmodel").value) { cfg.hordeModel = pickDefaultHordeModel(); $("hmodel").value = cfg.hordeModel; saveTeacherCfg(cfg); }
    const chosen = $("hmodel").value || pickDefaultHordeModel();
    const mm = hordeModels.find((m) => m.name === chosen);
    $("hstatus").textContent = `${hordeModels.length} real text models on AI Horde · default: ${chosen}${mm && mm.eta != null ? ` (ETA ~${mm.eta}s, ${mm.queued} queued)` : ""}`;
    applyTeacherUI();
  } catch (e) {
    $("hstatus").textContent = "AI Horde model list unavailable: " + e.message + " (the demo still works — submissions may be queued).";
  }
}

function init() {
  const sel = $("exsel");
  EXAMPLES.forEach((ex, i) => {
    const o = document.createElement("option");
    o.value = String(i); o.textContent = `${i + 1}. ${ex.name}`;
    sel.appendChild(o);
  });
  sel.addEventListener("change", () => { $("exhtml").textContent = EXAMPLES[+sel.value].html; });
  $("exhtml").textContent = EXAMPLES[0].html;

  $("tau").addEventListener("input", () => ($("tauVal").textContent = Number($("tau").value).toFixed(2)));
  $("eloadBtn").addEventListener("click", loadModel);
  $("runBtn").addEventListener("click", () => runExample(+$("exsel").value));
  $("autoBtn").addEventListener("click", autoRun);
  $("resetBtn").addEventListener("click", resetLearning);
  document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !$("runBtn").disabled) runExample(+$("exsel").value); });

  initTeacherUI();
  // The learner needs the criterion count; create it once criteria are known (loadModel does the work,
  // so build it eagerly — 63 is fixed by the spec, but keep it honest by defaulting to 63 until loaded).
  learner = new DeltaLearner(63, F_NAMES.length);
  if (learner.size()) addLog(`Restored in-browser learning: ${learner.size()} criterion delta heads, ${learner.mem.length} remembered answers.`);
  drawChart();
}
init();
