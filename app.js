import * as ort from "./vendor/ort.min.mjs";
import { Tokenizer } from "./vendor/tokenizers.min.mjs";
import { WcagLaya, loadCriteria } from "./laya-wcag-core.js";

const $ = (id) => document.getElementById(id);
// One bundled checkpoint: the WCAG 2.2 fine-tune of Laya's encoder, served from ./model/.
// Override with ?modelBase=<url-with-trailing-slash> (needs CORS if it points elsewhere).
let MODEL_DIR = new URLSearchParams(location.search).get("modelBase") || "./model/";
MODEL_DIR = MODEL_DIR.replace(/\/?$/, "/");

// Example HTML snippets that exercise common violation types. The model only sees the markup
// text (no CSS, no rendered layout), so contrast/layout-style checks are inherently weak here.
const EXAMPLES = {
  "Image without alt text": '<img src="chart.png" width="200">\n<p>Q3 results</p>',
  "Button with no accessible name": '<div role="button" class="btn"></div>',
  "Form input without label": '<form><input type="email" name="email" placeholder="you@example.com"><button>Go</button></form>',
  "Heading level skipped": '<h1>Page title</h1>\n<h4>First section</h4>\n<p>Body text</p>',
  "Link with no discernible text": '<a href="/next"><img src="arrow.gif"></a>',
  "Iframe without title": '<iframe src="https://example.com/widget" width="600" height="400"></iframe>',
  "Clean semantic snippet": '<main aria-label="Main">\n  <h1>Contact us</h1>\n  <label for="email">Email</label>\n  <input id="email" type="email" name="email">\n  <img src="logo.png" alt="Company logo">\n  <button type="submit">Send</button>\n</main>',
};

let wcag = null;
let backendUsed = "";
let lastResult = null;

function setStatus(msg, cls = "") { const s = $("status"); s.textContent = msg; s.className = "status " + cls; }
function setProgress(f) { $("progress").style.display = f == null ? "none" : "block"; if (f != null) $("progressBar").style.width = (f * 100).toFixed(1) + "%"; }
function badge(text, ok = false) { const b = document.createElement("span"); b.className = "badge" + (ok ? " ok" : ""); b.textContent = text; $("badges").appendChild(b); }

async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}. Has the model been built? See the README.`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onProgress?.(got, total);
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

// The weight file is stored as <= 24 MiB parts (hosting limits); stitch them back together here.
// Parts are kept in the browser's Cache Storage, keyed by the file's hash, so a repeat visit does not download them again.
async function fetchParts(entry, onProgress) {
  let cache = null;
  try { cache = await caches.open("laya-wcag-" + entry.sha256.slice(0, 16)); } catch { /* private mode etc.: just download */ }
  const out = new Uint8Array(entry.size); let off = 0, fromCache = 0;
  for (const part of entry.parts) {
    const url = MODEL_DIR + part;
    let bytes = null;
    try { const hit = cache && await cache.match(url); if (hit) { bytes = new Uint8Array(await hit.arrayBuffer()); fromCache++; } } catch {}
    if (!bytes) {
      bytes = await fetchBytes(url, (got) => onProgress(off + got, entry.size));
      try { if (cache) await cache.put(url, new Response(bytes)); } catch { /* quota exceeded: fine */ }
    }
    if (off + bytes.length > entry.size) throw new Error("weights are larger than the manifest says");
    out.set(bytes, off); off += bytes.length; onProgress(off, entry.size);
  }
  if (off !== entry.size) throw new Error(`weights incomplete: got ${off} of ${entry.size} bytes`);
  fetchParts.lastFromCache = fromCache;
  return out;
}

async function hasWebGPU() {
  try { return !!navigator.gpu && !!(await navigator.gpu.requestAdapter()); } catch { return false; }
}

async function loadModel() {
  $("loadBtn").disabled = true; $("runBtn").disabled = true;
  $("badges").innerHTML = ""; wcag = null;
  try {
    let want = $("backend").value;
    const gpu = await hasWebGPU();
    if (want === "auto") want = gpu ? "webgpu" : "wasm";
    if (want === "webgpu" && !gpu) {
      badge("WebGPU not available — using WASM");
      want = "wasm";
    }

    ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

    setStatus("Loading tokenizer, config and criteria list…"); setProgress(0);
    const [tj, tc, criteria, manifest] = await Promise.all(
      ["tokenizer.json", "tokenizer_config.json", "wcag_criteria.json", "manifest.json"].map((f) =>
        fetch(MODEL_DIR + f).then((r) => { if (!r.ok) throw new Error(f + ": HTTP " + r.status); return r.json(); })));
    const tokenizer = new Tokenizer(tj, tc);
    const v = manifest.variants.wcag;
    if (!v) throw new Error("manifest.json has no \"wcag\" variant");

    const t0 = performance.now();
    const progress = (name) => (got, total) => { setProgress(total ? got / total : null); setStatus(`Downloading ${name}: ${(got / 1048576).toFixed(0)}${total ? " / " + (total / 1048576).toFixed(0) : ""} MB`); };
    const graph = await fetchBytes(MODEL_DIR + v.onnx);
    const data = await fetchParts(v.data, progress(v.data.name));
    const dlMs = performance.now() - t0;

    setStatus("Creating inference session (this can take a while on first run)…"); setProgress(null);
    const t1 = performance.now();
    const session = await ort.InferenceSession.create(graph, {
      executionProviders: want === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
      graphOptimizationLevel: "all",
      // ORT-web mounts preloaded external data under the *location string inside the ONNX graph*
    // (v.data.name), not under the HTTP part filenames in v.data.parts.
    externalData: [{ path: v.data.name, data }],
    });
    const initMs = performance.now() - t1;
    wcag = new WcagLaya(ort, session, tokenizer, criteria);
    backendUsed = want;

    setStatus("Warming up…");
    const w0 = performance.now();
    await wcag.systemOne("<p>warm up</p>");
    const warmMs = performance.now() - w0;
    setStatus(`Ready. Download ${(dlMs / 1000).toFixed(1)} s${fetchParts.lastFromCache ? ` (${fetchParts.lastFromCache} of ${v.data.parts.length} parts from browser cache)` : ""}, session init ${(initMs / 1000).toFixed(1)} s, warm-up call ${warmMs.toFixed(0)} ms.`, "");
    badge(want === "webgpu" ? "WebGPU" : "WASM" + (ort.env.wasm.numThreads > 1 ? ` · ${ort.env.wasm.numThreads} threads` : " · 1 thread"), true);
    badge("WCAG 2.2 · 63 criteria");
    badge("cross-origin isolated: " + (self.crossOriginIsolated ? "yes" : "no"));
    $("runBtn").disabled = false;
  } catch (e) {
    console.error(e); setStatus("Could not load the model: " + (e?.message || e), "warn");
  } finally { setProgress(null); $("loadBtn").disabled = false; }
}

const fmtPct = (x) => (x * 100).toFixed(1) + "%";

function barRow(name, sub, p, top) {
  const d = document.createElement("div"); d.className = "bar" + (top ? " top" : "");
  d.innerHTML = `<span class="name"></span><span class="track"><span class="fill" style="display:block;width:${(p * 100).toFixed(1)}%"></span></span><span class="val">${fmtPct(p)}</span>`;
  const n = d.querySelector(".name");
  n.textContent = name;
  n.title = sub ? `${name} — ${sub}` : name;
  return d;
}

function render(res) {
  lastResult = res;
  $("resultCard").style.display = "block";
  $("sLatency").textContent = res.latency_ms.toFixed(0);
  $("sViolations").textContent = String(res.violations.length);
  $("sBackend").textContent = backendUsed === "webgpu" ? "WebGPU" : "WASM";

  // Violations table (criteria at/above the threshold), sorted by probability.
  const tbody = $("violRows");
  tbody.innerHTML = "";
  if (!res.violations.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="4" class="note">No criterion reached the threshold of ${fmtPct(Number($("thresh").value))}. Lower the threshold, or inspect the full ranking below.</td>`;
    tbody.appendChild(tr);
  }
  for (const v of res.violations) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><b></b> <span class="note"></span></td><td></td><td></td><td class="num"></td>`;
    tr.querySelector("b").textContent = v.criterion;
    tr.querySelector(".note").textContent = v.principle + " ·";
    tr.children[1].textContent = v.level;
    tr.children[2].textContent = v.description;
    tr.children[3].textContent = fmtPct(v.probability);
    tbody.appendChild(tr);
  }

  // Full ranking of all 63 criteria, top 12 by probability.
  const box = $("ranking");
  box.innerHTML = "";
  for (const c of res.criteria.slice(0, 12)) {
    box.appendChild(barRow(`${c.criterion} · ${c.description}`, `Level ${c.level} · ${c.principle}`, c.probability, c.violation));
  }

  $("raw").textContent = JSON.stringify(res, null, 2);
}

async function run() {
  if (!wcag) return;
  const text = $("state").value;
  if (!text.trim()) { setStatus("Paste some HTML first.", "warn"); return; }
  $("runBtn").disabled = true;
  try { render(await wcag.systemOne(text, { threshold: Number($("thresh").value) })); setStatus("Done."); }
  catch (e) { console.error(e); setStatus("Run failed: " + (e?.message || e), "warn"); }
  finally { $("runBtn").disabled = false; }
}

// wiring
for (const k of Object.keys(EXAMPLES)) $("example").add(new Option(k, k));
function applyExample() { $("state").value = EXAMPLES[$("example").value]; }
$("example").addEventListener("change", applyExample);
$("thresh").addEventListener("input", () => { $("threshVal").textContent = Number($("thresh").value).toFixed(2); if (lastResult) render(lastResult); });
$("loadBtn").addEventListener("click", loadModel);
$("runBtn").addEventListener("click", run);
document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run(); });
applyExample();
window.__wcag = { loadModel, run, get ready() { return !!wcag; }, get instance() { return wcag; } };
