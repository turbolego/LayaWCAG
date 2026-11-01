# LayaWCAG demo — scaling & contrast: research for the GitHub Pages demo

Review of the external comment, plus a concrete design for landing the three
suggestions in the browser demo that ships to https://turbolego.github.io/LayaWCAG/
via `.github/workflows/deploy.yml` (static `dist/` → GitHub Pages, no server).

## The comment, in three parts

> "For scanning large pages, since your build caps input at 256 tokens, sliding
> over DOM subtrees and rolling the scores up into a per-section heatmap could
> give you the prioritization view directly. The ModernBERT base can go much
> longer if you ever want to push MAX_LEN up. And for the contrast over-firing,
> adding a few computed-style hints (resolved foreground/background pairs) to
> the input might help more than adding more data."

All three are sound. Each maps to a different layer of the stack, and they are
independent — any one can ship without the others.

| # | Suggestion | Layer | Cost | Blocks-on |
|---|-----------|-------|------|-----------|
| 1 | Sliding-window DOM scan → per-section heatmap | inference / UI | low–med | none (client-only) |
| 2 | Push MAX_LEN beyond 256 | training + export | med | retrain + re-export |
| 3 | Computed-style contrast hints | input construction | med | none (client-only) |

---

## Ground truth (verified in this repo)

- **Model:** ModernBERT-large (421M params, **frozen**) + `Linear(1024, 63)`
  multi-label head, `sigmoid` → per-criterion probability. ONNX Runtime Web in
  the browser. 63 WCAG 2.2 criteria.
- **Context:** trained at **MAX_LEN = 256** (`kaggle_training/…ipynb`,
  `collate_fn`). The browser defaults `maxLen = 512` in
  `web/laya-wcag-core.js` (`buildWcagSequence`, `systemOne`, `systemBatch`).
- **ModernBERT max sequence = 8192** (rotary positional embeddings; HuggingFace
  docs). So 256/512 are *self-imposed caps*, not architectural limits.
- **The ONNX graph is length-dynamic.** The export passes
  `dynamic_axes = {input_ids:{1:"sequence_length"}, ...}` — the deployed graph
  accepts a variable sequence dim. So *inference-time* MAX_LEN can be raised
  **without re-exporting**. What is untrained is the model's *quality* past 256,
  not the graph's ability.
- **Deployment:** `web/` → `dist/` (`scripts/prepare_site.mjs`) → GitHub Pages
  (`deploy.yml`, `actions/deploy-pages@v4`). Static hosting. `web/coi.js`
  registers a service worker to add the COOP/COEP headers Pages can't send, so
  ONNX-WASM gets `SharedArrayBuffer`. Model is ~378 MB, streamed as ≤24 MiB
  parts and cached in Cache Storage.
- **Contrast today:** the model only sees **markup text** — no CSS cascade, no
  rendered layout. 1.4.3 / 1.4.11 therefore over-fire (see
  `DEFAULT_PER_CRITERION_THRESHOLDS` in `web/laya-wcag-core.js`, already bumped
  to 0.7 as a band-aid).

---

## 1. Sliding-window section scan → per-section heatmap

**Goal:** large pages exceed the token cap, so instead of one truncated call,
walk the document's top-level sections, score each, and roll the results up
into a section heatmap (the "prioritization view" the reviewer wants).

### Design

Everything is client-side, so it fits GitHub Pages with zero infra:

1. **Parse & segment.** For pasted HTML, build a DOM with
   `new DOMParser().parseFromString(html, "text/html")`. Split on a
   well-defined section boundary — `header | nav | main > section | section |
   aside | footer`, or fall back to `<h1>`–`<h2>` blocks. For a **live page**
   (new "scan this page" mode) walk `document` directly.
2. **Serialize each section to the model's input form** (`outerHTML`, trimmed).
   Keep each window ≤ the model's context (start at 256 tokens to stay in the
   trained range; the heatmap does not need long context — it *wants* small,
   local windows).
3. **Batch-score** the windows with `wcag.systemBatch([...])` (already
   supported — `systemBatch` pads to the longest seq and returns per-snippet
   results). One ONNX call for many sections.
4. **Roll up.** For each section, keep the 63-probability vector. Two useful
   aggregates:
   - **Section heat** = weighted max over the criteria at the user's threshold
     (drives the heatmap color): `heat = max_i p_i` (or `Σ p_i·w_level` to
     weight AAA > AA > A).
   - **Prioritization** = rank sections by `heat` descending → "fix this
     section first."
5. **Render the heatmap.** A vertical strip (one row per section) colored by
   heat, with the section's top-3 criteria as a tooltip/legend. No new
   dependencies — plain DOM/CSS (the demo already builds bars in
   `barRow()` in `web/app.js`).

### Why it's the cheapest win
- No retraining, no re-export, no new model bytes. Pure UI + a loop over
  `systemBatch`.
- It *directly* produces the prioritization view the reviewer describes, and it
  makes the "capped at 256" limit a feature (small local windows) rather than a
  bug (one truncated page).
- Works today on the "paste HTML" flow; adds nothing to the 378 MB download.

### Sketch (client, no new deps)
```js
// web/app.js  (new)
async function scanSections(html, threshold) {
  const doc = new DOMParser().parseFromString(html, "text/html");
  const sections = [...doc.body.querySelectorAll(
    "header,nav,main>section,section,aside,footer")];
  const chunks = sections.map(el => el.outerHTML);
  const results = await wcag.systemBatch(chunks, { threshold });
  return results.map((r, i) => ({
    el: sections[i],
    heat: Math.max(...r.criteria.map(c => c.probability)), // or level-weighted sum
    top: r.violations.slice(0, 3),
  })).sort((a, b) => b.heat - a.heat);
}
```

---

## 2. Pushing MAX_LEN

**What's true:** the graph is length-dynamic and ModernBERT tops out at 8192,
so *technically* we can feed 256 < L ≤ 8192 today. **What's unproven:** model
*quality* beyond the 256 it was trained on — the classifier head never saw
long-context CLS vectors.

**To actually use long context well, you must retrain:**
- Retrain/continue the WCAG fine-tune with `MAX_LEN` raised (e.g. 1024 or
  2048) on real long pages, then re-export (the dynamic-axes export already
  handles it), re-quantize, re-split to ≤24 MiB parts, re-upload to HF.
- Cost: a longer-context run is O(L²) attention but the encoder is **frozen**,
  so only the tiny `Linear(1024, 63)` head trains — retraining is cheap. The
  real cost is the data (long, multi-section pages labeled across all 63
  criteria) and the ONNX re-export/quantize/split cycle in `deploy.yml`.

**Recommendation:** treat #1 (sliding window) as the primary large-page path —
it deliberately *avoids* needing long context. Raising MAX_LEN is only worth it
if you want single-call whole-page scoring *and* you can produce long-context
training data. Sequence it after #1 ships, and validate with
`tests/browser_smoke_wcag.py` that a 1024-token window beats the truncated-512
baseline on a held-out set before committing to it.

---

## 3. Computed-style contrast hints (the "helps more than more data" idea)

**Why:** 1.4.3 over-fires because the model sees only markup. It cannot know
that `#111 on #fff` is 15.9:1 (fine) vs `#aaa on #fff` is 2.3:1 (fail). More
training data teaches the *distribution*, but the deterministic signal is right
in the browser: resolve the actual fg/bg and hand the model the ratio.

**Design (client-side, GitHub-Pages-safe):**
1. **Get a live DOM + computed styles.** Pasted HTML can't be styled in place
   (it's not in the document). Render it into a **sandboxed same-origin
   iframe** (`srcdoc`, `sandbox="allow-scripts"`), wait for layout, then
   `getComputedStyle` the text elements *inside that iframe* (same-origin →
   allowed; the current page or an about:blank srcdoc iframe is both
   reachable). For a "scan this page" mode, just use `document` directly.
2. **For each text-bearing element**, resolve:
   - `color` (fg) and the *effective* bg — walk up ancestors, alpha-composite
     the first non-transparent `background-color` (or the page bg).
   - font size / weight → "large text" test (≥ 24px, or ≥ 18.66px bold).
3. **Compute the WCAG 2.x contrast ratio** (relative-luminance formula).
   This is a pure, well-specified calculation — no model, no data.
4. **Inject as a hint token** into the section's model input, e.g. prepend
   `[contrast] <h1> fg=#111 bg=#fff 15.9:1 OK` / `[contrast] <p class=x>
   fg=#999 bg=#fff 2.9:1 FAIL-A` lines. The model then grounds its 1.4.3 /
   1.4.11 / 1.4.6 prediction in the real ratio instead of guessing.
5. **Down-weight the learned 1.4.3 logit** and **up-weight the deterministic
   ratio** — you can even short-circuit: if ratio < 4.5 (normal) or < 3.0
   (large) → force 1.4.3 to a violation; if passes → suppress it. This is
   effectively a **hybrid model+rule** detector, which is the honest
   architecture for contrast and matches what axe-core does (rule-based).

**GitHub Pages notes:** sandboxed `srcdoc` iframes are same-origin and can be
read from the parent (no COOP needed for the contrast math itself — only the
ONNX-WASM multi-thread path needs the COI shim, which `coi.js` already
handles). No server, no CORS, no new download. The contrast helper is pure JS
and unit-testable in Node (no browser, no model).

**Effort:** the ratio math + hint formatter is small and testable now; the
iframe-render plumbing is medium. This is the piece most likely to *actually*
fix the reported contrast over-fire, and it's the one that matches the
reviewer's "might help more than adding more data."

---

## Suggested ship order (all fit GitHub Pages, no infra)

1. **Contrast helper (pure JS) + hint injection** — highest value per line of
   code, directly targets the over-fire, and is unit-testable headless first.
   **DONE:** `web/contrast.mjs` (WCAG 2.x ratio, large-text, alpha compositing,
   hint formatter, `injectContrastHints`) + `tests/contrast.test.mjs`
   (27 headless tests, all passing). Shipped to Pages automatically —
   `prepare_site.mjs` copies `web/` wholesale into `dist/`. **REMAINING (needs
   a browser):** collect resolved fg/bg via `getComputedStyle` (live page or
   sandboxed `srcdoc` iframe for pasted HTML), build the `items` array, call
   `injectContrastHints(html, items)` in `web/app.js` before `systemOne`.
2. **Section scan + heatmap** — UI-only, no model change, delivers the
   prioritization view. Add `scanSections()` + a heatmap card to
   `web/index.html`/`web/app.js`. (Sketch in §1.)
3. **Push MAX_LEN** — only after 1+2, and only with retrained long-context
   data; validate against a held-out long-page set first.

## Open questions for @kveita
- Do we add a **"scan this page"** mode (needs the iframe/`getComputedStyle`
  path) or keep the demo paste-only (contrast hints then computed from a
  rendered `srcdoc` iframe)?
- Heatmap aggregation: plain `max(p)` vs level-weighted `Σ p·w`?
- Is a retrain on 1024/2048-context pages in scope for the Kaggle pipeline, or
  do we stay at 256 and rely on sliding windows?
