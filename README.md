# LayaWCAG

**A WCAG 2.2 accessibility violation detector that runs entirely in your browser.** Paste an HTML snippet, get a violation probability for each of the 63 WCAG 2.2 success criteria, in a single forward pass, on the visitor's device. No server, no API key, no data leaves the page.

**Live demo:** https://turbolego.github.io/LayaWCAG/

Built on the [Laya](https://huggingface.co/convaiinnovations/laya) ModernBERT-large encoder (Apache-2.0), fine-tuned for WCAG 2.2 detection, converted to ONNX, quantized to int8, and split into browser-sized parts for ONNX Runtime Web.

> **Honest status.** This is a research demo, not a conformance tool. The classifier was fine-tuned on a small, automatically-labeled corpus and works well on the criteria it has data for, but most of the 63 criteria have no training examples at all. Treat its output as a hint and pair it with a rule-based checker like [axe-core](https://github.com/dequelabs/axe-core). See [Results](#results) and [Limitations](#limitations) for the real numbers.

## What it does

You paste HTML. The model returns, for all 63 WCAG 2.2 success criteria at once:

- a **violation probability** (sigmoid over a per-criterion logit)
- the criteria sorted by probability, with those above a threshold flagged as violations

Unlike a chat model it does not generate text. It reads the markup once and scores every criterion in one pass, which makes it deterministic and small enough to ship to a browser.

The model sees **markup text only** — no CSS cascade beyond inline styles, no rendered layout, no assistive-technology simulation. Checks that need computed styles, real media, JavaScript behaviour or pointer hit-testing are inherently out of reach and are left untrained (see [Limitations](#limitations)).

## The model

| | |
|---|---|
| Base encoder | ModernBERT-large (421M parameters), **frozen** during fine-tuning |
| Head | `Linear(1024, 63)` multi-label classifier on the `[CLS]` pooled token |
| Output | `criterion_scores` → `[batch, 63]` logits; sigmoid applied in the browser |
| Loss | `BCEWithLogitsLoss` with per-criterion `pos_weight` (clamped to 50) against class imbalance |
| Input | `input_ids` + `attention_mask`, up to 256 tokens |
| Runtime | ONNX Runtime Web (WASM on CPU, WebGPU experimental) |
| Size | ~395 MB int8, split into 16 × 24 MiB parts |

The ONNX graph takes only `input_ids` and `attention_mask` (no marker/qtype inputs) and returns the 63 criterion logits. The browser runtime (`web/laya-wcag-core.js`) encodes `[CLS] text [SEP]`, runs the session, applies the sigmoid, and returns per-criterion probabilities.

## Training

### Base and head

The 421M-parameter Laya encoder is frozen; only the small classifier head is trained. This fits on a single Kaggle T4 and avoids optimizer-state OOM. The head is trained with `BCEWithLogitsLoss` and a `pos_weight` vector (negative-count / positive-count per criterion, clamped to 50) so the extreme class imbalance — most criteria have few or zero positives — cannot collapse the head to predicting all-zero.

### Data pipeline

The labels come from the **IBM Equal Access Accessibility Checker** (`accessibility-checker-engine`), the same tool the [A11YBench](https://huggingface.co/datasets/LLM4APR/A11YBench) benchmark uses, mapped from its rule IDs to WCAG 2.2 success-criterion numbers.

`axe-labeling/crawl_label.js` drives real Chromium (Playwright) over ~50 production websites — the A11YBench projects' own live deployments plus other high-traffic sites — 6 pages each, and runs the checker on each **live page document**. It emits two kinds of rows:

- **page-level**: the full page HTML with the set of criteria it violates
- **element-level**: each violating element on its own, so one page carrying 30 violations yields ~30 training examples

That element-level expansion is where the volume comes from: ~230 crawled pages became **7,162 labeled rows**. Running the checker in a real browser (rather than jsdom) is what makes layout-dependent rules — contrast (1.4.3), focus-visible (2.4.7), viewport-zoom (1.4.4) — actually evaluate.

`build_final_dataset.js` merges the crawl with the hand-written structural pairs, de-duplicates by content, and stratifies the train/val split so every criterion with positives appears in both sets.

Two labeling pitfalls the pipeline handles, worth knowing if you regenerate the data:

- The IBM engine emits **pass rows shaped like failures** (`value: ["VIOLATION","PASS"]`, `reasonId: Pass_0`). Filter on `reasonId` (`/^(fail|warning|potential)/i`), never on `value[0]`, or every page is labeled.
- Under jsdom there is no layout engine, so `element_tabbable_visible` fires `potential_visible` on every focusable element (noise) and contrast/target-size/focus rules never fire at all — which is why the real-browser crawler was necessary.

### Final corpus

| | |
|---|---|
| Train rows | 5,647 (3,740 with ≥1 violation) |
| Val rows | 1,515 (1,039 with ≥1 violation) |
| Criteria with ≥1 positive | **12 of 63** |

Per-criterion positive counts in train: 2.4.7 Focus Visible 1,230 · 1.4.3 Contrast 927 · 1.3.1 Info & Relationships 1,014 · 1.1.1 137 · 3.3.2 Labels 104 · 4.1.1 Parsing 233 · 2.4.4 Link Purpose 149 · 4.1.2 Name/Role/Value 81 · 2.1.1 Keyboard 37 · 4.1.1/3.1.1 and a few others in the low tens.

### Run configuration

Trained on a Kaggle T4 (kernel `hummern/laya-wcag-2-2-retrain-v6`): batch size 16, learning rate 2e-4, `MAX_LEN` 256, up to 15 epochs with early stopping on validation macro-F1 (patience 5). Best validation macro-F1 **0.0826** at epoch 13. `scripts/retrain_notebook.py` regenerates the Kaggle notebook from these settings.

## Results

Measured on the held-out validation set (1,515 rows). "Gap" is mean violation probability on violating rows minus mean on clean rows — a positive gap means the model ranks real violations above clean markup. Per-criterion F1 is at a 0.5 threshold.

| Criterion | Description | Gap | F1 @0.5 |
|---|---|---|---|
| 3.3.2 | Labels or Instructions | **+0.561** | 0.639 |
| 4.1.2 | Name, Role, Value | **+0.321** | 0.368 |
| 1.3.1 | Info and Relationships | **+0.281** | 0.594 |
| 4.1.1 | Parsing | +0.277 | 0.294 |
| 2.4.7 | Focus Visible | +0.216 | 0.421 |
| 1.1.1 | Non-text Content | +0.197 | 0.323 |
| 2.4.4 | Link Purpose (In Context) | +0.160 | 0.190 |
| 1.4.3 | Contrast (Minimum) | +0.122 | 0.446 |

- **Micro-F1 @0.5: 0.432**
- **Macro-F1 @0.5: 0.060** — low only because 52 of 63 criteria have no positive example and contribute zero; the criteria that have data discriminate clearly.

### Before and after

The previous model (trained on ~110 hand-written/jsdom rows) was effectively untrained: probabilities clustered near 0.5 for every input and the clean example flagged 19 of 63 criteria. The retrained model:

| Example | Criteria flagged (of 63) |
|---|---|
| Clean semantic page | 19 → **3** |
| Low-contrast text → 1.4.3 score | dead → **0.90** |
| Image without alt → 1.1.1 score | ~0.34 → **0.73** |

This was the first build where any criterion clearly worked. Micro-F1 went from ~0 to 0.43.

## Packaging

The int8 build is ~395 MB. `onnx.save_model` externalizes the weights with `size_threshold=1024` (attribute-sized tensors stay inline — at threshold 0 tiny `Slice` constants move external and ORT refuses them), the data file is split into 16 × 24 MiB parts, and `model/manifest.json` records the parts with a SHA-256 of the whole data file.

One detail that bites every deployment: ORT-web mounts preloaded external data under the **location string inside the ONNX graph** (`laya-wcag.onnx.data`), so `manifest.json`'s `data.name` must be that string — not the first part's filename. Get it wrong and the session fails with "File not found in preloaded files."

The browser caches the parts in Cache Storage keyed by the file hash, so repeat visits skip the download.

## Limitations

Read these before trusting any output:

- **52 of 63 criteria have no training data.** Media (1.2.x captions/audio-description), timing (2.2.1), sensory characteristics (1.3.3), reflow (1.4.10), target size (2.5.8) and others need real media, JavaScript, or pointer hit-testing the crawler does not exercise. Their columns are untrained and their probabilities are meaningless.
- **Contrast (1.4.3) over-fires.** It reads inline color styles but does not fully separate passing from failing pairs — good-contrast text also scores ~0.90. It needs more contrast *variety* in the corpus, not more volume of the same pairs.
- **Macro-F1 is low** (0.060) purely because of the empty columns; do not read it as "the working criteria are bad."
- The model is a **research demo, not a WCAG conformance tool.** Always pair it with axe-core or the IBM checker for real audits.

## Try it locally

```
npm ci
node scripts/prepare_site.mjs --out dist-test
SMOKE_DIST=dist-test python tests/browser_smoke_wcag.py
python serve.py    # http://localhost:8000
```

Open the page, pick **WASM**, press **Load model** (downloads ~395 MB once, then cached), and run an example. `tests/browser_smoke_wcag.py` drives the real page in headless Chromium and asserts the model loads and the violations table renders.

### Retrain on Kaggle

The model is retrained by regenerating the notebook and pushing it with the labeled data attached:

```
python scripts/retrain_notebook.py                 # rewrites notebooks/laya_wcag_training.ipynb
# upload the data (axe-labeling/train_final.jsonl, val_final.jsonl) as a Kaggle dataset,
# then push the notebook with that dataset in dataset_sources (GPU, T4).
# scripts/retrain_notebook.py's data cell locates the mount robustly and trains,
# exports, quantizes, splits, verifies, and writes the site-ready manifest.
```

## Deploy

Pages is configured as the legacy Jekyll branch build (source `main`, path `/`), so the repo tree is served directly and `dist/` is not published. The assembled site — including the vendored ONNX Runtime Web + tokenizers (`vendor/`) and a `.nojekyll` — is committed to the branch. Push to `main` and Pages rebuilds.

## The general decision model (still in the repo)

Before the WCAG fine-tune, this repo ported Laya's general **typed-decision** model to the browser: given text and typed questions (`choice` / `score` / `yes-no`), it returns calibrated probabilities and confidence in one pass. That code — `web/laya-core.js` (`Laya` class), `workflows.html`, `scripts/build_model.py`, `scripts/verify_model.py` — is unchanged and still builds the general checkpoint. The live demo now runs the WCAG model, but the decision-model runtime and its build scripts remain available for the general Laya checkpoint (`convaiinnovations/laya`).

## Repository layout

| Path | What it is |
|---|---|
| `index.html`, `app.js` | the deployed WCAG demo (synced from `web/`) |
| `web/` | source for the demo: `app.js`, `laya-wcag-core.js`, `index.html`, service worker |
| `web/laya-wcag-core.js` | `WcagLaya` runtime: encode → run → sigmoid over 63 criteria |
| `web/laya-core.js` | the general decision-model runtime (`Laya` class) |
| `model/` | the deployed WCAG build: graph, 16 data parts, `manifest.json`, tokenizer/config, `wcag_criteria.json` |
| `scripts/retrain_notebook.py` | regenerate the Kaggle training notebook |
| `scripts/build_wcag_model.py`, `train_wcag.py` | WCAG conversion / training helpers |
| `scripts/build_model.py`, `verify_model.py` | general-checkpoint ONNX build + PyTorch fidelity check |
| `scripts/prepare_site.mjs` | assemble `dist/` |
| `tests/browser_smoke_wcag.py` | headless-Chromium smoke test for the WCAG demo |
| `tests/data/` | 108 labeled cases for the general decision model |
| `axe-labeling/` (sibling dir) | the IBM-checker labeling + crawl pipeline that produced the training data |
| `serve.py` | local static server |
| `LICENSE`, `NOTICE.md`, `licenses/` | license text, notices, third-party licenses |

## License

The code in this repository (build scripts, demo page, JavaScript inference code) is licensed under the Apache License, Version 2.0; see `LICENSE`.

The model is a modified derivative of [Laya](https://huggingface.co/convaiinnovations/laya) by ConvAI Innovations (Apache-2.0), built on ModernBERT-large by Answer.AI and LightOn (Apache-2.0). The demo page ships ONNX Runtime Web (MIT) and Tokenizers.js (Apache-2.0). The training labels are produced by the IBM Equal Access Accessibility Checker (Apache-2.0). Copyright notices, the list of changes made to Laya, and the third-party license texts are in `NOTICE.md` and the `licenses/` folder.

This is an unofficial port and is not affiliated with ConvAI Innovations or IBM.
