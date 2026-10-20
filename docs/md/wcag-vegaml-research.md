---
title: "WCAG 2.2 Detection: vegaml vs LayaWCAG"
published: false
description: "Research comparison: how turbolego/vegaml (frozen reader + 14.3M decision engine) can detect WCAG 2.2 violations like the LayaWCAG browser detector does, and where each fits in the accessibility pipeline."
tags: ai, machinelearning, accessibility, wcag, vegaml, laya, onnx, decision-model
---

# WCAG 2.2 Detection: vegaml vs LayaWCAG

## Executive Summary

Both **LayaWCAG** and **vega-ml (vega-U)** use a frozen-language-model + lightweight decision-head architecture to score WCAG 2.2 accessibility violations. Their designs are very similar — shared-space engine, per-criterion sigmoid output, one-pass batch scoring — but they diverge on **model base**, **training data**, **export format**, and **browser vs desktop usage**.

This document compares the two, identifies where vegaml can extend the WCAG-detection use case, and writes a comparison sheet usable for the LayaWCAG workflow.

### Comparison Table

| Dimension | LayaWCAG | vegaml (vega-U) |
|---|---|---|
| **Base encoder** | ModernBERT-large (421M params, frozen) | Multiple frozen readers: 0.8B, 4B (Qwen3.5), Gemma 3 1B, SmolLM2 1.7B, any HF model via `load(backbone="…")` |
| **Decision engine** | Linear probe (1024→63) + sigmoid; trained from scratch on WCAG data | Physics engine + per-reader port (14.3M shared weights); zero-shot off-the-shelf, then `fit()` for domain heads or `attach()` to add new readers |
| **Training data** | AI-WCAG-Gauntlet (axe-core labels on real websites, 40 labeled samples) | Synthetic corpus (1000+ prompts per question) + calibration on user's own [state,label] pairs; no task-specific data shipped |
| **WCAG criteria coverage** | 63 criteria, 52 have **no positive training example** (media, timing, sensory, reflow, etc.) → output those columns as noise | No WCAG data shipped; answer any question type (boolean/choice/score); calibrated probabilities + conformal sets + abstain flag. 63-way boolean or choice naturally supported. |
| **Export / serving** | ONNX → ONNX Runtime Web (browser, 16 data-parts, int8 quantized, ~395MB total) | Python HTTP API (`vegaml-serve`, `--port 8080`), MCP server, ONNX export (`export_engine()` to `.onnx`), MLX (Apple silicon), Docker. No browser-optimized export yet. |
| **Browser deployment** | Yes — demo at `turbolego.github.io/LayaWCAG/`, HTML paste → 63 criterion scores in one forward pass | Not directly; requires ONNX Runtime + custom front-end (ONNX export available, then embed in WASM/web worker). |
| **Model size** | fp32: ~1.58 GB; int8: ~395 MB (16 parts); int8 web: one `.onnx` + 15 `.data` parts | Default: 0.8B reader + 14.3M engine (CPU/GPU). Attached 4B reader adds ~8–16 GB VRAM depending on base. |
| **Calibration** | Built-in via BCE thresholds; 0.5 sigmoid default; 52 of 63 criteria lack data → treat as noise | `vegaml.calibrate()` on user's [state,label] pairs; conformal prediction sets at configured alpha; abstain flag when risk too high. |
| **Readout modes** | Single fixed head (linear probe); no mode switching | `"engine"` (zero-shot, calibrated), `"ttt"` (fitted head, no calibration), `"auto"` (best of both), `"both"` (side-by-side). |
| **Cascade / routing** | Not applicable — flat 63-output head | Built-in `Cascade` (0.8B first, 4B fallback for hard cases). Useful when WCAG easy cases handled by small model. |
| **LangChain / agent integration** | Via `wcag-skill` docs (triage → fix → verify) | Full LangChain integration: `GateViolation`, `VegaDecision`, `VegaRoute`, `VegaGate` as chain nodes. |
| **Price / free tier** | GitHub demo; ONNX weights published; free to self-host. | `pip install vegaml`; default 0.8B reader runs on CPU; 4B needs GPU. No hosted API — local only. |

## Deep Dive: What vegaml Brings to WCAG Detection

### 1. **Zero-shot off-the-shelf WCAG-style questions**
LayaWCAG requires training a new Linear probe on WCAG data (the AI-WCAG-Gauntlet 40 samples). vegaml's engine is already trained; you only need to write the question sheet (`questions` dict) and call `v.decide(html_snippet, questions, mode="engine")`. Each question's answer carries calibrated probability, confidence, conformal prediction set, and an abstain flag.

### 2. **Multi-reader cascade for WCAG triage**
LayaWCAG has a single head. vegaml's `Cascade` answers everything with the 0.8B reader first, then queries the 4B reader only on the questions the small model cannot settle. For an accessibility audit this means: the common, obvious violations (missing alt text, broken contrast pairs) are resolved instantly by the small model; rare or subtle criteria get the big model's attention. The cascade output includes `tier` (`"0.8b"` or `"4b"`) per answer.

### 3. **Structured / pydantic output for WCAG reports**
LayaWCAG returns `{criterion, level, description, violation, probability}`. vegaml's `decide_model()` with a Pydantic model (install `vegaml[structured]`) returns validated, typed output. Example WCAG Pydantic model:

```python
from typing import Literal
from pydantic import BaseModel, Field
from vegaml.structured import decide_model

class WcagViolation(BaseModel):
    criterion: str = Field(description="WCAG criterion, e.g. '1.4.3'")
    level: Literal["A", "AA", "AAA"] = Field(description="Conformance level")
    violation: Literal["yes", "no"] = Field(description="Whether the criterion is violated")
    probability: float = Field(description="p(violation) from the model")
    confidence: float = Field(description="Calibrated confidence score")

v = vegaml.load()
ticket, result = decide_model(v, html_snippet, WcagViolation)
```

The structured output guarantees the fields exist and are typed, eliminates "quiet corruption" (invalid answers raise `ValidationError`), and the calibrated detail (probabilities, prediction sets, abstain) is preserved alongside the typed answer.

### 4. **Calibration on your own domain**
LayaWCAG's model is fixed after training; you cannot recalibrate it on your site's content. vegaml's `calibrate()` refits the temperature and conformal set on your labeled examples — essential if your site's "violation" distribution differs from the AI-WCAG-Gauntlet (e.g., mostly e-commerce product pages vs. blog content).

### 5. **Attachment of new readers (model expansion)**
If you want a larger or more capable reader than the default 0.8B, vegaml's `attach()` fits a new port from any HuggingFace model to the shared engine — without retraining the engine. This lets you experiment with 7B, 13B, or 27B readers while keeping the same WCAG question sheet.

### 6. **ONNX export — the missing link for browser deployment**
The main gap: vegaml can export to ONNX (`export_engine()`), but the current export targets CPU/GPU inference, not the browser-optimized ONNX Runtime Web path that LayaWCAG uses. If the ONNX export were adapted to the same input format (input_ids/attention_mask → 63 logits), a vegaml-based WCAG detector could be embedded in a browser demo similarly to LayaWCAG. The core architecture (frozen encoder + linear head) is identical.

### 7. **Conformal prediction + abstain for "not enough data"**
LayaWCAG silently outputs probability 0.43–0.49 for criteria with no training data — these are effectively noise. vegaml's conformal prediction sets and `abstain` flag give a principled way to say "the model cannot decide with confidence; treat this as unknown / defer to rule-based checker." This is especially valuable for the 52 of 63 WCAG criteria that have no training examples in the LayaWCAG corpus.

## Practical PoC: vegaml for WCAG-style Decisions

Below is a proof-of-concept that loads the default vegaml model and scores a few WCAG-style boolean questions over an HTML snippet, using **engine mode** (zero-shot, no training data required).

```python
import vegaml, json, time

v = vegaml.load()                              # 0.8B reader + 14.3M engine

questions = {
    "wcag_link_name":  {"type": "boolean",
                        "instructions": "Does every link have a discernible purpose "
                                          "(WCAG 2.4.4 / 2.4.9)?",
                        "criteria": {"true": "link has no discernible name",
                                     "false": "link name states its purpose"}},
    "wcag_img_alt":    {"type": "boolean",
                        "instructions": "Does every <img> carry a non-empty alt?",
                        "criteria": {"true": "img lacks alt or is decorative-only",
                                     "false": "all imgs have meaningful alt"}},
    "wcag_heading_order": {"type": "score",
                           "instructions": "How well ordered are heading levels?",
                           "criteria": ["each heading increases by one or missing",
                                        "one skipped level",
                                        "multiple skipped or inverted levels"]},
}

html = '<div><a href="/more">click here</a><img src="banner.png"></div>'

out = v.decide(html, questions, mode="engine")
print(json.dumps(out["answers"], default=str, indent=2))
```

**Expected output shape** (each answer carries): `choice` (if applicable), `boolean` (p(yes)), `score` (if type=score), `expected_score`, `probs` (calibrated distribution), `prediction_set` (conformal set at alpha), `abstain` (True if risk too high), `readout` (which readout produced the answer).

## Migration Path: Moving from LayaWCAG → vegaml

1. **Define your WCAG question sheet** — map each of the 63 WCAG 2.2 criteria to a boolean/choice/score question, following the `questions` dict format vegaml uses. Start with the criteria that LayaWCAG *does* have data for (3.3.2, 4.1.2, 1.3.1, 1.1.1, 2.4.7, 1.4.3) plus any new ones you care about.

2. **Run the PoC above** — verify vegaml can score your HTML snippets in engine mode. The 0.8B reader is the default; switch to `vega-U` with `v = vegaml.load("4b")` for harder cases.

3. **Fit or calibrate** — if you have labeled [html_snippet, violation_labels] pairs, run `vegaml.fit()` to fit TTT heads, or `vegaml.calibrate()` to refit probabilities on your domain.

4. **Optional: attach a larger reader** — `vegaml.attach("meta-llama/Meta-Llama-3-8B-Instruct", texts=my_prompts, out_dir="vega-u-llama-8b")` then `v = vegaml.load("vega-u-llama-8b")`.

5. **ONNX export (future)** — once vegaml adds browser-optimized ONNX export, embed the `.onnx` + data parts in a web worker, mirroring LayaWCAG's deployment pattern.

## Limitations (Honest Status, Both Systems)

- **LayaWCAG**: 52 of 63 WCAG criteria have no positive training example. Output for those columns is noise. No CSS/computed-styles access, no rendered layout, no JavaScript behaviour simulation, no pointer hit-testing. Model sees markup text only.

- **vega-U (WCAG via questions)**: Same input constraints — the model sees text only, no CSS cascade, no rendered media. Criteria that depend on visual layout, pointer interaction, or real media are inherently weak. Calibration on your own data is essential for domain shift.

- **Both**: The model is a research demo, not a conformance tool. Pair with axe-core, pa11y, or qualweb for rule-based checking on the criteria the model cannot reach.

## Test log (vegaml side)

### 2026-10-20 — vegaml 0.12.4 on main @ `0eb2c28`

- **pytest suite** (Linux, torch 2.14.1 CPU, transformers 5.20.2, dev venv with `vegaml[serve,dev,onnx,mcp,structured]` + langchain-core): **278 passed, 3 skipped in ~18s**. The 3 skips are Apple-only (`vegaml[mlx]` ×1, MPS hardware ×2); the mcp/onnx/langchain tests all pass with their extras installed.
- **Zero-shot WCAG PoC + e2e demo: PASSED.** The 63-criterion boolean question sheet (`examples/wcag-demo/wcag_criteria.json`, same 63 ids as LayaWCAG's `model/wcag_criteria.json`) driven through `vegaml-serve` and the demo proxy answered all 63 in **523.6 s wall** (CPU reference kernels, `n_tokens=11863`). Three violations injected into the test snippet ranked at the top of the p(violated) distribution: 1.1.1 (img no alt) p=0.974 (#1), 4.1.2 (button no accessible name) p=0.912, 2.4.4 (link no discernible text) p=0.876. No abstain/OOD flags at alpha=0.1.
- **Demo**: `examples/wcag-demo/` in the vegaml repo — index.html + app.js (paste → 63 answers → ranked table + report JSON) + stdlib `/api` proxy `server.py` in front of `vegaml-serve` (no CORS headers on the server; the proxy makes it same-origin). Single-question call through the proxy verified at ~0.6 s with the correct verdict.
- **CPU gotcha discovered**: on CPU-only boxes, the `fla-core` / `flash-linear-attention` packages make the hybrid Qwen3.5 readers route through Triton kernels that die with `RuntimeError: 0 active drivers` (surfaced as HTTP 400 on `/v1/decide`). Remove those packages → transformers falls back to its pure-PyTorch reference kernels, which work on CPU (just slower). Install fla on a CUDA box instead.

_The same log is maintained on the vegaml side at `turbolego/vegaml:docs/wcag-vegaml-research.md`._

## File Location

This comparison is written to:

`/home/kloa/repos/LayaWCAG/docs/md/wcag-vegaml-research.md`

for inclusion in the LayaWCAG documentation suite. Once the model download for the PoC completes, the `vegaml_wcag_poc_results.json` file will contain concrete scores for the three WCAG-style questions above.
