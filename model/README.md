---
license: mit
tags:
  - accessibility
  - wcag
  - wcag2.2
  - a11y
  - onnx
  - browser
  - modernbert
library_name: onnxruntime
pipeline_tag: text-classification
---

# Laya WCAG 2.2 — violation classifier (ONNX, browser-ready)

A WCAG 2.2 accessibility-violation classifier for HTML markup, exported to ONNX and packaged
for in-browser inference with [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web).
It powers <https://turbolego.github.io/layaForWeb/>.

**The model only sees the HTML text — no CSS, no rendered layout.** Its output is a hint,
not a verdict; run a real checker (axe-core, IBM Equal Access) for authoritative results.

## Architecture

| | |
|---|---|
| Encoder | `answerdotai/ModernBERT-base` (421M params), **frozen** |
| Head | `Linear(1024, 63)` + a small MLP on the 1.4.3 column only |
| Input | `[CLS] + html tokens[:254] + [SEP]`, `input_ids` + `attention_mask` |
| Output | `criterion_scores` — 63 logits (BCEWithLogitsLoss ⇒ apply sigmoid) |
| Precision | int8 encoder + fp32 MLP (~0.5 MB), 395 MB total |
| Max length | 256 tokens |

Each output column is one WCAG 2.2 criterion, in the order given by `wcag_criteria.json`
(column 14 = **1.4.3 Contrast (Minimum)**).

## Training

Labels came from the IBM Equal Access checker run in real Chromium over crawled production
pages (`axe-labeling/` in the repo) — real layout is what makes contrast (1.4.3),
focus-visible (2.4.7) and viewport-zoom (1.4.4) judgeable at all. The encoder is frozen;
only the classifier head is trained.

**v6** (joint head over all 63 criteria) was the first build where any criterion clearly
worked — pos/neg probability gaps on held-out data: +0.56 (3.3.2), +0.32 (4.1.2),
+0.28 (1.3.1), +0.20 (1.1.1), +0.22 (2.4.7).

**Contrast fine-tune** addressed 1.4.3, which v6 got backwards: good-contrast text scored
*higher* than low-contrast text, so no threshold could separate them. Because a full retrain
traded the working criteria against each other, only that one column was retrained:

| 1.4.3 Contrast | before | after |
|---|---|---|
| AUC (1,615 held-out rows) | 0.629 | **0.933** |
| Best F1 | 0.450 | **0.779** |

Method: precompute the frozen encoder's `[CLS]` embeddings once, train a `Linear(1024,128) →
GELU → Dropout → Linear(128,1)` head initialized from the deployed linear row, then bake the
MLP into the graph as an additive correction masked to column 14. The other 62 criteria are
provably untouched — verified that only column 14 moves in the ONNX.

## Results

- The site's "Clean semantic snippet" flags **0 of 63** criteria.
- fp32 (node/ORT) gives 0 flags; onnxruntime-web (wasm + int8) shows 1–2 flags from *other*
  criteria (4.1.2, 2.4.4) that sit at 0.43–0.49 and cross 0.5 only from wasm rounding.

## Known limits — read before trusting output

- **52 of 63 criteria have no positive training example** (media 1.2.x, timing 2.2.1,
  sensory 1.3.3, reflow 1.4.10). Those columns are noise and should be ignored.
- Criteria that do work: **3.3.2, 4.1.2, 1.3.1, 1.1.1, 2.4.7, 1.4.3**.
- No CSS or rendered layout is available, so any criterion that depends on visual layout
  beyond inline colour is inherently weak.
- Two hand-written micro-snippets (single `<p>` with a colour) still rank wrong — they are
  outside the training distribution. Real pages separate correctly.

## Files

```
manifest.json              variant manifest: part list, size, sha256, chunk size
laya-wcag.onnx             the graph (307 KB) — external-data location: laya-wcag.onnx.data
laya-wcag.0000.onnx.data   weight parts, 15 × 25,165,824 B + 1 × 18,058,240 B
laya-wcag.0015.onnx.data   (395,545,600 B total)
tokenizer.json             ModernBERT tokenizer
tokenizer_config.json      tokenizer config
wcag_criteria.json         the 63 criteria, in classifier-column order
```

`laya-wcag.onnx` has **no inline weights** — stitch the 16 `.onnx.data` parts (in order, per
`manifest.json`) into one `laya-wcag.onnx.data` next to the graph before loading.

## Usage

```python
import glob, json, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer

# stitch the parts
buf = bytearray()
for p in sorted(glob.glob("laya-wcag.00*.onnx.data")):
    buf += open(p, "rb").read()
open("laya-wcag.onnx.data", "wb").write(bytes(buf))

sess = ort.InferenceSession(
    "laya-wcag.onnx",
    sess_options=ort.SessionOptions(),
    providers=["CPUExecutionProvider"],
)
tok = Tokenizer.from_file("tokenizer.json")
criteria = json.load(open("wcag_criteria.json"))

ids = tok.encode("<main><h1>Hi</h1><p>Hello</p></main>", add_special_tokens=False).ids
seq = [tok.token_to_id("[CLS]")] + ids[:254] + [tok.token_to_id("[SEP]")]
a = np.array([seq], dtype=np.int64)
logits = sess.run(None, {"input_ids": a, "attention_mask": np.ones_like(a)})[0]["criterion_scores"][0]
probs = 1 / (1 + np.exp(-logits))

for i, p in enumerate(probs):
    if p > 0.5:
        print(criteria[i]["criterion"], round(float(p), 3))
```

In the browser, see `web/laya-wcag-core.js` in
[turbolego/layaForWeb](https://github.com/turbolego/LayaWCAG) — it mounts the stitched parts
via ORT-web's `externalData` option, with `path` matching the graph's location string
(`laya-wcag.onnx.data`, not a part filename).
