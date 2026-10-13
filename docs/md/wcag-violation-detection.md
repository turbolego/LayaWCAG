---
title: "WCAG 2.2 Violation Detection with Laya"
published: false
description: "Fine-tuned Laya model for detecting WCAG 2.2 accessibility violations in HTML snippets. Trained on Axe-core benchmark data, deployed to browser via ONNX Runtime Web."
tags: ai, machinelearning, accessibility, wcag, laya, onnx, webassembly
---

# WCAG 2.2 Violation Detection with Laya

*October 2026*

A [Laya](https://github.com/convaiinnovations/laya) decision model fine-tuned to detect [WCAG 2.2](https://www.w3.org/TR/WCAG22/) accessibility violations from HTML snippets. Given a snippet of HTML, the model scores whether specific WCAG success criteria are violated, with confidence scores for each decision.

## How It Works

1. **Input**: An HTML snippet (e.g., from a webpage or axe-core violation report)
2. **Questions**: WCAG success criteria to check (e.g., "color-contrast-enhanced", "heading-order", "target-size")
3. **Output**: Yes/No probability for each criterion + confidence score

Example:

```js
const html = `<a href="#">Learn more</a>`;

const questions = {
  wcag_color_contrast_enhanced: {
    type: "yes_no",
    instructions: "Elements must meet enhanced color contrast ratio thresholds",
    criteria: { yes: "Violation detected", no: "No violation" }
  }
};

// Model response:
// { "wcag_color_contrast_enhanced": { "yes": 0.92, "no": 0.08, confidence: 0.84 } }
```

## Training Data

The model was trained on 40 samples extracted from the [AI-WCAG-Gauntlet](https://github.com/turbolego/AI-WCAG-Gauntlet) benchmark, which uses axe-core to audit real websites. Each sample contains:

- **HTML snippet**: Code with an accessibility violation
- **Violation type**: axe-core rule ID mapped to WCAG 2.2 success criteria
- **Label**: Correct yes/no classification

Common violation types in the training data:

| Rule ID | Description | Count |
|---------|-------------|-------|
| target-size | Touch targets too small | 9 |
| color-contrast-enhanced | Insufficient color contrast | 5 |
| heading-order | Incorrect heading hierarchy | 5 |
| landmark-complementary-is-top-level | Complementary landmark not top-level | 4 |
| color-contrast | Insufficient color contrast | 2 |

## Deployment

### Browser (via layaForWeb)

The model is converted to ONNX and deployed using the existing layaForWeb infrastructure:

```bash
# Build the model
python scripts/build_wcag_model.py --checkpoint build/wcag --out build/wcag-onnx

# Serve locally
python serve.py --model wcag

# Or deploy via GitHub Actions
```

### GitHub Actions Workflow

Add `.github/workflows/deploy-wcag.yml` to automate:

```yaml
name: Deploy WCAG Model

on:
  push:
    branches: [main]
    paths:
      - 'build/wcag/**'

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Build model
        run: |
          pip install -r requirements-build.txt
          python scripts/build_wcag_model.py --checkpoint build/wcag --out build/wcag-export
      - name: Deploy to Pages
        uses: actions/jekyll-build-publish@v1
        with:
          source_dir: build/wcag-export/
          destination_dir: wcag-model/
```

### Hugging Face

Upload the ONNX model to a Hugging Face repository:

```bash
python scripts/upload_to_hf.py turbolego/laya-wcag-violations
```

## Usage

### Direct (PyTorch)

```python
import laya

agent = laya.load("turbolego/laya-wcag", device="cpu")

html = "<a href='#'>Link</a>"
questions = {
    "wcag_color_contrast": {
        "type": "yes_no",
        "instructions": "Does this element have sufficient color contrast?",
        "criteria": {"yes": "Violation", "no": "OK"}
    }
}

result = agent.system_one(html, questions)
print(result)
# { "wcag_color_contrast": { "yes": 0.85, "no": 0.15, confidence: 0.70 } }
```

### Browser (ONNX Runtime Web)

Include the ONNX model in your HTML:

```html
<script type="module">
  import { loadWCAGModel } from './laya-core.js';
  
  const model = await loadWCAGModel('model.onnx');
  const result = await model.detect(html, questions);
  console.log(result);
</script>
```

## Performance

### Training Metrics

- **Model**: Laya English (convaiinnovations/laya)
- **Parameters**: ~421M
- **Training samples**: 40 (from Axe-core benchmark)
- **Epochs**: 3

### Limitations

- **Small dataset**: 40 samples is insufficient for production use. More data needed from real-world WCAG audits.
- **Binary classification**: Current model only detects yes/no violations. Severity levels (A/AA/AAA) not yet mapped.
- **HTML context**: Model sees raw HTML snippets, not full page structure.

## Next Steps

1. **Expand dataset**: Add more axe-core violation reports from diverse websites
2. **Multi-label**: Support multiple criteria per input
3. **Severity scoring**: Map violations to WCAG levels (A/AA/AAA)
4. **Integration**: Embed in layaForWeb demo page with live accessibility checking

## Related

- [layaForWeb](https://github.com/turbolego/LayaWCAG) - Browser deployment of Laya models
- [AI-WCAG-Gauntlet](https://github.com/turbolego/AI-WCAG-Gauntlet) - WCAG 2.2 benchmark
- [Axe-core](https://github.com/dequelabs/axe-core) - Accessibility testing engine
- [WCAG 2.2 Spec](https://www.w3.org/TR/WCAG22/) - Official specification
