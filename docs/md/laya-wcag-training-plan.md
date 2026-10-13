# Plan: Re-train Laya Model for WCAG 2.2 Violation Detection

**Repository:** https://github.com/turbolego/LayaWCAG
**Goal:** Fine-tune the Laya model to classify web accessibility violations per WCAG 2.2 criteria, deploy via Kaggle, and publish the resulting model to the layaForWeb repository.

---

## Phase 1: Create Kaggle Notebook

### 1.1 Setup
- Create new Kaggle notebook: `laya-wcag-training`
- Configure compute: GPU T4 x2 (Kaggle provides free)
- Install dependencies:
  - `torch` (PyTorch)
  - `transformers` (HuggingFace)
  - `datasets` (HuggingFace)
  - `onnx` (for model export)
  - `onnxruntime` (for verification)
  - `onnxscript` (required for torch.onnx.export with newer PyTorch)

### 1.2 Data Preparation
- Source datasets for training:
  - **Option A:** Use WCAG violation corpus (existing labeled datasets from WAI-ARIA, axe-core datasets)
  - **Option B:** Generate synthetic training data using LLM-augmented labeling
  - **Option C:** Combine existing accessibility evaluation datasets
- Create training data in format: `{text: str, violations: list[WCAG_criteria]}`
- Fields per violation:
  - `criterion` (e.g., "1.1.1", "1.2.1")
  - `principle` (Perceivable, Operable, Understandable, Robust)
  - `level` (A, AA, AAA)
  - `description` (natural language violation description)

### 1.3 Next Steps
- [x] Create initial Kaggle notebook with data loading skeleton
- [x] Add data loading for WCAG 2.2 training corpus
- [x] Add model architecture (multi-label classification heads)
- [x] Add training loop
- [x] Add evaluation metrics
- [x] Add ONNX export and quantization
- [x] Expand WCAG 2.2 criteria from 5 to 63 (covers all Level A and AA criteria)

---

## Phase 2: Model Training

### 2.1 Architecture Adaptation
- Load base Laya model (ModernBERT-large encoder + 2-layer transformer head)
- Adapt for multi-label classification:
  - Add classification heads for each WCAG criterion
  - Output: probabilities for each WCAG 2.2 criterion (63+ criteria)
  - Yes/No decision per criterion
  - Confidence score per prediction
- **Implementation note:** Use `laya.load()` to get the agent, then access `agent.model` and `agent.tok`
- **Pitfall:** Do NOT use `AutoTokenizer.from_pretrained()` / `AutoModel.from_pretrained()` — raises `ValueError: Couldn't instantiate the backend tokenizer`

### 2.2 Training
- Freeze the encoder (421M params) — only train the classifier head (5K params)
- Use batch_size=4, learning_rate=2e-5, epochs=3
- Keep sequence length short (128 tokens) to avoid OOM on single T4
- Loss: Binary cross-entropy with logits (BCEWithLogitsLoss)

---

## Phase 3: Evaluation and Validation

### 3.1 Training Metrics
- Track: macro F1, precision, recall
- Current synthetic training: F1=0.20 (expected — synthetic data is trivial)

### 3.2 Validation Sets
- Hold-out test set: 20% of training data
- Cross-validation: 5-fold
- External validation: axe-core test cases, WAI website examples

### 3.3 Error Analysis
- Identify criteria with low recall
- Check for systematic false positives
- Analyze confusion matrix for common violations

---

## Phase 4: Model Export & Optimization

### 4.1 ONNX Export
```python
import torch
import onnx

# Export trained model
torch.onnx.export(
    model,
    dummy_input,
    "laya-wcag.onnx",
    opset_version=17,
    input_names=['input_ids', 'attention_mask'],
    output_names=['criterion_scores', 'confidence']
)
```

### 4.2 Quantization
- Apply weight-only quantization (similar to q8e8 build in layaForWeb)
- Verify accuracy retention (>95% of float32 performance)
- Target model size: <400MB

### 4.3 Manifest Generation
- Split into 24 MiB parts
- Generate SHA-256 hashes
- Create `manifest.json` for browser caching

---

## Phase 5: Integration with layaForWeb

### 5.1 Repository Updates
- Add new notebook to `/notebooks/`
- Add trained model weights to releases or HuggingFace Hub
- Update `serve.py` to load WCAG-specific model
- Add configuration for WCAG violation detection mode

### 5.2 API Extension
- Extend `systemOne()` to handle WCAG detection:
```javascript
const questions = {
  wcag_violations: {
    type: "choice",
    instructions: "Which WCAG 2.2 criteria are violated?",
    criteria: wcag_criteria_list  // 63+ criteria
  }
};
```

### 5.3 Documentation
- Add `docs/md/laya-wcag-training.md` explaining:
  - Training process
  - Model capabilities
  - Usage examples
  - Limitations
- Update README with WCAG detection as new use case

---

## Phase 6: Deployment & Publishing

### 6.1 Kaggle Publishing
- Push notebook to Kaggle
- Set as public (or private with share link)
- Document model weights download link

### 6.2 GitHub Release
- Create GitHub release with:
  - Model weights (.onnx files)
  - Manifest file
  - Training artifacts (config, metrics)
- Tag commit with `v0.1-wcag`

### 6.3 Model Registry
- Upload to HuggingFace Hub:
  - Repository: `turbolego/laya-wcag`
  - Include model card with training details
  - Add usage examples

---

## Current Status (2026-10-08)

- ✅ Notebook created at `/tmp/layaForWeb/notebooks/laya_wcag_training.ipynb`
- ✅ Kernel pushed to Kaggle (version 11)
- ✅ 63 WCAG 2.2 criteria defined (all Level A and AA)
- ✅ Synthetic training passes (F1=0.20 on synthetic data)
- ⏳ Running kernel v11: training + ONNX export with `onnxscript` fix
- ⏹ Previous run (v10) failed: `ModuleNotFoundError: No module named 'onnxscript'`
- 🔗 https://www.kaggle.com/code/hummern/laya-wcag-2-2-violation-detection-training

---

## Known Issues & Fixes

### Issue: `ValueError: Couldn't instantiate the backend tokenizer`
- **Cause:** Using `AutoTokenizer.from_pretrained("convaiinnovations/laya")`
- **Fix:** Use `laya.load("convaiinnovations/laya", device=device)` then `agent.tok`

### Issue: `ModuleNotFoundError: No module named 'onnxscript'`
- **Cause:** Newer PyTorch requires `onnxscript` for ONNX export
- **Fix:** Add `onnxscript` to pip install line

### Issue: OOM on single T4
- **Cause:** Too large batch size or sequence length
- **Fix:** batch_size=4, MAX_LEN=128, freeze encoder
