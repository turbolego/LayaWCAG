---
title: "Train Laya WCAG 2.2 Violation Detector with Unsloth on Kaggle"
published: 2026-10-09
description: "Train a Laya System 1 decision model to detect WCAG 2.2 violations from HTML snippets using Unsloth's FastDecisionModel on Kaggle free T4×2 GPUs. Convert to ONNX and publish to turbolego/LayaWCAG."
tags:
  - laya
  - wcag
  - accessibility
  - unsloth
  - decision-model
  - kaggle
  - onnx
---

## Context

- **Source repo**: `turbolego/LayaWCAG` (GitHub fork at `/media/kloa/.../repos/turbolego-layaForWeb`)
- **Base model**: `convaiinnovations/laya` (421M params, ModernBERT-large backbone + decision head)
- **Training method**: Unsloth's `FastDecisionModel` with LoRA (r=16) for decision model fine-tuning
- **Platform**: Kaggle free T4×2 (2x Tesla T4) via Unsloth notebooks
- **Benchmark data**: `AI-WCAG-Gauntlet` at `/media/kloa/.../repos/AI-WCAG-Gauntlet` with WCAG 2.2 Level AAA criteria and axe-core rules

## Research Findings: Unsloth + Laya Decision Model Training

### Unsloth Decision Model Capability

Unsloth's `FastDecisionModel` converts any LLM into a System 1 decision engine. Key results:

| Model | Dataset | Accuracy | Setup |
|-------|---------|----------|-------|
| Qwen3.5-0.8B (Unsloth) | BANKING77 | 75.4% | LoRA r=16, 2 epochs, lr=2e-4 |
| Qwen3.5-0.8B (Unsloth) | CLINC150 | 75.0% | LoRA r=16, 2 epochs, lr=2e-4 |
| Qwen3.5-4B (Unsloth) | typed-decisions | 77% | LoRA r=16, 2 epochs, lr=2e-4 |
| Laya (convaiinnovations) | typed-decisions | 76.6% | RLCD, 1 epoch |
| Jev 1.13.0 (TypeSafe) | typed-decisions | 72.7% | — |

Unsloth benchmarks use 2 epochs, LoRA rank 16, learning rate 2e-4, gradient_accumulation_steps=4, batch_size=8. Kaggle free T4×2 provides sufficient VRAM (16GB total).

### Laya Architecture

- **Backbone**: ModernBERT-large (395M params, fully fine-tuned)
- **Decision head**: 2 transformer layers + option marker scorer + act/escalate head
- **Input**: 512 tokens per question (question + options + state)
- **Output**: Typed decisions with probabilities and confidence scores
- **Training**: RLCD (Reinforcement Learning for Calibrated Decisions) with strictly proper scoring rules
- **Multilingual**: Supports 100+ languages via mmBERT-base (322M)

### Dataset Format (Laya/Unsloth Decision Model)

Each dataset row has three fields:

```json
{
  "text": "<html snippet or context>",
  "questions": {
    "wcag_1_1_1": {
      "type": "yes_no",
      "options": ["yes", "no"],
      "instructions": "Does all non-text content have a text alternative?"
    },
    "wcag_1_4_3": {
      "type": "yes_no",
      "options": ["yes", "no"],
      "instructions": "Can text be resized up to 200% without loss of content or functionality?"
    }
  },
  "gold": {
    "wcag_1_1_1": "yes",
    "wcag_1_4_3": "no"
  }
}
```

Available question types: `yes_no` (binary), `choice` (multi-class), `score` (1-5 scale), `yes_no_na` (ternary).

### Available WCAG 2.2 Training Data

| Dataset | Source | Description |
|---------|--------|-------------|
| **A11YBench** | `LLM4APR/A11YBench` | 60 real-world web projects, 147 pages, 8,886 violations from IBM Accessibility Checker. Multi-domain (UI libs, dev tools, cloud platforms, docs). Two configs: Lite (10 repos) and Full. |
| **Web Accessibility Compliance** | Kaggle | 50 major websites audited with axe-core via Playwright, 85 violations mapped to WCAG 2.0/2.1. |
| **frontend-accessibility-dataset** | GitHub | Structured dataset of frontend accessibility violations for AI code review. |
| **AI-WCAG-Gauntlet** (local) | `/repos/AI-WCAG-Gauntlet` | WCAG 2.2 Level AAA criteria in `resources/wcag_22.json`, axe-core rules in `resources/axe_rules_aaa.md`. Benchmark results in `benchmarks/*/axe_report.json`. |

### Training Pipeline (Unsloth + Kaggle)

1. **Data preparation**: Convert A11YBench/AI-WCAG-Gauntlet data to Laya decision format
2. **Kaggle notebook**: Use Unsloth's Kaggle notebook template with T4×2 GPU
3. **Training**: `FastDecisionModel.build_dataset()` → `DecisionTrainer` with LoRA r=16, 2 epochs, lr=2e-4
4. **Evaluation**: `FastDecisionModel.evaluate()` on held-out test set
5. **Conversion**: Convert fine-tuned model to ONNX for browser deployment

## Updated Plan

### Step 1: Build WCAG Decision Dataset

**Goal**: Create JSONL dataset mapping HTML snippets to WCAG 2.2 violation checks.

**Data sources**:
- `AI-WCAG-Gauntlet/benchmarks/*/axe_report.json` - extract HTML snippets with violations
- `LLM4APR/A11YBench` (Hugging Face) - 8,886 real violations from 60 projects
- Local axe-core reports

**Dataset format** (Laya decision model schema):

```json
{
  "text": "<html snippet extracted from benchmark>",
  "questions": {
    "wcag_1_1_1": {
      "type": "yes_no",
      "options": ["yes", "no"],
      "instructions": "Does all non-text content have a text alternative?"
    },
    "wcag_1_2_2": {
      "type": "yes_no",
      "options": ["yes", "no"],
      "instructions": "Is synchronized text provided for all prerecorded audio?"
    }
  },
  "gold": {
    "wcag_1_1_1": "no",
    "wcag_1_2_2": "yes"
  }
}
```

**Priority criteria** (Level A/AA):
- 1.1.1 (img alt), 1.2.1-3 (audio/video captions), 1.3.1 (semantic HTML)
- 1.4.3 (contrast), 1.4.4 (resize), 1.4.10-12 (responsive)
- 2.1.1 (keyboard), 2.1.2 (no keyboard trap), 2.4.1-6 (navigation)
- 2.5.1-5 (touch targets), 3.1.1 (language), 3.2.1-4 (predictable)
- 3.3.1-3 (form errors), 4.1.1-2 (parser/name)
- AAA: 1.2.6-8 (sign language/audio description), 1.3.5 (identify purpose), 1.4.13 (hover/focus), 2.4.10-12, 2.5.7-10, 3.1.3-6, 3.2.3-6, 3.3.4-6, 3.5.1-3, 4.1.3-5

### Step 2: Create Kaggle Training Notebook

**Template**: Use Unsloth's Kaggle notebook or `AAB20/kaggle-UNSLOTH-NOTEBOOKS` collection.

**Key configuration**:
```python
from unsloth import FastDecisionModel, DecisionTrainer, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments

model, tokenizer = FastDecisionModel.from_pretrained(
    model_name="convaiinnovations/laya",
    max_seq_length=2048,
    load_in_4bit=True,
)

dataset = load_dataset("parquet", data_files="laya_wcag_train.parquet")
items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
print(f"Skipped {report['skipped']} of {report['total']} decisions")

train_items, eval_items = FastDecisionModel.split_holdout(items, seed=3407)

trainer = DecisionTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_items,
    eval_dataset=eval_items,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        eval_strategy="epoch",
        logging_steps=10,
        output_dir="outputs",
        report_to="none",
        seed=3407,
    ),
)
```

**Training parameters**:
- LoRA rank: 16
- Epochs: 2 (adjust based on validation loss)
- Learning rate: 2e-4
- Batch size: 8 (per device), gradient accumulation: 4
- Max seq length: 2048 (wcag snippets + criteria text)
- Freeze: ModernBERT backbone optional (train only decision head first)

### Step 3: Build Laya Training Script

**File**: `scripts/train_laya_wcag.py`

**Sections**:
1. Data loading from Hugging Face or local JSONL/Parquet
2. Dataset building with `FastDecisionModel.build_dataset()`
3. Training with `DecisionTrainer`
4. Evaluation with `FastDecisionModel.evaluate()`
5. Save model checkpoint to `checkpoints/laya-wcag/`

### Step 4: Convert to ONNX and Integrate

**Conversion**: Use `scripts/build_model.py` from layaForWeb repo:
```bash
python scripts/build_model.py \
  --repo turbolego/laya-wcag \
  --out build-wcag \
  --variants q8e8,q4e8
```

**Integration into layaForWeb**:
- Place ONNX model in `build-wcag/`
- Update `src/laya-core.js` or equivalent to load new model
- Update UI to show WCAG criteria and violation results
- Add verification script `tests/seq_parity.mjs`

### Step 5: Document and Publish

**Doc page**: `docs/md/laya-wcag-training.md`
- Training methodology
- Dataset sources and size
- Benchmark results (before/after fine-tuning)
- WCAG criteria covered
- Usage examples

## Hyperparameter Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | convaiinnovations/laya (421M) | ModernBERT-large backbone |
| Training method | FastDecisionModel + LoRA | Unsloth's decision head |
| LoRA rank | 16 | Unsloth recommended default |
| Epochs | 2-3 | Adjust based on validation loss |
| Learning rate | 2e-4 | Unsloth recommended |
| Batch size | 8 (per device) | T4×2: effective 32 with grad_accum=4 |
| Max seq length | 2048 | WCAG snippets + criteria |
| Platform | Kaggle T4×2 | Free, sufficient VRAM |
| Expected accuracy | 70-80% | Based on Laya-typed-decisions (76.6%) |

## Key References

- [Unsloth Decision Model Guide](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)
- [Laya-typed-decisions benchmark](https://huggingface.co/convaiinnovations/laya-typed-decisions) - 0.766 accuracy
- [FastDecisionModel API](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)
- [A11YBench dataset](https://huggingface.co/datasets/LLM4APR/A11YBench) - 8,886 violations
- [Kaggle Unsloth notebooks](https://github.com/AAB20/kaggle-UNSLOTH-NOTEBOOKS)
- [Laya fine-tuning guide](https://nandhakishorm.github.io/laya/finetune/)
- [AI-WCAG-Gauntlet benchmarks](https://github.com/turbolego/AI-WCAG-Gauntlet)
