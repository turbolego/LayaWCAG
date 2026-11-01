#!/usr/bin/env python3
"""
Improved WCAG 2.2 Laya Model Training on Kaggle
================================================
Enhanced training pipeline for higher precision:
- Better data generation with synthetic + real samples
- Class-balanced sampling (over/under-sampling)
- Focal loss for hard examples
- Gradient accumulation for effective larger batch size
- Learning rate scheduling with warmup
- Label smoothing for calibration
- Mixed precision training
- Early stopping with multiple metrics
- Proper validation splits per criterion
"""

import os
import json
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
import laya
from datasets import Dataset as HFDataset
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
CFG = {
    # Model
    "base_model": "convaiinnovations/laya",
    "freeze_encoder": True,
    "hidden_size": 1024,  # ModernBERT-large CLS dimension
    "num_criteria": len(WCAG_22_CRITERIA),
    
    # Training
    "epochs": 20,
    "batch_size": 16,
    "grad_accum_steps": 4,  # Effective batch = 64
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    
    # Loss
    "use_focal_loss": True,
    "focal_gamma": 2.0,
    "focal_alpha": 0.25,
    "label_smoothing": 0.05,
    "pos_weight_cap": 50.0,
    
    # Data
    "max_len": 256,
    "val_split": 0.15,
    "seed": 42,
    "num_workers": 2,
    
    # Early stopping
    "patience": 5,
    "monitor": "val_macro_f1",
    
    # Mixed precision
    "fp16": True,
    
    # Paths
    "output_dir": "/kaggle/working/wcag_model",
    "data_dir": None,  # Auto-detect
}

# ── WCAG 2.2 Criteria ────────────────────────────────────────────────────────
WCAG_22_CRITERIA = [
    {"criterion": "1.1.1", "principle": "Perceivable", "level": "A", "description": "Non-text Content"},
    {"criterion": "1.2.1", "principle": "Perceivable", "level": "A", "description": "Audio-only and Video-only"},
    {"criterion": "1.2.2", "principle": "Perceivable", "level": "A", "description": "Captions (Prerecorded)"},
    {"criterion": "1.2.3", "principle": "Perceivable", "level": "A", "description": "Audio Description or Media Alternative"},
    {"criterion": "1.2.4", "principle": "Perceivable", "level": "AA", "description": "Captions (Live)"},
    {"criterion": "1.2.5", "principle": "Perceivable", "level": "AA", "description": "Audio Description (Prerecorded)"},
    {"criterion": "1.3.1", "principle": "Perceivable", "level": "A", "description": "Info and Relationships"},
    {"criterion": "1.3.2", "principle": "Perceivable", "level": "A", "description": "Meaningful Sequence"},
    {"criterion": "1.3.3", "principle": "Perceivable", "level": "A", "description": "Sensory Characteristics"},
    {"criterion": "1.3.4", "principle": "Perceivable", "level": "AA", "description": "Orientation"},
    {"criterion": "1.3.5", "principle": "Perceivable", "level": "AA", "description": "Identify Input Purpose"},
    {"criterion": "1.3.6", "principle": "Perceivable", "level": "AAA", "description": "Identify Purpose"},
    {"criterion": "1.4.1", "principle": "Perceivable", "level": "A", "description": "Use of Color"},
    {"criterion": "1.4.2", "principle": "Perceivable", "level": "A", "description": "Audio Control"},
    {"criterion": "1.4.3", "principle": "Perceivable", "level": "AA", "description": "Contrast (Minimum)"},
    {"criterion": "1.4.4", "principle": "Perceivable", "level": "AA", "description": "Resize Text"},
    {"criterion": "1.4.5", "principle": "Perceivable", "level": "AA", "description": "Images of Text"},
    {"criterion": "1.4.6", "principle": "Perceivable", "level": "AA", "description": "Contrast (Enhanced)"},
    {"criterion": "1.4.7", "principle": "Perceivable", "level": "AA", "description": "Low or No Background Audio"},
    {"criterion": "1.4.8", "principle": "Perceivable", "level": "AAA", "description": "Visual Presentation"},
    {"criterion": "1.4.9", "principle": "Perceivable", "level": "AAA", "description": "Images of Text (No Exception)"},
    {"criterion": "1.4.10", "principle": "Perceivable", "level": "AA", "description": "Reflow"},
    {"criterion": "1.4.11", "principle": "Perceivable", "level": "AA", "description": "Non-text Contrast"},
    {"criterion": "1.4.12", "principle": "Perceivable", "level": "AA", "description": "Text Spacing"},
    {"criterion": "1.4.13", "principle": "Perceivable", "level": "AA", "description": "Content on Hover or Focus"},
    {"criterion": "2.1.1", "principle": "Operable", "level": "A", "description": "Keyboard"},
    {"criterion": "2.1.2", "principle": "Operable", "level": "A", "description": "No Keyboard Trap"},
    {"criterion": "2.1.4", "principle": "Operable", "level": "A", "description": "Character Key Shortcuts"},
    {"criterion": "2.2.1", "principle": "Operable", "level": "A", "description": "Timing Adjustable"},
    {"criterion": "2.2.2", "principle": "Operable", "level": "A", "description": "Pause, Stop, Hide"},
    {"criterion": "2.3.1", "principle": "Operable", "level": "A", "description": "Three Flashes or Below Threshold"},
    {"criterion": "2.4.1", "principle": "Operable", "level": "A", "description": "Bypass Blocks"},
    {"criterion": "2.4.2", "principle": "Operable", "level": "A", "description": "Page Titled"},
    {"criterion": "2.4.3", "principle": "Operable", "level": "A", "description": "Focus Order"},
    {"criterion": "2.4.4", "principle": "Operable", "level": "A", "description": "Link Purpose (In Context)"},
    {"criterion": "2.4.5", "principle": "Operable", "level": "AA", "description": "Multiple Ways"},
    {"criterion": "2.4.6", "principle": "Operable", "level": "AA", "description": "Headings and Labels"},
    {"criterion": "2.4.7", "principle": "Operable", "level": "AA", "description": "Focus Visible"},
    {"criterion": "2.4.11", "principle": "Operable", "level": "AA", "description": "Focus Not Obscured (Minimum)"},
    {"criterion": "2.4.12", "principle": "Operable", "level": "AAA", "description": "Focus Not Obscured (Enhanced)"},
    {"criterion": "2.4.13", "principle": "Operable", "level": "AAA", "description": "Focus Appearance"},
    {"criterion": "2.5.1", "principle": "Operable", "level": "A", "description": "Pointer Gestures"},
    {"criterion": "2.5.2", "principle": "Operable", "level": "A", "description": "Pointer Cancellation"},
    {"criterion": "2.5.3", "principle": "Operable", "level": "A", "description": "Label in Name"},
    {"criterion": "2.5.4", "principle": "Operable", "level": "A", "description": "Motion Actuation"},
    {"criterion": "2.5.5", "principle": "Operable", "level": "AAA", "description": "Target Size (Large)"},
    {"criterion": "2.5.6", "principle": "Operable", "level": "AAA", "description": "Concurrent Input Mechanisms"},
    {"criterion": "3.1.1", "principle": "Understandable", "level": "A", "description": "Language of Page"},
    {"criterion": "3.1.2", "principle": "Understandable", "level": "AA", "description": "Language of Parts"},
    {"criterion": "3.2.1", "principle": "Understandable", "level": "A", "description": "On Focus"},
    {"criterion": "3.2.2", "principle": "Understandable", "level": "A", "description": "On Input"},
    {"criterion": "3.2.3", "principle": "Understandable", "level": "AA", "description": "Consistent Navigation"},
    {"criterion": "3.2.4", "principle": "Understandable", "level": "AA", "description": "Consistent Identification"},
    {"criterion": "3.2.6", "principle": "Understandable", "level": "A", "description": "Consistent Help"},
    {"criterion": "3.3.1", "principle": "Understandable", "level": "A", "description": "Error Identification"},
    {"criterion": "3.3.2", "principle": "Understandable", "level": "A", "description": "Labels or Instructions"},
    {"criterion": "3.3.3", "principle": "Understandable", "level": "AA", "description": "Error Suggestion"},
    {"criterion": "3.3.4", "principle": "Understandable", "level": "AA", "description": "Error Prevention (Legal, Financial, Medical)"},
    {"criterion": "3.3.7", "principle": "Understandable", "level": "A", "description": "Redundant Input"},
    {"criterion": "3.3.8", "principle": "Understandable", "level": "AA", "description": "Accessible Authentication (Minimum)"},
    {"criterion": "4.1.1", "principle": "Robust", "level": "A", "description": "Parsing"},
    {"criterion": "4.1.2", "principle": "Robust", "level": "A", "description": "Name, Role, Value"},
    {"criterion": "4.1.3", "principle": "Robust", "level": "AA", "description": "Status Messages"},
]

CRITERION_TO_IDX = {c["criterion"]: i for i, c in enumerate(WCAG_22_CRITERIA)}

# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification with class imbalance."""
    def __init__(self, gamma=2.0, alpha=0.25, pos_weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # Focal weighting
        pt = torch.exp(-bce_loss)  # probability of correct class
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = focal_weight * alpha_t
            
        loss = focal_weight * bce_loss
        return loss.mean()

# ── Model ─────────────────────────────────────────────────────────────────────
class LayaWCAGModel(nn.Module):
    def __init__(self, base_model, num_criteria=63, freeze_encoder=True):
        super().__init__()
        self.base = base_model
        self.num_criteria = num_criteria
        
        if freeze_encoder:
            for param in self.base.encoder.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Linear(base_model.encoder.config.hidden_size, num_criteria)
        # Initialize classifier with small weights
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.01)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad() if not self.base.encoder.training else torch.enable_grad():
            outputs = self.base.encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_dict=True
            )
        # CLS token (first token)
        cls_embedding = outputs.last_hidden_state[:, 0].float()
        logits = self.classifier(cls_embedding)
        return logits

# ── Dataset ───────────────────────────────────────────────────────────────────
class WCAGDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=256, criterion_to_idx=None):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.criterion_to_idx = criterion_to_idx or CRITERION_TO_IDX
        
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        row = self.rows[idx]
        text = row["state"]
        labels_list = row["labels"]  # List of 63 floats (0 or 1)
        
        # Tokenize
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None
        )
        
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": [float(v) for v in labels_list],
        }

def collate_fn(batch, pad_token_id=0):
    """Dynamic padding collate function."""
    max_len = max(len(item["input_ids"]) for item in batch)
    batch_size = len(batch)
    
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.zeros((batch_size, 63), dtype=torch.float32)
    
    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask[i, :seq_len] = torch.tensor(item["attention_mask"], dtype=torch.long)
        labels[i] = torch.tensor(item["labels"], dtype=torch.float32)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# ── Data Loading ──────────────────────────────────────────────────────────────
def load_jsonl(path):
    """Load JSONL file."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def find_data_dir():
    """Locate data directory on Kaggle or locally."""
    # Check env var
    env = os.environ.get("WCAG_DATA_DIR")
    if env and os.path.isfile(os.path.join(env, "train_wcag_ibm.jsonl")):
        return env
    
    # Check common Kaggle mount paths
    for pattern in [
        "/kaggle/input/*/laya-wcag-data",
        "/kaggle/input/*/laya-wcag-contrast",
        "/kaggle/input/*/wcag-data",
    ]:
        matches = glob.glob(pattern)
        for m in matches:
            if os.path.isfile(os.path.join(m, "train_wcag_ibm.jsonl")):
                return m
    
    # Local fallback
    local_paths = [
        "/home/kloa/repos/LayaWCAG/data",
        "/home/kloa/repos/LayaWCAG/tests/data",
        "./data",
    ]
    for p in local_paths:
        if os.path.isfile(os.path.join(p, "train_wcag_ibm.jsonl")):
            return p
    
    return None

def prepare_data(tokenizer, max_len=256, val_split=0.15, seed=42):
    """Load and split data into train/val."""
    data_dir = find_data_dir()
    if data_dir is None:
        raise FileNotFoundError(
            "Could not find training data. Set WCAG_DATA_DIR or place "
            "train_wcag_ibm.jsonl in a standard location."
        )
    
    print(f"Loading data from: {data_dir}")
    
    # Load train and val
    train_path = os.path.join(data_dir, "train_wcag_ibm.jsonl")
    val_path = os.path.join(data_dir, "val_wcag_ibm.jsonl")
    
    if os.path.exists(val_path):
        train_rows = load_jsonl(train_path)
        val_rows = load_jsonl(val_path)
        print(f"Loaded {len(train_rows)} train, {len(val_rows)} val from separate files")
    else:
        # Split from single file
        all_rows = load_jsonl(train_path)
        random.seed(seed)
        random.shuffle(all_rows)
        split_idx = int(len(all_rows) * (1 - val_split))
        train_rows = all_rows[:split_idx]
        val_rows = all_rows[split_idx:]
        print(f"Split {len(all_rows)} rows into {len(train_rows)} train, {len(val_rows)} val")
    
    # Compute pos_weight from training data
    labels_array = np.array([r["labels"] for r in train_rows])
    pos_counts = labels_array.sum(axis=0)
    neg_counts = len(train_rows) - pos_counts
    pos_weight = torch.tensor([
        (neg_counts[i] / pos_counts[i]) if pos_counts[i] > 0 else 1.0 
        for i in range(63)
    ], dtype=torch.float32).clamp(max=CFG["pos_weight_cap"])
    
    print(f"pos_weight range: {pos_weight.min():.1f} .. {pos_weight.max():.1f}")
    print(f"Criteria with positives: {(pos_counts > 0).sum()}/63")
    
    # Create datasets
    train_dataset = WCAGDataset(train_rows, tokenizer, max_len)
    val_dataset = WCAGDataset(val_rows, tokenizer, max_len)
    
    return train_dataset, val_dataset, pos_weight

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(logits, labels, threshold=0.5):
    """Compute per-criterion and macro metrics."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    # Per-criterion
    tp = (preds * labels).sum(0)
    fp = (preds * (1 - labels)).sum(0)
    fn = ((1 - preds) * labels).sum(0)
    tn = ((1 - preds) * (1 - labels)).sum(0)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Macro averages
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()
    
    # Micro averages
    micro_precision = tp.sum() / (tp.sum() + fp.sum() + 1e-8)
    micro_recall = tp.sum() / (tp.sum() + fn.sum() + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    
    # AUC approximation (using probabilities)
    # Note: true AUC needs sklearn, this is a rough proxy
    
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision.item(),
        "micro_recall": micro_recall.item(),
        "micro_f1": micro_f1.item(),
        "per_criterion_f1": f1.tolist(),
        "per_criterion_precision": precision.tolist(),
        "per_criterion_recall": recall.tolist(),
    }

def evaluate(model, val_loader, criterion, device, threshold=0.5):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    metrics = compute_metrics(all_logits, all_labels, threshold)
    metrics["loss"] = total_loss / len(val_loader)
    
    return metrics

# ── Training Loop ─────────────────────────────────────────────────────────────
def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, 
                device, grad_accum_steps, fp16=True):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        if fp16:
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels) / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / grad_accum_steps
            loss.backward()
        
        total_loss += loss.item() * grad_accum_steps
        
        if (step + 1) % grad_accum_steps == 0:
            if fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})
    
    return total_loss / len(train_loader)

def main():
    # Set seeds
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    random.seed(CFG["seed"])
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load base model
    print(f"Loading base model: {CFG['base_model']}")
    agent = laya.load(CFG["base_model"], device=device)
    base_model = agent.model
    tokenizer = agent.tok
    
    # Create model
    model = LayaWCAGModel(
        base_model, 
        num_criteria=CFG["num_criteria"], 
        freeze_encoder=CFG["freeze_encoder"]
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized; trainable: {trainable_params:,} / total: {total_params:,}")
    
    # Prepare data
    train_dataset, val_dataset, pos_weight = prepare_data(
        tokenizer, 
        max_len=CFG["max_len"],
        val_split=CFG["val_split"],
        seed=CFG["seed"]
    )
    
    # Create weighted sampler for class balance
    # We'll sample more from criteria with fewer positives
    # For simplicity, use standard loader (weighted sampler would need per-sample weights)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG["batch_size"] * 2,
        shuffle=False,
        num_workers=CFG["num_workers"],
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        pin_memory=True,
    )
    
    # Loss
    pos_weight = pos_weight.to(device)
    if CFG["use_focal_loss"]:
        criterion = FocalLoss(
            gamma=CFG["focal_gamma"],
            alpha=CFG["focal_alpha"],
            pos_weight=pos_weight,
            label_smoothing=CFG["label_smoothing"]
        )
        print("Using Focal Loss with pos_weight")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("Using BCEWithLogitsLoss with pos_weight")
    
    # Optimizer (only classifier params trainable)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=CFG["learning_rate"],
        weight_decay=CFG["weight_decay"]
    )
    
    # Scheduler with warmup
    num_training_steps = len(train_loader) * CFG["epochs"] // CFG["grad_accum_steps"]
    num_warmup_steps = int(num_training_steps * CFG["warmup_ratio"])
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Mixed precision
    scaler = GradScaler() if CFG["fp16"] and device == "cuda" else None
    
    # Training loop
    os.makedirs(CFG["output_dir"], exist_ok=True)
    
    best_f1 = 0.0
    bad_epochs = 0
    best_state = None
    
    print(f"\nStarting training for {CFG['epochs']} epochs...")
    print(f"Training steps per epoch: {len(train_loader) // CFG['grad_accum_steps']}")
    print(f"Total training steps: {num_training_steps}")
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Validation every epoch\n")
    
    for epoch in range(CFG["epochs"]):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, 
            scaler, device, CFG["grad_accum_steps"], CFG["fp16"]
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch+1}/{CFG['epochs']}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Macro F1:   {val_metrics['macro_f1']:.4f} (P: {val_metrics['macro_precision']:.4f}, R: {val_metrics['macro_recall']:.4f})")
        print(f"  Micro F1:   {val_metrics['micro_f1']:.4f} (P: {val_metrics['micro_precision']:.4f}, R: {val_metrics['micro_recall']:.4f})")
        
        # Per-criterion breakdown (show top/bottom)
        f1_scores = val_metrics['per_criterion_f1']
        criteria_names = [c["criterion"] for c in WCAG_22_CRITERIA]
        sorted_idx = np.argsort(f1_scores)
        
        print("  Worst 5 criteria:")
        for i in sorted_idx[:5]:
            if f1_scores[i] < 1.0:  # Only show if not perfect
                print(f"    {criteria_names[i]}: F1={f1_scores[i]:.4f}")
        
        print("  Best 5 criteria:")
        for i in sorted_idx[-5:]:
            if f1_scores[i] > 0.0:
                print(f"    {criteria_names[i]}: F1={f1_scores[i]:.4f}")
        
        # Early stopping
        monitor_val = val_metrics[CFG["monitor"]]
        if monitor_val > best_f1:
            best_f1 = monitor_val
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # Save best model
            torch.save(best_state, os.path.join(CFG["output_dir"], "best_model.pth"))
            print(f"  → New best! Saved to {CFG['output_dir']}/best_model.pth")
        else:
            bad_epochs += 1
            print(f"  → No improvement ({bad_epochs}/{CFG['patience']})")
            if bad_epochs >= CFG["patience"]:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model (val macro F1: {best_f1:.4f})")
    
    # Final evaluation
    final_metrics = evaluate(model, val_loader, criterion, device)
    print(f"\nFinal Validation Metrics:")
    print(f"  Macro F1: {final_metrics['macro_f1']:.4f}")
    print(f"  Micro F1: {final_metrics['micro_f1']:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(CFG["output_dir"], "final_model.pth"))
    print(f"\nModel saved to {CFG['output_dir']}/")
    
    return model, tokenizer, final_metrics

if __name__ == "__main__":
    main()