#!/usr/bin/env python3
"""Rewrite notebooks/laya_wcag_training.ipynb to train on real IBM-checker-labeled data.

Replaces the synthetic-data cells with a real-data pipeline:
  - loads train_wcag_ibm.jsonl / val_wcag_ibm.jsonl (produced by the axe/IBM labeling step,
    each line {state, labels: [0|1 x 63]})
  - keeps the frozen-encoder + Linear(1024, 63) head and BCEWithLogitsLoss
  - trains with pos_weight to counter the extreme class imbalance (9/63 criteria have any
    positive example locally; most criteria are all-negative)
  - fixes the ONNX export: strips the invalid num_outputs attribute from Split nodes and
    externalizes the weights with size_threshold=1024 so ORT-web can load the graph
  - writes the site-ready manifest in the variants format the web app reads

Usage:  python scripts/retrain_notebook.py
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "laya_wcag_training.ipynb"

CRITERIA_CELL = '''import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset

# Define WCAG 2.2 criteria (the classifier's 63 columns, in order)
WCAG_22_CRITERIA = %CRITERIA%
num_criteria = len(WCAG_22_CRITERIA)'''

DATA_CELL = '''## Load real labeled data (IBM Equal Access checker -> WCAG 2.2 SCs)

import json, os, glob

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

# The dataset slug is mounted at /kaggle/input/<owner>/<slug>/; locate it robustly because
# Kaggle sometimes mounts under a slightly different path for freshly created datasets.
def find_data_dir():
    env = os.environ.get("WCAG_DATA_DIR")
    cands = [env] if env else []
    cands += glob.glob("/kaggle/input/*/laya-wcag-data") + glob.glob("/kaggle/input/laya-wcag-data")
    cands += glob.glob("/kaggle/input/**/train_wcag_ibm.jsonl", recursive=True)
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "train_wcag_ibm.jsonl")):
            return c
    # last resort: search for the file anywhere under /kaggle/input
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train_wcag_ibm.jsonl" in files:
            return root
    raise FileNotFoundError("train_wcag_ibm.jsonl not found under /kaggle/input")

DATA_DIR = find_data_dir()
print("data dir:", DATA_DIR)
train_rows = load_jsonl(os.path.join(DATA_DIR, "train_wcag_ibm.jsonl"))
val_rows = load_jsonl(os.path.join(DATA_DIR, "val_wcag_ibm.jsonl"))
print(f"train: {len(train_rows)} rows, val: {len(val_rows)} rows")
print(f"train positives: {sum(1 for r in train_rows if any(r['labels']))}")
print(f"val positives:   {sum(1 for r in val_rows if any(r['labels']))}")

# Report per-criterion coverage so an all-zero column is visible before training.
import numpy as np
Ytr = np.array([r["labels"] for r in train_rows], dtype=np.float32)
print("per-criterion positives in train:")
for i, c in enumerate(WCAG_22_CRITERIA):
    n = int(Ytr[:, i].sum())
    if n:
        print(f"  {c['criterion']} {c['description']}: {n}")
print(f"criteria with no positive example: {int((Ytr.sum(axis=0) == 0).sum())}/{num_criteria}")'''

MODEL_CELL = '''## Multi-Label Classification Head (encoder frozen)

class LayaWCAGModel(nn.Module):
    def __init__(self, base_model, num_criteria):
        super().__init__()
        self.base = base_model
        # Freeze the 421M-param encoder: we train only the small classifier head.
        # This fits comfortably in a single T4 and avoids OOM from optimizer states.
        for p in self.base.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(base_model.encoder.config.hidden_size, num_criteria)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # no activations stored for the frozen encoder
            h = self.base.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = h[:, 0].float()  # CLS token
        logits = self.classifier(pooled)
        return logits

model = LayaWCAGModel(base_model, num_criteria)
model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model initialized; trainable parameters: {trainable:,}")'''

TRAIN_CFG_CELL = '''## Training Configuration

learning_rate = 5e-4   # head-only: the encoder is frozen, so a larger LR is safe and converges fast
batch_size = 8
num_epochs = 30        # small dataset: more epochs with early stopping on val F1
num_criteria = len(WCAG_22_CRITERIA)

# Class imbalance: most criteria have few or zero positives. pos_weight up-weights the
# positive class in BCEWithLogitsLoss so the head cannot collapse to predicting all-zero.
pos_counts = Ytr.sum(axis=0)
neg_counts = len(Ytr) - pos_counts
pos_weight = torch.tensor(
    [ (neg_counts[i] / pos_counts[i]) if pos_counts[i] > 0 else 1.0 for i in range(num_criteria) ],
    dtype=torch.float32, device=device,
).clamp(max=50.0)
print(f"pos_weight range: {pos_weight.min():.1f} .. {pos_weight.max():.1f}")'''

TOKENIZE_CELL = '''## Tokenize and build the DataLoader

MAX_LEN = 512  # the exported graph takes dynamic length; 512 matches the browser runtime

def encode_row(row):
    enc = tokenizer(row["state"], truncation=True, max_length=MAX_LEN, padding=False)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "labels": [float(v) for v in row["labels"]]}

train_enc = [encode_row(r) for r in train_rows]
val_enc = [encode_row(r) for r in val_rows]

class WcagDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        return self.rows[idx]

def collate(batch):
    L = max(len(r["input_ids"]) for r in batch)
    input_ids = torch.full((len(batch), L), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, r in enumerate(batch):
        n = len(r["input_ids"])
        input_ids[i, :n] = torch.tensor(r["input_ids"])
        attention_mask[i, :n] = 1
    labels = torch.tensor([r["labels"] for r in batch], dtype=torch.float32)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_dataset = WcagDataset(train_enc)
val_dataset = WcagDataset(val_enc)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
print(f"Created {len(train_loader)} train batches, {len(val_loader)} val batches")'''

TRAIN_LOOP_CELL = '''## Training Loop (head only, with val macro-F1 early stopping)

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["labels"].int().numpy())
    import numpy as np
    P = np.vstack(all_preds); L = np.vstack(all_labels)
    return f1_score(L, P, average="macro", zero_division=0), f1_score(L, P, average="micro", zero_division=0)

best_f1, best_state, patience, bad = -1.0, None, 5, 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        loss = criterion(logits, batch["labels"].to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    macro, micro = evaluate(val_loader)
    print(f"epoch {epoch+1}/{num_epochs} loss={total_loss/len(train_loader):.4f} val_macroF1={macro:.4f} val_microF1={micro:.4f}", flush=True)
    if macro > best_f1:
        best_f1, bad = macro, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= patience:
            print(f"early stop at epoch {epoch+1} (best val macro F1 {best_f1:.4f})")
            break

if best_state is not None:
    model.load_state_dict(best_state)
print(f"best val macro F1: {best_f1:.4f}")'''

EXPORT_CELL = '''## ONNX Export (with the Split-node fix)

import torch.onnx
import onnx

dummy_input_ids = torch.ones(1, 64, dtype=torch.long).to(device)
dummy_attention_mask = torch.ones(1, 64, dtype=torch.long).to(device)

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    "laya-wcag.onnx",
    opset_version=17,
    input_names=["input_ids", "attention_mask"],
    output_names=["criterion_scores"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "criterion_scores": {0: "batch_size"},
    },
    external_data=False,
)
print("Export finished.")

# PyTorch 2.x emits an invalid num_outputs attribute on Split nodes (opset 17); ORT rejects it.
# Output arity already encodes the information, so strip the stale attribute.
m = onnx.load("laya-wcag.onnx")
n_split = 0
for node in m.graph.node:
    if node.op_type == "Split":
        keep = [a for a in node.attribute if a.name != "num_outputs"]
        if len(keep) != len(node.attribute):
            del node.attribute[:]
            node.attribute.extend(keep)
            n_split += 1
# stale value_info for initializers also breaks quantize_dynamic's shape inference
vi = m.graph.value_info
keep_vi = [v for v in vi if v.name not in {i.name for i in m.graph.initializer}]
del vi[:]
vi.extend(keep_vi)
onnx.save(m, "laya-wcag-clean.onnx")
print(f"cleaned {n_split} Split nodes; saved laya-wcag-clean.onnx")
onnx.checker.check_model(onnx.load("laya-wcag-clean.onnx"))
print("onnx.checker OK")'''

PACKAGE_CELL = '''## Quantize, externalize, split, and write the site manifest

import onnx, os, hashlib, json as json_lib

# int8 dynamic quantization: 1.58 GB fp32 -> ~395 MB, small quality cost
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("laya-wcag-clean.onnx", "laya-wcag-q8.onnx", weight_type=QuantType.QUInt8)
print("quantized:", os.path.getsize("laya-wcag-q8.onnx") / 1e6, "MB")

# Externalize the weights so the browser can stream them: keep attribute-sized tensors
# (a 1-element Slice starts constant, etc.) INLINE — ORT refuses tiny external tensors.
if os.path.exists("laya-wcag.onnx.data"):
    os.remove("laya-wcag.onnx.data")
qm = onnx.load("laya-wcag-q8.onnx", load_external_data=False)
onnx.save_model(qm, "laya-wcag.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
                location="laya-wcag.onnx.data", size_threshold=1024, convert_attribute=False)
print("graph:", os.path.getsize("laya-wcag.onnx"), "bytes; data:", os.path.getsize("laya-wcag.onnx.data"), "bytes")

# Split the data file into <= 24 MiB parts (GitHub Pages / browser fetch limits)
OUT = "build/laya-wcag"
os.makedirs(OUT, exist_ok=True)
CHUNK = 24 * 1024 * 1024
sha = hashlib.sha256()
parts = []
with open("laya-wcag.onnx.data", "rb") as f:
    i = 0
    while True:
        chunk = f.read(CHUNK)
        if not chunk:
            break
        name = f"laya-wcag.{i:04d}.onnx.data"
        with open(os.path.join(OUT, name), "wb") as p:
            p.write(chunk)
        sha.update(chunk)
        parts.append(name)
        i += 1

import shutil
shutil.copy("laya-wcag.onnx", os.path.join(OUT, "laya-wcag.onnx"))

manifest = {
    "version": 1,
    "source": "turbolego/layaForWeb",
    "chunk_bytes": CHUNK,
    "variants": {
        "wcag": {
            "label": "WCAG 2.2 Violations",
            "onnx": "laya-wcag.onnx",
            # data.name MUST be the ONNX graph's external-data location string, NOT the first
            # part's filename — ORT-web mounts preloaded data under this key.
            "data": {
                "name": "laya-wcag.onnx.data",
                "size": os.path.getsize("laya-wcag.onnx.data"),
                "sha256": sha.hexdigest(),
                "parts": parts,
            },
        }
    },
}
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json_lib.dump(manifest, f, indent=2)

# also ship the criteria list and the tokenizer/config next to the model
with open(os.path.join(OUT, "wcag_criteria.json"), "w") as f:
    json_lib.dump(WCAG_22_CRITERIA, f, indent=1)

print(f"packaged: {len(parts)} parts, {manifest['variants']['wcag']['data']['size']/1e6:.0f} MB, sha256 {sha.hexdigest()[:16]}...")

# sanity: reload the packaged model and run one forward pass.
# The graph's external data file sits one directory up from OUT (/kaggle/working/laya-wcag.onnx.data
# while OUT is /kaggle/working/build/laya-wcag), so copy it next to the graph before loading —
# ORT resolves external data relative to the model file's own directory.
import onnxruntime as ort
import numpy as np
import shutil
shutil.copy("laya-wcag.onnx.data", os.path.join(OUT, "laya-wcag.onnx.data"))
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(os.path.join(OUT, "laya-wcag.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
ids = np.array([[1, 100, 200, 300, 2]], dtype=np.int64)
att = np.ones_like(ids)
out = sess.run(None, {"input_ids": ids, "attention_mask": att})
print("packaged model output shape:", out[0].shape, "sample:", np.round(out[0][0][:5], 3))
assert out[0].shape == (1, num_criteria), "unexpected output shape"
assert np.isfinite(out[0]).all(), "non-finite outputs"
print("PACKAGED MODEL VERIFIED")'''


def make_code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + "\n" for ln in src.rstrip("\n").split("\n")],
    }


def make_md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in src.rstrip("\n").split("\n")],
    }


def main() -> None:
    nb = json.loads(NB.read_text())
    criteria_literal = "".join(nb["cells"][4]["source"])
    # extract the WCAG_22_CRITERIA literal from the existing cell
    import ast
    start = criteria_literal.find("WCAG_22_CRITERIA = ") + len("WCAG_22_CRITERIA = ")
    depth = 0
    end = start
    for k, ch in enumerate(criteria_literal[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = k + 1
                break
    literal = criteria_literal[start:end]
    # validate it parses
    ast.literal_eval(literal)
    criteria = ast.literal_eval(literal)
    print(f"criteria literal: {len(criteria)} entries")

    cells = [
        make_md_cell("# Laya WCAG 2.2 violation detection — retrained on real checker-labeled data\n\n"
                     "Fine-tunes the Laya encoder (frozen) with a Linear(1024, 63) head on HTML snippets\n"
                     "labeled by the IBM Equal Access Accessibility Checker (the tool A11YBench uses),\n"
                     "mapped to the 63 WCAG 2.2 success criteria. Exports a quantized, split ONNX build\n"
                     "that the layaForWeb browser demo loads.\n\n"
                     "Input: `/kaggle/input/<you>/laya-wcag-data/` with `train_wcag_ibm.jsonl` and\n"
                     "`val_wcag_ibm.jsonl` (lines `{state, labels: [0|1 x 63]}`)."),
        make_code_cell("!pip install -q torch transformers datasets onnx onnxruntime onnxscript laya scikit-learn"),
        make_code_cell("import torch\nimport os\ndevice = \"cuda\" if torch.cuda.is_available() else \"cpu\"\nprint(f\"Using device: {device}\")"),
        make_code_cell("## Load base Laya model\n\nimport laya\n\nagent = laya.load(\"convaiinnovations/laya\", device=device)\nbase_model = agent.model\ntokenizer = agent.tok\nprint(f\"Model parameters: {sum(p.numel() for p in base_model.parameters()):,}\")"),
        make_code_cell(CRITERIA_CELL.replace("%CRITERIA%", literal)),
        make_code_cell(DATA_CELL),
        make_code_cell(MODEL_CELL),
        make_code_cell(TRAIN_CFG_CELL),
        make_code_cell(TOKENIZE_CELL),
        make_code_cell(TRAIN_LOOP_CELL),
        make_code_cell(EXPORT_CELL),
        make_code_cell(PACKAGE_CELL),
    ]
    nb["cells"] = cells
    NB.write_text(json.dumps(nb, indent=1))
    print(f"wrote {NB} with {len(cells)} cells")


if __name__ == "__main__":
    main()
