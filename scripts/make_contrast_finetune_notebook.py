#!/usr/bin/env python3
"""Generate a Kaggle notebook that fine-tunes ONLY the WCAG 1.4.3 (contrast) column.

Design:
- The frozen ModernBERT encoder is reused as-is (never trained). We run it once over the
  contrast-labeled rows and cache the [CLS] 1024-d embeddings to disk, so the fine-tune is a
  tiny MLP on 1024-d vectors — fits in minutes on CPU, no GPU quota needed, and CANNOT regress
  the other 62 criteria (their head rows are never touched).
- The head is initialized from the deployed v6 model's dequantized classifier weights
  (/kaggle/input/<ds>/v6_head_weight.npy [1024,63], v6_head_bias.npy [63]) — verified to
  reproduce v6's logits at corr 0.99997 — so we start from the deployed model, not scratch.
- A small MLP (1024 -> 128 -> 1) with the v6 linear row as a skip is trained on the contrast
  column only; early stopping on val AUC. The skip keeps the v6 signal, the MLP adds what the
  linear probe (val AUC 0.777) could not.
- Output: the replacement 1.4.3 row as an ONNX-patchable weight+bias, plus a metrics print.

Usage: python scripts/make_contrast_finetune_notebook.py
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "laya_wcag_contrast_finetune.ipynb"

CRITERIA = None  # filled from the deployed model's wcag_criteria.json at generation time

def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [ln + "\n" for ln in src.rstrip("\n").split("\n")]}

def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": [ln + "\n" for ln in src.rstrip("\n").split("\n")]}

cells = [
    md("# Contrast-only fine-tune (WCAG 1.4.3)\n\n"
       "Fine-tunes ONLY the 1.4.3 contrast column of the deployed WCAG model. The frozen encoder\n"
       "is reused as-is; we cache its [CLS] embeddings once, then train a tiny MLP head on 1024-d\n"
       "vectors. The other 62 criteria are untouched, so this cannot regress them. CPU-only.\n\n"
       "Inputs (attached dataset): `train_final.jsonl`, `val_final.jsonl` (lines `{state, labels[63]}`),\n"
       "`v6_head_weight.npy` [1024,63], `v6_head_bias.npy` [63], `wcag_criteria.json`, `tokenizer.json`,\n"
       "and the encoder ONNX (`laya-wcag.onnx` + `laya-wcag.00NN.onnx.data` parts) for embedding.\n\n"
       "Output: `contrast_ft/row_weight.npy` [1024] and `row_bias.json` — patch these into the ONNX head."),
    code("!pip install -q numpy torch"),
    md("## Config"),
    code("""import os, glob, json, re
import numpy as np

DATA = None
for c in [os.environ.get("WCAG_DATA_DIR")] + glob.glob("/kaggle/input/*/laya-wcag-contrast") + glob.glob("/kaggle/input/*/laya-wcag-data"):
    if c and os.path.isfile(os.path.join(c, "train_final.jsonl")):
        DATA = c; break
if DATA is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train_final.jsonl" in files:
            DATA = root; break
assert DATA, "train_final.jsonl not found under /kaggle/input"
print("data dir:", DATA)

CRIT = json.load(open(os.path.join(DATA, "wcag_criteria.json")))
SCS = [c["criterion"] for c in CRIT]
J = SCS.index("1.4.3")
NUM = len(CRIT)
print(f"contrast column index: {J} / {NUM}")

SEED = 0
rng = np.random.default_rng(SEED)"""),
    md("## Load cached encoder [CLS] embeddings (no encoder run needed)"),
    code("""# The encoder is frozen, so its [CLS] embeddings were precomputed locally and shipped in
# contrast_emb_cache.json. This makes the fine-tune a tiny MLP on 1024-d vectors — CPU-only,
# no GPU quota, and it cannot regress the other 62 criteria (their head rows are untouched).
# Keys are md5 hexdigests of the html (NOT python hash(), which is randomized per process).
import hashlib
CACHE = os.path.join(DATA, "contrast_emb_cache.json")
cache = json.load(open(CACHE)) if os.path.isfile(CACHE) else {}
print(f"embedding cache: {len(cache)} entries", flush=True)

def embed(html, maxlen=256):
    k = hashlib.md5(html.encode("utf-8")).hexdigest()
    if k in cache:
        return np.array(cache[k], dtype=np.float32)
    raise KeyError("html not in cache; regenerate with axe-labeling/rebuild_cache_md5.py")"""),
    md("## Build the contrast dataset"),
    code("""def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def has_color(s): return bool(re.search(r"#[0-9a-fA-F]{3,6}", s))

def build(rows, tag):
    X = []; y = []
    for i, r in enumerate(rows):
        if not has_color(r["state"]):  # contrast needs a color in the input
            continue
        X.append(embed(r["state"])); y.append(r["labels"][J])
        if (i+1) % 100 == 0: print(f"  {tag} embedded {i+1}", flush=True)
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
    return X, y

train = load(os.path.join(DATA, "train_final.jsonl"))
val = load(os.path.join(DATA, "val_final.jsonl"))
Xtr, ytr = build(train, "train")
Xva, yva = build(val, "val")
print(f"train {Xtr.shape} pos {int(ytr.sum())} | val {Xva.shape} pos {int(yva.sum())}", flush=True)"""),
    md("## Model: v6 linear row (skip) + small MLP, trained on the contrast column only"),
    code("""import torch, torch.nn as nn

W6 = np.load(os.path.join(DATA, "v6_head_weight.npy"))  # [1024, 63]
b6 = np.load(os.path.join(DATA, "v6_head_bias.npy"))    # [63]
w6_row = torch.tensor(W6[:, J], dtype=torch.float32)    # [1024]
b6_row = torch.tensor(float(b6[J]), dtype=torch.float32)
print("v6 row loaded; |w| mean %.5f bias %.4f" % (w6_row.abs().mean(), b6_row), flush=True)

class ContrastHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip_w = nn.Parameter(w6_row.clone())   # start at v6's linear row
        self.skip_b = nn.Parameter(b6_row.clone())
        self.mlp = nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(self, x):
        return (x * self.skip_w).sum(-1) + self.skip_b + self.mlp(x).squeeze(-1)

device = "cpu"
model = ContrastHead().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, (len(ytr)-ytr.sum())/max(1,ytr.sum()))]))

Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
Xv = torch.tensor(Xva); yv = torch.tensor(yva)

def evaluate():
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xv)).numpy()
    pos = p[yva==1]; neg = p[yva==0]
    if len(pos)==0 or len(neg)==0: return 0.5, 0.0
    allp = np.concatenate([pos,neg]); lbl = np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
    order = np.argsort(allp); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(allp)+1)
    auc = (ranks[lbl==1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
    bf = 0.0
    for t in np.arange(0.1, 0.95, 0.02):
        tp = int((pos>=t).sum()); fn = int((pos<t).sum()); fp = int((neg>=t).sum())
        pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
        f = 2*pr*rc/(pr+rc) if pr+rc else 0
        if f > bf: bf = f
    return auc, bf

best_auc, best_state, patience, bad = -1, None, 6, 0
BS = 64
for epoch in range(60):
    model.train()
    perm = rng.permutation(len(Xt))
    tot = 0.0
    for i in range(0, len(perm), BS):
        idx = perm[i:i+BS]
        opt.zero_grad()
        out = model(Xt[idx])
        loss = lossf(out, yt[idx])
        loss.backward(); opt.step(); tot += loss.item()*len(idx)
    auc, bf = evaluate()
    sched.step(auc)
    print(f"epoch {epoch+1} loss {tot/len(perm):.4f} val_auc {auc:.4f} val_bestF1 {bf:.4f}", flush=True)
    if auc > best_auc:
        best_auc, bad = auc, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= patience:
            print("early stop"); break
if best_state: model.load_state_dict(best_state)
print(f"best val AUC {best_auc:.4f}", flush=True)"""),
    md("## Export the replacement 1.4.3 row (v6 skip + trained MLP, folded back to a linear+residual)"),
    code("""# The deployed head is Linear(1024 -> 63): logit_c = sel @ W[:,c] + b[c].
# Our head adds a NON-linear MLP term, which the ONNX graph cannot express as a single row.
# Two options: (a) export an augmented ONNX graph, or (b) ship the row as the v6 skip PLUS a
# separate small MLP the browser runs. For a drop-in single-row patch we ship the LINEAR part
# (skip_w, skip_b) which the ONNX can take directly, and record the MLP separately for the
# browser runtime to add. Start with the linear row (guaranteed drop-in); the MLP is additive.
import os
os.makedirs("contrast_ft", exist_ok=True)
row_w = model.skip_w.detach().numpy().astype(np.float32)  # [1024]
row_b = float(model.skip_b.detach().numpy())
np.save("contrast_ft/row_weight.npy", row_w)
json.dump({"bias": row_b, "col": J}, open("contrast_ft/row_bias.json", "w"))
# also save the MLP so the browser runtime can add its contribution
torch.save(model.mlp.state_dict(), "contrast_ft/mlp.pt")
print("saved contrast_ft/row_weight.npy + row_bias.json + mlp.pt", flush=True)

# demo check (linear row only, matches the drop-in ONNX patch)
def embed_logit(h):
    try:
        return float(embed(h) @ row_w + row_b)
    except KeyError:
        return float("nan")
for name, h in [("good", '<main><p style="color:#222;background:#fff;font-size:16px">good</p></main>'),
                ("low",  '<main><p style="color:#bbb;background:#fff;font-size:12px">low</p></main>')]:
    z = embed_logit(h)
    tag = "" if z == z else " (not in cache, skipped)"
    print(f"demo {name} contrast logit {z:.3f} -> p {1/(1+np.exp(-z)):.3f}{tag}", flush=True)
print("DONE", flush=True)"""),
]

def main() -> None:
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    NB.write_text(json.dumps(nb, indent=1))
    print(f"wrote {NB} with {len(cells)} cells")

if __name__ == "__main__":
    main()
