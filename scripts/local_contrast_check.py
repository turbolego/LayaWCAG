#!/usr/bin/env python3
"""Local CPU validation of the contrast-only fine-tune, mirroring the Kaggle notebook exactly.

Caches encoder [CLS] embeddings once (slow, CPU), then trains the skip+MLP contrast head and
reports held-out val AUC / best-F1. If val AUC does not clearly beat the linear probe (~0.777),
the fine-tune is not worth shipping to Kaggle.
"""
import onnx, numpy as np, glob, tempfile, shutil, json, re, os, sys
from tokenizers import Tokenizer
import onnxruntime as ort
import torch, torch.nn as nn

MODEL_DIR = "model"
LABEL_DIR = "/home/kloa/axe-labeling"
CRIT = json.load(open(os.path.join(MODEL_DIR, "wcag_criteria.json")))
SCS = [c["criterion"] for c in CRIT]; J = SCS.index("1.4.3")

# --- trimmed encoder -> CLS embedding ---
buf = bytearray()
for p in sorted(glob.glob(os.path.join(MODEL_DIR, "laya-wcag.00*.onnx.data"))): buf += open(p, "rb").read()
d = tempfile.mkdtemp()
shutil.copy(os.path.join(MODEL_DIR, "laya-wcag.onnx"), d + "/laya-wcag.onnx")
open(d + "/laya-wcag.onnx.data", "wb").write(bytes(buf))
m = onnx.load(d + "/laya-wcag.onnx")
keep = []
for n in m.graph.node:
    keep.append(n)
    if "select" in n.output: break
del m.graph.node[:]; m.graph.node.extend(keep)
del m.graph.output[:]
m.graph.output.append(onnx.helper.make_tensor_value_info("select", onnx.TensorProto.FLOAT, ["batch", 1024]))
ki = [i for i in m.graph.initializer if not i.name.startswith(("classifier.weight", "classifier.bias"))]
del m.graph.initializer[:]; m.graph.initializer.extend(ki)
enc_path = d + "/encoder.onnx"; onnx.save(m, enc_path)
sess = ort.InferenceSession(enc_path, sess_options=ort.SessionOptions(), providers=["CPUExecutionProvider"])
tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
CLS = tok.token_to_id("[CLS]"); SEP = tok.token_to_id("[SEP]")
def embed(html, maxlen=256):
    ids = tok.encode(html, add_special_tokens=False).ids
    seq = [CLS] + ids[:maxlen-2] + [SEP]
    a = np.array([seq], dtype=np.int64)
    return sess.run(None, {"input_ids": a, "attention_mask": np.ones_like(a)})[0][0]

CACHE = os.path.join(LABEL_DIR, "contrast_emb_cache.json")
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def emb_cached(html):
    h = str(hash(html) & 0xffffffff)
    if h in cache: return np.array(cache[h], dtype=np.float32)
    e = embed(html); cache[h] = e.tolist(); return e.astype(np.float32)

def has_color(s): return bool(re.search(r"#[0-9a-fA-F]{3,6}", s))
def load(p): return [json.loads(l) for l in open(p) if l.strip()]

def build(rows):
    X = []; y = []
    for r in rows:
        if not has_color(r["state"]): continue
        X.append(emb_cached(r["state"])); y.append(r["labels"][J])
    return np.asarray(X, np.float32), np.asarray(y, np.float32)

train = [r for r in load(os.path.join(LABEL_DIR, "train_final.jsonl")) if has_color(r["state"])][:600]
val = [r for r in load(os.path.join(LABEL_DIR, "val_final.jsonl")) if has_color(r["state"])][:400]
print(f"embedding {len(train)} train + {len(val)} val (cached where possible)...", flush=True)
Xtr, ytr = build(train); Xva, yva = build(val)
json.dump(cache, open(CACHE, "w"))
print(f"train {Xtr.shape} pos {int(ytr.sum())} | val {Xva.shape} pos {int(yva.sum())}", flush=True)

W6 = np.load(os.path.join(LABEL_DIR, "v6_head_weight.npy")); b6 = np.load(os.path.join(LABEL_DIR, "v6_head_bias.npy"))
w6_row = torch.tensor(W6[:, J], dtype=torch.float32); b6_row = torch.tensor(float(b6[J]), dtype=torch.float32)

class ContrastHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip_w = nn.Parameter(w6_row.clone()); self.skip_b = nn.Parameter(b6_row.clone())
        self.mlp = nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(self, x):
        return (x * self.skip_w).sum(-1) + self.skip_b + self.mlp(x).squeeze(-1)

rng = np.random.default_rng(0)
model = ContrastHead()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
pos_weight = torch.tensor([max(1.0, (len(ytr)-ytr.sum())/max(1, ytr.sum()))])
lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
Xt = torch.tensor(Xtr); yt = torch.tensor(ytr); Xv = torch.tensor(Xva)

def evaluate():
    model.eval()
    with torch.no_grad(): p = torch.sigmoid(model(Xv)).numpy()
    pos = p[yva==1]; neg = p[yva==0]
    if len(pos)==0 or len(neg)==0: return 0.5, 0.0
    allp = np.concatenate([pos,neg]); lbl = np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
    order = np.argsort(allp); ranks = np.empty_like(order,float); ranks[order]=np.arange(1,len(allp)+1)
    auc = (ranks[lbl==1].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
    bf=0.0
    for t in np.arange(0.1,0.95,0.02):
        tp=int((pos>=t).sum()); fn=int((pos<t).sum()); fp=int((neg>=t).sum())
        pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
        f=2*pr*rc/(pr+rc) if pr+rc else 0
        if f>bf: bf=f
    return auc, bf

best_auc, best_state, bad = -1, None, 0
BS=64
for epoch in range(60):
    model.train(); perm = rng.permutation(len(Xt)); tot=0.0
    for i in range(0,len(perm),BS):
        idx=perm[i:i+BS]; opt.zero_grad()
        loss=lossf(model(Xt[idx]), yt[idx]); loss.backward(); opt.step(); tot+=loss.item()*len(idx)
    auc,bf=evaluate()
    if epoch%5==0 or auc>best_auc: print(f"epoch {epoch+1} loss {tot/len(perm):.4f} val_auc {auc:.4f} val_bestF1 {bf:.4f}", flush=True)
    if auc>best_auc: best_auc, bad = auc, 0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    else:
        bad+=1
        if bad>=6: print("early stop"); break
if best_state: model.load_state_dict(best_state)
print(f"BEST val AUC {best_auc:.4f}  (linear probe was ~0.777)", flush=True)
# demo with the FULL head (skip+mlp), which is what would ship
for name,h in [("good",'<main><p style="color:#222;background:#fff;font-size:16px">good</p></main>'),
               ("low", '<main><p style="color:#bbb;background:#fff;font-size:12px">low</p></main>')]:
    with torch.no_grad(): p=torch.sigmoid(model(torch.tensor(embed(h)))).item()
    print(f"demo {name} contrast -> {p:.3f}", flush=True)
print("DONE", flush=True)
