#!/usr/bin/env python3
"""Contrast-only head fix — bounded-sample version.

Fits a logistic-regression probe for criterion 1.4.3 on a bounded sample of the frozen
encoder's [CLS] embeddings (the held-out probe already showed AUC 0.999, so a few hundred
train rows are enough), validates on a held-out slice, and writes the replacement 1.4.3 row.

Caches embeddings to /tmp so re-runs are instant. No encoder fine-tune, no GPU.
"""
import onnx, numpy as np, glob, tempfile, shutil, json, re, os
from tokenizers import Tokenizer
import onnxruntime as ort

MODEL_DIR = "model"
CRIT = json.load(open(os.path.join(MODEL_DIR, "wcag_criteria.json")))
SCS = [c["criterion"] for c in CRIT]
J = SCS.index("1.4.3")
N_TRAIN = int(os.environ.get("N_TRAIN", "400"))
N_VAL = int(os.environ.get("N_VAL", "300"))

# ---- trimmed encoder graph -> CLS embedding ----------------------------------
buf = bytearray()
for p in sorted(glob.glob(os.path.join(MODEL_DIR, "laya-wcag.00*.onnx.data"))):
    buf += open(p, "rb").read()
d = tempfile.mkdtemp()
shutil.copy(os.path.join(MODEL_DIR, "laya-wcag.onnx"), d + "/laya-wcag.onnx")
open(d + "/laya-wcag.onnx.data", "wb").write(bytes(buf))
m = onnx.load(d + "/laya-wcag.onnx")
keep = []
for n in m.graph.node:
    keep.append(n)
    if "select" in n.output:
        break
del m.graph.node[:]; m.graph.node.extend(keep)
del m.graph.output[:]
m.graph.output.append(onnx.helper.make_tensor_value_info("select", onnx.TensorProto.FLOAT, ["batch", 1024]))
ki = [i for i in m.graph.initializer if not i.name.startswith(("classifier.weight", "classifier.bias"))]
del m.graph.initializer[:]; m.graph.initializer.extend(ki)
enc_path = d + "/encoder.onnx"
onnx.save(m, enc_path)
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(enc_path, sess_options=so, providers=["CPUExecutionProvider"])

tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
CLS = tok.token_to_id("[CLS]"); SEP = tok.token_to_id("[SEP]")
def emb(html):
    ids = tok.encode(html, add_special_tokens=False).ids
    seq = [CLS] + ids[:254] + [SEP]
    a = np.array([seq], dtype=np.int64)
    return sess.run(None, {"input_ids": a, "attention_mask": np.ones_like(a)})[0][0]

def has_color(s): return bool(re.search(r"#[0-9a-fA-F]{3,6}", s))
def load(path): return [json.loads(l) for l in open(path) if l.strip()]

def embed_set(rows, cache):
    X = []; y = []
    for i, r in enumerate(rows):
        h = hash(r["state"]) & 0xffffffff
        if h in cache:
            e = cache[h]
        else:
            e = emb(r["state"]); cache[h] = e
        X.append(e); y.append(r["labels"][J])
        if (i+1) % 50 == 0:
            print(f"  embedded {i+1}/{len(rows)}", flush=True)
    return np.array(X), np.array(y)

cache = {}
train = [r for r in load("/home/kloa/axe-labeling/train_final.jsonl") if has_color(r["state"])][:N_TRAIN]
val = [r for r in load("/home/kloa/axe-labeling/val_final.jsonl") if has_color(r["state"])][:N_VAL]
print(f"fitting on {len(train)} train rows, validating on {len(val)} val rows", flush=True)
Xtr, ytr = embed_set(train, cache)
Xva, yva = embed_set(val, cache)
print(f"train pos {int(ytr.sum())}/{len(ytr)}  val pos {int(yva.sum())}/{len(yva)}", flush=True)

mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
def fit(X, y, iters=2000, lr=0.2, l2=1e-3):
    Xs = (X - mu) / sd
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1/(1+np.exp(-(Xs@w+b)))
        w -= lr*(Xs.T@(p-y)/len(y) + l2*w); b -= lr*(p-y).mean()
    return w, b
w, b = fit(Xtr, ytr)

def metrics(X, y):
    Xs = (X - mu)/sd; p = 1/(1+np.exp(-(Xs@w+b)))
    pos = p[y==1]; neg = p[y==0]
    if len(pos)==0 or len(neg)==0: return float("nan"), 0, 0.5, pos, neg
    allp = np.concatenate([pos,neg]); lbl = np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
    order = np.argsort(allp); ranks = np.empty_like(order,float); ranks[order]=np.arange(1,len(allp)+1)
    auc = (ranks[lbl==1].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
    bf=0; bt=0.5
    for t in np.arange(0.1,0.95,0.02):
        tp=int((pos>=t).sum()); fn=int((pos<t).sum()); fp=int((neg>=t).sum())
        pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
        f=2*pr*rc/(pr+rc) if pr+rc else 0
        if f>bf: bf,bt=f,t
    return auc, bf, bt, pos, neg

auc, bf, bt, pos, neg = metrics(Xtr, ytr)
print(f"TRAIN: AUC {auc:.3f} bestF1 {bf:.3f}@{bt:.2f} pos {pos.mean():.3f} neg {neg.mean():.3f}", flush=True)
auc2, bf2, bt2, pos2, neg2 = metrics(Xva, yva)
print(f"VAL:   AUC {auc2:.3f} bestF1 {bf2:.3f}@{bt2:.2f} pos {pos2.mean():.3f} neg {neg2.mean():.3f}", flush=True)

# write replacement row in the head's native [1024] layout (fold standardization in)
w_native = (w / sd).astype(np.float32)
b_native = float(b - (mu * w / sd).sum())
np.save("/tmp/contrast_row_weight.npy", w_native)
json.dump({"bias": b_native, "col": J}, open("/tmp/contrast_row_bias.json", "w"))
print(f"wrote row weight [1024] + bias {b_native:.4f}", flush=True)

for name, h in [("good",'<main><p style="color:#222;background:#fff;font-size:16px">good</p></main>'),
                ("low", '<main><p style="color:#bbb;background:#fff;font-size:12px">low</p></main>')]:
    e = emb(h); pr = 1/(1+np.exp(-float(e @ w_native + b_native)))
    print(f"demo {name} contrast -> {pr:.3f}", flush=True)
print("DONE", flush=True)
