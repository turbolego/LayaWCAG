#!/usr/bin/env python3
"""Patch the deployed WCAG ONNX head with the contrast fine-tune's non-linear MLP.

The fine-tuned contrast head is skip(linear) + MLP(1024->128->1). The deployed ONNX head is a
plain linear MatMul(1024->63) and cannot express the MLP, so we bake the MLP into the graph:

    select[.,1024] -> MatMul(mlp0.w) -> +mlp0.b -> GELU -> MatMul(mlp3.w) -> +mlp3.b
    -> 1.4.3 MLP logit (scalar)
    -> multiply by a 0/1 mask vector that selects ONLY column 14 (1.4.3)
    -> add to criterion_scores

Net effect: criterion_scores[:,14] += mlp_logit ; all other columns unchanged.
The encoder and the other 62 criteria are untouched, so this cannot regress them.

Reads:
  model/laya-wcag.onnx (+ 00NN.onnx.data parts)  -- deployed v6
  /tmp/kaggle_out_contrast/contrast_ft/mlp.pt    -- fine-tuned MLP
  /tmp/kaggle_out_contrast/contrast_ft/row_bias.json  -- {bias, col}
Writes:
  build/contrast_patch/laya-wcag-mlp.onnx  -- patched fp32 graph (external data)
Then quantize_dynamic + split + manifest, mirroring the earlier packaging.
"""
import onnx, numpy as np, glob, tempfile, shutil, json, os, torch
from onnx import helper, TensorProto, numpy_helper

MODEL_DIR = "model"
FT = "/tmp/kaggle_out_contrast/contrast_ft"
J = json.load(open(os.path.join(FT, "row_bias.json")))["col"]  # 14 (1.4.3)
print("contrast column index:", J, flush=True)

# Load a CLEAN v6 base graph + data. BASE_ONNX/BASE_DATA_DIR override the default model/ paths
# (needed because model/ now holds the already-patched graph).
BASE_ONNX = os.environ.get("BASE_ONNX", os.path.join(MODEL_DIR, "laya-wcag.onnx"))
BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", MODEL_DIR)
buf = bytearray()
for p in sorted(glob.glob(os.path.join(BASE_DATA_DIR, "laya-wcag.00*.onnx.data"))): buf += open(p, "rb").read()
d = tempfile.mkdtemp()
shutil.copy(BASE_ONNX, d + "/laya-wcag.onnx")
open(d + "/laya-wcag.onnx.data", "wb").write(bytes(buf))
m = onnx.load(d + "/laya-wcag.onnx")
assert not any("mlp" in (n.name or "") for n in m.graph.node), "base graph already has MLP nodes"

sd = torch.load(os.path.join(FT, "mlp.pt"), map_location="cpu")
W0 = sd["0.weight"].detach().numpy().astype(np.float32)  # [128,1024]
B0 = sd["0.bias"].detach().numpy().astype(np.float32)    # [128]
W3 = sd["3.weight"].detach().numpy().astype(np.float32)  # [1,128]
B3 = sd["3.bias"].detach().numpy().astype(np.float32)    # [1]

inits = list(m.graph.initializer)
new_inits = [
    numpy_helper.from_array(W0.T.copy(), "mlp_w0_T"),  # [1024,128]
    numpy_helper.from_array(B0, "mlp_b0"),
    numpy_helper.from_array(W3.T.copy(), "mlp_w3_T"),  # [128,1]
    numpy_helper.from_array(B3, "mlp_b3"),
]
# 0/1 mask that selects only column J of the 63-d score vector
mask = np.zeros((1, 63), dtype=np.float32); mask[0, J] = 1.0
new_inits.append(numpy_helper.from_array(mask, "contrast_col_mask"))

nodes = [
    helper.make_node("MatMul", ["select", "mlp_w0_T"], ["mlp_h"], name="mlp_matmul0"),
    helper.make_node("Add", ["mlp_h", "mlp_b0"], ["mlp_hb"], name="mlp_add0"),
    helper.make_node("Gelu", ["mlp_hb"], ["mlp_g"], name="mlp_gelu", domain="com.microsoft"),
    helper.make_node("MatMul", ["mlp_g", "mlp_w3_T"], ["mlp_out"], name="mlp_matmul3"),
    helper.make_node("Add", ["mlp_out", "mlp_b3"], ["mlp_logit"], name="mlp_add3"),  # [batch,1]
    # broadcast the single MLP logit into a 63-d vector, masked to column J only
    helper.make_node("Mul", ["mlp_logit", "contrast_col_mask"], ["mlp_contrib"], name="mlp_mask"),
    # add the MLP correction back into the SAME output name the app reads: criterion_scores.
    # The old head Add produced 'criterion_scores'; rename it to a base tensor and write the
    # final sum into 'criterion_scores' so the graph output name the app reads is preserved.
    helper.make_node("Add", ["criterion_scores_base", "mlp_contrib"], ["criterion_scores"], name="mlp_add_scores"),
]
m.graph.node.extend(nodes)
m.graph.initializer.extend(new_inits)

# rename the existing head output 'criterion_scores' -> 'criterion_scores_base' (its producer
# and consumers), and keep the graph output named 'criterion_scores' (produced by mlp_add_scores).
old_producer = None
for n in m.graph.node:
    if "criterion_scores" in n.output and n.name != "mlp_add_scores":
        old_producer = n; break
assert old_producer is not None, "could not find the head Add producing criterion_scores"
old_producer.output[list(old_producer.output).index("criterion_scores")] = "criterion_scores_base"
# graph output stays named 'criterion_scores' (already declared) — no change needed.

# Gelu with com.microsoft domain requires opset import; add if missing
domains = {op.domain for op in m.opset_import}
if "com.microsoft" not in domains:
    m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
# fall back to a manual GELU if ORT-web lacks the MS domain: use Erf-based gelu built from core ops
# (keep MS Gelu; ORT-web supports com.microsoft Gelu.)

os.makedirs("build/contrast_patch", exist_ok=True)
# save with external data
# Force IR version to 9 (ORT-web 1.30 supports up to IR 9; onnx.save bumps it to 10 by default)
m.ir_version = 9
onnx.save(m, "build/contrast_patch/laya-wcag-mlp.onnx", save_as_external_data=True,
          all_tensors_to_one_file=True, location="laya-wcag-mlp.onnx.data",
          size_threshold=1024, convert_attribute=False)
print("saved build/contrast_patch/laya-wcag-mlp.onnx", flush=True)

# sanity: run it, confirm column 14 moves and others don't
import onnxruntime as ort
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession("build/contrast_patch/laya-wcag-mlp.onnx", sess_options=so, providers=["CPUExecutionProvider"])
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
CLS = tok.token_to_id("[CLS]"); SEP = tok.token_to_id("[SEP]")
def run(h):
    ids = tok.encode(h, add_special_tokens=False).ids; seq=[CLS]+ids[:254]+[SEP]
    a=np.array([seq],dtype=np.int64)
    return sess.run(None,{"input_ids":a,"attention_mask":np.ones_like(a)})[0][0]
# compare against v6
v6 = ort.InferenceSession(d+"/laya-wcag.onnx", sess_options=so, providers=["CPUExecutionProvider"])
def run6(h):
    ids = tok.encode(h, add_special_tokens=False).ids; seq=[CLS]+ids[:254]+[SEP]
    a=np.array([seq],dtype=np.int64)
    return v6.run(None,{"input_ids":a,"attention_mask":np.ones_like(a)})[0][0]
for h in ['<main><p style="color:#222;background:#fff;font-size:16px">good</p></main>',
          '<main><p style="color:#bbb;background:#fff;font-size:12px">low</p></main>']:
    a=run(h); b=run6(h)
    diff=np.abs(a-b)
    print(f"max diff {diff.max():.4f} at col {int(diff.argmax())} | col14 v6 {b[J]:.3f} -> patched {a[J]:.3f} | p v6 {1/(1+np.exp(-b[J])):.3f} -> {1/(1+np.exp(-a[J])):.3f}", flush=True)
print("DONE", flush=True)
