"""Whisper graphs for subthx from an optimum ONNX export (fp16 weights,
float32 I/O): two sessions instead of the export's three, half the cache
memory, attention that does not hold its score matrix.

encoder_kv_fp16.onnx
    in:  input_features [B, n_mels, 3000] float32
    out: cross.{l}.key / cross.{l}.value [B, H, 1500, 64] float16 per
         decoder layer: the decoder's cross-attention keys and values,
         projected once here (the export recomputes them in the first
         decoder step and hands them around as float32)
    The optimum export spells every self-attention out (Reshape, Transpose,
    scaled MatMul, Softmax, MatMul), so a 30 s window holds an H x 1500 x
    1500 score matrix per layer. ONNX Runtime's transformer optimizer fuses
    the layer norms and GELUs but does not recognize this attention (q and k
    are scaled separately); here each one becomes
    com.microsoft.MultiHeadAttention, which the CUDA provider runs as
    memory-efficient / flash attention.

decoder_fp16.onnx
    in:  input_ids [B, T] int64, past.{l}.key / past.{l}.value
         [B, H, P, 64] float16 (P = 0 for the first step), cross.*
    out: logits [B, 1, vocab] float32 (the last position only),
         present.{l}.key / present.{l}.value [B, H, P + T, 64] float16,
         alignment_heads [B, A, T, 1500] float32: the cross-attention of
         generation_config's alignment heads, in that order
    The export's decoder_with_past with the new-token count unfolded (it was
    traced with T = 1), so one graph runs the prompt, every greedy step, and
    the teacher-forced pass openai-whisper's `find_alignment` (CTranslate2's
    `align`) times the words with.

usage: python tools/whisper_onnx.py models/whisper-ja-anime-v0.3
"""
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx import numpy_helper as nh
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions

model = sys.argv[1]
cfg = json.load(open(os.path.join(model, "config.json")))
align_heads = json.load(open(os.path.join(model, "generation_config.json")))["alignment_heads"]
onnx_dir = os.path.join(model, "onnx")
d = cfg["d_model"]
dec_layers, dec_heads = cfg["decoder_layers"], cfg["decoder_attention_heads"]
head_dim = d // dec_heads
FP16 = TensorProto.FLOAT16


def init(g, name):
    return next(i for i in g.initializer if i.name == name)


def bypass_cast(g, cast, keep):
    """Drop a Cast node: consumers of its output read `keep` instead."""
    for n in g.node:
        for k, i in enumerate(n.input):
            if i == cast.output[0]:
                n.input[k] = keep
    g.node.remove(cast)


# ---------------------------------------------------------------- encoder
opt = optimizer.optimize_model(
    os.path.join(onnx_dir, "encoder_model_fp16.onnx"),
    model_type="bart",
    num_heads=cfg["encoder_attention_heads"],
    hidden_size=d,
    optimization_options=FusionOptions("bart"),
    opt_level=0,
    use_gpu=True,
)
g = opt.model.graph
for layer in range(cfg["encoder_layers"]):
    p = f"/layers.{layer}/self_attn/"
    names = {n.name: n for n in g.node}
    q, k, v = p + "q_proj/Add_output_0", p + "k_proj/MatMul_output_0", p + "v_proj/Add_output_0"
    out = p + "Reshape_3_output_0"
    # q and k each scaled by d^-1/4, or only q by d^-1/2 (newer exports)
    drop = [p + x for x in ["Reshape", "Reshape_1", "Reshape_2", "Transpose", "Transpose_1", "Transpose_2", "Mul", "Mul_1", "MatMul", "Softmax", "MatMul_1", "Transpose_3", "Reshape_3"]]
    drop = [n for n in drop if n in names]
    if not all(p + x in names for x in ["Mul", "MatMul", "Softmax", "MatMul_1", "Reshape_3"]):
        raise SystemExit(f"encoder layer {layer}: unexpected attention graph")
    at = list(g.node).index(names[drop[0]])
    for n in drop:
        g.node.remove(names[n])
    g.node.insert(
        at,
        helper.make_node(
            "MultiHeadAttention", [q, k, v], [out], name=p + "MultiHeadAttention", domain="com.microsoft",
            num_heads=cfg["encoder_attention_heads"], scale=(d // cfg["encoder_attention_heads"]) ** -0.5,
        ),
    )

# the final hidden state stays float16 and feeds the cross projections,
# whose weights come from the export's first-step decoder
last = next(n for n in g.node if n.output[0] == "last_hidden_state")
assert last.op_type == "Cast"
hidden = last.input[0]
g.node.remove(last)
del g.output[:]
first = onnx.load(os.path.join(onnx_dir, "decoder_model_fp16.onnx"))
fn = {n.name: n for n in first.graph.node}
g.initializer.append(nh.from_array(np.array([0, -1, dec_heads, head_dim], np.int64), "cross_shape"))
for layer in range(dec_layers):
    p = f"/model/decoder/layers.{layer}/encoder_attn/"
    for kind, proj in [("key", "k_proj"), ("value", "v_proj")]:
        w = fn[p + proj + "/MatMul"].input[1]
        g.initializer.append(nh.from_array(nh.to_array(init(first.graph, w)), f"cross.{layer}.{proj}.weight"))
        x = f"cross.{layer}.{kind}"
        g.node.append(helper.make_node("MatMul", [hidden, f"cross.{layer}.{proj}.weight"], [x + "/mm"], name=x + "/MatMul"))
        y = x + "/mm"
        if p + proj + "/Add" in fn:
            add = fn[p + proj + "/Add"]
            b = next(i for i in add.input if i != fn[p + proj + "/MatMul"].output[0])
            g.initializer.append(nh.from_array(nh.to_array(init(first.graph, b)), f"cross.{layer}.{proj}.bias"))
            g.node.append(helper.make_node("Add", [y, f"cross.{layer}.{proj}.bias"], [x + "/add"], name=x + "/Add"))
            y = x + "/add"
        g.node.append(helper.make_node("Reshape", [y, "cross_shape"], [x + "/heads"], name=x + "/Reshape"))
        g.node.append(helper.make_node("Transpose", [x + "/heads"], [x], perm=[0, 2, 1, 3], name=x + "/Transpose"))
        g.output.append(helper.make_tensor_value_info(x, FP16, ["batch_size", dec_heads, 1500, head_dim]))
del first, fn
opt.prune_graph()
if not any(o.domain == "com.microsoft" for o in opt.model.opset_import):
    opt.model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
dst = os.path.join(onnx_dir, "encoder_kv_fp16.onnx")
opt.save_model_to_file(dst)
print(dst, os.path.getsize(dst) // 1_000_000, "MB")

# ---------------------------------------------------------------- decoder
m = onnx.load(os.path.join(onnx_dir, "decoder_with_past_model_fp16.onnx"))
g = m.graph
del g.value_info[:]

# the new-token count, folded to 1 when traced: from input_ids instead
seq = "/model/decoder/Gather_1_output_0"
assert nh.to_array(init(g, seq)).tolist() == 1
g.initializer.remove(init(g, seq))
g.initializer.append(helper.make_tensor("seq_axis", TensorProto.INT64, [], [1]))
g.initializer.append(helper.make_tensor("zero_axis", TensorProto.INT64, [1], [0]))
fix = [
    helper.make_node("Shape", ["input_ids"], ["ids_shape"], name="/seq/Shape"),
    helper.make_node("Gather", ["ids_shape", "seq_axis"], [seq], axis=0, name="/seq/Gather"),
    helper.make_node("Unsqueeze", [seq, "zero_axis"], ["seq_1d"], name="/seq/Unsqueeze"),
]
for k, n in enumerate(fix):
    g.node.insert(k, n)
# the causal mask had one row
next(n for n in g.node if n.name == "/model/decoder/Concat_3").input[0] = "seq_1d"
# reshapes to [-1, 1, ...] (batch, one token, ...): keep the batch, any length
reshapes = {n.input[1] for n in g.node if n.op_type == "Reshape"}
for i in g.initializer:
    if i.name in reshapes:
        a = nh.to_array(i)
        if a.ndim == 1 and len(a) >= 3 and a[0] == -1 and a[1] == 1:
            b = a.copy()
            b[0], b[1] = 0, -1
            i.CopyFrom(nh.from_array(b, i.name))

# caches in and out as float16, renamed
rename = {}
for layer in range(dec_layers):
    for kv in ["key", "value"]:
        rename[f"past_key_values.{layer}.decoder.{kv}"] = f"past.{layer}.{kv}"
        rename[f"past_key_values.{layer}.encoder.{kv}"] = f"cross.{layer}.{kv}"
        rename[f"present.{layer}.decoder.{kv}"] = f"present.{layer}.{kv}"
for i in g.input:
    if i.name in rename:
        cast = next(n for n in g.node if n.op_type == "Cast" and n.input[0] == i.name)
        bypass_cast(g, cast, rename[i.name])
        i.name = rename[i.name]
        i.type.tensor_type.elem_type = FP16
for o in g.output:
    if o.name in rename:
        cast = next(n for n in g.node if n.output[0] == o.name)
        assert cast.op_type == "Cast"
        src = cast.input[0]
        g.node.remove(cast)
        # the float16 value under its output name
        for n in g.node:
            for k, x in enumerate(n.output):
                if x == src:
                    n.output[k] = rename[o.name]
            for k, x in enumerate(n.input):
                if x == src:
                    n.input[k] = rename[o.name]
        o.name = rename[o.name]
        o.type.tensor_type.elem_type = FP16

# logits of the last position only (the prompt and the alignment pass
# need none of the others)
lg = next(n for n in g.node if n.output[0] == "logits")
lg.output[0] = "logits_all"
g.initializer.append(helper.make_tensor("last_start", TensorProto.INT64, [1], [-1]))
g.initializer.append(helper.make_tensor("last_end", TensorProto.INT64, [1], [2**62]))
g.initializer.append(helper.make_tensor("last_axis", TensorProto.INT64, [1], [1]))
g.node.append(helper.make_node("Slice", ["logits_all", "last_start", "last_end", "last_axis"], ["logits"], name="/logits/Slice"))
next(o for o in g.output if o.name == "logits").type.tensor_type.shape.dim[1].dim_value = 1

# the alignment heads' cross-attention
assert [l for l, _ in align_heads] == sorted(l for l, _ in align_heads)
by_layer = {}
for layer, head in align_heads:
    by_layer.setdefault(layer, []).append(head)
parts = []
for layer, hs in by_layer.items():
    soft = f"/model/decoder/layers.{layer}/encoder_attn/Softmax_output_0"
    assert any(soft in n.output for n in g.node), soft
    g.initializer.append(helper.make_tensor(f"align_heads_{layer}", TensorProto.INT64, [len(hs)], hs))
    g.node.append(helper.make_node("Gather", [soft, f"align_heads_{layer}"], [f"align_{layer}"], axis=1, name=f"/align/Gather_{layer}"))
    parts.append(f"align_{layer}")
g.node.append(helper.make_node("Concat", parts, ["align_concat"], axis=1, name="/align/Concat"))
g.node.append(helper.make_node("Cast", ["align_concat"], ["alignment_heads"], to=TensorProto.FLOAT, name="/align/Cast"))
g.output.append(helper.make_tensor_value_info("alignment_heads", TensorProto.FLOAT, ["batch_size", len(align_heads), "decoder_sequence_length", 1500]))
for x in list(g.input) + list(g.output):
    dims = x.type.tensor_type.shape.dim
    if x.name == "input_ids":
        dims[1].dim_param = "decoder_sequence_length"
    if x.name.startswith("present."):
        dims[2].dim_param = "total_sequence_length"

onnx.checker.check_model(m)
dst = os.path.join(onnx_dir, "decoder_fp16.onnx")
onnx.save(m, dst)
print(dst, len(g.node), "nodes,", os.path.getsize(dst) // 1_000_000, "MB")
