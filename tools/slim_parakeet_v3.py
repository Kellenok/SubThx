"""Make Olicorne's Parakeet v3 ONNX exports faster on GPUs.

decoder: keep only what greedy TDT decoding reads (the LSTM states and the
token/duration argmax, computed in the graph; the export also returns all
logits, top-16 logits and log-sum-exps for beam search) and convert it to
fp16 with fp32 inputs/outputs.

encoder (Redux): convert the 4-bit repack to fp16 activations, so the CUDA
MatMulNBits kernel dequantizes into a cuBLAS GEMM for big batches instead of
its slow fp32 path.

Usage (SubPlz venv):
  python tools/slim_parakeet_v3.py decoder models/parakeet-ultra/decoder_joint-model.onnx models/parakeet-ultra/decoder_joint-model.fp16.onnx
  python tools/slim_parakeet_v3.py encoder models/parakeet-redux/encoder-model.w4.onnx models/parakeet-redux/encoder-model.w4.fp16.onnx
"""
import sys

import onnx
from onnxconverter_common import float16

N_DURATIONS = 5  # TDT durations 0..4

mode, src, dst = sys.argv[1:4]
m = onnx.load(src)
if mode == "decoder":
    # argmax inside the graph: two int64 ids per row instead of 8198 logits
    # (the tail of the joint output is the TDT durations)
    from onnx import TensorProto, helper
    vocab = m.graph.output[0].type.tensor_type.shape.dim[-1].dim_value - N_DURATIONS
    consts = {"am_zero": [0], "am_vocab": [vocab], "am_end": [vocab + N_DURATIONS], "am_axis": [-1]}
    for name, v in consts.items():
        m.graph.initializer.append(helper.make_tensor(name, TensorProto.INT64, [1], v))
    m.graph.node.extend([
        helper.make_node("Slice", ["outputs", "am_zero", "am_vocab", "am_axis"], ["am_tok_logits"]),
        helper.make_node("Slice", ["outputs", "am_vocab", "am_end", "am_axis"], ["am_dur_logits"]),
        helper.make_node("ArgMax", ["am_tok_logits"], ["token"], axis=-1, keepdims=0),
        helper.make_node("ArgMax", ["am_dur_logits"], ["duration"], axis=-1, keepdims=0),
    ])
    KEEP = ["token", "duration", "output_states_1", "output_states_2"]
    m.graph.output.extend([
        helper.make_tensor_value_info("token", TensorProto.INT64, None),
        helper.make_tensor_value_info("duration", TensorProto.INT64, None),
    ])
    keep = [o for o in m.graph.output if o.name in KEEP]
    del m.graph.output[:]
    m.graph.output.extend(keep)
    # drop nodes that no longer feed an output
    needed = set(KEEP)
    nodes = []
    for n in reversed(m.graph.node):
        if any(o in needed for o in n.output):
            nodes.append(n)
            needed.update(i for i in n.input if i)
    del m.graph.node[:]
    m.graph.node.extend(reversed(nodes))
    used = needed
    inits = [t for t in m.graph.initializer if t.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(inits)
# the model's own Cast-to-fp32 nodes will feed fp16 ops: they must cast to
# fp16 (casts the converter inserts around fp32-only ops stay as they are)
own_casts = {n.name for n in m.graph.node if n.op_type == "Cast"}
m = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True)
for n in m.graph.node:
    if n.op_type == "Cast" and n.name in own_casts:
        for a in n.attribute:
            if a.name == "to" and a.i == onnx.TensorProto.FLOAT:
                a.i = onnx.TensorProto.FLOAT16
onnx.save(m, dst)
print(f"{mode}: {len(m.graph.node)} nodes -> {dst}")
