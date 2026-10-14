"""FP32 Kitsune ONNX -> mixed FP16 (nodes whose activations overflow FP16 stay FP32).

Usage (SubPlz venv): python tools/mixed_fp16.py models/kitsune-p0.05b
Writes <dir>/model.mixed16.onnx; checks frame-id agreement against FP32 on real audio.
"""
import json, sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16

d = Path(sys.argv[1])
src = d / "model.fp32.onnx"
audio = np.load("fixtures/tantei16k.npy")
segs = json.load(open("fixtures/vad/segments.json"))["samples"]
pre = ort.InferenceSession("models/nemo80.onnx", providers=["CPUExecutionProvider"])


def batch(idx):
    w = [audio[s:e] for s, e in (segs[i] for i in idx)]
    L = max(map(len, w))
    x = np.zeros((len(w), L), np.float32)
    for k, a in enumerate(w):
        x[k, :len(a)] = a
    f, fl = pre.run(None, {"waveforms": x, "waveforms_lens": np.array([len(a) for a in w], np.int64)})
    return {"features": f, "lengths": fl}


calib = [batch(range(i, i + 16)) for i in (100, 3000, 6000)]
check = [batch(range(i, i + 32)) for i in (500, 2500, 4500, 7000)]

# 1. find activations whose magnitude does not fit FP16
m = onnx.load(str(src))
float_outs = []
vi = {v.name: v for v in list(m.graph.value_info) + list(m.graph.output)}
inferred = onnx.shape_inference.infer_shapes(m)
types = {v.name: v.type.tensor_type.elem_type for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
for n in m.graph.node:
    for o in n.output:
        if types.get(o) == onnx.TensorProto.FLOAT and o not in [x.name for x in m.graph.output]:
            m.graph.output.append(onnx.helper.make_value_info(o, onnx.TypeProto()))
            float_outs.append((n.name, o))
probe = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
names = [o for _, o in float_outs]
peak = dict.fromkeys(names, 0.0)
for feed in calib:
    for name, val in zip(names, probe.run(names, feed)):
        if val.size:
            peak[name] = max(peak[name], float(np.nanmax(np.abs(val))))
limit = 3e4  # head-room below fp16 max (65504)
block = sorted({n for n, o in float_outs if peak[o] > limit})
print(f"{len(float_outs)} float activations, {len(block)} nodes exceed {limit:g}: {block[:12]}{' ...' if len(block) > 12 else ''}")
del probe

# 2. convert, keeping those nodes (and graph IO) in fp32
model = onnx.load(str(src))
mixed = float16.convert_float_to_float16(model, keep_io_types=True, node_block_list=block,
                                         op_block_list=float16.DEFAULT_OP_BLOCK_LIST + ["Cast"])
out = d / "model.mixed16.onnx"
onnx.save(mixed, str(out))

# 3. agreement on held-out batches
ref = ort.InferenceSession(str(src), providers=["CUDAExecutionProvider"])
new = ort.InferenceSession(str(out), providers=["CUDAExecutionProvider"])
agree = total = 0
for feed in check:
    a, la = ref.run(None, feed)
    b, _ = new.run(None, feed)
    for k in range(len(la)):
        agree += int((a[k, :la[k]] == b[k, :la[k]]).sum())
        total += int(la[k])
print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB), frame-id agreement with fp32: {agree / total * 100:.2f}%")
