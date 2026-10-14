"""Repack Parakeet Redux's 2-bit MatMulNBits weights as 4-bit, so the CUDA
provider (which has no 2-bit kernels) can run the encoder.

The weights are ternary per 128-block (-1, 0, +1 times a scale), so 2- and
4-bit packings hold exactly the same values; only the storage changes.

Usage: python tools/repack_redux.py models/parakeet-redux/encoder-model.w2a8.onnx \
           models/parakeet-redux/encoder-model.w4.onnx
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
inits = {t.name: t for t in m.graph.initializer}
done = 0
for node in m.graph.node:
    if node.op_type != "MatMulNBits":
        continue
    attrs = {a.name: a for a in node.attribute}
    if onnx.helper.get_attribute_value(attrs["bits"]) != 2:
        continue
    if len(node.input) > 3 and node.input[3]:
        raise SystemExit(f"{node.name}: explicit zero points are not handled")
    t = inits[node.input[1]]
    q2 = numpy_helper.to_array(t)  # [N, blocks, block_size/4], 4 values per byte, low bits first
    vals = np.stack([(q2 >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(q2.shape[0], q2.shape[1], -1)
    w = vals.astype(np.int16) - 2  # default 2-bit zero point
    assert w.min() >= -2 and w.max() <= 1
    q4 = (w + 8).astype(np.uint8)  # default 4-bit zero point
    packed = (q4[..., 0::2] | (q4[..., 1::2] << 4)).astype(np.uint8)  # low nibble first
    new = numpy_helper.from_array(packed, t.name.replace("_Q2", "_Q4"))
    m.graph.initializer.remove(t)
    m.graph.initializer.append(new)
    node.input[1] = new.name
    attrs["bits"].i = 4
    done += 1

onnx.save(m, dst)
print(f"repacked {done} MatMulNBits nodes -> {dst}")
