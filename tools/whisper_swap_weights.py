"""Optimum-style ONNX graphs of a Whisper model built from a template
model's graphs and the target's weights. The architecture must match
(layers, width, heads); the vocabulary may differ.

The decoder graphs that tools/whisper_onnx.py knows how to rework come
from older optimum exports (whisper-ja-anime-v0.3's); newer exports
(onnx-community/whisper-large-v3-turbo) build the decoder differently. So
the graphs come from the template, and every initializer gets the
target's tensor of the same layer (MatMul weights are stored transposed
under generated names; the layer comes from the node's name), in the
graph's dtype. Every large initializer must be replaced, so a renamed
layer fails loudly instead of keeping the template's weights.

weights: a .safetensors file (a fine-tune: TheStageAI/thewhisper-large-v3-turbo)
         or a model folder with an optimum export in onnx/ (its tensors are read
         back from its own graphs)
The configs and tokenizer in <out dir> must be the target's.

usage: python tools/whisper_swap_weights.py <template dir> <weights> <out dir>
  e.g. python tools/whisper_swap_weights.py models/whisper-ja-anime-v0.3 models/whisper-large-v3-turbo models/whisper-large-v3-turbo
  then tools/whisper_onnx.py on <out dir>
"""
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper as nh

FILES = [("encoder_model_fp16.onnx", "encoder"), ("decoder_model_fp16.onnx", "decoder"), ("decoder_with_past_model_fp16.onnx", "decoder")]


def layer_of(node_name, part):
    """'/model/decoder/layers.0/fc1/MatMul' or '/layers.0/fc1/MatMul' ->
    'model.decoder.layers.0.fc1'"""
    p = node_name.strip("/").rsplit("/", 1)[0].replace("/", ".")
    if p.startswith("model.") or p == "proj_out":
        return p
    return f"model.{part}.{p}"


def params(g, part):
    """(initializer, parameter name, stored transposed) for a graph"""
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, []).append(n)
    for init in g.initializer:
        key, transposed = None, False
        if init.name.startswith("model.") or init.name == "proj_out.weight":
            key = init.name
        elif "/" not in init.name and not init.name.startswith("onnx::"):
            key = f"model.{part}.{init.name}"  # "layers.0.fc1.bias", "conv1.weight"
        else:
            for n in users.get(init.name, []):
                if n.op_type == "MatMul" and n.input[1] == init.name:
                    key, transposed = layer_of(n.name, part) + ".weight", True
                    break
                if n.op_type == "Add" and len(init.dims) == 1:
                    key = layer_of(n.name, part) + ".bias"
                    break
        yield init, key, transposed


template, src, out = sys.argv[1:4]
if src.endswith(".safetensors"):
    from safetensors.numpy import load_file

    w = load_file(src)
else:
    w = {}
    for name, part in FILES:
        g = onnx.load(os.path.join(src, "onnx", name)).graph
        for init, key, transposed in params(g, part):
            if key and key not in w and len(init.dims) >= 1:
                a = nh.to_array(init)
                w[key] = a.T if transposed else a
        del g
if "proj_out.weight" not in w:
    w["proj_out.weight"] = w["model.decoder.embed_tokens.weight"]

os.makedirs(os.path.join(out, "onnx"), exist_ok=True)
for name, part in FILES:
    m = onnx.load(os.path.join(template, "onnx", name))
    done, left = 0, []
    for init, key, transposed in list(params(m.graph, part)):
        a = nh.to_array(init)
        if key is None or key not in w:
            if a.size > 1000:
                left.append((init.name, key, a.shape))
            continue
        new = w[key].T if transposed else w[key]
        if a.ndim == new.ndim and a.shape[1:] != new.shape[1:] or a.ndim != new.ndim:
            raise SystemExit(f"{name}: {init.name} <- {key}: shape {new.shape} vs {a.shape}")
        # the vocabulary (embed_tokens, proj_out) may differ in size
        if new.shape != a.shape and not ("embed_tokens" in key or key == "proj_out.weight"):
            raise SystemExit(f"{name}: {init.name} <- {key}: shape {new.shape} vs {a.shape}")
        init.CopyFrom(nh.from_array(np.ascontiguousarray(new).astype(a.dtype), init.name))
        done += 1
    if left:
        raise SystemExit(f"{name}: no weight for {left[:8]} ({len(left)} in all)")
    dst = os.path.join(out, "onnx", name)
    onnx.save(m, dst)
    print(dst, done, "tensors from the target")
