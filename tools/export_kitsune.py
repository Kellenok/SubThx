"""Export Kitsune (HF ParakeetForCTC) to ONNX FP16 with argmax inside the graph.

Usage (SubPlz venv): python tools/export_kitsune.py [p0.05b p0.1b p0.3b]
Writes models/kitsune-<size>/{model.fp16.onnx, tokenizer.json, config.json}
Inputs:  features [B, 80, T] float32 (NeMo log-mel, normalised), lengths [B] int64
Outputs: ids [B, T/8] int64 (frame-wise argmax), out_lengths [B] int64
"""
import shutil, sys, time
from pathlib import Path

import torch
from transformers import AutoProcessor, ParakeetForCTC

args = [a for a in sys.argv[1:] if not a.startswith("--")]
sizes = args or ["p0.05b", "p0.1b", "p0.3b"]
FP32 = "--fp32" in sys.argv
# --mixed: fp16 everywhere except encoder self-attention (fp32), the model
# author's fp16 recipe; plain fp16 overflows in the attention scores
MIXED = "--mixed" in sys.argv
# --mixed2: fp16 everywhere incl. attention projections; only the attention
# scores (q.k, positional term, mask, softmax) are computed in fp32
MIXED2 = "--mixed2" in sys.argv
DT = torch.float32 if FP32 else torch.float16
NAME = ("model.fp32.onnx" if FP32 else "model.mixed16.onnx" if MIXED
        else "model.mixed16b.onnx" if MIXED2 else "model.fp16.onnx")


def fp32_scores_forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
    b, t = hidden_states.shape[:2]
    h, d = self.config.num_attention_heads, self.head_dim
    shape = (b, t, -1, d)
    q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
    k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
    v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    rel_k = self.relative_k_proj(position_embeddings).view(b, -1, h, d)
    q32 = q.float()
    q_u = q32 + self.bias_u.float().view(1, h, 1, d)
    q_v = q32 + self.bias_v.float().view(1, h, 1, d)
    bd = q_v @ rel_k.float().permute(0, 2, 3, 1)
    bd = self._rel_shift(bd)[..., :t]
    scores = (q_u @ k.float().transpose(2, 3) + bd) * self.scaling
    if attention_mask is not None:
        scores = scores.masked_fill(attention_mask.logical_not(), -10000.0)
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    out = (probs @ v).transpose(1, 2).reshape(b, t, -1)
    return self.o_proj(out), None


def fp32_scores(model):
    import types
    from transformers.models.parakeet.modeling_parakeet import ParakeetEncoderAttention
    n = 0
    for mod in model.modules():
        if isinstance(mod, ParakeetEncoderAttention):
            mod.forward = types.MethodType(fp32_scores_forward, mod)
            n += 1
    return n


def to_dtype(x, dt):
    if torch.is_tensor(x) and x.is_floating_point():
        return x.to(dt)
    if isinstance(x, tuple):
        return tuple(to_dtype(v, dt) for v in x)
    return x


def fp32_attention(model):
    from transformers.models.parakeet.modeling_parakeet import ParakeetEncoderAttention
    n = 0
    for mod in model.modules():
        if isinstance(mod, ParakeetEncoderAttention):
            mod.float()
            fwd = mod.forward

            def wrapped(*args, _fwd=fwd, **kw):
                args = to_dtype(args, torch.float32)
                kw = {k: to_dtype(v, torch.float32) for k, v in kw.items()}
                return to_dtype(_fwd(*args, **kw), torch.float16)

            mod.forward = wrapped
            n += 1
    return n


# HF masks attention with -inf (NeMo uses -10000). Fully padded query rows then
# become NaN after softmax; eager PyTorch hides that, the exported graph (and
# fp16) does not. Export with NeMo's finite value instead.
_masked_fill_ = torch.Tensor.masked_fill_


def _finite_masked_fill_(self, mask, value):
    if isinstance(value, float) and value == float("-inf"):
        value = -10000.0
    return _masked_fill_(self, mask, value)


torch.Tensor.masked_fill_ = _finite_masked_fill_


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, features, lengths):
        x = features.transpose(1, 2).to(DT)  # [B, T, 80]
        t = x.shape[1]
        mask = torch.arange(t, device=x.device)[None, :] < lengths[:, None]
        logits = self.m(input_features=x, attention_mask=mask).logits
        ids = logits.argmax(-1)
        # FastConformer 8x subsampling: three stride-2 convs, kernel 3, padding 1
        out_len = lengths
        for _ in range(3):
            out_len = torch.div(out_len + 2 * 1 - 3, 2, rounding_mode="floor") + 1
        return ids, out_len


for size in sizes:
    mid = f"Multy123/kitsune-transcribe-{size}-ja"
    out = Path(f"G:/subthx/models/kitsune-{size}")
    out.mkdir(parents=True, exist_ok=True)
    model = ParakeetForCTC.from_pretrained(mid, dtype=DT, attn_implementation="eager").cuda().eval()
    if MIXED:
        print(f"{size}: {fp32_attention(model)} attention modules kept in fp32")
    if MIXED2:
        print(f"{size}: {fp32_scores(model)} attention modules with fp32 scores")
    w = Wrap(model).eval()
    feats = torch.randn(2, 80, 801, device="cuda")
    lens = torch.tensor([801, 600], device="cuda")
    t = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(w, (feats, lens), str(out / NAME),
                          input_names=["features", "lengths"], output_names=["ids", "out_lengths"],
                          dynamic_axes={"features": {0: "B", 2: "T"}, "lengths": {0: "B"},
                                        "ids": {0: "B", 1: "T8"}, "out_lengths": {0: "B"}},
                          opset_version=17, dynamo=False)
    snap = Path(AutoProcessor.from_pretrained(mid).tokenizer.name_or_path)
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(mid, allow_patterns=["tokenizer.json", "config.json", "tokenizer_config.json"]))
    for f in ("tokenizer.json", "config.json", "tokenizer_config.json"):
        shutil.copy(src / f, out / f)
    size_mb = sum(p.stat().st_size for p in out.glob(NAME + "*")) / 1e6
    print(f"{size}: exported in {time.perf_counter() - t:.1f}s, {size_mb:.0f} MB")
    del model, w
    torch.cuda.empty_cache()
