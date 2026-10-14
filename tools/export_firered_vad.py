"""Export FireRedVAD (DFSMN) to ONNX + CMVN to JSON, verify against PyTorch, time ORT.

Usage (SubPlz venv): python tools/export_firered_vad.py [audio16k.npy]
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from fireredvad import FireRedVad, FireRedVadConfig

SRC = "G:/SubPlz/config/models/FireRedVAD/VAD"
OUT = Path("G:/subthx/models"); OUT.mkdir(exist_ok=True)
onnx_path = OUT / "firered_vad.onnx"

vad = FireRedVad.from_pretrained(SRC, FireRedVadConfig(use_gpu=False))
model = vad.vad_model.eval()
a = model.dfsmn
nparams = sum(p.numel() for p in model.parameters())
print(f"DFSMN params: {nparams:,}")


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, feat):  # [N, T, 80] (CMVN applied) -> [N, T]
        probs, _ = self.m(feat)
        return probs.squeeze(-1)


w = Wrap(model).eval()
dummy = torch.randn(1, 500, 80)
torch.onnx.export(w, (dummy,), str(onnx_path), input_names=["feat"], output_names=["probs"],
                  dynamic_axes={"feat": {0: "batch", 1: "time"}, "probs": {0: "batch", 1: "time"}},
                  opset_version=17, dynamo=False)
print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")

# fp16 copy for GPUs (DFSMN activations stay well inside the fp16 range)
import onnx  # noqa: E402
from onnxconverter_common import float16  # noqa: E402
onnx.save(float16.convert_float_to_float16(onnx.load(str(onnx_path)), keep_io_types=True),
          str(OUT / "firered_vad.fp16.onnx"))

cmvn = vad.audio_feat.cmvn
(OUT / "firered_vad_cmvn.json").write_text(json.dumps({
    "means": cmvn.means.tolist(), "inverse_std_variances": cmvn.inverse_std_variances.tolist()}))

# ---- verify + time on a real audiobook
npy = sys.argv[1] if len(sys.argv) > 1 else None
if not npy:
    raise SystemExit
audio = np.load(npy)
dur = len(audio) / 16000
wav = torch.from_numpy(audio).float().unsqueeze(0) * 32768.0
t = time.perf_counter()
feats = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80, frame_length=25.0, frame_shift=10.0,
                                          dither=0.0, sample_frequency=16000.0, snip_edges=True)
print(f"torch CPU fbank: {time.perf_counter() - t:.2f}s for {feats.shape[0]:,} frames")
feats = ((feats - torch.from_numpy(cmvn.means).float()) * torch.from_numpy(cmvn.inverse_std_variances).float())

torch.set_num_threads(os.cpu_count())
t = time.perf_counter()
with torch.no_grad():
    ref = torch.cat([w(c.unsqueeze(0)).squeeze(0) for c in feats.split(30000)]).numpy()
print(f"torch CPU model: {time.perf_counter() - t:.2f}s")

import onnxruntime as ort  # noqa: E402

fnp = feats.numpy()
for prov in (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]):
    sess = ort.InferenceSession(str(onnx_path), providers=prov)
    for chunk in (30000, 200000):
        sess.run(None, {"feat": fnp[None, :chunk]})  # warmup
        t = time.perf_counter()
        out = np.concatenate([sess.run(None, {"feat": fnp[None, s:s + chunk]})[0][0]
                              for s in range(0, len(fnp), chunk)])
        dt = time.perf_counter() - t
        diff = np.abs(out - ref).max() if chunk == 30000 else float("nan")
        print(f"ORT {prov[0]:<22} chunk {chunk:>6}: {dt:.2f}s (RTF x{dur / dt:,.0f}), "
              f"max|onnx-torch| {diff:.2e}, decisions differ {(np.abs((out >= .45) != (ref >= .45))).sum() if chunk == 30000 else '-'}")
