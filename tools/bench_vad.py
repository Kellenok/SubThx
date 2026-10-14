"""Benchmark the two SubPlz VADs (FireRedVAD, batched Silero) on one full audiobook.

Usage (SubPlz venv):  python tools/bench_vad.py <audio> [cache.npy]
"""
import os, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "G:/SubPlz")
from subplz.moonshine_model import numba_post_process  # noqa: E402

audio_path = sys.argv[1]
cache = Path(sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else Path(os.environ["TEMP"]) / "vad_bench_audio.npy")

if cache.exists():
    audio = np.load(cache)
else:
    t = time.perf_counter()
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", audio_path, "-map", "0:a:0", "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    np.save(cache, audio)
    print(f"decoded in {time.perf_counter() - t:.1f}s")
dur = len(audio) / 16000
print(f"audio: {dur / 3600:.2f} h, {len(audio):,} samples")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def report(name, t_model, t_post, segs, extra=""):
    lens = np.array([(e - s) / 16000 for s, e in segs]) if segs else np.zeros(1)
    total = t_model + t_post
    print(f"{name:<34} model {t_model:6.2f}s  post {t_post:5.2f}s  total {total:6.2f}s  "
          f"RTF x{dur / total:,.0f} | {len(segs)} segs, speech {lens.sum() / dur * 100:.1f}%, "
          f"mean {lens.mean():.2f}s, max {lens.max():.2f}s {extra}")


results = {}

# ---------------- Silero (torch JIT, batched streams) -- SubPlz path
def silero_jit(device, B=128):
    m, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True, verbose=False)
    m = m.to(device).eval()
    chunk = 512
    n = len(audio)
    tw = (n + chunk - 1) // chunk
    wps = (tw + B - 1) // B
    sync(); t = time.perf_counter()
    padded = torch.zeros(wps * B * chunk, device=device)
    padded[:n] = torch.from_numpy(audio).to(device)
    streams = padded.view(B, wps, chunk)
    m.reset_states()
    probs = torch.zeros(wps, B, device=device)
    with torch.no_grad():
        for i in range(wps):
            probs[i] = m(streams[:, i, :], 16000).squeeze(-1)
    p = probs.T.reshape(-1)[:tw].cpu().numpy()
    sync(); t_model = time.perf_counter() - t
    t = time.perf_counter()
    s, e = numba_post_process(p, total_samples=n)
    t_post = time.perf_counter() - t
    return p, list(zip(s.tolist(), e.tolist())), t_model, t_post


numba_post_process(np.zeros(1000, np.float32), total_samples=512000)  # JIT warmup

if torch.cuda.is_available():
    silero_jit("cuda", 128)  # warmup
    p_jit, segs, tm, tp = silero_jit("cuda", 128)
    report("silero JIT CUDA B=128 (SubPlz)", tm, tp, segs)
    results["silero_jit_cuda"] = segs

# ---------------- Silero ONNX, batched streams (what Rust/ort would do)
import onnxruntime as ort  # noqa: E402

onnx_path = Path.home() / ".cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.onnx"


def silero_onnx(providers, B, threads=None):
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), so, providers=providers)
    chunk, ctx = 512, 64
    n = len(audio)
    tw = (n + chunk - 1) // chunk
    wps = (tw + B - 1) // B
    t = time.perf_counter()
    padded = np.zeros(wps * B * chunk, np.float32)
    padded[:n] = audio
    streams = padded.reshape(B, wps, chunk)
    state = np.zeros((2, B, 128), np.float32)
    context = np.zeros((B, ctx), np.float32)
    sr = np.array(16000, dtype=np.int64)
    probs = np.zeros((wps, B), np.float32)
    for i in range(wps):
        x = np.concatenate([context, streams[:, i, :]], axis=1)
        out, state = sess.run(None, {"input": x, "state": state, "sr": sr})
        probs[i] = out[:, 0]
        context = x[:, -ctx:]
    p = probs.T.reshape(-1)[:tw]
    t_model = time.perf_counter() - t
    t = time.perf_counter()
    s, e = numba_post_process(p, total_samples=n)
    t_post = time.perf_counter() - t
    return p, list(zip(s.tolist(), e.tolist())), t_model, t_post


for B in (128, 512):
    p_onnx, segs, tm, tp = silero_onnx(["CPUExecutionProvider"], B)
    extra = f"| max|p-jit| {np.abs(p_onnx - p_jit).max():.4f}" if "silero_jit_cuda" in results else ""
    report(f"silero ONNX CPU B={B}", tm, tp, segs, extra)
    results[f"silero_onnx_cpu_{B}"] = segs
if "CUDAExecutionProvider" in ort.get_available_providers():
    silero_onnx(["CUDAExecutionProvider"], 128)
    p_onnx, segs, tm, tp = silero_onnx(["CUDAExecutionProvider"], 512)
    report("silero ONNX CUDA B=512", tm, tp, segs)

# ---------------- FireRedVAD -- SubPlz GPU path (torchaudio kaldi fbank + DFSMN)
import torchaudio  # noqa: E402
from fireredvad import FireRedVad, FireRedVadConfig  # noqa: E402


def firered(device):
    cfg = FireRedVadConfig(use_gpu=device == "cuda", speech_threshold=0.45, max_speech_frame=800,
                           min_silence_frame=30, merge_silence_frame=20, min_speech_frame=15)
    vad = FireRedVad.from_pretrained("G:/SubPlz/config/models/FireRedVAD/VAD", cfg)
    cmvn = vad.audio_feat.cmvn
    means = torch.from_numpy(cmvn.means).float().to(device)
    istd = torch.from_numpy(cmvn.inverse_std_variances).float().to(device)
    N_FRAMES, overlap = 60000, 240
    step = N_FRAMES * 160
    n = len(audio)
    sync(); t = time.perf_counter()
    chunk_probs = []
    t_feat = 0.0
    with torch.no_grad():
        for s in range(0, n - overlap, step):
            e = min(s + step + overlap, n)
            if e - s < 400:
                break
            tf = time.perf_counter()
            w = torch.from_numpy(audio[s:e]).to(device).float().unsqueeze(0) * 32768.0
            f = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, frame_length=25.0, frame_shift=10.0,
                                                  dither=0.0, sample_frequency=16000.0, snip_edges=True)
            f = (f - means) * istd
            sync(); t_feat += time.perf_counter() - tf
            for mc in f.split(30000, dim=0):
                cp, _ = vad.vad_model.forward(mc.unsqueeze(0))
                chunk_probs.append(cp.squeeze(0).cpu())
    probs = torch.cat(chunk_probs, dim=0).squeeze().numpy()
    sync(); t_model = time.perf_counter() - t
    t = time.perf_counter()
    dec = vad.vad_postprocessor.process(probs.tolist())
    ts = vad.vad_postprocessor.decision_to_segment(dec, n / 16000)
    t_post = time.perf_counter() - t
    return [(int(a * 16000), int(b * 16000)) for a, b in ts], t_model, t_post, t_feat


if torch.cuda.is_available():
    firered("cuda")
    segs, tm, tp, tf = firered("cuda")
    report("FireRedVAD CUDA (SubPlz)", tm, tp, segs, f"| fbank {tf:.2f}s")
    results["firered_cuda"] = segs
if "--firered-cpu" in sys.argv:
    torch.set_num_threads(os.cpu_count())
    segs, tm, tp, tf = firered("cpu")
    report("FireRedVAD CPU", tm, tp, segs, f"| fbank {tf:.2f}s")
