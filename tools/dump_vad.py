"""Golden FireRedVAD outputs (SubPlz config/chunking, PyTorch CPU) for the Rust port.

Usage (SubPlz venv): python tools/dump_vad.py <audio16k.npy> [out_dir]
Writes: fbank_head.npy (first 60000 frames, CMVN applied), probs.npy, segments.json
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from fireredvad import FireRedVad, FireRedVadConfig

audio = np.load(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else "fixtures/vad"); out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(os.cpu_count())

cfg = FireRedVadConfig(use_gpu=False, speech_threshold=0.45, max_speech_frame=800,
                       min_silence_frame=30, merge_silence_frame=20, min_speech_frame=15)
vad = FireRedVad.from_pretrained("G:/SubPlz/config/models/FireRedVAD/VAD", cfg)
cmvn = vad.audio_feat.cmvn
means = torch.from_numpy(cmvn.means).float()
istd = torch.from_numpy(cmvn.inverse_std_variances).float()

# SubPlz.run_vad (cuda branch) chunking, on CPU
N_FRAMES, overlap = 60000, 240
step = N_FRAMES * 160
n = len(audio)
probs, head = [], None
t = time.perf_counter()
with torch.no_grad():
    for s in range(0, n - overlap, step):
        e = min(s + step + overlap, n)
        if e - s < 400:
            break
        w = torch.from_numpy(audio[s:e]).float().unsqueeze(0) * 32768.0
        f = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, frame_length=25.0, frame_shift=10.0,
                                              dither=0.0, sample_frequency=16000.0, snip_edges=True)
        f = (f - means) * istd
        if head is None:
            head = f.numpy()
        for mc in f.split(30000, dim=0):
            cp, _ = vad.vad_model.forward(mc.unsqueeze(0))
            probs.append(cp.squeeze(0).squeeze(-1))
probs = torch.cat(probs).numpy()
print(f"features+model: {time.perf_counter() - t:.1f}s, {len(probs):,} frames")

t = time.perf_counter()
dec = vad.vad_postprocessor.process(probs.tolist())
ts = vad.vad_postprocessor.decision_to_segment(dec, n / 16000.0)
print(f"postprocess: {time.perf_counter() - t:.2f}s, {len(ts)} segments")

np.save(out / "fbank_head.npy", head.astype(np.float32))
np.save(out / "probs.npy", probs.astype(np.float32))
(out / "segments.json").write_text(json.dumps({
    "n_samples": n,
    "decisions_sum": int(sum(dec)),
    "seconds": ts,
    "samples": [[int(s * 16000), int(e * 16000)] for s, e in ts],
}))
