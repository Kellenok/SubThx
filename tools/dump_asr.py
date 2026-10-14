"""Golden Parakeet transcripts: SubPlz ParakeetModel.run_batched_asr on the fixture VAD chunks.

Usage (SubPlz venv): python tools/dump_asr.py fixtures/tantei16k.npy fixtures/vad/segments.json fixtures/asr/tantei.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "G:/SubPlz")
from subplz.parakeet_model import ParakeetModel

audio = np.load(sys.argv[1])
chunks = [{"start": s, "end": e} for s, e in json.load(open(sys.argv[2]))["samples"]]
m = ParakeetModel(device="cuda")
m.run_batched_asr(audio, chunks[:64], batch_size=64)  # warmup
t = time.perf_counter()
texts = m.run_batched_asr(audio, chunks, batch_size=64)
dt = time.perf_counter() - t
print(f"ASR: {dt:.2f}s for {len(chunks)} chunks ({len(audio) / 16000 / dt:,.0f}x realtime)")
out = Path(sys.argv[3]); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"seconds": dt, "texts": texts}, ensure_ascii=False), encoding="utf-8")
