"""Golden Kitsune transcripts (PyTorch BF16, run_full_book_final.py features) + ONNX export check.

Usage (SubPlz venv): python tools/dump_kitsune.py <size> fixtures/tantei16k.npy fixtures/vad/segments.json
Writes fixtures/asr/tantei.kitsune-<size>.json
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, ParakeetForCTC

size, npy, segs = sys.argv[1:4]
mid = f"Multy123/kitsune-transcribe-{size}-ja"
audio = np.load(npy)
chunks = json.load(open(segs))["samples"]
slices = [audio[s:e] for s, e in chunks]
order = sorted(range(len(slices)), key=lambda i: len(slices[i]))
batches = [order[i:i + 64] for i in range(0, len(order), 64)]

proc = AutoProcessor.from_pretrained(mid)
fe = proc.feature_extractor


def extract_mel_gpu(batch_wav_list, device="cuda"):  # from run_full_book_final.py
    lens = torch.tensor([len(w) for w in batch_wav_list], device=device)
    tensors = [torch.from_numpy(w).to(device) for w in batch_wav_list]
    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    timemask = torch.arange(padded.shape[1], device=device).unsqueeze(0) < lens.unsqueeze(1)
    padded = torch.cat([padded[:, :1], padded[:, 1:] - fe.preemphasis * padded[:, :-1]], dim=1)
    padded = padded.masked_fill(~timemask, 0.0)
    mel = fe._torch_extract_fbank_features(padded, device=device)
    fl = torch.floor_divide(lens + fe.n_fft // 2 * 2 - fe.n_fft, fe.hop_length)
    am = torch.arange(mel.shape[1], device=device)[None, :] < fl[:, None]
    mask = am.unsqueeze(-1)
    mm = mel * mask
    mean = (mm.sum(dim=1) / fl.unsqueeze(-1)).unsqueeze(1)
    var = (((mm - mean) ** 2 * mask).sum(dim=1) / (fl - 1).unsqueeze(-1)).unsqueeze(1)
    return ((mel - mean) / (torch.sqrt(var) + 1e-5)) * mask, am, fl


model = ParakeetForCTC.from_pretrained(mid, dtype=torch.bfloat16).to("cuda").eval()
f, am, _ = extract_mel_gpu(slices[:4])
with torch.inference_mode():
    model(input_features=f.to(torch.bfloat16), attention_mask=am)
torch.cuda.synchronize()
texts = [""] * len(slices)
t = time.perf_counter()
for bi, b in enumerate(batches):
    if bi % 25 == 0:
        print(f"  batch {bi}: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved, {time.perf_counter() - t:.1f}s", flush=True)
    f, am, _ = extract_mel_gpu([slices[i] for i in b])
    with torch.inference_mode():
        ids = model(input_features=f.to(torch.bfloat16), attention_mask=am).logits.argmax(-1)
    del f, am
    torch.cuda.empty_cache()  # growing batch shapes otherwise fragment the cache allocator (VRAM "leak")
    for i, txt in zip(b, proc.batch_decode(ids, skip_special_tokens=True)):
        texts[i] = txt
torch.cuda.synchronize()
dt = time.perf_counter() - t
print(f"torch bf16 {size}: {dt:.2f}s ({len(audio) / 16000 / dt:,.0f}x)")

out = Path(f"fixtures/asr/tantei.kitsune-{size}.json")
out.write_text(json.dumps({"seconds": dt, "texts": texts}, ensure_ascii=False), encoding="utf-8")
print(f"peak torch VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
torch.cuda.empty_cache()

# ONNX export check on a few batches (python ORT CUDA); "same as requested"
# arena so varying batch shapes do not keep growing VRAM
import onnxruntime as ort  # noqa: E402
sess = ort.InferenceSession(f"G:/subthx/models/kitsune-{size}/model.fp16.onnx",
                            providers=[("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
                                       "CPUExecutionProvider"])
same = total = 0
for b in batches[::25]:
    f, am, fl = extract_mel_gpu([slices[i] for i in b])
    ids, olen = sess.run(None, {"features": f.transpose(1, 2).float().cpu().numpy(), "lengths": fl.cpu().numpy()})
    with torch.inference_mode():
        lg = model(input_features=f.to(torch.bfloat16), attention_mask=am).logits
    assert ids.shape[1] == lg.shape[1], (ids.shape, lg.shape)
    for k, i in enumerate(b):
        o = proc.decode(torch.from_numpy(ids[k, :olen[k]]), skip_special_tokens=True)
        same += o == texts[i]
        total += 1
print(f"onnx fp16 vs torch bf16 texts identical: {same}/{total}")
