"""Whisper reference data for the Rust port.

- models/<model>/mel_filters.npy: the feature extractor's mel filterbank
  (n_fft/2+1 x n_mels, float32), like nemo*_fbanks.npy for NeMo models
- fixtures/whisper/: 30 s of audio and its log-mel features from
  transformers' WhisperFeatureExtractor, for tests/whisper.rs

usage: python tools/dump_whisper.py models/whisper-ja-anime-v0.3
       python tools/dump_whisper.py models/whisper-large-v3-turbo --filters-only
"""
import json
import os
import sys

import numpy as np
from transformers import WhisperFeatureExtractor

model = sys.argv[1]
fe = WhisperFeatureExtractor.from_pretrained(model)
np.save(os.path.join(model, "mel_filters.npy"), fe.mel_filters.astype(np.float32))
print("mel filters", fe.mel_filters.shape)
if "--filters-only" in sys.argv:
    sys.exit()

# 30 s of the audiobook fixture, from 10 minutes in, plus a short clip
# (padding path)
audio = np.load("fixtures/tantei16k.npy")[16000 * 600 : 16000 * 630].astype(np.float32)
short = audio[: 16000 * 7]
os.makedirs("fixtures/whisper", exist_ok=True)
for name, a in [("long", audio), ("short", short)]:
    f = fe(a, sampling_rate=16000, return_tensors="np").input_features[0].astype(np.float32)
    np.save(f"fixtures/whisper/{name}_audio.npy", a)
    np.save(f"fixtures/whisper/{name}_features.npy", f)
    print(name, f.shape, float(f.min()), float(f.max()))
json.dump({"model": model}, open("fixtures/whisper/meta.json", "w"))
