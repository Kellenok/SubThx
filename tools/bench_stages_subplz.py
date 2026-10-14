"""Stage timings of `subplz sync` (Whisper, stable-ts + faster-whisper) without
changing SubPlz: the library entry points are wrapped with timers, then the
real CLI runs in this process. Run with SubPlz's venv:

    G:/SubPlz/.venv/Scripts/python.exe tools/bench_stages_subplz.py AUDIO EPUB OUT_DIR CACHE_DIR [--model tiny]

Prints one line `STAGES {...}` at the end (seconds):
  load   WhisperModel construction (weights from the local HF cache)
  decode sum of the ffmpeg chunk decodes (the decode thread overlaps the ASR)
  vad    sum of faster-whisper's get_speech_timestamps (Silero VAD)
  asr    transcribe() time that is not VAD
  total  the whole `subplz sync` process (includes the book alignment)
"""
import json
import os
import runpy
import subprocess
import sys
import threading
import time

SUBPLZ = os.environ.get("SUBPLZ_ROOT", "G:/SubPlz")  # e.g. an export of the upstream tag v4.0.0
audio, epub, out_dir, cache_dir = sys.argv[1:5]
model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "tiny"
sys.path.insert(0, SUBPLZ)
os.chdir(SUBPLZ)  # SubPlz writes config/ relative to the working directory

import faster_whisper.transcribe as ftr  # noqa: E402

T = {"load": 0.0, "decode": 0.0, "vad": 0.0, "transcribe": 0.0, "match": 0.0}
lock = threading.Lock()


def add(key, dt):
    with lock:
        T[key] += dt


# model load
orig_init = ftr.WhisperModel.__init__


def timed_init(self, *a, **k):
    t = time.perf_counter()
    orig_init(self, *a, **k)
    add("load", time.perf_counter() - t)


ftr.WhisperModel.__init__ = timed_init

# VAD (faster-whisper calls this module-level function inside transcribe)
orig_vad = ftr.get_speech_timestamps


def timed_vad(*a, **k):
    t = time.perf_counter()
    r = orig_vad(*a, **k)
    add("vad", time.perf_counter() - t)
    return r


ftr.get_speech_timestamps = timed_vad

# the whole fast pipeline (decode thread + VAD + ASR); asr = pipeline - vad.
# stable-ts calls transcribe through its own path, so time the entry point instead.
import subplz.models as spm  # noqa: E402

orig_pipe = getattr(spm, "transcribe_whisper_audio_subs_fast", None)


def timed_pipe(self, stream, **k):
    t = time.perf_counter()
    r = orig_pipe(self, stream, **k)
    add("transcribe", time.perf_counter() - t)
    return r


if orig_pipe is not None:  # the batched 4-chunk pipeline (the modified SubPlz)
    spm.transcribe_whisper_audio_subs_fast = timed_pipe
elif hasattr(spm, "faster_transcribe"):  # upstream v4: one faster-whisper call per chapter
    orig_ft = spm.faster_transcribe


    def timed_ft(self, audio, name, **a):
        t = time.perf_counter()
        r = orig_ft(self, audio, name, **a)
        add("transcribe", time.perf_counter() - t)
        return r


    spm.faster_transcribe = timed_ft

# decode: SubPlz decodes each chunk with an ffmpeg subprocess.run
orig_run = subprocess.run


def timed_run(cmd, *a, **k):
    t = time.perf_counter()
    r = orig_run(cmd, *a, **k)
    if cmd and str(cmd[0]).lower().endswith("ffmpeg"):
        add("decode", time.perf_counter() - t)
    return r


subprocess.run = timed_run

# alignment: chapter matching (fuzzy_match_chapters) and per-chapter do_batch; the
# align wall time spans the first to the last do_batch (they run in a thread pool)
import subplz.sync as sy  # noqa: E402

ALIGN = []
orig_fm = sy.fuzzy_match_chapters


def timed_fm(*a, **k):
    t = time.perf_counter()
    r = orig_fm(*a, **k)
    add("match", time.perf_counter() - t)
    return r


sy.fuzzy_match_chapters = timed_fm

orig_do = sy.do_batch


def timed_do(*a, **k):
    t = time.perf_counter()
    r = orig_do(*a, **k)
    with lock:
        ALIGN.append((t, time.perf_counter()))
    return r


sy.do_batch = timed_do

if epub == "-":  # no book: `gen` (the stages up to the VAD are the same)
    cli = ["subplz", "gen", "--audio", audio, "--output-dir", out_dir,
           "--model", model, "--device", "cuda", "--cache-dir", cache_dir]
else:
    cli = ["subplz", "sync", "--audio", audio, "--text", epub, "--output-dir", out_dir,
           "--model", model, "--device", "cuda", "--cache-dir", cache_dir]
sys.argv = cli
t0 = time.perf_counter()
try:
    runpy.run_module("subplz", run_name="__main__", alter_sys=True)
finally:
    T["total"] = time.perf_counter() - t0
    T["asr"] = T["transcribe"] - T["vad"]
    if ALIGN:
        T["align"] = max(e for _, e in ALIGN) - min(s for s, _ in ALIGN)
        T["align_cpu"] = sum(e - s for s, e in ALIGN)
    print("STAGES " + json.dumps({k: round(v, 3) for k, v in T.items()}), flush=True)
