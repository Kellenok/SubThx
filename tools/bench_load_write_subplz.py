"""Model load and subtitle writing of an upstream SubPlz tree, timed on their own.

    SUBPLZ_ROOT=<export of the tag> PYTHONUTF8=1 G:/SubPlz/.venv/Scripts/python.exe \
        tools/bench_load_write_subplz.py load            # one cold process per run
    SUBPLZ_ROOT=<export> ... tools/bench_load_write_subplz.py write <srt>

load: subplz.models.get_model with the backend fields the CLI passes for the default
      (stable-ts + faster-whisper, whisper tiny, CUDA), clock started before the imports
      that the CLI does when it loads a model. Prints LOAD <seconds>.
write: subplz.files.Writer.write_sub (what the CLI calls) with the cues of an SRT file,
       to a file. Median of 5 writes in one process. Prints WRITE <seconds>.
"""
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = os.environ.get("SUBPLZ_ROOT", "G:/SubPlz")
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # SubPlz writes config/ relative to the working directory

mode = sys.argv[1]

if mode == "load":
    # the clock starts before the imports: the CLI imports torch, faster-whisper and
    # whisper when it loads the model, and that time belongs to the load
    t = time.perf_counter()
    from subplz.models import get_model

    backend = SimpleNamespace(
        model_name="tiny", device="cuda", faster_whisper=False, stable_ts=True,
        local_only=True, quantize=False, threads=16,
    )
    get_model(backend)
    print(f"LOAD {time.perf_counter() - t:.3f}", flush=True)

elif mode == "write":
    from subplz.sub import Segment
    from subplz.files import Writer  # what the CLI calls to write the subtitles

    def secs(ts):
        h, m, rest = ts.split(":")
        s, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    text = Path(sys.argv[2]).read_text(encoding="utf-8-sig")
    segments = []
    for block in re.split(r"\r?\n\r?\n", text.strip()):
        lines = block.splitlines()
        if len(lines) >= 3:
            a, b = (secs(x) for x in lines[1].split(" --> "))
            segments.append(Segment(start=a, end=b, text=" ".join(lines[2:]).strip()))
    times = []
    out = Path(tempfile.mkdtemp()) / "out.srt"
    writer = Writer("srt")
    for _ in range(5):
        t = time.perf_counter()
        writer.write_sub(segments, out)
        times.append(time.perf_counter() - t)
    print(f"CUES {len(segments)}")
    print(f"WRITE {statistics.median(times):.4f}", flush=True)
