"""Dump a golden fixture from SubPlz's sync pipeline using cached transcripts.

Runs the exact SubPlz alignment code (match_start -> expand_matches -> do_batch
-> shift_align -> write_srt) on transcripts already in the SubPlz cache, and
records every intermediate stage so the Rust port can be diffed step by step.

Usage (with the SubPlz venv):
    G:/SubPlz/.venv/Scripts/python.exe tools/dump_fixture.py \
        --audio "<book>.m4b" --epub "<book>.epub" \
        --model parakeet --out fixtures/<name>.json
"""

import argparse
import copy
import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SUBPLZ = Path(os.environ.get("SUBPLZ_DIR", "G:/SubPlz"))
sys.path.insert(0, str(SUBPLZ))

from ats.main import expand_matches, write_srt  # noqa: E402
from subplz import sync as S  # noqa: E402
from subplz.align import shift_align  # noqa: E402
from subplz.cache import Cache  # noqa: E402
from subplz.cli import START_PUNC, END_PUNC, OTHER_PUNC  # noqa: E402
from subplz.files import AudioSub  # noqa: E402
from subplz.text import Epub  # noqa: E402

NOPEND = "うぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ…\u3000\x20"
PREPEND = START_PUNC
APPEND = END_PUNC + OTHER_PUNC


def seg_dict(s):
    return {"text": s.text, "start": s.start, "end": s.end}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--epub", required=True)
    ap.add_argument("--model", default="parakeet")
    ap.add_argument("--cache-dir", default=str(SUBPLZ / "config" / "cache"))
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--record-opcodes",
        help="also write every Levenshtein.opcodes call (inputs+outputs) here",
    )
    a = ap.parse_args()

    calls = []
    if a.record_opcodes:
        import ats.align as ats_align

        real = ats_align.Levenshtein

        class Recorder:
            @staticmethod
            def opcodes(s1, s2):
                ops = real.opcodes(s1, s2)
                calls.append(
                    {
                        "s1": s1,
                        "s2": s2,
                        "ops": [
                            [o.tag[0], o.src_start, o.src_end, o.dest_start, o.dest_end]
                            for o in ops
                        ],
                    }
                )
                return ops

        ats_align.Levenshtein = Recorder

    cache_inputs = SimpleNamespace(
        model_name=a.model, cache_dir=a.cache_dir, enabled=True, overwrite=False
    )
    title, audio_subs = AudioSub.from_file(a.audio, cache_inputs)
    streams = [(os.path.basename(a.audio), title, audio_subs)]

    epub = Epub.from_file(Path(a.epub))
    chapters = [("book.txt", epub.chapters)]

    # Raw transcripts, deep-copied before do_batch mutates start/end in place.
    transcripts = {}
    for sub in audio_subs:
        t = sub.transcribe(None)
        if t is None:
            raise SystemExit(f"cache miss for chapter {sub.cid} ({a.model})")
        transcripts[str(sub.cid)] = copy.deepcopy(t)

    t0 = time.perf_counter()
    ats, sta = S.match_start(streams, chapters)
    batches = expand_matches(streams, chapters, ats, sta)
    t_match = time.perf_counter() - t0

    prepend, append, nopend = set(PREPEND), set(APPEND), set(NOPEND)
    stages = []
    all_segments = []
    t0 = time.perf_counter()
    ai = 0
    offset = 0
    for ajs, (chi, chjs), score in batches[ai]:
        ach = [(audio_subs[aj].transcribe(None), audio_subs[aj].duration) for aj in ajs]
        tch = (
            [chapters[chi][1][chj] for chj in chjs]
            if chi is not None and 0 <= chi < len(chapters)
            else []
        )
        segs = S._align_chapter_task((ach, tch, offset, prepend, append, nopend))
        stages.append(
            {
                "audio": ajs,
                "text_chapter": chi,
                "text_parts": chjs,
                "score": score,
                "offset": offset,
                "segments": [seg_dict(s) for s in segs],
            }
        )
        all_segments.extend(segs)
        offset += sum(x[1] for x in ach)
    t_align = time.perf_counter() - t0

    t0 = time.perf_counter()
    shifted = shift_align(all_segments)
    t_shift = time.perf_counter() - t0
    buf = io.StringIO()
    write_srt(shifted, buf)

    fixture = {
        "source": {
            "audio": os.path.basename(a.audio),
            "epub": os.path.basename(a.epub),
            "model": a.model,
        },
        "punctuation": {"prepend": PREPEND, "append": APPEND, "nopend": NOPEND},
        "audio_chapters": [
            {
                "cid": s.cid,
                "title": s.cn,
                "start": s.start_time,
                "end": s.end_time,
                "duration": s.duration,
            }
            for s in audio_subs
        ],
        "epub_chapters": [
            {
                "idx": c.idx,
                "title": c.title,
                "paragraphs": [p.text() for p in c.text()],
            }
            for c in epub.chapters
        ],
        "transcripts": transcripts,
        "match": {
            "ats": [[k[0], k[1], v[0], v[1], v[2]] for k, v in ats.items()],
            "batches": [
                [[ajs, [chi, chjs], score] for ajs, (chi, chjs), score in b]
                for b in batches
            ],
        },
        "stages": stages,
        "final_segments": [seg_dict(s) for s in shifted],
        "srt": buf.getvalue(),
        "timings_s": {"match": t_match, "align": t_align, "shift": t_shift},
    }
    if a.record_opcodes:
        rp = Path(a.record_opcodes)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(calls, ensure_ascii=False), encoding="utf-8")
        print(f"recorded {len(calls)} opcodes calls -> {rp}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    print(
        f"wrote {out}: {len(shifted)} cues, match {t_match:.2f}s, "
        f"align {t_align:.2f}s, shift {t_shift:.2f}s"
    )


if __name__ == "__main__":
    main()
