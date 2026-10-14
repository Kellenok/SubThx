"""Metrics of the subtitles a run actually wrote (same normalization and metrics as tools/eval_book.py),
for the chapters 2..12 of the book. Run with SubPlz's venv: .venv/Scripts/python.exe eval_srt.py NAME=FILE.srt ..."""
import re, sys, unicodedata
from pathlib import Path

sys.path.insert(0, "G:/SubPlz")
from rapidfuzz.distance import Levenshtein
from subplz.files import Epub

EPUB = r"E:\audiobooks\Audiobook Collection\[二語 十] 探偵はもう、死んでいる。\test\探偵はもう、死んでいる。【電子特典付き】.epub"
PUNC = re.compile(r'[\s、。！？!?…・—―–\-\(\)（）「」『』【】〜～"\'“”‘’＜＞<>:：;；*＊+＋/_／＼=＝♪]')


def norm(t):
    return PUNC.sub("", unicodedata.normalize("NFKC", t)).lower()


epub = Epub.from_file(Path(EPUB))
gt = norm("".join(p.text() for ch in [epub.chapters[i] for i in range(2, 13)] for p in ch.text()))


import ffmpeg  # noqa: E402

AUDIO = r"E:\audiobooks\Audiobook Collection\[二語 十] 探偵はもう、死んでいる。\test\[01] 探偵はもう、死んでいる。 [B0DQCZ9TNQ].m4b"
chapters = ffmpeg.probe(AUDIO, show_chapters=None)["chapters"]
# the story: audio chapters 1..11 (the same set eval_book.py aligns)
STORY = [(float(chapters[i]["start_time"]), float(chapters[i]["end_time"])) for i in range(1, 12)]


def secs(ts):
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def srt_text(path, story_only=True):
    text = Path(path).read_text(encoding="utf-8-sig")
    cues = []
    for block in re.split(r"\r?\n\r?\n", text.strip()):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        a, b = (secs(t) for t in lines[1].split(" --> "))
        mid = (a + b) / 2
        if story_only and not any(s <= mid < e for s, e in STORY):
            continue
        cues.append(" ".join(lines[2:]).strip())
    return cues


def metrics(ev):
    ops = Levenshtein.editops(ev, gt)
    missing = sum(op[0] == "insert" for op in ops)
    extra = sum(op[0] == "delete" for op in ops)
    replace = sum(op[0] == "replace" for op in ops)
    return {"match": (len(gt) - missing - replace) / len(gt) * 100, "extra": extra, "extra_pct": extra / max(len(ev), 1) * 100,
            "cer": Levenshtein.distance(ev, gt) / len(gt) * 100}


print(f"book (chapters 2..12, L1): {len(gt):,} chars")
print(f"{'run':<28} | {'cues':>5} {'book match':>10} {'extra':>13} {'final CER':>9}")
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    cues = srt_text(path)
    m = metrics(norm("".join(cues)))
    print(f"{name:<28} | {len(cues):5d} {m['match']:9.2f}% {m['extra']:6,} ({m['extra_pct']:.2f}%) {m['cer']:8.2f}%")
