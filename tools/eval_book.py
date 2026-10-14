"""Book-level quality, same metrics as the full-book benchmark (run_full_book_final.py):
raw ASR vs book text, and after per-chapter SubPlz alignment (do_batch), both
NFKC + punctuation-stripped (L1).

Usage (SubPlz venv):
  python tools/eval_book.py <name>=<transcripts.json | subplz-cache:<model>[@<cache dir>]> ...
transcripts.json: per kept audio chapter, list of {start, end, text} relative to the chapter
subplz-cache: the SubPlz cache of a run (default: G:/SubPlz/config/cache)
"""
import json, os, re, sys, unicodedata
from pathlib import Path

import ffmpeg
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "G:/SubPlz")
from subplz.files import Epub  # noqa: E402
from subplz.sync import do_batch  # noqa: E402

AUDIO = r"E:\audiobooks\Audiobook Collection\[二語 十] 探偵はもう、死んでいる。\test\[01] 探偵はもう、死んでいる。 [B0DQCZ9TNQ].m4b"
EPUB = r"E:\audiobooks\Audiobook Collection\[二語 十] 探偵はもう、死んでいる。\test\探偵はもう、死んでいる。【電子特典付き】.epub"
CACHE = Path("G:/SubPlz/config/cache")

PUNC = re.compile(r'[\s、。！？!?…・—―–\-\(\)（）「」『』【】〜～"\'“”‘’＜＞<>:：;；*＊+＋/_／＼=＝♪]')


def norm(t):
    return PUNC.sub("", unicodedata.normalize("NFKC", t)).lower()


epub = Epub.from_file(Path(EPUB))
gt = norm("".join(p.text() for ch in [epub.chapters[i] for i in range(2, 13)] for p in ch.text()))
chapters = ffmpeg.probe(AUDIO, show_chapters=None)["chapters"]
kept = [c for c in chapters if float(c["end_time"]) - float(c["start_time"]) > 1]
story = [{"start": float(chapters[i]["start_time"]), "end": float(chapters[i]["end_time"]), "epub": i + 1} for i in range(1, 12)]


def metrics(ev):
    ops = Levenshtein.editops(ev, gt)
    missing = sum(op[0] == "insert" for op in ops)
    extra = sum(op[0] == "delete" for op in ops)
    replace = sum(op[0] == "replace" for op in ops)
    return {"match": (len(gt) - missing - replace) / len(gt) * 100, "extra": extra, "extra_pct": extra / max(len(ev), 1) * 100,
            "cer": Levenshtein.distance(ev, gt) / len(gt) * 100}


def load(spec):
    if spec.startswith("subplz-cache:"):
        model, _, cache_dir = spec.split(":", 1)[1].partition("@")
        cache = Path(cache_dir) if cache_dir else CACHE
        per = []
        for c in kept:
            p = cache / f"{Path(AUDIO).name}.{c['id']}.{model}.subs"
            per.append(eval(p.read_text(encoding="utf-8"))["segments"])
    else:
        per = json.loads(Path(spec).read_text(encoding="utf-8"))
    segs = []
    for c, ch in zip(kept, per):
        st = float(c["start_time"])
        segs += [{"start": s["start"] + st, "end": s["end"] + st, "text": s["text"]} for s in (ch or [])]
    return segs


print(f"book (chapters 2..12, L1): {len(gt):,} chars")
print(f"{'model':<26} | {'raw match':>9} {'raw CER':>8} | {'cues':>5} {'book match':>10} {'extra':>13} {'final CER':>9}")
for arg in sys.argv[1:]:
    name, spec = arg.split("=", 1)
    segs = load(spec)
    raw = metrics(norm("".join(s["text"] for s in segs)))
    cues = []
    for ch in story:
        dur = ch["end"] - ch["start"]
        cs = [{"start": max(0.0, s["start"] - ch["start"]), "end": min(dur, s["end"] - ch["start"]), "text": s["text"]}
              for s in segs if ch["start"] <= (s["start"] + s["end"]) / 2 < ch["end"]]
        if cs:
            cues += do_batch([({"segments": cs, "language": "ja"}, dur)], [epub.chapters[ch["epub"]]], set(), set(), set(), ch["start"])
    fz = metrics(norm("".join(c.text.strip() for c in cues if c.text.strip())))
    print(f"{name:<26} | {raw['match']:8.2f}% {raw['cer']:7.2f}% | {len(cues):5d} {fz['match']:9.2f}% "
          f"{fz['extra']:6,} ({fz['extra_pct']:.2f}%) {fz['cer']:8.2f}%")
