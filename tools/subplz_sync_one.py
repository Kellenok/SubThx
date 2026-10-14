"""Oracle: SubPlz's alignment of one transcript against one text file
(do_batch + shift_align), for comparing with `subthx sync --transcripts`.

Usage (SubPlz venv, PYTHONUTF8=1):
  python tools/subplz_sync_one.py <transcripts.json> <book.txt> <language> <out.json>
transcripts.json: what `subthx sync --transcripts` writes (one chapter here).
Writes [[text, start, end], ...].
"""
import json
import sys

from ats.main import TextFile
from subplz.align import shift_align
from subplz.cli import END_PUNC, OTHER_PUNC, START_PUNC
from subplz.sync import do_batch

NOPEND = "うぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ…　\x20"

tr_path, book, language, out = sys.argv[1:5]
chapters = json.load(open(tr_path, encoding="utf-8"))
assert len(chapters) == 1, "one audio chapter expected"
segs = chapters[0]
duration = max(s["end"] for s in segs) + 1.0
segments = do_batch(
    [({"segments": segs, "language": language}, duration)],
    [TextFile(path=book, title=book)],
    set(START_PUNC),
    set(END_PUNC + OTHER_PUNC),
    set(NOPEND),
    0,
)
shifted = shift_align(segments)
json.dump([[s.text, s.start, s.end] for s in shifted], open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(len(shifted), "segments")
