"""pysbd (ja, clean=False) + SubPlz's quote fixes on real lines, as an oracle.

Usage (SubPlz venv): python tools/dump_sentences.py
"""
import json, sys, glob
sys.path.insert(0, "G:/SubPlz")
import pysbd
from subplz.text import fix_end_of_quotes, merge_short_lines_with_quotes

seg = pysbd.Segmenter(language="ja", clean=False)
lines = []
for f in sorted(glob.glob("fixtures/epub/*.json"))[:4]:
    d = json.load(open(f, encoding="utf-8"))
    lines += [p for c in d["chapters"] for p in c["paragraphs"]]
for f in sorted(glob.glob("fixtures/e2e/*.srt")):
    blocks = open(f, encoding="utf-8").read().split("\n\n")
    lines += [b.split("\n", 2)[2] for b in blocks if b.count("\n") >= 2]
rows = []
for text in lines:
    rows.append({"line": text, "pysbd": seg.segment(text)})
allseg = [s for r in rows for s in r["pysbd"]]
fixed = merge_short_lines_with_quotes(fix_end_of_quotes(allseg))
json.dump({"rows": rows, "fixed_all": fixed}, open("fixtures/text/sentences.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(rows), "lines,", len(allseg), "sentences")
