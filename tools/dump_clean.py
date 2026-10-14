"""Dump ats.lang Japanese clean()/normalize() outputs for every string in the fixtures."""
import glob, json, sys
sys.path.insert(0, "G:/SubPlz")
from ats.lang import get_lang

lang = get_lang("ja")
seen = set()
rows = []
for f in sorted(glob.glob("fixtures/*.json")):
    fx = json.load(open(f, encoding="utf-8"))
    strings = [p for c in fx["epub_chapters"] for p in c["paragraphs"]]
    strings += [s["text"] for t in fx["transcripts"].values() for s in t["segments"]]
    for s in strings:
        if s in seen:
            continue
        seen.add(s)
        rows.append([s, lang.clean(s), lang.translate(s), lang.normalize(s)])
json.dump(rows, open("fixtures/clean/clean.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(rows), "strings")
