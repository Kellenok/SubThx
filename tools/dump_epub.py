"""Dump SubPlz's EPUB chapter/paragraph extraction for every epub under a dir."""
import glob, json, sys, hashlib
from pathlib import Path
sys.path.insert(0, "G:/SubPlz")
from subplz.text import Epub

root = sys.argv[1] if len(sys.argv) > 1 else "E:/audiobooks"
out = Path("fixtures/epub"); out.mkdir(parents=True, exist_ok=True)
for f in sorted(glob.glob(root + "/**/*.epub", recursive=True)):
    e = Epub.from_file(Path(f))
    d = {
        "path": f,
        "title": e.title,
        "chapters": [
            {"idx": c.idx, "title": c.title, "paragraphs": [p.text() for p in c.text()]}
            for c in e.chapters
        ],
    }
    name = hashlib.md5(f.encode()).hexdigest()[:10] + ".json"
    (out / name).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print(name, len(d["chapters"]), sum(len(c["paragraphs"]) for c in d["chapters"]), f[-50:])
