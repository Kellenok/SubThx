"""Oracle for --respect-grouping (SubPlz align.nc_align).

  script <text file> <out.json>        sentence script as nc_align reads it
  align <script.txt> <subs.srt> <count> <out.json>   nc_align segments

The text file is copied next to the output first: SubPlz writes its
temporary sentence file beside the source. Run with PYTHONUTF8=1.
"""
import json
import shutil
import sys
from pathlib import Path

from subplz.align import nc_align, read_script
from subplz.files import get_chapters

mode = sys.argv[1]
if mode == "script":
    src, out = Path(sys.argv[2]), Path(sys.argv[3])
    work = out.parent / ("regroup_src" + src.suffix)
    shutil.copy(src, work)
    chapters = get_chapters([str(work)], "ja", False, None)
    with open(chapters[0][0], encoding="utf-8") as f:
        lines = list(read_script(f))
    out.write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
    print(f"{len(lines)} script lines")
elif mode == "align":
    script, subs, count, out = sys.argv[2], sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
    segs = nc_align(script, subs, count)
    out.write_text(json.dumps([[s.text, s.start, s.end] for s in segs], ensure_ascii=False), encoding="utf-8")
    print(f"{len(segs)} segments")
