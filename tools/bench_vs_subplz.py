"""Wall-clock: `subplz sync` (cached transcripts) vs `subthx sync`, same inputs."""
import os, subprocess, sys, tempfile, time
from pathlib import Path

books = [
    ("E:/audiobooks/Audiobook Collection/[二語 十] 探偵はもう、死んでいる。/test/[01] 探偵はもう、死んでいる。 [B0DQCZ9TNQ].m4b",
     "E:/audiobooks/Audiobook Collection/[二語 十] 探偵はもう、死んでいる。/test/探偵はもう、死んでいる。【電子特典付き】.epub"),
    ("E:/audiobooks/クラスの大嫌いな女子と結婚することになった。/7/[7巻] クラスの大嫌いな女子と結婚することになった。7꞉ (KADOKAWA) [B0DPM4NTF9].m4b",
     "E:/audiobooks/クラスの大嫌いな女子と結婚することになった。/7/クラスの大嫌いな女子と結婚することになった。７【電子特典付き】.epub"),
]
CACHE = "G:/SubPlz/config/cache"
PY = "G:/SubPlz/.venv/Scripts/python.exe"
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
env = dict(os.environ, PYTHONIOENCODING="utf-8")

def run(cmd, cwd):
    t = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True)
    dt = time.perf_counter() - t
    if r.returncode:
        print(r.stderr.decode("utf-8", "replace")[-2000:]); raise SystemExit("failed")
    return dt

for audio, epub in books:
    tmp = Path(tempfile.mkdtemp())
    sp, st = [], []
    for i in range(RUNS):
        out = tmp / f"sp{i}"; out.mkdir()
        sp.append(run([PY, "-m", "subplz", "sync", "--audio", audio, "--text", epub, "--output-dir", str(out),
                       "--model", "parakeet", "--cache-dir", CACHE], "G:/SubPlz"))
        st.append(run(["G:/subthx/target/release/subthx.exe", "sync", audio, epub, "--cache-dir", CACHE,
                       "-o", str(tmp / f"st{i}.srt"), "--overwrite"], "G:/subthx"))
    sp_srt = next((tmp / "sp0").glob("*.srt")).read_bytes()
    same = sp_srt == (tmp / "st0.srt").read_bytes()
    print(f"{Path(audio).name[:30]}: subplz {min(sp):.2f}s (runs {', '.join(f'{x:.2f}' for x in sp)}) | "
          f"subthx {min(st)*1000:.0f}ms (runs {', '.join(f'{x*1000:.0f}' for x in st)}) | "
          f"x{min(sp)/min(st):.0f} faster | identical srt: {same}")
