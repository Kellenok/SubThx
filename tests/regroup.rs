//! --respect-grouping against SubPlz (tools/dump_regroup.py outputs in
//! SUBTHX_REGROUP_DIR): the sentence script per source type, and nc_align.

use std::path::{Path, PathBuf};

fn dir() -> Option<PathBuf> {
    std::env::var_os("SUBTHX_REGROUP_DIR").map(PathBuf::from)
}

fn json<T: serde::de::DeserializeOwned>(p: &Path) -> T {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn script_lines_match_subplz() {
    let Some(d) = dir() else { return };
    let cases = [
        (
            PathBuf::from("E:/audiobooks/Audiobook Collection/[二語 十] 探偵はもう、死んでいる。/test/探偵はもう、死んでいる。【電子特典付き】.epub"),
            "script_epub.json",
        ),
        (
            PathBuf::from("E:/audiobooks/Subs collection/Part 1/[1巻] 86‐エイティシックス‐.srt"),
            "script_srt.json",
        ),
        (d.join("t.txt"), "script_txt.json"),
    ];
    for (src, want) in cases {
        let want: Vec<String> = json(&d.join(want));
        let got = subthx::textsrc::script_lines(&src).unwrap();
        let first = got.iter().zip(&want).position(|(a, b)| a != b);
        assert!(
            got == want,
            "{}: {} vs {} lines, first difference at {first:?}: {:?} vs {:?}",
            src.display(),
            got.len(),
            want.len(),
            first.map(|i| &got[i]),
            first.map(|i| &want[i])
        );
    }
}

#[test]
fn nc_align_matches_subplz() {
    let Some(d) = dir() else { return };
    for (script, subs, want) in [
        ("script_small.txt", "subs_small.srt", "nc_small.json"),
        ("script_full.txt", "subs_full.srt", "nc_full.json"),
    ] {
        if !d.join(want).exists() {
            continue;
        }
        let script: Vec<String> = std::fs::read_to_string(d.join(script))
            .unwrap()
            .lines()
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect();
        let subs = std::fs::read_to_string(d.join(subs)).unwrap();
        let t = std::time::Instant::now();
        let got = subthx::regroup::nc_align(&script, &subs, 6);
        eprintln!("{want}: {} segments in {:.2}s", got.len(), t.elapsed().as_secs_f64());
        let want: Vec<(String, f64, f64)> = json(&d.join(want));
        assert_eq!(got.len(), want.len(), "{subs}");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!((&g.text, g.start, g.end), (&w.0, w.1, w.2), "segment {i}");
        }
    }
}
