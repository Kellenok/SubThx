use std::time::Instant;
use subthx::media::probe;

#[test]
fn chapters_match_ffprobe() {
    let index = std::fs::read_to_string("fixtures/probe/index.tsv").unwrap();
    let mut bad = 0;
    for line in index.lines() {
        let (h, path) = line.split_once('\t').unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("fixtures/probe/{h}.json")).unwrap()).unwrap();
        let want = v["chapters"].as_array().unwrap();
        let t = Instant::now();
        let got = probe(std::path::Path::new(path)).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let mut issues = Vec::new();
        if got.chapters.len() != want.len() {
            issues.push(format!("{} chapters, want {}", got.chapters.len(), want.len()));
        }
        for (g, w) in got.chapters.iter().zip(want) {
            let ws: f64 = w["start_time"].as_str().unwrap().parse().unwrap();
            let we: f64 = w["end_time"].as_str().unwrap().parse().unwrap();
            let wt = w["tags"]["title"].as_str().unwrap_or("");
            if g.id != w["id"].as_i64().unwrap() || g.start != ws || g.end != we || g.title != wt {
                issues.push(format!("got {} {} {} {:?} want {} {} {} {:?}", g.id, g.start, g.end, g.title, w["id"], ws, we, wt));
            }
        }
        eprintln!("{h} ({ms:.1} ms) {} chapters {}", got.chapters.len(), if issues.is_empty() { "ok" } else { "MISMATCH" });
        for i in issues.iter().take(4) {
            eprintln!("   {i}");
        }
        if !issues.is_empty() {
            bad += 1;
        }
    }
    assert_eq!(bad, 0);
}
