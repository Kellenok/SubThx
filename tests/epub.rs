use std::time::Instant;
use subthx::epub::Epub;

#[derive(serde::Deserialize)]
struct Want {
    path: String,
    title: String,
    chapters: Vec<WantCh>,
}
#[derive(serde::Deserialize)]
struct WantCh {
    idx: usize,
    title: String,
    paragraphs: Vec<String>,
}

#[test]
fn epub_extraction_matches_subplz() {
    let mut files: Vec<_> = std::fs::read_dir("fixtures/epub").unwrap().map(|e| e.unwrap().path()).collect();
    files.sort();
    let mut bad = 0;
    for f in files {
        let w: Want = serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
        let t = Instant::now();
        let e = Epub::open(std::path::Path::new(&w.path)).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let mut issues = Vec::new();
        if e.title != w.title {
            issues.push(format!("book title {:?} != {:?}", e.title, w.title));
        }
        if e.chapters.len() != w.chapters.len() {
            issues.push(format!(
                "chapters {:?} != {:?}",
                e.chapters.iter().map(|c| c.idx).collect::<Vec<_>>(),
                w.chapters.iter().map(|c| c.idx).collect::<Vec<_>>()
            ));
        }
        let mut para_bad = 0;
        for (g, x) in e.chapters.iter().zip(&w.chapters) {
            if g.idx != x.idx || g.title != x.title {
                issues.push(format!("ch {} {:?} != ch {} {:?}", g.idx, g.title, x.idx, x.title));
            }
            if g.paragraphs.len() != x.paragraphs.len() {
                issues.push(format!("ch {} paragraphs {} != {}", g.idx, g.paragraphs.len(), x.paragraphs.len()));
            }
            for (a, b) in g.paragraphs.iter().zip(&x.paragraphs) {
                if a != b {
                    para_bad += 1;
                    if para_bad <= 3 {
                        issues.push(format!("para {:?} != {:?}", a, b));
                    }
                }
            }
        }
        eprintln!("{} ({:.1} ms): {} chapters, {para_bad} bad paragraphs", f.display(), ms, e.chapters.len());
        for i in issues.iter().take(6) {
            eprintln!("   {i}");
        }
        if !issues.is_empty() || para_bad > 0 {
            bad += 1;
        }
    }
    assert_eq!(bad, 0);
}
