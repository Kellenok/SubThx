use std::time::Instant;
use subthx::lev::{opcodes_chars, Tag};

#[derive(serde::Deserialize)]
struct Call {
    s1: String,
    s2: String,
    ops: Vec<(String, usize, usize, usize, usize)>,
}

fn tag_char(t: Tag) -> &'static str {
    match t {
        Tag::Equal => "e",
        Tag::Replace => "r",
        Tag::Insert => "i",
        Tag::Delete => "d",
    }
}

#[test]
fn opcodes_match_rapidfuzz() {
    let mut files: Vec<_> = std::fs::read_dir("fixtures/ops").expect("run tools/dump_fixture.py first")
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    for f in files {
        let calls: Vec<Call> = serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
        let inputs: Vec<(Vec<char>, Vec<char>)> =
            calls.iter().map(|c| (c.s1.chars().collect(), c.s2.chars().collect())).collect();
        let t = Instant::now();
        let outs: Vec<_> = inputs.iter().map(|(a, b)| opcodes_chars(a, b)).collect();
        let dt = t.elapsed();
        let mut bad = 0;
        for (k, (c, got)) in calls.iter().zip(&outs).enumerate() {
            let got: Vec<_> = got
                .iter()
                .map(|o| (tag_char(o.tag).to_string(), o.src_start, o.src_end, o.dest_start, o.dest_end))
                .collect();
            if got != c.ops {
                bad += 1;
                if bad <= 3 {
                    let i = got.iter().zip(&c.ops).position(|(a, b)| a != b).unwrap_or(got.len().min(c.ops.len()));
                    eprintln!(
                        "{} call {k} (len {}x{}): first diff at op {i}: got {:?} want {:?}",
                        f.display(), inputs[k].0.len(), inputs[k].1.len(), got.get(i), c.ops.get(i)
                    );
                }
            }
        }
        eprintln!("{}: {} calls, {} mismatched, rust {:.1} ms", f.display(), calls.len(), bad, dt.as_secs_f64() * 1e3);
        assert_eq!(bad, 0);
    }
}
