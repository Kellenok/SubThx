use subthx::textsrc::{pysbd_ja, split_sentences};

#[derive(serde::Deserialize)]
struct Row { line: String, pysbd: Vec<String> }
#[derive(serde::Deserialize)]
struct Fx { rows: Vec<Row>, fixed_all: Vec<String> }

#[test]
fn sentences_match_pysbd() {
    let fx: Fx = serde_json::from_str(&std::fs::read_to_string("fixtures/text/sentences.json").unwrap()).unwrap();
    let mut bad = 0;
    for r in &fx.rows {
        let got = pysbd_ja(&r.line);
        if got != r.pysbd {
            bad += 1;
            if bad <= 8 {
                eprintln!("line {:?}\n  got  {:?}\n  want {:?}", r.line, got, r.pysbd);
            }
        }
    }
    let lines: Vec<String> = fx.rows.iter().map(|r| r.line.clone()).collect();
    let all = split_sentences(&lines);
    let want: Vec<String> = fx.fixed_all.iter().flat_map(|l| l.split('\n')).map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
    let same = all.len() == want.len() && all.iter().zip(&want).all(|(a, b)| a.trim() == b);
    eprintln!("{} lines, {bad} differ from pysbd ({:.3}%); full split_sentences identical: {same} ({} vs {})",
        fx.rows.len(), bad as f64 * 100.0 / fx.rows.len() as f64, all.len(), want.len());
}
