use subthx::lang::Lang;

#[test]
fn clean_matches_python() {
    let rows: Vec<(String, String, String, String)> =
        serde_json::from_str(&std::fs::read_to_string("fixtures/clean/clean.json").unwrap()).unwrap();
    let l = Lang::Japanese;
    let mut bad = [0usize; 3];
    for (raw, clean, tr, norm) in &rows {
        for (k, (got, want)) in [(l.clean(raw), clean), (l.translate(raw), tr), (l.normalize(raw), norm)].iter().enumerate() {
            if got != *want {
                bad[k] += 1;
                if bad[k] <= 3 {
                    let d: Vec<_> = got.chars().zip(want.chars()).enumerate().filter(|(_, (a, b))| a != b).take(1).collect();
                    eprintln!("[{k}] {raw:?}\n  got  {got:?}\n  want {want:?}\n  first diff {d:?}");
                }
            }
        }
    }
    eprintln!("{} strings, mismatches clean/translate/normalize = {bad:?}", rows.len());
    assert_eq!(bad, [0, 0, 0]);
}
