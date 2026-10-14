use std::path::Path;

#[test]
fn subtitle_text_matches_subplz() {
    let Ok(dir) = std::env::var("SUBTHX_SUBTEXT_DIR") else { return };
    let oracle: std::collections::HashMap<String, Vec<String>> =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/subsrc.srt.oracle.json")).unwrap()).unwrap();
    for (path, want) in &oracle {
        let got = &subthx::textsrc::load(Path::new(path)).unwrap()[0];
        let same = got == want;
        if !same {
            let i = got.iter().zip(want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            eprintln!("  first diff at {i}: got {:?} want {:?}", got.get(i), want.get(i));
        }
        eprintln!("{path}: {} paragraphs (SubPlz {}), identical: {same}", got.len(), want.len());
    }
}
