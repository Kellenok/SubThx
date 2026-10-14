use std::time::Instant;
use subthx::fixture::{fixture_files, Fixture};
use subthx::sync::{align_batches, expand_matches, match_start};

#[test]
fn chapter_matching_and_batches_match_subplz() {
    let mut bad = 0;
    for f in fixture_files() {
        let fx = Fixture::load(&f);
        let (audio, text) = fx.inputs();
        let p = fx.punctuation.sets();

        let t = Instant::now();
        let (ats, sta) = match_start(&audio, &text);
        let batches = expand_matches(&audio, &text, &ats, &sta);
        let t_match = t.elapsed();
        let t = Instant::now();
        let segs = align_batches(&audio[0], &text, &batches[0], &p);
        let t_align = t.elapsed();

        let mut want_ats: Vec<_> = fx.matching.ats.clone();
        want_ats.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let mut got_ats: Vec<_> = ats.iter().map(|(k, v)| (k.0, k.1, v.0, v.1, v.2)).collect();
        got_ats.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        if got_ats != want_ats {
            bad += 1;
            eprintln!("ats differ:\n got  {got_ats:?}\n want {want_ats:?}");
        }
        let got_b: Vec<_> = batches[0]
            .iter()
            .map(|b| (b.audio.clone(), (b.text_file, b.text_parts.clone()), b.score))
            .collect();
        if got_b != fx.matching.batches[0] {
            bad += 1;
            eprintln!("batches differ:\n got  {got_b:?}\n want {:?}", fx.matching.batches[0]);
        }
        let mut seg_bad = 0;
        for (g, st) in segs.iter().zip(&fx.stages) {
            if g.len() != st.segments.len()
                || g.iter().zip(&st.segments).any(|(a, b)| a.text != b.text || a.start != b.start || a.end != b.end)
            {
                seg_bad += 1;
            }
        }
        bad += seg_bad;
        eprintln!(
            "{}: {} matches, {} batches, {seg_bad} bad stages | rust match {:.1} ms (py {:.1}), align {:.1} ms (py {:.1})",
            f.display(), ats.len(), batches[0].len(),
            t_match.as_secs_f64() * 1e3, fx.timings_s["match"] * 1e3,
            t_align.as_secs_f64() * 1e3, fx.timings_s["align"] * 1e3
        );
    }
    assert_eq!(bad, 0);
}
