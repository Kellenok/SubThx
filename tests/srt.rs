use std::time::Instant;
use subthx::fixture::{fixture_files, Fixture};
use subthx::shift::{shift_align, write_srt, END_PUNC, OTHER_PUNC, START_PUNC};
use subthx::sync::{align_batches, expand_matches, match_start};

#[test]
fn full_pipeline_srt_matches_subplz() {
    let mut bad = 0;
    for f in fixture_files() {
        let fx = Fixture::load(&f);
        assert_eq!(fx.punctuation.prepend, START_PUNC);
        assert_eq!(fx.punctuation.append, format!("{END_PUNC}{OTHER_PUNC}"));
        let (audio, text) = fx.inputs();
        let p = fx.punctuation.sets();

        let t = Instant::now();
        let (ats, sta) = match_start(&audio, &text);
        let batches = expand_matches(&audio, &text, &ats, &sta);
        let segs: Vec<_> = align_batches(&audio[0], &text, &batches[0], &p).into_iter().flatten().collect();
        let t_shift = Instant::now();
        let shifted = shift_align(segs);
        let srt = write_srt(&shifted);
        let shift_ms = t_shift.elapsed().as_secs_f64() * 1e3;
        let total_ms = t.elapsed().as_secs_f64() * 1e3;

        let mut cue_bad = 0;
        for (i, (g, w)) in shifted.iter().zip(&fx.final_segments).enumerate() {
            if g.text != w.text || g.start != w.start || g.end != w.end {
                cue_bad += 1;
                if cue_bad <= 3 {
                    eprintln!("cue {i}:\n  got  {:?}\n  want {:?}", g.text, w.text);
                }
            }
        }
        let srt_ok = srt == fx.srt;
        if !srt_ok {
            let line = srt.lines().zip(fx.srt.lines()).position(|(a, b)| a != b);
            if let Some(l) = line {
                eprintln!("srt line {l}: got {:?} want {:?}", srt.lines().nth(l), fx.srt.lines().nth(l));
            }
        }
        let py_ms: f64 = fx.timings_s.values().sum::<f64>() * 1e3;
        eprintln!(
            "{}: {} cues (want {}), {cue_bad} bad, srt identical: {srt_ok} | rust total {total_ms:.1} ms (shift {shift_ms:.1}) vs python {py_ms:.1} ms",
            f.display(), shifted.len(), fx.final_segments.len()
        );
        if cue_bad > 0 || !srt_ok || shifted.len() != fx.final_segments.len() {
            bad += 1;
        }
    }
    assert_eq!(bad, 0);
}
