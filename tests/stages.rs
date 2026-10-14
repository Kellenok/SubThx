use std::time::Instant;
use subthx::batch::{align_chapter_task, AudioPart};
use subthx::fixture::{fixture_files, Fixture};

#[test]
fn batch_alignment_matches_subplz() {
    let mut total_bad = 0;
    for f in fixture_files() {
        let fx = Fixture::load(&f);
        let p = fx.punctuation.sets();
        let t = Instant::now();
        let mut outs = Vec::new();
        for st in &fx.stages {
            let ach: Vec<AudioPart> = st
                .audio
                .iter()
                .map(|&aj| AudioPart { transcript: fx.transcript(aj), duration: fx.audio_chapters[aj].duration })
                .collect();
            let paras: Vec<&str> = if st.text_chapter >= 0 {
                st.text_parts.iter().flat_map(|&j| fx.epub_chapters[j].paragraphs.iter().map(|s| s.as_str())).collect()
            } else {
                Vec::new()
            };
            outs.push(align_chapter_task(&ach, &paras, st.offset, &p));
        }
        let dt = t.elapsed();
        let mut bad = 0;
        let mut cues = 0;
        for (k, (st, got)) in fx.stages.iter().zip(&outs).enumerate() {
            cues += st.segments.len();
            if got.len() != st.segments.len() {
                eprintln!("stage {k}: {} cues, want {}", got.len(), st.segments.len());
                bad += 1;
                continue;
            }
            for (i, (g, w)) in got.iter().zip(&st.segments).enumerate() {
                if g.text != w.text || g.start != w.start || g.end != w.end {
                    bad += 1;
                    if bad <= 5 {
                        eprintln!("stage {k} cue {i}:\n  got  {:?} {} {}\n  want {:?} {} {}", g.text, g.start, g.end, w.text, w.start, w.end);
                    }
                }
            }
        }
        eprintln!(
            "{}: {} stages, {cues} cues, {bad} mismatched | rust {:.1} ms vs python {:.1} ms",
            f.display(), fx.stages.len(), dt.as_secs_f64() * 1e3, fx.timings_s["align"] * 1e3
        );
        total_bad += bad;
    }
    assert_eq!(total_bad, 0);
}
