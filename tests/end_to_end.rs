//! EPUB parsed by Rust + cached transcripts -> SRT identical to SubPlz.
use std::time::Instant;
use subthx::epub::Epub;
use subthx::fixture::{fixture_files, Fixture};
use subthx::shift::{shift_align, write_srt};
use subthx::sync::{align_batches, expand_matches, match_start, TextFile};

fn epub_path(fixture: &std::path::Path) -> &'static str {
    let n = fixture.file_name().unwrap().to_string_lossy();
    if n.starts_with("tantei") {
        "E:/audiobooks/Audiobook Collection/[二語 十] 探偵はもう、死んでいる。/test/探偵はもう、死んでいる。【電子特典付き】.epub"
    } else {
        "E:/audiobooks/クラスの大嫌いな女子と結婚することになった。/7/クラスの大嫌いな女子と結婚することになった。７【電子特典付き】.epub"
    }
}

#[test]
fn epub_to_srt_end_to_end() {
    for f in fixture_files() {
        let fx = Fixture::load(&f);
        let (audio, _) = fx.inputs();
        let p = fx.punctuation.sets();
        let t = Instant::now();
        let epub = Epub::open(std::path::Path::new(epub_path(&f))).unwrap();
        let t_epub = t.elapsed();
        let text = vec![TextFile { chapters: epub.chapters.iter().map(|c| c.paragraphs.clone()).collect() }];
        let (ats, sta) = match_start(&audio, &text);
        let batches = expand_matches(&audio, &text, &ats, &sta);
        let segs: Vec<_> = align_batches(&audio[0], &text, &batches[0], &p).into_iter().flatten().collect();
        let srt = write_srt(&shift_align(segs));
        let total = t.elapsed();
        eprintln!(
            "{}: srt identical: {} | epub {:.1} ms, total {:.1} ms",
            f.display(), srt == fx.srt, t_epub.as_secs_f64() * 1e3, total.as_secs_f64() * 1e3
        );
        assert_eq!(srt, fx.srt);
    }
}
