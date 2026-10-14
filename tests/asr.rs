use std::path::Path;
use std::time::Instant;
use subthx::asr::{Chunk, Parakeet};
use subthx::npy::read_f32;
use subthx::vad::Device;

#[derive(serde::Deserialize)]
struct Segs { samples: Vec<(usize, usize)> }
#[derive(serde::Deserialize)]
struct Want { seconds: f64, texts: Vec<String> }

fn lev(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + (ca != cb) as usize).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

#[test]
fn parakeet_matches_subplz() {
    let device: Device = std::env::var("SUBTHX_DEVICE").unwrap_or_else(|_| "cuda".into()).parse().unwrap();
    let limit: usize = std::env::var("SUBTHX_ASR_LIMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let (_, audio) = read_f32(Path::new("fixtures/tantei16k.npy")).unwrap();
    let segs: Segs = serde_json::from_str(&std::fs::read_to_string("fixtures/vad/segments.json").unwrap()).unwrap();
    let want: Want = serde_json::from_str(&std::fs::read_to_string("fixtures/asr/tantei.json").unwrap()).unwrap();
    let chunks: Vec<Chunk> = segs.samples.iter().take(limit).map(|&(s, e)| Chunk { start: s, end: e }).collect();

    let t = Instant::now();
    let mut m = Parakeet::load(Path::new("G:/SubPlz/models/parakeet-tdt-ja-fp16"), Path::new(&std::env::var("SUBTHX_PRE").unwrap_or_else(|_| "models/nemo80_fbanks.npy".into())), device).unwrap();
    eprintln!("load: {:.1}s", t.elapsed().as_secs_f64());
    m.transcribe(&audio, &chunks[..64.min(chunks.len())], 64).unwrap(); // warmup
    let t = Instant::now();
    m.profile = [0.0; 4];
    let texts = m.transcribe(&audio, &chunks, 64).unwrap();
    eprintln!("profile: pre {:.2}s, encoder {:.2}s, decoder loop {:.2}s (of which runs {:.2}s)", m.profile[0], m.profile[1], m.profile[2], m.profile[3]);
    let dt = t.elapsed().as_secs_f64();

    let mut same = 0;
    let (mut edits, mut total) = (0usize, 0usize);
    let mut shown = 0;
    for (g, w) in texts.iter().zip(&want.texts) {
        let w = w.trim();
        if g == w {
            same += 1;
        } else if shown < 5 {
            shown += 1;
            eprintln!("  got  {g:?}\n  want {w:?}");
        }
        let (a, b): (Vec<char>, Vec<char>) = (g.chars().collect(), w.chars().collect());
        edits += lev(&a, &b);
        total += b.len();
    }
    eprintln!(
        "{device:?}: {} chunks in {dt:.2}s ({:.0}x realtime; SubPlz {:.2}s for all), identical {same}/{}, CER vs SubPlz {:.3}%",
        chunks.len(), (chunks.last().unwrap().end as f64 / 16000.0) / dt, want.seconds, texts.len(), edits as f64 * 100.0 / total as f64
    );
}
