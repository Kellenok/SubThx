use std::path::Path;
use std::time::Instant;
use subthx::asr::Chunk;
use subthx::kitsune::Kitsune;
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
fn kitsune_matches_torch() {
    let size = std::env::var("SUBTHX_KITSUNE").unwrap_or_else(|_| "p0.05b".into());
    let file = std::env::var("SUBTHX_KITSUNE_FILE").unwrap_or_else(|_| "model.fp32.onnx".into());
    let device: Device = std::env::var("SUBTHX_DEVICE").unwrap_or_else(|_| "cuda".into()).parse().unwrap();
    let (_, audio) = read_f32(Path::new("fixtures/tantei16k.npy")).unwrap();
    let segs: Segs = serde_json::from_str(&std::fs::read_to_string("fixtures/vad/segments.json").unwrap()).unwrap();
    let want: Want = serde_json::from_str(&std::fs::read_to_string(format!("fixtures/asr/tantei.kitsune-{size}.json")).unwrap()).unwrap();
    let limit: usize = std::env::var("SUBTHX_ASR_LIMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let chunks: Vec<Chunk> = segs.samples.iter().take(limit).map(|&(s, e)| Chunk { start: s, end: e }).collect();
    let mut k = Kitsune::load(Path::new(&format!("models/kitsune-{size}")), &file, Path::new("models/nemo80_fbanks.npy"), device).unwrap();
    k.transcribe(&audio, &chunks[..64.min(chunks.len())], 64).unwrap();
    k.profile = [0.0; 2];
    let t = Instant::now();
    let bs: usize = std::env::var("SUBTHX_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let texts = k.transcribe(&audio, &chunks, bs).unwrap();
    let dt = t.elapsed().as_secs_f64();
    if let Ok(path) = std::env::var("SUBTHX_DUMP") {
        std::fs::write(path, serde_json::to_string(&texts).unwrap()).unwrap();
    }
    let (mut same, mut edits, mut total, mut shown) = (0, 0, 0, 0);
    for (g, w) in texts.iter().zip(&want.texts) {
        let (g, w) = (g.trim(), w.trim());
        if g == w { same += 1 } else if shown < 4 { shown += 1; eprintln!("  got  {g:?}\n  want {w:?}"); }
        let (a, b): (Vec<char>, Vec<char>) = (g.chars().collect(), w.chars().collect());
        edits += lev(&a, &b);
        total += b.len();
    }
    eprintln!(
        "kitsune {size} {file} {device:?}: {} chunks, {:.0}s of speech in {dt:.2}s ({:.0}x; torch bf16 {:.2}s) | mel {:.2}s model {:.2}s | identical {same}/{} CER vs torch {:.3}%",
        chunks.len(), chunks.iter().map(|c| (c.end - c.start) as f64).sum::<f64>() / 16000.0, chunks.iter().map(|c| (c.end - c.start) as f64).sum::<f64>() / 16000.0 / dt, want.seconds, k.profile[0], k.profile[1], texts.len(), edits as f64 * 100.0 / total as f64
    );
}
