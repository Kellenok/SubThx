//! Whisper against transformers (fixtures from tools/dump_whisper.py).

use std::path::Path;
use subthx::asr::Chunk;
use subthx::npy::read_f32;
use subthx::vad::Device;
use subthx::whisper::{Whisper, WhisperMel};

const MODEL: &str = "models/whisper-ja-anime-v0.3";

#[test]
fn features_match_transformers() {
    let mel = WhisperMel::load(&Path::new(MODEL).join("mel_filters.npy")).unwrap();
    for name in ["long", "short"] {
        let (_, audio) = read_f32(Path::new(&format!("fixtures/whisper/{name}_audio.npy"))).unwrap();
        let (shape, want) = read_f32(Path::new(&format!("fixtures/whisper/{name}_features.npy"))).unwrap();
        let got = mel.features(&audio);
        assert_eq!(got.len(), shape.iter().product::<usize>());
        let max = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("{name}: max |diff| {max:.2e}");
        assert!(max < 1e-3, "{name}: features differ by {max}");
    }
}

#[test]
fn transcribes_a_window() {
    let device: Device = std::env::var("SUBTHX_DEVICE").unwrap_or_else(|_| "cuda".into()).parse().unwrap();
    let (_, audio) = read_f32(Path::new("fixtures/whisper/long_audio.npy")).unwrap();
    let mut w = Whisper::load(Path::new(MODEL), "ja", device).unwrap();
    let chunks = [Chunk { start: 0, end: audio.len() }];
    let t = std::time::Instant::now();
    let out = w.transcribe_with(&audio, &chunks, 1, &mut |_| Ok(())).unwrap();
    eprintln!("{:.2}s (features {:.2}, encoder {:.2}, decoder {:.2})", t.elapsed().as_secs_f64(), w.profile[0], w.profile[1], w.profile[2]);
    for u in &out {
        eprintln!("{:6.2} {:6.2} {}", u.start as f64 / 16000.0, u.end as f64 / 16000.0, u.text);
    }
    assert!(!out.is_empty());
    let text: String = out.iter().map(|u| u.text.as_str()).collect();
    assert!(text.chars().count() > 50, "too little text: {text}");
}
