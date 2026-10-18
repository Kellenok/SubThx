//! The VAD run on features made a group of chunks at a time must give the
//! same probabilities as the whole file's features (the CPU, a real model).

use subthx::audio::Pcm;
use subthx::vad::{Cmvn, Device, Fbank, Model};

fn models() -> Option<std::path::PathBuf> {
    let dir = std::env::var_os("SUBTHX_MODELS").map(std::path::PathBuf::from).unwrap_or_else(|| "models".into());
    dir.join("firered_vad.onnx").exists().then_some(dir)
}

/// speech-like bursts over a hum: a few minutes' worth of structure that is
/// not silence, so the model has something to say
fn audio(seconds: usize) -> Vec<f32> {
    let mut x = 12345u32;
    (0..seconds * 16000)
        .map(|i| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (x >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let t = i as f32 / 16000.0;
            let burst = ((t * 0.7).sin() * 3.0).clamp(0.0, 1.0);
            0.02 * noise + burst * 0.3 * (t * 220.0 * std::f32::consts::TAU).sin() * (1.0 + 0.5 * (t * 5.0).sin())
        })
        .collect()
}

#[test]
fn blocks_give_the_whole_files_probs() {
    let Some(dir) = models() else {
        eprintln!("no models: skipped");
        return;
    };
    let cmvn = Cmvn::load(&dir.join("firered_vad_cmvn.json")).unwrap();
    let fbank = Fbank::new();
    let mut model = Model::load(&dir.join("firered_vad.onnx"), Device::Cpu).unwrap();
    // two full 30000-frame chunks and a rest
    let pcm = Pcm::from_f32(&audio(13 * 60 + 7));
    let whole = model.probs(&fbank.compute(&pcm.to_f32(), &cmvn)).unwrap();
    let blocks = model.probs_audio(&pcm, &fbank, &cmvn).unwrap();
    assert_eq!(whole.len(), blocks.len());
    assert!(whole.iter().zip(&blocks).all(|(a, b)| a.to_bits() == b.to_bits()), "probabilities differ");
    assert!(whole.iter().any(|&p| p > 0.5) && whole.iter().any(|&p| p < 0.5), "the test audio says nothing");
}
