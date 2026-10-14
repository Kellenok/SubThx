use std::path::Path;
use std::time::Instant;
use subthx::mel::NemoMel;
use subthx::npy::read_f32;
use subthx::vad::{session, Device};

#[derive(serde::Deserialize)]
struct Segs { samples: Vec<(usize, usize)> }

#[test]
fn nemo_mel_matches_onnx_preprocessor() {
    let (_, audio) = read_f32(Path::new("fixtures/tantei16k.npy")).unwrap();
    let segs: Segs = serde_json::from_str(&std::fs::read_to_string("fixtures/vad/segments.json").unwrap()).unwrap();
    let mel = NemoMel::load(Path::new("models/nemo80_fbanks.npy")).unwrap();
    let mut pre = session(Path::new("models/nemo80.onnx"), Device::Cpu).unwrap();
    let (mut tn, mut to, mut maxd) = (0f64, 0f64, 0f32);
    let (mut sum, mut n, mut big) = (0f64, 0usize, 0usize);
    for batch in segs.samples.chunks(64).step_by(10) {
        let wavs: Vec<&[f32]> = batch.iter().map(|&(s, e)| &audio[s..e]).collect();
        let t = Instant::now();
        let (native, lens, _) = mel.batch(&wavs, 1);
        tn += t.elapsed().as_secs_f64();
        let max_len = wavs.iter().map(|w| w.len()).max().unwrap();
        let mut wav = vec![0f32; wavs.len() * max_len];
        for (k, w) in wavs.iter().enumerate() {
            wav[k * max_len..k * max_len + w.len()].copy_from_slice(w);
        }
        let wl: Vec<i64> = wavs.iter().map(|w| w.len() as i64).collect();
        let t = Instant::now();
        let out = pre
            .run(ort::inputs![
                "waveforms" => ort::value::TensorRef::from_array_view(([wavs.len(), max_len], &wav[..])).unwrap(),
                "waveforms_lens" => ort::value::TensorRef::from_array_view(([wavs.len()], &wl[..])).unwrap()
            ])
            .unwrap();
        to += t.elapsed().as_secs_f64();
        let (_, reference) = out["features"].try_extract_tensor::<f32>().unwrap();
        let (_, rl) = out["features_lens"].try_extract_tensor::<i64>().unwrap();
        assert_eq!(&lens[..], rl);
        assert_eq!(native.len(), reference.len());
        for (a, b) in native.iter().zip(reference) {
            let d = (a - b).abs();
            maxd = maxd.max(d);
            sum += d as f64;
            n += 1;
            big += (d > 1e-3) as usize;
        }
    }
    eprintln!("native {tn:.2}s vs onnx(cpu) {to:.2}s on every 10th batch, max |diff| {maxd:.2e}, mean {:.2e}, >1e-3: {big}/{n}", sum / n as f64);
    assert!(maxd < 0.1);
}
