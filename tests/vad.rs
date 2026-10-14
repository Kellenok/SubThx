use std::path::Path;
use std::time::Instant;
use subthx::npy::read_f32;
use subthx::vad::{decision_to_segment, Cmvn, Device, Fbank, Model, PostConfig};

#[derive(serde::Deserialize)]
struct Want {
    n_samples: usize,
    decisions_sum: usize,
    seconds: Vec<(f64, f64)>,
}

fn seg_diff(a: &[(f64, f64)], b: &[(f64, f64)]) -> usize {
    let sa: std::collections::HashSet<_> = a.iter().map(|&(s, e)| ((s * 1000.0) as i64, (e * 1000.0) as i64)).collect();
    b.iter().filter(|&&(s, e)| !sa.contains(&((s * 1000.0) as i64, (e * 1000.0) as i64))).count()
}

#[test]
fn firered_vad_matches_python() {
    let (_, audio) = read_f32(Path::new("fixtures/tantei16k.npy")).unwrap();
    let (hshape, head) = read_f32(Path::new("fixtures/vad/fbank_head.npy")).unwrap();
    let (_, want_probs) = read_f32(Path::new("fixtures/vad/probs.npy")).unwrap();
    let want: Want = serde_json::from_str(&std::fs::read_to_string("fixtures/vad/segments.json").unwrap()).unwrap();
    assert_eq!(want.n_samples, audio.len());
    let dur = audio.len() as f64 / 16000.0;

    let cmvn = Cmvn::load(Path::new("models/firered_vad_cmvn.json")).unwrap();
    let fb = Fbank::new();
    let t = Instant::now();
    let feats = fb.compute(&audio, &cmvn);
    let t_fbank = t.elapsed();
    let n = hshape[0] * hshape[1];
    let fdiff = feats[..n].iter().zip(&head).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("fbank: {:.2}s for {} frames, max|rust-torch| over first {} frames = {fdiff:.2e}", t_fbank.as_secs_f64(), feats.len() / 80, hshape[0]);

    let post = PostConfig::default();
    let t = Instant::now();
    let dec = post.process(&want_probs);
    let segs = decision_to_segment(&dec, Some(dur));
    let t_post = t.elapsed();
    eprintln!(
        "postprocess (python probs): {:.3}s, {} segments (want {}), decisions sum {} (want {}), segments not in python: {}",
        t_post.as_secs_f64(), segs.len(), want.seconds.len(), dec.iter().map(|&x| x as usize).sum::<usize>(), want.decisions_sum, seg_diff(&want.seconds, &segs)
    );
    assert_eq!(segs, want.seconds);

    let devices = std::env::var("SUBTHX_DEVICES").unwrap_or_else(|_| "cpu".into());
    for dev in devices.split(',') {
        let device: Device = dev.parse().unwrap();
        let mut model = Model::load(Path::new(&std::env::var("SUBTHX_VAD_MODEL").unwrap_or_else(|_| "models/firered_vad.onnx".into())), device).unwrap();
        model.probs(&feats[..30000 * 80]).unwrap(); // warmup
        let t = Instant::now();
        let probs = model.probs(&feats).unwrap();
        let t_model = t.elapsed();
        let pdiff = probs.iter().zip(&want_probs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let segs2 = decision_to_segment(&post.process(&probs), Some(dur));
        eprintln!(
            "model ({device:?}): {:.2}s, max|p-torch| {pdiff:.2e}; end-to-end {} segments, {} differ from python | VAD total {:.2}s",
            t_model.as_secs_f64(), segs2.len(), seg_diff(&want.seconds, &segs2), (t_fbank + t_model + t_post).as_secs_f64()
        );
    }
}
