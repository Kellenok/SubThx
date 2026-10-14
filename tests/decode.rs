use std::path::Path;
use std::time::Instant;
use subthx::audio::decode_16k;
use subthx::npy::read_f32;
use subthx::vad::{decision_to_segment, Cmvn, Device, Fbank, Model, PostConfig};

const BOOK: &str = "E:/audiobooks/Audiobook Collection/[二語 十] 探偵はもう、死んでいる。/test/[01] 探偵はもう、死んでいる。 [B0DQCZ9TNQ].m4b";

#[test]
fn decode_matches_ffmpeg_closely() {
    let t = Instant::now();
    let ours = decode_16k(Path::new(BOOK)).unwrap();
    let dt = t.elapsed();
    eprintln!("resample CPU time (all threads): {:.2}s, threads {}", subthx::audio::RESAMPLE_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9, rayon::current_num_threads());
    let (_, ff) = read_f32(Path::new("fixtures/tantei16k.npy")).unwrap();
    eprintln!("decode: {:.2}s, {} samples (ffmpeg {}), {:.0}x realtime", dt.as_secs_f64(), ours.len(), ff.len(), ours.len() as f64 / 16000.0 / dt.as_secs_f64());

    // best lag on a few windows
    let mut lags = Vec::new();
    for &w0 in &[16000 * 600usize, 16000 * 9000, 16000 * 20000] {
        let n = 16000 * 5;
        let a = &ours[w0..w0 + n];
        let mut best = (0i64, f64::MIN);
        for lag in -400i64..=400 {
            let b0 = (w0 as i64 + lag) as usize;
            let c: f64 = a.iter().zip(&ff[b0..b0 + n]).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
            if c > best.1 {
                best = (lag, c);
            }
        }
        lags.push(best.0);
    }
    let lag = lags[0];
    // SNR + gain over the whole aligned overlap
    let (mut sig, mut err, mut xy, mut xx) = (0f64, 0f64, 0f64, 0f64);
    for i in 0..ours.len().min(ff.len()) {
        let j = i as i64 + lag;
        if j < 0 || j as usize >= ff.len() {
            continue;
        }
        let (x, y) = (ours[i] as f64, ff[j as usize] as f64);
        sig += y * y;
        err += (x - y) * (x - y);
        xy += x * y;
        xx += x * x;
    }
    eprintln!("lag (ours vs ffmpeg, samples) {lags:?}, gain {:.4}, SNR {:.1} dB", xy / xx, 10.0 * (sig / err).log10());

    // VAD on our decode vs python reference segments
    #[derive(serde::Deserialize)]
    struct Want { seconds: Vec<(f64, f64)> }
    let want: Want = serde_json::from_str(&std::fs::read_to_string("fixtures/vad/segments.json").unwrap()).unwrap();
    let cmvn = Cmvn::load(Path::new("models/firered_vad_cmvn.json")).unwrap();
    let feats = Fbank::new().compute(&ours, &cmvn);
    let mut model = Model::load(Path::new("models/firered_vad.onnx"), Device::DirectMl).unwrap();
    let segs = decision_to_segment(&PostConfig::default().process(&model.probs(&feats).unwrap()), Some(ours.len() as f64 / 16000.0));
    let ws: std::collections::HashSet<_> = want.seconds.iter().map(|&(s, e)| ((s * 1000.0).round() as i64, (e * 1000.0).round() as i64)).collect();
    let same = segs.iter().filter(|&&(s, e)| ws.contains(&((s * 1000.0).round() as i64, (e * 1000.0).round() as i64))).count();
    let near = segs.iter().filter(|&&(s, e)| want.seconds.iter().any(|&(a, b)| (a - s).abs() <= 0.02 && (b - e).abs() <= 0.02)).count();
    eprintln!("VAD on our decode: {} segments (ffmpeg decode {}), identical {same}, within 20 ms {near}", segs.len(), want.seconds.len());
}
