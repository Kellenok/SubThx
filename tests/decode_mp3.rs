use std::path::Path;
use std::time::Instant;
use subthx::audio::decode_16k;

#[test]
fn mp3_decode_matches_ffmpeg() {
    let Ok(src) = std::env::var("SUBTHX_MP3") else { return };
    let reference = std::env::var("SUBTHX_MP3_REF").unwrap();
    let t = Instant::now();
    let ours = decode_16k(Path::new(&src)).unwrap();
    let dt = t.elapsed().as_secs_f64();
    let raw = std::fs::read(&reference).unwrap();
    // ffmpeg -f s16le, like SubPlz
    let ff: Vec<f32> = raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect();
    let mut lags = Vec::new();
    for &w0 in &[16000 * 600usize, 16000 * 9000, 16000 * 18000] {
        let n = 16000 * 5;
        if w0 + n + 2000 > ours.len().min(ff.len()) { continue; }
        let a = &ours[w0..w0 + n];
        let mut best = (0i64, f64::MIN);
        for lag in -1500i64..=1500 {
            let b0 = (w0 as i64 + lag) as usize;
            let c: f64 = a.iter().zip(&ff[b0..b0 + n]).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
            if c > best.1 { best = (lag, c); }
        }
        lags.push(best.0);
    }
    let lag = lags[0];
    // sub-sample lag by parabolic interpolation of the correlation peak
    let w0 = 16000 * 600usize;
    let n = 16000 * 20;
    let corr = |l: i64| -> f64 {
        let b0 = (w0 as i64 + l) as usize;
        ours[w0..w0 + n].iter().zip(&ff[b0..b0 + n]).map(|(x, y)| (*x as f64) * (*y as f64)).sum()
    };
    let (cm, c0, cp) = (corr(lag - 1), corr(lag), corr(lag + 1));
    let frac = 0.5 * (cm - cp) / (cm - 2.0 * c0 + cp);
    eprintln!("fractional lag {:.3} samples @16k = {:.1} samples @44.1k", lag as f64 + frac, (lag as f64 + frac) * 44100.0 / 16000.0);
    let (mut sig, mut errs) = (0f64, 0f64);
    for i in 0..ours.len().min(ff.len()) {
        let j = i as i64 + lag;
        if j < 0 || j as usize >= ff.len() { continue; }
        let (x, y) = (ours[i] as f64, ff[j as usize] as f64);
        sig += y * y;
        errs += (x - y) * (x - y);
    }
    let win = 160000;
    let mut snrs: Vec<f64> = Vec::new();
    let mut zero_windows = 0;
    let (mut xy, mut xx) = (0f64, 0f64);
    for w in (0..ours.len().min(ff.len()) / win).map(|k| k * win) {
        let (mut s2, mut e2, mut o2) = (0f64, 0f64, 0f64);
        for i in w..w + win {
            let (x, y) = (ours[i] as f64, ff[i] as f64);
            s2 += y * y; e2 += (x - y) * (x - y); o2 += x * x; xy += x * y; xx += x * x;
        }
        if o2 < 1e-9 && s2 > 1e-6 { zero_windows += 1; }
        if s2 > 1e-3 { snrs.push(10.0 * (s2 / e2.max(1e-20)).log10()); }
    }
    snrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| snrs[((snrs.len() - 1) as f64 * p) as usize];
    eprintln!("gain {:.4}; per-10s SNR p5 {:.1} p50 {:.1} p95 {:.1} dB; windows where we output silence: {zero_windows}/{}", xy / xx, q(0.05), q(0.5), q(0.95), snrs.len());
    eprintln!("mp3 decode {:.2}s ({:.0}x), {} samples vs ffmpeg {}, lag {lags:?}, SNR {:.1} dB",
        dt, ours.len() as f64 / 16000.0 / dt, ours.len(), ff.len(), 10.0 * (sig / errs).log10());
}
