//! Decoding of every container/codec we take, against ffmpeg's decode of the
//! same file (what SubPlz works from). Needs ffmpeg on PATH and test files:
//! SUBTHX_FORMATS_DIR with a.* / v.* files (the same 20 minutes of speech).

use std::path::PathBuf;

/// (lag, SNR in dB at the best gain) of `got` against `want` on one window
fn window(want: &[f32], got: &[f32], w0: usize, len: usize) -> (i64, f64) {
    let dot = |b0: usize| {
        let (mut xy, mut yy) = (0f64, 0f64);
        for i in 0..len {
            xy += want[w0 + i] as f64 * got[b0 + i] as f64;
            yy += (got[b0 + i] as f64).powi(2);
        }
        (xy, yy)
    };
    let mut best = (0i64, f64::MIN);
    for lag in -800i64..=800 {
        let b0 = w0 as i64 + lag;
        if b0 < 0 || b0 as usize + len > got.len() {
            continue;
        }
        let (xy, yy) = dot(b0 as usize);
        let c = xy / yy.sqrt().max(1e-12);
        if c > best.1 {
            best = (lag, c);
        }
    }
    let b0 = (w0 as i64 + best.0) as usize;
    let (xy, yy) = dot(b0);
    // ffmpeg's stereo downmix is -3 dB, ours the mean: compare at the best gain
    let g = xy / yy.max(1e-12);
    let (mut sig, mut err) = (0f64, 0f64);
    for i in 0..len {
        let (x, y) = (want[w0 + i] as f64, g * got[b0 + i] as f64);
        sig += x * x;
        err += (x - y) * (x - y);
    }
    (best.0, 10.0 * (sig / err.max(1e-12)).log10())
}

#[test]
fn formats_decode_like_ffmpeg() {
    let Some(dir) = std::env::var_os("SUBTHX_FORMATS_DIR").map(PathBuf::from) else {
        eprintln!("SUBTHX_FORMATS_DIR not set, skipping");
        return;
    };
    let mut names: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).collect();
    names.sort();
    let mut failed = Vec::new();
    for p in names {
        let n = p.file_name().unwrap().to_string_lossy().into_owned();
        if !(n.starts_with("a.") || n.starts_with("a_") || n.starts_with("v.")) {
            continue;
        }
        let out = std::process::Command::new("ffmpeg")
            .args(["-nostdin", "-loglevel", "error", "-i"])
            .arg(&p)
            .args(["-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"])
            .output()
            .unwrap();
        let want: Vec<f32> = out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let t = std::time::Instant::now();
        let got = match subthx::audio::decode_16k(&p) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{n:12} ERROR {e}");
                failed.push(n);
                continue;
            }
        };
        let dt = t.elapsed().as_secs_f64();
        // windows across the file, on both sides of chunk and piece borders
        let ws: Vec<(i64, f64)> = [7usize, 29, 31, 150, 299, 301, 450, 599, 601, 899, 901, 1150]
            .iter()
            .map(|&s| window(&want, &got, 16000 * s, 16000))
            .collect();
        let max_lag = ws.iter().map(|w| w.0.abs()).max().unwrap();
        let min_snr = ws.iter().map(|w| w.1).fold(f64::MAX, f64::min);
        let dlen = got.len() as i64 - want.len() as i64;
        eprintln!(
            "{n:12} {dt:5.2}s {:5.0}x  len {dlen:+5}  lags {:?}  min SNR {min_snr:4.1} dB",
            1200.0 / dt,
            ws.iter().map(|w| w.0).collect::<Vec<_>>()
        );
        // within 1 ms everywhere and the same sound. Lossy decoders differ a
        // little, and pieces ffmpeg decodes from a seek sit a fraction of a
        // sample off (a few tens of µs), which costs SNR but no timing
        if max_lag > 16 || min_snr < 10.0 || dlen.abs() > 16000 {
            failed.push(n);
        }
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}
