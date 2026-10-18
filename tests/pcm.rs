//! Decoding straight to 16-bit samples must give what quantizing the f32
//! decoding gives, on both the built-in decoder and ffmpeg (which cuts a long
//! file into pieces and splices them).

use std::path::Path;
use subthx::audio::{decode_16k, decode_pcm_for, Pcm};

const BOOK: &str = "E:/audiobooks/Audiobook Collection/[二語 十] 探偵はもう、死んでいる。/test/[01] 探偵はもう、死んでいる。 [B0DQCZ9TNQ].m4b";

#[test]
fn native_decoder_pcm_is_the_quantized_f32() {
    let p = Path::new(BOOK);
    if !p.exists() {
        eprintln!("no book: skipped");
        return;
    }
    let f = decode_16k(p).unwrap();
    let want = Pcm::from_f32(&f);
    let got = decode_pcm_for(p, &[]).unwrap();
    assert_eq!(got.len(), want.len());
    assert!(got.0 == want.0, "native pcm differs from quantized f32");
}

#[test]
fn ffmpeg_pcm_is_the_quantized_f32() {
    let p = Path::new(BOOK);
    if !p.exists() || subthx::ffmpeg::ffmpeg().is_none() {
        eprintln!("no book or ffmpeg: skipped");
        return;
    }
    let f: Vec<f32> = subthx::ffmpeg::decode_16k(p, &[]).unwrap();
    let i: Vec<i16> = subthx::ffmpeg::decode_16k(p, &[]).unwrap();
    assert_eq!(f.len(), i.len());
    let (mut max, mut diff) = (0i32, 0usize);
    for (a, b) in Pcm::from_f32(&f).0.iter().zip(&i) {
        let d = (*a as i32 - *b as i32).abs();
        max = max.max(d);
        diff += (d != 0) as usize;
    }
    eprintln!("ffmpeg: {} samples, {diff} differ, largest difference {max}", i.len());
    let q = Pcm::from_f32(&f).0;
    let bad: Vec<usize> = (0..i.len()).filter(|&k| (q[k] as i32 - i[k] as i32).abs() > 1).collect();
    eprintln!("first bad {:?}, count {}", &bad[..bad.len().min(8)], bad.len());
    if let Some(&k) = bad.first() {
        eprintln!("around {k}: f32->i16 {:?}, s16 {:?}", &q[k.saturating_sub(3)..k + 6], &i[k.saturating_sub(3)..k + 6]);
        for lag in -3i64..=3 {
            let m = (0..2000).filter(|&j| (q[(k as i64 + j as i64 + lag) as usize] as i32 - i[k + j] as i32).abs() <= 1).count();
            eprintln!("lag {lag}: {m}/2000 close");
        }
    }
    assert!(max <= 1, "ffmpeg's s16 and the quantized f32 are more than a step apart");
}

#[test]
fn mp3_decodes_the_whole_file() {
    let (Ok(src), Ok(reference)) = (std::env::var("SUBTHX_MP3"), std::env::var("SUBTHX_MP3_REF")) else { return };
    let ours = decode_16k(Path::new(&src)).unwrap();
    let want = std::fs::metadata(&reference).unwrap().len() as usize / 2;
    eprintln!("mp3: {} samples, ffmpeg {}", ours.len(), want);
    assert!(ours.len().abs_diff(want) < 4000, "length differs");
    let pcm = decode_pcm_for(Path::new(&src), &[]).unwrap();
    assert!(pcm.0 == Pcm::from_f32(&ours).0, "mp3 pcm differs from quantized f32");
}
