//! Decoded sound kept in a temporary file past a length must read back the
//! same as it does from memory.

use std::path::Path;
use subthx::audio::{decode_16k, decode_pcm_for, Pcm, Sink};

const BOOK: &str = "E:/audiobooks/Audiobook Collection/[二語 十] 探偵はもう、死んでいる。/test/[01] 探偵はもう、死んでいる。 [B0DQCZ9TNQ].m4b";

/// the same operations on a `Vec` and on a sink that spills
#[test]
fn sink_in_a_file_acts_like_a_vec() {
    let mem = Sink::with_limit(usize::MAX);
    let file = Sink::with_limit(10);
    let mut model: Vec<i16> = Vec::new();
    let data = |n: usize, seed: i16| (0..n).map(|i| (i as i16).wrapping_mul(7).wrapping_add(seed)).collect::<Vec<i16>>();
    for s in [&mem, &file] {
        s.extend(&data(7, 1)).unwrap();
        s.extend(&data(9, 2)).unwrap(); // crosses the limit
        s.resize(40).unwrap();
        s.write_at(30, &data(15, 3)).unwrap(); // past the end: grows
        s.truncate(33).unwrap();
        s.extend(&data(5, 4)).unwrap();
        s.resize(30).unwrap(); // shorter than it is: nothing
        s.resize(50).unwrap(); // silence after truncation must be silence
    }
    model.extend(data(7, 1));
    model.extend(data(9, 2));
    model.resize(40, 0);
    model.splice(30..30, std::iter::empty());
    model.resize(45, 0);
    model[30..45].copy_from_slice(&data(15, 3));
    model.truncate(33);
    model.extend(data(5, 4));
    model.resize(50, 0);
    let (m, f) = (mem.finish().unwrap(), file.finish().unwrap());
    assert_eq!(m.to_i16(), model);
    assert_eq!(f.to_i16(), model);
    assert_eq!(f.read(10, 20), model[10..20]);
    assert_eq!(f.read(45, 99), model[45..50]); // cut at the end
    assert_eq!(f.range(3, 6), Pcm::from_f32(&m.range(3, 6)).to_f32());
}

#[test]
fn a_long_book_in_a_file_is_the_book_in_memory() {
    let p = Path::new(BOOK);
    if !p.exists() {
        eprintln!("no book: skipped");
        return;
    }
    // 7.5 h: over the 30 minute limit, so it is in a file
    let spilled = decode_pcm_for(p, &[]).unwrap();
    let f = decode_16k(p).unwrap();
    assert_eq!(spilled.len(), f.len());
    let a = spilled.range(16000 * 3600, 16000 * 3600 + 1_000_000);
    assert_eq!(a, f[16000 * 3600..16000 * 3600 + 1_000_000]);
    assert_eq!(spilled.range(0, 5), f[..5]);
    assert_eq!(spilled.range(f.len() - 5, f.len() + 100), f[f.len() - 5..]);
}

#[test]
fn mp3_decodes_the_whole_file() {
    let (Ok(src), Ok(reference)) = (std::env::var("SUBTHX_MP3"), std::env::var("SUBTHX_MP3_REF")) else { return };
    let ours = decode_16k(Path::new(&src)).unwrap();
    let want = std::fs::metadata(&reference).unwrap().len() as usize / 2;
    assert!(ours.len().abs_diff(want) < 4000, "length differs");
}
