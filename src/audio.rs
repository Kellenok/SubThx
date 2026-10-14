//! Audio decoding to 16 kHz mono f32 (ffmpeg only as a last resort).
//!
//! MP4/M4B AAC: access units come straight from the MP4 sample tables and
//! are decoded with symphonia in parallel chunks (one packet of pre-roll,
//! enough for AAC-LC overlap-add), downmixed to mono and resampled with a
//! polyphase windowed-sinc filter. Chunks start on multiples of 441 packets
//! so 44.1 kHz -> 16 kHz output positions are integers and chunking does not
//! change the result.

use rayon::prelude::*;
use std::path::Path;
use symphonia::core::codecs::audio::well_known::CODEC_ID_AAC;
use symphonia::core::codecs::audio::{AudioCodecParameters, AudioDecoder, AudioDecoderOptions};
use symphonia::core::packet::Packet;

pub const OUT_RATE: u32 = 16000;
/// cumulative CPU time spent resampling (all threads), for profiling
pub static RESAMPLE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const AAC_FRAME: usize = 1024;

/// Rational resampler in -> out with a precomputed polyphase table.
pub struct Resampler {
    /// in/out reduced: output j sits at input position j * up_in / down_out
    num: usize,
    den: usize,
    half: usize,
    /// den phases x (2*half) taps
    table: Vec<f32>,
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let y = x * x / 4.0;
    for k in 1..50 {
        term *= y / (k as f64 * k as f64);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl Resampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Resampler {
        let g = gcd(in_rate as usize, out_rate as usize);
        let (num, den) = (in_rate as usize / g, out_rate as usize / g);
        // low-pass at 0.97 of the output Nyquist, Kaiser beta 9 (ffmpeg swr defaults)
        let ratio = (out_rate as f64 / in_rate as f64).min(1.0);
        let fc = 0.97 * ratio * 0.5; // cycles per input sample
        let half = (16.0 / ratio).ceil() as usize;
        let beta = 9.0;
        let i0b = bessel_i0(beta);
        let mut table = vec![0f32; den * 2 * half];
        for ph in 0..den {
            let frac = ph as f64 / den as f64;
            let row = &mut table[ph * 2 * half..(ph + 1) * 2 * half];
            let mut sum = 0.0;
            let mut tmp = vec![0f64; 2 * half];
            for (k, v) in tmp.iter_mut().enumerate() {
                // input index base - half + 1 + k, offset from the exact position
                let t = (k as f64 - half as f64 + 1.0) - frac;
                let x = 2.0 * fc * t;
                let sinc = if x.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
                let w = t / half as f64;
                let win = if w.abs() >= 1.0 { 0.0 } else { bessel_i0(beta * (1.0 - w * w).sqrt()) / i0b };
                *v = sinc * win;
                sum += *v;
            }
            for (r, v) in row.iter_mut().zip(&tmp) {
                *r = (v / sum) as f32;
            }
        }
        Resampler { num, den, half, table }
    }

    pub fn out_len(&self, in_len: usize) -> usize {
        (in_len * self.den).div_ceil(self.num)
    }

    /// Output samples `out_start..out_start+out.len()` given input `x` that
    /// starts at absolute input index `x_start` (missing samples are zero).
    pub fn process(&self, x: &[f32], x_start: usize, out_start: usize, out: &mut [f32]) {
        let taps = 2 * self.half;
        for (o, y) in out.iter_mut().enumerate() {
            let j = out_start + o;
            let pos = j * self.num;
            let base = pos / self.den;
            let ph = pos % self.den;
            let h = &self.table[ph * taps..(ph + 1) * taps];
            let first = base as isize - self.half as isize + 1 - x_start as isize;
            let mut acc = 0f32;
            if first >= 0 && first as usize + taps <= x.len() {
                acc = dot(&x[first as usize..first as usize + taps], h);
            } else {
                for k in 0..taps {
                    let i = first + k as isize;
                    if i >= 0 && (i as usize) < x.len() {
                        acc += x[i as usize] * h[k];
                    }
                }
            }
            *y = acc;
        }
    }
}

/// Dot product with 8 independent partial sums so it vectorises.
#[inline(always)]
fn dot_lanes(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..8 {
            acc[l] = x[l].mul_add(y[l], acc[l]);
        }
    }
    let mut s = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    dot_lanes(a, b)
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *AVX2.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            // SAFETY: the CPU supports AVX2 and FMA (checked above)
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_lanes(a, b)
}

/// Everything else symphonia reads: FLAC, WAV, AIFF, CAF, Ogg Vorbis, ALAC,
/// A file read on its own thread in big sequential blocks, up to
/// `AHEAD` blocks ahead of the reader. The demuxer's small reads come from
/// memory and the disk keeps streaming while a batch decodes: a film's
/// video is most of the file, and a hard disk read in the gaps between
/// batches ran at half its speed.
struct ReadAhead {
    path: std::path::PathBuf,
    len: u64,
    rx: std::sync::Mutex<std::sync::mpsc::Receiver<std::io::Result<Vec<u8>>>>,
    block: Vec<u8>,
    /// read position in `block`
    off: usize,
    /// file position of `block[0]`
    start: u64,
}

const AHEAD_BLOCK: usize = 4 << 20;
const AHEAD: usize = 16;

impl ReadAhead {
    fn open(path: &Path) -> std::io::Result<ReadAhead> {
        let len = std::fs::metadata(path)?.len();
        let rx = Self::spawn(path, 0)?;
        Ok(ReadAhead { path: path.to_path_buf(), len, rx: std::sync::Mutex::new(rx), block: Vec::new(), off: 0, start: 0 })
    }

    fn spawn(path: &Path, from: u64) -> std::io::Result<std::sync::mpsc::Receiver<std::io::Result<Vec<u8>>>> {
        use std::io::{Read, Seek};
        let mut f = std::fs::File::open(path)?;
        f.seek(std::io::SeekFrom::Start(from))?;
        let (tx, rx) = std::sync::mpsc::sync_channel(AHEAD);
        std::thread::spawn(move || loop {
            let mut buf = vec![0u8; AHEAD_BLOCK];
            let mut n = 0;
            while n < buf.len() {
                match f.read(&mut buf[n..]) {
                    Ok(0) => break,
                    Ok(k) => n += k,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                }
            }
            buf.truncate(n);
            // a dropped reader (seeked away or done) stops the thread
            if tx.send(Ok(buf)).is_err() || n == 0 {
                return;
            }
        });
        Ok(rx)
    }
}

impl std::io::Read for ReadAhead {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.off == self.block.len() {
            let next = self.rx.get_mut().unwrap().recv().unwrap_or_else(|_| Ok(Vec::new()))?;
            self.start += self.block.len() as u64;
            self.block = next;
            self.off = 0;
            if self.block.is_empty() {
                return Ok(0);
            }
        }
        let n = out.len().min(self.block.len() - self.off);
        out[..n].copy_from_slice(&self.block[self.off..self.off + n]);
        self.off += n;
        Ok(n)
    }
}

impl std::io::Seek for ReadAhead {
    fn seek(&mut self, to: std::io::SeekFrom) -> std::io::Result<u64> {
        let here = self.start + self.off as u64;
        let target = match to {
            std::io::SeekFrom::Start(p) => p as i128,
            std::io::SeekFrom::Current(d) => here as i128 + d as i128,
            std::io::SeekFrom::End(d) => self.len as i128 + d as i128,
        };
        if target < 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek before the start"));
        }
        let target = target as u64;
        if target >= self.start && target <= self.start + self.block.len() as u64 {
            self.off = (target - self.start) as usize;
        } else if target > here && target - here <= (AHEAD * AHEAD_BLOCK) as u64 {
            // a short skip forward: through the blocks already coming
            let mut skip = target - here;
            while skip > 0 {
                let left = (self.block.len() - self.off) as u64;
                if skip <= left {
                    self.off += skip as usize;
                    break;
                }
                skip -= left;
                self.off = self.block.len();
                let mut b = [0u8; 1];
                if std::io::Read::read(self, &mut b)? == 0 {
                    break;
                }
                skip -= 1;
            }
        } else {
            // anywhere else (an index at the end of the file): read from there
            *self.rx.get_mut().unwrap() = Self::spawn(&self.path, target)?;
            self.block.clear();
            self.off = 0;
            self.start = target;
        }
        Ok(self.start + self.off as u64)
    }
}

impl symphonia::core::io::MediaSource for ReadAhead {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.len)
    }
}

/// MKV/WebM, MP4/MOV without AAC. Packets are demuxed in batches (a WAV of a
/// whole audiobook does not fit in memory twice); each batch is decoded in
/// parallel chunks with pre-roll and placed by the packets' timestamps.
fn decode_any(path: &Path, langs: &[&str]) -> Result<Vec<f32>, String> {
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;

    let file = ReadAhead::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut hint = Hint::new();
    if let Some(e) = path.extension() {
        hint.with_extension(&e.to_string_lossy());
    }
    let mut reader = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(|e| e.to_string())?;
    let audio: Vec<&symphonia::core::formats::Track> = reader.tracks().iter().filter(|t| t.codec_params.as_ref().and_then(|p| p.audio()).is_some()).collect();
    let tags: Vec<(Option<&str>, bool)> =
        audio.iter().map(|t| (t.language.as_deref(), t.flags.contains(symphonia::core::formats::TrackFlags::DEFAULT))).collect();
    let track = (*audio.get(pick_track(&tags, langs)).ok_or("no audio track")?).clone();
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).unwrap().clone();
    let rate = params.sample_rate.ok_or("unknown sample rate")?;
    // fail early (and fall back to ffmpeg) when there is no decoder for it
    symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| e.to_string())?;
    // timestamps -> sample positions
    let (tb_num, tb_den) = match track.time_base {
        Some(tb) => (tb.numer.get() as i128, tb.denom.get() as i128),
        None => (1, rate as i128),
    };
    // timestamp 0 is time 0, as for ffmpeg (packets before it are pre-skip)
    let to_samples = |ts: i128| (ts * tb_num * rate as i128 + tb_den / 2).div_euclid(tb_den) as i64;
    // a tick longer than a sample (MKV counts milliseconds): rounded
    // durations would drift, so positions come from the timestamps and each
    // batch decodes as one piece
    let coarse = tb_num * rate as i128 > tb_den;

    let rs = Resampler::new(rate, OUT_RATE);
    let out_pos = |s: i64| (s.max(0) as usize * rs.den).div_ceil(rs.num);
    const PRE: usize = 8;
    const POST: usize = 4;
    let batch_samples = rate as i64 * 30 * 64;
    let chunk_samples = if coarse { batch_samples } else { rate as i64 * 30 };

    struct Pkt {
        pos: i64,
        packet: Packet,
    }
    let mut out: Vec<f32> = Vec::new();
    let mut pending: Vec<Pkt> = Vec::new();
    // index in `pending` of the first packet not yet output (earlier ones are pre-roll)
    let mut first = 0usize;
    let mut eof = false;
    // extra packets to read past a full batch (a chunk still waiting for its post-roll)
    let mut more = 0usize;
    let mut last_pos: Option<(i64, i64)> = None; // (pos, dur in samples) of the previous packet
    while !eof || first < pending.len() {
        // read until the batch covers enough audio
        while !eof && (more > 0 || pending.last().map_or(0, |p| p.pos) - pending.get(first).map_or(0, |p| p.pos) < batch_samples) {
            more = more.saturating_sub(1);
            match reader.next_packet() {
                Ok(Some(p)) if p.track_id == track.id => {
                    let dur = if coarse { 0 } else { to_samples(p.dur.get() as i128) };
                    let by_ts = to_samples(p.pts.get() as i128);
                    // positions add up packet durations (what the decoder puts
                    // out); timestamps only start the count and open real gaps
                    let pos = match last_pos {
                        Some((lp, ld)) if ld > 0 && by_ts <= lp + ld + rate as i64 / 10 => lp + ld,
                        Some((lp, _)) if by_ts <= lp => lp + 1,
                        _ => by_ts,
                    };
                    last_pos = Some((pos, dur));
                    pending.push(Pkt { pos, packet: p });
                }
                Ok(Some(_)) => {}
                Ok(None) => eof = true,
                Err(symphonia::core::errors::Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => eof = true,
                Err(e) => return Err(e.to_string()),
            }
        }
        if first >= pending.len() {
            break;
        }
        // chunks [p0, p1) of ~30 s; without EOF the last ones wait for their post-roll
        let mut bounds = vec![first];
        for i in first + 1..pending.len() {
            if pending[i].pos - pending[*bounds.last().unwrap()].pos >= chunk_samples {
                bounds.push(i);
            }
        }
        bounds.push(pending.len());
        let mut chunks: Vec<(usize, usize)> = bounds.windows(2).map(|w| (w[0], w[1])).collect();
        if !eof {
            while chunks.last().is_some_and(|&(_, p1)| p1 + POST > pending.len()) {
                chunks.pop();
            }
            if chunks.is_empty() {
                more = POST + 1;
                continue;
            }
        }
        let n = pending.len();
        let last_at_eof = eof && chunks.last().is_some_and(|&(_, p1)| p1 == n);
        let pieces: Vec<Vec<f32>> = chunks
            .par_iter()
            .enumerate()
            .map(|(k, &(p0, p1))| -> Result<Vec<f32>, String> {
                let d0 = p0.saturating_sub(PRE);
                let d1 = (p1 + POST).min(n);
                let mut dec = symphonia::default::get_codecs()
                    .make_audio_decoder(&params, &AudioDecoderOptions::default())
                    .map_err(|e| e.to_string())?;
                // decoded audio is laid end to end (container timestamps can be
                // coarse, MKV's are milliseconds); timestamps only anchor the
                // chunk and open gaps
                let mut mono: Vec<f32> = Vec::new();
                let mut inter: Vec<f32> = Vec::new();
                // the chunk is anchored at its first packet that puts out audio
                let mut anchor: Option<(i64, usize)> = None;
                let mut tl = pending[d0].pos; // timeline position of the end of `mono`
                for (idx, pk) in pending.iter().enumerate().take(d1).skip(d0) {
                    let gap = pk.pos - tl;
                    if gap > rate as i64 / 10 {
                        mono.resize(mono.len() + gap as usize, 0.0);
                        tl += gap;
                    }
                    let Ok(buf) = dec.decode(&pk.packet) else { continue };
                    let ch = buf.num_planes().max(1);
                    buf.copy_to_vec_interleaved(&mut inter);
                    let frames = inter.len() / ch;
                    let t0 = (pk.packet.trim_start.get() as usize).min(frames);
                    let t1 = frames.saturating_sub(pk.packet.trim_end.get() as usize).max(t0);
                    if idx >= p0 && anchor.is_none() && t1 > t0 {
                        anchor = Some((pk.pos, mono.len()));
                    }
                    mono.extend((t0..t1).map(|f| inter[f * ch..f * ch + ch].iter().sum::<f32>() / ch as f32));
                    tl += (t1 - t0) as i64;
                }
                let (apos, at) = anchor.unwrap_or((pending[p0].pos, 0));
                let x0 = apos - at as i64;
                let end = x0 + mono.len() as i64;
                let o0 = out_pos(pending[p0].pos);
                let o1 = if last_at_eof && k + 1 == chunks.len() { out_pos(end) } else { out_pos(pending[p1].pos) };
                let mut dst = vec![0f32; o1.saturating_sub(o0)];
                if x0 >= 0 {
                    rs.process(&mono, x0 as usize, o0, &mut dst);
                } else {
                    let skip = (-x0) as usize;
                    rs.process(&mono[skip.min(mono.len())..], 0, o0, &mut dst);
                }
                Ok(dst)
            })
            .collect::<Result<_, String>>()?;
        // place by position: a gap in the timestamps stays silent
        for (piece, &(p0, _)) in pieces.into_iter().zip(&chunks) {
            let o0 = out_pos(pending[p0].pos);
            if out.len() < o0 {
                out.resize(o0, 0.0);
            }
            out.truncate(o0);
            out.extend_from_slice(&piece);
        }
        let done = chunks.last().unwrap().1;
        // keep the pre-roll of what comes next
        let keep_from = done.saturating_sub(PRE);
        pending.drain(..keep_from);
        first = done - keep_from;
        if eof && done == n {
            break;
        }
    }
    if out.is_empty() {
        return Err("no audio decoded".into());
    }
    Ok(out)
}

/// ISO 639-2 codes (both forms) of the ISO 639-1 languages we transcribe.
const ISO_639_2: &[(&str, &[&str])] = &[
    ("ja", &["jpn", "jp"]),
    ("en", &["eng"]),
    ("zh", &["zho", "chi"]),
    ("ko", &["kor"]),
    ("bg", &["bul"]),
    ("cs", &["ces", "cze"]),
    ("da", &["dan"]),
    ("de", &["deu", "ger"]),
    ("el", &["ell", "gre"]),
    ("es", &["spa"]),
    ("et", &["est"]),
    ("fi", &["fin"]),
    ("fr", &["fra", "fre"]),
    ("hr", &["hrv"]),
    ("hu", &["hun"]),
    ("it", &["ita"]),
    ("lt", &["lit"]),
    ("lv", &["lav"]),
    ("mt", &["mlt"]),
    ("nl", &["nld", "dut"]),
    ("pl", &["pol"]),
    ("pt", &["por"]),
    ("ro", &["ron", "rum"]),
    ("ru", &["rus"]),
    ("sk", &["slk", "slo"]),
    ("sl", &["slv"]),
    ("sv", &["swe"]),
    ("uk", &["ukr"]),
];

/// The ISO 639-1 code of a language tag ("jpn", "ja", "ja-JP", "jp"), for
/// the languages we know.
pub fn lang_code(tag: &str) -> Option<&'static str> {
    let primary = tag.split(['-', '_']).next().unwrap_or("").to_ascii_lowercase();
    ISO_639_2.iter().find(|(a, codes)| *a == primary || codes.contains(&primary.as_str())).map(|(a, _)| *a)
}

/// Whether a track's language tag ("jpn", "ja", "ja-JP") is `iso1` ("ja").
pub fn lang_is(tag: &str, iso1: &str) -> bool {
    let primary = tag.split(['-', '_']).next().unwrap_or("").to_ascii_lowercase();
    primary == iso1 || ISO_639_2.iter().any(|(a, codes)| *a == iso1 && codes.contains(&primary.as_str()))
}

/// The audio track to transcribe among (language tag, default?) in file
/// order: the default track unless it is tagged with a language the model
/// does not speak (the dub that dual-audio anime releases put first), then
/// the first track in one of `langs`. Untagged ("und") tracks fit anything.
pub fn pick_track(tracks: &[(Option<&str>, bool)], langs: &[&str]) -> usize {
    let spoken = |l: &str| langs.iter().any(|w| lang_is(l, w));
    fn tagged<'a>(t: &(Option<&'a str>, bool)) -> Option<&'a str> {
        t.0.filter(|l| !l.is_empty() && !l.eq_ignore_ascii_case("und"))
    }
    let default = tracks.iter().position(|t| t.1).unwrap_or(0);
    match tracks.get(default) {
        Some(t) if langs.is_empty() || tagged(t).is_none_or(spoken) => default,
        _ => tracks.iter().position(|t| tagged(t).is_some_and(spoken)).unwrap_or(default),
    }
}

/// Decode an audio or video file to 16 kHz mono f32: our own fast paths for
/// AAC in MP4/M4B and for MP3, symphonia for the rest, ffmpeg (when
/// installed) for what symphonia cannot read.
pub fn decode_16k(path: &Path) -> Result<Vec<f32>, String> {
    decode_16k_for(path, &[])
}

/// [`decode_16k`] of the audio track in one of `langs` (ISO 639-1) when a
/// file has several (see [`pick_track`]).
pub fn decode_16k_for(path: &Path, langs: &[&str]) -> Result<Vec<f32>, String> {
    let mut head = [0u8; 8];
    {
        use std::io::Read;
        let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let n = f.read(&mut head).map_err(|e| e.to_string())?;
        if n < 8 {
            return Err(format!("{}: not an audio file", path.display()));
        }
    }
    // symphonia's WAV reader would take an AVI for a WAV: AVI goes to ffmpeg
    let avi = &head[..4] == b"RIFF" && {
        let mut h = [0u8; 12];
        std::fs::File::open(path).and_then(|mut f| std::io::Read::read_exact(&mut f, &mut h)).is_ok() && &h[8..12] == b"AVI "
    };
    let fast = if &head[4..8] == b"ftyp" {
        Some(decode_mp4_aac(path))
    } else if crate::media::is_mp3(&head) {
        Some(decode_mp3(path))
    } else {
        None
    };
    let first_err = match fast {
        Some(Ok(v)) => return Ok(v),
        Some(Err(e)) => e,
        None => String::new(),
    };
    let err = match if avi { Err("AVI needs ffmpeg".to_string()) } else { decode_any(path, langs) } {
        Ok(v) => return Ok(v),
        Err(e) if first_err.is_empty() => e,
        Err(e) => format!("{first_err}; {e}"),
    };
    if crate::ffmpeg::ffmpeg().is_some() {
        return crate::ffmpeg::decode_16k(path, langs).map_err(|e| format!("{}: {e}", path.display()));
    }
    Err(format!("{}: {err} (installing ffmpeg adds Opus, AC-3, WMA, AVI and more)", path.display()))
}

fn decode_mp4_aac(path: &Path) -> Result<Vec<f32>, String> {
    let info = crate::media::mp4_audio(path)?;
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    let n_packets = info.packets.len();
    // the edit list's priming is cut like ffmpeg does; input index i is
    // decoded sample i + skip
    let skip = info.skip.min(n_packets * AAC_FRAME);
    let in_len = n_packets * AAC_FRAME - skip;
    let rs = Resampler::new(info.sample_rate, OUT_RATE);
    let out_len = rs.out_len(in_len);
    let out_pos = |p: usize| ((p * AAC_FRAME).saturating_sub(skip) * rs.den).div_ceil(rs.num).min(out_len);
    // chunk boundaries must map to integer output positions
    let align = rs.num / gcd(rs.num, AAC_FRAME);
    let chunk_packets = align * 16;
    // AAC-LC output of packet n overlaps frame n-1; the resampler also reads
    // `half` samples before the chunk, so decode two packets of pre-roll
    let pre = 2usize;
    let post = rs.half.div_ceil(AAC_FRAME) + 1;

    let mut params = AudioCodecParameters::new();
    params
        .for_codec(CODEC_ID_AAC)
        .with_sample_rate(info.sample_rate)
        .with_extra_data(info.asc.clone().into_boxed_slice());

    let mut out = vec![0f32; out_len];
    let chunks: Vec<(usize, &mut [f32])> = {
        let mut v = Vec::new();
        let mut rest: &mut [f32] = &mut out;
        let mut p0 = 0;
        while p0 < n_packets {
            let p1 = (p0 + chunk_packets).min(n_packets);
            let o0 = out_pos(p0);
            let o1 = if p1 == n_packets { out_len } else { out_pos(p1) };
            let (a, b) = rest.split_at_mut(o1 - o0);
            v.push((p0, a));
            rest = b;
            p0 = p1;
        }
        v
    };

    chunks.into_par_iter().try_for_each(|(p0, dst)| -> Result<(), String> {
        let p1 = (p0 + chunk_packets).min(n_packets);
        let d0 = p0.saturating_sub(pre);
        let d1 = (p1 + post).min(n_packets);
        let mut dec = symphonia::default::codecs::AacDecoder::try_new(&params, &AudioDecoderOptions::default())
            .map_err(|e| e.to_string())?;
        let mut mono = Vec::with_capacity((d1 - d0) * AAC_FRAME);
        let mut inter: Vec<f32> = Vec::new();
        for p in d0..d1 {
            let (off, size) = info.packets[p];
            let bytes = &data[off as usize..off as usize + size as usize];
            let pkt = Packet::new(0, ((p * AAC_FRAME) as i64).into(), (AAC_FRAME as u64).into(), bytes.to_vec());
            match dec.decode(&pkt) {
                Ok(buf) => {
                    let ch = buf.num_planes().max(1);
                    buf.copy_to_vec_interleaved(&mut inter);
                    let frames = inter.len() / ch;
                    for f in 0..frames {
                        let s: f32 = inter[f * ch..f * ch + ch].iter().sum();
                        mono.push(s / ch as f32);
                    }
                    // keep the packet grid even if a decoder returns short frames
                    mono.resize(mono.len() + AAC_FRAME.saturating_sub(frames), 0.0);
                }
                Err(_) => mono.resize(mono.len() + AAC_FRAME, 0.0),
            }
        }
        let o0 = out_pos(p0);
        let t = std::time::Instant::now();
        let x_start = (d0 * AAC_FRAME) as isize - skip as isize;
        if x_start >= 0 {
            rs.process(&mono, x_start as usize, o0, dst);
        } else {
            rs.process(&mono[((-x_start) as usize).min(mono.len())..], 0, o0, dst);
        }
        RESAMPLE_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    })?;
    Ok(out)
}

/// MP3: frames are parsed sequentially by symphonia's MPEG reader, then decoded
/// in parallel chunks with pre-roll (bit reservoir + MDCT overlap). Encoder
/// delay and padding from the LAME tag are trimmed like ffmpeg does
/// (delay = enc_delay + 529).
fn decode_mp3(path: &Path) -> Result<Vec<f32>, String> {
    use symphonia::core::formats::{FormatOptions, FormatReader};
    use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};

    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(std::io::Cursor::new(data)), MediaSourceStreamOptions::default());
    let mut reader = symphonia::default::formats::MpaReader::try_new(mss, FormatOptions::default()).map_err(|e| e.to_string())?;
    let track = reader.tracks().first().ok_or("mp3: no track")?.clone();
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or("mp3: no audio codec parameters")?
        .clone();
    let in_rate = params.sample_rate.ok_or("mp3: unknown sample rate")?;
    let delay = track.delay.unwrap_or(0) as usize;
    let padding = track.padding.unwrap_or(0) as usize;

    let mut packets: Vec<Box<[u8]>> = Vec::new();
    let mut frame = 0usize;
    while let Some(p) = reader.next_packet().map_err(|e| e.to_string())? {
        if frame == 0 {
            frame = p.dur.get() as usize;
        }
        packets.push(p.data);
    }
    if packets.is_empty() || frame == 0 {
        return Err("mp3: no audio frames".into());
    }
    let n = packets.len();
    let in_len = (n * frame).saturating_sub(delay + padding);
    let rs = Resampler::new(in_rate, OUT_RATE);
    let out_len = rs.out_len(in_len);
    let mut out = vec![0f32; out_len];

    // output j belongs to the chunk whose input range [s0, s1) contains j*num/den
    let chunk_packets: usize = std::env::var("SUBTHX_MP3_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let pre: usize = std::env::var("SUBTHX_MP3_PRE").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let post = rs.half.div_ceil(frame) + 1;
    let out_pos = |s: usize| (s * rs.den).div_ceil(rs.num).min(out_len);
    let mut chunks: Vec<(usize, usize, &mut [f32])> = Vec::new();
    {
        let mut rest: &mut [f32] = &mut out;
        let mut p0 = 0;
        let mut o_prev = 0;
        while p0 < n {
            let p1 = (p0 + chunk_packets).min(n);
            let s1 = if p1 == n { in_len } else { (p1 * frame).saturating_sub(delay) };
            let o1 = if p1 == n { out_len } else { out_pos(s1) };
            let (a, b) = rest.split_at_mut(o1 - o_prev);
            chunks.push((p0, o_prev, a));
            rest = b;
            o_prev = o1;
            p0 = p1;
        }
    }
    chunks.into_par_iter().try_for_each(|(p0, o0, dst)| -> Result<(), String> {
        let p1 = (p0 + chunk_packets).min(n);
        let d0 = p0.saturating_sub(pre);
        let d1 = (p1 + post).min(n);
        let mut dec = symphonia::default::codecs::MpaDecoder::try_new(&params, &AudioDecoderOptions::default())
            .map_err(|e| e.to_string())?;
        let mut mono: Vec<f32> = Vec::with_capacity((d1 - d0) * frame);
        let mut inter: Vec<f32> = Vec::new();
        for (p, bytes) in packets.iter().enumerate().take(d1).skip(d0) {
            let pkt = Packet::new(0, ((p * frame) as i64).into(), (frame as u64).into(), bytes.clone());
            match dec.decode(&pkt) {
                Ok(buf) => {
                    let ch = buf.num_planes().max(1);
                    buf.copy_to_vec_interleaved(&mut inter);
                    let frames = inter.len() / ch;
                    for f in 0..frames.min(frame) {
                        let s: f32 = inter[f * ch..f * ch + ch].iter().sum();
                        mono.push(s / ch as f32);
                    }
                    mono.resize(mono.len() + frame.saturating_sub(frames), 0.0);
                }
                Err(_) => mono.resize(mono.len() + frame, 0.0),
            }
        }
        // absolute (trimmed) input index of mono[0]
        let x_start = (d0 * frame) as isize - delay as isize;
        if x_start >= 0 {
            rs.process(&mono, x_start as usize, o0, dst);
        } else {
            let skip = (-x_start) as usize;
            rs.process(&mono[skip.min(mono.len())..], 0, o0, dst);
        }
        Ok(())
    })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_track_in_the_models_language() {
        // dual-audio anime: English dub first and default
        let dual = [(Some("eng"), true), (Some("jpn"), false)];
        assert_eq!(pick_track(&dual, &["ja"]), 1);
        assert_eq!(pick_track(&dual, &["en", "de"]), 0);
        assert_eq!(pick_track(&dual, &[]), 0);
        // untagged: the default one
        assert_eq!(pick_track(&[(None, false), (Some("und"), true)], &["ja"]), 1);
        assert_eq!(pick_track(&[(Some("ja-JP"), false), (Some("fre"), true)], &["ja"]), 0);
        // nothing in the language: the default one
        assert_eq!(pick_track(&[(Some("eng"), false), (Some("spa"), true)], &["ja"]), 1);
    }
}
