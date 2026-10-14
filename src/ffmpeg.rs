//! ffmpeg as a fallback for what symphonia cannot read (Opus, AC-3, DTS,
//! WMA, AVI, FLV, MPEG-TS…). Used only when an `ffmpeg` binary is found:
//! `SUBTHX_FFMPEG` (`none` to turn it off), next to our executable, or on PATH.

use rayon::prelude::*;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

fn find(name: &str) -> Option<PathBuf> {
    let file = if cfg!(windows) { format!("{name}.exe") } else { name.to_string() };
    let env = format!("SUBTHX_{}", name.to_uppercase());
    if let Some(p) = std::env::var_os(&env).map(PathBuf::from) {
        // "none" turns the fallback off
        return Some(p).filter(|p| p.is_file());
    }
    let beside = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join(&file)));
    if let Some(p) = beside.filter(|p| p.is_file()) {
        return Some(p);
    }
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(&file)).find(|p| p.is_file())
}

pub fn ffmpeg() -> Option<&'static Path> {
    static P: OnceLock<Option<PathBuf>> = OnceLock::new();
    P.get_or_init(|| find("ffmpeg")).as_deref()
}

pub fn ffprobe() -> Option<&'static Path> {
    static P: OnceLock<Option<PathBuf>> = OnceLock::new();
    P.get_or_init(|| find("ffprobe")).as_deref()
}

/// A console-less child process (the app has no console to borrow).
fn command(bin: &Path) -> Command {
    let mut c = Command::new(bin);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        c.creation_flags(CREATE_NO_WINDOW);
    }
    c.stdin(Stdio::null());
    c
}

#[derive(serde::Deserialize, Default)]
pub struct Probe {
    #[serde(default)]
    pub format: Format,
    #[serde(default)]
    pub chapters: Vec<ProbeChapter>,
    #[serde(default)]
    pub streams: Vec<ProbeStream>,
}

#[derive(serde::Deserialize, Default)]
pub struct ProbeStream {
    #[serde(default)]
    pub codec_type: String,
    #[serde(default)]
    pub codec_name: String,
    #[serde(default)]
    pub tags: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub disposition: std::collections::HashMap<String, i64>,
}

#[derive(serde::Deserialize, Default)]
pub struct Format {
    pub duration: Option<String>,
    #[serde(default)]
    pub tags: std::collections::HashMap<String, String>,
}

#[derive(serde::Deserialize)]
pub struct ProbeChapter {
    pub id: i64,
    pub start_time: String,
    pub end_time: String,
    #[serde(default)]
    pub tags: std::collections::HashMap<String, String>,
}

/// `ffprobe -show_format -show_chapters -show_streams` (None without ffprobe).
pub fn probe(path: &Path) -> Option<Result<Probe, String>> {
    let bin = ffprobe()?;
    let out = command(bin)
        .args(["-v", "error", "-print_format", "json", "-show_format", "-show_chapters", "-show_streams"])
        .arg(path)
        .output();
    Some(match out {
        Ok(o) if o.status.success() => serde_json::from_slice(&o.stdout).map_err(|e| format!("ffprobe: {e}")),
        Ok(o) => Err(format!("ffprobe: {}", String::from_utf8_lossy(&o.stderr).trim())),
        Err(e) => Err(format!("ffprobe: {e}")),
    })
}

/// 16 kHz mono f32 of `[start, start+len)` seconds (whole file without `start`).
fn run(bin: &Path, path: &Path, track: usize, span: Option<(f64, f64)>) -> Result<Vec<f32>, String> {
    let mut c = command(bin);
    c.args(["-hide_banner", "-loglevel", "error"]);
    // no -ss for the first piece: even `-ss 0` drops a leading frame of some codecs
    if let Some((start, _)) = span.filter(|&(start, _)| start > 0.0) {
        c.args(["-ss", &format!("{start:.6}")]);
    }
    c.arg("-i").arg(path);
    if let Some((_, len)) = span {
        c.args(["-t", &format!("{len:.6}")]);
    }
    c.args(["-map", &format!("0:a:{track}"), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]);
    let mut child = c.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| format!("ffmpeg: {e}"))?;
    let mut bytes = Vec::new();
    child.stdout.take().unwrap().read_to_end(&mut bytes).map_err(|e| format!("ffmpeg: {e}"))?;
    let mut err = String::new();
    child.stderr.take().unwrap().read_to_string(&mut err).ok();
    let status = child.wait().map_err(|e| format!("ffmpeg: {e}"))?;
    if !status.success() {
        return Err(format!("ffmpeg: {}", err.trim()));
    }
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
}

/// Bytes per second above which a file holds more than sound (2 Mbit/s;
/// lossless stereo stays under it).
const MAX_AUDIO_RATE: f64 = 250_000.0;

/// Decode with ffmpeg. Long audio files are cut into pieces decoded by
/// parallel ffmpeg processes; a video is read once, in order. `-ss` is only roughly exact (tens of ms in AVI), so
/// each piece starts a little early and is spliced where its overlap matches
/// what came before; every non-final piece then has its exact length, so the
/// timeline never drifts.
pub fn decode_16k(path: &Path, langs: &[&str]) -> Result<Vec<f32>, String> {
    const M: usize = 16000; // matched overlap
    const R: usize = 1600; // search range either way
    let bin = ffmpeg().ok_or("ffmpeg not found")?;
    let probed = probe(path).and_then(Result::ok).unwrap_or_default();
    let duration = probed.format.duration.as_deref().and_then(|d| d.parse::<f64>().ok());
    let audio: Vec<(Option<&str>, bool)> = probed
        .streams
        .iter()
        .filter(|s| s.codec_type == "audio")
        .map(|s| (s.tags.get("language").map(String::as_str), s.disposition.get("default") == Some(&1)))
        .collect();
    let track = crate::audio::pick_track(&audio, langs);
    // a video is mostly picture: its pieces would read the whole file at
    // once from several places, which a hard disk serves at a fraction of
    // its speed, while one process decodes the sound faster than the disk
    // reads the file
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let pieces = match duration {
        Some(d) if d > 600.0 && (bytes as f64) < d * MAX_AUDIO_RATE => ((d / 300.0).ceil() as usize).min(rayon::current_num_threads()).max(1),
        _ => 1,
    };
    if pieces == 1 {
        return run(bin, path, track, None);
    }
    let d = duration.unwrap();
    let bound = |i: usize| ((d * i as f64 / pieces as f64) * 16000.0).round() as usize;
    let parts: Vec<Vec<f32>> = (0..pieces)
        .into_par_iter()
        .map(|i| {
            let s0 = if i == 0 { 0 } else { bound(i) - M - R };
            let len = if i + 1 == pieces { 1e9 } else { (bound(i + 1) + R - s0) as f64 / 16000.0 };
            run(bin, path, track, Some((s0 as f64 / 16000.0, len)))
        })
        .collect::<Result<_, String>>()?;
    let mut out: Vec<f32> = Vec::with_capacity(bound(pieces) + 16000);
    for (i, part) in parts.into_iter().enumerate() {
        let skip = if i == 0 { 0 } else { (M + R) as i64 + splice_lag(&out[bound(i) - M..bound(i)], &part, R) };
        let rest = part.get(skip.max(0) as usize..).unwrap_or(&[]);
        if i + 1 == pieces {
            out.extend_from_slice(rest);
        } else {
            let n = bound(i + 1) - bound(i);
            out.extend_from_slice(&rest[..n.min(rest.len())]);
            out.resize(bound(i + 1), 0.0);
        }
    }
    Ok(out)
}

/// How far `tail` (the last samples so far) sits in `part` from where it
/// should (at `r`): the best normalized correlation within ±r; 0 when the
/// overlap is too quiet or too unlike to tell.
fn splice_lag(tail: &[f32], part: &[f32], r: usize) -> i64 {
    let m = tail.len();
    let e: f64 = tail.iter().map(|x| (*x as f64).powi(2)).sum();
    if e / (m as f64) < 1e-7 || part.len() < m + 2 * r {
        return 0;
    }
    let mut best = (0i64, 0.5f64);
    for l in -(r as i64)..=r as i64 {
        let b0 = (r as i64 + l) as usize;
        let w = &part[b0..b0 + m];
        let (mut xy, mut yy) = (0f64, 0f64);
        for (x, y) in tail.iter().zip(w) {
            xy += *x as f64 * *y as f64;
            yy += (*y as f64).powi(2);
        }
        let c = xy / (e * yy).sqrt().max(1e-12);
        if c > best.1 {
            best = (l, c);
        }
    }
    best.0
}

/// A frame a third of the way in, as a JPEG: the cover of a video. Cached,
/// since every card asks for it.
pub fn frame(path: &Path) -> Option<Vec<u8>> {
    static CACHE: OnceLock<std::sync::Mutex<std::collections::HashMap<PathBuf, Option<Vec<u8>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(hit) = cache.lock().unwrap().get(path) {
        return hit.clone();
    }
    let bin = ffmpeg()?;
    let at = match probe(path) {
        Some(Ok(p)) => p.format.duration.and_then(|d| d.parse::<f64>().ok()).map_or(0.0, |d| d / 3.0),
        _ => 0.0,
    };
    let out = command(bin)
        .args(["-hide_banner", "-loglevel", "error", "-ss", &format!("{at:.3}"), "-i"])
        .arg(path)
        .args(["-map", "0:v:0", "-frames:v", "1", "-vf", "scale=-2:240", "-c:v", "mjpeg", "-q:v", "4", "-f", "image2pipe", "pipe:1"])
        .output()
        .ok();
    let jpeg = out.filter(|o| o.status.success() && !o.stdout.is_empty()).map(|o| o.stdout);
    cache.lock().unwrap().insert(path.to_path_buf(), jpeg.clone());
    jpeg
}

/// Subtitle codecs that are text (PGS and DVD subtitles are pictures).
const TEXT_SUBS: &[&str] = &["subrip", "ass", "ssa", "webvtt", "mov_text", "text"];

/// The embedded text subtitle track in `lang` (ISO 639-1) of a video:
/// (index among its subtitle streams, codec). A track tagged with the
/// language (full dialogue before signs-only or SDH ones), else the only
/// subtitle track when it is untagged.
pub fn subtitle_track(path: &Path, lang: &str) -> Option<(usize, String)> {
    let p = probe(path)?.ok()?;
    let subs: Vec<&ProbeStream> = p.streams.iter().filter(|s| s.codec_type == "subtitle").collect();
    fn tag(s: &ProbeStream) -> Option<&str> {
        s.tags.get("language").map(String::as_str).filter(|l| !l.is_empty() && *l != "und")
    }
    let text = |s: &ProbeStream| TEXT_SUBS.contains(&s.codec_name.as_str());
    // full dialogue first: not forced (signs only), not for the hearing
    // impaired (sound descriptions between the lines)
    let partial = |s: &ProbeStream| {
        let title = s.tags.get("title").map(|t| t.to_lowercase()).unwrap_or_default();
        let flag = |k: &str| s.disposition.get(k) == Some(&1);
        flag("forced") || flag("hearing_impaired") || ["sdh", "forced", "sign", "song", "cc"].iter().any(|w| title.contains(w))
    };
    let fits: Vec<usize> = (0..subs.len()).filter(|&i| text(subs[i]) && tag(subs[i]).is_some_and(|l| crate::audio::lang_is(l, lang))).collect();
    if let Some(&i) = fits.iter().find(|&&i| !partial(subs[i])).or(fits.first()) {
        return Some((i, subs[i].codec_name.clone()));
    }
    match subs.as_slice() {
        [s] if text(s) && tag(s).is_none() => Some((0, s.codec_name.clone())),
        _ => None,
    }
}

/// The embedded subtitles in `lang` written to a file (ASS stays ASS, the
/// rest becomes SRT), cached in the temp folder by the video's path, size
/// and time.
pub fn extract_subtitles(path: &Path, lang: &str) -> Result<PathBuf, String> {
    let bin = ffmpeg().ok_or("embedded subtitles need ffmpeg")?;
    let (track, codec) = subtitle_track(path, lang).ok_or_else(|| format!("{}: no embedded text subtitles in {lang:?}", path.display()))?;
    let ext = if codec == "ass" || codec == "ssa" { "ass" } else { "srt" };
    let md = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&(path, md.len(), md.modified().ok(), track), &mut h);
    let dir = std::env::temp_dir().join("subthx-subs");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let out = dir.join(format!("{stem}.{:016x}.{lang}.{ext}", std::hash::Hasher::finish(&h)));
    if out.exists() {
        return Ok(out);
    }
    let tmp = out.with_extension(format!("part.{ext}"));
    let o = command(bin)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(path)
        .args(["-map", &format!("0:s:{track}"), "-f", ext])
        .arg(&tmp)
        .output()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    if !o.status.success() {
        std::fs::remove_file(&tmp).ok();
        return Err(format!("ffmpeg: {}", String::from_utf8_lossy(&o.stderr).trim()));
    }
    std::fs::rename(&tmp, &out).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(out)
}

