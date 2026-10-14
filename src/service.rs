//! The sync/gen steps shared by the CLI and the HTTP server: pairing files,
//! transcribing one audio file into per-chapter transcripts, aligning them
//! against the book, and naming/writing subtitle files.

use crate::align::Punct;
use crate::batch::{Segment, Transcript};
use crate::media::Chapter;
use crate::pipeline::{Engine, Span, Stage, Timings};
use crate::shift::{shift_align, write_srt, write_vtt, END_PUNC, OTHER_PUNC, START_PUNC};
use crate::sync::{align_batches, expand_matches, match_start, AudioChapter, AudioFile, TextFile};
use std::path::{Path, PathBuf};

const NOPEND: &str = "うぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ…\u{3000}\u{20}";
/// Audio and video files we take (ffmpeg, if installed, covers the odd ones).
pub const AUDIO_EXTS: &[&str] = &[
    "m4b", "m4a", "mp4", "mp3", "aac", "flac", "wav", "ogg", "oga", "opus", "aiff", "aif", "caf", "wma", "mka", "mkv", "webm", "mov", "m4v",
    "avi", "wmv", "flv", "ts", "mts", "m2ts", "mpg", "mpeg", "3gp",
];

pub fn ext(p: &Path) -> String {
    p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default()
}

pub fn is_audio(p: &Path) -> bool {
    AUDIO_EXTS.contains(&ext(p).as_str())
}

/// natural ("os") sort key: digit runs compare as numbers
pub fn natural_key(p: &Path) -> Vec<(u8, u64, String)> {
    let s = p.to_string_lossy().to_lowercase();
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut digits = false;
    let flush = |buf: &mut String, digits: bool, out: &mut Vec<(u8, u64, String)>| {
        if !buf.is_empty() {
            if digits {
                out.push((0, buf.parse().unwrap_or(u64::MAX), String::new()));
            } else {
                out.push((1, 0, buf.clone()));
            }
            buf.clear();
        }
    };
    for c in s.chars() {
        if c.is_ascii_digit() != digits {
            flush(&mut buf, digits, &mut out);
            digits = c.is_ascii_digit();
        }
        buf.push(c);
    }
    flush(&mut buf, digits, &mut out);
    out
}

/// One book: its audio files (in reading order) and text sources.
#[derive(Clone, Debug)]
pub struct Job {
    pub audios: Vec<PathBuf>,
    pub texts: Vec<PathBuf>,
}

/// Split explicit files into a job; gen makes one job per audio file.
pub fn file_jobs(files: &[PathBuf], sync: bool) -> Result<Vec<Job>, String> {
    let audios: Vec<PathBuf> = files.iter().filter(|p| is_audio(p)).cloned().collect();
    let texts: Vec<PathBuf> = files.iter().filter(|p| crate::textsrc::is_text(p)).cloned().collect();
    if let Some(p) = files.iter().find(|p| !is_audio(p) && !crate::textsrc::is_text(p)) {
        return Err(format!("{}: unsupported file type", p.display()));
    }
    if audios.is_empty() {
        return Err("no audio files given".into());
    }
    if !sync {
        return Ok(audios.into_iter().map(|a| Job { audios: vec![a], texts: vec![] }).collect());
    }
    if texts.is_empty() {
        return Err("sync needs the book: add an epub/txt/subtitle file".into());
    }
    if audios.len() > 1 && texts.iter().all(|t| crate::textsrc::is_subtitle(t)) {
        // episodes and their subtitles, not one book
        return Ok(pair_subtitles(audios, texts.clone(), None, &mut |_| {}));
    }
    Ok(vec![Job { audios, texts }])
}

/// Subtitles to retime: they keep their own lines, only the times change.
/// A video among the texts stands for its embedded subtitles.
pub fn is_retime(texts: &[PathBuf]) -> bool {
    !texts.is_empty() && texts.iter().all(|t| crate::textsrc::is_subtitle(t) || is_audio(t))
}

/// Where subthx keeps what it needs between runs (SUBTHX_STATE_DIR, else
/// %LOCALAPPDATA%\subthx or ~/.local/share/subthx).
fn state_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("SUBTHX_STATE_DIR") {
        return Some(d.into());
    }
    std::env::var_os("LOCALAPPDATA")
        .map(|d| PathBuf::from(d).join("subthx"))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share/subthx")))
}

/// path -> (size, modified ns): files as subthx wrote them
type Stamps = std::collections::BTreeMap<String, (u64, u64)>;

fn stamp(p: &Path) -> Option<(String, (u64, u64))> {
    let full = std::fs::canonicalize(p).ok()?;
    let md = std::fs::metadata(&full).ok()?;
    let ns = md.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos() as u64;
    Some((full.to_string_lossy().into_owned(), (md.len(), ns)))
}

fn outputs_file() -> Option<PathBuf> {
    state_dir().map(|d| d.join("outputs.json"))
}

static OUTPUTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The subtitle files subthx wrote, unchanged since: planning does not take
/// them for subtitles to retime (after a gen, `ep01.ja.srt` next to the
/// video is ours, not the user's).
pub struct OwnOutputs(Stamps);

impl OwnOutputs {
    pub fn load() -> OwnOutputs {
        let read = || -> Option<Stamps> { serde_json::from_slice(&std::fs::read(outputs_file()?).ok()?).ok() };
        OwnOutputs(read().unwrap_or_default())
    }

    pub fn contains(&self, p: &Path) -> bool {
        stamp(p).is_some_and(|(k, v)| self.0.get(&k) == Some(&v))
    }
}

/// Remember a file subthx just wrote (see [`OwnOutputs`]).
pub fn record_output(p: &Path) {
    let (Some(file), Some((k, v))) = (outputs_file(), stamp(p)) else { return };
    let _lock = OUTPUTS.lock().unwrap_or_else(|e| e.into_inner());
    let mut all = OwnOutputs::load().0;
    all.insert(k, v);
    if all.len() > 20_000 {
        // forget files that are gone or changed since
        all.retain(|k, v| stamp(Path::new(k)).is_some_and(|(_, now)| now == *v));
    }
    if let Some(d) = file.parent() {
        std::fs::create_dir_all(d).ok();
    }
    let tmp = file.with_extension("json.tmp");
    if std::fs::write(&tmp, serde_json::to_vec(&all).unwrap_or_default()).is_ok() {
        std::fs::rename(&tmp, &file).ok();
    }
}

/// The language a subtitle file's name gives (`ep01.ja.srt`,
/// `ep01.jpn.ass`, `ep01.en-US.vtt`), ISO 639-1.
fn name_language(p: &Path) -> Option<&'static str> {
    let stem = p.file_stem()?.to_string_lossy().into_owned();
    let parts: Vec<&str> = stem.split('.').collect();
    // the base name itself is not a tag ("en.srt" is a file called "en")
    parts.iter().skip(1).rev().take(2).find_map(|t| crate::audio::lang_code(t))
}

/// Whether text is written in the script of `lang`: kana/kanji for
/// Japanese, hangul, Cyrillic, Greek, else Latin letters. Too little text
/// to tell fits.
pub fn script_fits(text: &str, lang: &str) -> bool {
    let (mut kana, mut han, mut hangul, mut cyr, mut greek, mut latin, mut all) = (0usize, 0, 0, 0, 0, 0, 0);
    for c in text.chars().filter(|c| c.is_alphabetic()) {
        all += 1;
        match c as u32 {
            0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => kana += 1,
            0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF => han += 1,
            0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F => hangul += 1,
            0x0400..=0x052F => cyr += 1,
            0x0370..=0x03FF | 0x1F00..=0x1FFF => greek += 1,
            _ if c.is_ascii_alphabetic() || (0x00C0..=0x024F).contains(&(c as u32)) => latin += 1,
            _ => {}
        }
    }
    if all < 20 {
        return true;
    }
    let share = |n: usize| n as f64 / all as f64;
    match lang {
        "ja" => share(kana + han) >= 0.3,
        "zh" => share(han) >= 0.3 && kana * 10 < han,
        "ko" => share(hangul) >= 0.3,
        "ru" | "uk" | "bg" | "sr" | "mk" | "be" => share(cyr) >= 0.5,
        "el" => share(greek) >= 0.5,
        _ => share(latin) >= 0.5,
    }
}

/// Whether subtitles are in `lang`: the language in their name (when
/// there is one) and the script of their text.
pub fn subtitle_fits(p: &Path, lang: &str) -> bool {
    if name_language(p).is_some_and(|l| l != lang) {
        return false;
    }
    // a name can lie (an English file called .ja.srt): the text decides too
    let Ok(bytes) = std::fs::read(p) else { return false };
    let src = String::from_utf8_lossy(&bytes[..bytes.len().min(1 << 20)]).into_owned();
    script_fits(&crate::textsrc::subtitle_lines(&src, &ext(p)).join("\n"), lang)
}

/// What planning needs to choose subtitles: the language of the job and
/// the files subthx wrote itself.
pub struct Choose {
    pub lang: String,
    pub own: OwnOutputs,
}

impl Choose {
    pub fn new(lang: &str) -> Choose {
        Choose { lang: lang.to_string(), own: OwnOutputs::load() }
    }

    /// Subtitles worth retiming: the user's, in the job's language (not
    /// earlier outputs or their `.old` backups).
    pub fn usable(&self, p: &Path) -> bool {
        !p.file_stem().is_some_and(|s| s.to_string_lossy().ends_with(".old")) && !self.own.contains(p) && subtitle_fits(p, &self.lang)
    }
}

/// SubPlz `--respect-grouping-count` for a job: subtitles being retimed
/// keep their lines (15: SubPlz's depth for subtitles with songs).
pub fn grouping_for(texts: &[PathBuf], asked: Option<usize>) -> Option<usize> {
    asked.or(is_retime(texts).then_some(15))
}

/// Videos (or audio files) and subtitle files: each subtitle goes with the
/// media file its name starts with (`ep01.mkv` <- `ep01.ja.srt`), the rest
/// pair in sort order when the counts match. With `embedded` (the job's
/// language), a video left without subtitles retimes its own embedded
/// text track in that language; media still without get generated ones.
pub fn pair_subtitles(audios: Vec<PathBuf>, mut subs: Vec<PathBuf>, embedded: Option<&str>, warn: &mut dyn FnMut(String)) -> Vec<Job> {
    let lower = |p: &Path| p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
    let mut pairs: Vec<(PathBuf, Option<PathBuf>)> = Vec::new();
    for a in audios {
        let stem = format!("{}.", a.file_stem().unwrap_or_default().to_string_lossy().to_lowercase());
        let hit = subs.iter().position(|s| lower(s).starts_with(&stem));
        pairs.push((a, hit.map(|i| subs.remove(i))));
    }
    let unmatched = pairs.iter().filter(|(_, s)| s.is_none()).count();
    if unmatched > 0 && unmatched == subs.len() {
        let mut rest = subs.drain(..);
        for (_, s) in pairs.iter_mut().filter(|(_, s)| s.is_none()) {
            *s = rest.next();
        }
    }
    for s in &subs {
        warn(format!("{}: no video or audio of the same name, skipping", s.display()));
    }
    if let Some(lang) = embedded.filter(|_| crate::ffmpeg::ffprobe().is_some()) {
        use rayon::prelude::*;
        pairs.par_iter_mut().filter(|(_, s)| s.is_none()).for_each(|(a, s)| {
            if crate::ffmpeg::subtitle_track(a, lang).is_some() {
                *s = Some(a.clone());
            }
        });
    }
    pairs
        .into_iter()
        .map(|(a, s)| Job { audios: vec![a], texts: s.into_iter().collect() })
        .collect()
}

/// Pair a folder's audio and epub/txt files like SubPlz `-d`: several audio
/// files and one text are one book, otherwise they pair up in sort order.
/// Without a book, the videos retime their subtitles (files `choose` finds
/// usable, else embedded tracks). Problems that skip or change the pairing
/// go to `warn`.
pub fn dir_jobs(dir: &Path, sync: bool, choose: &Choose, warn: &mut dyn FnMut(String)) -> Result<Vec<Job>, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    files.sort_by_key(|p| natural_key(p));
    let audios: Vec<PathBuf> = files.iter().filter(|p| is_audio(p)).cloned().collect();
    if !sync {
        return Ok(audios.into_iter().map(|a| Job { audios: vec![a], texts: vec![] }).collect());
    }
    let texts: Vec<PathBuf> = files
        .iter()
        .filter(|p| crate::textsrc::TEXT_EXTS.contains(&ext(p).as_str()))
        .cloned()
        .collect();
    // no book but subtitles: retime them
    let subs: Vec<PathBuf> = files.iter().filter(|p| crate::textsrc::is_subtitle(p) && choose.usable(p)).cloned().collect();
    if texts.is_empty() && !audios.is_empty() {
        let jobs = pair_subtitles(audios.clone(), subs, Some(&choose.lang), warn);
        if jobs.iter().any(|j| !j.texts.is_empty()) {
            return Ok(jobs);
        }
    }
    if audios.is_empty() || texts.is_empty() {
        warn(format!("{}: needs audio and an epub/txt, skipping", dir.display()));
        return Ok(vec![]);
    }
    if audios.len() > 1 && texts.len() == 1 {
        // one book read across several audio files
        return Ok(vec![Job { audios, texts }]);
    }
    if audios.len() != texts.len() {
        warn(format!(
            "{}: {} audio files but {} texts, pairing them by sort order",
            dir.display(),
            audios.len(),
            texts.len()
        ));
    }
    Ok(audios.into_iter().zip(texts).map(|(a, t)| Job { audios: vec![a], texts: vec![t] }).collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Srt,
    Vtt,
}

impl std::str::FromStr for Format {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "srt" => Ok(Format::Srt),
            "vtt" => Ok(Format::Vtt),
            _ => Err(format!("unsupported format {s:?} (srt, vtt)")),
        }
    }
}

impl Format {
    pub fn ext(self) -> &'static str {
        match self {
            Format::Srt => "srt",
            Format::Vtt => "vtt",
        }
    }

    pub fn mime(self) -> &'static str {
        match self {
            Format::Srt => "application/x-subrip; charset=utf-8",
            Format::Vtt => "text/vtt; charset=utf-8",
        }
    }

    pub fn render(self, segs: &[Segment]) -> String {
        match self {
            Format::Srt => write_srt(segs),
            Format::Vtt => write_vtt(segs),
        }
    }
}

/// `<output_dir or the audio's folder>/<audio stem>[.<lang_ext>].<fmt>`
pub fn output_path(audio: &Path, output_dir: Option<&Path>, lang_ext: Option<&str>, fmt: Format) -> PathBuf {
    let dir = output_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| audio.parent().map(Path::to_path_buf).unwrap_or_default());
    let stem = audio.file_stem().unwrap_or_default().to_string_lossy().into_owned();
    let le = lang_ext.map(|l| format!(".{l}")).unwrap_or_default();
    dir.join(format!("{stem}{le}.{}", fmt.ext()))
}

/// Make room for `path`. Ok(false) = it exists and may not be replaced.
/// With `overwrite` the previous file is kept once as `<name>.old.<ext>`.
pub fn prepare_output(path: &Path, overwrite: bool) -> Result<bool, String> {
    if !path.exists() {
        if let Some(d) = path.parent() {
            if !d.as_os_str().is_empty() {
                std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
            }
        }
        return Ok(true);
    }
    if !overwrite {
        return Ok(false);
    }
    let old = path.with_extension(format!("old.{}", ext(path)));
    if !old.exists() {
        std::fs::rename(path, &old).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(true)
}

pub fn write_subs(path: &Path, segs: &[Segment], fmt: Format) -> Result<(), String> {
    write_subs_raw(path, segs, fmt)?;
    record_output(path);
    Ok(())
}

fn write_subs_raw(path: &Path, segs: &[Segment], fmt: Format) -> Result<(), String> {
    let mut s = fmt.render(segs);
    if cfg!(windows) {
        // SubPlz writes in Python text mode
        s = s.replace('\n', "\r\n");
    }
    std::fs::write(path, s).map_err(|e| format!("{}: {e}", path.display()))
}

/// One audio file's transcripts, one per kept chapter.
#[derive(Clone)]
pub struct Transcribed {
    /// chapters longer than a second (SubPlz drops the rest); empty = whole file
    pub chapters: Vec<Chapter>,
    pub audio: AudioFile,
    /// None when the transcripts came from a cache
    pub timings: Option<Timings>,
}

impl Transcribed {
    /// Audio chapter names for the match report (the file name without chapters).
    pub fn titles(&self, audio: &Path) -> Vec<String> {
        if self.chapters.is_empty() {
            vec![audio.file_name().unwrap_or_default().to_string_lossy().into_owned()]
        } else {
            self.chapters.iter().map(|c| c.title.clone()).collect()
        }
    }
}

fn kept_chapters(audio: &Path) -> Result<(Vec<Chapter>, Option<f64>), String> {
    let info = crate::media::probe(audio)?;
    let kept = info.chapters.iter().filter(|c| c.end - c.start > 1.0).cloned().collect();
    Ok((kept, info.duration))
}

/// `language` (ISO 639-1) picks the text cleaning used to match the book.
pub fn transcribe(
    engine: &mut Engine,
    audio: &Path,
    language: &str,
    progress: &mut dyn FnMut(Stage, f64) -> Result<(), String>,
) -> Result<Transcribed, String> {
    let (kept, _) = kept_chapters(audio)?;
    let spans: Vec<Span> = kept.iter().map(|c| Span { start: c.start, end: c.end }).collect();
    engine.set_language(language)?;
    let (mut transcripts, tm) = engine.transcribe_with(audio, &spans, progress)?;
    for t in &mut transcripts {
        t.language = language.to_string();
    }
    let durations: Vec<f64> = if kept.is_empty() {
        vec![tm.audio_seconds]
    } else {
        kept.iter().map(|c| c.end - c.start).collect()
    };
    let chapters = transcripts
        .into_iter()
        .zip(durations)
        .map(|(tr, duration)| AudioChapter {
            transcript: Some(tr),
            duration,
        })
        .collect();
    Ok(Transcribed {
        chapters: kept,
        audio: AudioFile { chapters },
        timings: Some(tm),
    })
}

/// SubPlz's cached transcripts: `<cache_dir>/<audio name>.<chapter id>.<model>.subs`
pub fn from_subplz_cache(cache_dir: &Path, audio: &Path, model: &str) -> Result<Transcribed, String> {
    let (kept, duration) = kept_chapters(audio)?;
    let ids: Vec<(i64, f64)> = if kept.is_empty() {
        vec![(0, duration.ok_or("audio has no chapters and unknown duration")?)]
    } else {
        kept.iter().map(|c| (c.id, c.end - c.start)).collect()
    };
    let mut chapters = Vec::new();
    for (cid, duration) in ids {
        let name = format!("{}.{cid}.{model}.subs", audio.file_name().unwrap().to_string_lossy());
        let p = cache_dir.join(&name);
        let src = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let v = crate::pyrepr::parse(&src).map_err(|e| format!("{}: {e}", p.display()))?;
        let tr: Transcript = serde_json::from_value(v).map_err(|e| format!("{}: {e}", p.display()))?;
        chapters.push(AudioChapter {
            transcript: Some(tr),
            duration,
        });
    }
    Ok(Transcribed {
        chapters: kept,
        audio: AudioFile { chapters },
        timings: None,
    })
}

/// SubPlz gen: chapter cues shifted by the running sum of chapter durations.
/// `shift`: SubPlz's `shift_align`, which moves a short lead-in (up to two
/// characters before a comma) to the cue before; off for a model that
/// breaks its own lines (Whisper), where that lead-in is the next speaker's.
pub fn gen(tr: &Transcribed, shift: bool) -> Vec<Segment> {
    let mut segs = Vec::new();
    let mut offset = 0.0f64;
    for ch in &tr.audio.chapters {
        for c in ch.transcript.iter().flat_map(|t| &t.segments) {
            segs.push(Segment {
                text: c.text.clone(),
                start: c.start + offset,
                end: c.end + offset,
            });
        }
        offset += ch.duration;
    }
    if shift {
        shift_align(segs)
    } else {
        segs
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ChapterMatch {
    pub audio_file: usize,
    pub audio_chapter: String,
    /// None = no book chapter matched; the transcript is used as is
    pub text_chapter: Option<String>,
    pub score: Option<f64>,
}

pub struct Synced {
    /// subtitles per audio file
    pub files: Vec<Vec<Segment>>,
    pub matches: Vec<ChapterMatch>,
    /// share of the matched book chapters' text that made it into the
    /// subtitles, 0..100 (the benchmark's "book match")
    pub book_match: Option<f64>,
}

/// Benchmark text normalization: NFKC, no whitespace or punctuation, lowercase.
fn bench_norm(s: &str) -> Vec<char> {
    use unicode_normalization::UnicodeNormalization;
    const PUNC: &str = "、。！？!?…・—―–-()（）「」『』【】〜～\"'“”‘’＜＞<>:：;；*＊+＋/_／＼=＝♪";
    s.nfkc()
        .filter(|c| !c.is_whitespace() && !PUNC.contains(*c))
        .flat_map(char::to_lowercase)
        .collect()
}

/// `(len(book) - missing - replaced) / len(book)` over a Levenshtein
/// alignment of the subtitles onto the book (tools/eval_book.py).
fn book_match(subs: &[char], book: &[char]) -> Option<f64> {
    if book.is_empty() {
        return None;
    }
    let (a, b, n) = crate::lev::encode(subs, book);
    let ops = crate::lev::editops(&a, &b, n);
    let lost = ops.iter().filter(|o| matches!(o.tag, crate::lev::Tag::Insert | crate::lev::Tag::Replace)).count();
    Some((book.len() - lost) as f64 / book.len() as f64 * 100.0)
}

/// Match audio chapters to book chapters and align them (all audio files
/// of one book together, so chapters may cross file boundaries).
///
/// `respect_grouping` (SubPlz `--respect-grouping-count`, applied only with a
/// single text source like SubPlz) regroups each file's cues into whole
/// sentences of the book.
pub fn sync(audios: &[PathBuf], trs: &[Transcribed], texts: &[PathBuf], respect_grouping: Option<usize>) -> Result<Synced, String> {
    // the language the audio was transcribed in is the book's
    let lang = trs
        .iter()
        .flat_map(|t| &t.audio.chapters)
        .find_map(|c| c.transcript.as_ref().map(|t| t.language.clone()))
        .unwrap_or_else(|| "ja".into());
    // a video among the texts: its embedded subtitles
    let texts: Vec<PathBuf> =
        texts.iter().map(|t| if is_audio(t) { crate::ffmpeg::extract_subtitles(t, &lang) } else { Ok(t.clone()) }).collect::<Result<_, _>>()?;
    let texts = texts.as_slice();
    let mut text = Vec::new();
    let mut text_titles = Vec::new();
    for p in texts {
        let (titles, chapters): (Vec<String>, Vec<Vec<String>>) = crate::textsrc::load_titled_in(p, &lang)?.into_iter().unzip();
        text_titles.push(titles);
        text.push(TextFile { chapters });
    }
    let audio_files: Vec<AudioFile> = trs.iter().map(|t| t.audio.clone()).collect();
    let p = Punct {
        prepend: START_PUNC.chars().collect(),
        append: END_PUNC.chars().chain(OTHER_PUNC.chars()).collect(),
        nopend: NOPEND.chars().collect(),
    };
    let (ats, sta) = match_start(&audio_files, &text);
    let batches = expand_matches(&audio_files, &text, &ats, &sta);
    let mut files: Vec<Vec<Segment>> = (0..audio_files.len())
        .map(|ai| shift_align(align_batches(&audio_files[ai], &text, &batches[ai], &p).into_iter().flatten().collect()))
        .collect();
    if let (Some(count), [book]) = (respect_grouping, texts) {
        // subtitles keep their own cues (SubPlz would split them into
        // sentences, which merges and splits the lines being retimed)
        let keep = crate::textsrc::is_subtitle(book);
        let script = if keep { crate::textsrc::subtitle_cues(book)? } else { crate::textsrc::script_lines_in(book, &lang)? };
        for segs in files.iter_mut().filter(|s| !s.is_empty()) {
            *segs = crate::regroup::nc_align_with(&script, &write_srt(segs), count, keep);
        }
    }
    let mut matches = Vec::new();
    for (ai, (tr, audio)) in trs.iter().zip(audios).enumerate() {
        for (i, title) in tr.titles(audio).into_iter().enumerate() {
            let m = ats.get(&(ai, i));
            matches.push(ChapterMatch {
                audio_file: ai,
                audio_chapter: title,
                text_chapter: m.map(|&(ti, j, _)| text_titles[ti][j].clone()),
                score: m.map(|&(_, _, s)| s),
            });
        }
    }
    // the book chapters the audio was aligned against, in book order
    let mut used: Vec<(usize, usize)> = batches
        .iter()
        .flatten()
        .filter(|b| b.text_file >= 0)
        .flat_map(|b| b.text_parts.iter().map(move |&j| (b.text_file as usize, j)))
        .collect();
    used.sort();
    used.dedup();
    let book: String = used.iter().flat_map(|&(ti, j)| text[ti].chapters[j].iter().map(String::as_str)).collect();
    let subs: String = files.iter().flatten().map(|s| crate::batch::py_strip(&s.text)).collect();
    let book_match = book_match(&bench_norm(&subs), &bench_norm(&book));
    Ok(Synced { files, matches, book_match })
}

/// SubPlz's failure note: `<output without its extension>.subfail`.
pub fn subfail_path(output: &Path) -> PathBuf {
    output.with_extension("subfail")
}

pub fn write_subfail(audio: &Path, output: &Path, error: &str) -> Result<PathBuf, String> {
    let p = subfail_path(output);
    let mut s = format!("Error processing subtitle for {}:\n{error}\n", audio.display());
    if cfg!(windows) {
        s = s.replace('\n', "\r\n");
    }
    std::fs::write(&p, s).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(p)
}

/// SubPlz `cleanup_subfail`: drop `.subfail` notes in the output folders
/// whose subtitles now exist. Returns the removed notes.
pub fn cleanup_subfail(outputs: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = outputs.iter().map(|o| o.parent().map(Path::to_path_buf).unwrap_or_default()).collect();
    dirs.sort();
    dirs.dedup();
    let mut removed = Vec::new();
    for d in dirs {
        let rd = if d.as_os_str().is_empty() { std::fs::read_dir(".") } else { std::fs::read_dir(&d) };
        let Ok(rd) = rd else { continue };
        for p in rd.filter_map(|e| e.ok().map(|e| e.path())) {
            if ext(&p) != "subfail" {
                continue;
            }
            if ["ass", "srt", "vtt", "ssa", "idx"].iter().any(|f| p.with_extension(f).exists()) && std::fs::remove_file(&p).is_ok() {
                removed.push(p);
            }
        }
    }
    removed
}

/// Put `dir` first on PATH so ONNX Runtime finds the CUDA/cuDNN DLLs there
/// (they are loaded lazily through the normal search path).
pub fn add_dll_dir(dir: &Path) -> Result<(), String> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut dirs = vec![dir.to_path_buf()];
    dirs.extend(std::env::split_paths(&path));
    std::env::set_var("PATH", std::env::join_paths(dirs).map_err(|e| e.to_string())?);
    Ok(())
}

/// Model options shared by `sync`, `gen` and `serve`.
#[derive(clap::Args, Clone, Debug)]
pub struct ModelArgs {
    /// speech recognition model: kitsune-p0.05b, kitsune-p0.1b, kitsune-p0.3b, parakeet-ultra, parakeet-redux, whisper-*
    #[arg(long, default_value = "kitsune-p0.05b")]
    pub asr: crate::pipeline::AsrModel,
    /// cuda, directml or cpu
    #[arg(long, default_value = "cuda")]
    pub device: crate::vad::Device,
    /// model folder (FireRedVAD, mel filterbank, kitsune-*)
    #[arg(long, default_value = "models")]
    pub models: PathBuf,
    /// utterances per ASR batch (default: 64 on GPUs, 16 on the CPU)
    #[arg(long)]
    pub batch_size: Option<usize>,
    /// folder with CUDA/cuDNN DLLs, added to the DLL search path
    #[arg(long)]
    pub cuda_dir: Option<PathBuf>,
}

impl ModelArgs {
    pub fn config(&self) -> crate::pipeline::Config {
        crate::pipeline::Config {
            asr: self.asr.clone(),
            device: self.device,
            models: self.models.clone(),
            batch_size: self.batch_size,
        }
    }
}
