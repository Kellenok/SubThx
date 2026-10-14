//! Audiobook container probing without ffmpeg: chapter list and duration the
//! way `ffprobe -show_chapters` reports them (SubPlz derives chapter durations
//! from ffprobe's `%f`-formatted start/end times, so we reproduce that too).
//!
//! Supported: MP4/M4B QuickTime chapter tracks (`tref/chap`), Nero `chpl`,
//! MP3 ID3v2 `CHAP` frames.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Chapter {
    pub id: i64,
    /// start/end in seconds, as ffprobe prints them (`%f`) and Python parses back
    pub start: f64,
    pub end: f64,
    pub title: String,
}

#[derive(Debug, Clone)]
pub struct MediaInfo {
    pub title: Option<String>,
    /// duration of the selected audio stream, seconds (if known)
    pub duration: Option<f64>,
    pub chapters: Vec<Chapter>,
}

/// ffprobe `print_time`: ts * av_q2d(tb) formatted with "%f"; Python float() of it.
fn ff_time(ts: i64, num: i64, den: i64) -> f64 {
    let d = ts as f64 * (num as f64 / den as f64);
    format!("{d:.6}").parse().unwrap()
}

pub fn probe(path: &Path) -> Result<MediaInfo, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut head = [0u8; 12];
    let n = f.read(&mut head).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    if n >= 8 && &head[4..8] == b"ftyp" {
        mp4::probe(&mut f)
    } else if n >= 4 && is_mp3(&head) {
        mp3::probe(&mut f)
    } else {
        Ok(probe_other(path))
    }
}

/// MP3: an ID3v2 tag or an MPEG audio frame sync with a layer (ADTS AAC has
/// the same sync but layer 0).
pub fn is_mp3(head: &[u8]) -> bool {
    head.starts_with(b"ID3") || (head.len() >= 2 && head[0] == 0xFF && head[1] & 0xE0 == 0xE0 && (head[1] >> 1) & 3 != 0)
}

/// Other containers: chapters and duration from ffprobe when it is
/// installed, else none (the duration then comes from the decoded audio).
fn probe_other(path: &Path) -> MediaInfo {
    let Some(Ok(p)) = crate::ffmpeg::probe(path) else {
        return MediaInfo { title: None, duration: None, chapters: Vec::new() };
    };
    let tag = |tags: &std::collections::HashMap<String, String>, k: &str| {
        tags.iter().find(|(key, _)| key.eq_ignore_ascii_case(k)).map(|(_, v)| v.clone())
    };
    MediaInfo {
        title: tag(&p.format.tags, "title"),
        duration: p.format.duration.and_then(|d| d.parse().ok()),
        chapters: p
            .chapters
            .iter()
            .filter_map(|c| {
                Some(Chapter {
                    id: c.id,
                    start: c.start_time.parse().ok()?,
                    end: c.end_time.parse().ok()?,
                    title: tag(&c.tags, "title").unwrap_or_default(),
                })
            })
            .collect(),
    }
}

/// Cover art: (image bytes, MIME type).
pub type Cover = (Vec<u8>, &'static str);

fn image_mime(b: &[u8]) -> &'static str {
    if b.starts_with(&[0x89, b'P', b'N', b'G']) {
        "image/png"
    } else if b.len() > 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        "image/webp"
    } else {
        "image/jpeg"
    }
}

/// Cover art embedded in an audio file (MP4 `covr`, ID3 `APIC`), else an
/// image next to it: `<stem>.jpg/png/webp`, `cover.*`, `folder.*`, or the
/// only image in the folder; for a video without one, a frame of it.
pub fn cover(path: &Path) -> Result<Option<Cover>, String> {
    let embedded = (|| -> Result<Option<Cover>, String> {
        let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut head = [0u8; 12];
        let n = f.read(&mut head).map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        if n >= 8 && &head[4..8] == b"ftyp" {
            mp4::cover(&mut f)
        } else if n >= 3 && &head[..3] == b"ID3" {
            mp3::cover(&mut f)
        } else {
            Ok(None)
        }
    })()?;
    if embedded.is_some() {
        return Ok(embedded);
    }
    if let Some(img) = sidecar_image(path) {
        return Ok(Some(img));
    }
    // a video: one of its frames (needs ffmpeg)
    const VIDEO: [&str; 14] = ["mp4", "m4v", "mkv", "webm", "mov", "avi", "wmv", "flv", "ts", "mts", "m2ts", "mpg", "mpeg", "3gp"];
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    Ok(VIDEO.contains(&ext.as_str()).then(|| crate::ffmpeg::frame(path)).flatten().map(|b| (b, "image/jpeg")))
}

pub fn sidecar_image(path: &Path) -> Option<Cover> {
    const EXTS: [&str; 4] = ["jpg", "jpeg", "png", "webp"];
    let dir = path.parent()?;
    let stem = path.file_stem()?.to_string_lossy().to_lowercase();
    let images: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| EXTS.contains(&e.to_string_lossy().to_lowercase().as_str())))
        .collect();
    let named = |want: &str| {
        images
            .iter()
            .find(|p| p.file_stem().is_some_and(|s| s.to_string_lossy().to_lowercase() == want))
    };
    let pick = named(&stem)
        .or_else(|| named("cover"))
        .or_else(|| named("folder"))
        .or(if images.len() == 1 { images.first() } else { None })?;
    let b = std::fs::read(pick).ok()?;
    let mime = image_mime(&b);
    Some((b, mime))
}

pub struct Mp4Audio {
    pub channels: u16,
    pub sample_rate: u32,
    /// AudioSpecificConfig
    pub asc: Vec<u8>,
    /// (file offset, size) per AAC access unit
    pub packets: Vec<(u64, u32)>,
    /// leading samples to drop (edit list: encoder priming)
    pub skip: usize,
}

pub fn mp4_audio(path: &Path) -> Result<Mp4Audio, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    mp4::audio(&mut f)
}

mod mp4 {
    use super::*;

    struct BoxHdr {
        typ: [u8; 4],
        body: u64,
        end: u64,
    }

    fn read_boxes(f: &mut File, start: u64, end: u64) -> Result<Vec<BoxHdr>, String> {
        let mut out = Vec::new();
        let mut pos = start;
        while pos + 8 <= end {
            f.seek(SeekFrom::Start(pos)).map_err(|e| e.to_string())?;
            let mut h = [0u8; 8];
            f.read_exact(&mut h).map_err(|e| e.to_string())?;
            let mut size = u32::from_be_bytes(h[..4].try_into().unwrap()) as u64;
            let typ: [u8; 4] = h[4..8].try_into().unwrap();
            let mut hdr = 8;
            if size == 1 {
                let mut l = [0u8; 8];
                f.read_exact(&mut l).map_err(|e| e.to_string())?;
                size = u64::from_be_bytes(l);
                hdr = 16;
            } else if size == 0 {
                size = end - pos;
            }
            if size < hdr {
                break;
            }
            out.push(BoxHdr {
                typ,

                body: pos + hdr,
                end: (pos + size).min(end),
            });
            pos += size;
        }
        Ok(out)
    }

    fn read_at(f: &mut File, off: u64, len: usize) -> Result<Vec<u8>, String> {
        f.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
        let mut v = vec![0u8; len];
        f.read_exact(&mut v).map_err(|e| e.to_string())?;
        Ok(v)
    }

    fn body(f: &mut File, b: &BoxHdr) -> Result<Vec<u8>, String> {
        read_at(f, b.body, (b.end - b.body) as usize)
    }

    fn child<'a>(bs: &'a [BoxHdr], t: &[u8; 4]) -> Option<&'a BoxHdr> {
        bs.iter().find(|b| &b.typ == t)
    }

    fn be32(b: &[u8], o: usize) -> u32 {
        u32::from_be_bytes(b[o..o + 4].try_into().unwrap())
    }
    fn be64(b: &[u8], o: usize) -> u64 {
        u64::from_be_bytes(b[o..o + 8].try_into().unwrap())
    }

    #[derive(Default)]
    struct Track {
        id: u32,
        handler: [u8; 4],
        timescale: u32,
        duration: u64,
        chap_refs: Vec<u32>,
        stts: Vec<(u32, u32)>,
        stsz: Vec<u32>,
        stsc: Vec<(u32, u32)>,
        chunk_offsets: Vec<u64>,
        /// mp4a: (channels, sample rate, AudioSpecificConfig)
        mp4a: Option<(u16, u32, Vec<u8>)>,
        /// edit list: media time where playback starts (encoder priming), media timescale
        skip: u64,
    }

    fn read_track(f: &mut File, trak: &BoxHdr) -> Result<Track, String> {
        let mut t = Track::default();
        let kids = read_boxes(f, trak.body, trak.end)?;
        if let Some(tkhd) = child(&kids, b"tkhd") {
            let b = body(f, tkhd)?;
            t.id = if b[0] == 1 { be32(&b, 20) } else { be32(&b, 12) };
        }
        if let Some(tref) = child(&kids, b"tref") {
            for r in read_boxes(f, tref.body, tref.end)? {
                if &r.typ == b"chap" {
                    let b = body(f, &r)?;
                    t.chap_refs.extend((0..b.len() / 4).map(|i| be32(&b, i * 4)));
                }
            }
        }
        if let Some(edts) = child(&kids, b"edts") {
            if let Some(elst) = read_boxes(f, edts.body, edts.end)?.into_iter().find(|b| &b.typ == b"elst") {
                let b = body(f, &elst)?;
                let (n, size) = (be32(&b, 4) as usize, if b[0] == 1 { 20 } else { 12 });
                // the first edit that is not an empty one (media time -1)
                t.skip = (0..n)
                    .filter(|i| 8 + (i + 1) * size <= b.len())
                    .map(|i| {
                        let o = 8 + i * size;
                        if b[0] == 1 { be64(&b, o + 8) as i64 } else { be32(&b, o + 4) as i32 as i64 }
                    })
                    .find(|&m| m >= 0)
                    .unwrap_or(0) as u64;
            }
        }
        let Some(mdia) = child(&kids, b"mdia") else { return Ok(t) };
        let mk = read_boxes(f, mdia.body, mdia.end)?;
        if let Some(mdhd) = child(&mk, b"mdhd") {
            let b = body(f, mdhd)?;
            if b[0] == 1 {
                t.timescale = be32(&b, 20);
                t.duration = be64(&b, 24);
            } else {
                t.timescale = be32(&b, 12);
                t.duration = be32(&b, 16) as u64;
            }
        }
        if let Some(hdlr) = child(&mk, b"hdlr") {
            let b = body(f, hdlr)?;
            t.handler = b[8..12].try_into().unwrap();
        }
        let Some(minf) = child(&mk, b"minf") else { return Ok(t) };
        let mi = read_boxes(f, minf.body, minf.end)?;
        let Some(stbl) = child(&mi, b"stbl") else { return Ok(t) };
        for b in read_boxes(f, stbl.body, stbl.end)? {
            let d = body(f, &b)?;
            match &b.typ {
                b"stts" => {
                    let n = be32(&d, 4) as usize;
                    t.stts = (0..n).map(|i| (be32(&d, 8 + i * 8), be32(&d, 12 + i * 8))).collect();
                }
                b"stsz" => {
                    let size = be32(&d, 4);
                    let n = be32(&d, 8) as usize;
                    t.stsz = if size != 0 {
                        vec![size; n]
                    } else {
                        (0..n).map(|i| be32(&d, 12 + i * 4)).collect()
                    };
                }
                b"stsc" => {
                    let n = be32(&d, 4) as usize;
                    t.stsc = (0..n).map(|i| (be32(&d, 8 + i * 12), be32(&d, 12 + i * 12))).collect();
                }
                b"stco" => {
                    let n = be32(&d, 4) as usize;
                    t.chunk_offsets = (0..n).map(|i| be32(&d, 8 + i * 4) as u64).collect();
                }
                b"stsd" => t.mp4a = parse_mp4a(&d),
                b"co64" => {
                    let n = be32(&d, 4) as usize;
                    t.chunk_offsets = (0..n).map(|i| be64(&d, 8 + i * 8)).collect();
                }
                _ => {}
            }
        }
        Ok(t)
    }

    /// stsd body -> mp4a (channels, sample rate, ASC from esds)
    fn parse_mp4a(d: &[u8]) -> Option<(u16, u32, Vec<u8>)> {
        // full box header (4) + entry count (4), then the first sample entry
        let e = d.get(8..)?;
        if e.len() < 36 || &e[4..8] != b"mp4a" {
            return None;
        }
        let version = u16::from_be_bytes([e[16], e[17]]);
        let channels = u16::from_be_bytes([e[24], e[25]]);
        let rate = u32::from_be_bytes(e[32..36].try_into().ok()?) >> 16;
        let o = 36 + match version {
            1 => 16,
            2 => 36,
            _ => 0,
        };
        let entry_len = (be32(e, 0) as usize).min(e.len());
        // QuickTime (.mov) keeps esds inside a `wave` atom
        fn esds(e: &[u8], mut o: usize, end: usize) -> Option<Vec<u8>> {
            while o + 8 <= end {
                let size = be32(e, o) as usize;
                if size < 8 {
                    break;
                }
                let stop = (o + size).min(end);
                match &e[o + 4..o + 8] {
                    b"esds" => return find_asc(e.get(o + 12..stop)?),
                    b"wave" => {
                        if let Some(a) = esds(e, o + 8, stop) {
                            return Some(a);
                        }
                    }
                    _ => {}
                }
                o += size;
            }
            None
        }
        esds(e, o, entry_len).map(|a| (channels, rate, a))
    }

    /// Walk MPEG-4 descriptors: ES(3) -> DecoderConfig(4) -> DecoderSpecificInfo(5).
    fn find_asc(mut b: &[u8]) -> Option<Vec<u8>> {
        fn read_desc(b: &[u8]) -> Option<(u8, &[u8], &[u8])> {
            let tag = *b.first()?;
            let mut len = 0usize;
            let mut i = 1;
            loop {
                let c = *b.get(i)?;
                len = (len << 7) | (c & 0x7F) as usize;
                i += 1;
                if c & 0x80 == 0 || i > 4 {
                    break;
                }
            }
            let body = b.get(i..i + len)?;
            Some((tag, body, &b[i + len..]))
        }
        while !b.is_empty() {
            let (tag, body, rest) = read_desc(b)?;
            match tag {
                3 => {
                    let flags = *body.get(2)?;
                    let mut o = 3;
                    if flags & 0x80 != 0 {
                        o += 2;
                    }
                    if flags & 0x40 != 0 {
                        o += 1 + *body.get(o)? as usize;
                    }
                    if flags & 0x20 != 0 {
                        o += 2;
                    }
                    return find_asc(body.get(o..)?);
                }
                4 => return find_asc(body.get(13..)?),
                5 => return Some(body.to_vec()),
                _ => b = rest,
            }
        }
        None
    }

    /// (file offset, size, timestamp) per sample
    fn samples(t: &Track) -> Vec<(u64, u32, i64)> {
        let mut out = Vec::with_capacity(t.stsz.len());
        let mut ts_iter = t.stts.iter().flat_map(|&(n, d)| std::iter::repeat_n(d, n as usize));
        let mut ts: i64 = 0;
        let mut s = 0usize;
        for (ci, &off) in t.chunk_offsets.iter().enumerate() {
            let chunk_no = ci as u32 + 1;
            let per = t
                .stsc
                .iter()
                .rev()
                .find(|&&(first, _)| first <= chunk_no)
                .map(|&(_, n)| n)
                .unwrap_or(1);
            let mut o = off;
            for _ in 0..per {
                if s >= t.stsz.len() {
                    return out;
                }
                out.push((o, t.stsz[s], ts));
                o += t.stsz[s] as u64;
                ts += ts_iter.next().unwrap_or(0) as i64;
                s += 1;
            }
        }
        out
    }

    fn sample_title(f: &mut File, off: u64, size: u32) -> Result<Option<String>, String> {
        if size < 2 {
            return Ok(None);
        }
        let d = read_at(f, off, size as usize)?;
        let len = u16::from_be_bytes([d[0], d[1]]) as usize;
        if len > size as usize - 2 {
            return Ok(None);
        }
        let t = &d[2..2 + len];
        let s = if t.len() >= 2 && t[..2] == [0xFE, 0xFF] {
            let u: Vec<u16> = t[2..].chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
            String::from_utf16_lossy(&u)
        } else if t.len() >= 2 && t[..2] == [0xFF, 0xFE] {
            let u: Vec<u16> = t[2..].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            String::from_utf16_lossy(&u)
        } else {
            String::from_utf8_lossy(t).into_owned()
        };
        Ok(Some(s.trim_end_matches('\0').to_string()))
    }

    fn ilst_title(f: &mut File, udta: &BoxHdr) -> Result<Option<String>, String> {
        for m in read_boxes(f, udta.body, udta.end)? {
            if &m.typ != b"meta" {
                continue;
            }
            for i in read_boxes(f, m.body + 4, m.end)? {
                if &i.typ != b"ilst" {
                    continue;
                }
                for item in read_boxes(f, i.body, i.end)? {
                    if item.typ != [0xA9, b'n', b'a', b'm'] {
                        continue;
                    }
                    for d in read_boxes(f, item.body, item.end)? {
                        if &d.typ == b"data" {
                            let b = body(f, &d)?;
                            return Ok(Some(String::from_utf8_lossy(&b[8..]).into_owned()));
                        }
                    }
                }
            }
        }
        Ok(None)
    }

    /// `moov/udta/meta/ilst/covr/data`
    pub fn cover(f: &mut File) -> Result<Option<super::Cover>, String> {
        let len = f.metadata().map_err(|e| e.to_string())?.len();
        let top = read_boxes(f, 0, len)?;
        let Some(moov) = child(&top, b"moov") else { return Ok(None) };
        let mk = read_boxes(f, moov.body, moov.end)?;
        let Some(udta) = child(&mk, b"udta") else { return Ok(None) };
        for m in read_boxes(f, udta.body, udta.end)? {
            if &m.typ != b"meta" {
                continue;
            }
            for i in read_boxes(f, m.body + 4, m.end)? {
                if &i.typ != b"ilst" {
                    continue;
                }
                for item in read_boxes(f, i.body, i.end)? {
                    if &item.typ != b"covr" {
                        continue;
                    }
                    for d in read_boxes(f, item.body, item.end)? {
                        if &d.typ == b"data" && d.end - d.body > 8 {
                            let b = body(f, &d)?[8..].to_vec();
                            let mime = super::image_mime(&b);
                            return Ok(Some((b, mime)));
                        }
                    }
                }
            }
        }
        Ok(None)
    }

    /// AAC packets of the audio track (with several, the general decoder
    /// picks one by language).
    pub fn audio(f: &mut File) -> Result<super::Mp4Audio, String> {
        let len = f.metadata().map_err(|e| e.to_string())?.len();
        let top = read_boxes(f, 0, len)?;
        let moov = child(&top, b"moov").ok_or("no moov box")?;
        let mut sound = Vec::new();
        for b in read_boxes(f, moov.body, moov.end)?.iter().filter(|b| &b.typ == b"trak") {
            let t = read_track(f, b)?;
            if t.handler == *b"soun" {
                sound.push(t);
            }
        }
        if sound.len() > 1 {
            return Err("several audio tracks".into());
        }
        if let Some(t) = sound.into_iter().next() {
            let (channels, sample_rate, asc) = t.mp4a.clone().ok_or("audio track is not AAC (mp4a/esds)")?;
            let packets = samples(&t).into_iter().map(|(o, s, _)| (o, s)).collect();
            let skip = if t.timescale > 0 { (t.skip as u128 * sample_rate as u128 / t.timescale as u128) as usize } else { 0 };
            return Ok(super::Mp4Audio {
                channels,
                sample_rate,
                asc,
                packets,
                skip,
            });
        }
        Err("no audio track".into())
    }

    pub fn probe(f: &mut File) -> Result<MediaInfo, String> {
        let len = f.metadata().map_err(|e| e.to_string())?.len();
        let top = read_boxes(f, 0, len)?;
        let moov = child(&top, b"moov").ok_or("no moov box")?;
        let mk = read_boxes(f, moov.body, moov.end)?;
        let mut tracks = Vec::new();
        for b in mk.iter().filter(|b| &b.typ == b"trak") {
            tracks.push(read_track(f, b)?);
        }
        let title = match child(&mk, b"udta") {
            Some(u) => ilst_title(f, u)?,
            None => None,
        };
        let audio = tracks.iter().find(|t| t.handler == *b"soun");
        let duration = audio.map(|t| t.duration as f64 / t.timescale as f64);

        let mut chapters = Vec::new();
        // QuickTime chapter track referenced from any track
        let chap_id = tracks.iter().flat_map(|t| t.chap_refs.iter().copied()).next();
        if let Some(ct) = chap_id.and_then(|id| tracks.iter().find(|t| t.id == id)) {
            let ss = samples(ct);
            // ffmpeg's st->duration here is the sum of sample durations, not mdhd
            let track_end: i64 = ct.stts.iter().map(|&(n, d)| n as i64 * d as i64).sum();
            for (i, &(off, size, ts)) in ss.iter().enumerate() {
                let end = if i + 1 < ss.len() { ss[i + 1].2 } else { track_end };
                let Some(title) = sample_title(f, off, size)? else { continue };
                chapters.push(Chapter {
                    id: i as i64,
                    start: ff_time(ts, 1, ct.timescale as i64),
                    end: ff_time(end, 1, ct.timescale as i64),
                    title,
                });
            }
        } else if let Some(udta) = child(&mk, b"udta") {
            // Nero chpl: 100ns units
            for c in read_boxes(f, udta.body, udta.end)? {
                if &c.typ != b"chpl" {
                    continue;
                }
                let b = body(f, &c)?;
                let n = b[8] as usize;
                let mut o = 9;
                let mut raw = Vec::new();
                for _ in 0..n {
                    let start = be64(&b, o) as i64;
                    let l = b[o + 8] as usize;
                    let t = String::from_utf8_lossy(&b[o + 9..o + 9 + l]).into_owned();
                    raw.push((start, t));
                    o += 9 + l;
                }
                let total = duration.map(|d| (d * 1e7) as i64).unwrap_or(0);
                for (i, (start, t)) in raw.iter().enumerate() {
                    let end = raw.get(i + 1).map(|r| r.0).unwrap_or(total);
                    chapters.push(Chapter {
                        id: i as i64,
                        start: ff_time(*start, 1, 10_000_000),
                        end: ff_time(end, 1, 10_000_000),
                        title: t.clone(),
                    });
                }
            }
        }
        Ok(MediaInfo {
            title,
            duration,
            chapters,
        })
    }
}

mod mp3 {
    use super::*;

    fn syncsafe(b: &[u8]) -> u32 {
        b.iter().fold(0u32, |a, &x| (a << 7) | (x & 0x7F) as u32)
    }

    fn text_frame(d: &[u8]) -> String {
        if d.is_empty() {
            return String::new();
        }
        let enc = d[0];
        let t = &d[1..];
        let s = match enc {
            1 | 2 => {
                let (be, body) = match t.get(..2) {
                    Some([0xFF, 0xFE]) => (false, &t[2..]),
                    Some([0xFE, 0xFF]) => (true, &t[2..]),
                    _ => (enc == 2, t),
                };
                let u: Vec<u16> = body
                    .chunks_exact(2)
                    .map(|c| if be { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) })
                    .collect();
                String::from_utf16_lossy(&u)
            }
            3 => String::from_utf8_lossy(t).into_owned(),
            _ => t.iter().map(|&c| c as char).collect(),
        };
        s.trim_end_matches('\0').to_string()
    }

    struct Frame<'a> {
        id: [u8; 4],
        data: &'a [u8],
    }

    fn frames(buf: &[u8], ver: u8) -> Vec<Frame<'_>> {
        let mut out = Vec::new();
        let mut o = 0;
        while o + 10 <= buf.len() && buf[o] != 0 {
            let id: [u8; 4] = buf[o..o + 4].try_into().unwrap();
            let size = if ver >= 4 {
                syncsafe(&buf[o + 4..o + 8])
            } else {
                u32::from_be_bytes(buf[o + 4..o + 8].try_into().unwrap())
            } as usize;
            let end = (o + 10 + size).min(buf.len());
            out.push(Frame {
                id,
                data: &buf[o + 10..end],
            });
            o = end;
        }
        out
    }

    /// ID3v2 `APIC`: encoding, MIME\0, picture type, description\0, data
    pub fn cover(f: &mut File) -> Result<Option<super::Cover>, String> {
        let mut hdr = [0u8; 10];
        f.read_exact(&mut hdr).map_err(|e| e.to_string())?;
        if &hdr[..3] != b"ID3" {
            return Ok(None);
        }
        let ver = hdr[3];
        let mut buf = vec![0u8; syncsafe(&hdr[6..10]) as usize];
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        for fr in frames(&buf, ver) {
            if &fr.id != b"APIC" || fr.data.len() < 4 {
                continue;
            }
            let d = fr.data;
            let enc = d[0];
            let Some(z) = d[1..].iter().position(|&c| c == 0) else { continue };
            let mut o = 1 + z + 1 + 1; // MIME, its NUL, picture type
            // description: NUL-terminated, two NULs for UTF-16
            if enc == 1 || enc == 2 {
                while o + 1 < d.len() && !(d[o] == 0 && d[o + 1] == 0) {
                    o += 2;
                }
                o += 2;
            } else {
                while o < d.len() && d[o] != 0 {
                    o += 1;
                }
                o += 1;
            }
            if o < d.len() {
                let b = d[o..].to_vec();
                let mime = super::image_mime(&b);
                return Ok(Some((b, mime)));
            }
        }
        Ok(None)
    }

    pub fn probe(f: &mut File) -> Result<MediaInfo, String> {
        let mut hdr = [0u8; 10];
        f.read_exact(&mut hdr).map_err(|e| e.to_string())?;
        let mut chapters = Vec::new();
        let mut title = None;
        if &hdr[..3] == b"ID3" {
            let ver = hdr[3];
            let size = syncsafe(&hdr[6..10]) as usize;
            let mut buf = vec![0u8; size];
            f.read_exact(&mut buf).map_err(|e| e.to_string())?;
            let mut raw = Vec::new();
            for fr in frames(&buf, ver) {
                match &fr.id {
                    b"TIT2" => title = Some(text_frame(fr.data)),
                    b"CHAP" => {
                        let d = fr.data;
                        let Some(z) = d.iter().position(|&c| c == 0) else { continue };
                        let o = z + 1;
                        if d.len() < o + 16 {
                            continue;
                        }
                        let start = u32::from_be_bytes(d[o..o + 4].try_into().unwrap()) as i64;
                        let end = u32::from_be_bytes(d[o + 4..o + 8].try_into().unwrap()) as i64;
                        let mut t = String::new();
                        for sub in frames(&d[o + 16..], ver) {
                            if &sub.id == b"TIT2" {
                                t = text_frame(sub.data);
                            }
                        }
                        raw.push((start, end, t));
                    }
                    _ => {}
                }
            }
            raw.sort_by_key(|r| r.0);
            for (i, (s, e, t)) in raw.into_iter().enumerate() {
                chapters.push(Chapter {
                    id: i as i64,
                    start: ff_time(s, 1, 1000),
                    end: ff_time(e, 1, 1000),
                    title: t,
                });
            }
        }
        Ok(MediaInfo {
            title,
            duration: None,
            chapters,
        })
    }
}
