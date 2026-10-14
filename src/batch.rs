//! Port of SubPlz `sync.do_batch` / `_anchor_guided_align` / `_align_chapter_task`
//! and `ats.main.to_subs`.

use crate::align::{align, py_slice, Alignment, Punct};
use crate::lang::Lang;

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Cue {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Transcript {
    pub language: String,
    pub segments: Vec<Cue>,
}

/// One audio chapter going into a batch: its transcript and duration.
pub struct AudioPart<'a> {
    pub transcript: &'a Transcript,
    pub duration: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Segment {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// `_align_chapter_task`: align a batch of audio chapters against the
/// paragraphs of the matched book chapters (empty -> raw transcript).
pub fn align_chapter_task(ach: &[AudioPart], paragraphs: &[&str], offset: f64, p: &Punct) -> Vec<Segment> {
    if !paragraphs.is_empty() {
        do_batch(ach, paragraphs, p, offset)
    } else {
        let mut res = Vec::new();
        for a in ach {
            for seg in &a.transcript.segments {
                res.push(Segment {
                    text: seg.text.clone(),
                    start: seg.start + offset,
                    end: seg.end + offset,
                });
            }
        }
        res
    }
}

fn do_batch(ach: &[AudioPart], tcontent: &[&str], p: &Punct, offset: f64) -> Vec<Segment> {
    let mut acontent: Vec<Cue> = Vec::new();
    // python: boff = 0 (int); boff += a[1]
    let mut boff: Option<f64> = None;
    for a in ach {
        for c in &a.transcript.segments {
            let mut c = c.clone();
            if let Some(b) = boff {
                c.start += b;
                c.end += b;
            }
            acontent.push(c);
        }
        boff = Some(match boff {
            None => a.duration,
            Some(b) => b + a.duration,
        });
    }
    let language = Lang::from_code(&ach[0].transcript.language);
    let n = acontent.len();
    let m = tcontent.len();
    let asr: Vec<&str> = acontent.iter().map(|c| c.text.as_str()).collect();
    let alignment = if n * m > 40_000 && n >= 50 && m >= 50 {
        anchor_guided_align(&asr, tcontent, language, p)
    } else {
        align(language, &asr, tcontent, p)
    };
    to_subs(tcontent, &acontent, &alignment, offset)
}

fn anchor_guided_align(asr: &[&str], txt: &[&str], language: Lang, p: &Punct) -> Alignment {
    let n = asr.len();
    let m = txt.len();
    let clean_asr: Vec<String> = asr.iter().map(|s| language.clean(s)).collect();
    let clean_txt: Vec<String> = txt.iter().map(|s| language.clean(s)).collect();

    let mut asr_cumlens = vec![0usize; n + 1];
    for (i, s) in clean_asr.iter().enumerate() {
        asr_cumlens[i + 1] = asr_cumlens[i] + s.chars().count();
    }
    let mut txt_cumlens = vec![0usize; m + 1];
    for (i, s) in clean_txt.iter().enumerate() {
        txt_cumlens[i + 1] = txt_cumlens[i] + s.chars().count();
    }
    let asr_joined = clean_asr.concat();
    let txt_joined = clean_txt.concat();
    if asr_joined.is_empty() || txt_joined.is_empty() {
        return vec![Vec::new(); m];
    }
    let asr_len = asr_cumlens[n] as f64;
    let txt_len = txt_cumlens[m] as f64;

    // python str.count (non-overlapping) == 1, plus str.index (char offset)
    let unique_pos = |hay: &str, needle: &str| -> Option<usize> {
        let mut it = memchr::memmem::find_iter(hay.as_bytes(), needle.as_bytes());
        let first = it.next()?;
        if it.next().is_some() {
            return None;
        }
        Some(hay[..first].chars().count())
    };

    let mut raw_anchors = Vec::new();
    const STEP: usize = 25;
    let mut si = 15;
    while si + 15 < n {
        let sub: Vec<char> = clean_asr[si].chars().collect();
        if sub.len() >= 8 {
            for k in 0..sub.len() - 7 {
                let ngram: String = sub[k..k + 8].iter().collect();
                let Some(a_pos) = unique_pos(&asr_joined, &ngram) else { continue };
                let Some(t_pos) = unique_pos(&txt_joined, &ngram) else { continue };
                let r_a = a_pos as f64 / asr_len;
                let r_t = t_pos as f64 / txt_len;
                if (r_a - r_t).abs() < 0.035 {
                    let a_idx = asr_cumlens.partition_point(|&x| x <= a_pos);
                    let t_idx = txt_cumlens.partition_point(|&x| x <= t_pos);
                    raw_anchors.push((a_idx, t_idx));
                    break;
                }
            }
        }
        si += STEP;
    }

    let mut filtered = Vec::new();
    let (mut last_a, mut last_t) = (0usize, 0usize);
    for (a_idx, t_idx) in raw_anchors {
        if a_idx >= last_a + 15 && t_idx >= last_t + 10 {
            filtered.push((a_idx, t_idx));
            last_a = a_idx;
            last_t = t_idx;
        }
    }
    if filtered.is_empty() {
        return align(language, asr, txt, p);
    }

    let mut chunks = Vec::new();
    let (mut pa, mut pt) = (0, 0);
    for &(a, t) in &filtered {
        chunks.push(((pa, a), (pt, t)));
        pa = a;
        pt = t;
    }
    chunks.push(((pa, n), (pt, m)));

    let mut merged: Alignment = Vec::new();
    for ((a_start, a_end), (t_start, t_end)) in chunks {
        let chunk_asr = &asr[a_start.min(n)..a_end.min(n)];
        let chunk_txt = &txt[t_start.min(m)..t_end.min(m)];
        if chunk_asr.is_empty() || chunk_txt.is_empty() {
            merged.extend((t_start..t_end).map(|_| Vec::new()));
            continue;
        }
        let mut chunk_align = align(language, chunk_asr, chunk_txt, p);
        for seg_list in chunk_align.iter_mut() {
            for e in seg_list.iter_mut() {
                e[2] += a_start as i64;
            }
        }
        merged.extend(chunk_align);
    }
    while merged.len() < m {
        merged.push(Vec::new());
    }
    merged
}

fn to_subs(text: &[&str], subs: &[Cue], alignment: &Alignment, offset: f64) -> Vec<Segment> {
    // flatten to (start, end, cue, paragraph)
    let flat: Vec<(i64, i64, i64, usize)> = alignment
        .iter()
        .enumerate()
        .flat_map(|(i, a)| a.iter().map(move |t| (t[0], t[1], t[2], i)))
        .collect();
    let text_chars: Vec<Vec<char>> = text.iter().map(|s| s.chars().collect()).collect();
    let (mut start, mut end) = (0usize, 0usize);
    let mut segments = Vec::with_capacity(subs.len());
    for (si, s) in subs.iter().enumerate() {
        while end < flat.len() && flat[end].2 == si as i64 {
            end += 1;
        }
        let mut r = String::new();
        for a in &flat[start..end] {
            let t = &text_chars[a.3];
            let (lo, hi) = py_slice(t.len(), a.0, a.1);
            r.extend(&t[lo..hi]);
        }
        let text = if !py_strip(&r).is_empty() {
            r
        } else {
            format!("＊{}", s.text)
        };
        segments.push(Segment {
            text,
            start: s.start + offset,
            end: s.end + offset,
        });
        start = end;
    }
    segments
}

/// Python `str.isspace` per char.
pub fn py_isspace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

/// Python `str.strip()` (no args).
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}
