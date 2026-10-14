//! Port of `subplz/align.py::shift_align` (move short punctuation-delimited
//! fragments between neighbouring cues) and `ats.main.write_srt`.
//!
//! The Python code mutates Segment objects that are shared between several
//! lists (`segments`, `new_segments`, `adjusted_segments`), so segments live
//! in an arena here and lists hold ids, preserving the aliasing exactly.

use crate::batch::{py_strip, Segment};
use std::collections::HashSet;
use std::sync::LazyLock;

pub const START_PUNC: &str = "『「(（《｟[{\"'“¿'“\"¿([{-『「（〈《〔【｛［｟＜<‘“〝※";
pub const END_PUNC: &str = "'\"・.。!！?？:：”>＞⦆)]}』」）〉》〕】｝］’〟／＼～〜~;；─―–-➡";
pub const OTHER_PUNC: &str = "＊　,，、…";

struct Arena {
    segs: Vec<Segment>,
}

impl Arena {
    fn new_seg(&mut self, text: String, start: f64, end: f64) -> usize {
        self.segs.push(Segment { text, start, end });
        self.segs.len() - 1
    }
}

static START_SET: LazyLock<HashSet<char>> = LazyLock::new(|| START_PUNC.chars().collect());
static END_SET: LazyLock<HashSet<char>> = LazyLock::new(|| END_PUNC.chars().collect());
static PUNCT_SET: LazyLock<HashSet<char>> =
    LazyLock::new(|| START_PUNC.chars().chain(END_PUNC.chars()).chain(OTHER_PUNC.chars()).collect());

#[inline]
fn is_punct(c: char) -> bool {
    PUNCT_SET.contains(&c)
}

fn has_ending_punctuation(s: &[char]) -> bool {
    s.iter().any(|c| END_SET.contains(c))
}

fn count_non_punctuation(s: &[char]) -> usize {
    s.iter().filter(|&&c| !is_punct(c)).count()
}

fn has_double_comma(a: &[char], b: &[char]) -> bool {
    *a.last().expect("IndexError") == '、' && *b.last().expect("IndexError") == '、'
}

fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

fn s(c: &[char]) -> String {
    c.iter().collect()
}

const RSTRIP: [char; 5] = ['\u{3000}', ' ', '\t', '\r', '\n'];

pub fn shift_align(input: Vec<Segment>) -> Vec<Segment> {
    let n = input.len();
    let mut ar = Arena { segs: input };
    let segments: Vec<usize> = (0..n).collect();

    let mut new_segments: Vec<usize> = Vec::with_capacity(n);
    for (i, &seg) in segments.iter().enumerate() {
        let mut text = chars(&ar.segs[seg].text);
        let indices: Vec<usize> = text.iter().enumerate().filter(|(_, &c)| is_punct(c)).map(|(k, _)| k).collect();
        if indices.is_empty() {
            new_segments.push(seg);
            continue;
        }
        // find_index_with_non_punctuation_start(indices)[0] == indices[0]
        let start_index = indices[0];
        let non_punc_count = count_non_punctuation(&text[..start_index]);
        if non_punc_count == 0 || count_non_punctuation(&text[start_index + 1..]) == 0 {
            new_segments.push(seg);
            continue;
        }
        if non_punc_count <= 2 {
            let prev_clean: Vec<char> = match new_segments.last() {
                Some(&p) => chars(ar.segs[p].text.trim_end_matches(RSTRIP)),
                None => Vec::new(),
            };
            if i > 0
                && !new_segments.is_empty()
                && !prev_clean.is_empty()
                && !has_ending_punctuation(&prev_clean[prev_clean.len() - 1..])
                && !has_double_comma(&chars(&ar.segs[*new_segments.last().unwrap()].text), &text[..start_index + 1])
            {
                let prev = new_segments.pop().unwrap();
                ar.segs[prev].text.push_str(&s(&text[..start_index + 1]));
                new_segments.push(prev);
                text = text[start_index + 1..].to_vec();
            }
        }
        let (st, en) = (ar.segs[seg].start, ar.segs[seg].end);
        let id = ar.new_seg(s(&text), st, en);
        new_segments.push(id);
    }

    let mut final_segments: Vec<usize> = Vec::with_capacity(n);
    let mut i = 0;
    while i < new_segments.len() {
        let seg = new_segments[i];
        let mut text = chars(&ar.segs[seg].text);
        let indices: Vec<usize> = text.iter().enumerate().filter(|(_, &c)| is_punct(c)).map(|(k, _)| k).collect();
        if let Some(&last_index) = indices.last() {
            // find_index_with_non_punctuation_end(indices)[-1] == indices[-1]
            let non_punc_count = count_non_punctuation(&text[last_index..]);
            if non_punc_count == 0 || text.len() - last_index == text.len() {
                final_segments.push(seg);
                i += 1;
                continue;
            }
            if count_non_punctuation(&text[last_index..]) <= 2 {
                let next_clean: Vec<char> = if i + 1 < new_segments.len() {
                    chars(ar.segs[new_segments[i + 1]].text.trim_start_matches(RSTRIP))
                } else {
                    Vec::new()
                };
                if i + 1 < new_segments.len()
                    && !next_clean.is_empty()
                    && !has_ending_punctuation(&next_clean[..1])
                    && !has_double_comma(&next_clean[..1], &text[last_index + 1..])
                {
                    // python uses the ORIGINAL list here: segments[i + 1]
                    let next_segment = segments[i + 1];
                    let moved = s(&text[last_index + 1..]);
                    let old = std::mem::take(&mut ar.segs[next_segment].text);
                    ar.segs[next_segment].text = moved + &old;
                    text.truncate(last_index + 1);
                    let (st, en) = (ar.segs[seg].start, ar.segs[seg].end);
                    let id = ar.new_seg(s(&text), st, en);
                    final_segments.push(id);
                    new_segments[i + 1] = next_segment;
                    i += 1;
                    continue;
                }
            }
        }
        let (st, en) = (ar.segs[seg].start, ar.segs[seg].end);
        let id = ar.new_seg(s(&text), st, en);
        final_segments.push(id);
        i += 1;
    }

    let adjusted = double_check_misaligned_pairs(&mut ar, final_segments);
    adjusted
        .into_iter()
        .map(|id| {
            let sg = &ar.segs[id];
            Segment {
                text: py_strip(&sg.text).to_string(),
                start: sg.start,
                end: sg.end,
            }
        })
        .collect()
}

fn double_check_misaligned_pairs(ar: &mut Arena, segments: Vec<usize>) -> Vec<usize> {
    if segments.len() < 2 {
        return segments;
    }
    let mut adjusted: Vec<usize> = Vec::with_capacity(segments.len());
    for (index, &seg) in segments.iter().enumerate() {
        // handle_specific_pattern: r"」「(.{1,2})、$" | r"」「(.{1})、$"
        let text = chars(&ar.segs[seg].text);
        if let Some(p) = specific_pattern_start(&text) {
            if index < segments.len() - 1 {
                let end = if text.last() == Some(&'\n') { text.len() - 1 } else { text.len() };
                let matched = s(&text[p..end]);
                let nx = segments[index + 1];
                let old = std::mem::take(&mut ar.segs[nx].text);
                ar.segs[nx].text = matched + &old;
                ar.segs[seg].text = s(&text[..p]);
            }
        }
        // handle_starting_punctuation
        let text = chars(&ar.segs[seg].text);
        if !text.is_empty() && END_SET.contains(&text[0]) && index > 0 {
            let prev = *adjusted.last().unwrap();
            ar.segs[prev].text.push(text[0]);
            ar.segs[seg].text = s(&text[1..]);
        }
        // handle_ending_punctuation
        let text = chars(&ar.segs[seg].text);
        if !text.is_empty() && START_SET.contains(text.last().unwrap()) && index < segments.len() - 1 {
            let nx = segments[index + 1];
            let old = std::mem::take(&mut ar.segs[nx].text);
            ar.segs[nx].text = format!("{}{}", text[text.len() - 1], old);
            ar.segs[seg].text = s(&text[..text.len() - 1]);
        }
        adjusted.push(seg);
    }
    adjusted
}

/// Leftmost start of `」「.{1,2}、$` (python `$`: end, or before a final \n).
fn specific_pattern_start(t: &[char]) -> Option<usize> {
    let end = if t.last() == Some(&'\n') { t.len() - 1 } else { t.len() };
    for width in [2usize, 1] {
        let len = 2 + width + 1;
        if end < len {
            continue;
        }
        let p = end - len;
        if t[p] == '」' && t[p + 1] == '「' && t[p + 2..p + 2 + width].iter().all(|&c| c != '\n') && t[end - 1] == '、'
        {
            return Some(p);
        }
    }
    None
}

/// CPython float `divmod(x, y)` for y > 0.
fn py_divmod(vx: f64, wx: f64) -> (f64, f64) {
    let mut m = vx % wx;
    let mut div = (vx - m) / wx;
    if m != 0.0 {
        if (wx < 0.0) != (m < 0.0) {
            m += wx;
            div -= 1.0;
        }
    } else {
        m = 0.0f64.copysign(wx);
    }
    let floordiv = if div != 0.0 {
        let mut f = div.floor();
        if div - f > 0.5 {
            f += 1.0;
        }
        f
    } else {
        0.0f64.copysign(vx / wx)
    };
    (floordiv, m)
}

fn pad0(s: String, width: usize) -> String {
    let n = s.chars().count();
    if n >= width {
        s
    } else {
        "0".repeat(width - n) + &s
    }
}

/// `ats.main.sexagesimal`
pub fn sexagesimal(secs: f64, use_comma: bool) -> String {
    let (mm, ss) = py_divmod(secs, 60.0);
    let (hh, mm) = py_divmod(mm, 60.0);
    let r = format!(
        "{}:{}:{}",
        pad0(format!("{hh:.0}"), 2),
        pad0(format!("{mm:.0}"), 2),
        pad0(format!("{ss:.3}"), 6)
    );
    if use_comma {
        r.replace('.', ",")
    } else {
        r
    }
}

pub fn write_srt(segments: &[Segment]) -> String {
    segments
        .iter()
        .enumerate()
        .map(|(i, s)| {
            format!(
                "{}\n{} --> {}\n{}",
                i + 1,
                sexagesimal(s.start, true),
                sexagesimal(s.end, true),
                s.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// `ats.main.write_vtt`
pub fn write_vtt(segments: &[Segment]) -> String {
    let body = segments
        .iter()
        .map(|s| format!("{} --> {}\n{}", sexagesimal(s.start, false), sexagesimal(s.end, false), s.text))
        .collect::<Vec<_>>()
        .join("\n\n");
    format!("WEBVTT\n\n{body}")
}
