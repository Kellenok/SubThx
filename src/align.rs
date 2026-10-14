//! Port of `ats/align.py`: map cleaned book text onto ASR cues via Levenshtein
//! opcodes, then snap cue boundaries in the original text.
//!
//! Python index semantics (negative indices wrap, slices clamp) are reproduced
//! on purpose: the heuristics rely on them in edge cases.

use crate::lang::Lang;
use crate::lev;
use std::collections::HashSet;

/// One alignment piece: `[start, end, cue_index]` in a paragraph's original text.
pub type Piece = [i64; 3];
/// Per text paragraph, the pieces assigned to cues.
pub type Alignment = Vec<Vec<Piece>>;

pub struct Punct {
    pub prepend: HashSet<char>,
    pub append: HashSet<char>,
    pub nopend: HashSet<char>,
}

/// Python `t[i]` with negative wrap-around.
#[inline]
fn at(t: &[char], i: i64) -> char {
    let n = t.len() as i64;
    let j = if i < 0 { i + n } else { i };
    assert!(j >= 0 && j < n, "python IndexError: index {i} len {n}");
    t[j as usize]
}

/// Python slice bounds `t[a:b]` -> clamped `(lo, hi)` with lo <= hi.
#[inline]
pub fn py_slice(n: usize, a: i64, b: i64) -> (usize, usize) {
    let n = n as i64;
    let fix = |x: i64| -> i64 {
        if x < 0 {
            (x + n).max(0)
        } else {
            x.min(n)
        }
    };
    let (lo, hi) = (fix(a), fix(b));
    (lo as usize, hi.max(lo) as usize)
}

pub fn align(lang: Lang, transcript: &[&str], text: &[&str], p: &Punct) -> Alignment {
    let transcript_clean: Vec<Vec<char>> = transcript
        .iter()
        .map(|s| lang.clean(s).chars().collect())
        .collect();
    let transcript_joined: Vec<char> = transcript_clean.concat();
    let text_clean: Vec<Vec<char>> = text.iter().map(|s| lang.clean(s).chars().collect()).collect();
    let text_joined: Vec<char> = text_clean.concat();
    if text_joined.is_empty() || transcript_joined.is_empty() {
        return Vec::new();
    }

    let ops = lev::opcodes_chars(&text_joined, &transcript_joined);
    let mut xs = vec![0i64];
    let mut ys = vec![0i64];
    for op in &ops {
        xs.push(op.src_end as i64);
        ys.push(op.dest_end as i64);
    }
    let mut coords: Vec<(i64, i64)> = vec![(xs[0], ys[0])];
    for i in 1..xs.len() - 1 {
        let (cx, cy) = *coords.last().unwrap();
        let (dx1, dy1) = (xs[i] - cx, ys[i] - cy);
        let (dx2, dy2) = (xs[i + 1] - xs[i], ys[i + 1] - ys[i]);
        if dx1 * dy2 == dx2 * dy1 && (dx1 * dx2 + dy1 * dy2) >= 0 {
            continue;
        }
        coords.push((xs[i], ys[i]));
    }
    coords.push((*xs.last().unwrap(), *ys.last().unwrap()));

    let text_chars: Vec<Vec<char>> = text.iter().map(|s| s.chars().collect()).collect();
    let mut segments = align_sub(&coords, &text_clean, &transcript_clean, 2);
    fix(lang, &text_chars, &text_clean, &mut segments);
    smart_snap_boundaries(&text_chars, &mut segments, &p.append);
    fix_punc(&text_chars, &mut segments, p);
    segments
}

fn align_sub(coords: &[(i64, i64)], text: &[Vec<char>], subs: &[Vec<char>], thing: i64) -> Alignment {
    let ntext = text.len();
    let mut cur0: usize = 0;
    let mut cur1: usize = 0;
    let mut pos = [0i64, 0];
    let mut toff: i64 = 0;
    let mut p = [0i64, 0];
    let mut gaps = [0i64, 0];
    let mut segments: Alignment = vec![Vec::new()];
    let mut off: i64 = 0;

    let finish = |mut s: Alignment| {
        s.truncate(ntext);
        s
    };

    for &(c0, c1) in coords {
        let isgap = c0 - p[0] == 0 || c1 - p[1] == 0;

        while cur1 < subs.len() && pos[1] + subs[cur1].len() as i64 <= c1 {
            if cur0 >= ntext {
                return finish(segments);
            }
            let sublen = subs[cur1].len() as i64;
            pos[1] += sublen;
            if isgap {
                gaps[0] += (pos[1] - p[1]).max(0);
                gaps[1] += (pos[0] - p[0]).max(0);
            }
            let diff = sublen + gaps[1] - gaps[0];
            if diff > sublen.div_euclid(4) {
                let mut target = toff + diff + off;
                off = 0;
                while cur0 < ntext && target >= text[cur0].len() as i64 {
                    let (start, end) = (toff, text[cur0].len() as i64);
                    if end - start != 0 {
                        let (lo, hi) = py_slice(text[cur0].len(), start, end);
                        let region = &text[cur0][lo..hi];
                        let distinct = region
                            .iter()
                            .filter(|&&c| c != '。')
                            .collect::<HashSet<_>>()
                            .len() as i64;
                        let prev = segments.last_mut().unwrap();
                        if end - start < thing || distinct < thing {
                            if let Some(last) = prev.last_mut() {
                                last[1] = end;
                            } else {
                                prev.push([start, end, cur1 as i64]);
                            }
                        } else {
                            prev.push([start, end, cur1 as i64]);
                        }
                    }
                    segments.push(Vec::new());
                    pos[0] += end - start;
                    target -= text[cur0].len() as i64;
                    cur0 += 1;
                    toff = 0;
                }
                pos[0] += target - toff;
                segments.last_mut().unwrap().push([toff, target, cur1 as i64]);
                toff = target;
            } else {
                let half = (text[cur0].len() as i64).div_euclid(2);
                let prev = segments.last_mut().unwrap();
                if toff >= half && !prev.is_empty() {
                    prev.last_mut().unwrap()[1] += diff;
                    toff += diff;
                } else {
                    off += diff;
                }
            }
            cur1 += 1;
            gaps = [0, 0];
            p = [pos[0].max(p[0]), pos[1].max(p[1])];
        }

        if isgap {
            gaps[0] += c1 - p[1];
            gaps[1] += c0 - p[0];
        }
        p = [c0, c1];
    }
    finish(segments)
}

/// Map positions in the cleaned text back to the translated original text.
fn fix(lang: Lang, original: &[Vec<char>], edited: &[Vec<char>], segments: &mut Alignment) {
    for (l, s) in segments.iter_mut().enumerate() {
        let o: Vec<char> = lang
            .translate(&original[l].iter().collect::<String>())
            .chars()
            .collect();
        let e = &edited[l];
        let mut m: Vec<Option<i64>> = vec![None; e.len() + 1];
        let mut ei = 0usize;
        for (oi, &oc) in o.iter().enumerate() {
            if ei < e.len() && oc == e[ei] {
                m[ei] = Some(oi as i64);
                ei += 1;
            }
        }
        // the loop body ends with `oi += 1`, so afterwards oi == len(o)
        let last_oi = o.len() as i64;
        if ei >= m.len() {
            m.resize(ei + 1, None);
        }
        m[ei] = Some(last_oi);
        m[0] = Some(0);
        let mut last = 0usize;
        for i in 0..e.len() {
            if m[i].is_some() {
                last = i;
            } else {
                m[i] = m[last];
            }
        }
        let el = e.len() as i64;
        for f in s.iter_mut() {
            f[0] = m[f[0].clamp(0, el) as usize].expect("python KeyError in fix()");
            f[1] = m[f[1].clamp(0, el) as usize].expect("python KeyError in fix()");
        }
    }
}

fn is_kanji(c: char) -> bool {
    ('\u{4e00}'..='\u{9faf}').contains(&c)
}

fn is_ws(c: char) -> bool {
    c == '\u{3000}' || c == ' '
}

fn smart_snap_boundaries(text: &[Vec<char>], segments: &mut Alignment, append: &HashSet<char>) {
    let mut close: HashSet<char> = ['」', '』', '。', '、', '！', '？', '…', '・'].into_iter().collect();
    close.extend(append.iter().copied());
    let open: HashSet<char> = ['「', '『', '（', '“'].into_iter().collect();

    for (l, s) in segments.iter_mut().enumerate() {
        if s.is_empty() {
            continue;
        }
        let t = &text[l];
        let n = t.len() as i64;
        for k in 0..s.len() - 1 {
            let mut idx = s[k][1];
            while 0 < idx && idx <= n && (open.contains(&at(t, idx - 1)) || is_ws(at(t, idx - 1))) {
                idx -= 1;
                s[k][1] = idx;
            }
            let mut snapped = false;
            for delta in 1..=3 {
                let target_idx = idx + delta;
                if target_idx <= n {
                    let ch = at(t, target_idx - 1);
                    if close.contains(&ch) {
                        let (lo, hi) = py_slice(t.len(), idx, target_idx - 1);
                        if t[lo..hi].iter().all(|&c| close.contains(&c) || is_ws(c)) {
                            idx = target_idx;
                            s[k][1] = idx;
                            snapped = true;
                            break;
                        }
                    }
                }
            }
            if !snapped && idx < n && close.contains(&at(t, idx)) {
                s[k][1] = idx + 1;
                snapped = true;
            }
            if !snapped && 0 < idx && idx < n && is_kanji(at(t, idx - 1)) && is_kanji(at(t, idx)) {
                let mut forward = idx;
                while forward < n && is_kanji(at(t, forward)) {
                    forward += 1;
                }
                if forward - idx <= 2 {
                    s[k][1] = forward;
                } else {
                    let mut backward = idx;
                    while backward > 0 && is_kanji(at(t, backward - 1)) {
                        backward -= 1;
                    }
                    if idx - backward <= 2 {
                        s[k][1] = backward;
                    }
                }
            }
            s[k + 1][0] = s[k][1];
        }
    }
}

fn fix_punc(text: &[Vec<char>], segments: &mut Alignment, p: &Punct) {
    let (prepend, append, nopend) = (&p.prepend, &p.append, &p.nopend);
    for (l, s) in segments.iter_mut().enumerate() {
        if s.is_empty() {
            continue;
        }
        let t = &text[l];
        let n = t.len() as i64;
        for k in 0..s.len() {
            let fi = (k + 1).min(s.len() - 1);
            let connected = s[fi][0] == s[k][1];
            let mut lp = 0;
            loop {
                if lp > 20 {
                    break;
                }
                let p1 = s[k][1];
                if p1 < n && append.contains(&at(t, p1)) {
                    s[k][1] += 1;
                } else if prepend.contains(&at(t, p1 - 1)) {
                    s[k][1] -= 1;
                } else if (p1 > 0 && nopend.contains(&at(t, p1 - 1)))
                    || (p1 < n && nopend.contains(&at(t, p1)))
                    || (p1 < n - 1 && nopend.contains(&at(t, p1 + 1)))
                {
                    let (mut start, mut end) = (p1 - 1, p1);
                    if p1 < n - 1
                        && ((nopend.contains(&at(t, p1 + 1)) && 0x4E00 > at(t, p1) as u32)
                            || at(t, p1) as u32 > 0x9FAF)
                    {
                        end += 1;
                    }
                    while start > 0 && nopend.contains(&at(t, start)) {
                        start -= 1;
                    }
                    while end < n - 1 && nopend.contains(&at(t, end)) {
                        end += 1;
                    }
                    if prepend.contains(&at(t, start)) {
                        if p1 == start {
                            break;
                        }
                        s[k][1] = start;
                    } else if append.contains(&at(t, start)) {
                        if p1 == start + 1 {
                            break;
                        }
                        s[k][1] = start + 1;
                    } else if end < n && prepend.contains(&at(t, end)) {
                        if p1 == end {
                            break;
                        }
                        s[k][1] = end;
                    } else if end < n && append.contains(&at(t, end)) {
                        if p1 == end + 1 {
                            break;
                        }
                        s[k][1] = end + 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
                lp += 1;
            }
            if connected {
                s[fi][0] = s[k][1];
            }
        }
    }
}
