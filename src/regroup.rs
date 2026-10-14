//! `--respect-grouping`: port of SubPlz `align.nc_align`. Regroups finished
//! subtitles so every cue is one or more whole lines of the book script
//! (its sentence-split text), timed from the subtitles it covers. A
//! divide-and-conquer DP picks, around the middle script line, the best
//! mapping of up to `max_merge_count` script lines onto subtitle runs.

use crate::batch::Segment;

struct Sub {
    text: String,
    start: String,
    end: String,
}

/// `read_subtitles` on SubPlz's own SRT/VTT output.
fn read_subtitles(src: &str) -> Vec<Sub> {
    // Python text mode: universal newlines
    let src = src.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines = src.split('\n');
    // file.readlines() yields no final empty line after a trailing "\n"
    let mut all: Vec<&str> = lines.by_ref().collect();
    if all.last() == Some(&"") {
        all.pop();
    }
    let mut it = all.into_iter();
    let mut subs: Vec<Sub> = Vec::new();
    let Some(first) = it.next() else {
        return subs;
    };
    if first == "WEBVTT" {
        it.next();
    }
    let mut last_sub = " ".to_string();
    while let Some(line) = it.next() {
        let Some((start, end)) = find_times(line) else {
            continue;
        };
        let mut text = Vec::new();
        let mut line = it.next();
        while let Some(l) = line.filter(|l| !l.is_empty()) {
            text.push(remove_tags(l));
            line = it.next();
            if line == Some("") {
                break;
            }
        }
        let sub = text.join(" ");
        let start = start.replace(',', ".");
        let end = end.replace(',', ".");
        if !sub.is_empty() && last_sub != sub && sub != " " && sub != "[音楽]" {
            last_sub = sub.clone();
            subs.push(Sub { text: sub, start, end });
        } else if last_sub == sub && !subs.is_empty() {
            subs.last_mut().unwrap().end = end;
        }
    }
    subs
}

fn remove_tags(line: &str) -> String {
    let mut out = String::new();
    let mut rest = line;
    while let Some(i) = rest.find('<') {
        match rest[i..].find('>') {
            Some(j) => {
                out.push_str(&rest[..i]);
                rest = &rest[i + j + 1..];
            }
            None => break,
        }
    }
    out.push_str(rest);
    out
}

/// `\d\d:\d\d:\d\d.\d\d\d` (long) or `\d\d:\d\d.\d\d\d` (short), `.` = any char
fn ts_at(c: &[char], i: usize, long: bool) -> Option<usize> {
    let pat: &[u8] = if long { b"dd:dd:dd.ddd" } else { b"dd:dd.ddd" };
    if i + pat.len() > c.len() {
        return None;
    }
    for (k, &p) in pat.iter().enumerate() {
        let ch = c[i + k];
        let ok = match p {
            b'd' => ch.is_ascii_digit(),
            b'.' => ch != '\n',
            _ => ch == p as char,
        };
        if !ok {
            return None;
        }
    }
    Some(i + pat.len())
}

/// First match of SubPlz's timestamp-pair regex (leftmost, alternatives in order).
fn find_times(line: &str) -> Option<(String, String)> {
    let c: Vec<char> = line.chars().collect();
    let arrow: Vec<char> = " --> ".chars().collect();
    // (start long, end long) in the regex's alternative order
    const ALTS: [(bool, bool); 3] = [(true, true), (false, false), (false, true)];
    for i in 0..c.len() {
        for (sl, el) in ALTS {
            let Some(a) = ts_at(&c, i, sl) else { continue };
            if c.len() < a + arrow.len() || c[a..a + arrow.len()] != arrow[..] {
                continue;
            }
            let b = a + arrow.len();
            if let Some(e) = ts_at(&c, b, el) {
                return Some((c[i..a].iter().collect(), c[b..e].iter().collect()));
            }
        }
    }
    None
}

fn to_float(t: &str) -> f64 {
    let mut total = 0.0;
    for (i, comp) in t.split(':').rev().enumerate() {
        total += comp.parse::<f64>().unwrap_or(0.0) * 60f64.powi(i as i32);
    }
    total
}

/// Bit-parallel LCS (Hyyrö) of a fixed pattern against a text fed in pieces.
struct Lcs {
    /// open addressing: char -> row in `rows`
    keys: Vec<u32>,
    slot_row: Vec<u32>,
    rows: Vec<u64>,
    words: usize,
    len: usize,
    s: Vec<u64>,
}

impl Lcs {
    fn new(pattern: &[char]) -> Lcs {
        let words = pattern.len().div_ceil(64).max(1);
        let cap = (pattern.len() * 2).next_power_of_two().max(16);
        let mut l = Lcs {
            keys: vec![u32::MAX; cap],
            slot_row: vec![0; cap],
            rows: Vec::new(),
            words,
            len: pattern.len(),
            s: vec![!0u64; words],
        };
        for (i, &c) in pattern.iter().enumerate() {
            let slot = l.slot(c);
            if l.keys[slot] == u32::MAX {
                l.keys[slot] = c as u32;
                l.slot_row[slot] = (l.rows.len() / words) as u32;
                l.rows.resize(l.rows.len() + words, 0);
            }
            let row = l.slot_row[slot] as usize;
            l.rows[row * words + i / 64] |= 1u64 << (i % 64);
        }
        l
    }

    fn slot(&self, c: char) -> usize {
        let mask = self.keys.len() - 1;
        let mut i = (c as u32).wrapping_mul(0x9E37_79B9) as usize & mask;
        while self.keys[i] != u32::MAX && self.keys[i] != c as u32 {
            i = (i + 1) & mask;
        }
        i
    }

    fn feed(&mut self, text: &[char]) {
        for &c in text {
            let slot = self.slot(c);
            if self.keys[slot] == u32::MAX {
                continue; // no match bits: the state is unchanged
            }
            let row = &self.rows[self.slot_row[slot] as usize * self.words..][..self.words];
            let mut carry = 0u64;
            for (s, &r) in self.s.iter_mut().zip(row) {
                let u = *s & r;
                let (t1, c1) = s.overflowing_add(u);
                let (t2, c2) = t1.overflowing_add(carry);
                carry = (c1 || c2) as u64;
                *s = t2 | (*s - u);
            }
        }
    }

    fn lcs(&self) -> usize {
        let mut n = 0;
        for (w, s) in self.s.iter().enumerate() {
            let bits = self.len.saturating_sub(w * 64).min(64);
            let mask = if bits == 64 { !0 } else { (1u64 << bits) - 1 };
            n += (!s & mask).count_ones() as usize;
        }
        n
    }
}

/// rapidfuzz `fuzz.ratio` from the lengths and the LCS
fn ratio(la: usize, lb: usize, lcs: usize) -> f64 {
    let lensum = la + lb;
    if lensum == 0 {
        return 100.0;
    }
    let dist = lensum - 2 * lcs;
    (1.0 - dist as f64 / lensum as f64) * 100.0
}

#[derive(Clone, Copy)]
struct Memo {
    score: f64,
    used_sub: usize,
    used_script: usize,
}

/// One `recursively_find_match` call: memo over its script window x sub range.
struct Dp<'a> {
    script: &'a [Vec<char>],
    subs: &'a [Vec<char>],
    max_merge: usize,
    sp0: usize,
    sub0: usize,
    width: usize,
    memo: Vec<Option<Memo>>,
    /// per script position: (best score, sub position)
    best: Vec<Option<(f64, usize)>>,
}

impl Dp<'_> {
    fn idx(&self, sp: usize, sub: usize) -> usize {
        (sp - self.sp0) * self.width + (sub - self.sub0)
    }

    /// `get_best_sub_n`; the joined subs grow one sub at a time, so one LCS
    /// pass scores every count
    fn best_sub_n(&mut self, sp: usize, n_script: usize, last_sp: usize, sub: usize, max_subs: usize, last_sub: usize) -> (f64, usize) {
        let mut best = 0.0;
        let mut best_used = 1;
        let end = self.script.len().min(sp + n_script);
        let line: Vec<char> = self.script[sp.min(end)..end].concat();
        let mut lcs = Lcs::new(&line);
        let mut base_len = 0;
        let remaining = last_sub as i64 - sub as i64;
        let top = (max_subs as i64).min(remaining);
        for n_sub in 1..=top.max(0) as usize {
            // get_base clamps to the end of the subs
            if let Some(piece) = self.subs.get(sub + n_sub - 1) {
                lcs.feed(piece);
                base_len += piece.len();
            }
            let cur = ratio(base_len, line.len(), lcs.lcs()) / 100.0 * line.len().min(base_len) as f64;
            let tot = cur + self.score(sp + n_script, last_sp, sub + n_sub, last_sub);
            if tot > best {
                best = tot;
                best_used = n_sub;
            }
        }
        (best, best_used)
    }

    /// `calc_best_score`
    fn score(&mut self, sp: usize, last_sp: usize, sub: usize, last_sub: usize) -> f64 {
        if sp >= self.script.len() || sub >= self.subs.len() {
            return 0.0;
        }
        let k = self.idx(sp, sub);
        if let Some(m) = self.memo[k] {
            return m.score;
        }
        let mm = self.max_merge;
        let (mut best, mut best_sub, mut best_script) = (0.0, 1, 1);
        let remaining = last_sp as i64 - sp as i64;
        let top = (mm as i64).min(remaining);
        for n_script in 1..=top.max(0) as usize {
            let max_subs = if n_script == 1 { mm } else { 1 };
            let (s, u) = self.best_sub_n(sp, n_script, last_sp, sub, max_subs, last_sub);
            if s > best {
                best = s;
                best_sub = u;
                best_script = n_script;
            }
        }
        if best_script > 1 {
            // one more fitting
            let (s, u) = self.best_sub_n(sp, best_script, last_sp, sub, mm, last_sub);
            if s > best {
                best = s;
                best_sub = u;
            }
        }
        self.memo[k] = Some(Memo {
            score: best,
            used_sub: best_sub,
            used_script: best_script,
        });
        let b = &mut self.best[sp - self.sp0];
        if best >= b.map_or(0.0, |b| b.0) {
            *b = Some((best, sub));
        }
        best
    }

    fn best_path(&self, mut sp: usize, n: usize, last_sp: usize, last_sub: usize) -> Vec<(usize, usize)> {
        let mut sub = self.best[sp - self.sp0].expect("scored script position").1;
        let mut out = Vec::new();
        let mut i = 0;
        while i < n && sp < last_sp && sub < last_sub {
            out.push((sp, sub));
            let d = self.memo[self.idx(sp, sub)].expect("scored state");
            sub += d.used_sub;
            sp += d.used_script;
            i += 1;
        }
        out
    }
}

/// (script_pos, script lines used, sub_pos, subs used)
type Match = (usize, usize, usize, usize);

/// `recursively_find_match`. SubPlz clears its memo on every call, so the
/// two halves are independent and run in parallel.
fn find(script: &[Vec<char>], subs: &[Vec<char>], mm: usize, first_sp: usize, last_sp: usize, first_sub: usize, last_sub: usize) -> Vec<Match> {
    if first_sp == last_sp || first_sub == last_sub {
        return vec![];
    }
    let ctx = mm * 2;
    let mid = (first_sp + last_sp) / 2;
    let start = first_sp.max(mid.saturating_sub(ctx));
    let end = (mid + ctx).min(last_sp);
    let width = last_sub - first_sub + 1;
    let mut dp = Dp {
        script,
        subs,
        max_merge: mm,
        sp0: start,
        sub0: first_sub,
        width,
        memo: vec![None; (end - start + 1) * width],
        best: vec![None; end - start + 1],
    };
    for sp in (start..end).rev() {
        for sub in (first_sub..last_sub).rev() {
            dp.score(sp, end, sub, last_sub);
        }
    }
    let path = dp.best_path(start, end - start, end, last_sub);
    let Some(&(sp, sub)) = path.iter().take_while(|p| p.0 <= mid).last() else {
        return vec![];
    };
    let m = dp.memo[dp.idx(sp, sub)].unwrap();
    drop(dp);
    let (n_script, n_sub) = (m.used_script, m.used_sub);
    let (mut left, right) = rayon::join(
        || find(script, subs, mm, first_sp, sp, first_sub, sub),
        || find(script, subs, mm, sp + n_script, last_sp, sub + n_sub, last_sub),
    );
    left.push((sp, n_script, sub, n_sub));
    left.extend(right);
    left
}

/// `nc_align(script, subs_file, max_merge_count)`; `subs` is the rendered
/// SRT/VTT the sync step wrote, `script` the book's sentence lines.
pub fn nc_align(script: &[String], subs: &str, max_merge_count: usize) -> Vec<Segment> {
    nc_align_with(script, subs, max_merge_count, false)
}

/// [`nc_align`]; with `keep_lines` every script line stays a cue of its own
/// (subtitles being retimed): a group of lines that matched the same cues
/// shares their time span by length of text.
pub fn nc_align_with(script: &[String], subs: &str, max_merge_count: usize, keep_lines: bool) -> Vec<Segment> {
    let subs = read_subtitles(subs);
    let script_c: Vec<Vec<char>> = script.iter().map(|s| s.chars().collect()).collect();
    let subs_c: Vec<Vec<char>> = subs.iter().map(|s| s.text.chars().collect()).collect();
    let result = find(&script_c, &subs_c, max_merge_count, 0, script_c.len(), 0, subs_c.len());

    let mut out = Vec::new();
    for i in 0..result.len() {
        let (sp, sub) = if i == 0 { (0, 0) } else { (result[i].0, result[i].2) };
        let (n_script, n_sub) = match result.get(i + 1) {
            Some(next) => (next.0 - sp, next.2 - sub),
            None => (script_c.len() - sp, subs.len() - sub),
        };
        let end = script.len().min(sp + n_script);
        // python index: subs[sub + n_sub - 1] wraps to the end when negative
        let last = (sub + n_sub) as i64 - 1;
        let last = if last < 0 { subs.len() as i64 + last } else { last } as usize;
        let lines = &script[sp.min(end)..end];
        if keep_lines && lines.len() > 1 && last >= sub {
            out.extend(split_span(lines, &subs[sub..=last]));
            continue;
        }
        out.push(Segment {
            text: lines.concat(),
            start: to_float(&subs[sub].start),
            end: to_float(&subs[last].end),
        });
    }
    out
}

/// Spread script lines over the cues they matched: the cues' text laid out
/// along their times (gaps between cues skipped), each line taking its
/// share by length.
fn split_span(lines: &[String], cues: &[Sub]) -> Vec<Segment> {
    let lens: Vec<f64> = cues.iter().map(|c| c.text.chars().count().max(1) as f64).collect();
    let total: f64 = lens.iter().sum();
    let at = |pos: f64| -> f64 {
        let mut acc = 0.0;
        for (c, l) in cues.iter().zip(&lens) {
            if pos <= acc + l {
                let (s, e) = (to_float(&c.start), to_float(&c.end));
                return s + (pos - acc) / l * (e - s);
            }
            acc += l;
        }
        to_float(&cues.last().unwrap().end)
    };
    let weights: Vec<f64> = lines.iter().map(|l| l.chars().count().max(1) as f64).collect();
    let wsum: f64 = weights.iter().sum();
    let mut done = 0.0;
    lines
        .iter()
        .zip(&weights)
        .map(|(l, w)| {
            let (a, b) = (done / wsum * total, (done + w) / wsum * total);
            done += w;
            // round like the rest of the pipeline's times
            let r = |t: f64| (t * 1000.0).round() / 1000.0;
            Segment { text: l.clone(), start: r(at(a)), end: r(at(b)) }
        })
        .collect()
}
