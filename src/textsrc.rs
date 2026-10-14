//! Book text sources, as SubPlz `files.get_chapters` reads them:
//! - .epub: chapters of paragraphs (see `epub`);
//! - .txt: one chapter, its non-empty stripped lines;
//! - subtitles (.srt/.vtt/.ass): cue text without timing/tags, split into
//!   sentences like pysbd (ja) + SubPlz's quote fixes, one chapter.
//!
//! Sentences of languages that write with spaces (European ones) come from
//! [`sentences_spaced`] instead: pysbd's Japanese rules would cut after
//! "Mr." or an initial and leave closing quotes to the next sentence.

use crate::batch::py_strip;
use std::path::Path;

pub const TEXT_EXTS: &[&str] = &["epub", "txt"];
pub const SUB_EXTS: &[&str] = &["srt", "vtt", "ass", "ssa"];

fn ext(p: &Path) -> String {
    p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default()
}

pub fn is_subtitle(p: &Path) -> bool {
    SUB_EXTS.contains(&ext(p).as_str())
}

pub fn is_text(p: &Path) -> bool {
    let e = ext(p);
    TEXT_EXTS.contains(&e.as_str()) || SUB_EXTS.contains(&e.as_str())
}

fn read_utf8(p: &Path) -> Result<String, String> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let s = String::from_utf8_lossy(&b).into_owned();
    Ok(s.strip_prefix('\u{feff}').map(str::to_string).unwrap_or(s))
}

/// Chapters of paragraphs for one text file.
pub fn load(p: &Path) -> Result<Vec<Vec<String>>, String> {
    Ok(load_titled(p)?.into_iter().map(|(_, c)| c).collect())
}

/// Chapters with display titles (epub chapter titles, else the file name).
pub fn load_titled(p: &Path) -> Result<Vec<(String, Vec<String>)>, String> {
    load_titled_in(p, "ja")
}

/// [`load_titled`] for a book in `lang` (ISO 639-1): subtitles are split
/// into that language's sentences.
pub fn load_titled_in(p: &Path, lang: &str) -> Result<Vec<(String, Vec<String>)>, String> {
    if ext(p) == "epub" {
        return Ok(crate::epub::Epub::open(p)?.chapters.into_iter().map(|c| (c.title, c.paragraphs)).collect());
    }
    let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
    Ok(load_plain(p, lang)?.into_iter().map(|c| (name.clone(), c)).collect())
}

fn load_plain(p: &Path, lang: &str) -> Result<Vec<Vec<String>>, String> {
    let e = ext(p);
    match e.as_str() {
        "txt" => Ok(vec![non_empty_lines(&read_utf8(p)?)]),
        _ if SUB_EXTS.contains(&e.as_str()) => {
            let lines = subtitle_lines(&read_utf8(p)?, &e);
            Ok(vec![split_sentences_in(&lines, lang)])
        }
        _ => Err(format!("{}: unsupported text format", p.display())),
    }
}

fn non_empty_lines(s: &str) -> Vec<String> {
    s.split('\n')
        .map(|l| py_strip(l).to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

/// Text lines of a subtitle file (SubPlz converts to SRT with ffmpeg, then
/// `remove_timing_and_metadata`).
pub fn subtitle_lines(src: &str, ext: &str) -> Vec<String> {
    subtitle_cues_raw(src, ext).into_iter().flatten().filter_map(|l| clean_sub_line(&l)).collect()
}

/// One string per cue (its lines joined): what retiming keeps together.
pub fn subtitle_cues(p: &Path) -> Result<Vec<String>, String> {
    let e = ext(p);
    let cues = subtitle_cues_raw(&read_utf8(p)?, &e);
    Ok(cues
        .into_iter()
        .filter_map(|c| {
            let mut out = String::new();
            for l in c.iter().filter_map(|l| clean_sub_line(l)) {
                // words in Latin script need a space where the lines meet
                if out.chars().last().is_some_and(|a| a.is_ascii_alphanumeric() || ",.!?;:".contains(a))
                    && l.chars().next().is_some_and(|b| b.is_ascii_alphanumeric())
                {
                    out.push(' ');
                }
                out.push_str(&l);
            }
            (!out.is_empty()).then_some(out)
        })
        .collect())
}

/// Text lines grouped by cue: SRT/VTT blocks, ASS Dialogue events.
fn subtitle_cues_raw(src: &str, ext: &str) -> Vec<Vec<String>> {
    let src = src.replace("\r\n", "\n");
    let mut cues: Vec<Vec<String>> = Vec::new();
    if ext == "ass" || ext == "ssa" {
        let mut in_events = false;
        let mut text_field = 9;
        for line in src.lines() {
            let t = line.trim();
            if t.starts_with('[') {
                in_events = t.eq_ignore_ascii_case("[events]");
                continue;
            }
            if !in_events {
                continue;
            }
            if let Some(f) = t.strip_prefix("Format:") {
                if let Some(i) = f.split(',').position(|x| x.trim().eq_ignore_ascii_case("text")) {
                    text_field = i;
                }
            } else if let Some(d) = t.strip_prefix("Dialogue:") {
                if let Some(text) = d.splitn(text_field + 1, ',').nth(text_field) {
                    cues.push(text.replace("\\N", "\n").replace("\\n", "\n").split('\n').map(str::to_string).collect());
                }
            }
        }
    } else {
        let mut cur: Vec<String> = Vec::new();
        for line in src.lines() {
            let t = line.trim();
            if t == "WEBVTT" || t.starts_with("NOTE") {
                continue;
            }
            if t.is_empty() {
                if !cur.is_empty() {
                    cues.push(std::mem::take(&mut cur));
                }
                continue;
            }
            cur.push(line.to_string());
        }
        if !cur.is_empty() {
            cues.push(cur);
        }
    }
    cues
}

/// A subtitle text line without tags, numbering, timing or ASS drawings.
fn clean_sub_line(l: &str) -> Option<String> {
    let tag = |s: &str, open: char, close: char| -> String {
        let mut out = String::new();
        let mut depth = 0;
        for c in s.chars() {
            if c == open {
                depth += 1;
            } else if c == close && depth > 0 {
                depth -= 1;
            } else if depth == 0 {
                out.push(c);
            }
        }
        out
    };
    let t = py_strip(l);
    if t.is_empty() || t.chars().all(|c| c.is_ascii_digit()) || t.contains("-->") {
        return None;
    }
    let mut c = tag(&tag(t, '<', '>'), '{', '}');
    // ASS drawing commands: "m 0 0 l ..."
    if let Some(i) = drawing_start(&c) {
        c.truncate(i);
    }
    Some(c)
}

/// position of `m\s\d+\s\d+\s.+?$` (the regex SubPlz strips)
fn drawing_start(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    for i in 0..b.len() {
        if b[i] != b'm' {
            continue;
        }
        let rest = &s[i + 1..];
        let mut it = rest.char_indices().peekable();
        let ok = (|| {
            let (_, c) = it.next()?;
            if !c.is_whitespace() {
                return None;
            }
            for _ in 0..2 {
                let mut digits = 0;
                while let Some(&(_, d)) = it.peek() {
                    if d.is_ascii_digit() {
                        digits += 1;
                        it.next();
                    } else {
                        break;
                    }
                }
                if digits == 0 {
                    return None;
                }
                let (_, w) = it.next()?;
                if !w.is_whitespace() {
                    return None;
                }
            }
            it.next().map(|_| ())
        })();
        if ok.is_some() {
            return Some(i);
        }
    }
    None
}

const TERMINAL: &[char] = &['。', '．', '.', '！', '!', '?', '？'];
const END_MARK: char = '\u{0238}'; // pysbd's 'ȸ' end-of-text marker

/// pysbd `Segmenter(language="ja", clean=False).segment(line)` for the
/// constructs that occur in Japanese prose and subtitles.
pub fn pysbd_ja(line: &str) -> Vec<String> {
    if line.is_empty() {
        return Vec::new();
    }
    let orig: Vec<char> = line.chars().collect();
    let mut sents: Vec<String> = Vec::new();
    for part in line.split(['\n', '\r']) {
        let chars: Vec<char> = part.chars().collect();
        if chars.is_empty() {
            continue;
        }
        if !chars.iter().any(|c| TERMINAL.contains(c)) {
            let s = py_strip(part);
            if !s.is_empty() {
                sents.push(s.to_string());
            }
            continue;
        }
        // protect punctuation inside the innermost 「…」 and （…）
        let mut protected = vec![false; chars.len()];
        for (open, close) in [('（', '）'), ('「', '」')] {
            let mut i = 0;
            while i < chars.len() {
                if chars[i] == open {
                    if let Some(j) = (i + 1..chars.len()).find(|&j| chars[j] == open || chars[j] == close) {
                        if chars[j] == close {
                            for p in protected.iter_mut().take(j).skip(i + 1) {
                                *p = true;
                            }
                            i = j;
                        }
                    }
                }
                i += 1;
            }
        }
        // an ASCII period inside a token (3.14, ver.001, www.example.jp) is
        // not a boundary
        for i in 0..chars.len().saturating_sub(1) {
            if chars[i] == '.' && chars[i + 1].is_ascii_alphanumeric() {
                protected[i] = true;
            }
        }
        // pysbd DoublePunctuationRules: ?! !? ?? !! act as one terminal
        // (only when the text does not start with one of them)
        let starts_double = chars.len() >= 2 && double_mark(chars[0], chars[1]).is_some();
        let mut c: Vec<char> = Vec::with_capacity(chars.len());
        let mut prot2: Vec<bool> = Vec::with_capacity(chars.len());
        let mut k = 0;
        while k < chars.len() {
            if !starts_double && k + 1 < chars.len() && !protected[k] {
                if let Some(m) = double_mark(chars[k], chars[k + 1]) {
                    c.push(m);
                    prot2.push(false);
                    k += 2;
                    continue;
                }
            }
            c.push(chars[k]);
            prot2.push(protected[k]);
            k += 1;
        }
        let mut protected = prot2;
        if !is_terminal(*c.last().unwrap()) {
            c.push(END_MARK);
            protected.push(false);
        }
        let term = |i: usize| (is_terminal(c[i]) || c[i] == END_MARK) && !protected[i];
        let run = |i: usize| (TERMINAL.contains(&c[i]) && !protected[i]) || c[i] == ' ';
        let mut i = 0;
        while i < c.len() {
            // [。．.！!?？ ]{2,}
            let mut j = i;
            while j < c.len() && run(j) {
                j += 1;
            }
            if j - i >= 2 {
                push_sent(&mut sents, &c[i..j]);
                i = j;
                continue;
            }
            // \S.*?[terminal]
            if !c[i].is_whitespace() {
                if let Some(k) = (i + 1..c.len()).find(|&k| term(k)) {
                    push_sent(&mut sents, &c[i..=k]);
                    i = k + 1;
                    continue;
                }
            }
            // [terminal]
            if term(i) {
                push_sent(&mut sents, &c[i..=i]);
            }
            i += 1;
        }
    }
    // map back onto the original line, keeping trailing whitespace
    let mut out = Vec::new();
    let mut prior_end = 0usize;
    for s in sents {
        let sc: Vec<char> = s.chars().collect();
        let mut found = None;
        let mut start = 0;
        while start + sc.len() <= orig.len() {
            if orig[start..start + sc.len()] == sc[..] {
                let mut end = start + sc.len();
                while end < orig.len() && orig[end].is_whitespace() {
                    end += 1;
                }
                if end > prior_end {
                    found = Some((start, end));
                    break;
                }
            }
            start += 1;
        }
        if let Some((a, b)) = found {
            out.push(orig[a..b].iter().collect());
            prior_end = b;
        }
    }
    out
}

const DOUBLE: [(char, char, char); 4] = [('?', '!', '\u{2609}'), ('!', '?', '\u{2608}'), ('?', '?', '\u{2607}'), ('!', '!', '\u{2604}')];

fn double_mark(a: char, b: char) -> Option<char> {
    DOUBLE.iter().find(|d| d.0 == a && d.1 == b).map(|d| d.2)
}

fn is_terminal(c: char) -> bool {
    TERMINAL.contains(&c) || DOUBLE.iter().any(|d| d.2 == c)
}

fn push_sent(sents: &mut Vec<String>, c: &[char]) {
    let mut s = String::new();
    for &x in c {
        if x == END_MARK {
            continue;
        }
        match DOUBLE.iter().find(|d| d.2 == x) {
            Some(d) => {
                s.push(d.0);
                s.push(d.1);
            }
            None => s.push(x),
        }
    }
    let s = py_strip(&s.replace('\n', "")).to_string();
    if !s.is_empty() {
        sents.push(s);
    }
}

/// SubPlz `split_sentences_from_input` (pysbd path) + reading the result back
/// as paragraphs (non-empty stripped lines).
pub fn split_sentences(lines: &[String]) -> Vec<String> {
    split_sentences_in(lines, "ja")
}

/// [`split_sentences`] in `lang` (ISO 639-1).
pub fn split_sentences_in(lines: &[String], lang: &str) -> Vec<String> {
    // written one per line, read back with strip()
    split_sentences_raw_in(lines, lang)
        .iter()
        .flat_map(|l| l.split('\n'))
        .map(|l| py_strip(l).to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

/// The book as SubPlz's `--respect-grouping` reads it: the sentence-split
/// text file (`read_script`: lines without the newline, empty ones dropped).
pub fn script_lines(p: &Path) -> Result<Vec<String>, String> {
    script_lines_in(p, "ja")
}

/// [`script_lines`] of a book in `lang` (ISO 639-1).
pub fn script_lines_in(p: &Path, lang: &str) -> Result<Vec<String>, String> {
    let e = ext(p);
    let input: Vec<String> = if e == "epub" {
        crate::epub::Epub::open(p)?.chapters.into_iter().flat_map(|c| c.paragraphs).collect()
    } else if e == "txt" {
        universal_lines(&read_utf8(p)?)
    } else if SUB_EXTS.contains(&e.as_str()) {
        subtitle_lines(&read_utf8(p)?, &e)
    } else {
        return Err(format!("{}: unsupported text format", p.display()));
    };
    let written: String = split_sentences_raw_in(&input, lang)
        .into_iter()
        .filter(|l| !l.is_empty())
        .map(|l| l + "\n")
        .collect();
    Ok(universal_lines(&written).into_iter().filter(|l| !l.is_empty()).collect())
}

/// Python text-mode lines: \r\n, \r and \n all end a line.
fn universal_lines(s: &str) -> Vec<String> {
    let s = s.replace("\r\n", "\n").replace('\r', "\n");
    let mut v: Vec<String> = s.split('\n').map(str::to_string).collect();
    if v.last().is_some_and(|l| l.is_empty()) {
        v.pop();
    }
    v
}

/// `split_sentences_from_input` before writing: pysbd per line,
/// `fix_end_of_quotes`, `merge_short_lines_with_quotes`.
pub fn split_sentences_raw(lines: &[String]) -> Vec<String> {
    split_sentences_raw_in(lines, "ja")
}

/// [`split_sentences_raw`] in `lang`: spaced languages get
/// [`sentences_spaced`] (their closing quotes already stay with the sentence).
pub fn split_sentences_raw_in(lines: &[String], lang: &str) -> Vec<String> {
    if !unspaced(lang) {
        return lines.iter().flat_map(|l| sentences_spaced(l.trim_end_matches('\n'), lang)).collect();
    }
    let mut segs: Vec<String> = Vec::new();
    for l in lines {
        segs.extend(pysbd_ja(l.trim_end_matches('\n')));
    }
    // fix_end_of_quotes
    let mut fixed: Vec<String> = Vec::new();
    for (i, line) in segs.into_iter().enumerate() {
        let mut line = line;
        if i > 0 {
            if let Some(c) = line.chars().next() {
                if c == '」' || c == '’' {
                    if let Some(last) = fixed.last_mut() {
                        last.push(c);
                    }
                    line = line[c.len_utf8()..].to_string();
                }
            }
        }
        if !line.is_empty() {
            fixed.push(line);
        }
    }
    // merge_short_lines_with_quotes
    let mut merged: Vec<String> = Vec::new();
    for line in fixed {
        if let Some(last) = merged.last_mut() {
            let first = last.chars().next();
            if matches!(first, Some('」') | Some('’')) && (last.chars().count() + line.chars().count()) <= 1 {
                last.push_str(&line);
                continue;
            }
        }
        merged.push(line);
    }
    merged
}

/// Languages written without spaces between words: pysbd's Japanese rules.
pub fn unspaced(lang: &str) -> bool {
    matches!(lang, "ja" | "zh" | "yue" | "th" | "lo" | "my" | "km")
}

/// What ends a sentence in a spaced language.
const SPACED_END: &[char] = &['.', '!', '?', '…'];
/// Closing quotes and brackets that stay with the sentence they end.
const CLOSERS: &[char] = &['"', '\'', '”', '’', '»', '«', ')', ']', '›'];

/// Words that end in a period without ending the sentence, by language
/// (lowercase, without the final period).
fn abbreviations(lang: &str) -> &'static [&'static str] {
    match lang {
        "en" => &[
            "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "no", "vol", "fig", "approx", "dept", "inc", "ltd", "co", "jan", "feb",
            "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "gen", "col", "lt", "sgt", "capt", "rev", "gov", "sen", "rep",
        ],
        "de" => &["hr", "hrn", "fr", "dr", "prof", "bzw", "usw", "ca", "evtl", "ggf", "nr", "str", "vgl", "bd", "abs", "inkl", "zzgl", "geb", "jh", "sog", "tel"],
        "fr" => &["m", "mm", "mme", "mmes", "mlle", "mlles", "dr", "st", "ste", "av", "apr", "env", "etc", "cf", "vol", "chap", "p"],
        "es" => &["sr", "sra", "srta", "sres", "dr", "dra", "ud", "uds", "d", "dña", "av", "pág", "etc", "aprox", "núm", "tel"],
        "it" => &["sig", "sigg", "sigra", "dott", "dr", "prof", "ing", "avv", "geom", "ecc", "pag", "cap", "tel"],
        "pt" => &["sr", "sra", "srta", "dr", "dra", "prof", "av", "pág", "etc", "aprox", "tel"],
        "nl" => &["dhr", "mevr", "mw", "dr", "prof", "ir", "drs", "mr", "ca", "enz", "nr", "blz", "jl"],
        "ru" | "uk" | "bg" => &["г", "гг", "ул", "стр", "им", "тов", "др", "проф", "рис", "см", "т", "с", "д", "пр", "акад", "доц", "млн", "млрд", "тыс", "руб"],
        "pl" => &["dr", "prof", "inż", "mgr", "ul", "np", "tzn", "itd", "itp", "tj", "wg", "ok", "godz", "nr", "str", "zob"],
        "cs" | "sk" => &["dr", "prof", "ing", "mgr", "např", "tzn", "atd", "apod", "tj", "ul", "č", "str", "resp"],
        _ => &["dr", "prof", "st", "nr", "no", "etc"],
    }
}

/// Sentences of one line of a spaced language: a sentence ends at . ! ? …
/// (with any closing quotes) before whitespace and a word that may start a
/// sentence, unless the period closes an abbreviation, an initial ("J.") or
/// a word with inner periods ("e.g.", "z.B.").
pub fn sentences_spaced(line: &str, lang: &str) -> Vec<String> {
    let c: Vec<char> = line.chars().collect();
    let n = c.len();
    let abbr = abbreviations(lang);
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < n {
        if !SPACED_END.contains(&c[i]) {
            i += 1;
            continue;
        }
        let mark = i;
        let mut j = i;
        while j < n && SPACED_END.contains(&c[j]) {
            j += 1;
        }
        while j < n && CLOSERS.contains(&c[j]) {
            j += 1;
        }
        // French spacing: « Bonjour. »
        let mut q = j;
        while q < n && c[q].is_whitespace() {
            q += 1;
        }
        if q > j && q < n && matches!(c[q], '»' | '›') {
            j = q + 1;
        }
        if j < n && !c[j].is_whitespace() {
            // 3.14, www.example.com, "?!" inside a word
            i = j;
            continue;
        }
        let mut k = j;
        while k < n && c[k].is_whitespace() {
            k += 1;
        }
        let ends = if k == n {
            true
        } else if c[k].is_lowercase() {
            // "etc. and so on", "what? he said"
            false
        } else if c[mark] == '.' && j - mark == 1 + c[mark + 1..j].iter().filter(|x| CLOSERS.contains(x)).count() {
            // a single period: what word does it close?
            let w0 = c[..mark].iter().rposition(|x| x.is_whitespace() || matches!(x, '(' | '"' | '«' | '“' | '„' | '[')).map_or(0, |p| p + 1);
            let word: String = c[w0..mark].iter().collect::<String>().to_lowercase();
            let initial = mark - w0 == 1 && c[w0].is_uppercase();
            let inner = word.contains('.');
            !(initial || inner || abbr.contains(&word.as_str()))
        } else {
            true
        };
        if ends {
            let sent: String = c[start..j].iter().collect();
            let sent = sent.trim();
            if !sent.is_empty() {
                out.push(sent.to_string());
            }
            start = k;
        }
        i = k.max(j);
    }
    let rest: String = c[start.min(n)..].iter().collect();
    let rest = rest.trim();
    if !rest.is_empty() {
        out.push(rest.to_string());
    }
    out
}

/// Whether a sentence of a spaced language ends between `before` and the
/// word `next` (with its leading space), by [`sentences_spaced`].
pub fn sentence_break(before: &str, next: &str, lang: &str) -> bool {
    let joined = format!("{before}{next}");
    let s = sentences_spaced(&joined, lang);
    s.len() >= 2 && s.last().map(String::as_str) == Some(next.trim())
}

#[cfg(test)]
mod spaced_tests {
    use super::sentences_spaced;

    #[test]
    fn english() {
        assert_eq!(
            sentences_spaced("Mr. Smith met Dr. Jones at 3.15 p.m. on Main St. today. \"Really?\" she asked. J. R. R. Tolkien wrote it... and then left!", "en"),
            vec!["Mr. Smith met Dr. Jones at 3.15 p.m. on Main St. today.", "\"Really?\" she asked.", "J. R. R. Tolkien wrote it... and then left!"]
        );
        assert_eq!(sentences_spaced("Wait... What? No!", "en"), vec!["Wait...", "What?", "No!"]);
        assert_eq!(sentences_spaced("He said: \"Go.\" Then he left.", "en"), vec!["He said: \"Go.\"", "Then he left."]);
    }

    #[test]
    fn other_languages() {
        assert_eq!(sentences_spaced("Das ist z.B. gut. Herr Dr. Müller kommt.", "de"), vec!["Das ist z.B. gut.", "Herr Dr. Müller kommt."]);
        assert_eq!(sentences_spaced("Он живёт на ул. Ленина. Это далеко.", "ru"), vec!["Он живёт на ул. Ленина.", "Это далеко."]);
        assert_eq!(sentences_spaced("¿Vienes? ¡Claro! Sr. García también.", "es"), vec!["¿Vienes?", "¡Claro!", "Sr. García también."]);
        assert_eq!(sentences_spaced("« Bonjour. » M. Dupont sourit.", "fr"), vec!["« Bonjour. »", "M. Dupont sourit."]);
    }
}
