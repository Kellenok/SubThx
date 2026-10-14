//! Port of SubPlz `sync.match_start` / `ats.main.expand_matches` and the batch
//! loop of `sync.sync`: find which book chapter each audio chapter starts, group
//! chapters into batches, align each batch.

use crate::align::Punct;
use crate::batch::{align_chapter_task, AudioPart, Segment, Transcript};
use crate::lang::Lang;
use crate::lev::indel_distance;
use rayon::prelude::*;
use std::collections::HashMap;

pub const SCORE_THRESHOLD: f64 = 40.0;

#[derive(Clone)]
pub struct AudioChapter {
    pub transcript: Option<Transcript>,
    pub duration: f64,
}

/// One audio file split into chapters.
#[derive(Clone)]
pub struct AudioFile {
    pub chapters: Vec<AudioChapter>,
}

/// One text source (an epub): chapters of paragraphs.
pub struct TextFile {
    pub chapters: Vec<Vec<String>>,
}

/// (text_file, text_chapter, score)
pub type Ats = HashMap<(usize, usize), (usize, usize, f64)>;
/// (audio_file, audio_chapter, score)
pub type Sta = HashMap<(usize, usize), (usize, usize, f64)>;

#[derive(Clone, Debug, PartialEq)]
pub struct Batch {
    pub audio: Vec<usize>,
    /// -1 when the audio chapters matched no text
    pub text_file: i64,
    pub text_parts: Vec<usize>,
    pub score: Option<f64>,
}

/// rapidfuzz `fuzz.ratio`
pub fn fuzz_ratio(a: &[char], b: &[char]) -> f64 {
    let lensum = a.len() + b.len();
    if lensum == 0 {
        return 100.0;
    }
    let dist = indel_distance(a, b);
    let norm_dist = dist as f64 / lensum as f64;
    (1.0 - norm_dist) * 100.0
}

fn norm_clean(lang: Lang, s: &str) -> Vec<char> {
    lang.normalize(&lang.clean(s)).chars().collect()
}

pub fn match_start(audio: &[AudioFile], text: &[TextFile]) -> (Ats, Sta) {
    let mut ats: Ats = HashMap::new();
    let mut sta: Sta = HashMap::new();
    // text content per (ti, j); SubPlz caches with the language of the first
    // audio chapter that needed it, all chapters are the same language in practice
    let mut textcache: HashMap<(usize, usize), Vec<char>> = HashMap::new();

    for (ai, af) in audio.iter().enumerate() {
        for (i, ch) in af.chapters.iter().enumerate() {
            let Some(l) = &ch.transcript else { continue };
            let lang = Lang::from_code(&l.language);
            let joined: String = l.segments.iter().map(|s| s.text.as_str()).collect();
            let acontent = norm_clean(lang, &joined);

            // the candidate scores are independent; compute in parallel
            let mut cands: Vec<(usize, usize)> = Vec::new();
            for (ti, tf) in text.iter().enumerate() {
                for j in 0..tf.chapters.len() {
                    if sta.contains_key(&(ti, j)) {
                        continue;
                    }
                    textcache.entry((ti, j)).or_insert_with(|| {
                        let joined: String = tf.chapters[j].concat();
                        norm_clean(lang, &joined)
                    });
                    cands.push((ti, j));
                }
            }
            let scores: Vec<Option<f64>> = cands
                .par_iter()
                .map(|k| {
                    let tcontent = &textcache[k];
                    if acontent.len() < 100 || tcontent.len() < 100 {
                        return None;
                    }
                    let n = tcontent.len().min(acontent.len()).min(2000);
                    Some(fuzz_ratio(&acontent[..n], &tcontent[..n]))
                })
                .collect();

            let mut best: (i64, i64, f64) = (-1, -1, 0.0);
            for (&(ti, j), score) in cands.iter().zip(scores) {
                let Some(score) = score else { continue };
                if score > SCORE_THRESHOLD && score > best.2 {
                    best = (ti as i64, j as i64, score);
                }
            }
            if best.0 >= 0 {
                let key = (best.0 as usize, best.1 as usize);
                if !sta.contains_key(&key) {
                    ats.insert((ai, i), (key.0, key.1, best.2));
                    sta.insert(key, (ai, i, best.2));
                }
            }
        }
    }
    (ats, sta)
}

pub fn expand_matches(audio: &[AudioFile], text: &[TextFile], ats: &Ats, sta: &Sta) -> Vec<Vec<Batch>> {
    let mut out = Vec::new();
    for (ai, af) in audio.iter().enumerate() {
        let mut batch: Vec<Batch> = Vec::new();
        let add = |batch: &mut Vec<Batch>, idx: usize, other: &[usize]| {
            let (chi, chj, score) = ats[&(ai, idx)];
            let mut z = vec![chj];
            z.extend((chj + 1..text[chi].chapters.len()).take_while(|&j| !sta.contains_key(&(chi, j))));
            let mut a = vec![idx];
            a.extend_from_slice(other);
            batch.push(Batch {
                audio: a,
                text_file: chi as i64,
                text_parts: z,
                score: Some(score),
            });
        };
        let n = af.chapters.len();
        let mut prev: Option<usize> = None;
        let mut i = 0;
        while i < n {
            let t = ats.contains_key(&(ai, i));
            let mut k = Vec::new();
            while i < n && ats.contains_key(&(ai, i)) == t {
                k.push(i);
                i += 1;
            }
            if t {
                for &x in &k[..k.len() - 1] {
                    add(&mut batch, x, &[]);
                }
                prev = Some(*k.last().unwrap());
            } else if let Some(p) = prev {
                add(&mut batch, p, &k);
            } else {
                batch.push(Batch {
                    audio: k,
                    text_file: -1,
                    text_parts: Vec::new(),
                    score: None,
                });
            }
        }
        if n > 0 && prev == Some(n - 1) {
            add(&mut batch, n - 1, &[]);
        }
        out.push(batch);
    }
    out
}

/// Align every batch of one audio file; returns the unshifted segments
/// (the input of `shift_align`) per batch.
pub fn align_batches(af: &AudioFile, text: &[TextFile], batches: &[Batch], p: &Punct) -> Vec<Vec<Segment>> {
    let empty = Transcript {
        language: "ja".into(),
        segments: Vec::new(),
    };
    // python: offset = 0; offset += sum(durations)  (sum starts at int 0)
    let mut offsets = Vec::with_capacity(batches.len());
    let mut offset = 0.0f64;
    for b in batches {
        offsets.push(offset);
        let mut s = 0.0f64;
        for &aj in &b.audio {
            s += af.chapters[aj].duration;
        }
        offset += s;
    }
    batches
        .par_iter()
        .zip(offsets)
        .map(|(b, off)| {
            let ach: Vec<AudioPart> = b
                .audio
                .iter()
                .map(|&aj| AudioPart {
                    transcript: af.chapters[aj].transcript.as_ref().unwrap_or(&empty),
                    duration: af.chapters[aj].duration,
                })
                .collect();
            let paras: Vec<&str> = if b.text_file >= 0 && (b.text_file as usize) < text.len() {
                let tf = &text[b.text_file as usize];
                b.text_parts
                    .iter()
                    .flat_map(|&j| tf.chapters[j].iter().map(|s| s.as_str()))
                    .collect()
            } else {
                Vec::new()
            };
            align_chapter_task(&ach, &paras, off, p)
        })
        .collect()
}
