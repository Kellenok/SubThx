//! Golden fixtures produced by `tools/dump_fixture.py` from SubPlz.

use crate::align::Punct;
use crate::batch::{Cue, Transcript};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct Fixture {
    pub punctuation: PunctStrings,
    pub audio_chapters: Vec<AudioChapter>,
    pub epub_chapters: Vec<EpubChapter>,
    pub transcripts: HashMap<String, Transcript>,
    #[serde(rename = "match")]
    pub matching: Match,
    pub stages: Vec<Stage>,
    pub final_segments: Vec<Cue>,
    pub srt: String,
    pub timings_s: HashMap<String, f64>,
}

#[derive(Deserialize)]
pub struct PunctStrings {
    pub prepend: String,
    pub append: String,
    pub nopend: String,
}

impl PunctStrings {
    pub fn sets(&self) -> Punct {
        Punct {
            prepend: self.prepend.chars().collect(),
            append: self.append.chars().collect(),
            nopend: self.nopend.chars().collect(),
        }
    }
}

#[derive(Deserialize)]
pub struct AudioChapter {
    pub cid: i64,
    pub title: String,
    pub start: f64,
    pub end: f64,
    pub duration: f64,
}

#[derive(Deserialize)]
pub struct EpubChapter {
    pub idx: i64,
    pub title: String,
    pub paragraphs: Vec<String>,
}

#[derive(Deserialize)]
pub struct Match {
    /// [audio_file, audio_chapter, text_file, text_chapter, score]
    pub ats: Vec<(usize, usize, usize, usize, f64)>,
    /// per audio file: [audio_chapters, [text_file, text_chapters], score]
    pub batches: Vec<Vec<(Vec<usize>, (i64, Vec<usize>), Option<f64>)>>,
}

#[derive(Deserialize)]
pub struct Stage {
    pub audio: Vec<usize>,
    pub text_chapter: i64,
    pub text_parts: Vec<usize>,
    pub offset: f64,
    pub segments: Vec<Cue>,
}

impl Fixture {
    pub fn load(path: &std::path::Path) -> Fixture {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    pub fn transcript(&self, audio_idx: usize) -> &Transcript {
        &self.transcripts[&self.audio_chapters[audio_idx].cid.to_string()]
    }
}

pub fn fixture_files() -> Vec<std::path::PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir("fixtures")
        .expect("run tools/dump_fixture.py first")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    v.sort();
    v
}

impl Fixture {
    /// Inputs in the shape the sync pipeline takes (one audio file, one epub).
    pub fn inputs(&self) -> (Vec<crate::sync::AudioFile>, Vec<crate::sync::TextFile>) {
        let audio = crate::sync::AudioFile {
            chapters: self
                .audio_chapters
                .iter()
                .map(|c| crate::sync::AudioChapter {
                    transcript: self.transcripts.get(&c.cid.to_string()).cloned(),
                    duration: c.duration,
                })
                .collect(),
        };
        let text = crate::sync::TextFile {
            chapters: self.epub_chapters.iter().map(|c| c.paragraphs.clone()).collect(),
        };
        (vec![audio], vec![text])
    }
}
