//! Audio -> speech chunks (FireRedVAD) -> text (Parakeet or Kitsune) ->
//! per-chapter transcripts, following SubPlz's ParakeetModel
//! `transcribe_audio_subs_fast` (single-pass branch). Models are loaded once
//! in an [`Engine`] and reused for every audio file.

use crate::asr::{Chunk, Parakeet};
use crate::batch::{py_strip, Cue, Transcript};
use crate::kitsune::Kitsune;
use crate::vad::{self, round3, Device};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AsrModel {
    Kitsune(String),
    /// a parakeet-tdt-0.6b-v3 derivative in `models/parakeet-<name>` (128 mel bins)
    ParakeetV3(String),
    /// an optimum ONNX export of a Whisper model in `models/whisper-<name>`
    Whisper(String),
}

impl std::fmt::Display for AsrModel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AsrModel::Kitsune(size) => write!(f, "kitsune-{size}"),
            AsrModel::ParakeetV3(name) => write!(f, "parakeet-{name}"),
            AsrModel::Whisper(name) => write!(f, "whisper-{name}"),
        }
    }
}

impl std::str::FromStr for AsrModel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "kitsune-p0.05b" | "kitsune-p0.1b" | "kitsune-p0.3b" => Ok(AsrModel::Kitsune(s["kitsune-".len()..].to_string())),
            "parakeet-ultra" | "parakeet-redux" => Ok(AsrModel::ParakeetV3(s["parakeet-".len()..].to_string())),
            "whisper-ja-anime-v0.3" | "whisper-large-v3-turbo" | "whisper-thewhisper-large-v3-turbo" => Ok(AsrModel::Whisper(s["whisper-".len()..].to_string())),
            _ => Err(format!("unknown ASR model {s:?} ({})", CATALOG.iter().map(|m| m.id).collect::<Vec<_>>().join(", "))),
        }
    }
}

/// A speech recognition model the app can offer.
pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub family: &'static str,
    /// ISO 639-1 codes of the subtitle languages it transcribes
    pub languages: &'static [&'static str],
}

/// Every model subthx knows, installed or not. New families (Whisper,
/// Moonshine, …) add entries here and an [`Asr`] backend.
pub const CATALOG: &[ModelInfo] = &[
    ModelInfo { id: "kitsune-p0.05b", name: "Kitsune 0.05B", family: "kitsune", languages: &["ja"] },
    ModelInfo { id: "kitsune-p0.1b", name: "Kitsune 0.1B", family: "kitsune", languages: &["ja"] },
    ModelInfo { id: "kitsune-p0.3b", name: "Kitsune 0.3B", family: "kitsune", languages: &["ja"] },
    ModelInfo { id: "parakeet-ultra", name: "Parakeet Ultra", family: "parakeet", languages: V3_LANGUAGES },
    ModelInfo { id: "parakeet-redux", name: "Parakeet Redux", family: "parakeet", languages: V3_LANGUAGES },
    ModelInfo { id: "whisper-ja-anime-v0.3", name: "Whisper JA Anime v0.3", family: "whisper", languages: &["ja"] },
    ModelInfo { id: "whisper-large-v3-turbo", name: "Whisper Large v3 Turbo", family: "whisper", languages: WHISPER_LANGUAGES },
    // the languages of its model card: the fine-tune lost Japanese (it writes romaji)
    ModelInfo { id: "whisper-thewhisper-large-v3-turbo", name: "TheWhisper Large v3 Turbo", family: "whisper", languages: THEWHISPER_LANGUAGES },
];

/// 30 s windows per Whisper batch: 8 holds ~5 GB of GPU memory with the
/// fused encoder at nearly the speed of 16 (SUBTHX_WHISPER_BATCH overrides).
fn whisper_batch(asked: usize) -> usize {
    std::env::var("SUBTHX_WHISPER_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(asked.min(8))
}

/// The files a Whisper export loads.
pub const WHISPER_FILES: [&str; 3] = [
    // tools/whisper_onnx.py, tools/dump_whisper.py
    "onnx/encoder_kv_fp16.onnx",
    "onnx/decoder_fp16.onnx",
    "mel_filters.npy",
];

/// Whether the model breaks its transcript into lines itself (Whisper, by
/// timestamps): its cues need no regrouping after the fact.
pub fn breaks_lines(asr: &AsrModel) -> bool {
    model_info(asr).family == "whisper"
}

/// The languages of multilingual Whisper (openai `LANGUAGES`, without the
/// odd codes "jw" and "haw").
const WHISPER_LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs",
    "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
    "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
    "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "ln", "ha", "ba", "su", "yue",
];

/// TheStageAI/thewhisper-large-v3-turbo's languages.
const THEWHISPER_LANGUAGES: &[&str] = &[
    "en", "ar", "bg", "bn", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
    "vi",
];

/// The 25 European languages of parakeet-tdt-0.6b-v3.
const V3_LANGUAGES: &[&str] = &["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk"];

/// (encoder file, decoder file, encoder on the CPU?) of a v3 model on `device`.
/// Ultra: fp16 encoder on GPUs, int8 on the CPU. Redux ships a 2-bit
/// encoder the CUDA provider cannot run; on GPUs it uses the same weights
/// repacked as 4-bit (tools/repack_redux.py). An fp32 or fp16 decoder (not
/// the int8 one) keeps a batch's transcripts independent of each other.
pub fn parakeet_v3_files(dir: &Path, device: Device) -> (&'static str, &'static str, bool) {
    let gpu = device != Device::Cpu;
    // on GPUs: the slimmed fp16 decoder (tools/slim_parakeet_v3.py), if made
    let dec = if gpu && dir.join("decoder_joint-model.fp16.onnx").exists() {
        "decoder_joint-model.fp16.onnx"
    } else {
        "decoder_joint-model.onnx"
    };
    for (file, on_gpu) in [
        ("encoder-model.w4.fp16.onnx", true),
        ("encoder-model.w4.onnx", true),
        ("encoder-model.fp16.onnx", true),
        ("encoder-model.w2a8.onnx", false),
        ("encoder-model.int8.onnx", false),
    ] {
        if on_gpu == gpu && dir.join(file).exists() {
            return (file, dec, !on_gpu);
        }
    }
    // a GPU without a GPU encoder: the CPU one
    let enc = if dir.join("encoder-model.w2a8.onnx").exists() { "encoder-model.w2a8.onnx" } else { "encoder-model.int8.onnx" };
    (enc, dec, true)
}

pub fn model_info(asr: &AsrModel) -> &'static ModelInfo {
    let id = asr.to_string();
    CATALOG.iter().find(|m| m.id == id).expect("every AsrModel is in the catalog")
}

/// The Kitsune file used on `device`: fp16 with fp32 attention scores on
/// GPUs, int8 matmuls (fp32 convs) on the CPU.
pub fn kitsune_file(dir: &Path, device: Device) -> &'static str {
    if device != Device::Cpu {
        "model.mixed16b.onnx"
    } else if dir.join("model.int8.onnx").exists() {
        "model.int8.onnx"
    } else {
        "model.fp32.onnx"
    }
}

/// Bytes on disk of what `asr` loads on `device` (None: not installed).
pub fn model_size(models: &Path, asr: &AsrModel, device: Device) -> Option<u64> {
    match asr {
        AsrModel::Kitsune(size) => {
            let dir = models.join(format!("kitsune-{size}"));
            std::fs::metadata(dir.join(kitsune_file(&dir, device))).ok().map(|m| m.len())
        }
        AsrModel::ParakeetV3(name) => {
            let dir = models.join(format!("parakeet-{name}"));
            let (enc, dec, _) = parakeet_v3_files(&dir, device);
            let size = |f: &str| std::fs::metadata(dir.join(f)).ok().map(|m| m.len());
            Some(size(enc)? + size(dec)? + size(&format!("{dec}.data")).unwrap_or(0))
        }
        AsrModel::Whisper(name) => {
            let dir = models.join(format!("whisper-{name}"));
            WHISPER_FILES.iter().map(|f| std::fs::metadata(dir.join(f)).ok().map(|m| m.len())).sum()
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub asr: AsrModel,
    pub device: Device,
    /// holds firered_vad*.onnx, firered_vad_cmvn.json, nemo80_fbanks.npy, kitsune-*/
    pub models: PathBuf,
    /// utterances per ASR batch; None = 64 on GPUs, 16 on the CPU (less padding)
    pub batch_size: Option<usize>,
}

/// Chapter span in seconds (as ffprobe reports it).
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub start: f64,
    pub end: f64,
}

/// Processing step reported by [`Engine::transcribe_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Decode,
    Vad,
    Asr,
}

#[derive(Default, Debug, Clone)]
pub struct Timings {
    pub decode: f64,
    pub vad: f64,
    pub asr: f64,
    pub chunks: usize,
    pub audio_seconds: f64,
}

enum Asr {
    Parakeet(Parakeet),
    Kitsune(Kitsune),
    Whisper(crate::whisper::Whisper),
}

pub struct Engine {
    cmvn: vad::Cmvn,
    fbank: vad::Fbank,
    vad: vad::Model,
    asr: Asr,
    /// what the model speaks: picks the audio track of a multi-track file
    languages: &'static [&'static str],
    /// the language of the next transcription, when set (see `set_language`)
    language: Option<String>,
    batch_size: usize,
    /// seconds spent loading the models
    pub load_seconds: f64,
}

/// The error of a model that is not downloaded (the clients word it).
pub const MODEL_MISSING: &str = "model_missing";

impl Engine {
    pub fn load(cfg: &Config) -> Result<Engine, String> {
        // not downloaded yet: say so, not the OS's "file not found"
        if model_size(&cfg.models, &cfg.asr, cfg.device).is_none() || !cfg.models.join("firered_vad.onnx").exists() {
            return Err(MODEL_MISSING.into());
        }
        let t = Instant::now();
        // fp16 halves the VAD model time on GPUs (~0.5% of segment boundaries
        // move by a few frames); the CPU keeps fp32
        let fp16 = cfg.models.join("firered_vad.fp16.onnx");
        let vad_path = if cfg.device != Device::Cpu && fp16.exists() {
            fp16
        } else {
            cfg.models.join("firered_vad.onnx")
        };
        let mel = cfg.models.join("nemo80_fbanks.npy");
        let asr = match &cfg.asr {
            AsrModel::Kitsune(size) => {
                let dir = cfg.models.join(format!("kitsune-{size}"));
                Asr::Kitsune(Kitsune::load(&dir, kitsune_file(&dir, cfg.device), &mel, cfg.device)?)
            }
            AsrModel::ParakeetV3(name) => {
                let dir = cfg.models.join(format!("parakeet-{name}"));
                let (enc, dec, enc_cpu) = parakeet_v3_files(&dir, cfg.device);
                let enc_device = if enc_cpu { Device::Cpu } else { cfg.device };
                Asr::Parakeet(Parakeet::load_with(
                    &dir,
                    enc,
                    dec,
                    &cfg.models.join("nemo128_fbanks.npy"),
                    enc_device,
                    cfg.device,
                )?)
            }
            AsrModel::Whisper(name) => {
                let lang = model_info(&cfg.asr).languages[0];
                let mut m = crate::whisper::Whisper::load(&cfg.models.join(format!("whisper-{name}")), lang, cfg.device)?;
                // TheStage's fine-tune lost timestamps: with them it drops
                // text (English WER 10.7% against 8.5% without, on an hour
                // of This American Life; turbo: 8.0% / 7.8%)
                m.set_timestamps(!name.starts_with("thewhisper"));
                Asr::Whisper(m)
            }
        };
        Ok(Engine {
            cmvn: vad::Cmvn::load(&cfg.models.join("firered_vad_cmvn.json"))?,
            fbank: vad::Fbank::new(),
            vad: vad::Model::load(&vad_path, cfg.device)?,
            asr,
            languages: model_info(&cfg.asr).languages,
            language: None,
            batch_size: cfg.batch_size.unwrap_or(if cfg.device == Device::Cpu { 16 } else { 64 }),
            load_seconds: t.elapsed().as_secs_f64(),
        })
    }

    /// Transcribe in `language` (ISO 639-1) from now on: Whisper's language
    /// token, and the audio track of files that have several.
    pub fn set_language(&mut self, language: &str) -> Result<(), String> {
        if let Asr::Whisper(m) = &mut self.asr {
            m.set_language(language)?;
        }
        self.language = Some(language.to_string());
        Ok(())
    }

    /// Transcribe one audio file and split the cues into `chapters` (the
    /// whole file when empty). Cue times are relative to their chapter.
    pub fn transcribe(&mut self, audio: &Path, chapters: &[Span]) -> Result<(Vec<Transcript>, Timings), String> {
        self.transcribe_with(audio, chapters, &mut |_, _| Ok(()))
    }

    /// [`Engine::transcribe`] reporting `progress(stage, fraction)`; an error
    /// from it (e.g. a cancelled job) stops the run between steps.
    pub fn transcribe_with(
        &mut self,
        audio: &Path,
        chapters: &[Span],
        progress: &mut dyn FnMut(Stage, f64) -> Result<(), String>,
    ) -> Result<(Vec<Transcript>, Timings), String> {
        let mut tm = Timings::default();

        progress(Stage::Decode, 0.0)?;
        let t = Instant::now();
        let wanted: Vec<&str> = match &self.language {
            Some(l) => vec![l.as_str()],
            None => self.languages.to_vec(),
        };
        let wav = crate::audio::decode_16k_for(audio, &wanted)?;
        tm.decode = t.elapsed().as_secs_f64();
        tm.audio_seconds = wav.len() as f64 / 16000.0;

        progress(Stage::Vad, 0.0)?;
        let t = Instant::now();
        let feats = self.fbank.compute(&wav, &self.cmvn);
        let probs = self.vad.probs(&feats)?;
        drop(feats);
        let decisions = vad::PostConfig::default().process(&probs);
        let chunks: Vec<Chunk> = vad::segments_samples(&decisions, wav.len())
            .into_iter()
            .map(|(start, end)| Chunk {
                start,
                end: end.min(wav.len()),
            })
            .filter(|c| c.end > c.start)
            .collect();
        tm.vad = t.elapsed().as_secs_f64();
        tm.chunks = chunks.len();

        progress(Stage::Asr, 0.0)?;
        let t = Instant::now();
        let n = chunks.len().max(1) as f64;
        let mut on_batch = |done: usize| progress(Stage::Asr, done as f64 / n);
        let with_chunks = |texts: Vec<String>| -> Vec<(usize, usize, String)> { chunks.iter().zip(texts).map(|(c, t)| (c.start, c.end, t)).collect() };
        let utterances = match &mut self.asr {
            Asr::Parakeet(m) => with_chunks(m.transcribe_with(&wav, &chunks, self.batch_size, &mut on_batch)?),
            Asr::Kitsune(m) => with_chunks(m.transcribe_with(&wav, &chunks, self.batch_size, &mut on_batch)?),
            // an encoder-decoder: windows of 30 s, far heavier per item
            Asr::Whisper(m) => m
                .transcribe_with(&wav, &chunks, whisper_batch(self.batch_size), &mut on_batch)?
                .into_iter()
                .map(|u| (u.start, u.end, u.text))
                .collect(),
        };
        tm.asr = t.elapsed().as_secs_f64();
        if std::env::var_os("SUBTHX_PROFILE").is_some() {
            if let Asr::Parakeet(m) = &self.asr {
                let p = m.profile;
                eprintln!("  parakeet: mel {:.2}s, encoder {:.2}s, decode loop {:.2}s (decoder runs {:.2}s)", p[0], p[1], p[2], p[3]);
            }
            if let Asr::Whisper(m) = &self.asr {
                let p = m.profile;
                eprintln!("  whisper: features {:.2}s, encoder {:.2}s, decoder {:.2}s, alignment {:.2}s", p[0], p[1], p[2], p[3]);
            }
        }

        // all_segments with absolute times, empty texts dropped
        let mut all: Vec<Cue> = Vec::new();
        for (start, end, text) in utterances {
            let t = py_strip(&text);
            if !t.is_empty() {
                all.push(Cue {
                    start: round3(start as f64 / 16000.0),
                    end: round3(end as f64 / 16000.0),
                    text: t.to_string(),
                });
            }
        }
        let spans: Vec<Span> = if chapters.is_empty() {
            vec![Span {
                start: 0.0,
                end: tm.audio_seconds,
            }]
        } else {
            chapters.to_vec()
        };
        let out = spans
            .iter()
            .map(|sp| Transcript {
                language: "ja".into(),
                segments: all
                    .iter()
                    .filter(|s| sp.start <= s.start && s.start < sp.end)
                    .map(|s| Cue {
                        start: round3(s.start - sp.start),
                        end: round3(s.end - sp.start),
                        text: s.text.clone(),
                    })
                    .collect(),
            })
            .collect();
        Ok((out, tm))
    }
}
