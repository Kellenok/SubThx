//! Whisper (encoder-decoder, e.g. whisper-ja-anime-v0.3) through ONNX
//! Runtime, from an optimum export reworked by tools/whisper_onnx.py into two
//! fp16 graphs: the encoder (fused attention) handing out the decoder's
//! cross-attention caches, and one decoder for the prompt, every step and
//! the alignment pass.
//!
//! What faster-whisper's batched pipeline with word timestamps does (as
//! SubPlz runs it): consecutive VAD chunks are packed into 30 s windows (the
//! audio outside speech zeroed, so music does not turn into words), encoded
//! in batches and decoded greedily with timestamps (the model learned them
//! from subtitles: each line, usually one speaker, ends with one; they mark
//! cue edges, the times come from alignment). One more decoder pass
//! over each transcript reads the alignment heads' cross-attention and DTW
//! times every word (openai `find_alignment`, CTranslate2 `align`); each word
//! then goes to the VAD chunk it was said in, so cues keep the VAD's edges.
//! The key/value caches stay on the GPU between steps (IoBinding); only
//! token ids go in and logits come out.

use crate::asr::Chunk;
use crate::vad::Device;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::{DynValue, Tensor};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

const RATE: usize = 16000;
const WINDOW: usize = 30 * RATE;
const N_FFT: usize = 400;
const HOP: usize = 160;
const FRAMES: usize = 3000;
/// longest decode per window (tokens after the prompt)
const MAX_TOKENS: usize = 440;
/// decoder steps between arena shrinks
const SHRINK_EVERY: usize = 8;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// transformers' WhisperFeatureExtractor (numpy path): periodic Hann STFT
/// with reflect padding, power spectrum, mel filterbank, log10, clamp to
/// max - 8, (x + 4) / 4; 30 s padded with zeros.
pub struct WhisperMel {
    /// (n_fft/2+1) x n_mels
    filters: Vec<f32>,
    n_mels: usize,
    window: Vec<f32>,
    fft: Arc<dyn realfft::RealToComplex<f32>>,
}

impl WhisperMel {
    pub fn load(path: &Path) -> Result<WhisperMel, String> {
        let (shape, filters) = crate::npy::read_f32(path)?;
        let bins = N_FFT / 2 + 1;
        if shape.len() != 2 || shape[0] != bins {
            return Err(format!("{}: expected {bins} x n_mels mel filters, got {shape:?}", path.display()));
        }
        let window = (0..N_FFT)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32)
            .collect();
        Ok(WhisperMel {
            n_mels: shape[1],
            filters,
            window,
            fft: realfft::RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT),
        })
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// n_mels x 3000 features of up to 30 s of audio.
    pub fn features(&self, audio: &[f32]) -> Vec<f32> {
        let bins = N_FFT / 2 + 1;
        let pad = N_FFT / 2;
        // the 30 s window with reflect padding on both sides
        let mut x = vec![0f32; WINDOW + 2 * pad];
        let n = audio.len().min(WINDOW);
        x[pad..pad + n].copy_from_slice(&audio[..n]);
        for i in 0..pad {
            x[pad - 1 - i] = x[pad + 1 + i];
            x[pad + WINDOW + i] = x[pad + WINDOW - 2 - i];
        }
        let mut mel = vec![0f32; self.n_mels * FRAMES];
        let mut buf = vec![0f32; N_FFT];
        let mut spec = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut power = vec![0f32; bins];
        for f in 0..FRAMES {
            let s = &x[f * HOP..f * HOP + N_FFT];
            for ((b, v), w) in buf.iter_mut().zip(s).zip(&self.window) {
                *b = v * w;
            }
            self.fft.process_with_scratch(&mut buf, &mut spec, &mut scratch).expect("fft sizes");
            for (p, c) in power.iter_mut().zip(&spec) {
                *p = c.re * c.re + c.im * c.im;
            }
            for m in 0..self.n_mels {
                let mut acc = 0f32;
                for (k, p) in power.iter().enumerate() {
                    acc += self.filters[k * self.n_mels + m] * p;
                }
                mel[m * FRAMES + f] = acc.max(1e-10).log10();
            }
        }
        let top = mel.iter().cloned().fold(f32::MIN, f32::max);
        for v in &mut mel {
            *v = (v.max(top - 8.0) + 4.0) / 4.0;
        }
        mel
    }
}

/// GPT-2 byte-level BPE: token strings use printable stand-ins for bytes.
fn byte_decoder() -> std::collections::HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter().zip(cs).map(|(b, c)| (char::from_u32(c).unwrap(), b as u8)).collect()
}

/// A word of a window's transcript, times in seconds from the window start.
struct Word {
    text: String,
    /// DTW span (may hold a pause next to the word)
    start: f64,
    end: f64,
    /// the middle of its speech (its DTW span over speech frames only)
    peak: f64,
    /// the model began a new segment (subtitle line) with it
    line: bool,
}

/// A recognized utterance, in samples of the whole file.
#[derive(Clone, Debug)]
pub struct Utterance {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

pub struct Whisper {
    mel: WhisperMel,
    /// input_features -> cross.{l}.key/value
    enc: Session,
    /// input_ids, past.{l}.*, cross.{l}.* -> logits (last position),
    /// present.{l}.*, alignment_heads
    dec: Session,
    layers: usize,
    heads: usize,
    head_dim: usize,
    /// bytes of each text token (ids below `eot`)
    vocab: Vec<Vec<u8>>,
    sot: i64,
    eot: i64,
    lang: i64,
    /// the language writes spaces between words (cue rules differ)
    spaced: bool,
    /// decode with timestamp tokens (lines); SUBTHX_WHISPER_TS=0 turns off
    timestamps: bool,
    /// speech packed into one window, in samples (at most 30 s);
    /// SUBTHX_WHISPER_WINDOW in seconds
    window: usize,
    lang_code: String,
    /// language code (ISO 639-1) -> its token
    languages: std::collections::HashMap<String, i64>,
    transcribe: i64,
    no_timestamps: i64,
    /// the first timestamp token (<|0.00|>)
    ts_begin: i64,
    /// the latest timestamp a window may start with
    max_initial: i64,
    begin_suppress: Vec<i64>,
    median_width: usize,
    device: Device,
    /// seconds in (features, encoder, decoder, alignment)
    pub profile: [f64; 4],
}

/// One batch of windows on its way through the model.
struct Window<'a> {
    /// first sample
    w0: usize,
    /// the VAD chunks in it
    idx: &'a [usize],
    /// samples from `w0` to the end of the last chunk
    span: usize,
    /// the chunks as encoder frames (20 ms) from `w0`
    speech: Vec<(usize, usize)>,
}

impl Whisper {
    /// `dir`: tokenizer.json, generation_config.json, config.json,
    /// mel_filters.npy (tools/dump_whisper.py) and
    /// onnx/{encoder_kv,decoder}_fp16.onnx (tools/whisper_onnx.py).
    /// `language`: ISO 639-1.
    pub fn load(dir: &Path, language: &str, device: Device) -> Result<Whisper, String> {
        let read = |f: &str| -> Result<serde_json::Value, String> {
            serde_json::from_str(&std::fs::read_to_string(dir.join(f)).map_err(|e| format!("{}: {e}", dir.join(f).display()))?).map_err(err)
        };
        let gen = read("generation_config.json")?;
        let tok = read("tokenizer.json")?;
        let cfg = read("config.json")?;
        let id = |v: &serde_json::Value| v.as_i64().ok_or_else(|| format!("{}: bad generation_config.json", dir.display()));
        let eot = id(&gen["eos_token_id"])?;
        let languages: std::collections::HashMap<String, i64> = gen["lang_to_id"]
            .as_object()
            .into_iter()
            .flatten()
            .filter_map(|(k, v)| Some((k.trim_start_matches("<|").trim_end_matches("|>").to_string(), v.as_i64()?)))
            .collect();
        let lang = *languages.get(language).ok_or_else(|| format!("Whisper has no language {language:?}"))?;

        let dec = byte_decoder();
        let mut vocab = vec![Vec::new(); eot as usize];
        for (s, v) in tok["model"]["vocab"].as_object().ok_or("tokenizer.json: no vocab")? {
            if let Some(i) = v.as_u64().filter(|&i| i < eot as u64) {
                vocab[i as usize] = s.chars().filter_map(|c| dec.get(&c).copied()).collect();
            }
        }
        // the decoder's tensors grow every step: its arena grows by
        // doubling; the encoder's shapes repeat, so it takes exactly what it
        // needs
        let onnx = |f: &str, growing: bool| -> Result<Session, String> {
            let p = dir.join("onnx").join(f);
            crate::vad::session_opts(&p, device, false, growing).map_err(|e| format!("{}: {e}", p.display()))
        };
        Ok(Whisper {
            mel: WhisperMel::load(&dir.join("mel_filters.npy"))?,
            enc: onnx("encoder_kv_fp16.onnx", false)?,
            dec: onnx("decoder_fp16.onnx", true)?,
            layers: cfg["decoder_layers"].as_u64().unwrap_or(4) as usize,
            heads: cfg["decoder_attention_heads"].as_u64().unwrap_or(20) as usize,
            head_dim: (cfg["d_model"].as_u64().unwrap_or(1280) / cfg["decoder_attention_heads"].as_u64().unwrap_or(20)) as usize,
            vocab,
            sot: id(&gen["decoder_start_token_id"])?,
            eot,
            lang,
            spaced: !crate::textsrc::unspaced(language),
            timestamps: std::env::var("SUBTHX_WHISPER_TS").map_or(true, |v| v != "0"),
            window: std::env::var("SUBTHX_WHISPER_WINDOW").ok().and_then(|v| v.parse::<usize>().ok()).map_or(WINDOW, |s| (s * RATE).min(WINDOW)),
            lang_code: language.to_string(),
            languages,
            transcribe: id(&gen["task_to_id"]["transcribe"])?,
            no_timestamps: id(&gen["no_timestamps_token_id"])?,
            ts_begin: id(&gen["no_timestamps_token_id"])? + 1,
            max_initial: gen["max_initial_timestamp_index"].as_i64().unwrap_or(50),
            begin_suppress: gen["begin_suppress_tokens"].as_array().into_iter().flatten().filter_map(|v| v.as_i64()).collect(),
            median_width: cfg["median_filter_width"].as_u64().unwrap_or(7) as usize,
            device,
            profile: [0.0; 4],
        })
    }

    fn device_mem(device: Device) -> Result<MemoryInfo<'static>, String> {
        let dev = match device {
            Device::Cuda => AllocationDevice::CUDA,
            Device::DirectMl => AllocationDevice::DIRECTML,
            Device::Cpu => AllocationDevice::CPU,
        };
        MemoryInfo::new(dev, 0, AllocatorType::Device, MemoryType::Default).map_err(err)
    }

    fn text(&self, ids: &[i64]) -> String {
        let bytes: Vec<u8> = ids.iter().filter(|&&i| i >= 0 && i < self.eot).flat_map(|&i| self.vocab[i as usize].iter().copied()).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode with timestamp tokens (the model's lines) or without; the
    /// SUBTHX_WHISPER_TS environment variable wins.
    pub fn set_timestamps(&mut self, on: bool) {
        if std::env::var_os("SUBTHX_WHISPER_TS").is_none() {
            self.timestamps = on;
        }
    }

    /// Transcribe in `language` (ISO 639-1) from now on.
    pub fn set_language(&mut self, language: &str) -> Result<(), String> {
        self.lang = *self.languages.get(language).ok_or_else(|| format!("Whisper has no language {language:?}"))?;
        self.spaced = !crate::textsrc::unspaced(language);
        self.lang_code = language.to_string();
        Ok(())
    }

    fn prompt(&self) -> [i64; 4] {
        [self.sot, self.lang, self.transcribe, self.no_timestamps]
    }

    /// The prompt to decode with timestamps: the model was trained on
    /// subtitles and closes every line with one, so a change of speaker
    /// shows even where no punctuation does.
    fn timed_prompt(&self) -> [i64; 3] {
        [self.sot, self.lang, self.transcribe]
    }

    /// `{kind}.{layer}.key`, `{kind}.{layer}.value` for every decoder layer
    fn cache_names(&self, kind: &str) -> Vec<String> {
        (0..self.layers).flat_map(|l| ["key", "value"].map(|kv| format!("{kind}.{l}.{kv}"))).collect()
    }

    /// One decoder pass: `ids` [b, T] after the self-attention caches `past`
    /// (none: from the start) over the windows' `cross` caches. Gives the
    /// last position's logits (on the CPU), the new caches (on the device)
    /// and, when `align`, the alignment heads' attention (on the CPU).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn decode(
        &mut self,
        ids: &[i64],
        b: usize,
        past: Option<&[DynValue]>,
        cross: &[DynValue],
        align: bool,
        opts: Option<&ort::session::RunOptions>,
    ) -> Result<(DynValue, Vec<DynValue>, Option<DynValue>), String> {
        let dev = Self::device_mem(self.device)?;
        let cpu = MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::CPUOutput).map_err(err)?;
        let ids = Tensor::from_array(([b, ids.len() / b], ids.to_vec())).map_err(err)?;
        let empty: Vec<DynValue> = match past {
            Some(_) => Vec::new(),
            None => (0..2 * self.layers)
                .map(|_| Tensor::from_array(([b, self.heads, 0, self.head_dim], Vec::<half::f16>::new())).map(|t| t.into_dyn()))
                .collect::<Result<_, _>>()
                .map_err(err)?,
        };
        let past = past.unwrap_or(&empty);
        let present = self.cache_names("present");
        let mut bind = self.dec.create_binding().map_err(err)?;
        bind.bind_input("input_ids", &ids).map_err(err)?;
        for (n, kv) in self.cache_names("past").iter().zip(past) {
            bind.bind_input(n.as_str(), kv).map_err(err)?;
        }
        for (n, kv) in self.cache_names("cross").iter().zip(cross) {
            bind.bind_input(n.as_str(), kv).map_err(err)?;
        }
        bind.bind_output_to_device("logits", &cpu).map_err(err)?;
        bind.bind_output_to_device("alignment_heads", if align { &cpu } else { &dev }).map_err(err)?;
        for n in &present {
            bind.bind_output_to_device(n.as_str(), &dev).map_err(err)?;
        }
        let mut outs = match opts {
            Some(o) => self.dec.run_binding_with_options(&bind, o),
            None => self.dec.run_binding(&bind),
        }
        .map_err(err)?;
        let logits = outs.remove("logits").ok_or("decoder: no logits")?;
        let att = outs.remove("alignment_heads").filter(|_| align);
        let kv = present.iter().map(|n| outs.remove(n.as_str()).ok_or("decoder: no cache")).collect::<Result<_, _>>()?;
        Ok((logits, kv, att))
    }

    /// Transcribe the speech in `chunks`, `batch` windows at a time,
    /// calling `on_batch(chunks done)` after each batch. Consecutive chunks
    /// share a 30 s window; the window's words are timed by alignment and go
    /// back to the chunks they were said in, so every cue keeps the VAD's
    /// edges.
    pub fn transcribe_with(
        &mut self,
        audio: &[f32],
        chunks: &[Chunk],
        batch: usize,
        on_batch: &mut dyn FnMut(usize) -> Result<(), String>,
    ) -> Result<Vec<Utterance>, String> {
        let mut windows: Vec<(usize, Vec<usize>)> = Vec::new();
        for (i, c) in chunks.iter().enumerate() {
            match windows.last_mut() {
                Some((w0, idx)) if c.end <= *w0 + self.window => idx.push(i),
                _ => windows.push((c.start, vec![i])),
            }
        }
        let mut out = Vec::new();
        let mut done = 0;
        for group in windows.chunks(batch.max(1)) {
            let wins: Vec<Window> = group
                .iter()
                .map(|(w0, idx)| Window {
                    w0: *w0,
                    idx,
                    span: chunks[*idx.last().unwrap()].end.min(audio.len()) - w0,
                    speech: idx.iter().map(|&i| ((chunks[i].start - w0) / (2 * HOP), (chunks[i].end - w0).div_ceil(2 * HOP))).collect(),
                })
                .collect();
            let t = std::time::Instant::now();
            let feats: Vec<f32> = wins
                .par_iter()
                .flat_map_iter(|w| {
                    // only the speech: music between chunks would turn into words
                    let mut wav = vec![0f32; WINDOW];
                    for &i in w.idx {
                        let c = &chunks[i];
                        let (a, b) = (c.start.max(w.w0), c.end.min(w.w0 + WINDOW).min(audio.len()));
                        if b > a {
                            wav[a - w.w0..b - w.w0].copy_from_slice(&audio[a..b]);
                        }
                    }
                    self.mel.features(&wav)
                })
                .collect();
            self.profile[0] += t.elapsed().as_secs_f64();
            let speech: Vec<f64> = wins.iter().map(|w| w.idx.iter().map(|&i| (chunks[i].end - chunks[i].start) as f64 / RATE as f64).sum()).collect();
            let (seqs, words) = self.run_batch(&wins, &speech, feats)?;
            for ((w, seq), words) in wins.iter().zip(&seqs).zip(words) {
                out.extend(self.place(w, chunks, seq, &words));
                done += w.idx.len();
            }
            on_batch(done)?;
        }
        Ok(out)
    }

    /// Encode, decode greedily and align a batch of windows: per window its
    /// text tokens and its words as (text, start s, end s) in the window.
    #[allow(clippy::type_complexity)]
    fn run_batch(&mut self, wins: &[Window], speech: &[f64], feats: Vec<f32>) -> Result<(Vec<Vec<i64>>, Vec<Vec<Word>>), String> {
        let b = wins.len();
        let n_mels = self.mel.n_mels();
        let dev = Self::device_mem(self.device)?;
        // arenas give back what a run no longer holds: shapes change every
        // step (the caches grow by a token) and freed blocks never fit again
        let mut shrink = ort::session::RunOptions::new().map_err(err)?;
        if self.device == Device::Cuda {
            shrink.set("memory.enable_memory_arena_shrinkage", "gpu:0").map_err(err)?;
        }

        let t = std::time::Instant::now();
        let input = Tensor::from_array(([b, n_mels, FRAMES], feats)).map_err(err)?;
        let cross_names = self.cache_names("cross");
        let mut bind = self.enc.create_binding().map_err(err)?;
        bind.bind_input("input_features", &input).map_err(err)?;
        for n in &cross_names {
            bind.bind_output_to_device(n.as_str(), &dev).map_err(err)?;
        }
        let mut outs = self.enc.run_binding_with_options(&bind, &shrink).map_err(err)?;
        let cross: Vec<DynValue> = cross_names.iter().map(|n| outs.remove(n.as_str()).ok_or("encoder: no cache")).collect::<Result<_, _>>()?;
        drop(outs);
        drop(bind);
        self.profile[1] += t.elapsed().as_secs_f64();

        // greedy decoding, no timestamps (the alignment times the words)
        let t = std::time::Instant::now();
        let prompt: Vec<i64> = if self.timestamps { self.timed_prompt().to_vec() } else { self.prompt().to_vec() };
        let ids: Vec<i64> = (0..b).flat_map(|_| prompt.iter().copied()).collect();
        let (mut logits, mut selfkv, _) = self.decode(&ids, b, None, &cross, false, Some(&shrink))?;

        // no window says more than its speech allows (a looping one would
        // hold up the whole batch)
        let limits: Vec<usize> = speech.iter().map(|s| ((TOKENS_PER_SECOND * s) as usize + 8).min(MAX_TOKENS)).collect();
        let longest = limits.iter().copied().max().unwrap_or(0);
        // text and timestamp tokens
        let mut seqs: Vec<Vec<i64>> = vec![Vec::new(); b];
        let mut text_len = vec![0usize; b];
        let mut finished = vec![false; b];
        for step in 0..longest {
            let (_, data) = logits.try_extract_tensor::<f32>().map_err(err)?;
            let v = data.len() / b;
            let mut next = vec![self.eot; b];
            for k in 0..b {
                if finished[k] {
                    continue;
                }
                let tok = self.pick(&data[k * v..(k + 1) * v], &seqs[k], text_len[k] == 0);
                if tok == self.eot {
                    finished[k] = true;
                    continue;
                }
                seqs[k].push(tok);
                text_len[k] += (tok < self.eot) as usize;
                if text_len[k] >= limits[k] || seqs[k].len() >= MAX_TOKENS || cut_loop(&mut seqs[k]) {
                    finished[k] = true;
                    continue;
                }
                next[k] = tok;
            }
            if finished.iter().all(|&f| f) || step + 1 == longest {
                break;
            }
            let opts = (step % SHRINK_EVERY == SHRINK_EVERY - 1).then_some(&shrink);
            (logits, selfkv, _) = self.decode(&next, b, Some(&selfkv), &cross, false, opts)?;
        }
        drop((selfkv, logits));
        self.profile[2] += t.elapsed().as_secs_f64();

        // the text, and where the model's lines start in it
        let mut texts = Vec::with_capacity(b);
        let mut lines = Vec::with_capacity(b);
        for seq in &seqs {
            let (mut text, mut starts, mut after_ts) = (Vec::new(), Vec::new(), false);
            for &t in seq {
                if t >= self.ts_begin {
                    after_ts = true;
                } else if t < self.eot {
                    if after_ts && !text.is_empty() {
                        starts.push(text.len());
                    }
                    after_ts = false;
                    text.push(t);
                }
            }
            texts.push(text);
            lines.push(starts);
        }

        // alignment: one teacher-forced pass over prompt + text + eot
        let t = std::time::Instant::now();
        let words = self.align_batch(wins, &texts, &lines, &cross, &shrink)?;
        self.profile[3] += t.elapsed().as_secs_f64();
        Ok((texts, words))
    }

    /// Greedy choice under Whisper's timestamp rules (openai
    /// `ApplyTimestampRules`, as faster-whisper decodes): text comes in
    /// segments `<t0> ... <t1>`, the first token is a timestamp, times never
    /// go back, and timestamps win when together they are likelier than any
    /// text token. Nothing blank starts the text.
    fn pick(&self, row: &[f32], seq: &[i64], first_text: bool) -> i64 {
        if !self.timestamps {
            let mut best = (row[self.eot as usize], self.eot);
            for (i, &x) in row[..self.eot as usize].iter().enumerate() {
                if x > best.0 && !(first_text && self.begin_suppress.contains(&(i as i64))) {
                    best = (x, i as i64);
                }
            }
            return best.1;
        }
        let tb = self.ts_begin;
        let last_ts = seq.last().is_some_and(|&t| t >= tb);
        let penultimate_ts = seq.len() < 2 || seq[seq.len() - 2] >= tb;
        let mut text_ok = !seq.is_empty();
        let mut ts_ok = true;
        if last_ts {
            if penultimate_ts {
                ts_ok = false;
            } else {
                text_ok = false;
            }
        }
        let ts_min = match seq.iter().rev().find(|&&t| t >= tb) {
            Some(&l) if last_ts && !penultimate_ts => l,
            Some(&l) => l + 1,
            None => tb,
        };
        let ts_max = if seq.is_empty() { tb + self.max_initial } else { row.len() as i64 - 1 };
        // text tokens and eot; the other specials never
        let mut best_text = (f32::NEG_INFINITY, self.eot);
        if !seq.is_empty() {
            best_text = (row[self.eot as usize], self.eot);
        }
        if text_ok {
            for (i, &x) in row[..self.eot as usize].iter().enumerate() {
                if x > best_text.0 && !(first_text && self.begin_suppress.contains(&(i as i64))) {
                    best_text = (x, i as i64);
                }
            }
        }
        let mut best_ts = (f32::NEG_INFINITY, self.eot);
        let mut sum = 0f64;
        if ts_ok && ts_min <= ts_max {
            let span = &row[ts_min as usize..=(ts_max as usize).min(row.len() - 1)];
            for (i, &x) in span.iter().enumerate() {
                if x > best_ts.0 {
                    best_ts = (x, ts_min + i as i64);
                }
            }
            sum = span.iter().map(|&x| ((x - best_ts.0) as f64).exp()).sum();
        }
        // log-sum-exp of the timestamps against the best text token
        let ts_mass = best_ts.0 as f64 + sum.ln();
        if best_ts.0 > f32::NEG_INFINITY && (ts_mass > best_text.0 as f64 || best_ts.0 > best_text.0) {
            best_ts.1
        } else {
            best_text.1
        }
    }

    /// openai-whisper `find_alignment` (CTranslate2 `align`): the alignment
    /// heads' attention over the frames, normalized per frame across tokens,
    /// median-filtered, averaged, and walked with DTW; word edges are where
    /// the path moves to the word's first token.
    fn align_batch(
        &mut self,
        wins: &[Window],
        seqs: &[Vec<i64>],
        lines: &[Vec<usize>],
        cross: &[DynValue],
        shrink: &ort::session::RunOptions,
    ) -> Result<Vec<Vec<Word>>, String> {
        let b = wins.len();
        let prompt = self.prompt();
        let rows: Vec<Vec<i64>> = seqs.iter().map(|s| prompt.iter().chain(s.iter()).chain([self.eot].iter()).copied().collect()).collect();
        let t_len = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        // padding after eot: causal attention keeps it out of earlier rows
        let ids: Vec<i64> = rows.iter().flat_map(|r| r.iter().copied().chain(std::iter::repeat(self.eot)).take(t_len)).collect();
        let (_, _, att) = self.decode(&ids, b, None, cross, true, Some(shrink))?;
        let att = att.ok_or("alignment: no output")?;
        let (shape, data) = att.try_extract_tensor::<f32>().map_err(err)?;
        let heads = shape[1] as usize;
        let frames_all = shape[3] as usize;
        let sot_len = 3; // sot, language, task
        let median = self.median_width;
        let out: Vec<Vec<Word>> = (0..b)
            .into_par_iter()
            .map(|k| {
                let text = &seqs[k];
                if text.is_empty() {
                    return Vec::new();
                }
                // encoder frames are 20 ms: the window's speech only
                let nf = (wins[k].span.div_ceil(2 * HOP)).clamp(1, frames_all);
                // rows: positions no_timestamps .. last text token (eot dropped)
                let r0 = sot_len;
                let nr = 1 + text.len();
                let mut matrix = vec![0f32; nr * nf];
                for h in 0..heads {
                    // w[t][f] for the rows used, then standardized per frame over all rows
                    let at = |t: usize, f: usize| data[((k * heads + h) * t_len + t) * frames_all + f];
                    let all_rows = rows[k].len();
                    let mut w = vec![0f32; all_rows * nf];
                    for t in 0..all_rows {
                        for f in 0..nf {
                            w[t * nf + f] = at(t, f);
                        }
                    }
                    for f in 0..nf {
                        let n = all_rows as f32;
                        let mean = (0..all_rows).map(|t| w[t * nf + f]).sum::<f32>() / n;
                        let var = (0..all_rows).map(|t| (w[t * nf + f] - mean).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
                        let sd = var.sqrt().max(1e-10);
                        for t in 0..all_rows {
                            w[t * nf + f] = (w[t * nf + f] - mean) / sd;
                        }
                    }
                    for (i, t) in (r0..r0 + nr).enumerate() {
                        let filtered = median_filter(&w[t * nf..(t + 1) * nf], median);
                        for f in 0..nf {
                            matrix[i * nf + f] += filtered[f] / heads as f32;
                        }
                    }
                }
                // the path only walks speech frames (stable-ts's silence
                // suppression): no word can be timed into a pause, and the
                // edge between two chunks is an edge between columns
                let mut cols: Vec<usize> = wins[k].speech.iter().flat_map(|&(a, b)| a.min(nf)..b.min(nf)).collect();
                cols.dedup();
                if cols.is_empty() {
                    cols = (0..nf).collect();
                }
                let nc = cols.len();
                let mut neg = vec![0f32; nr * nc];
                for r in 0..nr {
                    for (j, &f) in cols.iter().enumerate() {
                        neg[r * nc + j] = -matrix[r * nf + f];
                    }
                }
                let (text_idx, time_idx) = dtw(&neg, nr, nc);
                // the column where the path enters each row
                let mut jump_cols = Vec::with_capacity(nr);
                for i in 0..text_idx.len() {
                    if i == 0 || text_idx[i] != text_idx[i - 1] {
                        jump_cols.push(time_idx[i]);
                    }
                }
                let t_of = |c: usize| cols[c.min(nc - 1)] as f64 * 2.0 * HOP as f64 / RATE as f64;
                let words = self.split_words(text);
                let mut bounds = vec![0usize];
                for (_, toks) in &words {
                    bounds.push(bounds.last().unwrap() + toks);
                }
                words
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, (w, _))| {
                        let shift = ROW_SHIFT.min(1);
                        let c0 = *jump_cols.get(bounds[i] + shift).or(jump_cols.last())?;
                        let c1 = jump_cols.get(bounds[i + 1] + shift).copied().unwrap_or(nc).max(c0 + 1);
                        let line = lines[k].iter().any(|&l| bounds[i] <= l && l < bounds[i + 1]);
                        Some(Word { text: w, start: t_of(c0), end: t_of(c1 - 1) + 0.02, peak: t_of((c0 + c1 - 1) / 2), line })
                    })
                    .collect()
            })
            .collect();
        Ok(out)
    }

    /// Words of a transcript: for languages without spaces, every run of
    /// tokens that decodes to whole characters (openai `split_tokens_on_unicode`).
    fn split_words(&self, text: &[i64]) -> Vec<(String, usize)> {
        let mut words = Vec::new();
        let mut cur: Vec<i64> = Vec::new();
        for &t in text {
            cur.push(t);
            let s = self.text(&cur);
            if !s.contains('\u{FFFD}') {
                words.push((s, cur.len()));
                cur.clear();
            }
        }
        if !cur.is_empty() {
            words.push((self.text(&cur), cur.len()));
        }
        if !self.spaced {
            return words;
        }
        // openai `split_tokens_on_spaces`: a word starts at a space, and
        // punctuation stands alone
        let mut joined: Vec<(String, usize)> = Vec::new();
        for (w, n) in words {
            let punct = !w.trim().is_empty() && w.trim().chars().all(|c| c.is_ascii_punctuation());
            match joined.last_mut() {
                Some(last) if !w.starts_with(' ') && !punct => {
                    last.0.push_str(&w);
                    last.1 += n;
                }
                _ => joined.push((w, n)),
            }
        }
        joined
    }

    /// Cues of a window from its timed words, the way stable-ts regroups
    /// them for SubPlz: a cue ends at the end of a sentence or at a real pause
    /// between VAD chunks (the VAD also cuts long speech every few seconds,
    /// often mid-phrase; those cuts are not cue edges). A cue that starts or
    /// ends with its chunk takes the chunk's edge, otherwise the words' time.
    fn place(&self, w: &Window, chunks: &[Chunk], seq: &[i64], words: &[Word]) -> Vec<Utterance> {
        let sec = |s: usize| s as f64 / RATE as f64;
        let base = sec(w.w0);
        if words.is_empty() {
            // nothing to align: the whole window as one cue
            let text = self.text(seq);
            let (a, b) = (chunks[w.idx[0]].start, chunks[*w.idx.last().unwrap()].end);
            return if text.trim().is_empty() { vec![] } else { vec![Utterance { start: a, end: b, text }] };
        }
        // the chunk each word was said in
        let chunk_of = |mid: f64| -> usize {
            let dist = |i: usize| {
                let c = &chunks[w.idx[i]];
                let (a, b) = (sec(c.start), sec(c.end));
                if mid < a {
                    a - mid
                } else if mid > b {
                    mid - b
                } else {
                    0.0
                }
            };
            (0..w.idx.len()).min_by(|&a, &b| dist(a).total_cmp(&dist(b))).unwrap_or(0)
        };
        // a word goes to the chunk of its attention peak; punctuation stays
        // with the word before it
        let mut timed: Vec<(&str, f64, f64, usize)> = Vec::with_capacity(words.len());
        for wd in words {
            let punct = wd.text.chars().all(|c| !c.is_alphanumeric());
            let c = match timed.last() {
                Some(p) if punct => p.3,
                _ => chunk_of(base + wd.peak),
            };
            timed.push((wd.text.as_str(), base + wd.start, base + wd.end, c));
        }

        let mut out: Vec<Utterance> = Vec::new();
        let mut cur: Vec<usize> = Vec::new(); // indices into `timed`
        let flush = |cur: &mut Vec<usize>, out: &mut Vec<Utterance>| {
            if cur.is_empty() {
                return;
            }
            let (f, l) = (cur[0], *cur.last().unwrap());
            let text: String = cur.iter().map(|&i| timed[i].0).collect();
            let (ci, cj) = (timed[f].3, timed[l].3);
            // a chunk's first word starts the cue at the chunk's start
            let first_of_chunk = f == 0 || timed[f - 1].3 != ci;
            let last_of_chunk = l + 1 == timed.len() || timed[l + 1].3 != cj;
            let (c0, c1) = (&chunks[w.idx[ci]], &chunks[w.idx[cj]]);
            let mut start = if first_of_chunk { c0.start } else { (timed[f].1 * RATE as f64) as usize };
            let mut end = if last_of_chunk { c1.end } else { (timed[l].2 * RATE as f64) as usize };
            start = start.clamp(c0.start, c0.end);
            end = end.clamp(c1.start, c1.end);
            if let Some(p) = out.last() {
                start = start.max(p.end);
            }
            end = end.max(start + RATE / 10);
            // a cue of punctuation alone ("," between two lines) says nothing
            if text.chars().any(char::is_alphanumeric) {
                out.push(Utterance { start, end, text });
            }
            cur.clear();
        };
        for i in 0..timed.len() {
            // the model's own line breaks: a new speaker, as a rule
            if words[i].line {
                flush(&mut cur, &mut out);
            }
            if let Some(&p) = cur.last() {
                let (pc, ic) = (timed[p].3, timed[i].3);
                let pause = pc != ic && sec(chunks[w.idx[ic]].start) - sec(chunks[w.idx[pc]].end) >= PAUSE;
                let len: usize = cur.iter().map(|&k| timed[k].0.chars().count()).sum();
                // long cues also break at commas, and at the spaces that
                // separate the lines of lyrics
                // (a space is a line break only where words are not spaced)
                let comma = if self.spaced {
                    len >= LONG_CUE_SPACED && timed[p].0.ends_with([',', ';', ':'])
                } else {
                    len >= LONG_CUE && (timed[p].0.ends_with(['、', ',', ' ', '　']) || timed[i].0.starts_with([' ', '　']))
                };
                if pause {
                    // alignment runs a little early: the first word after a
                    // pause often lands at the end of the chunk before it.
                    // A short tail after the last punctuation goes along.
                    let tail = cur.iter().rposition(|&k| timed[k].0.trim_end().ends_with(['、', ',', '。', '！', '？', '!', '?', '…'])).map_or(0, |p| p + 1);
                    let tail_len: usize = cur[tail..].iter().map(|&k| timed[k].0.chars().count()).sum();
                    if tail_len > 0 && tail_len <= CARRY && !self.spaced {
                        let carried = cur.split_off(tail);
                        flush(&mut cur, &mut out);
                        cur = carried;
                    } else {
                        flush(&mut cur, &mut out);
                    }
                } else if comma {
                    flush(&mut cur, &mut out);
                }
            }
            cur.push(i);
            if timed[i].0.trim_end().ends_with(SENTENCE_END) {
                // spaced languages: not after "Mr." or an initial, not
                // before a lowercase word (the books' sentence rules)
                let ends = !self.spaced
                    || timed.get(i + 1).is_none_or(|next| {
                        let before: String = cur.iter().map(|&k| timed[k].0).collect();
                        crate::textsrc::sentence_break(&before, next.0, &self.lang_code)
                    });
                if ends {
                    flush(&mut cur, &mut out);
                }
            }
        }
        flush(&mut cur, &mut out);
        out
    }
}

/// A gap between VAD chunks this long (s) is a pause that ends a cue.
const PAUSE: f64 = 0.3;
/// Characters before a pause that move to the cue after it.
const CARRY: usize = 1;
/// Characters after which a comma ends a cue.
const LONG_CUE: usize = 24;
/// The same where words are spaced (a subtitle line holds about 42).
const LONG_CUE_SPACED: usize = 60;
/// What ends a sentence (and its cue); closing quotes go with their sentence.
const SENTENCE_END: [char; 10] = ['。', '！', '？', '!', '?', '…', '」', '』', '♪', '.'];

/// Alignment rows: 0 = openai's (the position predicting a token), 1 = the
/// token's own position.
const ROW_SHIFT: usize = 1;

/// Generation limit per second of speech in a window.
const TOKENS_PER_SECOND: f64 = 25.0;

/// Whisper is looping when the end repeats an n-gram (n = 2..=8) three
/// times, or one token six times: the repeats are cut off, and true says
/// the sequence is over.
fn cut_loop(seq: &mut Vec<i64>) -> bool {
    let len = seq.len();
    for n in 1..=8 {
        let reps = if n == 1 { 6 } else { 3 };
        if len < reps * n {
            continue;
        }
        let last = &seq[len - n..];
        if (1..reps).all(|r| &seq[len - (r + 1) * n..len - r * n] == last) {
            seq.truncate(len - (reps - 1) * n);
            return true;
        }
    }
    false
}

/// scipy-style median filter along a row (reflect padding), as openai uses.
fn median_filter(x: &[f32], width: usize) -> Vec<f32> {
    let n = x.len();
    if width <= 1 || n < width / 2 + 1 {
        return x.to_vec();
    }
    let pad = width / 2;
    let at = |i: isize| -> f32 {
        // numpy "reflect": ... 2 1 | 0 1 2 ... n-1 | n-2 n-3 ...
        let mut i = i;
        if i < 0 {
            i = -i;
        }
        if i >= n as isize {
            i = 2 * (n as isize - 1) - i;
        }
        x[i.clamp(0, n as isize - 1) as usize]
    };
    let mut buf = vec![0f32; width];
    (0..n)
        .map(|i| {
            for (k, b) in buf.iter_mut().enumerate() {
                *b = at(i as isize + k as isize - pad as isize);
            }
            buf.sort_by(|a, b| a.total_cmp(b));
            buf[pad]
        })
        .collect()
}

/// openai `dtw_cpu`: monotonic path through an n x m cost matrix.
fn dtw(x: &[f32], n: usize, m: usize) -> (Vec<usize>, Vec<usize>) {
    let w = m + 1;
    let mut cost = vec![f32::INFINITY; (n + 1) * w];
    let mut trace = vec![-1i8; (n + 1) * w];
    cost[0] = 0.0;
    for j in 1..=m {
        for i in 1..=n {
            let (c0, c1, c2) = (cost[(i - 1) * w + j - 1], cost[(i - 1) * w + j], cost[i * w + j - 1]);
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            cost[i * w + j] = x[(i - 1) * m + j - 1] + c;
            trace[i * w + j] = t;
        }
    }
    for j in 0..w {
        trace[j] = 2;
    }
    for i in 0..=n {
        trace[i * w] = 1;
    }
    let (mut i, mut j) = (n, m);
    let mut path = Vec::new();
    while i > 0 || j > 0 {
        // row 0 and column 0 are only ever left at (0, 0): finite costs
        // start there
        path.push((i.saturating_sub(1), j.saturating_sub(1)));
        match trace[i * w + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    path.reverse();
    path.into_iter().unzip()
}
