//! Parakeet-TDT (NeMo FastConformer + TDT) through ONNX Runtime, a port of
//! SubPlz `ParakeetModel.run_batched_asr` (length-bucketed batches, greedy
//! TDT decoding) and onnx_asr's token-to-text rules.

use crate::vad::Device;
use std::path::Path;

pub struct Parakeet {
    /// onnx_asr nemo80.onnx; only used when `mel` is None
    pre: Option<ort::session::Session>,
    mel: Option<crate::mel::NemoMel>,
    enc: ort::session::Session,
    dec: ort::session::Session,
    vocab: Vec<String>,
    blank: usize,
    max_tokens_per_step: i32,
    /// seconds spent in (preprocessor, encoder, decoder loop, decoder runs)
    pub profile: [f64; 4],
    /// the decoder returns token/duration ids (argmax in the graph,
    /// tools/slim_parakeet_v3.py) instead of all joint logits
    dec_argmax: bool,
}

/// Speech chunk in samples (16 kHz).
#[derive(Clone, Copy, Debug)]
pub struct Chunk {
    pub start: usize,
    pub end: usize,
}

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

impl Parakeet {
    /// `dir` holds encoder-model.fp16.onnx, decoder_joint-model.fp16.onnx,
    /// vocab.txt; `preprocessor` is the NeMo mel filterbank (.npy, native
    /// features) or onnx_asr's nemo80.onnx (reference, CPU).
    pub fn load(dir: &Path, preprocessor: &Path, device: Device) -> Result<Parakeet, String> {
        Parakeet::load_with(
            dir,
            "encoder-model.fp16.onnx",
            "decoder_joint-model.fp16.onnx",
            preprocessor,
            device,
            device,
        )
    }

    /// [`Parakeet::load`] with explicit files: v3 exports ship fp16/int8/2-bit
    /// encoders and an fp32 decoder. `enc_device` lets an encoder that only
    /// runs on the CPU (2-bit Redux) sit next to a GPU decoder.
    pub fn load_with(
        dir: &Path,
        encoder: &str,
        decoder: &str,
        preprocessor: &Path,
        enc_device: Device,
        dec_device: Device,
    ) -> Result<Parakeet, String> {
        let vocab_src = std::fs::read_to_string(dir.join("vocab.txt")).map_err(err)?;
        let mut vocab = Vec::new();
        for line in vocab_src.lines() {
            let (tok, id) = line.rsplit_once(' ').ok_or("bad vocab line")?;
            let id: usize = id.parse().map_err(err)?;
            if vocab.len() <= id {
                vocab.resize(id + 1, String::new());
            }
            vocab[id] = tok.replace('\u{2581}', " ");
        }
        let blank = vocab.iter().position(|t| t == "<blk>").ok_or("no <blk> in vocab")?;
        let dec = crate::vad::session_with(&dir.join(decoder), dec_device, true)?;
        let dec_argmax = dec.outputs().iter().any(|o| o.name() == "token");
        Ok(Parakeet {
            pre: if preprocessor.extension().is_some_and(|e| e == "onnx") {
                Some(crate::vad::session(preprocessor, Device::Cpu)?)
            } else {
                None
            },
            mel: if preprocessor.extension().is_some_and(|e| e == "onnx") {
                None
            } else {
                Some(crate::mel::NemoMel::load(preprocessor)?)
            },
            enc: crate::vad::session_with(&dir.join(encoder), enc_device, true)?,
            dec,
            dec_argmax,
            vocab,
            blank,
            max_tokens_per_step: 10,
            profile: [0.0; 4],
        })
    }

    /// Transcribe `chunks` of `audio`; returns one text per chunk (input order).
    pub fn transcribe(&mut self, audio: &[f32], chunks: &[Chunk], batch_size: usize) -> Result<Vec<String>, String> {
        self.transcribe_with(audio, chunks, batch_size, &mut |_| Ok(()))
    }

    /// [`Parakeet::transcribe`] calling `on_batch(chunks done)` after every
    /// batch; an error from it stops the run.
    pub fn transcribe_with(
        &mut self,
        audio: &[f32],
        chunks: &[Chunk],
        batch_size: usize,
        on_batch: &mut dyn FnMut(usize) -> Result<(), String>,
    ) -> Result<Vec<String>, String> {
        let mut done = 0;
        let mut order: Vec<usize> = (0..chunks.len()).collect();
        order.sort_by_key(|&i| chunks[i].end - chunks[i].start); // stable, like python sorted()
        let mut texts = vec![String::new(); chunks.len()];
        for batch in order.chunks(batch_size.max(1)) {
            let out = self.batch(audio, chunks, batch)?;
            for (&i, t) in batch.iter().zip(out) {
                texts[i] = t;
            }
            done += batch.len();
            on_batch(done)?;
        }
        Ok(texts)
    }

    fn batch(&mut self, audio: &[f32], chunks: &[Chunk], idx: &[usize]) -> Result<Vec<String>, String> {
        let b = idx.len();
        let t0 = std::time::Instant::now();
        let (feats, flens, fshape) = if let Some(mel) = &self.mel {
            let wavs: Vec<&[f32]> = idx.iter().map(|&i| &audio[chunks[i].start..chunks[i].end]).collect();
            let pad = std::env::var("SUBTHX_PAD_SAMPLES").ok().and_then(|v| v.parse().ok()).unwrap_or(16000);
            let (f, l, t) = mel.batch(&wavs, pad);
            (f, l, vec![b, mel.n_mels(), t])
        } else {
            self.onnx_features(audio, chunks, idx)?
        };
        self.profile[0] += t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();

        // 2. encoder -> [B, 1024, T]
        let enc_out = self
            .enc
            .run(ort::inputs![
                "audio_signal" => ort::value::TensorRef::from_array_view((fshape.clone(), &feats[..])).map_err(err)?,
                "length" => ort::value::TensorRef::from_array_view(([b], &flens[..])).map_err(err)?
            ])
            .map_err(err)?;
        let (eshape, enc) = enc_out["outputs"].try_extract_tensor::<f32>().map_err(err)?;
        let (_, elens) = enc_out["encoded_lengths"].try_extract_tensor::<i64>().map_err(err)?;
        let (dim, t_max) = (eshape[1] as usize, eshape[2] as usize);
        let enc = enc.to_vec();
        let elens: Vec<i64> = elens.to_vec();
        drop(enc_out);
        self.profile[1] += t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();

        // 3. greedy TDT, all utterances in lockstep (SubPlz vectorised loop)
        let vocab_size = self.vocab.len();
        let mut t_idx = vec![0i64; b];
        let mut state1 = vec![0f32; 2 * b * 640];
        let mut state2 = vec![0f32; 2 * b * 640];
        let mut last_token = vec![self.blank as i32; b];
        let target_length = vec![1i32; b];
        let mut tokens: Vec<Vec<usize>> = vec![Vec::new(); b];
        let mut emitted = vec![0i32; b];
        let max_steps = elens.iter().max().copied().unwrap_or(0) * 2 + 50;
        let mut steps = 0;
        let mut enc_slice = vec![0f32; b * dim];
        while (0..b).any(|k| t_idx[k] < elens[k]) {
            steps += 1;
            if steps > max_steps {
                break;
            }
            for k in 0..b {
                let t = t_idx[k].min(elens[k] - 1).max(0) as usize;
                let base = k * dim * t_max;
                for d in 0..dim {
                    enc_slice[k * dim + d] = enc[base + d * t_max + t];
                }
            }
            let td = std::time::Instant::now();
            let out = self
                .dec
                .run(ort::inputs![
                    "encoder_outputs" => ort::value::TensorRef::from_array_view(([b, dim, 1], &enc_slice[..])).map_err(err)?,
                    "targets" => ort::value::TensorRef::from_array_view(([b, 1], &last_token[..])).map_err(err)?,
                    "target_length" => ort::value::TensorRef::from_array_view(([b], &target_length[..])).map_err(err)?,
                    "input_states_1" => ort::value::TensorRef::from_array_view(([2, b, 640], &state1[..])).map_err(err)?,
                    "input_states_2" => ort::value::TensorRef::from_array_view(([2, b, 640], &state2[..])).map_err(err)?
                ])
                .map_err(err)?;
            self.profile[3] += td.elapsed().as_secs_f64();
            // (token, duration) per row
            let picks: Vec<(usize, i64)> = if self.dec_argmax {
                let (_, tk) = out["token"].try_extract_tensor::<i64>().map_err(err)?;
                let (_, du) = out["duration"].try_extract_tensor::<i64>().map_err(err)?;
                tk.iter().zip(du).map(|(&t, &d)| (t as usize, d)).collect()
            } else {
                let (oshape, logits) = out["outputs"].try_extract_tensor::<f32>().map_err(err)?;
                let width = oshape[3] as usize;
                (0..b)
                    .map(|k| {
                        let row = &logits[k * width..(k + 1) * width];
                        (argmax(&row[..vocab_size]), argmax(&row[vocab_size..]) as i64)
                    })
                    .collect()
            };
            let (_, s1) = out["output_states_1"].try_extract_tensor::<f32>().map_err(err)?;
            let (_, s2) = out["output_states_2"].try_extract_tensor::<f32>().map_err(err)?;
            for k in 0..b {
                if t_idx[k] >= elens[k] {
                    continue;
                }
                let (tok, dur) = picks[k];
                if tok != self.blank {
                    tokens[k].push(tok);
                    last_token[k] = tok as i32;
                    for l in 0..2 {
                        let o = (l * b + k) * 640;
                        state1[o..o + 640].copy_from_slice(&s1[o..o + 640]);
                        state2[o..o + 640].copy_from_slice(&s2[o..o + 640]);
                    }
                    emitted[k] += 1;
                }
                if dur > 0 {
                    t_idx[k] += dur;
                    emitted[k] = 0;
                } else if tok == self.blank || emitted[k] == self.max_tokens_per_step {
                    t_idx[k] += 1;
                    emitted[k] = 0;
                }
            }
        }
        self.profile[2] += t0.elapsed().as_secs_f64();
        Ok(tokens.iter().map(|t| self.detokenize(t)).collect())
    }

    fn onnx_features(&mut self, audio: &[f32], chunks: &[Chunk], idx: &[usize]) -> Result<(Vec<f32>, Vec<i64>, Vec<usize>), String> {
        let b = idx.len();
        let lens: Vec<i64> = idx.iter().map(|&i| (chunks[i].end - chunks[i].start) as i64).collect();
        let max_len = *lens.iter().max().unwrap() as usize;
        let mut wav = vec![0f32; b * max_len];
        for (k, &i) in idx.iter().enumerate() {
            let c = chunks[i];
            wav[k * max_len..k * max_len + (c.end - c.start)].copy_from_slice(&audio[c.start..c.end]);
        }
        let pre = self.pre.as_mut().ok_or("no preprocessor")?;
        let pre_out = pre
            .run(ort::inputs![
                "waveforms" => ort::value::TensorRef::from_array_view(([b, max_len], &wav[..])).map_err(err)?,
                "waveforms_lens" => ort::value::TensorRef::from_array_view(([b], &lens[..])).map_err(err)?
            ])
            .map_err(err)?;
        let (fshape, feats) = pre_out["features"].try_extract_tensor::<f32>().map_err(err)?;
        let (_, flens) = pre_out["features_lens"].try_extract_tensor::<i64>().map_err(err)?;
        Ok((feats.to_vec(), flens.to_vec(), fshape.iter().map(|&d| d as usize).collect()))
    }

    /// onnx_asr `_decode_tokens` text, then SubPlz's `.strip()`.
    fn detokenize(&self, ids: &[usize]) -> String {
        let joined: Vec<char> = ids.iter().flat_map(|&i| self.vocab[i].chars()).collect();
        // re.sub(r"\A\s|\s\B|(\s)\b", lambda m: " " if m.group(1) else "", s)
        let is_word = |c: char| c.is_alphanumeric() || c == '_';
        let mut out = String::with_capacity(joined.len() * 3);
        for (i, &c) in joined.iter().enumerate() {
            if c.is_whitespace() {
                let next_word = joined.get(i + 1).is_some_and(|&n| is_word(n));
                if i > 0 && next_word {
                    out.push(' ');
                }
            } else {
                out.push(c);
            }
        }
        crate::batch::py_strip(&out).to_string()
    }
}

/// np.argmax: first maximum
fn argmax(v: &[f32]) -> usize {
    let mut m = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[m] {
            m = i;
        }
    }
    m
}
