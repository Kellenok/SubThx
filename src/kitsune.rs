//! Kitsune (FastConformer CTC, HF `ParakeetForCTC`) through ONNX Runtime.
//! Export: tools/export_kitsune.py (argmax inside the graph). Features are the
//! same NeMo log-mel as Parakeet. Text follows HF `ParakeetProcessor.batch_decode`
//! (group repeats, drop blank and special tokens, Metaspace decoder).

use crate::asr::Chunk;
use crate::mel::NemoMel;
use crate::vad::Device;
use std::path::Path;

pub struct Kitsune {
    mel: NemoMel,
    session: ort::session::Session,
    vocab: Vec<String>,
    blank: i64,
    unk: i64,
    /// seconds in (features, model)
    pub profile: [f64; 2],
}

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

impl Kitsune {
    /// `dir` holds `model_file` (e.g. model.mixed16b.onnx) and tokenizer.json.
    pub fn load(dir: &Path, model_file: &str, fbank: &Path, device: Device) -> Result<Kitsune, String> {
        let tok: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer.json")).map_err(err)?).map_err(err)?;
        let mut vocab = vec![String::new(); 0];
        let mut put = |id: usize, s: &str| {
            if vocab.len() <= id {
                vocab.resize(id + 1, String::new());
            }
            vocab[id] = s.to_string();
        };
        for (s, id) in tok["model"]["vocab"].as_object().ok_or("tokenizer.json: no vocab")? {
            put(id.as_u64().ok_or("bad id")? as usize, s);
        }
        for a in tok["added_tokens"].as_array().into_iter().flatten() {
            put(a["id"].as_u64().unwrap_or(0) as usize, a["content"].as_str().unwrap_or(""));
        }
        let find = |name: &str| vocab.iter().position(|t| t == name).map(|i| i as i64);
        let blank = find("<blank>").ok_or("no <blank> token")?;
        let unk = find("<unk>").unwrap_or(-1);
        Ok(Kitsune {
            mel: NemoMel::load(fbank)?,
            session: crate::vad::session_with(&dir.join(model_file), device, true)?,
            vocab,
            blank,
            unk,
            profile: [0.0; 2],
        })
    }

    /// Transcribe `chunks`; features for the next batches are computed on the
    /// CPU while the GPU runs the current one.
    pub fn transcribe(&mut self, audio: &[f32], chunks: &[Chunk], batch_size: usize) -> Result<Vec<String>, String> {
        self.transcribe_with(audio, chunks, batch_size, &mut |_| Ok(()))
    }

    /// [`Kitsune::transcribe`] calling `on_batch(chunks done)` after every
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
        order.sort_by_key(|&i| chunks[i].end - chunks[i].start);
        let batches: Vec<&[usize]> = order.chunks(batch_size.max(1)).collect();
        let mut texts = vec![String::new(); chunks.len()];
        let Kitsune {
            mel,
            session,
            vocab,
            blank,
            unk,
            profile,
        } = self;
        let (mel, vocab, blank, unk) = (&*mel, &*vocab, *blank, *unk);
        let batches_ref = &batches;
        std::thread::scope(|scope| -> Result<(), String> {
            let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Vec<f32>, Vec<i64>, usize, f64)>(2);
            scope.spawn(move || {
                for (bi, batch) in batches_ref.iter().enumerate() {
                    let t0 = std::time::Instant::now();
                    let wavs: Vec<&[f32]> = batch.iter().map(|&i| &audio[chunks[i].start..chunks[i].end]).collect();
                    let (feats, lens, t) = mel.batch(&wavs, 16000);
                    if tx.send((bi, feats, lens, t, t0.elapsed().as_secs_f64())).is_err() {
                        return;
                    }
                }
            });
            for (bi, feats, lens, t, mel_s) in rx {
                profile[0] += mel_s;
                let t0 = std::time::Instant::now();
                let batch = batches_ref[bi];
                let b = batch.len();
                let out = session
                    .run(ort::inputs![
                        "features" => ort::value::TensorRef::from_array_view(([b, mel.n_mels(), t], &feats[..])).map_err(err)?,
                        "lengths" => ort::value::TensorRef::from_array_view(([b], &lens[..])).map_err(err)?
                    ])
                    .map_err(err)?;
                let (shape, ids) = out["ids"].try_extract_tensor::<i64>().map_err(err)?;
                let (_, olens) = out["out_lengths"].try_extract_tensor::<i64>().map_err(err)?;
                let t8 = shape[1] as usize;
                for (k, &i) in batch.iter().enumerate() {
                    let n = (olens[k].max(0) as usize).min(t8);
                    texts[i] = decode(vocab, blank, unk, &ids[k * t8..k * t8 + n]);
                }
                profile[1] += t0.elapsed().as_secs_f64();
                done += b;
                // dropping rx on error stops the mel producer
                on_batch(done)?;
            }
            Ok(())
        })?;
        Ok(texts)
    }
}

/// HF ParakeetProcessor CTC decode: group repeats, drop blank/unk, Metaspace.
fn decode(vocab: &[String], blank: i64, unk: i64, ids: &[i64]) -> String {
    let mut out = String::new();
    let mut prev = -1i64;
    let mut first = true;
    for &id in ids {
        if id == prev {
            continue;
        }
        prev = id;
        if id == blank || id == unk || id < 0 || id as usize >= vocab.len() {
            continue;
        }
        let tok = vocab[id as usize].replace('\u{2581}', " ");
        // Metaspace decoder with prepend_scheme "always": drop the leading space of the first token
        if first {
            out.push_str(tok.strip_prefix(' ').unwrap_or(&tok));
            first = false;
        } else {
            out.push_str(&tok);
        }
    }
    out
}
