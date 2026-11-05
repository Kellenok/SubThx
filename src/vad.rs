//! FireRedVAD (DFSMN) as configured in SubPlz: kaldi fbank (torchaudio
//! `compliance.kaldi.fbank` semantics) -> CMVN -> ONNX model on 30000-frame
//! chunks -> FireRed `VadPostprocessor`.

use rayon::prelude::*;
use realfft::RealFftPlanner;
use std::path::Path;

pub const SAMPLE_RATE: usize = 16000;
const WIN: usize = 400; // 25 ms
const SHIFT: usize = 160; // 10 ms
const NFFT: usize = 512;
const NBINS: usize = 80;
const MODEL_CHUNK: usize = 30000;
const FRAME_SHIFT_S: f64 = 0.01;
const FRAME_LENGTH_S: f64 = 0.025;

pub struct Fbank {
    window: Vec<f32>,
    /// per mel bin: (first fft bin, weights)
    banks: Vec<(usize, Vec<f32>)>,
}

impl Default for Fbank {
    fn default() -> Self {
        Self::new()
    }
}

impl Fbank {
    pub fn new() -> Fbank {
        // povey window: hann(periodic=False)^0.85
        let window = (0..WIN)
            .map(|n| {
                let h = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (WIN - 1) as f64).cos();
                (h as f32).powf(0.85)
            })
            .collect();
        // torchaudio get_mel_banks(80, 512, 16000, low_freq=20, high_freq=8000)
        let mel = |f: f64| 1127.0 * (1.0 + f / 700.0).ln();
        let mel_low = mel(20.0);
        let mel_high = mel(8000.0);
        let delta = (mel_high - mel_low) / (NBINS + 1) as f64;
        let fft_bin_width = SAMPLE_RATE as f32 / NFFT as f32;
        let fft_mel: Vec<f32> = (0..NFFT / 2)
            .map(|k| 1127.0f32 * (1.0f32 + fft_bin_width * k as f32 / 700.0f32).ln())
            .collect();
        let banks = (0..NBINS)
            .map(|b| {
                let left = (mel_low + b as f64 * delta) as f32;
                let center = (mel_low + (b + 1) as f64 * delta) as f32;
                let right = (mel_low + (b + 2) as f64 * delta) as f32;
                let w: Vec<f32> = fft_mel
                    .iter()
                    .map(|&m| {
                        let up = (m - left) / (center - left);
                        let down = (right - m) / (right - center);
                        up.min(down).max(0.0)
                    })
                    .collect();
                let first = w.iter().position(|&x| x > 0.0).unwrap_or(0);
                let last = w.iter().rposition(|&x| x > 0.0).unwrap_or(0);
                (first, w[first..=last].to_vec())
            })
            .collect();
        Fbank { window, banks }
    }

    pub fn num_frames(n_samples: usize) -> usize {
        if n_samples < WIN {
            0
        } else {
            1 + (n_samples - WIN) / SHIFT
        }
    }

    /// Log mel features for `audio` in [-1, 1] (scaled by 32768 like SubPlz),
    /// CMVN applied, row-major [frames, 80].
    pub fn compute(&self, audio: &[f32], cmvn: &Cmvn) -> Vec<f32> {
        let frames = Self::num_frames(audio.len());
        let mut out = vec![0f32; frames * NBINS];
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(NFFT);
        out.par_chunks_mut(NBINS * 1024).enumerate().for_each_init(
            || (fft.make_input_vec(), fft.make_output_vec(), fft.make_scratch_vec(), vec![0f32; NFFT / 2 + 1]),
            |(buf, spec, scratch, power), (ci, rows)| {
                for (ri, row) in rows.chunks_mut(NBINS).enumerate() {
                    let f = ci * 1024 + ri;
                    let x = &audio[f * SHIFT..f * SHIFT + WIN];
                    let mean = x.iter().map(|&v| v * 32768.0).sum::<f32>() / WIN as f32;
                    let mut prev = x[0] * 32768.0 - mean;
                    for i in 0..WIN {
                        let cur = x[i] * 32768.0 - mean;
                        let pre = if i == 0 { cur - 0.97 * cur } else { cur - 0.97 * prev };
                        buf[i] = pre * self.window[i];
                        prev = cur;
                    }
                    buf[WIN..].fill(0.0);
                    fft.process_with_scratch(buf, spec, scratch).unwrap();
                    for (p, c) in power.iter_mut().zip(spec.iter()) {
                        // |c| as f32 `hypot` (what torch's abs() gives) but
                        // without a libm call per bin: the squares of two f32
                        // are exact in f64, and on 3e8 random pairs the result
                        // matched `f32::hypot` bit for bit
                        let a = ((c.re as f64) * (c.re as f64) + (c.im as f64) * (c.im as f64)).sqrt() as f32;
                        *p = a * a;
                    }
                    for (b, (first, w)) in self.banks.iter().enumerate() {
                        let e: f32 = w.iter().zip(&power[*first..]).map(|(a, b)| a * b).sum();
                        row[b] = (e.max(f32::EPSILON).ln() - cmvn.means[b]) * cmvn.inv_std[b];
                    }
                }
            },
        );
        out
    }
}

pub struct Cmvn {
    pub means: Vec<f32>,
    pub inv_std: Vec<f32>,
}

impl Cmvn {
    pub fn load(path: &Path) -> Result<Cmvn, String> {
        #[derive(serde::Deserialize)]
        struct J {
            means: Vec<f64>,
            inverse_std_variances: Vec<f64>,
        }
        let j: J = serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        Ok(Cmvn {
            means: j.means.iter().map(|&x| x as f32).collect(),
            inv_std: j.inverse_std_variances.iter().map(|&x| x as f32).collect(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
    DirectMl,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
            Device::DirectMl => "directml",
        })
    }
}

impl std::str::FromStr for Device {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "cuda" => Ok(Device::Cuda),
            "dml" | "directml" => Ok(Device::DirectMl),
            _ => Err(format!("unknown device {s:?} (cpu, cuda, directml)")),
        }
    }
}

fn environment() -> Result<&'static ort::environment::Environment, String> {
    static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
    if let Some(e) = ENV.get() {
        return Ok(e);
    }
    let e = ort::init().with_name("subthx").build().map_err(|e| e.to_string())?;
    Ok(ENV.get_or_init(|| e))
}

/// ONNX Runtime session on the requested device (fails instead of silently
/// falling back to CPU). TF32 off keeps fp32 models bit-close to CPU.
pub fn session(path: &Path, device: Device) -> Result<ort::session::Session, String> {
    session_with(path, device, false)
}

/// Like [`session`]; `tf32` enables TF32 math in cuDNN/cuBLAS (CUDA only).
/// The Parakeet encoder needs it: without TF32 ORT >= 1.21 finds no cuDNN
/// frontend plan for its convolutions and falls back to slow kernels.
pub fn session_with(path: &Path, device: Device, tf32: bool) -> Result<ort::session::Session, String> {
    session_opts(path, device, tf32, false)
}

/// `grow_by_doubling`: for models whose tensors grow a little every run (an
/// autoregressive decoder's caches). Growing the CUDA arena by exactly what
/// is asked would never reuse a freed block and leak a block per step.
pub fn session_opts(path: &Path, device: Device, tf32: bool, grow_by_doubling: bool) -> Result<ort::session::Session, String> {
    use ort::ep::{cuda::ConvAlgorithmSearch, DirectML, CUDA};
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let e = |e: ort::Error<ort::session::builder::SessionBuilder>| e.to_string();
    let mut b = ort::session::Session::builder(environment()?).map_err(|e| e.to_string())?;
    b = b.with_intra_threads(threads).map_err(e)?;
    b = match device {
        Device::Cpu => b,
        Device::Cuda => b
            .with_execution_providers([{
                // SUBTHX_CUDA_OPTS="search=exhaustive,nhwc=1,fuse=1,tf32=1" for experiments
                let opts = std::env::var("SUBTHX_CUDA_OPTS").unwrap_or_default();
                let get = |k: &str| opts.split(',').find_map(|kv| kv.strip_prefix(&format!("{k}=")).map(String::from));
                // grow the arena only by what is requested: batch shapes vary
                // all the time and the default power-of-two growth hoards VRAM
                let mut ep = CUDA::default()
                    .with_arena_extend_strategy(if grow_by_doubling {
                        ort::ep::ArenaExtendStrategy::NextPowerOfTwo
                    } else {
                        ort::ep::ArenaExtendStrategy::SameAsRequested
                    })
                    .with_tf32(get("tf32").map(|v| v == "1").unwrap_or(tf32))
                    .with_conv_algorithm_search(match get("search").as_deref() {
                        Some("exhaustive") => ConvAlgorithmSearch::Exhaustive,
                        Some("default") => ConvAlgorithmSearch::Default,
                        _ => ConvAlgorithmSearch::Heuristic,
                    });
                if let Some(v) = get("nhwc") {
                    ep = ep.with_prefer_nhwc(v == "1");
                }
                if let Some(v) = get("fuse") {
                    ep = ep.with_fuse_conv_bias(v == "1");
                }
                if let Some(v) = get("maxws") {
                    ep = ep.with_conv_max_workspace(v == "1");
                }
                ep.build().error_on_failure()
            }])
            .map_err(e)?,
        Device::DirectMl => b
            // SUBTHX_DML_OPT=0: no graph rewriting, for a GPU whose driver computes a fused node wrongly
            .with_optimization_level(if std::env::var("SUBTHX_DML_OPT").is_ok_and(|v| v.trim() == "0") {
                ort::session::builder::GraphOptimizationLevel::Disable
            } else {
                ort::session::builder::GraphOptimizationLevel::All
            })
            .and_then(|b| b.with_memory_pattern(false))
            .and_then(|b| b.with_parallel_execution(false))
            .and_then(|b| b.with_execution_providers([DirectML::default().build().error_on_failure()]))
            .map_err(e)?,
    };
    b.commit_from_file(path).map_err(|e| format!("{}: {e}", path.display()))
}

pub struct Model {
    session: ort::session::Session,
    /// 30000-frame chunks per run: batching helps GPUs, hurts the CPU EP
    batch: usize,
    /// seconds in (features, waiting for them, model), for SUBTHX_PROFILE
    pub profile: [f64; 3],
}

impl Model {
    pub fn load(path: &Path, device: Device) -> Result<Model, String> {
        Ok(Model {
            session: session(path, device)?,
            batch: if device == Device::Cpu { 1 } else { 8 },
            profile: [0.0; 3],
        })
    }

    /// Speech probability per frame; the model sees independent 30000-frame
    /// chunks, exactly like SubPlz. Full chunks are batched (same result,
    /// fewer runs).
    pub fn probs(&mut self, feats: &[f32]) -> Result<Vec<f32>, String> {
        let batch = std::env::var("SUBTHX_VAD_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(self.batch);
        let frames = feats.len() / NBINS;
        let full = frames / MODEL_CHUNK;
        let mut out = Vec::with_capacity(frames);
        let mut run = |out: &mut Vec<f32>, start: usize, b: usize, t: usize| -> Result<(), String> {
            let input = ort::value::TensorRef::from_array_view((
                [b, t, NBINS],
                &feats[start * NBINS..(start + b * t) * NBINS],
            ))
            .map_err(|e| e.to_string())?;
            let res = self.session.run(ort::inputs!["feat" => input]).map_err(|e| e.to_string())?;
            let (_, p) = res["probs"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
            out.extend_from_slice(p);
            Ok(())
        };
        let mut c = 0;
        while c < full {
            let b = batch.max(1).min(full - c);
            run(&mut out, c * MODEL_CHUNK, b, MODEL_CHUNK)?;
            c += b;
        }
        let rest = frames - full * MODEL_CHUNK;
        if rest > 0 {
            run(&mut out, full * MODEL_CHUNK, 1, rest)?;
        }
        Ok(out)
    }
}

impl Model {
    /// `probs(&fbank.compute(audio, cmvn))` with the features made a group of
    /// chunks at a time: the whole file's features are over 100 MB per hour
    /// of audio. A frame's features depend on its 400 samples only and the
    /// model sees independent chunks, so the result is the same.
    pub fn probs_audio(&mut self, audio: &crate::audio::Pcm, fbank: &Fbank, cmvn: &Cmvn) -> Result<Vec<f32>, String> {
        self.probs_frames(audio, fbank, cmvn, 0, Fbank::num_frames(audio.len()))
    }

    /// Frames `f0..f1` of [`Model::probs_audio`]: `f0` on a 30000-frame chunk
    /// boundary, so a file done a block at a time gives the same result.
    pub fn probs_frames(&mut self, audio: &crate::audio::Pcm, fbank: &Fbank, cmvn: &Cmvn, f0: usize, f1: usize) -> Result<Vec<f32>, String> {
        assert_eq!(f0 % MODEL_CHUNK, 0, "VAD blocks start on a chunk");
        let batch = std::env::var("SUBTHX_VAD_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(self.batch);
        let group = batch.max(1) * MODEL_CHUNK;
        let mut out = Vec::with_capacity(f1.saturating_sub(f0));
        // the next group's features are computed while the model runs on the current one
        let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<f32>, f64)>(1);
        std::thread::scope(|sc| {
            sc.spawn(move || {
                let mut start = f0;
                while start < f1 {
                    let t = std::time::Instant::now();
                    let n = group.min(f1 - start);
                    let x = audio.range(start * SHIFT, (start + n - 1) * SHIFT + WIN);
                    let feats = fbank.compute(&x, cmvn);
                    if tx.send((feats, t.elapsed().as_secs_f64())).is_err() {
                        break;
                    }
                    start += n;
                }
            });
            loop {
                let t = std::time::Instant::now();
                let Ok((feats, made)) = rx.recv() else { break };
                self.profile[0] += made;
                self.profile[1] += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                out.extend(self.probs(&feats)?);
                self.profile[2] += t.elapsed().as_secs_f64();
            }
            Ok::<(), String>(())
        })?;
        Ok(out)
    }
}

/// Frames per block of a file decoded and recognized a block at a time: 8
/// model chunks (what one VAD run takes on a GPU), 40 minutes.
pub const BLOCK_FRAMES: usize = 8 * MODEL_CHUNK;

/// Samples a block of frames `f0..f1` reads.
pub fn frames_samples(f1: usize) -> usize {
    if f1 == 0 {
        0
    } else {
        (f1 - 1) * SHIFT + WIN
    }
}

/// FireRed `VadPostprocessor` with SubPlz's FireRedVadConfig.
pub struct PostConfig {
    pub smooth_window_size: usize,
    pub prob_threshold: f64,
    pub min_speech_frame: usize,
    pub max_speech_frame: usize,
    pub min_silence_frame: usize,
    pub merge_silence_frame: usize,
}

impl Default for PostConfig {
    fn default() -> Self {
        PostConfig {
            smooth_window_size: 5,
            prob_threshold: 0.45,
            min_speech_frame: 15,
            max_speech_frame: 800,
            min_silence_frame: 30,
            merge_silence_frame: 20,
        }
    }
}

/// Python `round(x, 3)` (correctly rounded, like `format(x, ".3f")`)
pub fn round3(x: f64) -> f64 {
    format!("{x:.3}").parse().unwrap()
}

impl PostConfig {
    pub fn process(&self, raw: &[f32]) -> Vec<u8> {
        self.process_settled(raw).0
    }

    /// [`PostConfig::process`] of the start of a file's probabilities, and
    /// the frame up to which its segments are settled: a segment that ends at
    /// or before it comes out the same however the file goes on.
    ///
    /// Smoothing and the state machine look back only, except that speech is
    /// filled back to its start once it lasts `min_speech_frame` frames and
    /// `fix_smooth_window_start` reaches `smooth_window_size` frames before
    /// an onset: frames up to those two before the end are as they will be.
    /// A silence there of `merge_silence_frame` frames or more is never
    /// merged away, so the segments before it are final (with their split
    /// points, which only look inside a segment).
    pub fn process_settled(&self, raw: &[f32]) -> (Vec<u8>, usize) {
        if raw.is_empty() {
            return (Vec::new(), 0);
        }
        let probs: Vec<f64> = raw.iter().map(|&p| p as f64).collect();
        let w = self.smooth_window_size.max(1);
        // np.convolve(probs, ones(w)/w, 'full')[:n], first w-1 replaced by running mean
        let k = 1.0 / w as f64;
        let mut smoothed = vec![0f64; probs.len()];
        for i in 0..probs.len() {
            if i + 1 < w {
                smoothed[i] = probs[..=i].iter().sum::<f64>() / (i + 1) as f64;
            } else {
                let mut s = 0.0;
                for j in (i + 1 - w..=i).rev() {
                    s += probs[j] * k;
                }
                smoothed[i] = s;
            }
        }
        let binary: Vec<bool> = smoothed.iter().map(|&p| p >= self.prob_threshold).collect();
        let decisions = self.state_machine(&binary);
        let fixed = self.fix_smooth_window_start(&decisions);
        let merged = self.merge_short_silence(&fixed);
        let settled = self.settled(&merged);
        (self.split_long_speech(&merged, &probs), settled)
    }

    /// the start of the last long enough silence among the frames that will not change
    fn settled(&self, merged: &[u8]) -> usize {
        let sure = merged.len().saturating_sub(self.min_speech_frame + self.smooth_window_size + 1);
        let need = self.merge_silence_frame.max(1);
        let (mut settled, mut run_start, mut run) = (0, 0, 0);
        for (t, &d) in merged.iter().enumerate().take(sure) {
            if d == 0 {
                if run == 0 {
                    run_start = t;
                }
                run += 1;
                if run == need {
                    settled = run_start;
                }
            } else {
                run = 0;
            }
        }
        settled
    }

    fn state_machine(&self, binary: &[bool]) -> Vec<u8> {
        #[derive(PartialEq)]
        enum S {
            Silence,
            PossibleSpeech,
            Speech,
            PossibleSilence,
        }
        let mut d = vec![0u8; binary.len()];
        let mut state = S::Silence;
        let mut speech_start = 0usize;
        let mut silence_start = 0usize;
        for (t, &is_speech) in binary.iter().enumerate() {
            match state {
                S::Silence => {
                    if is_speech {
                        state = S::PossibleSpeech;
                        speech_start = t;
                    }
                }
                S::PossibleSpeech => {
                    if is_speech {
                        if t - speech_start >= self.min_speech_frame {
                            state = S::Speech;
                            d[speech_start..t].fill(1);
                        }
                    } else {
                        state = S::Silence;
                    }
                }
                S::Speech => {
                    if !is_speech {
                        state = S::PossibleSilence;
                        silence_start = t;
                    }
                }
                S::PossibleSilence => {
                    if !is_speech {
                        if t - silence_start >= self.min_silence_frame {
                            state = S::Silence;
                        }
                    } else {
                        state = S::Speech;
                    }
                }
            }
            d[t] = matches!(state, S::Speech | S::PossibleSilence) as u8;
        }
        d
    }

    fn fix_smooth_window_start(&self, d: &[u8]) -> Vec<u8> {
        let mut n = d.to_vec();
        for t in 1..d.len() {
            if d[t - 1] == 0 && d[t] == 1 {
                let start = t.saturating_sub(self.smooth_window_size);
                n[start..t].fill(1);
            }
        }
        n
    }

    fn merge_short_silence(&self, d: &[u8]) -> Vec<u8> {
        if self.merge_silence_frame == 0 {
            return d.to_vec();
        }
        let mut n = d.to_vec();
        let mut silence_start: Option<usize> = None;
        for t in 1..d.len() {
            if d[t - 1] == 1 && d[t] == 0 && silence_start.is_none() {
                silence_start = Some(t);
            } else if d[t - 1] == 0 && d[t] == 1 {
                if let Some(s) = silence_start {
                    if t - s < self.merge_silence_frame {
                        n[s..t].fill(1);
                    }
                    silence_start = None;
                }
            }
        }
        n
    }

    fn split_long_speech(&self, d: &[u8], probs: &[f64]) -> Vec<u8> {
        let mut n = d.to_vec();
        for (s, e) in decision_to_segment(d, None) {
            let start_frame = (s / FRAME_SHIFT_S) as usize;
            let end_frame = (e / FRAME_SHIFT_S) as usize;
            if end_frame - start_frame > self.max_speech_frame {
                let seg = &probs[start_frame..end_frame.min(probs.len())];
                for sp in self.find_split_points(seg) {
                    n[start_frame + sp] = 0;
                }
            }
        }
        n
    }

    fn find_split_points(&self, probs: &[f64]) -> Vec<usize> {
        let mut out = Vec::new();
        let len = probs.len();
        let mut start = 0usize;
        while start < len {
            if len - start <= self.max_speech_frame {
                break;
            }
            let ws = (start as f64 + self.max_speech_frame as f64 / 2.0) as usize;
            let we = start + self.max_speech_frame;
            let win = &probs[ws..we.min(len)];
            // np.argmin: first minimum
            let mut mi = 0;
            for (i, &p) in win.iter().enumerate() {
                if p < win[mi] {
                    mi = i;
                }
            }
            let min_index = ws + mi;
            out.push(min_index);
            start = min_index + 1;
        }
        out
    }
}

/// FireRed `decision_to_segment`, seconds rounded to 3 decimals.
pub fn decision_to_segment(d: &[u8], wav_dur: Option<f64>) -> Vec<(f64, f64)> {
    let mut segs = Vec::new();
    let mut speech_start: Option<usize> = None;
    for (t, &x) in d.iter().enumerate() {
        if x == 1 && speech_start.is_none() {
            speech_start = Some(t);
        } else if x == 0 {
            if let Some(s) = speech_start.take() {
                segs.push((s as f64 * FRAME_SHIFT_S, t as f64 * FRAME_SHIFT_S));
            }
        }
    }
    if let Some(s) = speech_start {
        let mut end = d.len() as f64 * FRAME_SHIFT_S + FRAME_LENGTH_S;
        if let Some(w) = wav_dur {
            end = end.min(w);
        }
        segs.push((s as f64 * FRAME_SHIFT_S, end));
    }
    segs.into_iter().map(|(s, e)| (round3(s), round3(e))).collect()
}

/// Speech segments in samples, as SubPlz's `run_vad` returns them.
pub fn segments_samples(decisions: &[u8], n_samples: usize) -> Vec<(usize, usize)> {
    decision_to_segment(decisions, Some(n_samples as f64 / SAMPLE_RATE as f64))
        .into_iter()
        .map(|(s, e)| ((s * SAMPLE_RATE as f64) as usize, (e * SAMPLE_RATE as f64) as usize))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// probabilities of speech and pauses of every length (long speech that
    /// gets split, pauses shorter than a merge)
    fn probs(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut out = Vec::with_capacity(n);
        let mut speech = false;
        while out.len() < n {
            let len = match rnd() % 6 {
                0 => 1 + rnd() % 10,
                1 => 10 + rnd() % 30,
                2 => 30 + rnd() % 100,
                3 => 100 + rnd() % 400,
                4 => 800 + rnd() % 1500,
                _ => 15 + rnd() % 10,
            } as usize;
            for _ in 0..len {
                let noise = (rnd() % 1000) as f32 / 1000.0 * 0.5;
                out.push(if speech { 0.5 + noise } else { noise * 0.9 });
            }
            speech = !speech;
        }
        out.truncate(n);
        out
    }

    #[test]
    fn settled_segments_do_not_change_later() {
        let cfg = PostConfig::default();
        for seed in 1..6u64 {
            let p = probs(60_000, seed * 0x9E37_79B9);
            let full = decision_to_segment(&cfg.process(&p), None);
            let mut last = 0;
            for end in (500..p.len()).step_by(997) {
                let (d, settled) = cfg.process_settled(&p[..end]);
                let cut = round3(settled as f64 * FRAME_SHIFT_S);
                let segs: Vec<_> = decision_to_segment(&d, None).into_iter().filter(|s| s.1 <= cut).collect();
                assert_eq!(segs[..], full[..segs.len()], "seed {seed}, first {end} frames");
                assert!(settled >= last, "settled went back");
                // it does not lag far behind
                assert!(end - settled < 20_000, "seed {seed}: settled {settled} of {end}");
                last = settled;
            }
        }
    }
}
