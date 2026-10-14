//! NeMo log-mel features for Parakeet, port of onnx_asr's NeMo preprocessor
//! (`nemo80.onnx` / `NemoPreprocessorNumpy`): preemphasis 0.97, centered STFT
//! (n_fft 512, Hann 400, hop 160), slaney mel bank, log(x + 2^-24),
//! per-feature normalisation over the valid frames, zeros beyond.

use rayon::prelude::*;
use realfft::RealFftPlanner;
use std::path::Path;
use std::sync::Arc;

const N_FFT: usize = 512;
const WIN: usize = 400;
const HOP: usize = 160;
const PREEMPH: f32 = 0.97;

pub struct NemoMel {
    /// [257][n_mels], from onnx_asr fbanks.npz
    fbank: Vec<f32>,
    n_mels: usize,
    /// np.hanning(400) zero-padded to 512, centered
    window: Vec<f64>,
    fft: Arc<dyn realfft::RealToComplex<f64>>,
}

impl NemoMel {
    pub fn load(fbank_npy: &Path) -> Result<NemoMel, String> {
        let (shape, fbank) = crate::npy::read_f32(fbank_npy)?;
        if shape.len() != 2 || shape[0] != N_FFT / 2 + 1 {
            return Err(format!("unexpected filterbank shape {shape:?}"));
        }
        let off = (N_FFT - WIN) / 2;
        let mut window = vec![0f64; N_FFT];
        for n in 0..WIN {
            window[off + n] = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (WIN - 1) as f64).cos();
        }
        let fft = RealFftPlanner::<f64>::new().plan_fft_forward(N_FFT);
        Ok(NemoMel {
            fbank,
            n_mels: shape[1],
            window,
            fft,
        })
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Batch features [B, n_mels, T] (T = max_len/160 + 1) and valid lengths.
    /// `pad_to` rounds max_len up to a multiple (fewer distinct encoder
    /// shapes); 1 keeps onnx_asr's exact shape.
    pub fn batch(&self, wavs: &[&[f32]], pad_to: usize) -> (Vec<f32>, Vec<i64>, usize) {
        let max_len = wavs.iter().map(|w| w.len()).max().unwrap_or(0).div_ceil(pad_to.max(1)) * pad_to.max(1);
        let t = max_len / HOP + 1;
        let m = self.n_mels;
        let mut out = vec![0f32; wavs.len() * m * t];
        let lens: Vec<i64> = wavs.iter().map(|w| (w.len() / HOP) as i64).collect();
        out.par_chunks_mut(m * t)
            .zip(wavs.par_iter())
            .for_each(|(dst, w)| self.one(w, max_len, t, dst));
        (out, lens, t)
    }

    fn one(&self, w: &[f32], max_len: usize, t: usize, dst: &mut [f32]) {
        let m = self.n_mels;
        let valid = w.len() / HOP;
        // preemphasis, zero beyond the utterance, centre padding of n_fft/2
        let pad = N_FFT / 2;
        let mut x = vec![0f32; max_len + 2 * pad];
        for i in 0..w.len() {
            let prev = if i == 0 { 0.0 } else { w[i - 1] };
            x[pad + i] = w[i] - PREEMPH * prev;
        }
        let mut buf = self.fft.make_input_vec();
        let mut spec = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut power = vec![0f32; N_FFT / 2 + 1];
        let mut logmel = vec![0f32; valid * m]; // [frame][mel]
        for f in 0..valid {
            let s = f * HOP;
            for i in 0..N_FFT {
                buf[i] = x[s + i] as f64 * self.window[i];
            }
            self.fft.process_with_scratch(&mut buf, &mut spec, &mut scratch).unwrap();
            for (p, c) in power.iter_mut().zip(&spec) {
                let a = c.norm() as f32;
                *p = a * a;
            }
            let row = &mut logmel[f * m..(f + 1) * m];
            row.fill(0.0);
            for (k, &p) in power.iter().enumerate() {
                if p == 0.0 {
                    continue;
                }
                let fb = &self.fbank[k * m..(k + 1) * m];
                for (r, &b) in row.iter_mut().zip(fb) {
                    *r += p * b;
                }
            }
            for r in row.iter_mut() {
                *r = (*r + 2f32.powi(-24)).ln();
            }
        }
        if valid == 0 {
            return;
        }
        for k in 0..m {
            let mean = (0..valid).map(|f| logmel[f * m + k]).sum::<f32>() / valid as f32;
            let var = (0..valid).map(|f| (logmel[f * m + k] - mean).powi(2)).sum::<f32>() / (valid as f32 - 1.0);
            let denom = var.sqrt() + 1e-5;
            let col = &mut dst[k * t..k * t + t];
            for f in 0..valid {
                col[f] = (logmel[f * m + k] - mean) / denom;
            }
        }
    }
}
