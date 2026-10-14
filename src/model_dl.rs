//! Model downloads from the subthx organization on Hugging Face. Each repo
//! mirrors a part of the models folder: `kitsune-onnx` holds
//! `kitsune-p0.05b/…`, so a file at `<repo>/<path>` goes to `<models>/<path>`.
//! Every repo has a `manifest.json` (sizes and sha256) that the downloads
//! are checked against. A model needs the VAD (firered-vad-onnx) too.

use std::io::{Read, Write};
use std::path::Path;

use serde::Deserialize;

use crate::cuda::Progress;

const ORG: &str = "subthx";

/// Where a model's files are: `folder` in `repo`, plus `extra` files at the
/// repo root (the mel filterbanks Kitsune and Parakeet use).
pub struct Source {
    pub repo: &'static str,
    pub folder: String,
    pub extra: &'static [&'static str],
}

/// The source of a catalog model (None: not published).
pub fn source(id: &str) -> Option<Source> {
    let s = match id {
        "kitsune-p0.05b" | "kitsune-p0.1b" | "kitsune-p0.3b" => Source { repo: "kitsune-onnx", folder: id.to_string(), extra: &["nemo80_fbanks.npy"] },
        "parakeet-ultra" | "parakeet-redux" => Source { repo: "parakeet-v3-onnx", folder: id.to_string(), extra: &["nemo128_fbanks.npy"] },
        "whisper-ja-anime-v0.3" => Source { repo: "whisper-ja-anime-onnx", folder: id.to_string(), extra: &[] },
        "whisper-large-v3-turbo" => Source { repo: "whisper-large-v3-turbo-onnx", folder: id.to_string(), extra: &[] },
        "whisper-thewhisper-large-v3-turbo" => Source { repo: "thewhisper-large-v3-turbo-onnx", folder: id.to_string(), extra: &[] },
        _ => return None,
    };
    Some(s)
}

/// The voice activity model every transcription needs.
const VAD_REPO: &str = "firered-vad-onnx";

#[derive(Deserialize)]
struct Manifest {
    files: std::collections::BTreeMap<String, Entry>,
}

#[derive(Deserialize)]
struct Entry {
    size: u64,
    sha256: String,
}

fn manifest(agent: &ureq::Agent, repo: &str) -> Result<Manifest, String> {
    let url = format!("https://huggingface.co/{ORG}/{repo}/resolve/main/manifest.json");
    let body = agent
        .get(&url)
        .call()
        .map_err(|e| format!("{url}: {e}"))?
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("{url}: {e}"))?;
    serde_json::from_str(&body).map_err(|e| format!("{url}: {e}"))
}

/// The files of `id` that are not in `models` yet, as (repo, path, entry).
fn wanted(agent: &ureq::Agent, models: &Path, id: &str) -> Result<Vec<(&'static str, String, u64, String)>, String> {
    let src = source(id).ok_or_else(|| format!("{id}: not in the subthx repos"))?;
    let mut out = vec![];
    let main = manifest(agent, src.repo)?;
    for (path, e) in main.files {
        let mine = path.starts_with(&format!("{}/", src.folder)) || src.extra.contains(&path.as_str());
        if mine {
            out.push((src.repo, path, e.size, e.sha256));
        }
    }
    if out.is_empty() {
        return Err(format!("{id}: no files in {}", src.repo));
    }
    // the VAD's files only: the repo's root config.json is for the counter
    let vad = manifest(agent, VAD_REPO)?;
    for (path, e) in vad.files.into_iter().filter(|(p, _)| p.starts_with("firered_")) {
        out.push((VAD_REPO, path, e.size, e.sha256));
    }
    // files already in place (same size; they only get there verified)
    out.retain(|(_, path, size, _)| std::fs::metadata(models.join(path)).map(|m| m.len() != *size).unwrap_or(true));
    Ok(out)
}

/// Download the files of `id` that are missing into `models`, each checked
/// against its sha256 before it takes its final name. `progress` is called as
/// bytes arrive; an error from it cancels.
pub fn download(models: &Path, id: &str, progress: &mut dyn FnMut(&Progress) -> Result<(), String>) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    let agent = crate::cuda::agent();
    let src = source(id).ok_or_else(|| format!("{id}: not in the subthx repos"))?;
    // the Hub counts a download as a request to the repo's root config.json
    // (its download stats); a HEAD is enough, and a missing file is no error
    for repo in [src.repo, VAD_REPO] {
        let _ = agent.head(&format!("https://huggingface.co/{ORG}/{repo}/resolve/main/config.json")).call();
    }
    let files = wanted(&agent, models, id)?;
    // Hugging Face counts downloads by requests to a repo's root config.json
    // (HEAD or GET): one per repo the model's files come from
    let src = source(id).ok_or_else(|| format!("{id}: not in the subthx repos"))?;
    for repo in [src.repo, VAD_REPO] {
        agent.head(&format!("https://huggingface.co/{ORG}/{repo}/resolve/main/config.json")).call().ok();
    }
    let mut st = Progress { total: files.iter().map(|f| f.2).sum(), files: files.len(), stage: "downloading", ..Progress::default() };
    for (i, (repo, path, size, sha)) in files.iter().enumerate() {
        st.file = i + 1;
        let url = format!("https://huggingface.co/{ORG}/{repo}/resolve/main/{path}");
        let dst = path.split('/').fold(models.to_path_buf(), |p, part| p.join(part));
        if let Some(dir) = dst.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let part = dst.with_file_name(format!("{}.part", dst.file_name().unwrap_or_default().to_string_lossy()));
        let mut resp = agent.get(&url).call().map_err(|e| format!("{url}: {e}"))?;
        let mut body = resp.body_mut().with_config().limit(u64::MAX).reader();
        let mut file = std::fs::File::create(&part).map_err(|e| format!("{}: {e}", part.display()))?;
        let mut hash = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut got = 0u64;
        loop {
            let n = body.read(&mut buf).map_err(|e| format!("{url}: {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| format!("{}: {e}", part.display()))?;
            hash.update(&buf[..n]);
            got += n as u64;
            st.done += n as u64;
            progress(&st)?;
        }
        drop(file);
        let hex: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if got != *size || hex != *sha {
            std::fs::remove_file(&part).ok();
            return Err(format!("{url}: checksum mismatch"));
        }
        std::fs::rename(&part, &dst).map_err(|e| format!("{}: {e}", dst.display()))?;
    }
    Ok(())
}
