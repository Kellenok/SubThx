//! The CUDA libraries ONNX Runtime's CUDA provider loads (CUDA 13: cuBLAS,
//! cuDNN 9, NVRTC), downloaded on request from NVIDIA's redistributable
//! wheels on PyPI into `<models>/cuda`, which is put on the DLL search path.
//! Only the DLLs are kept (about 1.7 GB unpacked, 0.9 GB to download).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// (PyPI package, version) of each wheel.
const WHEELS: &[(&str, &str)] = &[("nvidia-cublas", "13.8.0.4"), ("nvidia-cudnn-cu13", "9.27.0.42"), ("nvidia-cuda-nvrtc", "13.4.92")];

/// What the provider loads (found by watching which DLLs a run of every
/// model loads); the rest of a wheel's bin folder comes along.
pub const REQUIRED: &[&str] = &[
    "cublas64_13.dll",
    "cublasLt64_13.dll",
    "cudnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_adv64_9.dll",
    "nvrtc64_130_0.dll",
];

/// `<models>/cuda`
pub fn dir(models: &Path) -> PathBuf {
    models.join("cuda")
}

/// The required DLLs found neither in `<models>/cuda` nor on PATH.
pub fn missing(models: &Path) -> Vec<&'static str> {
    let mut dirs = vec![dir(models)];
    dirs.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()));
    REQUIRED.iter().copied().filter(|f| !dirs.iter().any(|d| d.join(f).is_file())).collect()
}

/// Whether `<models>/cuda` holds every required DLL.
pub fn installed(models: &Path) -> bool {
    let d = dir(models);
    REQUIRED.iter().all(|f| d.join(f).is_file())
}

/// Put `<models>/cuda` on the DLL search path when it is there.
pub fn use_downloaded(models: &Path) -> Result<(), String> {
    if installed(models) {
        crate::service::add_dll_dir(&dir(models))?;
    }
    Ok(())
}

/// Download progress.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Progress {
    /// bytes downloaded, of `total`
    pub done: u64,
    pub total: u64,
    /// which wheel (1-based) of `files`
    pub file: usize,
    pub files: usize,
    /// "downloading" or "unpacking"
    pub stage: &'static str,
}

pub fn agent() -> ureq::Agent {
    use ureq::tls::{TlsConfig, TlsProvider};
    ureq::Agent::config_builder()
        .tls_config(TlsConfig::builder().provider(TlsProvider::NativeTls).build())
        .build()
        .new_agent()
}

/// (url, size, sha256) of a package's Windows x64 wheel
fn wheel(agent: &ureq::Agent, package: &str, version: &str) -> Result<(String, u64, String), String> {
    let url = format!("https://pypi.org/pypi/{package}/{version}/json");
    let body = agent
        .get(&url)
        .call()
        .map_err(|e| format!("{url}: {e}"))?
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("{url}: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| format!("{url}: {e}"))?;
    let f = v["urls"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|u| u["filename"].as_str().is_some_and(|n| n.ends_with("win_amd64.whl")))
        .ok_or_else(|| format!("{package} {version}: no Windows x64 wheel on PyPI"))?;
    Ok((
        f["url"].as_str().unwrap_or_default().to_string(),
        f["size"].as_u64().unwrap_or(0),
        f["digests"]["sha256"].as_str().unwrap_or_default().to_string(),
    ))
}

/// Download the wheels and unpack their DLLs into `<models>/cuda`.
/// `progress` is called as bytes arrive; an error from it cancels.
pub fn download(models: &Path, progress: &mut dyn FnMut(&Progress) -> Result<(), String>) -> Result<PathBuf, String> {
    use sha2::{Digest, Sha256};
    let agent = agent();
    let found: Vec<(String, u64, String)> = WHEELS.iter().map(|(p, v)| wheel(&agent, p, v)).collect::<Result<_, _>>()?;
    let mut st = Progress { total: found.iter().map(|f| f.1).sum(), files: found.len(), stage: "downloading", ..Progress::default() };
    let out = dir(models);
    let part = models.join("cuda.part");
    std::fs::create_dir_all(&part).map_err(|e| format!("{}: {e}", part.display()))?;
    for (i, (url, _, sha)) in found.iter().enumerate() {
        st.file = i + 1;
        st.stage = "downloading";
        let tmp = part.join(format!("{i}.whl"));
        let mut resp = agent.get(url).call().map_err(|e| format!("{url}: {e}"))?;
        let mut body = resp.body_mut().with_config().limit(u64::MAX).reader();
        let mut file = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let mut hash = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut since = 0u64;
        loop {
            let n = body.read(&mut buf).map_err(|e| format!("{url}: {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| format!("{}: {e}", tmp.display()))?;
            hash.update(&buf[..n]);
            st.done += n as u64;
            since += n as u64;
            if since >= 4 << 20 {
                since = 0;
                progress(&st)?;
            }
        }
        drop(file);
        let got: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if !sha.is_empty() && got != *sha {
            return Err(format!("{url}: checksum mismatch"));
        }
        st.stage = "unpacking";
        progress(&st)?;
        // the DLLs of the wheel's bin folder (nvidia/cublas/bin/…)
        let mut zip = zip::ZipArchive::new(std::fs::File::open(&tmp).map_err(|e| e.to_string())?).map_err(|e| format!("{url}: {e}"))?;
        for k in 0..zip.len() {
            let mut entry = zip.by_index(k).map_err(|e| e.to_string())?;
            let name = entry.name().map_err(|e| e.to_string())?.to_string();
            let Some(base) = name.rsplit('/').next().filter(|b| b.to_ascii_lowercase().ends_with(".dll") && name.contains("/bin/")) else {
                continue;
            };
            let dst = part.join(base);
            let mut f = std::fs::File::create(&dst).map_err(|e| format!("{}: {e}", dst.display()))?;
            std::io::copy(&mut entry, &mut f).map_err(|e| format!("{}: {e}", dst.display()))?;
        }
        std::fs::remove_file(&tmp).ok();
    }
    if let Some(f) = REQUIRED.iter().find(|f| !part.join(f).is_file()) {
        return Err(format!("the downloaded wheels have no {f}"));
    }
    if out.exists() {
        std::fs::remove_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    }
    std::fs::rename(&part, &out).map_err(|e| format!("{}: {e}", out.display()))?;
    crate::service::add_dll_dir(&out)?;
    Ok(out)
}
