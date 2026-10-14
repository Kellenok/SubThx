//! subthx desktop app: runs the subthx server in-process (models stay loaded,
//! other programs can use the same API) and shows a window that talks to it.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use subthx::server::ServeArgs;
use subthx::service::ModelArgs;
use tauri::{Manager, State};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    asr: String,
    device: String,
    format: String,
    lang_ext: String,
    overwrite: bool,
    respect_grouping: bool,
    /// None = next to each audio file
    output_dir: Option<PathBuf>,
    models_dir: PathBuf,
    /// folder with the CUDA/cuDNN DLLs (None = the system search path)
    cuda_dir: Option<PathBuf>,
    port: u16,
    /// minutes; 0 = keep the models loaded
    idle_unload: f64,
    /// interface language: auto, ru, en, ja
    lang: String,
    /// subtitle language (ISO 639-1): picks which models fit
    language: String,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            asr: "kitsune-p0.05b".into(),
            device: default_device().into(),
            format: "srt".into(),
            lang_ext: "ja".into(),
            overwrite: true,
            respect_grouping: false,
            output_dir: None,
            models_dir: find_models(),
            cuda_dir: None,
            port: 7865,
            idle_unload: 0.0,
            lang: "auto".into(),
            language: "ja".into(),
        }
    }
}

/// CUDA when its libraries are there (downloaded or on the PATH), else the CPU:
/// a new install then works without the CUDA download
fn default_device() -> &'static str {
    if subthx::cuda::missing(&find_models()).is_empty() {
        "cuda"
    } else {
        "cpu"
    }
}

/// `models` next to the exe, or up the tree (a dev build in target/…)
fn find_models() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    for dir in exe.ancestors().skip(1).take(5) {
        let m = dir.join("models");
        if m.join("firered_vad.onnx").exists() {
            return m;
        }
    }
    exe.parent().map(|d| d.join("models")).unwrap_or_else(|| PathBuf::from("models"))
}

struct AppState {
    settings: Mutex<Settings>,
    /// what the running server was started with (paths and port need a restart)
    started: Settings,
    path: PathBuf,
    api: Result<String, String>,
}

fn load_settings(path: &Path) -> Settings {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn start_server(s: &Settings) -> Result<String, String> {
    let args = ServeArgs {
        host: "127.0.0.1".into(),
        port: s.port,
        model: ModelArgs {
            asr: s.asr.parse()?,
            device: s.device.parse()?,
            models: s.models_dir.clone(),
            batch_size: None,
            cuda_dir: s.cuda_dir.clone(),
        },
        lazy: true,
        idle_unload: s.idle_unload,
        token: None,
        allow_dir: vec![],
        keep_jobs: 24.0 * 60.0,
        cache_size: 8,
        cors: vec![
            "http://tauri.localhost".into(),
            "https://tauri.localhost".into(),
            "tauri://localhost".into(),
            "http://127.0.0.1:1420".into(),
        ],
    };
    let addr = subthx::server::spawn(args)?;
    Ok(format!("http://{addr}"))
}

#[derive(Serialize)]
struct Snapshot {
    settings: Settings,
    api: Option<String>,
    api_error: Option<String>,
    restart_needed: bool,
    default_models: PathBuf,
}

fn snapshot(st: &AppState) -> Snapshot {
    let s = st.settings.lock().unwrap().clone();
    let restart_needed = s.models_dir != st.started.models_dir || s.cuda_dir != st.started.cuda_dir || s.port != st.started.port;
    Snapshot {
        settings: s,
        api: st.api.clone().ok(),
        api_error: st.api.clone().err(),
        restart_needed,
        default_models: find_models(),
    }
}

#[tauri::command]
fn get_state(st: State<AppState>) -> Snapshot {
    snapshot(&st)
}

#[tauri::command]
fn save_settings(st: State<AppState>, settings: Settings) -> Result<Snapshot, String> {
    if let Some(d) = st.path.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(&settings).map_err(|e| e.to_string())?;
    std::fs::write(&st.path, json).map_err(|e| e.to_string())?;
    *st.settings.lock().unwrap() = settings;
    Ok(snapshot(&st))
}

#[tauri::command]
fn reveal(path: PathBuf) -> Result<(), String> {
    tauri_plugin_opener::reveal_item_in_dir(path).map_err(|e| e.to_string())
}

#[tauri::command]
fn open_path(path: PathBuf) -> Result<(), String> {
    tauri_plugin_opener::open_path(path, None::<&str>).map_err(|e| e.to_string())
}

#[tauri::command]
fn restart(app: tauri::AppHandle) {
    app.restart();
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let path = app.path().app_config_dir()?.join("settings.json");
            let settings = load_settings(&path);
            let api = start_server(&settings);
            if let Err(e) = &api {
                eprintln!("server: {e}");
            }
            app.manage(AppState {
                started: settings.clone(),
                settings: Mutex::new(settings),
                path,
                api,
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_state, save_settings, reveal, open_path, restart])
        .run(tauri::generate_context!())
        .expect("error while running subthx");
}
