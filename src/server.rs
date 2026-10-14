//! `subthx serve`: keeps the models loaded and runs sync/gen jobs for other
//! programs over HTTP. One worker thread owns the [`Engine`] and takes jobs
//! from a queue in order; the HTTP side only queues jobs and reports on them.
//!
//! Also answers OpenAI's `POST /v1/audio/transcriptions`, so programs that
//! talk to the Whisper API can use it as is.

use crate::batch::Segment;
use crate::pipeline::{Config, Engine, Stage};
use crate::service::{self, ChapterMatch, Format, ModelArgs, Transcribed};
use axum::extract::{DefaultBodyLimit, Multipart, Path as UrlPath, Query, Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{oneshot, watch};

#[derive(clap::Args, Clone, Debug)]
pub struct ServeArgs {
    /// address to listen on (127.0.0.1 = this computer only)
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, default_value_t = 7865)]
    pub port: u16,
    #[command(flatten)]
    pub model: ModelArgs,
    /// load the models on the first job instead of at startup
    #[arg(long)]
    pub lazy: bool,
    /// unload the models after this many idle minutes, freeing VRAM (0 = never)
    #[arg(long, default_value_t = 0.0)]
    pub idle_unload: f64,
    /// require "Authorization: Bearer <token>" on every request
    #[arg(long)]
    pub token: Option<String>,
    /// only accept file paths under these folders (repeatable; default: any)
    #[arg(long)]
    pub allow_dir: Vec<PathBuf>,
    /// forget finished jobs after this many minutes
    #[arg(long, default_value_t = 60.0)]
    pub keep_jobs: f64,
    /// transcripts kept in memory, so re-syncing the same audio skips ASR
    #[arg(long, default_value_t = 8)]
    pub cache_size: usize,
    /// web origins allowed to call the API from a browser/webview (CORS)
    #[arg(long)]
    pub cors: Vec<String>,
}

// ---------------------------------------------------------------- jobs

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Sync,
    Gen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Queued,
    Loading,
    Decoding,
    Vad,
    Asr,
    Aligning,
    Done,
    Failed,
    Cancelled,
}

impl JobState {
    fn finished(self) -> bool {
        matches!(self, JobState::Done | JobState::Failed | JobState::Cancelled)
    }
}

fn one_or_many<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<PathBuf>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(PathBuf),
        Many(Vec<PathBuf>),
    }
    Ok(match OneOrMany::deserialize(d)? {
        OneOrMany::One(p) => vec![p],
        OneOrMany::Many(v) => v,
    })
}

#[derive(Clone, Debug, Deserialize)]
pub struct JobRequest {
    #[serde(default)]
    pub mode: Mode,
    /// audio file path(s) in reading order
    #[serde(deserialize_with = "one_or_many")]
    pub audio: Vec<PathBuf>,
    /// book: epub/txt/srt/vtt/ass path(s); sync only
    #[serde(default, deserialize_with = "one_or_many")]
    pub text: Vec<PathBuf>,
    /// also write subtitle files like the CLI does (next to the audio or in output_dir)
    #[serde(default)]
    pub write: bool,
    pub output_dir: Option<PathBuf>,
    pub lang_ext: Option<String>,
    /// format of written files: srt (default) or vtt
    pub format: Option<String>,
    #[serde(default)]
    pub overwrite: bool,
    /// regroup cues into whole sentences of the book (single text source)
    #[serde(default)]
    pub respect_grouping: bool,
    /// subtitle language (ISO 639-1); the loaded model must support it
    pub language: Option<String>,
    /// search depth for respect_grouping (default 6; 15 for subtitles with songs)
    pub respect_grouping_count: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Status {
    pub id: String,
    pub mode: Mode,
    pub state: JobState,
    /// 0..1 over the whole job
    pub progress: f64,
    /// audio file being processed (0-based) and how many there are
    pub file: usize,
    pub files: usize,
    /// jobs before this one in the queue
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queued_ahead: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// subtitle files written (with "write": true)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub written: Vec<PathBuf>,
    pub elapsed: f64,
    /// processing time without the wait in the queue (when finished)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub took: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FileResult {
    pub audio: PathBuf,
    pub segments: Vec<Segment>,
}

#[derive(Clone, Debug, Serialize)]
pub struct JobResult {
    pub files: Vec<FileResult>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub matches: Vec<ChapterMatch>,
    /// audio length in seconds per file
    pub durations: Vec<f64>,
    /// share of the matched book text that made it into the subtitles, 0..100 (sync)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub book_match: Option<f64>,
    /// where the time went, seconds
    pub stages: Stages,
    /// model and device that transcribed (cached transcripts keep theirs)
    pub model: String,
    pub device: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Stages {
    pub load: f64,
    pub decode: f64,
    pub vad: f64,
    pub asr: f64,
    pub align: f64,
    /// transcripts served from the cache
    pub cached: usize,
}

struct Job {
    id: String,
    seq: u64,
    req: JobRequest,
    format: Format,
    created: Instant,
    status: watch::Sender<Status>,
    cancel: AtomicBool,
    result: Mutex<Option<JobResult>>,
    finished: Mutex<Option<Instant>>,
    /// when the worker picked the job up
    started: Mutex<Option<Instant>>,
    /// uploaded files deleted with the job
    temp: Vec<PathBuf>,
}

impl Job {
    fn update(&self, f: impl FnOnce(&mut Status)) {
        let elapsed = self.created.elapsed().as_secs_f64();
        self.status.send_modify(|s| {
            f(s);
            s.elapsed = elapsed;
        });
    }

    fn finish(&self, res: Result<JobResult, String>) {
        let cancelled = self.cancel.load(Ordering::Relaxed);
        match res {
            Ok(r) => {
                *self.result.lock().unwrap() = Some(r);
                self.update(|s| {
                    s.state = JobState::Done;
                    s.progress = 1.0;
                });
            }
            Err(e) => self.update(|s| {
                s.state = if cancelled { JobState::Cancelled } else { JobState::Failed };
                s.error = if cancelled { None } else { Some(e) };
            }),
        }
        let now = Instant::now();
        *self.finished.lock().unwrap() = Some(now);
        let took = self.started.lock().unwrap().map(|t| (now - t).as_secs_f64());
        self.status.send_modify(|s| s.took = took);
        for p in &self.temp {
            let _ = std::fs::remove_file(p);
        }
    }

    fn state(&self) -> JobState {
        self.status.borrow().state
    }
}

// ---------------------------------------------------------------- worker

#[derive(Clone, Debug, Serialize)]
struct ModelState {
    asr: String,
    device: String,
    loaded: bool,
    load_seconds: Option<f64>,
    /// job the worker is running
    busy: Option<String>,
}

enum Msg {
    Job(Arc<Job>),
    Load(Config, oneshot::Sender<Result<f64, String>>),
    Unload(oneshot::Sender<()>),
}

type CacheKey = (PathBuf, u64, Option<SystemTime>, String, String, String);

struct Worker {
    cfg: Config,
    engine: Option<Engine>,
    cache: Vec<(CacheKey, Transcribed)>,
    cache_size: usize,
    app: Arc<App>,
}

impl Worker {
    fn set_model(&self, f: impl FnOnce(&mut ModelState)) {
        f(&mut self.app.model.lock().unwrap());
    }

    fn load(&mut self) -> Result<f64, String> {
        if let Some(e) = &self.engine {
            return Ok(e.load_seconds);
        }
        let e = Engine::load(&self.cfg)?;
        let secs = e.load_seconds;
        eprintln!("models loaded ({} on {}) in {secs:.2}s", self.cfg.asr, self.cfg.device);
        self.engine = Some(e);
        self.set_model(|m| {
            m.loaded = true;
            m.load_seconds = Some(secs);
        });
        Ok(secs)
    }

    fn unload(&mut self) {
        if self.engine.take().is_some() {
            eprintln!("models unloaded");
        }
        self.set_model(|m| m.loaded = false);
    }

    fn run(mut self, rx: mpsc::Receiver<Msg>, idle: Option<Duration>) {
        loop {
            let msg = match idle {
                Some(d) if self.engine.is_some() => match rx.recv_timeout(d) {
                    Ok(m) => m,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        self.unload();
                        continue;
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                },
                _ => match rx.recv() {
                    Ok(m) => m,
                    Err(_) => return,
                },
            };
            match msg {
                Msg::Job(job) => {
                    if job.cancel.load(Ordering::Relaxed) {
                        job.finish(Err("cancelled".into()));
                        continue;
                    }
                    self.set_model(|m| m.busy = Some(job.id.clone()));
                    *job.started.lock().unwrap() = Some(Instant::now());
                    let t = Instant::now();
                    let res = self.run_job(&job);
                    if job.req.write {
                        // SubPlz-style failure notes next to the subtitles
                        let outs: Vec<PathBuf> = job
                            .req
                            .audio
                            .iter()
                            .map(|a| service::output_path(a, job.req.output_dir.as_deref(), job.req.lang_ext.as_deref(), job.format))
                            .collect();
                        match &res {
                            Err(e) if !job.cancel.load(Ordering::Relaxed) => {
                                let _ = service::write_subfail(&job.req.audio[0], &outs[0], e);
                            }
                            _ => {
                                service::cleanup_subfail(&outs);
                            }
                        }
                    }
                    match &res {
                        Ok(_) => eprintln!("job {} done in {:.2}s", job.id, t.elapsed().as_secs_f64()),
                        Err(_) if job.cancel.load(Ordering::Relaxed) => eprintln!("job {} cancelled", job.id),
                        Err(e) => eprintln!("job {} failed: {e}", job.id),
                    }
                    job.finish(res);
                    self.set_model(|m| m.busy = None);
                }
                Msg::Load(cfg, reply) => {
                    if cfg.asr != self.cfg.asr || cfg.device != self.cfg.device || cfg.batch_size != self.cfg.batch_size {
                        self.unload();
                        self.cfg = cfg;
                        self.set_model(|m| {
                            m.asr = self.cfg.asr.to_string();
                            m.device = self.cfg.device.to_string();
                            m.load_seconds = None;
                        });
                    }
                    let _ = reply.send(self.load());
                }
                Msg::Unload(reply) => {
                    self.unload();
                    let _ = reply.send(());
                }
            }
        }
    }

    fn cache_key(&self, audio: &Path, language: &str) -> Option<CacheKey> {
        let p = std::fs::canonicalize(audio).ok()?;
        let md = std::fs::metadata(&p).ok()?;
        Some((p, md.len(), md.modified().ok(), self.cfg.asr.to_string(), self.cfg.device.to_string(), language.into()))
    }

    fn transcribe(&mut self, job: &Job, i: usize, audio: &Path, st: &mut Stages) -> Result<Transcribed, String> {
        let language = job
            .req
            .language
            .clone()
            .unwrap_or_else(|| crate::pipeline::model_info(&self.cfg.asr).languages[0].to_string());
        let key = self.cache_key(audio, &language);
        if let Some(k) = &key {
            if let Some(pos) = self.cache.iter().position(|(c, _)| c == k) {
                let hit = self.cache.remove(pos);
                let tr = hit.1.clone();
                self.cache.push(hit);
                st.cached += 1;
                return Ok(tr);
            }
        }
        if self.engine.is_none() {
            job.update(|s| s.state = JobState::Loading);
            let t = Instant::now();
            self.load()?;
            st.load += t.elapsed().as_secs_f64();
        }
        let n = job.req.audio.len() as f64;
        let mut progress = |stage: Stage, frac: f64| -> Result<(), String> {
            if job.cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            // decode ~15%, VAD ~15%, ASR ~70% of a file's time
            let (state, within) = match stage {
                Stage::Decode => (JobState::Decoding, 0.15 * frac),
                Stage::Vad => (JobState::Vad, 0.15 + 0.15 * frac),
                Stage::Asr => (JobState::Asr, 0.3 + 0.7 * frac),
            };
            job.update(|s| {
                s.state = state;
                s.file = i;
                s.progress = ((i as f64 + within) / n * 0.99).min(0.99);
            });
            Ok(())
        };
        let tr = service::transcribe(self.engine.as_mut().unwrap(), audio, &language, &mut progress)?;
        if let Some(tm) = &tr.timings {
            st.decode += tm.decode;
            st.vad += tm.vad;
            st.asr += tm.asr;
        }
        if let (Some(k), true) = (key, self.cache_size > 0) {
            if self.cache.len() >= self.cache_size {
                self.cache.remove(0);
            }
            self.cache.push((k, tr.clone()));
        }
        Ok(tr)
    }

    fn run_job(&mut self, job: &Job) -> Result<JobResult, String> {
        let req = &job.req;
        let outs: Vec<PathBuf> = req
            .audio
            .iter()
            .map(|a| service::output_path(a, req.output_dir.as_deref(), req.lang_ext.as_deref(), job.format))
            .collect();
        if req.write && !req.overwrite {
            if let Some(o) = outs.iter().find(|o| o.exists()) {
                return Err(format!("{} already exists (set \"overwrite\": true)", o.display()));
            }
        }
        let mut stages = Stages::default();
        let mut trs = Vec::new();
        for (i, audio) in req.audio.iter().enumerate() {
            trs.push(self.transcribe(job, i, audio, &mut stages)?);
        }
        let t_align = Instant::now();
        if job.cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        job.update(|s| s.state = JobState::Aligning);
        let mut match_pct = None;
        let (segs, matches) = match req.mode {
            Mode::Sync => {
                let s = service::sync(
                    &req.audio,
                    &trs,
                    &req.text,
                    service::grouping_for(&req.text, req.respect_grouping.then_some(req.respect_grouping_count.unwrap_or(6))),
                )?;
                match_pct = s.book_match;
                (s.files, s.matches)
            }
            Mode::Gen => {
                let shift = !crate::pipeline::breaks_lines(&self.cfg.asr);
                (trs.iter().map(|t| service::gen(t, shift)).collect(), vec![])
            }
        };
        stages.align = t_align.elapsed().as_secs_f64();
        let t_write = Instant::now();
        if req.write {
            for (o, s) in outs.iter().zip(&segs) {
                service::prepare_output(o, req.overwrite)?;
                service::write_subs(o, s, job.format)?;
            }
            job.update(|s| s.written = outs.clone());
        }
        let write_s = t_write.elapsed().as_secs_f64();
        eprintln!(
            "job {}: load {:.2}s, decode {:.2}s, VAD {:.2}s, ASR {:.2}s, align {:.2}s, write {:.2}s{}",
            job.id,
            stages.load,
            stages.decode,
            stages.vad,
            stages.asr,
            stages.align,
            write_s,
            if stages.cached > 0 { format!(" ({} cached)", stages.cached) } else { String::new() }
        );
        let durations = trs
            .iter()
            .map(|t| match &t.timings {
                Some(tm) => tm.audio_seconds,
                None => t.audio.chapters.iter().map(|c| c.duration).sum::<f64>(),
            })
            .collect();
        Ok(JobResult {
            files: req
                .audio
                .iter()
                .zip(segs)
                .map(|(a, s)| FileResult {
                    audio: a.clone(),
                    segments: s,
                })
                .collect(),
            matches,
            durations,
            book_match: match_pct,
            stages,
            model: self.cfg.asr.to_string(),
            device: self.cfg.device.to_string(),
        })
    }
}

// ---------------------------------------------------------------- HTTP

struct App {
    jobs: Mutex<HashMap<String, Arc<Job>>>,
    queue: Mutex<mpsc::Sender<Msg>>,
    seq: AtomicU64,
    model: Mutex<ModelState>,
    base: Config,
    token: Option<String>,
    allow: Vec<PathBuf>,
    keep: Duration,
    uploads: PathBuf,
    cors: Vec<String>,
    cuda: Mutex<Download>,
    /// model downloads by catalog id (see `crate::model_dl`)
    models_dl: Mutex<HashMap<String, Download>>,
}

/// The CUDA libraries' download (see `crate::cuda`).
#[derive(Clone, Default, Serialize)]
struct Download {
    running: bool,
    progress: Option<crate::cuda::Progress>,
    error: Option<String>,
}

type Shared = Arc<App>;

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": { "message": self.1 } }))).into_response()
    }
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

fn not_found(id: &str) -> ApiError {
    ApiError(StatusCode::NOT_FOUND, format!("no job {id}"))
}

type ApiResult<T> = Result<T, ApiError>;

async fn cors(State(app): State<Shared>, req: Request, next: Next) -> Response {
    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .filter(|o| app.cors.iter().any(|c| c == o || c == "*"))
        .map(str::to_string);
    let Some(origin) = origin else {
        return next.run(req).await;
    };
    let mut res = if req.method() == axum::http::Method::OPTIONS {
        StatusCode::NO_CONTENT.into_response()
    } else {
        next.run(req).await
    };
    let h = res.headers_mut();
    let v = |s: &str| axum::http::HeaderValue::from_str(s).unwrap();
    h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, v(&origin));
    h.insert(header::ACCESS_CONTROL_ALLOW_METHODS, v("GET, POST, DELETE, OPTIONS"));
    h.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, v("content-type, authorization"));
    h.insert(header::VARY, v("Origin"));
    res
}

#[derive(Deserialize)]
struct CoverQuery {
    /// an audio file or an epub
    path: PathBuf,
}

/// Cover art of an audio file (embedded or an image beside it), a frame of a
/// video, or an epub's cover.
async fn cover(State(app): State<Shared>, Query(q): Query<CoverQuery>) -> ApiResult<Response> {
    app.check_path(&q.path, "file")?;
    // a video's frame comes from ffmpeg: off the async workers
    let path = q.path.clone();
    let found = tokio::task::spawn_blocking(move || {
        if service::ext(&path) == "epub" {
            crate::epub::cover(&path)
        } else {
            crate::media::cover(&path).map(|c| c.map(|(b, m)| (b, m.to_string())))
        }
    })
    .await
    .map_err(|e| bad(e.to_string()))?
    .map_err(bad)?;
    let Some((bytes, mime)) = found else {
        return Err(ApiError(StatusCode::NOT_FOUND, format!("no cover in {}", q.path.display())));
    };
    Ok(([(header::CONTENT_TYPE, mime), (header::CACHE_CONTROL, "max-age=3600".to_string())], bytes).into_response())
}

#[derive(Deserialize)]
struct PlanRequest {
    /// files and folders, e.g. what was dropped on a window
    paths: Vec<PathBuf>,
    /// the subtitle language (ISO 639-1); default: the loaded model's first
    #[serde(default)]
    language: Option<String>,
}

#[derive(Serialize)]
struct PlannedJob {
    mode: Mode,
    audio: Vec<PathBuf>,
    text: Vec<PathBuf>,
    /// the texts are subtitles: they keep their lines and get new times
    retime: bool,
    /// the subtitles are the video's own embedded track (the text is the video)
    embedded: bool,
}

/// Turn dropped files/folders into jobs, the mode from what is there: a book
/// -> sync; subtitles in the job's language (the user's, not our earlier
/// outputs), dropped or next to a video, or a video's embedded text track
/// -> retime; nothing -> gen. Folders pair like `sync -d`, loose audio
/// files go with loose texts.
async fn plan(State(app): State<Shared>, Json(req): Json<PlanRequest>) -> ApiResult<Json<serde_json::Value>> {
    let lang = match req.language.clone() {
        Some(l) => l,
        None => {
            let asr: crate::pipeline::AsrModel = app.model.lock().unwrap().asr.parse().map_err(bad)?;
            crate::pipeline::model_info(&asr).languages[0].to_string()
        }
    };
    let paths = req.paths.clone();
    for p in &paths {
        app.check_allowed(p)?;
    }
    // probing videos for subtitle tracks reads files: off the async runtime
    let (jobs, warnings) = tokio::task::spawn_blocking(move || plan_paths(&paths, &lang)).await.map_err(|e| bad(e.to_string()))?.map_err(bad)?;
    let out: Vec<PlannedJob> = jobs
        .into_iter()
        .map(|j| PlannedJob {
            mode: if j.texts.is_empty() { Mode::Gen } else { Mode::Sync },
            retime: service::is_retime(&j.texts),
            embedded: j.texts.iter().any(|t| service::is_audio(t)),
            audio: j.audios,
            text: j.texts,
        })
        .collect();
    Ok(Json(serde_json::json!({ "jobs": out, "warnings": warnings })))
}

fn plan_paths(paths: &[PathBuf], lang: &str) -> Result<(Vec<service::Job>, Vec<String>), String> {
    let choose = service::Choose::new(lang);
    let mut jobs = Vec::new();
    let mut warnings = Vec::new();
    let (mut audios, mut texts, mut other) = (vec![], vec![], vec![]);
    for p in paths {
        if p.is_dir() {
            let mut warn = |w: String| warnings.push(w);
            let mut js = service::dir_jobs(p, true, &choose, &mut warn)?;
            if js.is_empty() {
                js = service::dir_jobs(p, false, &choose, &mut |_| {})?;
                if !js.is_empty() {
                    warnings.pop();
                }
            }
            jobs.extend(js);
        } else if service::is_audio(p) {
            audios.push(p.clone());
        } else if crate::textsrc::is_text(p) {
            texts.push(p.clone());
        } else {
            other.push(p.display().to_string());
        }
    }
    if !other.is_empty() {
        warnings.push(format!("unsupported files skipped: {}", other.join(", ")));
    }
    audios.sort_by_key(|p| service::natural_key(p));
    // dropped subtitles that are not in the job's language or are our own
    // earlier output: generate instead
    texts.retain(|t: &PathBuf| {
        let keep = !crate::textsrc::is_subtitle(t) || choose.usable(t);
        if !keep {
            let name = t.file_name().unwrap_or_default().to_string_lossy();
            warnings.push(if choose.own.contains(t) {
                format!("{name}: made by subthx earlier, not taken as subtitles to retime")
            } else {
                format!("{name}: not in {lang:?}, not taken as subtitles to retime")
            });
        }
        keep
    });
    if !audios.is_empty() {
        if texts.is_empty() {
            jobs.extend(service::pair_subtitles(audios, vec![], Some(lang), &mut |w| warnings.push(w)));
        } else if texts.iter().all(|t| crate::textsrc::is_subtitle(t)) {
            jobs.extend(service::pair_subtitles(audios, texts, Some(lang), &mut |w| warnings.push(w)));
        } else {
            let mut loose = audios.clone();
            loose.extend(texts.iter().cloned());
            jobs.extend(service::file_jobs(&loose, true)?);
        }
    } else if !texts.is_empty() {
        warnings.push("a book without audio: add the audiobook too".into());
    }
    Ok((jobs, warnings))
}

/// API reference page, built from /openapi.json (no external assets).
async fn docs() -> Response {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], include_str!("server/docs.html")).into_response()
}

async fn openapi() -> Response {
    ([(header::CONTENT_TYPE, "application/json")], include_str!("server/openapi.json")).into_response()
}

async fn auth(State(app): State<Shared>, req: Request, next: Next) -> Response {
    // the reference is public: it explains how to authenticate
    let public = matches!(req.uri().path(), "/docs" | "/openapi.json");
    if let (Some(tok), false) = (&app.token, public) {
        let ok = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|t| t == tok);
        if !ok {
            return ApiError(StatusCode::UNAUTHORIZED, "missing or wrong bearer token".into()).into_response();
        }
    }
    next.run(req).await
}

impl App {
    fn check_path(&self, p: &Path, what: &str) -> ApiResult<()> {
        if !p.is_file() {
            return Err(bad(format!("{what} {} not found", p.display())));
        }
        self.check_allowed(p)
    }

    fn check_allowed(&self, p: &Path) -> ApiResult<()> {
        if self.allow.is_empty() {
            return Ok(());
        }
        let c = std::fs::canonicalize(p).map_err(|e| bad(format!("{}: {e}", p.display())))?;
        if self.allow.iter().any(|a| c.starts_with(a)) {
            Ok(())
        } else {
            Err(ApiError(StatusCode::FORBIDDEN, format!("{} is outside the allowed folders", p.display())))
        }
    }

    /// Drop finished jobs older than `keep`.
    fn sweep(&self) {
        let mut jobs = self.jobs.lock().unwrap();
        jobs.retain(|_, j| j.finished.lock().unwrap().map_or(true, |t| t.elapsed() < self.keep));
    }

    fn get(&self, id: &str) -> ApiResult<Arc<Job>> {
        self.jobs.lock().unwrap().get(id).cloned().ok_or_else(|| not_found(id))
    }

    fn status(&self, job: &Job) -> Status {
        let mut s = job.status.borrow().clone();
        if s.state == JobState::Queued {
            let jobs = self.jobs.lock().unwrap();
            s.queued_ahead = Some(jobs.values().filter(|j| j.seq < job.seq && !j.state().finished()).count());
        }
        s.elapsed = match *job.finished.lock().unwrap() {
            Some(t) => (t - job.created).as_secs_f64(),
            None => job.created.elapsed().as_secs_f64(),
        };
        s
    }

    fn send(&self, msg: Msg) -> ApiResult<()> {
        self.queue
            .lock()
            .unwrap()
            .send(msg)
            .map_err(|_| ApiError(StatusCode::INTERNAL_SERVER_ERROR, "worker stopped".into()))
    }

    fn submit(&self, req: JobRequest, temp: Vec<PathBuf>) -> ApiResult<Arc<Job>> {
        self.sweep();
        let format: Format = req.format.as_deref().unwrap_or("srt").parse().map_err(bad)?;
        let id = uuid::Uuid::new_v4().simple().to_string();
        let status = Status {
            id: id.clone(),
            mode: req.mode,
            state: JobState::Queued,
            progress: 0.0,
            file: 0,
            files: req.audio.len(),
            queued_ahead: None,
            error: None,
            written: vec![],
            elapsed: 0.0,
            took: None,
        };
        let job = Arc::new(Job {
            id: id.clone(),
            seq: self.seq.fetch_add(1, Ordering::Relaxed),
            req,
            format,
            created: Instant::now(),
            status: watch::channel(status).0,
            cancel: AtomicBool::new(false),
            result: Mutex::new(None),
            finished: Mutex::new(None),
            started: Mutex::new(None),
            temp,
        });
        self.jobs.lock().unwrap().insert(id, job.clone());
        self.send(Msg::Job(job.clone()))?;
        Ok(job)
    }
}

async fn wait_done(job: &Job) {
    let mut rx = job.status.subscribe();
    let _ = rx.wait_for(|s| s.state.finished()).await;
}

#[derive(Deserialize)]
struct SubmitQuery {
    /// hold the request until the job finishes
    #[serde(default)]
    wait: bool,
    /// with wait: answer with the subtitles themselves (srt, vtt or json)
    format: Option<String>,
    #[serde(default)]
    index: usize,
}

async fn create_job(State(app): State<Shared>, Query(q): Query<SubmitQuery>, Json(req): Json<JobRequest>) -> ApiResult<Response> {
    if req.audio.is_empty() {
        return Err(bad("\"audio\" is empty"));
    }
    for a in &req.audio {
        app.check_path(a, "audio")?;
        if !service::is_audio(a) {
            return Err(bad(format!("{}: not an audio file ({})", a.display(), service::AUDIO_EXTS.join(", "))));
        }
    }
    match req.mode {
        Mode::Sync if req.text.is_empty() => return Err(bad("sync needs \"text\": the book (epub/txt/srt/vtt/ass)")),
        Mode::Gen if !req.text.is_empty() => return Err(bad("gen takes no \"text\"")),
        _ => {}
    }
    for t in &req.text {
        app.check_path(t, "text")?;
        // a video: its embedded subtitles
        if !crate::textsrc::is_text(t) && !service::is_audio(t) {
            return Err(bad(format!("{}: not a text file", t.display())));
        }
    }
    if req.write {
        if let Some(d) = &req.output_dir {
            std::fs::create_dir_all(d).map_err(|e| bad(format!("{}: {e}", d.display())))?;
            app.check_allowed(d)?;
        }
    }
    if let Some(lang) = &req.language {
        let asr: crate::pipeline::AsrModel = app.model.lock().unwrap().asr.parse().map_err(bad)?;
        let info = crate::pipeline::model_info(&asr);
        if !info.languages.contains(&lang.as_str()) {
            return Err(bad(format!("{} does not transcribe language {lang:?}; pick a model for it", info.name)));
        }
    }
    let job = app.submit(req, vec![])?;
    if !q.wait {
        return Ok((StatusCode::ACCEPTED, Json(app.status(&job))).into_response());
    }
    wait_done(&job).await;
    match q.format {
        Some(f) => render_result(&app, &job, &f, q.index),
        None => Ok(Json(app.status(&job)).into_response()),
    }
}

async fn list_jobs(State(app): State<Shared>) -> Json<Vec<Status>> {
    app.sweep();
    let jobs: Vec<Arc<Job>> = app.jobs.lock().unwrap().values().cloned().collect();
    let mut v: Vec<(u64, Status)> = jobs.iter().map(|j| (j.seq, app.status(j))).collect();
    v.sort_by_key(|(s, _)| *s);
    Json(v.into_iter().map(|(_, s)| s).collect())
}

async fn job_status(State(app): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult<Json<Status>> {
    let job = app.get(&id)?;
    Ok(Json(app.status(&job)))
}

async fn job_events(
    State(app): State<Shared>,
    UrlPath(id): UrlPath<String>,
) -> ApiResult<Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>>> {
    let job = app.get(&id)?;
    let mut rx = job.status.subscribe();
    // every status change, closing after the final one
    let (tx, out) = tokio::sync::mpsc::channel(16);
    tokio::spawn(async move {
        loop {
            let s = rx.borrow_and_update().clone();
            let done = s.state.finished();
            let ev = Event::default().json_data(&s).unwrap_or_default();
            if tx.send(Ok(ev)).await.is_err() || done || rx.changed().await.is_err() {
                return;
            }
        }
    });
    let stream = tokio_stream::wrappers::ReceiverStream::new(out);
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

#[derive(Deserialize)]
struct ResultQuery {
    format: Option<String>,
    #[serde(default)]
    index: usize,
}

fn render_result(app: &App, job: &Job, format: &str, index: usize) -> ApiResult<Response> {
    let st = app.status(job);
    if !st.state.finished() {
        return Err(ApiError(StatusCode::CONFLICT, format!("job is {:?}", st.state).to_lowercase()));
    }
    let res = job.result.lock().unwrap();
    let Some(res) = res.as_ref() else {
        return Err(ApiError(
            StatusCode::CONFLICT,
            st.error.unwrap_or_else(|| "job was cancelled".into()),
        ));
    };
    if format == "json" {
        return Ok(Json(res).into_response());
    }
    let fmt: Format = format.parse().map_err(bad)?;
    let f = res
        .files
        .get(index)
        .ok_or_else(|| bad(format!("index {index} out of range (0..{})", res.files.len())))?;
    Ok(([(header::CONTENT_TYPE, fmt.mime())], fmt.render(&f.segments)).into_response())
}

async fn job_result(State(app): State<Shared>, UrlPath(id): UrlPath<String>, Query(q): Query<ResultQuery>) -> ApiResult<Response> {
    let job = app.get(&id)?;
    render_result(&app, &job, q.format.as_deref().unwrap_or("srt"), q.index)
}

async fn job_delete(State(app): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult<Json<Status>> {
    let job = app.get(&id)?;
    if job.state().finished() {
        app.jobs.lock().unwrap().remove(&id);
    } else {
        job.cancel.store(true, Ordering::Relaxed);
        if job.state() == JobState::Queued {
            job.update(|s| s.state = JobState::Cancelled);
        }
    }
    Ok(Json(app.status(&job)))
}

/// Where the CUDA libraries stand: in `<models>/cuda`, elsewhere on PATH,
/// or missing; and the download, while there is one.
async fn cuda_status(State(app): State<Shared>) -> Json<serde_json::Value> {
    let models = &app.base.models;
    let missing = crate::cuda::missing(models);
    Json(serde_json::json!({
        "dir": crate::cuda::dir(models),
        "installed": crate::cuda::installed(models),
        "missing": missing,
        "available": missing.is_empty(),
        "download": app.cuda.lock().unwrap().clone(),
    }))
}

/// Start downloading the CUDA libraries into `<models>/cuda` (about 0.9 GB;
/// GET /v1/cuda follows it). A loaded model keeps its device; the next load
/// finds the libraries.
async fn cuda_download(State(app): State<Shared>) -> ApiResult<Response> {
    {
        let mut c = app.cuda.lock().unwrap();
        if c.running {
            return Ok((StatusCode::ACCEPTED, Json(c.clone())).into_response());
        }
        *c = Download { running: true, ..Download::default() };
    }
    let a = app.clone();
    std::thread::spawn(move || {
        let r = crate::cuda::download(&a.base.models, &mut |p| {
            a.cuda.lock().unwrap().progress = Some(p.clone());
            Ok(())
        });
        let mut c = a.cuda.lock().unwrap();
        c.running = false;
        c.error = r.err().map(|e| failure_code("cuda download", e));
    });
    let c = app.cuda.lock().unwrap().clone();
    Ok((StatusCode::ACCEPTED, Json(c)).into_response())
}

async fn health(State(app): State<Shared>) -> Json<serde_json::Value> {
    let model = app.model.lock().unwrap().clone();
    let jobs = app.jobs.lock().unwrap();
    let pending = jobs.values().filter(|j| !j.state().finished()).count();
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "model": model,
        "pending_jobs": pending,
    }))
}

/// The short code a download failure shows as (the full message goes to the
/// log): `network`, `checksum`, or `download_failed`.
fn failure_code(what: &str, e: String) -> String {
    eprintln!("subthx: {what}: {e}");
    let code = if e.contains("checksum") {
        "checksum"
    } else if e.starts_with("https://") {
        "network"
    } else {
        "download_failed"
    };
    code.to_string()
}

/// Start downloading a model's missing files from the subthx repos
/// (`crate::model_dl`); GET /v1/models follows it.
async fn model_download(State(app): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult<Response> {
    if crate::model_dl::source(&id).is_none() {
        return Err(bad(format!("{id}: no download for this model")));
    }
    {
        let mut m = app.models_dl.lock().unwrap();
        let c = m.entry(id.clone()).or_default();
        if c.running {
            return Ok((StatusCode::ACCEPTED, Json(c.clone())).into_response());
        }
        *c = Download { running: true, ..Download::default() };
    }
    let (a, key) = (app.clone(), id.clone());
    std::thread::spawn(move || {
        let r = crate::model_dl::download(&a.base.models, &key, &mut |p| {
            if let Some(c) = a.models_dl.lock().unwrap().get_mut(&key) {
                c.progress = Some(p.clone());
            }
            Ok(())
        });
        if let Some(c) = a.models_dl.lock().unwrap().get_mut(&key) {
            c.running = false;
            c.error = r.err().map(|e| failure_code("model download", e));
        }
    });
    let c = app.models_dl.lock().unwrap().get(&id).cloned().unwrap_or_default();
    Ok((StatusCode::ACCEPTED, Json(c)).into_response())
}

async fn models(State(app): State<Shared>) -> Json<serde_json::Value> {
    let model = app.model.lock().unwrap().clone();
    let device: crate::vad::Device = model.device.parse().unwrap_or(crate::vad::Device::Cpu);
    let mut models = vec![];
    let mut languages: Vec<&str> = vec![];
    for m in crate::pipeline::CATALOG {
        let asr: crate::pipeline::AsrModel = m.id.parse().unwrap();
        // bytes on disk of what this model loads on the current device
        let size = crate::pipeline::model_size(&app.base.models, &asr, device);
        let download = app.models_dl.lock().unwrap().get(m.id).cloned();
        languages.extend(m.languages);
        models.push(serde_json::json!({
            "id": m.id,
            "name": m.name,
            "family": m.family,
            "languages": m.languages,
            "installed": size.is_some(),
            "size": size,
            "downloadable": crate::model_dl::source(m.id).is_some(),
            "download": download,
        }));
    }
    languages.sort();
    languages.dedup();
    Json(serde_json::json!({
        "current": model,
        "models": models,
        "languages": languages,
        "devices": ["cuda", "directml", "cpu"],
    }))
}

#[derive(Deserialize)]
struct LoadRequest {
    asr: Option<String>,
    device: Option<String>,
    batch_size: Option<usize>,
}

async fn models_load(State(app): State<Shared>, body: Option<Json<LoadRequest>>) -> ApiResult<Json<serde_json::Value>> {
    let mut cfg = app.base.clone();
    {
        let m = app.model.lock().unwrap();
        cfg.asr = m.asr.parse().map_err(bad)?;
        cfg.device = m.device.parse().map_err(bad)?;
    }
    if let Some(Json(r)) = body {
        if let Some(a) = r.asr {
            cfg.asr = a.parse().map_err(bad)?;
        }
        if let Some(d) = r.device {
            cfg.device = d.parse().map_err(bad)?;
        }
        if r.batch_size.is_some() {
            cfg.batch_size = r.batch_size;
        }
    }
    let on_cuda = cfg.device == crate::vad::Device::Cuda;
    let (tx, rx) = oneshot::channel();
    app.send(Msg::Load(cfg, tx))?;
    // queued behind running jobs
    let secs = rx
        .await
        .map_err(|_| ApiError(StatusCode::INTERNAL_SERVER_ERROR, "worker_stopped".into()))?
        .map_err(|e| {
            // the message is for the log; the client gets a code it can word
            eprintln!("subthx: model load: {e}");
            let libs = on_cuda && !crate::cuda::missing(&app.base.models).is_empty();
            let code = if e == crate::pipeline::MODEL_MISSING {
                "model_missing"
            } else if libs {
                "cuda_missing"
            } else {
                "load_failed"
            };
            ApiError(StatusCode::INTERNAL_SERVER_ERROR, code.into())
        })?;
    let model = app.model.lock().unwrap().clone();
    Ok(Json(serde_json::json!({ "model": model, "load_seconds": secs })))
}

async fn models_unload(State(app): State<Shared>) -> ApiResult<Json<serde_json::Value>> {
    let (tx, rx) = oneshot::channel();
    app.send(Msg::Unload(tx))?;
    let _ = rx.await;
    let model = app.model.lock().unwrap().clone();
    Ok(Json(serde_json::json!({ "model": model })))
}

/// OpenAI `POST /v1/audio/transcriptions` (multipart: file, model,
/// response_format = json | text | srt | vtt | verbose_json). Runs `gen`.
async fn openai_transcribe(State(app): State<Shared>, mut mp: Multipart) -> ApiResult<Response> {
    use tokio::io::AsyncWriteExt;
    let mut file: Option<PathBuf> = None;
    let mut response_format = "json".to_string();
    let mut cleanup = TempFile(None);
    while let Some(mut field) = mp.next_field().await.map_err(|e| bad(e.to_string()))? {
        match field.name().unwrap_or_default() {
            "file" => {
                let name = field.file_name().unwrap_or("audio").to_string();
                let ext = service::ext(Path::new(&name));
                if !service::AUDIO_EXTS.contains(&ext.as_str()) {
                    return Err(bad(format!("{name}: unsupported audio type ({})", service::AUDIO_EXTS.join(", "))));
                }
                let path = app.uploads.join(format!("{}.{ext}", uuid::Uuid::new_v4().simple()));
                cleanup.0 = Some(path.clone());
                let mut out = tokio::fs::File::create(&path)
                    .await
                    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                while let Some(chunk) = field.chunk().await.map_err(|e| bad(e.to_string()))? {
                    out.write_all(&chunk)
                        .await
                        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                }
                out.flush().await.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                file = Some(path);
            }
            "response_format" => response_format = field.text().await.map_err(|e| bad(e.to_string()))?,
            // model, language, prompt, temperature, timestamp_granularities: the
            // loaded model is used, Japanese only
            _ => {}
        }
    }
    let Some(file) = file else {
        return Err(bad("missing \"file\""));
    };
    let req = JobRequest {
        mode: Mode::Gen,
        audio: vec![file.clone()],
        text: vec![],
        write: false,
        output_dir: None,
        lang_ext: None,
        format: None,
        overwrite: false,
        respect_grouping: false,
        respect_grouping_count: None,
        language: None,
    };
    // the job deletes the upload when it finishes
    cleanup.0 = None;
    let job = app.submit(req, vec![file])?;
    wait_done(&job).await;
    let res = job.result.lock().unwrap().clone();
    let Some(res) = res else {
        let st = app.status(&job);
        return Err(ApiError(StatusCode::INTERNAL_SERVER_ERROR, st.error.unwrap_or_else(|| "cancelled".into())));
    };
    let segs = &res.files[0].segments;
    let text: String = segs.iter().map(|s| s.text.as_str()).collect();
    Ok(match response_format.as_str() {
        "text" => ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response(),
        "srt" | "vtt" => {
            let f: Format = response_format.parse().map_err(bad)?;
            ([(header::CONTENT_TYPE, f.mime())], f.render(segs)).into_response()
        }
        "verbose_json" => Json(serde_json::json!({
            "task": "transcribe",
            "language": "japanese",
            "duration": res.durations[0],
            "text": text,
            "segments": segs.iter().enumerate().map(|(i, s)| serde_json::json!({
                "id": i, "start": s.start, "end": s.end, "text": s.text,
            })).collect::<Vec<_>>(),
        }))
        .into_response(),
        "json" => Json(serde_json::json!({ "text": text })).into_response(),
        f => return Err(bad(format!("unsupported response_format {f:?} (json, text, srt, vtt, verbose_json)"))),
    })
}

/// Deletes an upload when the request fails before it becomes a job.
struct TempFile(Option<PathBuf>);

impl Drop for TempFile {
    fn drop(&mut self) {
        if let Some(p) = &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn router(app: Shared) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/models/load", post(models_load))
        .route("/v1/models/unload", post(models_unload))
        .route("/v1/models/{id}/download", post(model_download))
        .route("/v1/cuda", get(cuda_status))
        .route("/v1/cuda/download", post(cuda_download))
        .route("/v1/jobs", post(create_job).get(list_jobs))
        .route("/v1/jobs/{id}", get(job_status).delete(job_delete))
        .route("/v1/jobs/{id}/events", get(job_events))
        .route("/v1/jobs/{id}/result", get(job_result))
        .route(
            "/v1/audio/transcriptions",
            post(openai_transcribe).layer(DefaultBodyLimit::max(4 << 30)),
        )
        .route("/v1/plan", post(plan))
        .route("/v1/cover", get(cover))
        .route("/docs", get(docs))
        .route("/openapi.json", get(openapi))
        .layer(middleware::from_fn_with_state(app.clone(), auth))
        // outermost: preflight requests carry no token
        .layer(middleware::from_fn_with_state(app.clone(), cors))
        .with_state(app)
}

/// Start the worker; returns the shared state for [`router`].
fn start(a: &ServeArgs) -> Result<Shared, String> {
    if let Some(dir) = &a.model.cuda_dir {
        service::add_dll_dir(dir)?;
    }
    crate::cuda::use_downloaded(&a.model.models)?;
    let cfg = a.model.config();
    let allow = a
        .allow_dir
        .iter()
        .map(|d| std::fs::canonicalize(d).map_err(|e| format!("{}: {e}", d.display())))
        .collect::<Result<Vec<_>, _>>()?;
    let uploads = std::env::temp_dir().join("subthx-uploads");
    std::fs::create_dir_all(&uploads).map_err(|e| format!("{}: {e}", uploads.display()))?;
    let (tx, rx) = mpsc::channel();
    let app = Arc::new(App {
        jobs: Mutex::new(HashMap::new()),
        queue: Mutex::new(tx),
        seq: AtomicU64::new(0),
        model: Mutex::new(ModelState {
            asr: cfg.asr.to_string(),
            device: cfg.device.to_string(),
            loaded: false,
            load_seconds: None,
            busy: None,
        }),
        base: cfg.clone(),
        token: a.token.clone(),
        allow,
        keep: Duration::from_secs_f64(a.keep_jobs.max(0.0) * 60.0),
        uploads,
        cors: a.cors.clone(),
        cuda: Mutex::new(Download::default()),
        models_dl: Mutex::new(HashMap::new()),
    });
    let mut worker = Worker {
        cfg,
        engine: None,
        cache: Vec::new(),
        cache_size: a.cache_size,
        app: app.clone(),
    };
    if !a.lazy {
        worker.load()?;
    }
    let idle = (a.idle_unload > 0.0).then(|| Duration::from_secs_f64(a.idle_unload * 60.0));
    std::thread::Builder::new()
        .name("subthx-worker".into())
        .spawn(move || worker.run(rx, idle))
        .map_err(|e| e.to_string())?;
    Ok(app)
}

pub fn serve(a: ServeArgs) -> Result<(), String> {
    let app = start(&a)?;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(async move {
        let addr = format!("{}:{}", a.host, a.port);
        let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| format!("{addr}: {e}"))?;
        eprintln!("listening on http://{addr}");
        axum::serve(listener, router(app))
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
            .map_err(|e| e.to_string())
    })
}

/// Run a server in the background on its own thread; when `host:port` is
/// taken, a free port is used. Returns the address it listens on.
pub fn spawn(a: ServeArgs) -> Result<std::net::SocketAddr, String> {
    let app = start(&a)?;
    let listener = std::net::TcpListener::bind((a.host.as_str(), a.port))
        .or_else(|_| std::net::TcpListener::bind((a.host.as_str(), 0)))
        .map_err(|e| e.to_string())?;
    serve_on(app, listener)
}

/// Run a server on `127.0.0.1:<random port>` in the background (tests).
pub fn spawn_for_tests(a: ServeArgs) -> Result<std::net::SocketAddr, String> {
    let app = start(&a)?;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    serve_on(app, listener)
}

fn serve_on(app: Shared, listener: std::net::TcpListener) -> Result<std::net::SocketAddr, String> {
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async move {
            let l = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(l, router(app)).await.unwrap();
        });
    });
    Ok(addr)
}
