use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Instant;
use subthx::pipeline::Engine;
use subthx::service::{self, Format, Job, ModelArgs, Transcribed};

#[derive(Parser)]
#[command(version, about = "Sync audiobooks with their book text into subtitles")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Transcribe audiobook(s), align against the book (epub/txt/subtitles), write subtitles
    Sync(Args),
    /// Transcribe audio into subtitles without a book
    Gen(Args),
    /// Keep the models loaded and serve sync/gen jobs over HTTP
    Serve(subthx::server::ServeArgs),
}

#[derive(clap::Args)]
struct Args {
    /// audio or video files (m4b, mp3, flac, opus, mp4, mkv…) and, for sync, book text files (epub/txt/srt/vtt/ass)
    files: Vec<PathBuf>,
    /// process folders: each folder's audio + epub/txt are paired like SubPlz -d
    #[arg(short = 'd', long = "dir")]
    dirs: Vec<PathBuf>,
    /// output file (only with a single audio file)
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// output folder (default: next to each audio file)
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// srt or vtt (default: from -o's extension, else srt)
    #[arg(long)]
    format: Option<Format>,
    /// language extension in output names: <audio>.<lang-ext>.srt
    #[arg(long)]
    lang_ext: Option<String>,
    /// replace existing subtitles (the previous file is kept once as <name>.old.<ext>)
    #[arg(long)]
    overwrite: bool,
    #[command(flatten)]
    model: ModelArgs,
    /// subtitle language (ISO 639-1); default: the model's first language
    #[arg(long)]
    language: Option<String>,
    /// use SubPlz's cached transcripts (<audio>.<chapter>.<model>.subs) instead of running ASR
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// model name in SubPlz cache file names (with --cache-dir)
    #[arg(long, default_value = "parakeet")]
    cache_model: String,
    /// regroup cues into whole sentences of the book (single text source)
    #[arg(long)]
    respect_grouping: bool,
    /// how deep --respect-grouping searches: 6 suits audiobooks, 15 subtitles with songs
    #[arg(long, default_value_t = 6)]
    respect_grouping_count: usize,
    /// also write the per-chapter transcripts as JSON (<output>.transcripts.json)
    #[arg(long)]
    transcripts: bool,
}

fn main() {
    let cli = Cli::parse();
    let res = match cli.cmd {
        Cmd::Sync(a) => run(a, true),
        Cmd::Gen(a) => run(a, false),
        Cmd::Serve(a) => subthx::server::serve(a),
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn jobs(a: &Args, sync: bool) -> Result<Vec<Job>, String> {
    let mut jobs = Vec::new();
    if !a.files.is_empty() {
        jobs.extend(service::file_jobs(&a.files, sync)?);
    }
    for dir in &a.dirs {
        let lang = a.language.clone().unwrap_or_else(|| subthx::pipeline::model_info(&a.model.asr).languages[0].to_string());
        jobs.extend(service::dir_jobs(dir, sync, &service::Choose::new(&lang), &mut |w| eprintln!("{w}"))?);
    }
    if jobs.is_empty() {
        return Err("nothing to do: give files or -d folders".into());
    }
    Ok(jobs)
}

fn format_of(a: &Args) -> Format {
    a.format
        .or_else(|| a.output.as_ref().and_then(|o| service::ext(o).parse().ok()))
        .unwrap_or(Format::Srt)
}

fn output_path(a: &Args, audio: &Path, fmt: Format) -> PathBuf {
    match &a.output {
        Some(o) => o.clone(),
        None => service::output_path(audio, a.output_dir.as_deref(), a.lang_ext.as_deref(), fmt),
    }
}

/// false = skip this output
fn prepare_output(path: &Path, overwrite: bool) -> Result<bool, String> {
    let existed = path.exists();
    let ok = service::prepare_output(path, overwrite)?;
    if !ok {
        eprintln!("{} already exists, skipping (use --overwrite)", path.display());
    } else if existed {
        eprintln!("previous file kept as {}", path.with_extension(format!("old.{}", service::ext(path))).display());
    }
    Ok(ok)
}

fn transcribe_audio(a: &Args, engine: &mut Option<Engine>, audio: &Path) -> Result<Transcribed, String> {
    if let Some(cache) = &a.cache_dir {
        return service::from_subplz_cache(cache, audio, &a.cache_model);
    }
    if engine.is_none() {
        let e = Engine::load(&a.model.config())?;
        eprintln!("models loaded in {:.2}s", e.load_seconds);
        *engine = Some(e);
    }
    let language = a
        .language
        .clone()
        .unwrap_or_else(|| subthx::pipeline::model_info(&a.model.asr).languages[0].to_string());
    let tr = service::transcribe(engine.as_mut().unwrap(), audio, &language, &mut |_, _| Ok(()))?;
    if let Some(tm) = &tr.timings {
        eprintln!(
            "  {:.2} h audio: decode {:.2}s, VAD {:.2}s ({} chunks), ASR {:.2}s ({:.0}x)",
            tm.audio_seconds / 3600.0,
            tm.decode,
            tm.vad,
            tm.chunks,
            tm.asr,
            tm.audio_seconds / tm.asr.max(1e-9)
        );
    }
    Ok(tr)
}

fn dump_transcripts(a: &Args, out: &Path, tr: &Transcribed) -> Result<(), String> {
    if !a.transcripts {
        return Ok(());
    }
    let v: Vec<_> = tr.audio.chapters.iter().map(|c| c.transcript.as_ref().map(|t| &t.segments)).collect();
    let p = out.with_extension("transcripts.json");
    std::fs::write(&p, serde_json::to_string(&v).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

/// A model that is not in the models folder is downloaded on first use, as SubPlz does.
fn ensure_model(a: &Args) -> Result<(), String> {
    let m = &a.model;
    let have = subthx::pipeline::model_size(&m.models, &m.asr, m.device).is_some()
        && m.models.join("firered_vad.onnx").exists();
    if have {
        return Ok(());
    }
    let id = m.asr.to_string();
    if subthx::model_dl::source(&id).is_none() {
        return Err(format!("{id} is not downloadable: put its files in {}", m.models.display()));
    }
    eprintln!("{id} is not in {}: downloading it and the VAD from Hugging Face (subthx)", m.models.display());
    let mut last = 0;
    subthx::model_dl::download(&m.models, &id, &mut |p| {
        if p.file != last {
            last = p.file;
            eprintln!("  file {}/{}", p.file, p.files);
        }
        Ok(())
    })?;
    eprintln!("{id} is ready");
    Ok(())
}

fn run(a: Args, sync: bool) -> Result<(), String> {
    let t0 = Instant::now();
    if let Some(dir) = &a.model.cuda_dir {
        service::add_dll_dir(dir)?;
    }
    subthx::cuda::use_downloaded(&a.model.models)?;
    ensure_model(&a)?;
    let fmt = format_of(&a);
    let jobs = jobs(&a, sync)?;
    let n_audio: usize = jobs.iter().map(|j| j.audios.len()).sum();
    if a.output.is_some() && n_audio != 1 {
        return Err("-o works with a single audio file; use --output-dir".into());
    }
    let mut engine: Option<Engine> = None;
    let mut failed = 0;
    let mut outputs = Vec::new();
    for job in &jobs {
        // a video left without subtitles in a folder gets generated ones
        let res = if sync && !job.texts.is_empty() { run_sync(&a, fmt, job, &mut engine) } else { run_gen(&a, fmt, job, &mut engine) };
        let first_out = output_path(&a, &job.audios[0], fmt);
        if let Err(e) = res {
            failed += 1;
            eprintln!("error: {}: {e}", job.audios[0].display());
            // like SubPlz: a note next to where the subtitles would be
            match service::write_subfail(&job.audios[0], &first_out, &e) {
                Ok(p) => eprintln!("error log written to {}", p.display()),
                Err(e) => eprintln!("failed to write the error log: {e}"),
            }
        }
        outputs.extend(job.audios.iter().map(|au| output_path(&a, au, fmt)));
    }
    for p in service::cleanup_subfail(&outputs) {
        eprintln!("removed {} (subtitles exist now)", p.display());
    }
    eprintln!("done in {:.2}s", t0.elapsed().as_secs_f64());
    if failed > 0 {
        return Err(format!("{failed} of {} jobs failed", jobs.len()));
    }
    Ok(())
}

fn run_gen(a: &Args, fmt: Format, job: &Job, engine: &mut Option<Engine>) -> Result<(), String> {
    let audio = &job.audios[0];
    let out = output_path(a, audio, fmt);
    if !prepare_output(&out, a.overwrite)? {
        return Ok(());
    }
    eprintln!("{}", audio.display());
    let tr = transcribe_audio(a, engine, audio)?;
    dump_transcripts(a, &out, &tr)?;
    let segs = service::gen(&tr, !subthx::pipeline::breaks_lines(&a.model.asr));
    service::write_subs(&out, &segs, fmt)?;
    eprintln!("wrote {} ({} cues)", out.display(), segs.len());
    Ok(())
}

fn run_sync(a: &Args, fmt: Format, job: &Job, engine: &mut Option<Engine>) -> Result<(), String> {
    let outs: Vec<PathBuf> = job.audios.iter().map(|au| output_path(a, au, fmt)).collect();
    // only a check here: the old file is moved aside after the texts are read
    // (subtitles being retimed may be the very file written)
    if let Some(o) = outs.iter().find(|o| o.exists() && !a.overwrite) {
        eprintln!("{} already exists, skipping (use --overwrite)", o.display());
        return Ok(());
    }
    let mut trs = Vec::new();
    for (audio, out) in job.audios.iter().zip(&outs) {
        eprintln!("{}", audio.display());
        let tr = transcribe_audio(a, engine, audio)?;
        dump_transcripts(a, out, &tr)?;
        trs.push(tr);
    }
    let t = Instant::now();
    let grouping = service::grouping_for(&job.texts, a.respect_grouping.then_some(a.respect_grouping_count));
    let synced = service::sync(&job.audios, &trs, &job.texts, grouping)?;
    for m in &synced.matches {
        match (&m.text_chapter, m.score) {
            (Some(tc), Some(s)) => eprintln!("  {:<28} -> {tc} ({s:.0})", m.audio_chapter),
            _ => eprintln!("  {:<28} -> (no text match)", m.audio_chapter),
        }
    }
    if let Some(p) = synced.book_match {
        eprintln!("  book match {p:.2}%");
    }
    for (segs, out) in synced.files.iter().zip(&outs) {
        prepare_output(out, a.overwrite)?;
        service::write_subs(out, segs, fmt)?;
        eprintln!("wrote {} ({} cues)", out.display(), segs.len());
    }
    eprintln!("  text + alignment {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    Ok(())
}
