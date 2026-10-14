# Architecture of subthx

subthx makes subtitles for audiobooks: it recognizes speech and aligns it to the book's text, or just recognizes speech without a book. It is a Rust port of the synchronization from [SubPlz](https://github.com/kanjieater/SubPlz): no Python, PyTorch or ffmpeg, with the same result — the output matches SubPlz byte for byte on identical transcripts.

One library works in three forms:

| What | Where | For whom |
|---|---|---|
| CLI `subthx sync / gen` | `src/main.rs` | scripts, batch processing of folders |
| HTTP server `subthx serve` | `src/server.rs` | other programs; reference at `/docs` |
| Application (Tauri) | `app/` | people; runs the same server inside |

```
subthx/
├── src/            library + CLI
│   └── server/     docs.html and openapi.json, compiled into the binary
├── app/            desktop app: src-tauri (Rust) + src (HTML/CSS/JS)
├── tests/          integration tests, checked against SubPlz
├── tools/          Python scripts: test oracles, model export to ONNX
├── models/         models (not in git)
└── fixtures/       reference outputs of SubPlz (not in git)
```

## Pipeline

```
audio or video (m4b, mp3, flac, ogg, opus, wav, mp4, mkv, webm…)
  │  media.rs   chapters: QuickTime chapter track, chpl, ID3 CHAP, the rest via ffprobe
  │  audio.rs   decoding (symphonia, ffmpeg when needed) → 16 kHz mono
  ▼
FireRedVAD     vad.rs   fbank (kaldi) → CMVN → DFSMN (ONNX) → smoothing
  │                     → speech segments (up to 8 s)
  ▼
recognition    mel.rs   log-mel as in NeMo (80 or 128 bands)
  │            kitsune.rs  CTC, argmax inside the graph
  │            asr.rs      Parakeet TDT, greedy decoding
  ▼
transcript per audio chapter (pipeline.rs: Engine)
  │
  │   book: epub.rs / textsrc.rs (epub, txt, srt/vtt/ass → sentences, pysbd)
  ▼
chapter matching     sync.rs   match_start + expand_matches (fuzz ratio)
  ▼
alignment            batch.rs + align.rs + lev.rs
  │                  Levenshtein editops (port of rapidfuzz) + SubPlz heuristics
  ▼
shift.rs    shifting cue boundaries by punctuation
regroup.rs  (optional) cues regrouped into whole sentences of the book
  ▼
SRT / VTT  +  book match (share of the book's text that ended up in the subtitles)
```

`gen` skips the book: the chapter transcripts are joined with offsets equal to the chapter durations and passed through `shift_align`.

## Modules

| Module | What it does |
|---|---|
| `media.rs` | parses MP4/MOV/MP3: chapters, duration, AAC samples, edit list (priming); chapters of other containers via ffprobe |
| `audio.rs` | decoding to 16 kHz mono, in parallel chunks: AAC from MP4/MOV and MP3 (trimming LAME delay and priming, as ffmpeg does) with its own parser; FLAC, WAV, AIFF, CAF, Ogg Vorbis, ALAC, MKV/WebM through symphonia, chunks laid out by cumulative packet durations (by timestamps when the timescale is coarser than a sample, as in MKV); resampler on AVX2 |
| `whisper.rs` | Whisper (whisper-ja-anime-v0.3) on ONNX Runtime, as batched faster-whisper with word-level timestamps (the way SubPlz runs it): VAD segments are packed into 30 s windows (silence between them zeroed), the encoder runs in batches, greedy decoding with timestamps following Whisper's rules (the model was trained on subtitles and closes every line with a timestamp — usually one character's line, even without punctuation; these boundaries become cue boundaries, and SubPlz's `shift_align` is disabled for Whisper), a limit on length by speech duration, cutting of repeats; then one decoder pass over the text, cross-attention weights of the alignment heads → normalization, median filter, DTW over speech frames only (as `find_alignment` / CTranslate2 `align` with stable-ts silence suppression). Cues are built from words: end of sentence, a pause between VAD segments, and for long ones a comma or a space (song lyrics). The graphs are prepared by `tools/whisper_onnx.py` from an optimum export; there are two of them instead of three or four: the encoder with attention fused into `MultiHeadAttention` (memory-efficient attention on CUDA: 1.8× faster and no 1500×1500 matrices in memory), outputting fp16 cross-attention caches at once; one decoder (decoder_with_past with an expanded length — it was 1 when traced) runs the prompt, every step and the alignment pass, with decoder weights loaded into memory once. A batch is 8 windows (+3.3 GB of VRAM, ~270× on an RTX 4070; SubPlz with CTranslate2 on the same series: +3.9 GB). KV caches live on the GPU (IoBinding); CUDA arenas grow in powers of two and are shrunk periodically, otherwise the tensors that grow with each step leak |
| `ffmpeg.rs` | fallback for what symphonia lacks (Opus, AC-3, DTS, WMA, AVI, FLV, MPEG-TS): ffmpeg from `SUBTHX_FFMPEG`, next to the exe, or from PATH; long files are split into parallel processes with `-ss`, and the parts are joined by overlap correlation (`-ss` accuracy in AVI is tens of ms) |
| `vad.rs` | FireRedVAD as a whole: fbank, CMVN, ONNX model, postprocessing; creation of ONNX sessions for a device (`session_with`) |
| `mel.rs` | NeMo log-mel features; the filterbank is loaded from `nemo*_fbanks.npy` |
| `kitsune.rs` | Kitsune (HF ParakeetForCTC): batches by length, mel in a separate thread, CTC decoding |
| `asr.rs` | Parakeet TDT: encoder + decoder/joint, greedy batched decoder as in SubPlz |
| `pipeline.rs` | `Engine`: models loaded once, `transcribe_with` with progress and cancellation; model catalog `CATALOG` |
| `epub.rs`, `html.rs` | reading EPUB as SubPlz does (ebooklib + BeautifulSoup): chapters, headings, paragraphs |
| `textsrc.rs` | text sources: epub, txt, subtitles; a port of pysbd for Japanese |
| `lang.rs` | cleaning and normalizing text for matching: Japanese separately, other languages as English in SubPlz (lowercase) |
| `lev.rs` | 1:1 port of `rapidfuzz` editops/opcodes, including tie splitting; LCS |
| `align.rs`, `batch.rs` | aligning the transcript to the book's paragraphs (`do_batch` of SubPlz) |
| `sync.rs` | finding which book chapter each audio chapter starts at, and grouping into batches |
| `shift.rs` | `shift_align`, SRT/VTT writing with time formatting as in Python |
| `regroup.rs` | `--respect-grouping`: a port of `nc_align` (DP), 20× faster than the original |
| `service.rs` | shared steps for the CLI and the server: file pairs, output names, `.subfail`, `sync`/`gen`, book match |
| `server.rs` | HTTP API, queue, worker, transcript cache, SSE, CORS |
| `pyrepr.rs`, `npy.rs`, `fixture.rs` | reading SubPlz's cache (Python repr), `.npy`, test fixtures |

## Accuracy against SubPlz

The main rule of the port: every stage is checked against the original on real books. To do that:

- `tools/dump_*.py` run parts of SubPlz (from its venv) and save the reference: transcripts, editops, cleaned text, epub chapters, sentences, VAD, ASR.
- `tests/*.rs` compare our result with the reference. Alignment, chapter matching, SRT, epub, sentences and `nc_align` match exactly; features and models match within numeric tolerance.
- Python quirks are reproduced explicitly: slices and negative indices (`py_slice`), `round`, `divmod`, `%f` formatting, `str.strip`, universal newlines.

Peculiarities of SubPlz found along the way (for example, reading `.txt` as cp1252 on Windows) are not copied: subthx reads UTF-8 everywhere.

## Models and devices

`pipeline::CATALOG` is the list of all known models: id, name, family and subtitle languages. The server returns it on `/v1/models` together with what is installed and how much it weighs; the application builds the language and model picker from it.

| Model | Type | Languages | GPU | CPU |
|---|---|---|---|---|
| Kitsune 0.05B / 0.1B / 0.3B | CTC | ja | fp16, attention in fp32 (`mixed16b`) | int8 for MatMul |
| Parakeet Ultra (v3) | TDT | 25 European | encoder fp16, decoder fp16 with argmax in the graph | encoder int8, decoder fp32 |
| Parakeet Redux (v3) | TDT | 25 European | ternary weights in 4 bits, fp16 graph | encoder w2a8, decoder fp32 |
| FireRedVAD | DFSMN | any | fp16 | fp32 |

Everything runs on ONNX Runtime (`ort`): CUDA, DirectML or CPU. CUDA needs the CUDA 13 and cuDNN 9 DLLs. They come from `<models>/cuda` (the "Download" button in the app's settings: the nvidia-cublas, nvidia-cudnn-cu13 and nvidia-cuda-nvrtc wheels from PyPI, about 0.9 GB, checked against the sha256 in the PyPI JSON; only the DLLs are kept; `src/cuda.rs`), or from `--cuda-dir` / the folder setting, or from PATH. Models: the model menu shows a download button next to the ones not yet downloaded. Files come from the subthx organization's repositories on HF (`src/model_dl.rs`): each repository mirrors part of the `models/` folder, `manifest.json` gives sizes and sha256, and each file is verified before it gets its final name. The VAD (firered-vad-onnx) is downloaded with the model. Encoders on CUDA run with TF32: without it cuDNN does not find fast convolution plans.

On European languages, alignment works as in SubPlz (it matches it line for line), but the SubPlz algorithm compares text character by character and was designed for Japanese: on an hour of an English podcast (This American Life), about 91% of the transcript ends up in the subtitles, against 95% for the recognition itself. The losses are where voices overlap.

Speed on a 7.5-hour book (RTX, CUDA): Kitsune 0.05B takes about 8.5 s for the whole `sync`, of which decoding is 2.8 s, VAD 1.9 s, recognition 3.3 s, alignment about 0.1 s. On the CPU with int8, about 70 s.

### Preparing Parakeet v3 models

Ultra and Redux come from Olicorne's exports on Hugging Face (`fp16/`, `int8/`, `fp32/` and `w2a8/`). For the GPU, three files are made from them; the models work without them, but more slowly:

```bash
python tools/repack_redux.py models/parakeet-redux/encoder-model.w2a8.onnx models/parakeet-redux/encoder-model.w4.onnx
python tools/slim_parakeet_v3.py encoder models/parakeet-redux/encoder-model.w4.onnx models/parakeet-redux/encoder-model.w4.fp16.onnx
python tools/slim_parakeet_v3.py decoder models/parakeet-ultra/decoder_joint-model.onnx models/parakeet-ultra/decoder_joint-model.fp16.onnx
```

- `repack_redux.py`: the CUDA provider does not support 2-bit `MatMulNBits`; ternary weights are repacked losslessly into 4 bits.
- `slim_parakeet_v3.py encoder`: the graph in fp16, so `MatMulNBits` on large batches goes through cuBLAS. Redux encoder: 123 → 13 s on 7.5 h.
- `slim_parakeet_v3.py decoder`: removes the outputs for beam search, computes argmax inside the graph and converts the decoder to fp16. Decoder: 7.7 → 3.4 s on 7.5 h.

`pipeline::parakeet_v3_files` picks the best file for the device by itself. Recognition of 7.5 h on an RTX: Ultra 23 s (1160×), Redux 19 s (1400×), Japanese Parakeet 16 s (1645×; v3 has 128 mel bands and a vocabulary twice as large).

### Preparing Whisper

From the willopcbeta/whisper-ja-anime-v0.3-ONNX export (`onnx/*_fp16.onnx`), the scripts make `mel_filters.npy` and two graphs that `whisper.rs` reads (Python with `onnx` and `onnxruntime`, for example SubPlz's venv):

```bash
python tools/dump_whisper.py models/whisper-ja-anime-v0.3
python tools/whisper_onnx.py models/whisper-ja-anime-v0.3
```

After that, the original `encoder_model_fp16.onnx`, `decoder_model_fp16.onnx` and `decoder_with_past_model_fp16.onnx` are not needed.

**Multilingual turbo.** The decoder of onnx-community/whisper-large-v3-turbo was exported with a newer transformers and does not accept the "one-token length" fix, so the graphs are taken from the Japanese model (same architecture: 32+4 layers, 1280, 20 heads; a different vocabulary), and the weights from the target model (`tools/whisper_swap_weights.py`, matching by layer names; all large tensors must be replaced):

```bash
# whisper-large-v3-turbo: weights from the downloaded onnx-community ONNX (folder src/onnx); generation_config.json from openai (onnx-community's alignment heads are from large-v3)
python tools/whisper_swap_weights.py models/whisper-ja-anime-v0.3 models/whisper-large-v3-turbo/src models/whisper-large-v3-turbo
# TheStageAI/thewhisper-large-v3-turbo: their model.safetensors (fp32 → fp16), tokenizer.json and mel_filters.npy — as for turbo
python tools/whisper_swap_weights.py models/whisper-ja-anime-v0.3 models/whisper-thewhisper-large-v3-turbo/src/model.safetensors models/whisper-thewhisper-large-v3-turbo
python tools/dump_whisper.py models/whisper-large-v3-turbo --filters-only
python tools/whisper_onnx.py models/whisper-large-v3-turbo
```

The Whisper language is set per job (a language token; it also selects the audio track). For languages with spaces, words are assembled by spaces (openai `split_tokens_on_spaces`), the end of a sentence is checked by the book segmenter (not after "Mr."), and a long line is split at a comma from 60 characters, not at a space.

One hour of an English podcast (This American Life, WER against the transcript): turbo 8.0% with line timestamps, 7.8% without; TheWhisper 8.5% without timestamps, 10.7% with them — the fine-tune lost the timestamps, so it decodes without them (`SUBTHX_WHISPER_TS=0/1` overrides, `SUBTHX_WHISPER_WINDOW` sets the window length); Parakeet Redux 12.3%. Japanese TheWhisper writes romaji — the catalog lists only the languages of its card. Speed ~230–260× on an RTX 4070.

### How to add a model

1. An entry in `CATALOG` with its languages.
2. A variant in `AsrModel` and the recognition implementation in `Engine` (Whisper and Moonshine have their own decoding; VAD, audio decoding and the server are shared).
3. The file choice for a device (`kitsune_file`) and the size on disk (`model_size`).
4. For a new language: text cleaning in `lang.rs` and sentence splitting in `textsrc.rs`.

## Server

```
HTTP (axum, tokio)                     worker (one thread)
  POST /v1/jobs ──► queue (mpsc) ────► Engine: decoding → VAD → ASR
  GET  …/events ◄── watch channel ◄─── progress, cancellation checked between batches
  GET  …/result ◄── Mutex<JobResult> ◄ alignment, writing files
```

- The graphics card handles one job at a time; the others wait in the queue. `Engine` belongs to the worker and is not shared between threads.
- Job states: `queued → loading → decoding → vad → asr → aligning → done | failed | cancelled`.
- In-memory transcript cache (key: path, size, modification time, model, device): re-syncing the same recording with another text takes about 0.1 s.
- Models can be switched (`/v1/models/load`) or unloaded; `--idle-unload` unloads them after idling.
- Security: listens on `127.0.0.1` by default; `--token` enables a Bearer token; `--allow-dir` restricts paths; `--cors` allows the app's windows.
- `/v1/audio/transcriptions` mirrors the OpenAI Whisper API, so existing programs work without changes.
- The full reference with examples is at `/docs` of the running server; the machine-readable one is `/openapi.json`.

## Application

```
app/
├── index.html, src/main.js, src/i18n.js, src/styles.css   interface (no framework)
└── src-tauri/src/main.rs                                   shell
```

- At startup the shell reads the settings (`%APPDATA%/app.subthx/settings.json`) and runs the subthx server in the same process. The window talks to it over HTTP like any other program, so everything the window can do is also available through the API.
- Shell commands: settings, "show in folder", restart. Paths, port and the CUDA folder apply after a restart; the model, device and subtitle parameters apply at once.
- Interface: one "Books" list (in progress at the top, finished below, history in `localStorage`), drag and drop of files and folders (`/v1/plan` sorts them into books), its own window title bar, settings in a side panel, three interface languages.
- Design: paper, ink and one accent — the vermilion of a seal. Tokens in `:root` in `styles.css`, dark theme through `prefers-color-scheme`.

## Build and checks

```bash
cargo build --release                 # CLI and server: target/release/subthx.exe
cargo test --release                  # tests; many need the books in E:\audiobooks and the CUDA DLLs in PATH
cd app && npm install && npx tauri build --no-bundle   # app: target/release/subthx-app.exe
```

Tests that need external data look for it through environment variables (`SUBTHX_DEVICE`, `SUBTHX_REGROUP_DIR` and others) and are silently skipped when the data is missing.

## Model licenses

Models are not in the repository; the catalog of sources and licenses is in [MODELS.md](MODELS.md). Kitsune is distributed under CC BY-NC 4.0 (non-commercial use, with attribution). The upstream cards of Parakeet (redux, ultra) and Whisper JA Anime have no license; Whisper Large v3 Turbo is MIT, TheWhisper is CC BY 4.0, FireRedVAD is Apache-2.0. Authors must be credited when models are distributed with the application.

## Releases

A release has two executables and no installer: `subthx` (CLI and HTTP server) and `subthx-gui` (the desktop app). Both look for the `models` folder next to the executable, or up the directory tree. Model downloads and the CUDA libraries are fetched by the app itself, so nothing else ships in the release.

## How the mode is chosen

There is no mode switch: `/v1/plan` (`server::plan_paths`) decides from the dropped files. A book (epub/txt) → sync; subtitles in the job's language → retiming; nothing → generation.

- **Own outputs.** `service::write_subs` records each file it writes in `%LOCALAPPDATA%\subthx\outputs.json` (path, size, mtime; `SUBTHX_STATE_DIR` overrides the folder). Such a file, while it is unedited, and `.old` copies are not treated as the user's subtitles: dropping the folder again after generation means generation again, not retiming of its own result.
- **Subtitle language.** `service::subtitle_fits`: a language tag in the name (`.ja.`, `.jpn.`, `.en-US.`) must match the job's language, and the script of the text must fit (kana and kanji for Japanese, Cyrillic, Latin…; a name can lie). Subtitles that do not fit are a warning and generation.
- **Embedded tracks.** A video without suitable external subtitles is checked with `ffprobe`: a text track (subrip/ass/webvtt/mov_text) with a language tag — a full one, not forced/SDH/signs — gives retiming; in a job where the video itself is the text, that is the source. `ffmpeg::extract_subtitles` pulls it out when the job starts (ASS stays ASS) into a cache in a temporary folder.
- **Errors that remain.** A book match below 60% — the card says "book does not fit" / "subtitles do not fit" with a "Generate" button; the ⋯ menu on any card redoes it as generation, as sync with another file, or the same again (with overwrite).

## Sentences in European languages

Japanese books and subtitles are split like pysbd (ja) — `textsrc::pysbd_ja`. For languages with spaces (`textsrc::sentences_spaced`, the language is taken from the transcript), a sentence ends at `. ! ? …` with closing quotes before a space and not before a lowercase letter, except for abbreviations of that language (Mr., z.B., Sr.…), initials and words with dots inside; French `« … »` with a space stays with its sentence.
