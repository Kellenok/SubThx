// subthx window: drop an audiobook (+ its book), watch it sync, find the file.
// Everything goes through the in-process subthx HTTP API.

import { LANGS, fmtDate, fmtDuration, fmtNum, lang, setLang, t, tn } from "./i18n.js";

const TAURI = "__TAURI_INTERNALS__" in window;

const tauri = TAURI
  ? {
      core: await import("@tauri-apps/api/core"),
      webview: await import("@tauri-apps/api/webview"),
      window: await import("@tauri-apps/api/window"),
      dialog: await import("@tauri-apps/plugin-dialog"),
    }
  : null;

// ---------------------------------------------------------------- host

/** Settings + API address from the app shell (or a browser preview). */
const host = {
  async state() {
    if (tauri) return tauri.core.invoke("get_state");
    const saved = JSON.parse(localStorage.getItem("subthx-settings") || "null");
    return {
      settings: {
        asr: "kitsune-p0.05b",
        device: "cuda",
        format: "srt",
        lang_ext: "ja",
        overwrite: true,
        respect_grouping: false,
        output_dir: null,
        models_dir: "models",
        cuda_dir: null,
        port: 7865,
        idle_unload: 0,
        lang: "auto",
        ...saved,
      },
      api: new URLSearchParams(location.search).get("api") || "http://127.0.0.1:7865",
      api_error: null,
      restart_needed: false,
    };
  },
  async save(settings) {
    if (tauri) return tauri.core.invoke("save_settings", { settings });
    localStorage.setItem("subthx-settings", JSON.stringify(settings));
    return { ...state.snap, settings };
  },
  async pick({ folder = false, multiple = true } = {}) {
    if (!tauri) {
      toast(t("pickInApp"));
      return [];
    }
    const filters = folder
      ? undefined
      : [{ name: t("filesFilter"), extensions: [...MEDIA_EXTS, "epub", "txt", "srt", "vtt", "ass", "ssa"] }];
    const r = await tauri.dialog.open({ directory: folder, multiple, filters });
    if (!r) return [];
    return Array.isArray(r) ? r : [r];
  },
  reveal: (path) => tauri?.core.invoke("reveal", { path }),
  restart: () => tauri?.core.invoke("restart"),
};

// audio and video the server takes (service::AUDIO_EXTS)
const MEDIA_EXTS = [
  "m4b", "m4a", "mp4", "mp3", "aac", "flac", "wav", "ogg", "oga", "opus", "aiff", "aif", "caf", "wma", "mka", "mkv", "webm", "mov", "m4v",
  "avi", "wmv", "flv", "ts", "mts", "m2ts", "mpg", "mpeg", "3gp",
];

// ---------------------------------------------------------------- api

async function api(path, { method = "GET", body } = {}) {
  const res = await fetch(state.snap.api + path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(data?.error?.message || res.statusText);
  return data;
}

// ---------------------------------------------------------------- state

const HISTORY_KEY = "subthx-history";
const HISTORY_MAX = 300;

const state = {
  snap: null,
  health: null,
  /** /v1/models: every known model with languages, install state, size */
  catalog: [],
  /** subtitle languages some model covers */
  languages: [],
  loading: false,
  loadError: null,
  /** running and queued jobs */
  jobs: [],
  /** finished jobs, newest first (summaries only) */
  history: loadHistory(),
};

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history.slice(0, HISTORY_MAX)));
  } catch {}
}

// One line per model: speed and accuracy relative to its siblings (real
// times depend on the machine). Unknown catalog entries just show their name.
const MODEL_HINTS = {
  "kitsune-p0.05b": "m005",
  "kitsune-p0.1b": "m01",
  "kitsune-p0.3b": "m03",
  "parakeet-ultra": "mUltra",
  "parakeet-redux": "mRedux",
  "whisper-ja-anime-v0.3": "mWhisperAnime",
  "whisper-large-v3-turbo": "mWhisperTurbo",
  "whisper-thewhisper-large-v3-turbo": "mTheWhisper",
};
const modelName = (id) => state.catalog.find((m) => m.id === id)?.name ?? id;
const DEVICES = { cuda: "CUDA", directml: "DirectML", cpu: "CPU" };
const RUNNING = new Set(["decoding", "vad", "asr", "aligning"]);
const finished = (s) => s === "done" || s === "failed" || s === "cancelled";

// ---------------------------------------------------------------- helpers

const $ = (sel) => document.querySelector(sel);

/** h("div.class", {attrs}, ...children); null/false children are skipped */
function h(tag, attrs, ...kids) {
  const [name, ...cls] = tag.split(".");
  const el = document.createElement(name);
  if (cls.length) el.className = cls.join(" ");
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style") el.style.cssText = v;
    else if (k in el && k !== "list") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  el.append(...kids.flat().filter((k) => k != null && k !== false));
  return el;
}

function svg(paths) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 20 20");
  s.setAttribute("aria-hidden", "true");
  s.innerHTML = paths;
  return s;
}

const ICON = {
  download: '<path d="M10 3.5v9M6.5 9.5 10 13l3.5-3.5M4.5 15.5v.5A1.5 1.5 0 0 0 6 17.5h8a1.5 1.5 0 0 0 1.5-1.5v-.5"/>',
  plus: '<path d="M10 4.5v11M4.5 10h11"/>',
  folder: '<path d="M3 6.5A1.5 1.5 0 0 1 4.5 5h3l1.5 1.5h6.5A1.5 1.5 0 0 1 17 8v6.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 3 14.5z"/>',
  close: '<path d="M6 6l8 8M14 6l-8 8"/>',
  reveal: '<path d="M8 4.5H5.5a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1V12M11 4.5h4.5V9M15.5 4.5L9 11"/>',
  retry: '<path d="M15.5 10a5.5 5.5 0 1 1-1.6-3.9M15.5 4v3h-3"/>',
  info: '<circle cx="10" cy="10" r="6.75"/><path d="M10 9v4.25"/><circle cx="10" cy="6.6" r=".55" fill="currentColor" stroke="none"/>',
  copy: '<rect x="7" y="7" width="9" height="9" rx="1.5"/><path d="M13 7V5.5A1.5 1.5 0 0 0 11.5 4h-6A1.5 1.5 0 0 0 4 5.5v6A1.5 1.5 0 0 0 5.5 13H7"/>',
  more: '<circle cx="5" cy="10" r="1.25" fill="currentColor" stroke="none"/><circle cx="10" cy="10" r="1.25" fill="currentColor" stroke="none"/><circle cx="15" cy="10" r="1.25" fill="currentColor" stroke="none"/>',
};

const base = (p) => p.split(/[\\/]/).pop();
const stem = (p) => base(p).replace(/\.[^.]+$/, "");
const ext = (p) => (base(p).match(/\.([^.]+)$/)?.[1] || "").toLowerCase();
const fmtBytes = (n) => (n >= 1e9 ? t("gb", fmtNum(n / 1e9, 1)) : t("mb", fmtNum(Math.round(n / 1e6))));
const pct = (p) => `${fmtNum(Math.floor(p * 10) / 10, p >= 99.95 ? 0 : 1)}%`;

let toastTimer;
/** what an error shows: a known code's line, else the first line of the message, cut */
function errorText(e) {
  const keys = {
    cuda_missing: "errCudaMissing",
    load_failed: "errLoadFailed",
    worker_stopped: "errLoadFailed",
    network: "errNetwork",
    checksum: "errChecksum",
    download_failed: "errDownload",
    model_missing: "errModelMissing",
  };
  if (keys[e]) return t(keys[e]);
  const line = String(e ?? "").split("\n")[0].trim();
  return line.length > 90 ? `${line.slice(0, 89)}…` : line;
}

function toast(text) {
  const el = $("#toast");
  el.textContent = text;
  el.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("on"), 3600);
}

// ---------------------------------------------------------------- top bar

function renderStatic() {
  $("#open-settings").setAttribute("aria-label", t("settings"));
  $("#open-settings").title = t("settings");
  $("#settings-title").textContent = t("settings");
  $("#settings-close").setAttribute("aria-label", t("close"));
  $("#veil-text").textContent = t("dropRelease");
  $("#win-min").setAttribute("aria-label", t("minimize"));
  $("#win-max").setAttribute("aria-label", t("maximize"));
  $("#win-close").setAttribute("aria-label", t("close"));
}

function renderEngine() {
  const el = $("#engine");
  const s = state.snap?.settings;
  const model = state.health?.model;
  let st = "idle",
    text = state.snap ? t("serverDown") : t("starting");
  if (state.snap?.api_error) [st, text] = ["error", t("serverFailed")];
  else if (state.loading) [st, text] = ["loading", t("loadingModel", modelName(s.asr))];
  else if (state.loadError) [st, text] = ["error", t("modelFailed")];
  else if (model) {
    const name = `${modelName(model.asr)} · ${DEVICES[model.device] ?? model.device}`;
    if (model.busy) [st, text] = ["busy", name];
    else if (model.loaded) [st, text] = ["ready", name];
    else [st, text] = ["idle", t("unloaded", modelName(model.asr))];
  }
  el.dataset.state = st;
  el.querySelector(".engine-text").textContent = text;
  el.title = state.loadError ? errorText(state.loadError) : state.snap?.api_error ? errorText(state.snap.api_error) : t("engineHint");
}

async function loadModel() {
  if (!state.snap?.api) return;
  const { asr, device } = state.snap.settings;
  state.loading = true;
  state.loadError = null;
  renderEngine();
  try {
    await api("/v1/models/load", { method: "POST", body: { asr, device } });
  } catch (e) {
    state.loadError = e.message;
    toast(t("modelFailedToast", errorText(e.message)));
  }
  state.loading = false;
  await pollHealth();
}

async function pollHealth() {
  if (!state.snap?.api) return renderEngine();
  try {
    state.health = await api("/v1/health");
  } catch {
    state.health = null;
  }
  renderEngine();
}

// ---------------------------------------------------------------- jobs

let seq = Date.now();

async function addPaths(paths) {
  if (!paths.length) return;
  if (!state.snap?.api) return toast(t("serverNotRunning"));
  let plan;
  try {
    plan = await api("/v1/plan", { method: "POST", body: { paths, language: state.snap.settings.language || null } });
  } catch (e) {
    return toast(errorText(e.message));
  }
  for (const w of plan.warnings) toast(w);
  if (!plan.jobs.length) {
    if (!plan.warnings.length) toast(t("noAudio"));
    return;
  }
  for (const j of plan.jobs) submit({ key: ++seq, mode: j.mode, audio: j.audio, text: j.text });
  render();
}

// jobs go to the server one at a time, so its queue keeps the list's order
let posting = Promise.resolve();

function submit(job) {
  Object.assign(job, { id: null, status: { state: "queued", progress: 0 }, runStart: null });
  state.jobs.push(job);
  posting = posting.then(() => post(job));
}

async function post(job) {
  const s = state.snap.settings;
  try {
    const st = await api("/v1/jobs", {
      method: "POST",
      body: {
        mode: job.mode,
        audio: job.audio,
        text: job.text,
        write: true,
        // redoing a card replaces what it wrote
        overwrite: job.overwrite ?? s.overwrite,
        format: s.format,
        lang_ext: s.lang_ext || null,
        output_dir: s.output_dir,
        respect_grouping: s.respect_grouping,
        language: s.language || null,
      },
    });
    job.id = st.id;
    update(job, st);
    watch();
  } catch (e) {
    finish(job, { state: "failed", progress: 0, error: e.message, elapsed: 0 }, null);
  }
}

// One poll of /v1/jobs for every job: an event stream per job would hit the
// webview's limit of six connections per host and starve other requests.
let polling = null;

function watch() {
  polling ??= setInterval(poll, 400);
}

let inFlight = false;
async function poll() {
  if (inFlight) return;
  const live = state.jobs.filter((j) => j.id && !finished(j.status.state));
  if (!live.length) {
    clearInterval(polling);
    polling = null;
    return;
  }
  inFlight = true;
  try {
    const all = await api("/v1/jobs");
    const byId = new Map(all.map((st) => [st.id, st]));
    for (const job of live) {
      const st = byId.get(job.id);
      if (st) update(job, st);
      // the server forgot it (restarted): nothing more will come
      else finish(job, { state: "failed", progress: 0, error: t("serverDown"), elapsed: 0 }, null);
    }
  } catch {
  } finally {
    inFlight = false;
  }
}

async function update(job, st) {
  if (RUNNING.has(st.state) && !job.runStart) job.runStart = performance.now();
  const was = job.status.state;
  job.status = st;
  if (!finished(st.state)) return renderJob(job);
  if (finished(was)) return;
  let result = null;
  if (st.state === "done") {
    try {
      result = await api(`/v1/jobs/${job.id}/result?format=json`);
    } catch {}
  }
  finish(job, st, result);
  pollHealth();
}

/** A finished job turns into its result card (cancelled ones vanish). */
function finish(job, st, result) {
  const cover = document.querySelector(`.job:not(.job-done)[data-key="${job.key}"] .cover`);
  if (cover) {
    cover.style.setProperty("--p", "1");
    movedCovers.set(job.key, cover);
  }
  state.jobs = state.jobs.filter((j) => j !== job);
  if (st.state !== "cancelled") {
    const entry = summarize(job, st, result);
    entry.fresh = true;
    state.history.unshift(entry);
    saveHistory();
  }
  render();
}

function summarize(job, st, r) {
  const entry = {
    key: job.key,
    mode: job.mode,
    audio: job.audio,
    text: job.text,
    state: st.state,
    error: st.error || null,
    written: st.written || [],
    elapsed: st.took ?? st.elapsed ?? 0,
    finishedAt: Date.now(),
  };
  if (r) {
    entry.cues = r.files.reduce((n, f) => n + f.segments.length, 0);
    entry.audioSec = r.durations.reduce((a, b) => a + b, 0);
    entry.bookMatch = r.book_match ?? null;
    entry.model = r.model ?? null;
    entry.device = r.device ?? null;
    entry.stages = r.stages ?? null;
    entry.tint = job.tint ?? null;
    entry.chapters = (r.matches || []).map((m) => ({ a: m.audio_chapter, t: m.text_chapter }));
  }
  return entry;
}

async function cancel(job) {
  if (job.id) {
    try {
      await api(`/v1/jobs/${job.id}`, { method: "DELETE" });
    } catch {}
  }
}

function removeEntry(entry) {
  state.history = state.history.filter((e) => e !== entry);
  saveHistory();
  render();
}

function retry(entry) {
  redo(entry, entry.mode, entry.text);
}

/** The card's audio again, as `mode` with `text`; the card makes way. */
function redo(entry, mode, text) {
  removeEntry(entry);
  submit({ key: ++seq, mode, audio: entry.audio, text, overwrite: entry.state === "done" ? true : undefined });
  render();
}

const TEXT_EXTS = ["epub", "txt", "srt", "vtt", "ass", "ssa"];

async function redoWithText(entry) {
  const text = (await host.pick()).filter((p) => TEXT_EXTS.includes(ext(p)));
  if (!text.length) return toast(t("pickTextFirst"));
  redo(entry, "sync", text);
}

/** ⋯ on a card: redo it as generation, or as a sync with another text. */
function redoMenu(e) {
  const id = `redo-${e.key}`;
  const item = (label, fn) =>
    h(
      "button.menu-item",
      {
        type: "button",
        onclick: () => {
          menu.hidePopover();
          fn();
        },
      },
      label,
    );
  const menu = h(
    "div.menu",
    { id, popover: "auto" },
    h("div.menu-head", {}, t("redoAs")),
    e.mode === "sync" ? item(t("redoGen"), () => redo(e, "gen", [])) : null,
    item(t("redoSyncWith"), () => redoWithText(e)),
    item(t("redoSame"), () => retry(e)),
  );
  menu.style.setProperty("position-anchor", `--${id}`);
  const btn = iconBtn(ICON.more, t("redoAs"), null, { popovertarget: id });
  btn.style.setProperty("anchor-name", `--${id}`);
  return [btn, menu];
}

function clearHistory() {
  state.history = [];
  saveHistory();
  render();
}

// covers and their color

const tints = new Map(); // cover url -> css color (null: no cover)

// `cors=1`: these are always fetched with CORS (the canvas reads them), so
// they never share a cache entry with a plain image request
const coverUrl = (path) => `${state.snap.api}/v1/cover?cors=1&path=${encodeURIComponent(path)}`;

/** The cover's dominant color, made usable as a tint: the heaviest hue
 *  bucket among saturated, mid-light pixels, then normalized in HSL. */
function sampleTint(img) {
  const n = 24;
  const c = document.createElement("canvas");
  c.width = c.height = n;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(img, 0, 0, n, n);
  let px;
  try {
    px = g.getImageData(0, 0, n, n).data;
  } catch {
    return null; // tainted canvas: no CORS
  }
  const bins = Array.from({ length: 12 }, () => ({ w: 0, r: 0, g: 0, b: 0 }));
  for (let i = 0; i < px.length; i += 4) {
    const [r, gg, b] = [px[i] / 255, px[i + 1] / 255, px[i + 2] / 255];
    const max = Math.max(r, gg, b),
      min = Math.min(r, gg, b);
    const l = (max + min) / 2;
    const d = max - min;
    const sat = d === 0 ? 0 : d / (1 - Math.abs(2 * l - 1));
    let hue = 0;
    if (d) hue = max === r ? ((gg - b) / d) % 6 : max === gg ? (b - r) / d + 2 : (r - gg) / d + 4;
    // vivid, mid-light pixels count most; skin and paper tones barely
    // (covers are mostly faces: their warm tones would win every time)
    const deg = (hue * 60 + 360) % 360;
    const skin = deg >= 8 && deg <= 45 && sat < 0.75 ? 0.15 : 1;
    const w = sat * sat * sat * Math.max(0, 1 - Math.abs(l - 0.5) * 1.8) * skin + 0.0001;
    const bin = bins[Math.floor(deg / 30)];
    bin.w += w;
    bin.r += r * w;
    bin.g += gg * w;
    bin.b += b * w;
  }
  const best = bins.reduce((a, b) => (b.w > a.w ? b : a));
  if (!best.w) return null;
  const [r, gg, b] = [best.r / best.w, best.g / best.w, best.b / best.w];
  const max = Math.max(r, gg, b),
    min = Math.min(r, gg, b);
  const d = max - min;
  let hue = 0;
  if (d) hue = max === r ? ((gg - b) / d) % 6 : max === gg ? (b - r) / d + 2 : (r - gg) / d + 4;
  const sat = d === 0 ? 0 : d / (1 - Math.abs(max + min - 1));
  return `hsl(${Math.round((hue * 60 + 360) % 360)} ${Math.round(Math.min(0.9, Math.max(0.45, sat * 1.2)) * 100)}% 55%)`;
}

/** Cover tile. `progress` (0..1) shows it grey with the color rising; null = done. */
function coverEl(job, progress, card) {
  const wrap = h("div.cover", { "aria-hidden": "true" });
  if (progress != null) wrap.style.setProperty("--p", String(progress));
  // skip leading [series] [vol] tags
  const name = titleOf(job).replace(/^(\s*[\[［(（【][^\]］)）】]*[\]］)）】])+\s*/, "");
  const letter = h("span.cover-letter", {}, name.charAt(0));
  wrap.append(letter);
  const sources = [job.audio[0], ...job.text.filter((x) => ext(x) === "epub")];
  let i = 0;
  const next = () => {
    if (i >= sources.length || !state.snap?.api) return;
    const url = coverUrl(sources[i++]);
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.decoding = "async";
    img.onerror = next;
    img.onload = () => {
      letter.remove();
      if (progress != null) {
        const base = new Image();
        base.crossOrigin = "anonymous"; // same cache entry as the color copy
        base.className = "base";
        base.src = url;
        img.className = "color";
        wrap.append(base);
      }
      wrap.append(img);
      if (!tints.has(url)) tints.set(url, sampleTint(img));
      const tint = tints.get(url);
      if (tint) {
        job.tint = tint;
        (card ?? wrap.closest(".job"))?.style.setProperty("--tint", tint);
      }
    };
    img.src = url;
  };
  next();
  return wrap;
}

/** subtitles as the text: they keep their lines and get new times (a video
 * as the text: its embedded subtitles) */
const SUB_EXTS = ["srt", "vtt", "ass", "ssa"];
const isRetime = (job) => job.text.length > 0 && job.text.every((x) => SUB_EXTS.includes(ext(x)) || MEDIA_EXTS.includes(ext(x)));
const isEmbedded = (x) => MEDIA_EXTS.includes(ext(x));
/** below this book match (%) the text was most likely not this audio's */
const LOW_MATCH = 60;

function titleOf(job) {
  if (job.audio.length === 1) return stem(job.audio[0]);
  if (job.text.length === 1 && ext(job.text[0]) === "epub") return stem(job.text[0]);
  return stem(job.audio[0]);
}

function metaOf(job) {
  const parts = [job.mode !== "sync" ? t("modeGen") : isRetime(job) ? t("modeRetime") : t("modeSync")];
  const n = job.audio.length;
  parts.push(n === 1 ? ext(job.audio[0]).toUpperCase() : tn("audioFiles", n));
  for (const x of job.text) parts.push(isEmbedded(x) ? t("embeddedSubs") : ext(x).toUpperCase());
  return parts;
}

// running / queued card: the cover fills with color, a thin line under it

function jobCard(job, cover = null) {
  const st = job.status.state;
  const p = job.status.progress || 0;
  let line = st === "queued" && job.status.queued_ahead ? t("queuedAhead", job.status.queued_ahead) : t(st);
  if (RUNNING.has(st) && job.status.files > 1) line += ` · ${t("fileOf", job.status.file + 1, job.status.files)}`;
  if (job.runStart && p > 0.04) {
    const spent = (performance.now() - job.runStart) / 1000;
    if (spent > 1) line += ` · ${t("eta", fmtDuration(Math.max(1, (spent * (1 - p)) / p)))} ${t("left")}`.trimEnd();
  }
  const card = h("li.job", { "data-state": st, "data-key": job.key });
  if (job.tint) card.style.setProperty("--tint", job.tint);
  card.append(
    cover ?? coverEl(job, p, card),
    h("div.job-main", {}, h("div.job-title", { title: [...job.audio, ...job.text].join("\n") }, titleOf(job)), h("div.job-sub", {}, line)),
    h(
      "div.job-end",
      {},
      RUNNING.has(st) ? h("span.job-pct", {}, `${Math.round(p * 100)}%`) : null,
      iconBtn(ICON.close, t("cancel"), () => cancel(job)),
    ),
    h("div.job-track", {}, h("i", { style: `width:${(p * 100).toFixed(1)}%` })),
  );
  return card;
}

/** Progress updates swap the card but keep its cover (no reload flash). */
function renderJob(job) {
  const old = document.querySelector(`.job[data-key="${job.key}"]`);
  if (!old) return render();
  const oldCover = old.querySelector(".cover");
  oldCover?.style.setProperty("--p", String(job.status.progress || 0));
  const card = jobCard(job, oldCover);
  card.style.animation = "none";
  old.replaceWith(card);
}

const iconBtn = (icon, label, onclick, attrs = {}) =>
  h("button.icon-btn", { type: "button", "aria-label": label, title: label, onclick, ...attrs }, svg(icon));

/** "decoding 2.2 s · speech 1.4 s · …" */
function stagesText(st) {
  if (!st) return null;
  const parts = [];
  if (st.load > 0.05) parts.push(`${t("loading")} ${fmtDuration(st.load)}`);
  if (st.cached) parts.push(t("fromCache"));
  else parts.push(`${t("decoding")} ${fmtDuration(st.decode)}`, `${t("vad")} ${fmtDuration(st.vad)}`, `${t("asr")} ${fmtDuration(st.asr)}`);
  parts.push(`${t("aligning")} ${fmtDuration(st.align)}`);
  return parts.join(" · ");
}

// finished card: the match, how long the audio is, when. The rest under ⓘ.

const opened = new Set(); // keys of finished cards showing their details
/** covers of just-finished jobs, handed to their result cards (no reload) */
const movedCovers = new Map();
/** result cards by entry key, kept across renders so covers never reload */
const doneNodes = new Map();

function doneCard(e) {
  const ok = e.state === "done";
  const card = h("li.job.job-done", { "data-state": e.state, "data-fresh": e.fresh ? "" : null, "data-key": e.key });
  if (e.tint) card.style.setProperty("--tint", e.tint);
  const dot = () => h("span.dot", {}, "·");
  const sub = h("div.job-sub");
  // a text that is not this audio's: say so, generation is one click away
  const low = ok && e.mode === "sync" && e.bookMatch != null && e.bookMatch < LOW_MATCH;
  if (!ok) sub.append(h("span.job-failed", {}, t("failed")));
  else if (low) sub.append(h("b.job-low", {}, pct(e.bookMatch)), h("span.job-low", {}, isRetime(e) ? t("lowMatchSubs") : t("lowMatchBook")));
  else if (e.bookMatch != null) sub.append(h("b", {}, pct(e.bookMatch)), h("span", {}, t("matchShort")));
  else sub.append(h("span", {}, tn("cues", e.cues ?? 0)));
  if (ok && e.audioSec) sub.append(dot(), h("span", {}, fmtDuration(e.audioSec)));
  sub.append(dot(), h("span", {}, fmtDate(e.finishedAt)));

  const open = opened.has(e.key);
  const info = iconBtn(ICON.info, t("details"), () => {
    const now = !opened.has(e.key);
    now ? opened.add(e.key) : opened.delete(e.key);
    info.setAttribute("aria-expanded", String(now));
    if (now) card.append(details(e));
    else card.querySelector(".job-more")?.remove();
  }, { "aria-expanded": String(open) });
  const cover = movedCovers.get(e.key);
  movedCovers.delete(e.key);
  card.append(
    cover ?? coverEl(e, null, card),
    h("div.job-main", {}, h("div.job-title", { title: [...e.audio, ...e.text].join("\n") }, titleOf(e)), sub),
    h(
      "div.job-end",
      {},
      low ? h("button.btn.btn-text.job-action", { type: "button", onclick: () => redo(e, "gen", []) }, t("generateInstead")) : null,
      info,
      ok && e.written.length ? iconBtn(ICON.reveal, t("showInFolder"), () => host.reveal(e.written[0])) : null,
      !ok ? iconBtn(ICON.retry, t("retry"), () => retry(e)) : null,
      ...redoMenu(e),
      iconBtn(ICON.close, t("removeFromList"), () => removeEntry(e)),
    ),
  );
  if (open) card.append(details(e));
  return card;
}

function details(e) {
  if (e.state !== "done") return h("div.job-more", {}, h("div.job-error", { title: e.error || null }, e.error ? errorText(e.error) : t("failed")));
  const fact = (label, value) => (value ? h("div", {}, h("dt", {}, label), h("dd", {}, value)) : null);
  const model = e.model ? `${modelName(e.model)} · ${DEVICES[e.device] ?? e.device}` : null;
  return h(
    "div.job-more",
    {},
    h(
      "dl",
      {},
      fact(t("factMode"), metaOf(e)[0]),
      e.bookMatch != null ? fact(t("bookMatchLabel"), pct(e.bookMatch)) : null,
      e.chapters?.length ? fact(t("chapters"), chaptersFact(e.chapters)) : null,
      fact(t("factCues"), fmtNum(e.cues ?? 0)),
      fact(t("factAudio"), [e.audioSec ? fmtDuration(e.audioSec) : null, ...metaOf(e).slice(1)].filter(Boolean).join(" · ")),
      fact(t("factDone"), `${t("tookTime", fmtDuration(e.elapsed))} · ${fmtDate(e.finishedAt)}`),
      fact(t("factModel"), model),
    ),
    e.stages ? h("p.job-stages", {}, stagesText(e.stages)) : null,
  );
}

/** "11 of 13"; hovering names the chapters the book does not have. */
function chaptersFact(chapters) {
  const missed = chapters.filter((c) => !c.t);
  const text = t("chaptersOf", chapters.length - missed.length, chapters.length);
  if (!missed.length) return text;
  return h(
    "span.has-tip",
    { tabindex: 0 },
    text,
    h("span.tip", { role: "tooltip" }, h("span.tip-head", {}, t("missingChapters")), missed.map((c) => h("span.tip-row", {}, c.a))),
  );
}

// ---------------------------------------------------------------- main view

const FORMATS = ["M4B", "MP3", "FLAC", "OPUS", "MP4", "MKV", "EPUB", "TXT", "SRT", "ASS"];

const pickButtons = (big) => [
  h(
    "button.btn.btn-quiet",
    { type: "button", onclick: async () => addPaths(await host.pick({ folder: true })) },
    svg(ICON.folder),
    t("pickFolder"),
  ),
  h("button.btn.btn-ink", { type: "button", onclick: async () => addPaths(await host.pick()) }, svg(ICON.plus), big ? t("pickFiles") : t("files")),
];

function render() {
  const main = $("#main");
  if (!state.jobs.length && !state.history.length) {
    main.replaceChildren(
      h(
        "section.drop",
        {},
        h(
          "div.drop-inner",
          {},
          h("span.seal.seal-lg", { "aria-hidden": "true" }, h("span")),
          h("h1", {}, t("dropTitle")),
          h("p", {}, t("dropText")),
          h("div.formats", { "aria-label": t("formats") }, FORMATS.map((f) => h("span", {}, f))),
          h("div.actions", {}, pickButtons(true).reverse()),
        ),
      ),
    );
    return;
  }
  // one list: what is running on top, finished books right under it
  main.replaceChildren(
    ...[
      h("div.list-head", {}, h("h2", {}, t("books")), h("div.actions", {}, pickButtons(false))),
      state.jobs.length ? h("ul.jobs", {}, state.jobs.map((j) => kept(document.querySelector(`.job:not(.job-done)[data-key="${j.key}"]`)) ?? jobCard(j))) : null,
      state.history.length
        ? h(
            "div.section-head",
            {},
            h("h3", {}, t("tabDone"), h("span.count", {}, fmtNum(state.history.length))),
            h("button.btn.btn-text", { type: "button", onclick: clearHistory }, t("clearHistory")),
          )
        : null,
      state.history.length ? h("ul.jobs", {}, state.history.map(doneNode)) : null,
    ].filter(Boolean),
  );
  for (const e of state.history) delete e.fresh;
  const live = new Set(state.history.map((e) => e.key));
  for (const k of doneNodes.keys()) if (!live.has(k)) doneNodes.delete(k);
}

/** A card moved to its new place keeps its look: no entry animation again. */
function kept(el) {
  if (el) el.style.animation = "none";
  return el;
}

function doneNode(e) {
  let el = kept(doneNodes.get(e.key));
  if (!el) doneNodes.set(e.key, (el = doneCard(e)));
  return el;
}

// ---------------------------------------------------------------- settings

async function saveSettings(patch) {
  const prev = state.snap.settings;
  const next = { ...prev, ...patch };
  state.snap = await host.save(next);
  if (next.asr !== prev.asr || next.device !== prev.device) {
    loadModel();
    // the file each model loads (and so its size) depends on the device
    if (next.device !== prev.device) refreshCatalog().then(() => $("#settings").open && renderSettings());
  }
  if (next.lang !== prev.lang) {
    setLang(next.lang);
    renderAll();
  }
  renderSettings();
}

/** GET /v1/cuda: the CUDA libraries and their download */
let cuda = null;
async function refreshCuda() {
  try {
    cuda = await api("/v1/cuda");
  } catch {
    cuda = null;
  }
  if (cuda?.download?.running) setTimeout(() => refreshCuda().then(() => $("#settings").open && renderSettings()), 500);
}

/** the device row's hint: a short line; a missing CUDA library offers its download */
function deviceHint() {
  if (!state.loadError) return t("deviceHint");
  if (state.loadError !== "cuda_missing") return errorText(state.loadError);
  return h("span", {}, t("errCudaMissing"), " ", h("button.btn.btn-text", { type: "button", onclick: cudaPrompt }, t("cudaDownload")));
}

/** CUDA without its libraries asks first; the other devices switch now */
async function chooseDevice(v) {
  if (v === "cuda") {
    await refreshCuda();
    if (cuda && !cuda.available) return cudaPrompt();
  }
  saveSettings({ device: v });
}

/** a small modal over the page: a title, a body and the buttons */
function prompt(title, body, actions) {
  const dlg = h("dialog.prompt", { "aria-labelledby": "prompt-title" }, h("h2", { id: "prompt-title" }, title), body, actions);
  dlg.addEventListener("close", () => dlg.remove());
  document.body.append(dlg);
  dlg.showModal();
  return dlg;
}

/** asks before the CUDA download, shows its progress, then offers CUDA */
function cudaPrompt() {
  const body = h("div.prompt-body");
  const actions = h("div.prompt-actions");
  const dlg = prompt(t("cudaPromptTitle"), body, actions);
  dlg.addEventListener("close", () => $("#settings").open && renderSettings());
  const close = (label) => h("button.btn.btn-text", { type: "button", onclick: () => dlg.close() }, label);
  const start = async () => {
    try {
      await api("/v1/cuda/download", { method: "POST" });
    } catch (e) {
      toast(errorText(e.message));
    }
    watch();
  };
  const render = () => {
    const d = cuda?.download;
    if (d?.running) {
      const p = d.progress;
      const fraction = !p ? 0 : p.stage === "unpacking" ? 1 : p.done / p.total;
      const line = !p ? t("cudaStarting") : p.stage === "unpacking" ? t("cudaUnpacking", p.file, p.files) : t("cudaDownloading", p.file, p.files, fmtNum(Math.round(fraction * 100)));
      body.replaceChildren(h("p", {}, line), meterBar(fraction));
      actions.replaceChildren(close(t("close")));
    } else if (cuda?.installed) {
      body.replaceChildren(h("p", {}, t("cudaReady")));
      actions.replaceChildren(
        close(t("close")),
        h("button.btn.btn-ink", { type: "button", onclick: () => { dlg.close(); saveSettings({ device: "cuda" }); } }, t("cudaUseNow")),
      );
    } else {
      body.replaceChildren(h("p", {}, d?.error ? errorText(d.error) : t("cudaPromptText")));
      actions.replaceChildren(close(t("cancel")), h("button.btn.btn-ink", { type: "button", onclick: start }, d?.error ? t("retry") : t("cudaDownload")));
    }
  };
  const watch = async () => {
    await refreshCuda();
    if (!dlg.open) return;
    render();
    if (cuda?.download?.running) setTimeout(watch, 500);
  };
  watch();
  return dlg;
}

/** a bar of the given fraction (0..1) */
function meterBar(fraction) {
  const fill = h("span", {});
  fill.style.width = `${(fraction * 100).toFixed(1)}%`;
  return h("div.meter", { role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": Math.round(fraction * 100) }, fill);
}

function cudaRow() {
  const d = cuda?.download;
  let hint = null;
  let fraction = null;
  if (d?.running) {
    const p = d.progress;
    if (!p) hint = t("cudaStarting");
    else if (p.stage === "unpacking") {
      fraction = 1;
      hint = t("cudaUnpacking", p.file, p.files);
    } else {
      fraction = p.total ? p.done / p.total : 0;
      hint = t("cudaDownloading", p.file, p.files, fmtNum(Math.round(fraction * 100)));
    }
  } else if (d?.error) hint = t("cudaFailed", errorText(d.error));
  else if (cuda?.installed) hint = t("cudaReady");
  else if (cuda) hint = t("cudaMissing", cuda.missing.length);
  let meter = null;
  if (fraction !== null) {
    const fill = h("span", {});
    fill.style.width = `${(fraction * 100).toFixed(1)}%`;
    meter = h("div.meter", { role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": Math.round(fraction * 100) }, fill);
  }
  const button =
    cuda && !cuda.installed && !d?.running
      ? h(
          "button.btn.btn-ink",
          {
            type: "button",
            onclick: async () => {
              try {
                await api("/v1/cuda/download", { method: "POST" });
              } catch (e) {
                toast(errorText(e.message));
              }
              await refreshCuda();
              renderSettings();
            },
          },
          t("cudaDownload"),
        )
      : null;
  return row(t("cudaDl"), hint || meter ? h("div.cuda-status", {}, hint, meter) : null, button);
}

async function refreshCatalog() {
  try {
    const m = await api("/v1/models");
    state.catalog = m.models;
    state.languages = m.languages;
  } catch {}
}

/** Switching the subtitle language also switches to a model that speaks it. */
function setLanguage(language) {
  const fits = (m) => m.languages.includes(language);
  const current = state.catalog.find((m) => m.id === state.snap.settings.asr);
  const patch = { language };
  // the file suffix follows the language unless it was set to something else
  const prev = state.snap.settings;
  if (!prev.lang_ext || prev.lang_ext === (prev.language || "ja")) patch.lang_ext = language;
  if (!current || !fits(current)) {
    const pick = state.catalog.find((m) => fits(m) && m.installed) ?? state.catalog.find(fits);
    if (pick) patch.asr = pick.id;
  }
  saveSettings(patch);
}

/**
 * A customizable native <select> (appearance: base-select): keyboard and
 * screen-reader behavior of the platform, rich options.
 * options: [{ value, label, hint?, meta?, disabled? }]
 */
function select(label, value, options, onchange, cls = "") {
  return h(
    `select.select${cls}`,
    { "aria-label": label, onchange: (e) => onchange(e.target.value) },
    h("button", { type: "button" }, document.createElement("selectedcontent")),
    options.map((o) =>
      h(
        "option",
        { value: o.value, selected: o.value === value, disabled: o.disabled },
        h("span.opt-text", {}, h("span.opt-label", {}, o.label), o.hint ? h("span.opt-hint", {}, o.hint) : null),
        o.meta ? h("span.opt-meta", {}, o.meta) : null,
      ),
    ),
  );
}

/** native name, plus the name in the interface language when it differs */
function languageLabel(code) {
  const native = new Intl.DisplayNames([code], { type: "language" }).of(code);
  const local = new Intl.DisplayNames([lang], { type: "language" }).of(code);
  const cap = (x) => x.charAt(0).toLocaleUpperCase(code) + x.slice(1);
  return native === local ? cap(native) : `${cap(native)} · ${local}`;
}

function seg(name, value, options, onchange) {
  return h(
    "div.seg",
    { role: "radiogroup" },
    options.map(([v, label]) =>
      h("label", {}, h("input", { type: "radio", name, value: v, checked: v === value, onchange: () => onchange(v) }), h("span", {}, label)),
    ),
  );
}

function row(label, hint, control) {
  return h("div.row", {}, h("div.row-text", {}, h("div.row-label", {}, label), hint ? h("div.row-hint", {}, hint) : null), control);
}

/** label above, a full-width control below (for rich dropdowns) */
function stackRow(label, hint, control) {
  return h(
    "div.row.row-stack",
    {},
    h("div.row-text", {}, h("div.row-label", {}, label), hint ? h("div.row-hint", {}, hint) : null),
    control,
  );
}

function group(label, ...rows) {
  return h("section.group", {}, h("h3.group-label", {}, label), h("div.list", {}, rows));
}

function pathControl(value, key) {
  return h(
    "div.path-actions",
    {},
    h(
      "button.btn.btn-text",
      {
        type: "button",
        onclick: async () => {
          const [p] = await host.pick({ folder: true, multiple: false });
          if (p) saveSettings({ [key]: p });
        },
      },
      t("choose"),
    ),
    value && key !== "models_dir"
      ? h("button.icon-btn", { type: "button", "aria-label": t("reset"), title: t("reset"), onclick: () => saveSettings({ [key]: null }) }, svg(ICON.close))
      : null,
  );
}

/** the models for the subtitle language, as rows of the model menu */
function modelChoices() {
  const s = state.snap.settings;
  const fitting = state.catalog.filter((m) => m.languages.includes(s.language || "ja"));
  const list = fitting.length ? fitting : [{ id: s.asr, name: modelName(s.asr), installed: true }];
  return list.map((m) => ({
    value: m.id,
    label: m.name,
    installed: m.installed !== false,
    downloadable: !!m.downloadable,
    download: m.download ?? null,
    size: m.size,
    hint: MODEL_HINTS[m.id] ? t(MODEL_HINTS[m.id]) : null,
  }));
}

let modelMenu = null;
let pollTimer = 0;

/** the model's trigger and its menu: installed models pick, the rest download */
function modelPicker() {
  const id = "model-menu";
  const trigger = h("button.model-trigger", { type: "button", popovertarget: id, "aria-haspopup": "menu" });
  const menu = h("div.menu.model-menu", { id, popover: "auto", role: "menu" });
  trigger.style.setProperty("anchor-name", `--${id}`);
  menu.style.setProperty("position-anchor", `--${id}`);
  modelMenu = { trigger, menu };
  renderModelMenu();
  return [trigger, menu];
}

function renderModelMenu() {
  if (!modelMenu) return;
  const s = state.snap.settings;
  const choices = modelChoices();
  const cur = choices.find((c) => c.value === s.asr);
  modelMenu.trigger.replaceChildren(
    h("span.model-name", {}, cur?.label ?? modelName(s.asr)),
    cur?.size ? h("span.opt-meta", {}, fmtBytes(cur.size)) : null,
  );
  modelMenu.menu.replaceChildren(...choices.map(modelRow));
}

function modelRow(c) {
  const s = state.snap.settings;
  const dl = c.download;
  const downloading = !!dl?.running;
  const pct = dl?.progress?.total ? Math.round((dl.progress.done / dl.progress.total) * 100) : 0;
  let hint;
  if (downloading) hint = t("modelDownloading", pct);
  else if (dl?.error) hint = t("modelDownloadFailed", errorText(dl.error));
  else if (!c.installed) hint = t("notInModels");
  else hint = [c.hint, c.size ? fmtBytes(c.size) : null].filter(Boolean).join(" · ");
  const pick = h(
    "button.menu-item.model-pick",
    {
      type: "button",
      role: "menuitemradio",
      "aria-checked": String(c.value === s.asr),
      disabled: !c.installed || downloading,
      onclick: () => {
        modelMenu.menu.hidePopover();
        saveSettings({ asr: c.value });
      },
    },
    h("span.model-name", {}, c.label),
    hint ? h("span.model-hint", {}, hint) : null,
  );
  let action = null;
  if (downloading) {
    const fill = h("span", {});
    fill.style.width = `${pct}%`;
    action = h("div.meter.model-meter", { role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": pct }, fill);
  } else if (!c.installed && c.downloadable) {
    action = iconBtn(ICON.download, t("modelDownload"), () => downloadModel(c.value));
  }
  return h("div.model-row", {}, pick, action);
}

async function downloadModel(id) {
  try {
    await api(`/v1/models/${id}/download`, { method: "POST" });
  } catch (e) {
    toast(errorText(e.message));
  }
  pollModels();
}

/** GET /v1/models while a download runs: the menu's progress and the rows */
async function pollModels() {
  clearTimeout(pollTimer);
  await refreshCatalog();
  renderModelMenu();
  if (state.catalog.some((m) => m.download?.running)) pollTimer = setTimeout(pollModels, 500);
}

function renderSettings() {
  const s = state.snap.settings;
  const language = s.language || "ja";
  const languages = state.languages.length ? state.languages : [language];

  const apiUrl = state.snap.api;
  const switchRow = (label, hint, key) =>
    row(label, hint, h("input.switch", { type: "checkbox", role: "switch", checked: s[key], "aria-label": label, onchange: (e) => saveSettings({ [key]: e.target.checked }) }));

  $("#settings-body").replaceChildren(
    ...[
      group(
        t("groupAsr"),
        row(
          t("subLanguage"),
          t("subLanguageHint"),
          select(t("subLanguage"), language, languages.map((c) => ({ value: c, label: languageLabel(c) })), setLanguage),
        ),
        stackRow(t("model"), null, modelPicker()),
        row(t("device"), deviceHint(), seg("device", s.device, Object.entries(DEVICES), chooseDevice)),
      ),
      group(
        t("groupSubs"),
        row(t("format"), null, seg("format", s.format, [["srt", "SRT"], ["vtt", "VTT"]], (v) => saveSettings({ format: v }))),
        row(
          t("langExt"),
          h("span", {}, t("fileName"), h("code", {}, `${t("book")}${s.lang_ext ? "." + s.lang_ext : ""}.${s.format}`)),
          h("input.field", {
            type: "text",
            value: s.lang_ext || "",
            placeholder: t("none"),
            maxLength: 8,
            "aria-label": t("langExt"),
            onchange: (e) => saveSettings({ lang_ext: e.target.value.trim() }),
          }),
        ),
        switchRow(t("overwrite"), t("overwriteHint"), "overwrite"),
        switchRow(t("grouping"), t("groupingHint"), "respect_grouping"),
        row(t("outDir"), s.output_dir || t("nextToAudio"), pathControl(s.output_dir, "output_dir")),
      ),
      group(
        t("groupApi"),
        row(
          t("apiUrl"),
          apiUrl ? h("code", {}, apiUrl) : state.snap.api_error,
          apiUrl
            ? h(
                "button.icon-btn",
                {
                  type: "button",
                  "aria-label": t("copy"),
                  title: t("copy"),
                  onclick: async () => {
                    await navigator.clipboard.writeText(apiUrl);
                    toast(t("copied"));
                  },
                },
                svg(ICON.copy),
              )
            : null,
        ),
        row(
          t("port"),
          t("portHint"),
          h("input.field", {
            type: "number",
            min: 1024,
            max: 65535,
            value: s.port,
            "aria-label": t("port"),
            onchange: (e) => saveSettings({ port: Number(e.target.value) || 7865 }),
          }),
        ),
        row(
          t("idle"),
          t("idleHint"),
          seg("idle", String(s.idle_unload), [["0", t("never")], ["15", t("min15")], ["60", t("hour1")]], (v) => saveSettings({ idle_unload: Number(v) })),
        ),
      ),
      group(
        t("groupApp"),
        row(
          t("language"),
          null,
          select(
            t("language"),
            s.lang || "auto",
            [{ value: "auto", label: t("auto") }, ...LANGS.map(([value, label]) => ({ value, label }))],
            (v) => saveSettings({ lang: v }),
          ),
        ),
      ),
      group(
        t("groupFolders"),
        row(t("models"), s.models_dir, pathControl(s.models_dir, "models_dir")),
        row(t("cuda"), s.cuda_dir || t("systemPath"), pathControl(s.cuda_dir, "cuda_dir")),
        cudaRow(),
      ),
      state.snap.restart_needed
        ? h("div.restart", {}, h("p", {}, t("restartNote")), h("button.btn.btn-ink", { type: "button", onclick: () => host.restart() }, t("restart")))
        : null,
    ].filter(Boolean),
  );
}

async function openSettings() {
  await Promise.all([refreshCatalog(), refreshCuda()]);
  renderSettings();
  $("#settings").showModal();
}

// ---------------------------------------------------------------- window + drop

async function wireWindow() {
  if (!tauri) return;
  const win = tauri.window.getCurrentWindow();
  $("#winctl").hidden = false;
  $("#win-min").addEventListener("click", () => win.minimize());
  $("#win-max").addEventListener("click", () => win.toggleMaximize());
  $("#win-close").addEventListener("click", () => win.close());
  $(".bar").addEventListener("dblclick", (e) => {
    if (e.target.closest("button, nav")) return;
    win.toggleMaximize();
  });
}

async function wireDrop() {
  const veil = $("#veil");
  if (tauri) {
    await tauri.webview.getCurrentWebview().onDragDropEvent((e) => {
      const kind = e.payload.type;
      veil.classList.toggle("on", kind === "enter" || kind === "over");
      if (kind === "drop") addPaths(e.payload.paths);
    });
  } else {
    // browser preview: no file paths (add them from the console)
    window.subthxAdd = addPaths;
    addEventListener("dragover", (e) => {
      e.preventDefault();
      veil.classList.add("on");
    });
    addEventListener("dragleave", (e) => e.relatedTarget || veil.classList.remove("on"));
    addEventListener("drop", (e) => {
      e.preventDefault();
      veil.classList.remove("on");
      toast(t("dropInApp"));
    });
  }
}

// ---------------------------------------------------------------- start

function renderAll() {
  doneNodes.clear();
  renderStatic();
  renderEngine();
  render();
}

$("#open-settings").addEventListener("click", openSettings);
$("#engine").addEventListener("click", () => {
  if (state.health && !state.health.model.loaded && !state.loading && !state.loadError) loadModel();
  else openSettings();
});
$("#settings").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) e.currentTarget.close(); // backdrop
});

state.snap = await host.state();
setLang(state.snap.settings.lang);
renderAll();
if (state.snap.api_error) toast(t("serverFailedToast", errorText(state.snap.api_error)));
wireWindow();
wireDrop();
await refreshCatalog();
render(); // model names on finished cards come from the catalog
await pollHealth();
setInterval(pollHealth, 3000);
// keep ETAs moving between progress events
setInterval(() => state.jobs.filter((j) => !finished(j.status.state)).forEach(renderJob), 1000);
// the server may hold another model (another app, an earlier session): match the settings
{
  const m = state.health?.model;
  const s = state.snap.settings;
  if (m && (!m.loaded || m.asr !== s.asr || m.device !== s.device)) loadModel();
}
