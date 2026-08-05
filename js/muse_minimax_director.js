const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

// Muse Minimax Director — timeline UI.
//
// H3 has no chunk/segment concept in its own architecture — the whole clip comes from
// one prompt in one sampling pass. So the ruler/track below is a real duration-proportioned
// visual (drag block edges to change how much of the total clip each CUT gets, drag whole
// blocks to reorder) but it only ever compiles down to prose + a soft "(~Xs)" pacing hint
// per CUT — there's no actual per-segment sampling behind it, unlike the LTX timeline this
// is visually modeled on.
//
// mode/clip/vae/duration_seconds/resolution_preset/ref_image_size are real native widgets
// declared in the Python node — this file re-skins them into boxed sections rather than
// managing separate state for them. Character/background reference images, and reference
// video/audio clips, are all real uploaded files (via ComfyUI's own /upload/image endpoint,
// same as LTX Director's audio/video tracks) with scrub + Set In/Set Out trim controls where
// relevant — file path + trim window live in timeline_data; Python resolves/decodes them
// (with PyAV for video/audio) at execute time. There are no first_frame/last_frame graph
// sockets — First/Last Frame mode reuses Ref 1 / Ref 2 (the same character-slot UI) as its
// first/last frame source instead, with every other reference slot disabled in that mode.

const MAX_CHARACTER_SLOTS = 9;
const REF_AV_SLOTS = 3;
const HIDDEN_WIDGET_NAMES = ["timeline_data"];
const BOXED_WIDGET_NAMES = [
  "mode", "duration_seconds", "chunk_duration_seconds",
  "aspect_ratio", "megapixels", "multiple",
  "steps", "sampler_name", "scheduler", "seed", "control_after_generate", "shift_video", "shift_audio",
  "ref_image_size", "hybrid_continuation",
];
// "seed" uses a text input below (see _seedRow), not the generic number row — avoids
// <input type="number">'s tendency to mangle very large integers into scientific
// notation on display/edit. It's still ultimately a plain JS Number under the hood,
// same as ComfyUI's own native seed widget — this doesn't add or remove precision
// versus what the native widget already has, just a cleaner text-based editor for it.
const MODE_REFERENCE_PREFIX = "Reference (Omni)";

// Fixed-vocabulary retention markers from MiniMax's own reference-mode prompt
// guide (retention_analysis section) — visual markers apply to Subject/Picture/
// Video, audio markers apply to Audio. Values are the literal English tokens the
// guide requires; labels are just the friendlier on-screen text.
const VISUAL_RETENTION_OPTIONS = [
  { value: "fully_preserved", label: "fully preserved" },
  { value: "partially_preserved", label: "partially preserved" },
  { value: "attribute_transfer", label: "attribute transfer" },
  { value: "weak_reference", label: "weak reference" },
];
const AUDIO_RETENTION_OPTIONS = [
  { value: "reference", label: "reference (timbre/style only)" },
  { value: "fully_copy", label: "fully copy" },
  { value: "partially_copy", label: "partially copy" },
  { value: "weak_reference", label: "weak reference" },
];
const VIDEO_ROLE_OPTIONS = [
  { value: "reference", label: "Reference (motion/style/camera)" },
  { value: "editing_source", label: "Editing source (replace something in it)" },
  { value: "continuation_source", label: "Continuation source (extend from it)" },
];

const CUT_COLORS = [
  { bar: "#4F8EF7", glow: "rgba(79,142,247,0.35)" },
  { bar: "#33C481", glow: "rgba(51,196,129,0.35)" },
  { bar: "#F0665B", glow: "rgba(240,102,91,0.35)" },
  { bar: "#B26BF7", glow: "rgba(178,107,247,0.35)" },
  { bar: "#F7B94F", glow: "rgba(247,185,79,0.35)" },
  { bar: "#4FD1F7", glow: "rgba(79,209,247,0.35)" },
];

const ICON_UPLOAD = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>`;
const ICON_TRASH = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>`;
const ICON_DRAG = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><circle cx="8" cy="6" r="1.6"/><circle cx="16" cy="6" r="1.6"/><circle cx="8" cy="12" r="1.6"/><circle cx="16" cy="12" r="1.6"/><circle cx="8" cy="18" r="1.6"/><circle cx="16" cy="18" r="1.6"/></svg>`;
const ICON_PLUS = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>`;
const ICON_CHEV = `<svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`;
const ICON_PLAY = `<svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 21 12 6 21 6 3"/></svg>`;
const ICON_PAUSE = `<svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;

function injectStyles() {
  if (document.getElementById("mmd-styles")) return;
  const style = document.createElement("style");
  style.id = "mmd-styles";
  style.textContent = `
  .mmd-root {
    display: flex; flex-direction: column; gap: 14px;
    background: linear-gradient(180deg, #14141a 0%, #0d0d11 100%);
    border: 1px solid #3a3a48; border-radius: 12px;
    padding: 14px; box-sizing: border-box; width: 100%;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #e4e4ea;
  }
  .mmd-section-title {
    font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: #7a7a8c; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;
  }
  .mmd-section-title .mmd-badge {
    background: #ff4d4d22; color: #ff6b6b; border-radius: 4px; padding: 1px 6px;
    font-size: 9px; letter-spacing: 0.04em;
  }

  /* Boxed settings panel */
  .mmd-boxes-row { display: flex; gap: 10px; flex-wrap: wrap; }
  .mmd-box {
    flex: 1; min-width: 170px; background: #1e1e26; border: 1px solid #3a3a48;
    border-top: 3px solid #3a3a48; border-radius: 10px; padding: 10px 12px; box-sizing: border-box;
  }
  .mmd-box-generation { border-top-color: #4F8EF7; }
  .mmd-box-resolution { border-top-color: #33C481; }
  .mmd-box-sampling { border-top-color: #F0665B; }
  .mmd-box-reference { border-top-color: #B26BF7; }
  .mmd-box-title {
    font-size: 10px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    color: #6a6a7a; margin-bottom: 8px;
  }
  .mmd-box-row {
    display: flex; align-items: center; justify-content: space-between; gap: 8px;
    padding: 5px 0; border-top: 1px solid #22222b;
  }
  .mmd-box-row:first-of-type { border-top: none; }
  .mmd-box-row label { font-size: 10.5px; color: #9a9aa8; white-space: nowrap; }
  .mmd-box-select, .mmd-box-number {
    background: #101015; border: 1px solid #2e2e3a; border-radius: 6px; color: #e4e4ea;
    font-size: 11px; padding: 4px 6px; max-width: 62%; box-sizing: border-box;
  }
  .mmd-box-select option { background: #1a1a22; color: #e4e4ea; font-size: 11px; }
  .mmd-box-checkbox { width: 15px; height: 15px; accent-color: #4F8EF7; cursor: pointer; }
  .mmd-box-select:focus, .mmd-box-number:focus { outline: none; border-color: #4F8EF7; }
  .mmd-mode-pill {
    display: inline-block; font-size: 9px; font-weight: 700; letter-spacing: 0.03em;
    padding: 2px 7px; border-radius: 20px; margin-bottom: 8px;
  }
  .mmd-mode-pill.ref { background: #4F8EF722; color: #4F8EF7; }
  .mmd-mode-pill.fl { background: #F7B94F22; color: #F7B94F; }

  .mmd-style-input {
    width: 100%; box-sizing: border-box; background: #1a1a22; border: 1px solid #2e2e3a;
    border-radius: 8px; color: #e4e4ea; padding: 8px 10px; font-size: 12px; resize: vertical;
    min-height: 36px; font-family: inherit;
  }
  .mmd-style-input:focus { outline: none; border-color: #4F8EF7; }

  /* Timeline / ruler */
  .mmd-ruler-wrap { position: relative; }
  .mmd-ruler {
    position: relative; height: 20px; margin-bottom: 4px; border-bottom: 1px solid #3a3a48;
  }
  .mmd-ruler-tick {
    position: absolute; top: 0; bottom: 0; width: 1px; background: #3a3a48;
  }
  .mmd-ruler-tick-label {
    position: absolute; top: 0; font-size: 8.5px; color: #55556a; transform: translateX(2px);
  }
  .mmd-track {
    display: flex; height: 118px; border-radius: 10px; overflow: hidden; border: 1px solid #3a3a48;
    background: #18181f;
  }
  .mmd-cut-block {
    position: relative; display: flex; flex-direction: column; min-width: 40px;
    border-right: 1px solid #0d0d11; cursor: grab; box-sizing: border-box;
  }
  .mmd-cut-block.mmd-dragging { opacity: 0.35; }
  .mmd-cut-block.mmd-drag-over { box-shadow: inset 3px 0 0 #fff; }
  .mmd-cut-bar { height: 4px; width: 100%; flex-shrink: 0; }
  .mmd-cut-head {
    display: flex; align-items: center; justify-content: space-between; padding: 5px 7px 3px 7px;
    flex-shrink: 0;
  }
  .mmd-cut-label { font-size: 10.5px; font-weight: 700; letter-spacing: 0.02em; white-space: nowrap; }
  .mmd-cut-actions { display: flex; align-items: center; gap: 5px; color: #5a5a6a; flex-shrink: 0; }
  .mmd-cut-actions svg { display: block; }
  .mmd-cut-del { cursor: pointer; }
  .mmd-cut-del:hover { color: #ff6b6b; }
  .mmd-cut-text {
    width: 100%; flex: 1; box-sizing: border-box; background: transparent; border: none; color: #d4d4dc;
    font-size: 10.5px; line-height: 1.4; padding: 0 7px 8px 7px; resize: none; font-family: inherit;
  }
  .mmd-cut-text:focus { outline: none; }
  .mmd-cut-resize {
    position: absolute; top: 0; right: -4px; bottom: 0; width: 8px; cursor: ew-resize; z-index: 4;
  }
  .mmd-cut-resize:hover, .mmd-cut-resize.mmd-active { background: rgba(255,255,255,0.08); }
  .mmd-add-cut {
    width: 44px; flex-shrink: 0; border-left: 1.5px dashed #33333f;
    background: #131318; display: flex; align-items: center; justify-content: center;
    cursor: pointer; color: #5a5a6a; transition: all 0.15s ease;
  }
  .mmd-add-cut:hover { color: #4F8EF7; background: #1e1e26; }
  .mmd-track-hint { font-size: 9.5px; color: #4a4a58; margin-top: 5px; }

  /* References row — grid, not flex-wrap. auto-fit (not auto-fill) is the part that
     matters: auto-fill would reserve invisible empty column tracks when the width
     doesn't divide evenly, leaving a gap instead of the row actually reaching the
     edge; auto-fit collapses those empty tracks so the real columns stretch via 1fr
     to consume 100% of the width, always equal size to each other, at any node size. */
  .mmd-char-row {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(88px, 1fr)); gap: 10px;
  }
  .mmd-char-slot {
    background: #1e1e26; border: 1.5px dashed rgba(255,255,255,0.22); border-radius: 10px;
    display: flex; flex-direction: column; align-items: center; justify-content: flex-start;
    padding: 5px; cursor: pointer; position: relative; transition: all 0.15s ease; box-sizing: border-box;
  }
  .mmd-char-slot:hover { border-color: #4F8EF7; background: #1a1c24; }
  .mmd-char-slot.mmd-filled { border-style: solid; border-color: rgba(255,255,255,0.35); }
  .mmd-char-slot.mmd-bg-slot { border-color: #F7B94F99; }
  .mmd-char-slot.mmd-bg-slot:hover { border-color: #F7B94F; }
  .mmd-char-slot.mmd-char-slot-disabled {
    opacity: 0.35; cursor: not-allowed; pointer-events: none;
  }
  .mmd-char-slot.mmd-char-slot-disabled:hover { border-color: rgba(255,255,255,0.22); background: #1e1e26; }
  .mmd-char-label {
    position: absolute; top: 4px; left: 5px; font-size: 8.5px; font-weight: 700;
    color: #fff; background: rgba(0,0,0,0.55); border-radius: 4px; padding: 1px 4px;
    z-index: 2; pointer-events: none;
  }
  .mmd-char-placeholder { color: #4a4a58; font-size: 9px; text-align: center; margin-top: 20px; line-height: 1.4; }
  .mmd-char-preview { width: 100%; height: 60px; border-radius: 6px; overflow: hidden; background: #0a0a0d; margin-top: 15px; }
  .mmd-char-preview img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .mmd-char-del {
    position: absolute; top: 3px; right: 3px; width: 16px; height: 16px; border-radius: 50%;
    background: rgba(0,0,0,0.6); color: #fff; border: none; display: flex; align-items: center;
    justify-content: center; cursor: pointer; z-index: 3; font-size: 11px; line-height: 1;
  }
  .mmd-char-del:hover { background: #d33; }
  .mmd-analyze-btn {
    margin-top: 4px; width: 100%; background: #232330; border: 1px solid #3a3a48; color: #b8b8c8;
    border-radius: 6px; padding: 3px 0; font-size: 9px; cursor: pointer; transition: all 0.15s ease;
  }
  .mmd-analyze-btn:hover { background: #2a2a38; color: #fff; border-color: #4F8EF7; }
  .mmd-analyze-btn.mmd-loading { opacity: 0.6; pointer-events: none; }
  .mmd-desc-input {
    margin-top: 4px; width: 100%; box-sizing: border-box; background: #101015; border: 1px solid #26262f;
    border-radius: 5px; color: #c8c8d4; font-size: 8.5px; padding: 3px 4px; resize: vertical;
    min-height: 28px; font-family: inherit; line-height: 1.3;
  }
  .mmd-fl-note {
    font-size: 10.5px; color: #8a8a98; background: #1e1e26; border: 1px solid #3a3a48;
    border-radius: 8px; padding: 10px 12px; line-height: 1.5;
  }

  /* Reference video / audio slots */
  .mmd-av-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
  .mmd-av-slot {
    flex: 1; min-width: 220px; background: #1e1e26; border: 1.5px dashed rgba(255,255,255,0.22); border-radius: 10px;
    padding: 8px 10px; box-sizing: border-box; position: relative;
  }
  .mmd-av-slot.mmd-filled { border-style: solid; border-color: rgba(255,255,255,0.35); }
  .mmd-av-slot-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
  .mmd-av-slot-label { font-size: 9.5px; font-weight: 700; letter-spacing: 0.04em; color: #7a7a8c; text-transform: uppercase; }
  .mmd-av-slot-del {
    width: 16px; height: 16px; border-radius: 50%; background: #232330; color: #fff; border: none;
    display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 11px; line-height: 1;
  }
  .mmd-av-slot-del:hover { background: #d33; }
  .mmd-av-placeholder {
    color: #4a4a58; font-size: 9.5px; text-align: center; padding: 14px 0; cursor: pointer;
    display: flex; flex-direction: column; align-items: center; gap: 4px;
  }
  .mmd-av-placeholder:hover { color: #4F8EF7; }
  .mmd-av-media video { width: 100%; max-height: 90px; border-radius: 6px; background: #000; display: block; }
  .mmd-av-canvas { width: 100%; height: 40px; display: block; border-radius: 6px; background: #0a0a0d; }
  .mmd-av-scrub-row { display: flex; align-items: center; gap: 6px; margin: 6px 0 4px 0; }
  .mmd-av-play-btn {
    width: 22px; height: 22px; border-radius: 50%; background: #232330; border: 1px solid #3a3a48;
    color: #fff; display: flex; align-items: center; justify-content: center; cursor: pointer;
    flex-shrink: 0; padding: 0;
  }
  .mmd-av-play-btn:hover { background: #2a2a38; border-color: #4F8EF7; }
  .mmd-av-scrub {
    flex: 1; margin: 0; accent-color: #4F8EF7; cursor: pointer;
  }
  .mmd-av-trim-row { display: flex; align-items: center; gap: 6px; margin-top: 2px; }
  .mmd-av-trim-btn {
    flex: 1; background: #232330; border: 1px solid #3a3a48; color: #b8b8c8; border-radius: 6px;
    padding: 3px 0; font-size: 9px; cursor: pointer;
  }
  .mmd-av-trim-btn:hover { background: #2a2a38; color: #fff; border-color: #4F8EF7; }
  .mmd-av-trim-readout { font-size: 8.5px; color: #5a5a6a; text-align: center; margin-top: 3px; }
  .mmd-av-filename { font-size: 8.5px; color: #5a5a6a; margin-top: 3px; word-break: break-all; }
  .mmd-av-audio-toggle {
    display: flex; align-items: center; gap: 6px; margin-top: 6px; cursor: pointer;
    font-size: 9px; color: #9a9aa8;
  }
  .mmd-audio-pair-note { font-size: 9px; color: #6a9a7a; margin-bottom: 4px; line-height: 1.4; }

  /* Small secondary selectors (retention markers, video role, CUT speaker) — kept
     visually secondary via color/weight, NOT via tiny type — legibility comes
     first. Options list styled explicitly too, since browsers otherwise render
     a plain white system popup regardless of the select's own dark styling. */
  .mmd-mini-row {
    display: flex; align-items: center; justify-content: space-between; gap: 8px;
    margin-top: 6px;
  }
  .mmd-mini-row label { font-size: 11px; color: #8a8a98; }
  .mmd-mini-select {
    background: #101015; border: 1px solid #2e2e3a; border-radius: 5px; color: #d4d4dc;
    font-size: 11.5px; padding: 4px 7px; max-width: 62%; box-sizing: border-box;
  }
  .mmd-mini-select option { background: #1a1a22; color: #e4e4ea; font-size: 11.5px; }

  .mmd-cut-speaker-row {
    display: flex; align-items: center; gap: 6px; padding: 0 7px 8px 7px; flex-wrap: wrap;
  }
  .mmd-cut-speaker-row label { font-size: 11px; color: #8a8a98; white-space: nowrap; }
  .mmd-speaker-chip {
    background: #101015; border: 1px solid #2e2e3a; border-radius: 20px; color: #8a8a98;
    font-size: 10.5px; padding: 3px 10px; cursor: pointer; transition: all 0.15s ease;
  }
  .mmd-speaker-chip:hover { border-color: #4F8EF7; color: #d4d4dc; }
  .mmd-speaker-chip-active {
    background: #4F8EF722; border-color: #4F8EF7; color: #cfe0ff;
  }

  .mmd-lang-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 5px 0; border-top: 1px solid #22222b; }
  .mmd-lang-row label { font-size: 10.5px; color: #9a9aa8; white-space: nowrap; }
  .mmd-lang-input {
    background: #101015; border: 1px solid #2e2e3a; border-radius: 6px; color: #e4e4ea;
    font-size: 10.5px; padding: 4px 6px; max-width: 62%; box-sizing: border-box;
  }
  `;
  document.head.appendChild(style);
}

// Proven-safe widget hider, copied exactly from LTXInfiniteDirector's own
// muse_director_v2.js — an earlier, simpler version of this (just setting
// options.hidden + a bare computeSize override, no vueNodesMode guard, no draw
// override) caused a white-screen crash / "failed to save workflow draft" on
// this ComfyUI frontend. This guarded version is what actually works.
function hideWidget(w) {
  if (!w) return;
  w.hidden = true;
  if (!w.options) w.options = {};
  w.options.hidden = true;

  if (!window.LiteGraph || !window.LiteGraph.vueNodesMode) {
    w.computeSize = () => [0, -4]; // -4 cancels ComfyUI's hardcoded 4px widget padding
    if (!w._hiddenDrawHooked) {
      w._origDraw = w.hasOwnProperty("draw") ? w.draw : undefined;
      w._hiddenDrawHooked = true;
    }
    w.draw = () => {};
  }

  if (w.element) w.element.style.display = "none";
  if (w.callback) w.callback(w.value);
}

// Reference video/audio clips are real uploaded files (not base64, unlike the small
// character portraits) — same ComfyUI /upload/image endpoint LTX Director's own
// audio/video tracks use (it accepts any file type despite the name).
async function uploadRefFile(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("subfolder", "musedirector");
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) throw new Error("Upload failed: " + resp.status);
  const data = await resp.json();
  const subfolder = data.subfolder || "";
  return { file: subfolder ? subfolder + "/" + data.name : data.name, fileName: file.name };
}

function comfyViewUrl(entryFile) {
  const idx = entryFile.lastIndexOf("/");
  const subfolder = idx >= 0 ? entryFile.slice(0, idx) : "";
  const name = idx >= 0 ? entryFile.slice(idx + 1) : entryFile;
  return api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=input&subfolder=${encodeURIComponent(subfolder)}`);
}

async function urlToB64(url) {
  const resp = await fetch(url);
  const blob = await resp.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function extractAudioPeaks(file, numPeaks = 120) {
  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const channelData = audioBuffer.getChannelData(0);
  const peaks = [];
  const step = Math.max(1, Math.floor(channelData.length / numPeaks));
  for (let i = 0; i < numPeaks; i++) {
    let max = 0;
    for (let j = 0; j < step; j++) {
      const val = Math.abs(channelData[i * step + j] || 0);
      if (val > max) max = val;
    }
    peaks.push(max);
  }
  return { peaks, duration: audioBuffer.duration };
}

class MinimaxTimelineEditor {
  constructor(node) {
    this.node = node;
    this.timelineDataWidget = node.widgets.find((w) => w.name === "timeline_data");
    this.realWidgets = {};
    for (const name of BOXED_WIDGET_NAMES) {
      const w = node.widgets.find((x) => x.name === name);
      if (w) this.realWidgets[name] = w;
    }
    this.timeline = this._loadState();
    this.container = document.createElement("div");
    this.container.className = "mmd-root";
    injectStyles();
    this.build();
  }

  _loadState() {
    let parsed = {};
    try {
      parsed = JSON.parse(this.timelineDataWidget?.value || "{}");
    } catch (e) {
      parsed = {};
    }
    if (!Array.isArray(parsed.characters)) parsed.characters = [];
    if (!Array.isArray(parsed.segments) || parsed.segments.length === 0) {
      parsed.segments = [{ prompt: "", weight: 1 }];
    }
    parsed.segments.forEach((s) => { if (!s.weight) s.weight = 1; });
    if (typeof parsed.style_line !== "string") parsed.style_line = "";
    if (typeof parsed.background !== "object" || parsed.background === null) parsed.background = {};
    if (!Array.isArray(parsed.refVideos)) parsed.refVideos = [];
    while (parsed.refVideos.length < REF_AV_SLOTS) parsed.refVideos.push(null);
    if (!Array.isArray(parsed.refAudios)) parsed.refAudios = [];
    while (parsed.refAudios.length < REF_AV_SLOTS) parsed.refAudios.push(null);
    // Six-section MiniMax H3 prompt-format fields — see MiniMax's own reference-mode
    // prompt-writing guide. overall_soundscape/non_diegetic_music are their own two
    // required sections in that format (ambience/physical sound vs audience-only
    // score); dialogue_language feeds every <d>[Language]...</d> dialogue tag.
    if (typeof parsed.overall_soundscape !== "string") parsed.overall_soundscape = "";
    if (typeof parsed.non_diegetic_music !== "string") parsed.non_diegetic_music = "";
    if (typeof parsed.dialogue_language !== "string" || !parsed.dialogue_language) parsed.dialogue_language = "English";
    return parsed;
  }

  commitChanges() {
    if (this.timelineDataWidget) {
      this.timelineDataWidget.value = JSON.stringify(this.timeline);
    }
    this.node.setDirtyCanvas(true, true);
  }

  isReferenceMode() {
    const w = this.realWidgets.mode;
    return !w || String(w.value || "").startsWith(MODE_REFERENCE_PREFIX);
  }

  build() {
    this.container.innerHTML = "";
    this.container.appendChild(this._buildSettingsBoxes());

    const styleTitle = document.createElement("div");
    styleTitle.className = "mmd-section-title";
    styleTitle.textContent = "Overall Style";
    this.container.appendChild(styleTitle);

    const styleHint = document.createElement("div");
    styleHint.className = "mmd-track-hint";
    styleHint.style.marginTop = "-4px";
    styleHint.textContent = "One line describing the look of the whole clip (lighting, palette, camera feel) — prepended before the reference declarations and every CUT.";
    this.container.appendChild(styleHint);

    const styleInput = document.createElement("textarea");
    styleInput.className = "mmd-style-input";
    styleInput.placeholder = "e.g. Photorealistic, warm golden-hour light, cinematic shallow depth of field...";
    styleInput.value = this.timeline.style_line || "";
    styleInput.addEventListener("input", () => {
      this.timeline.style_line = styleInput.value;
      this.commitChanges();
    });
    this.container.appendChild(styleInput);

    this.soundTitle = document.createElement("div");
    this.soundTitle.className = "mmd-section-title";
    this.soundTitle.textContent = "Soundscape & Music";
    this.container.appendChild(this.soundTitle);

    this.soundHint = document.createElement("div");
    this.soundHint.className = "mmd-track-hint";
    this.soundHint.style.marginTop = "-4px";
    this.soundHint.textContent = "Ambience/physical sound across the whole clip, and any background score only the audience can hear — Reference mode only, kept separate the way MiniMax's own prompt format expects.";
    this.container.appendChild(this.soundHint);

    this.soundRow = document.createElement("div");
    this.soundRow.className = "mmd-boxes-row";

    const soundscapeBox = document.createElement("div");
    soundscapeBox.className = "mmd-box mmd-box-generation";
    const soundscapeTitle = document.createElement("div");
    soundscapeTitle.className = "mmd-box-title";
    soundscapeTitle.textContent = "Overall Soundscape";
    soundscapeBox.appendChild(soundscapeTitle);
    const soundscapeInput = document.createElement("textarea");
    soundscapeInput.className = "mmd-style-input";
    soundscapeInput.style.minHeight = "30px";
    soundscapeInput.placeholder = "e.g. Quiet indoor room tone and a low ventilation hum continue throughout.";
    soundscapeInput.value = this.timeline.overall_soundscape || "";
    soundscapeInput.addEventListener("input", () => {
      this.timeline.overall_soundscape = soundscapeInput.value;
      this.commitChanges();
    });
    soundscapeBox.appendChild(soundscapeInput);
    this.soundRow.appendChild(soundscapeBox);

    const musicBox = document.createElement("div");
    musicBox.className = "mmd-box mmd-box-generation";
    const musicTitle = document.createElement("div");
    musicTitle.className = "mmd-box-title";
    musicTitle.textContent = "Non-Diegetic Music";
    musicBox.appendChild(musicTitle);
    const musicInput = document.createElement("textarea");
    musicInput.className = "mmd-style-input";
    musicInput.style.minHeight = "30px";
    musicInput.placeholder = "e.g. A restrained solo-piano score at a slow tempo. Leave blank or type N/A for no music.";
    musicInput.value = this.timeline.non_diegetic_music || "";
    musicInput.addEventListener("input", () => {
      this.timeline.non_diegetic_music = musicInput.value;
      this.commitChanges();
    });
    musicBox.appendChild(musicInput);
    this.soundRow.appendChild(musicBox);

    this.container.appendChild(this.soundRow);

    const cutsTitle = document.createElement("div");
    cutsTitle.className = "mmd-section-title";
    cutsTitle.innerHTML = `Timeline <span class="mmd-badge">drag edges to time, drag blocks to reorder</span>`;
    this.container.appendChild(cutsTitle);

    this.rulerWrap = document.createElement("div");
    this.rulerWrap.className = "mmd-ruler-wrap";
    this.container.appendChild(this.rulerWrap);

    const hint = document.createElement("div");
    hint.className = "mmd-track-hint";
    hint.textContent = "H3 generates the whole clip in one pass — block widths are a pacing guide only, compiled into each CUT's own \"(~Xs)\" hint.";
    this.container.appendChild(hint);

    this.refsTitle = document.createElement("div");
    this.refsTitle.className = "mmd-section-title";
    this.refsTitle.style.marginTop = "4px";
    this.refsTitle.textContent = "References";
    this.container.appendChild(this.refsTitle);

    this.refsBox = document.createElement("div");
    this.refsBox.className = "mmd-box mmd-box-reference";
    this.refsArea = document.createElement("div");
    this.refsBox.appendChild(this.refsArea);
    this.container.appendChild(this.refsBox);

    this.renderTimeline();
    this.renderReferences();
    this._toggleReferenceBox();
  }

  // ── Boxed settings panel (re-skins the real native widgets) ────────────────
  _buildSettingsBoxes() {
    const row = document.createElement("div");
    row.className = "mmd-boxes-row";

    row.appendChild(this._buildGenerationBox());
    row.appendChild(this._buildResolutionBox());
    row.appendChild(this._buildSamplingBox());
    row.appendChild(this._buildReferenceBox());
    return row;
  }

  _buildGenerationBox() {
    const box = document.createElement("div");
    box.className = "mmd-box mmd-box-generation";

    const title = document.createElement("div");
    title.className = "mmd-box-title";
    title.textContent = "Generation";
    box.appendChild(title);

    const modeW = this.realWidgets.mode;
    if (modeW) {
      const pill = document.createElement("div");
      pill.className = "mmd-mode-pill " + (this.isReferenceMode() ? "ref" : "fl");
      pill.textContent = this.isReferenceMode() ? "Reference (Omni)" : "First / Last Frame";
      box.appendChild(pill);

      box.appendChild(this._selectRow("Mode", modeW, () => {
        pill.className = "mmd-mode-pill " + (this.isReferenceMode() ? "ref" : "fl");
        pill.textContent = this.isReferenceMode() ? "Reference (Omni)" : "First / Last Frame";
        this.renderReferences();
        this._toggleReferenceBox();
      }));
    }
    if (this.realWidgets.duration_seconds) {
      box.appendChild(this._numberRow("Total Duration (s)", this.realWidgets.duration_seconds, () => this.renderTimeline()));
    }
    if (this.realWidgets.chunk_duration_seconds) {
      box.appendChild(this._numberRow("Chunk Size (s)", this.realWidgets.chunk_duration_seconds, () => this.renderTimeline()));
    }
    return box;
  }

  _buildResolutionBox() {
    const box = document.createElement("div");
    box.className = "mmd-box mmd-box-resolution";

    const title = document.createElement("div");
    title.className = "mmd-box-title";
    title.textContent = "Resolution";
    box.appendChild(title);

    if (this.realWidgets.aspect_ratio) box.appendChild(this._selectRow("Aspect Ratio", this.realWidgets.aspect_ratio));
    if (this.realWidgets.megapixels) box.appendChild(this._numberRow("Megapixels", this.realWidgets.megapixels));
    if (this.realWidgets.multiple) box.appendChild(this._numberRow("Multiple Of", this.realWidgets.multiple));
    return box;
  }

  _buildSamplingBox() {
    const box = document.createElement("div");
    box.className = "mmd-box mmd-box-sampling";

    const title = document.createElement("div");
    title.className = "mmd-box-title";
    title.textContent = "Sampling";
    box.appendChild(title);

    if (this.realWidgets.sampler_name) box.appendChild(this._selectRow("Sampler", this.realWidgets.sampler_name));
    if (this.realWidgets.scheduler) box.appendChild(this._selectRow("Scheduler", this.realWidgets.scheduler));
    if (this.realWidgets.steps) box.appendChild(this._numberRow("Steps", this.realWidgets.steps));
    if (this.realWidgets.seed) box.appendChild(this._seedRow(this.realWidgets.seed));
    if (this.realWidgets.control_after_generate) box.appendChild(this._selectRow("After Generate", this.realWidgets.control_after_generate));
    if (this.realWidgets.shift_video) box.appendChild(this._numberRow("Shift (video)", this.realWidgets.shift_video));
    if (this.realWidgets.shift_audio) box.appendChild(this._numberRow("Shift (audio)", this.realWidgets.shift_audio));

    return box;
  }

  _buildReferenceBox() {
    this.refBox = document.createElement("div");
    this.refBox.className = "mmd-box mmd-box-reference";

    const title = document.createElement("div");
    title.className = "mmd-box-title";
    title.textContent = "Reference Settings";
    this.refBox.appendChild(title);

    if (this.realWidgets.ref_image_size) {
      this.refBox.appendChild(this._selectRow("Ref Image Size", this.realWidgets.ref_image_size));
    }
    this.refBox.appendChild(this._dialogueLanguageRow());
    if (this.realWidgets.hybrid_continuation) {
      this.refBox.appendChild(this._boolRow("Hybrid Continuation", this.realWidgets.hybrid_continuation));
      const hint = document.createElement("div");
      hint.className = "mmd-track-hint";
      hint.textContent = "Continuation chunks hard-lock to the previous chunk's last frame via the separate First/Last-Frame checkpoint, instead of the soft ref_video/ref_audio reference. Needs model_fl2va connected on the node itself.";
      this.refBox.appendChild(hint);
    }
    this._toggleReferenceBox();
    return this.refBox;
  }

  _toggleReferenceBox() {
    const show = this.isReferenceMode() ? "" : "none";
    if (this.refBox) this.refBox.style.display = this.isReferenceMode() ? "block" : "none";
    // Soundscape/Music are only meaningful in Reference mode — H3's First/Last
    // Frame checkpoint never encodes reference audio at all, so there's no
    // overall_soundscape/non_diegetic_music section to fill for that mode.
    if (this.soundTitle) this.soundTitle.style.display = show;
    if (this.soundHint) this.soundHint.style.display = show;
    if (this.soundRow) this.soundRow.style.display = show === "" ? "flex" : "none";
  }

  _selectRow(labelText, widget, onChange) {
    const rowEl = document.createElement("div");
    rowEl.className = "mmd-box-row";
    const label = document.createElement("label");
    label.textContent = labelText;
    rowEl.appendChild(label);

    const select = document.createElement("select");
    select.className = "mmd-box-select";
    let opts = widget.options?.values;
    if (!Array.isArray(opts)) opts = Array.isArray(widget.options) ? widget.options : [];
    (opts || []).forEach((v) => {
      const o = document.createElement("option");
      o.value = v; o.textContent = v;
      if (v === widget.value) o.selected = true;
      select.appendChild(o);
    });
    select.addEventListener("change", () => {
      widget.value = select.value;
      if (widget.callback) widget.callback(select.value);
      this.node.setDirtyCanvas(true, true);
      if (onChange) onChange();
    });
    rowEl.appendChild(select);
    return rowEl;
  }

  _numberRow(labelText, widget, onChange) {
    const rowEl = document.createElement("div");
    rowEl.className = "mmd-box-row";
    const label = document.createElement("label");
    label.textContent = labelText;
    rowEl.appendChild(label);

    const input = document.createElement("input");
    input.type = "number";
    input.className = "mmd-box-number";
    input.min = widget.options?.min ?? 1;
    input.max = widget.options?.max ?? 15;
    input.step = widget.options?.step ?? 0.5;
    input.value = widget.value;
    input.addEventListener("change", () => {
      widget.value = parseFloat(input.value);
      if (widget.callback) widget.callback(widget.value);
      this.node.setDirtyCanvas(true, true);
      if (onChange) onChange();
    });
    rowEl.appendChild(input);
    return rowEl;
  }

  _boolRow(labelText, widget, onChange) {
    const rowEl = document.createElement("div");
    rowEl.className = "mmd-box-row";
    const label = document.createElement("label");
    label.textContent = labelText;
    rowEl.appendChild(label);

    const input = document.createElement("input");
    input.type = "checkbox";
    input.className = "mmd-box-checkbox";
    input.checked = !!widget.value;
    input.addEventListener("change", () => {
      widget.value = input.checked;
      if (widget.callback) widget.callback(widget.value);
      this.node.setDirtyCanvas(true, true);
      if (onChange) onChange();
    });
    rowEl.appendChild(input);
    return rowEl;
  }

  // Small secondary select bound directly to a timeline_data object field (not a
  // real ComfyUI widget) — used for retention markers and video role, which only
  // exist in timeline_data, not as node inputs.
  _miniSelectRow(labelText, currentValue, options, onChange) {
    const rowEl = document.createElement("div");
    rowEl.className = "mmd-mini-row";
    rowEl.addEventListener("click", (e) => e.stopPropagation());
    const label = document.createElement("label");
    label.textContent = labelText;
    rowEl.appendChild(label);

    const select = document.createElement("select");
    select.className = "mmd-mini-select";
    options.forEach((opt) => {
      const o = document.createElement("option");
      o.value = opt.value;
      o.textContent = opt.label;
      if (opt.value === currentValue) o.selected = true;
      select.appendChild(o);
    });
    select.addEventListener("change", () => {
      onChange(select.value);
      this.commitChanges();
    });
    rowEl.appendChild(select);
    return rowEl;
  }

  // Feeds every <d>[Language]...</d> dialogue tag in the compiled prompt — lives in
  // timeline_data, not a real widget, same as style_line.
  _dialogueLanguageRow() {
    const rowEl = document.createElement("div");
    rowEl.className = "mmd-lang-row";
    const label = document.createElement("label");
    label.textContent = "Dialogue Language";
    rowEl.appendChild(label);

    const input = document.createElement("input");
    input.type = "text";
    input.className = "mmd-lang-input";
    input.value = this.timeline.dialogue_language || "English";
    input.addEventListener("input", () => {
      this.timeline.dialogue_language = input.value || "English";
      this.commitChanges();
    });
    rowEl.appendChild(input);
    return rowEl;
  }

  _seedRow(widget) {
    const rowEl = document.createElement("div");
    rowEl.className = "mmd-box-row";
    const label = document.createElement("label");
    label.textContent = "Seed";
    rowEl.appendChild(label);

    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = "numeric";
    input.pattern = "[0-9]*";
    input.className = "mmd-box-number";
    input.value = String(widget.value);
    input.addEventListener("change", () => {
      const digits = input.value.replace(/[^0-9]/g, "");
      input.value = digits || "0";
      widget.value = Number(digits || "0");
      if (widget.callback) widget.callback(widget.value);
      this.node.setDirtyCanvas(true, true);
    });
    rowEl.appendChild(input);
    return rowEl;
  }

  // ── Timeline / ruler ─────────────────────────────────────────────────────
  get durationSeconds() {
    return this.realWidgets.duration_seconds ? Number(this.realWidgets.duration_seconds.value) || 10 : 10;
  }

  renderTimeline() {
    this.rulerWrap.innerHTML = "";

    const ruler = document.createElement("div");
    ruler.className = "mmd-ruler";
    const dur = this.durationSeconds;
    const tickCount = Math.min(15, Math.max(4, Math.round(dur)));
    for (let t = 0; t <= tickCount; t++) {
      const pct = (t / tickCount) * 100;
      const tick = document.createElement("div");
      tick.className = "mmd-ruler-tick";
      tick.style.left = pct + "%";
      ruler.appendChild(tick);
      const lbl = document.createElement("div");
      lbl.className = "mmd-ruler-tick-label";
      lbl.style.left = pct + "%";
      lbl.textContent = ((t / tickCount) * dur).toFixed(1) + "s";
      ruler.appendChild(lbl);
    }
    this.rulerWrap.appendChild(ruler);

    const track = document.createElement("div");
    track.className = "mmd-track";
    this.track = track;
    this.rulerWrap.appendChild(track);

    const totalWeight = this.timeline.segments.reduce((s, seg) => s + (seg.weight || 1), 0) || 1;
    this.timeline.segments.forEach((seg, i) => {
      track.appendChild(this._buildCutBlock(seg, i, totalWeight));
    });

    const addBtn = document.createElement("div");
    addBtn.className = "mmd-add-cut";
    addBtn.innerHTML = ICON_PLUS;
    addBtn.title = "Add CUT";
    addBtn.addEventListener("click", () => {
      this.timeline.segments.push({ prompt: "", weight: 1 });
      this.commitChanges();
      this.renderTimeline();
    });
    track.appendChild(addBtn);
  }

  // Multi-select, not single — a CUT is one shot, but a shot can have more than
  // one person talking in it (a whole 5s clip is often one continuous shot with
  // two or three characters in it, not one cut per line of dialogue). Whichever
  // Ref chips are ticked here each get their own (Sx) tag wherever their own
  // <Subject N> appears in this CUT's text, and each keeps their own paired
  // voice (Ref Audio N <-> Ref N by position) automatically.
  _buildCutSpeakerRow(seg) {
    if (!Array.isArray(seg.speakerCharIdxs)) seg.speakerCharIdxs = [];
    const rowEl = document.createElement("div");
    rowEl.className = "mmd-cut-speaker-row";
    const label = document.createElement("label");
    label.textContent = "Who's speaking:";
    rowEl.appendChild(label);

    for (let i = 0; i < MAX_CHARACTER_SLOTS; i++) {
      const ch = this.timeline.characters[i];
      if (!ch || !(ch.file || ch.image_b64)) continue;
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "mmd-speaker-chip" + (seg.speakerCharIdxs.includes(i) ? " mmd-speaker-chip-active" : "");
      chip.textContent = `Ref ${i + 1}`;
      chip.addEventListener("click", (e) => {
        e.stopPropagation();
        const idx = seg.speakerCharIdxs.indexOf(i);
        if (idx === -1) seg.speakerCharIdxs.push(i);
        else seg.speakerCharIdxs.splice(idx, 1);
        chip.classList.toggle("mmd-speaker-chip-active");
        this.commitChanges();
      });
      rowEl.appendChild(chip);
    }
    return rowEl;
  }

  _buildCutBlock(seg, i, totalWeight) {
    const color = CUT_COLORS[i % CUT_COLORS.length];
    const dur = this.durationSeconds;
    const seconds = ((seg.weight || 1) / totalWeight) * dur;
    seg.duration_hint = seconds.toFixed(1);

    const block = document.createElement("div");
    block.className = "mmd-cut-block";
    block.draggable = true;
    block.style.flex = `${seg.weight || 1} 0 0`;

    const bar = document.createElement("div");
    bar.className = "mmd-cut-bar";
    bar.style.background = color.bar;
    block.appendChild(bar);

    const head = document.createElement("div");
    head.className = "mmd-cut-head";
    const label = document.createElement("div");
    label.className = "mmd-cut-label";
    label.style.color = color.bar;
    label.textContent = `CUT ${i + 1} · ~${seconds.toFixed(1)}s`;
    head.appendChild(label);

    const actions = document.createElement("div");
    actions.className = "mmd-cut-actions";
    const dragHandle = document.createElement("span");
    dragHandle.innerHTML = ICON_DRAG;
    actions.appendChild(dragHandle);
    if (this.timeline.segments.length > 1) {
      const del = document.createElement("span");
      del.className = "mmd-cut-del";
      del.innerHTML = ICON_TRASH;
      del.title = "Delete CUT";
      del.addEventListener("click", () => {
        this.timeline.segments.splice(i, 1);
        this.commitChanges();
        this.renderTimeline();
      });
      actions.appendChild(del);
    }
    head.appendChild(actions);
    block.appendChild(head);

    const text = document.createElement("textarea");
    text.className = "mmd-cut-text";
    text.placeholder = "What happens in this shot — action, dialogue, camera move...";
    text.value = seg.prompt || "";
    block.appendChild(text);

    // Speaker tagging only matters in Reference mode (it drives the (Sx) tag next
    // to that character's <Subject N> mentions) and only once this CUT actually
    // has quoted dialogue to attribute — appears/disappears live as you type
    // rather than needing a full re-render, so it never steals textarea focus.
    const speakerRow = this.isReferenceMode() ? this._buildCutSpeakerRow(seg) : null;
    if (speakerRow) {
      speakerRow.style.display = (seg.prompt || "").includes('"') ? "flex" : "none";
      block.appendChild(speakerRow);
    }
    text.addEventListener("input", () => {
      seg.prompt = text.value;
      this.commitChanges();
      if (speakerRow) speakerRow.style.display = text.value.includes('"') ? "flex" : "none";
    });

    // Resize handle (adjusts this block's weight vs. its right neighbor's)
    if (i < this.timeline.segments.length - 1) {
      const handle = document.createElement("div");
      handle.className = "mmd-cut-resize";
      handle.addEventListener("mousedown", (e) => this._startResize(e, i));
      block.appendChild(handle);
    }

    // Drag-to-reorder (whole block)
    block.addEventListener("dragstart", (e) => {
      block.classList.add("mmd-dragging");
      e.dataTransfer.setData("text/plain", String(i));
      e.dataTransfer.effectAllowed = "move";
    });
    block.addEventListener("dragend", () => block.classList.remove("mmd-dragging"));
    block.addEventListener("dragover", (e) => { e.preventDefault(); block.classList.add("mmd-drag-over"); });
    block.addEventListener("dragleave", () => block.classList.remove("mmd-drag-over"));
    block.addEventListener("drop", (e) => {
      e.preventDefault();
      block.classList.remove("mmd-drag-over");
      const fromIdx = parseInt(e.dataTransfer.getData("text/plain"), 10);
      if (Number.isNaN(fromIdx) || fromIdx === i) return;
      const [moved] = this.timeline.segments.splice(fromIdx, 1);
      this.timeline.segments.splice(i, 0, moved);
      this.commitChanges();
      this.renderTimeline();
    });

    return block;
  }

  _startResize(e, i) {
    e.preventDefault();
    e.stopPropagation();
    const trackRect = this.track.getBoundingClientRect();
    const segA = this.timeline.segments[i];
    const segB = this.timeline.segments[i + 1];
    const totalWeight = this.timeline.segments.reduce((s, seg) => s + (seg.weight || 1), 0);
    const pairWeight = (segA.weight || 1) + (segB.weight || 1);
    const startX = e.clientX;
    const startAWeight = segA.weight || 1;

    const onMove = (ev) => {
      const deltaPx = ev.clientX - startX;
      const deltaWeight = (deltaPx / trackRect.width) * totalWeight;
      let newA = startAWeight + deltaWeight;
      newA = Math.max(0.15, Math.min(pairWeight - 0.15, newA));
      segA.weight = newA;
      segB.weight = pairWeight - newA;
      this.renderTimeline();
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      this.commitChanges();
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  // ── Character / background reference slots ─────────────────────────────────
  renderReferences() {
    this.refsArea.innerHTML = "";
    this.refsTitle.style.display = "flex";

    const isFL = !this.isReferenceMode();

    if (isFL) {
      const note = document.createElement("div");
      note.className = "mmd-fl-note";
      note.textContent = "First/Last Frame mode — Ref 1 is used as the first frame, Ref 2 as the last frame (leave either empty to skip it). Other reference slots and reference video/audio don't apply in this mode.";
      this.refsArea.appendChild(note);
    }

    const charRow = document.createElement("div");
    charRow.className = "mmd-char-row";
    for (let i = 0; i < MAX_CHARACTER_SLOTS; i++) {
      const labelOverride = isFL ? (i === 0 ? "First Frame" : i === 1 ? "Last Frame" : null) : null;
      const disabled = isFL && i > 1;
      charRow.appendChild(this._buildCharSlot(i, disabled, labelOverride));
    }
    charRow.appendChild(this._buildCharSlot("bg", isFL));
    this.refsArea.appendChild(charRow);

    if (isFL) return; // no reference video/audio in First/Last Frame mode

    const avHint = document.createElement("div");
    avHint.className = "mmd-track-hint";
    avHint.style.marginTop = "10px";
    avHint.textContent = "Reference video / audio — upload a clip, skim it with the scrub bar, then Set In / Set Out to pick the exact window that gets sent as a reference. Video slot 1 is reserved internally for chunk-to-chunk continuity once a chunk has a predecessor. A video's own audio is off by default (motion only) — tick \"Include this clip's audio\" to also send it as a paired reference (e.g. for reperforming its dialogue in a different voice). Leave a video's \"Subject description\" blank for a pure motion/camera reference, or fill it in if the video shows a person/element you want reused. Ref Audio N automatically becomes Ref N's voice (Ref Audio 1 = Ref 1's voice, Ref Audio 2 = Ref 2's, etc.) — no need to describe whose voice it is, and no need to mention it in your CUT text. Just tick which characters are speaking on each CUT below and their paired voice is used automatically.";
    this.refsArea.appendChild(avHint);

    const videoRow = document.createElement("div");
    videoRow.className = "mmd-av-row";
    for (let i = 0; i < REF_AV_SLOTS; i++) videoRow.appendChild(this._buildRefAvSlot("video", i));
    this.refsArea.appendChild(videoRow);

    const audioRow = document.createElement("div");
    audioRow.className = "mmd-av-row";
    for (let i = 0; i < REF_AV_SLOTS; i++) audioRow.appendChild(this._buildRefAvSlot("audio", i));
    this.refsArea.appendChild(audioRow);
  }

  // ── Reference video / audio slots (upload + scrub + trim) ──────────────────
  _buildRefAvSlot(kind, idx) {
    const isVideo = kind === "video";
    const listKey = isVideo ? "refVideos" : "refAudios";
    const entry = this.timeline[listKey][idx];

    const slot = document.createElement("div");
    slot.className = "mmd-av-slot" + (entry ? " mmd-filled" : "");

    const head = document.createElement("div");
    head.className = "mmd-av-slot-head";
    const label = document.createElement("div");
    label.className = "mmd-av-slot-label";
    label.textContent = `${isVideo ? "Ref Video" : "Ref Audio"} ${idx + 1}`;
    head.appendChild(label);
    if (entry) {
      const del = document.createElement("button");
      del.className = "mmd-av-slot-del";
      del.innerHTML = "&times;";
      del.title = "Remove";
      del.addEventListener("click", () => {
        this.timeline[listKey][idx] = null;
        this.commitChanges();
        this.renderReferences();
      });
      head.appendChild(del);
    }
    slot.appendChild(head);

    if (!entry) {
      const placeholder = document.createElement("div");
      placeholder.className = "mmd-av-placeholder";
      placeholder.innerHTML = `${ICON_UPLOAD}<span>Drop or click to upload ${isVideo ? "video" : "audio"}</span>`;
      placeholder.addEventListener("click", () => this._promptAvFilePick(kind, idx));
      slot.appendChild(placeholder);
    } else {
      slot.appendChild(this._buildAvMedia(kind, idx, entry));
    }

    slot.addEventListener("dragover", (e) => { e.preventDefault(); slot.style.borderColor = "#4F8EF7"; });
    slot.addEventListener("dragleave", () => { slot.style.borderColor = ""; });
    slot.addEventListener("drop", async (e) => {
      e.preventDefault();
      slot.style.borderColor = "";
      const file = e.dataTransfer.files?.[0];
      if (file) await this._setAvSlot(kind, idx, file);
    });

    return slot;
  }

  _promptAvFilePick(kind, idx) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = kind === "video" ? "video/*" : "audio/*";
    input.onchange = async () => {
      if (input.files?.[0]) await this._setAvSlot(kind, idx, input.files[0]);
    };
    input.click();
  }

  async _setAvSlot(kind, idx, file) {
    const listKey = kind === "video" ? "refVideos" : "refAudios";
    try {
      const uploaded = await uploadRefFile(file);
      const entry = { ...uploaded, trimStartSec: 0, trimEndSec: null, sourceDurationSec: null };
      if (kind === "video") {
        entry.includeAudio = false;
      }
      if (kind === "audio") {
        try {
          const { peaks, duration } = await extractAudioPeaks(file);
          entry.waveformPeaks = peaks;
          entry.sourceDurationSec = duration;
          entry.trimEndSec = duration;
        } catch (err) {
          console.warn("[MuseMinimaxDirector] waveform peak extraction failed", err);
        }
      }
      entry._blobUrl = URL.createObjectURL(file);
      this.timeline[listKey][idx] = entry;
      this.commitChanges();
      this.renderReferences();
    } catch (err) {
      console.error("[MuseMinimaxDirector] reference upload failed", err);
      alert("Upload failed — see console for details.");
    }
  }

  _buildAvMedia(kind, idx, entry) {
    const listKey = kind === "video" ? "refVideos" : "refAudios";
    const wrap = document.createElement("div");
    wrap.className = "mmd-av-media";

    // _blobUrl only lives for the page session that created it via
    // URL.createObjectURL — it goes dead after any reload, but the string itself
    // gets saved into timeline_data and reloaded right along with it. entry.file
    // (the real uploaded server path) is always durable, so it must win once it
    // exists; _blobUrl is only a same-session fast path for brand new uploads.
    const src = entry.file ? comfyViewUrl(entry.file) : entry._blobUrl;
    let mediaEl;
    if (kind === "video") {
      mediaEl = document.createElement("video");
      mediaEl.src = src;
      mediaEl.playsInline = true;
      mediaEl.controls = false;
      wrap.appendChild(mediaEl);
    } else {
      mediaEl = document.createElement("audio");
      mediaEl.src = src;
      mediaEl.style.display = "none";
      wrap.appendChild(mediaEl);

      const canvas = document.createElement("canvas");
      canvas.className = "mmd-av-canvas";
      canvas.width = 260;
      canvas.height = 40;
      wrap.appendChild(canvas);
      requestAnimationFrame(() => this._drawWaveform(canvas, entry));
    }

    // The <audio>/<video> element's own duration (surfaced via loadedmetadata) is the
    // authoritative one for playback/scrubbing purposes — some containers (e.g. certain
    // FLAC files) report a misleadingly short duration from decodeAudioData, which is
    // only used above for waveform peaks. Always sync to the media element once it's
    // known, but don't clobber a trim point the user already set deliberately.
    const finalizeDuration = () => {
      if (!mediaEl.duration || !isFinite(mediaEl.duration)) return;
      const trimWasAtOldEnd = entry.trimEndSec === null || entry.trimEndSec === entry.sourceDurationSec;
      entry.sourceDurationSec = mediaEl.duration;
      if (trimWasAtOldEnd) entry.trimEndSec = mediaEl.duration;
      scrub.max = String(mediaEl.duration);
      this.commitChanges();
      this._updateAvReadout(readout, entry);
    };
    mediaEl.addEventListener("loadedmetadata", finalizeDuration);

    const scrubRow = document.createElement("div");
    scrubRow.className = "mmd-av-scrub-row";

    const playBtn = document.createElement("button");
    playBtn.className = "mmd-av-play-btn";
    playBtn.innerHTML = ICON_PLAY;
    playBtn.title = "Play/Pause";
    playBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (mediaEl.paused) mediaEl.play(); else mediaEl.pause();
    });
    mediaEl.addEventListener("play", () => { playBtn.innerHTML = ICON_PAUSE; });
    mediaEl.addEventListener("pause", () => { playBtn.innerHTML = ICON_PLAY; });
    mediaEl.addEventListener("ended", () => { playBtn.innerHTML = ICON_PLAY; });
    scrubRow.appendChild(playBtn);

    const scrub = document.createElement("input");
    scrub.type = "range";
    scrub.className = "mmd-av-scrub";
    scrub.min = "0";
    scrub.max = String(entry.sourceDurationSec || 100);
    scrub.step = "0.05";
    scrub.value = "0";
    scrub.addEventListener("input", () => {
      mediaEl.currentTime = parseFloat(scrub.value);
    });
    mediaEl.addEventListener("timeupdate", () => {
      scrub.value = String(mediaEl.currentTime);
    });
    scrubRow.appendChild(scrub);
    wrap.appendChild(scrubRow);

    const trimRow = document.createElement("div");
    trimRow.className = "mmd-av-trim-row";
    const setInBtn = document.createElement("button");
    setInBtn.className = "mmd-av-trim-btn";
    setInBtn.textContent = "Set In";
    setInBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      entry.trimStartSec = Math.min(mediaEl.currentTime, (entry.trimEndSec ?? mediaEl.duration ?? mediaEl.currentTime + 1) - 0.05);
      entry.trimStartSec = Math.max(0, entry.trimStartSec);
      this.commitChanges();
      this._updateAvReadout(readout, entry);
    });
    const setOutBtn = document.createElement("button");
    setOutBtn.className = "mmd-av-trim-btn";
    setOutBtn.textContent = "Set Out";
    setOutBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      entry.trimEndSec = Math.max(mediaEl.currentTime, entry.trimStartSec + 0.05);
      this.commitChanges();
      this._updateAvReadout(readout, entry);
    });
    trimRow.appendChild(setInBtn);
    trimRow.appendChild(setOutBtn);
    wrap.appendChild(trimRow);

    const readout = document.createElement("div");
    readout.className = "mmd-av-trim-readout";
    wrap.appendChild(readout);
    this._updateAvReadout(readout, entry);

    const filename = document.createElement("div");
    filename.className = "mmd-av-filename";
    filename.textContent = entry.fileName || "";
    wrap.appendChild(filename);

    if (kind === "audio") {
      // Ref Audio N auto-pairs with Ref (character) N by position — Ref Audio 1 is
      // always Ref 1's voice, Ref Audio 2 is always Ref 2's, etc. No need to
      // describe whose voice it is when that's already known from the pairing;
      // the description field below only matters as a fallback when there's no
      // character in the matching Ref slot (e.g. an off-screen voice).
      const pairedChar = this.timeline.characters[idx];
      const pairedFilled = pairedChar && (pairedChar.file || pairedChar.image_b64);
      const pairNote = document.createElement("div");
      pairNote.className = "mmd-audio-pair-note";
      pairNote.textContent = pairedFilled
        ? `Paired with Ref ${idx + 1} — this is automatically her/his voice reference.`
        : `No character in Ref ${idx + 1} — describe the voice below, or leave blank for an unattributed reference.`;
      wrap.appendChild(pairNote);

      const descInput = document.createElement("textarea");
      descInput.className = "mmd-desc-input";
      descInput.placeholder = "optional — only used when there's no matching character, e.g. \"Sarah — warm, mid-range\"";
      descInput.value = entry.description || "";
      descInput.addEventListener("click", (e) => e.stopPropagation());
      descInput.addEventListener("input", () => {
        entry.description = descInput.value;
        this.commitChanges();
      });
      wrap.appendChild(descInput);

      wrap.appendChild(this._miniSelectRow(
        "Retention", entry.retention || "reference", AUDIO_RETENTION_OPTIONS,
        (v) => { entry.retention = v; },
      ));
    }

    if (kind === "video") {
      const audioToggleRow = document.createElement("label");
      audioToggleRow.className = "mmd-av-audio-toggle";
      audioToggleRow.addEventListener("click", (e) => e.stopPropagation());

      const audioToggle = document.createElement("input");
      audioToggle.type = "checkbox";
      audioToggle.className = "mmd-box-checkbox";
      audioToggle.checked = !!entry.includeAudio;
      audioToggle.addEventListener("change", () => {
        entry.includeAudio = audioToggle.checked;
        this.commitChanges();
      });
      audioToggleRow.appendChild(audioToggle);

      const audioToggleLabel = document.createElement("span");
      audioToggleLabel.textContent = "Include this clip's audio (as its own reference)";
      audioToggleRow.appendChild(audioToggleLabel);

      wrap.appendChild(audioToggleRow);

      const subjectDescInput = document.createElement("textarea");
      subjectDescInput.className = "mmd-desc-input";
      subjectDescInput.placeholder = "if this video shows a person/element to reuse, describe them here (creates a Subject) — leave blank for a pure motion/camera reference";
      subjectDescInput.value = entry.description || "";
      subjectDescInput.addEventListener("click", (e) => e.stopPropagation());
      subjectDescInput.addEventListener("input", () => {
        entry.description = subjectDescInput.value;
        this.commitChanges();
      });
      wrap.appendChild(subjectDescInput);

      wrap.appendChild(this._miniSelectRow(
        "Role", entry.role || "reference", VIDEO_ROLE_OPTIONS,
        (v) => { entry.role = v; },
      ));
      wrap.appendChild(this._miniSelectRow(
        "Retention", entry.retention || (entry.description ? "fully_preserved" : "weak_reference"), VISUAL_RETENTION_OPTIONS,
        (v) => { entry.retention = v; },
      ));
    }

    return wrap;
  }

  _updateAvReadout(readoutEl, entry) {
    const inSec = (entry.trimStartSec || 0).toFixed(2);
    const outSec = entry.trimEndSec !== null && entry.trimEndSec !== undefined ? entry.trimEndSec.toFixed(2) : "end";
    readoutEl.textContent = `In ${inSec}s — Out ${outSec}s`;
  }

  _drawWaveform(canvas, entry) {
    const peaks = entry.waveformPeaks;
    if (!peaks || !peaks.length) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#4F8EF7";
    const barW = w / peaks.length;
    for (let i = 0; i < peaks.length; i++) {
      const amp = Math.max(1, peaks[i] * h * 0.9);
      ctx.fillRect(i * barW, (h - amp) / 2, Math.max(1, barW - 1), amp);
    }
  }

  _buildCharSlot(idx, disabled = false, labelOverride = null) {
    const isBg = idx === "bg";
    const data = isBg ? this.timeline.background : this.timeline.characters[idx];
    const filled = data && (data.file || data.image_b64);

    const slot = document.createElement("div");
    slot.className = "mmd-char-slot" + (filled ? " mmd-filled" : "") + (isBg ? " mmd-bg-slot" : "")
      + (disabled ? " mmd-char-slot-disabled" : "");

    const label = document.createElement("div");
    label.className = "mmd-char-label";
    label.textContent = labelOverride || (isBg ? "Location" : `Ref ${idx + 1}`);
    slot.appendChild(label);

    if (disabled) {
      const note = document.createElement("div");
      note.className = "mmd-char-placeholder";
      note.textContent = "Not used in this mode";
      slot.appendChild(note);
      return slot;
    }

    if (filled) {
      const del = document.createElement("button");
      del.className = "mmd-char-del";
      del.innerHTML = "&times;";
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        if (isBg) this.timeline.background = {};
        else this.timeline.characters[idx] = null;
        this.commitChanges();
        this.renderReferences();
      });
      slot.appendChild(del);

      const preview = document.createElement("div");
      preview.className = "mmd-char-preview";
      const img = document.createElement("img");
      // See the same comment on the ref video/audio media src above — a dead
      // _blobUrl from a previous page session must never win over the durable
      // uploaded file path once one exists.
      img.src = data.file ? comfyViewUrl(data.file) : (data._blobUrl || data.image_b64);
      preview.appendChild(img);
      slot.appendChild(preview);

      const analyzeBtn = document.createElement("button");
      analyzeBtn.className = "mmd-analyze-btn";
      analyzeBtn.textContent = data.description ? "Re-Analyze" : "Analyze";
      analyzeBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        await this.runAnalysis(idx, isBg, analyzeBtn, descInput);
      });
      slot.appendChild(analyzeBtn);

      var descInput = document.createElement("textarea");
      descInput.className = "mmd-desc-input";
      descInput.placeholder = "description...";
      descInput.value = data.description || "";
      descInput.addEventListener("click", (e) => e.stopPropagation());
      descInput.addEventListener("input", () => {
        data.description = descInput.value;
        this.commitChanges();
      });
      slot.appendChild(descInput);

      if (this.isReferenceMode()) {
        slot.appendChild(this._miniSelectRow(
          "Retention", data.retention || "fully_preserved", VISUAL_RETENTION_OPTIONS,
          (v) => { data.retention = v; },
        ));
      }
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "mmd-char-placeholder";
      placeholder.innerHTML = `${ICON_UPLOAD}<br>Drop image`;
      slot.appendChild(placeholder);
      slot.addEventListener("click", () => this._promptFilePick(idx, isBg));
    }

    slot.addEventListener("dragover", (e) => { e.preventDefault(); slot.style.borderColor = "#4F8EF7"; });
    slot.addEventListener("dragleave", () => { slot.style.borderColor = ""; });
    slot.addEventListener("drop", async (e) => {
      e.preventDefault();
      slot.style.borderColor = "";
      const file = e.dataTransfer.files?.[0];
      if (file) await this._setSlotImage(idx, isBg, file);
    });

    return slot;
  }

  _promptFilePick(idx, isBg) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.onchange = async () => {
      if (input.files?.[0]) await this._setSlotImage(idx, isBg, input.files[0]);
    };
    input.click();
  }

  async _setSlotImage(idx, isBg, file) {
    // Uploaded (like ref_video/ref_audio), not base64-embedded — a base64 image
    // inline in timeline_data can push a single workflow draft past the browser's
    // localStorage per-entry autosave budget (~750KB), which silently evicts the
    // draft and loses any unsaved edits every time you leave and return to the
    // workflow. A tiny file-path string avoids that entirely.
    try {
      const uploaded = await uploadRefFile(file);
      const entry = { ...uploaded, description: "", _blobUrl: URL.createObjectURL(file) };
      if (isBg) this.timeline.background = entry;
      else this.timeline.characters[idx] = entry;
      this.commitChanges();
      this.renderReferences();
    } catch (err) {
      console.error("[MuseMinimaxDirector] reference image upload failed", err);
      alert("Image upload failed — see console for details.");
    }
  }

  async _resolveImageB64(data) {
    if (data.image_b64) return data.image_b64;
    const src = data.file ? comfyViewUrl(data.file) : data._blobUrl;
    if (!src) throw new Error("No image source available for this reference.");
    return await urlToB64(src);
  }

  async runAnalysis(idx, isBg, btn, descInput) {
    if (btn.classList.contains("mmd-loading")) return;
    btn.classList.add("mmd-loading");
    btn.textContent = "Analyzing...";
    const data = isBg ? this.timeline.background : this.timeline.characters[idx];
    try {
      const imageB64 = await this._resolveImageB64(data);
      // Self-contained route owned by this package (no dependency on any other
      // Muse package) — tuned to produce <Subject N>-style sentences directly.
      const resp = await api.fetchApi("/muse_minimax_director/analyze_character", {
        method: "POST",
        body: JSON.stringify({
          image_b64: [imageB64],
          char_index: isBg ? MAX_CHARACTER_SLOTS : idx,
          provider: "ollama",
        }),
      });
      const result = await resp.json();
      if (result.status === "success") {
        data.description = result.description;
        descInput.value = result.description;
        btn.textContent = "Success!";
        this.commitChanges();
        setTimeout(() => { btn.classList.remove("mmd-loading"); btn.textContent = "Re-Analyze"; }, 1200);
      } else {
        alert("Analysis error: " + result.message);
        btn.classList.remove("mmd-loading");
        btn.textContent = "Analyze";
      }
    } catch (err) {
      console.error("[MuseMinimaxDirector] analysis request failed", err);
      alert("Analyze request failed — is ComfyUI running, and is Ollama (or your chosen provider) reachable?");
      btn.classList.remove("mmd-loading");
      btn.textContent = "Analyze";
    }
  }
}

app.registerExtension({
  name: "Muse.MinimaxDirector",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MuseMinimaxDirector") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

      // Hide the raw JSON widget, and the native copies of the widgets we re-skin
      // into boxed sections — the DOM editor below is the real UI for all of them.
      for (const w of this.widgets || []) {
        if (HIDDEN_WIDGET_NAMES.includes(w.name) || BOXED_WIDGET_NAMES.includes(w.name)) {
          hideWidget(w);
        }
      }

      this._museMinimaxEditor = new MinimaxTimelineEditor(this);
      this.addDOMWidget("mmd_timeline_ui", "mmd_timeline_ui", this._museMinimaxEditor.container, {
        serialize: false,
        hideOnZoom: false,
      });

      this.size = [560, 640];
      return r;
    };

    // onNodeCreated fires before ComfyUI applies a loaded workflow's saved
    // widgets_values onto the widgets — so building the DOM editor's in-memory
    // state there (in onNodeCreated) only ever sees timeline_data's default "{}",
    // never the real saved characters/segments/images. Without this hook, the
    // editor would silently keep showing (and re-saving) that empty default even
    // though the actual saved data was intact in the widget/file the whole time.
    // onConfigure fires after the real widgets_values has been applied, so
    // re-syncing here is what actually picks up the loaded data.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      if (this._museMinimaxEditor) {
        this._museMinimaxEditor.timeline = this._museMinimaxEditor._loadState();
        this._museMinimaxEditor.build();
      }
      return r;
    };
  },
});
