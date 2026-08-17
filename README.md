# Muse Minimax Director (Seed Hunt)

**Timeline-based director node for MiniMax H3 in ComfyUI, with Seed Hunt scouting**

Built by [Muse Collective](https://musecollective.co.uk) — write a single flowing script broken into CUTs, drop in reference character images/video/audio, and let the node handle chunking, prompt-per-chunk splitting, and reference-tag numbering for you.

This is a fork of [Muse Minimax Director](https://github.com/muse-collective-26/MiniMaxH3-Director) that adds a **Seed Hunt** toggle — scout 4 candidate seeds in one run at low resolution, then pick the best one and refine it at full resolution with the companion [Muse Minimax Refine](https://github.com/muse-collective-26/Muse-MiniMax-H3-Refine) node. The original repo stays as the simpler, single-generation version; this one is for anyone who wants the scouting workflow built in.

![ComfyUI Custom Node](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange?style=flat-square)
![MiniMax H3](https://img.shields.io/badge/MiniMax-H3-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Changelog

### v1.2.0
- **Seed Hunt is now three independent toggles instead of one all-or-nothing switch.** `candidate_2` / `candidate_3` / `candidate_4` each run their own extra pass (or don't) — pick exactly which ones you want to pay for instead of always paying for all 3. `candidate_1` still always runs for free, same as before. Turning all three off runs exactly like a normal single-generation workflow, same as `seed_hunt` off used to. The old `seed_hunt` widget still exists on already-saved workflows but no longer does anything — it's kept only so its saved widget position doesn't shift and corrupt older saved workflows' stored data, not because it's still functional. If your old workflow relied on it, tick the three new toggles instead.
- **New node: `MuseModelRoute`**, bundled in this same repo — the reverse of a switch. Feed it one MODEL and a boolean, and it sends that model to exactly one of two outputs (`on_true` / `on_false`), leaving the other empty. Pairs with this node's `model`/`model_fl2va` inputs so a single boolean can pick which checkpoint (Reference or First/Last-Frame) is actually active, instead of maintaining two separate model chains permanently wired in. See [Switching between Reference and First/Last Frame checkpoints, in detail](#switching-between-reference-and-firstlast-frame-checkpoints-in-detail).
- **`model` no longer forces a checkpoint load it doesn't need.** Previously, the sigma-shift step run on `model` happened unconditionally regardless of `mode` — meaning `model` had to always be a real, connected checkpoint even in First/Last Frame mode using `model_fl2va`. It's now skipped when `model` is empty, which is what makes the `MuseModelRoute` pattern above possible without crashing. Doesn't change anything if you wire `model` normally, as before.
- **Updated example workflow** — now wired with `MuseModelRoute` and a single boolean switch for the Reference/First-Last-Frame model choice, and the three independent candidate toggles in place of the old single `seed_hunt` checkbox.

### v1.1.0
- **Fixed a crash on old saved workflows.** If you'd saved a workflow before this update and it failed to load with an error mentioning `Cannot create property 'characters'`, that's fixed — the node now resets its timeline gracefully instead of blocking the whole workflow from loading. If you hit this, the safest fix is to delete the Muse Minimax Director node from your old workflow and add a fresh one in its place, then re-enter your references/CUTs.
- **Analyze button improvements:** it no longer leaks background/pose/setting details into character descriptions, and it now has its own settings (gear icon) so you can pick which vision provider/model it uses instead of being locked to a hidden default.
- **Reference video:** added a "duplicate video, replace character" retention option for people who want to reuse a reference video's motion/scene but swap in a different character.
- **Updated example workflow** (`workflows/muse_minimax_h3_director_scout_v1.json`) — now wired with **Patch Sol-Attn** as the active speed optimization. Based on direct same-seed testing, Sol-Attn was the only speed node that gave a real, consistent speed-up without any visible quality loss on hands/hair. The other speed nodes (Sage Attention variants, EasyCache, Spectrum) are left in the workflow but bypassed, so you can switch them on to compare for yourself — in our tests they either made no measurable difference or came with a quality cost.

---

## Powered by MiniMax H3

This node is a control layer built entirely on top of **MiniMax H3**, an open-weights omni-modal video generation model published by [MiniMax](https://www.minimax.io/). MiniMax H3 is the actual model doing every bit of the generation work here — this repository contains none of the model's weights, training code, or inference architecture. All of that lives in MiniMax H3 itself and in ComfyUI core's own stock support for it (`comfy_extras/nodes_minimax_h3.py`). What this node adds on top is a visual timeline/scripting layer, automatic chunking for videos longer than a single H3 generation, and correct handling of H3's reference-tag numbering — nothing more.

- **Model page:** [huggingface.co/MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- **License (MiniMax H3 Community License Agreement):** [huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- **Official prompt-writing guide:** [VIDEO_PROMPT_WRITING_GUIDE_ref_en.md](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)

You must download MiniMax H3's weights yourself, directly from the official source above, and you are bound by MiniMax's own license terms for using them — this node does not change, replace, or grant any rights over that license. See [Credits & Licensing](#credits--licensing) below for the full detail, and see [NOTICE](NOTICE) and [LICENSE](LICENSE) in this repository.

**If you build something with this node, please credit MiniMax H3 in your own output/description — "Powered by MiniMax H3" — the same courtesy this README extends to them.**

---

## What it does

MiniMax H3 is a strong omni-modal model, but its native inputs are low-level: numbered `<Picture N>` / `<Video N>` / `<Audio N>` reference tags that have to line up exactly with the order references are actually passed in, and a hard ceiling of roughly 15 seconds per single generation call. `MuseMinimaxDirector` sits on top of the real, stock ComfyUI MiniMax H3 nodes (`MiniMaxH3ReferenceToVideo`, `MiniMaxH3ImageToVideo`, `MiniMaxH3SigmaShift`) and handles all of that bookkeeping for you, so you can write a script instead of hand-managing tag numbers and generation-length limits.

### Key features

- **Visual timeline editor** — a script made of CUTs (segments) laid out along a single timeline, each with its own prompt text
- **Automatic chunking** — write one script of any total length; the node splits it into H3-sized chunks itself. Every CUT's text is included in every chunk it actually overlaps in time, regardless of how many CUTs you use or how their weights divide up the total duration
- **Two generation modes**, switched with a single toggle in the node's Reference Settings box:
  - **Reference mode** — up to 9 character reference images, plus reference video and reference audio, all combined via H3's soft reference conditioning
  - **First/Last Frame mode** — Ref 1 and Ref 2 become a hard-locked first frame and last frame instead. This is a genuine positional lock (frame 0 and the final frame), not conditioning — the other reference slots grey out in this mode because H3's First/Last Frame checkpoint has no reference-tag system to feed them into
- **Hybrid Continuation** (Reference mode only) — an optional toggle that hard-locks chunk-to-chunk transitions by routing the boundary frame through the First/Last Frame checkpoint instead of relying purely on Reference mode's softer carry-over conditioning. Needs a First/Last Frame model wired into the node's `model_fl2va` input
- **Reference video** — motion/style reference, or full "video editing" style person-swap-in-scene, per H3's own documented task types
- **Reference video audio** — a per-clip toggle to also pull that reference video's own embedded audio out and use it as a separate voice/timbre reference — useful for lipsync-style "reperformance," where the same spoken words come out in a different, reference-specified voice
- **Reference audio** — standalone voice cloning / timbre reference, independent of any video, for direct dialogue-in-a-cloned-voice generation
- **Correct `<Picture N>` / `<Video N>` / `<Audio N>` tag numbering** — built to match MiniMax H3's actual assignment rule, which is iteration order over the reference dictionary's values, **not** the numeric suffix of the input slot's own key. Reference items always land in the compiled prompt with the same tag numbers H3 itself will actually assign them
- **MiniMax's full six-section reference-mode prompt format** — `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`, `<Subject N>` abstraction layer, fixed-vocabulary retention markers, `[Shot N] At MM:SS.mmm` shot timestamps, and `(Sx)` speaker tags — built automatically from the timeline UI, per MiniMax's own official prompt-writing guide
- **Multi-speaker CUTs** — pick one or more speaking characters per CUT via chips in the timeline UI; quoted dialogue gets auto-attributed and `(Sx)`-tagged against the right `<Subject N>`, with positional Ref Audio ↔ Ref character voice pairing
- **Seed Hunt** — three independent toggles (`candidate_2`/`candidate_3`/`candidate_4`), each running one extra full pass at identical settings, seed only, and filling its own candidate image/audio pair — pick exactly how many you want to pay for (see [Seed Hunt, in detail](#seed-hunt-in-detail) below)
- **`ref_images_used` output** — the exact reference photos used for `<Picture N>` tagging on this run, ready to wire into [Muse Minimax Refine](https://github.com/muse-collective-26/Muse-MiniMax-H3-Refine)'s own `ref_images` input for identity-locked second-pass refining
- **Sigma shift controls** exposed directly on the node, applied via the real `MiniMaxH3SigmaShift` node
- **Image + audio output**, not a bundled video file — wire straight into a standard Video Combine node alongside the rest of your pipeline
- **`compiled_prompt` output** — the exact, fully-resolved prompt text sent to H3 for every chunk, so you can see precisely what tags and continuity language the node generated

---

## Nodes included

| Node | Description |
|------|-------------|
| `MuseMinimaxDirector` | Timeline-based director for MiniMax H3 |
| `MuseModelRoute` | Reverse of a switch — routes one MODEL to exactly one of two outputs based on a boolean, leaving the other empty. See [Switching between Reference and First/Last Frame checkpoints, in detail](#switching-between-reference-and-firstlast-frame-checkpoints-in-detail) |

---

## Requirements

- A recent ComfyUI install with the stock `MiniMaxH3ReferenceToVideo` / `MiniMaxH3ImageToVideo` / `MiniMaxH3SigmaShift` nodes available (these ship with ComfyUI core — no separate node pack needed for the model support itself, only for this timeline layer). If this node fails to load with `ModuleNotFoundError: No module named 'comfy_extras.nodes_minimax_h3'` in the console, your ComfyUI core build predates native MiniMax H3 support — update ComfyUI core itself (not this node) and restart.
- MiniMax H3 model weights, downloaded separately by you — see [Model setup](#model-setup)

### Python packages

```bash
pip install av
```

`av` is used to decode and trim uploaded reference video/audio clips.

---

## Installation

### Via ComfyUI Manager
Search for **Muse Minimax Director** and click Install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/muse-collective-26/MiniMaxH3-Director-Seed-Hunt
```

Restart ComfyUI after installing.

---

## Model setup

MiniMax H3 weights are **not** included in this repository and must be downloaded separately, directly from the official publisher:

**https://huggingface.co/MiniMaxAI/MiniMax-H3**

Depending on which mode(s) you plan to use, you need:

- The **Reference-to-Video** checkpoint — wired into this node's `model` input. Required for Reference mode.
- The **Image-to-Video (First/Last Frame)** checkpoint — wired into this node's `model_fl2va` input. Required for First/Last Frame mode, and also required if you want to use the Hybrid Continuation toggle while in Reference mode.

These are two separate weight files for two separate H3 checkpoints — they are not interchangeable, and each needs its own `MiniMaxH3SigmaShift` pass, which this node handles internally per-checkpoint.

You will also need a matching CLIP text encoder, a video VAE, and an audio VAE — wired the same way as any other ComfyUI MiniMax H3 workflow. Follow the official Hugging Face repo's own instructions for exact filenames and folder placement, and check ComfyUI's built-in MiniMax H3 example workflow/template (if your ComfyUI version ships one) for the current expected layout — deliberately not hardcoded here, since filenames and folder conventions can and do change between model/ComfyUI releases.

---

## Quick start

1. Add a **Muse Minimax Director** node to your graph.
2. Wire `model` (Reference-to-Video checkpoint), `clip`, `vae`, and `audio_vae`. Optionally wire `model_fl2va` (First/Last-Frame checkpoint) if you plan to use First/Last Frame mode or Hybrid Continuation.
3. On the node's timeline UI, drop a character reference image into **Ref 1**, and optionally a **Location** background image.
4. Add one or more **CUT** segments along the timeline and write a prompt for each.
5. Set `duration_seconds` for your total target length, and `chunk_duration_seconds` for how long each individual H3 generation call should be (stay under H3's own per-call ceiling — roughly 15 seconds).
6. Run. Wire the node's `images` and `audio` outputs into a standard Video Combine node to get a playable file.
7. Check the `compiled_prompt` output text if anything looks off — it shows exactly what was sent to H3, including every resolved `<Picture N>` / `<Video N>` / `<Audio N>` tag.

---

## Example workflow

A ready-to-load workflow is included at [`workflows/muse_minimax_h3_director_scout_v1.json`](workflows/muse_minimax_h3_director_scout_v1.json) — the full Seed Hunt scouting pipeline: model loaders wired up through a `MuseModelRoute` node and a boolean switch (see [Switching between Reference and First/Last Frame checkpoints, in detail](#switching-between-reference-and-firstlast-frame-checkpoints-in-detail)) so one toggle picks Reference vs. First/Last-Frame, sigma-shift patches applied, `MuseMinimaxDirector` with the `candidate_2`/`candidate_3`/`candidate_4` toggles on, all candidates previewed through their own Video Combine nodes, and two [Muse Minimax Refine](https://github.com/muse-collective-26/Muse-MiniMax-H3-Refine) nodes wired up (one candidate-driven, one taking the plain `images`/`audio` output for a single non-scouted run) feeding a final high-resolution output. Reference slots are left empty on purpose so you drop in your own characters and location rather than inheriting someone else's.

It also uses a few extra nodes purely for convenience/performance, on top of what's required above — search ComfyUI Manager for these if they show as missing when you load it:

- **KJNodes** (`ModelPreviewOverrideKJ`) — a model-preview wrapper, purely decorative, safe to delete and rewire straight through if you don't have it
- **ComfyUI-LayerStyle** (`LayerUtility: PurgeVRAM`) — clears VRAM between preview steps, not required for generation to work
- A text-preview node (`iToolsPreviewText`) showing the `compiled_prompt` output — swap for any other STRING preview node (e.g. "Show Text" from ComfyUI-Custom-Scripts) if you don't have this one
- A SageAttention memory-efficiency patch (`MiniMaxH3MemoryEfficientSageAttentionPatch`) applied to the model — optional; the node still runs without it, just less memory-efficient
- **ComfyUI-Nvidia-RTX-Nodes** (`RTXVideoSuperResolution`) — an additional generic upscaler on some of the candidate preview branches, on top of/instead of Muse Minimax Refine's own img2img-style refine; entirely optional, disabled branches (`mode: 4`) by default in the saved workflow
- **[ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3)** (`SpectrumApplyMiniMaxH3`) — a third-party speed optimization that forecasts some sampling steps instead of running the full transformer on them, wired into the model chain at its conservative default preset. **Worth knowing**: its own docs state forecasted steps change the denoising trajectory, so outputs can differ from a fully native run even with an identical seed — if you're doing careful seed-to-seed comparison (which is the whole point of Seed Hunt), consider disabling this node so what you're comparing is genuine seed variation, not forecasting variance. In our own same-seed testing it was also the one config that visibly softened hand/finger detail, so it's left **bypassed** by default in the saved workflow.
- **[ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton)** (`SolAttnPatch`) — the active speed optimization in this example workflow's model chain. This one **isn't on the ComfyUI Registry**, so ComfyUI Manager can't auto-install it from a "missing custom nodes" prompt the way it can for everything else here — use Manager's **Install via Git URL** with the link above instead. In our own same-seed testing, this was the only speed optimization that gave a real, consistent speedup with no visible quality cost.

None of these are needed for `MuseMinimaxDirector` itself to work — only for this specific example graph exactly as saved.

---

## Full node reference

### Inputs

| Input | Type | Description |
|-------|------|--------------|
| `mode` | Combo | `Reference` or `First/Last Frame` — switches which H3 checkpoint and reference system is used |
| `model` | MODEL | MiniMax H3 Reference-to-Video checkpoint |
| `model_fl2va` | MODEL (optional) | MiniMax H3 First/Last-Frame checkpoint — required for First/Last Frame mode, or to enable Hybrid Continuation while in Reference mode |
| `clip` | CLIP | MiniMax H3 text encoder |
| `vae` | VAE | MiniMax H3 video VAE |
| `audio_vae` | VAE | MiniMax H3 audio VAE — required in both modes, since H3's latent is always a joint audio+video structure internally |
| `aspect_ratio` | Combo | Output aspect ratio |
| `megapixels` | FLOAT | Output resolution budget |
| `multiple` | INT | Resolution rounding constraint |
| `resize_method` | Combo | How every character/background reference image and First/Last Frame image gets fit to the output resolution when its own aspect ratio doesn't match. `crop` scales up and center-crops the excess (no distortion, may crop the edges). `pad` scales down to fit and adds black bars (nothing cropped, but the bars become visible reference content). `stretch` resizes directly, distorting proportions — this is what H3 does internally on its own if you don't fit the image yourself, so it's here for parity, not as the recommended choice |
| `duration_seconds` | FLOAT | Total output length. If this is longer than one H3 generation can produce in a single call, the node automatically splits the render into multiple chunks |
| `chunk_duration_seconds` | FLOAT | Target length per chunk when chunking is needed |
| `ref_image_size` | Combo | Resolution reference images are resized to before being sent to H3 |
| `hybrid_continuation` | BOOLEAN | Reference mode only. When on (and `model_fl2va` is connected), chunk boundaries are hard-locked via the First/Last Frame checkpoint instead of soft carry-over conditioning |
| `seed` | INT | Sampler seed for the main run (also the base seed for Seed Hunt's candidates) |
| `seed_hunt` | BOOLEAN | Legacy, hidden from the node's UI. No longer has any effect — kept only so its saved widget position doesn't shift and corrupt older saved workflows' data. Use `candidate_2`/`candidate_3`/`candidate_4` instead |
| `candidate_2` | BOOLEAN | Runs one extra full pass at `seed + 1,000,003` and fills `candidate_2_images`/`candidate_2_audio`. Independent of `candidate_3`/`candidate_4` — turn on only the ones you want to pay for |
| `candidate_3` | BOOLEAN | Runs one extra full pass at `seed + 2,000,006` and fills `candidate_3_images`/`candidate_3_audio`. Independent of `candidate_2`/`candidate_4` |
| `candidate_4` | BOOLEAN | Runs one extra full pass at `seed + 3,000,009` and fills `candidate_4_images`/`candidate_4_audio`. Independent of `candidate_2`/`candidate_3` |
| `steps` | INT | Sampler steps |
| `sampler_name` | Combo | Sampler algorithm |
| `scheduler` | Combo | Noise scheduler |
| `shift_video` | FLOAT | Sigma shift value applied to the video branch via `MiniMaxH3SigmaShift` |
| `shift_audio` | FLOAT | Sigma shift value applied to the audio branch via `MiniMaxH3SigmaShift` |
| `timeline_data` | Hidden | Populated automatically by the visual timeline editor UI — not meant to be edited by hand |

### Outputs

| Output | Type | Description |
|--------|------|--------------|
| `images` | IMAGE | Generated video frames for the main run. **Blocked (not populated) whenever any of `candidate_2`/`candidate_3`/`candidate_4` are on** — scouting extra candidates means this isn't a picked final result, so this only ever means "the one real generation" when none of them are on. Use `candidate_1..4` instead when scouting |
| `audio` | AUDIO | Generated/mixed audio track, same blocking behavior as `images` |
| `compiled_prompt` | STRING | The exact per-chunk prompt(s) actually sent to H3, including every resolved section, tag, and continuity language. The single best debugging tool for this node — if a render doesn't look right, check this first. Always populated |
| `ref_images_used` | IMAGE | The static `<Picture N>` reference image set actually used on this run's first chunk (character/product photos + background, in H3's own tag order). Reference mode only — empty in First/Last Frame mode. Wire into [Muse Minimax Refine](https://github.com/muse-collective-26/Muse-MiniMax-H3-Refine)'s `ref_images` input for identity-locked refining. Always populated (Reference mode) |
| `candidate_1_images` / `candidate_1_audio` | IMAGE / AUDIO | Always mirrors `images`/`audio` (the main run), at zero extra cost — populated regardless of the candidate toggles |
| `candidate_2..4_images` / `candidate_2..4_audio` | IMAGE / AUDIO | Each candidate's own extra scouting pass (seed + N×1,000,003) — only populated when that specific candidate's toggle is on, independent of the others |

`images`/`audio` being blocked when a candidate toggle is on, and an unused `candidate_N` output being empty when its toggle is off, both use ComfyUI's `ExecutionBlocker` — anything wired to an unpopulated output stops silently (a console warning, no red error, no interruption to the rest of the queue) rather than running on the wrong data.

---

## Using the timeline UI

The timeline editor has three parts:

- **Characters** — up to 9 character reference image slots (Ref 1 through Ref 9). Only filled slots get sent to H3, and they are packed densely in fill order — so if you only fill Ref 1 and Ref 3, they still become `<Picture 1>` and `<Picture 2>` in the compiled prompt with no gap, matching exactly how H3 itself will number them. Each slot has its own free-text description field, used to build the compiled prompt's `<Picture N> = ...` label line.
- **Location** — a single background/setting reference image slot, sent as the final `<Picture N>` after all filled character slots.
- **CUTs** — your script, written as a sequence of timed segments along the timeline. Each CUT has its own prompt text and a weight that controls how much of the total duration it covers. CUTs are mapped into chunks by actual time overlap, not by a fixed split — a CUT that spans a chunk boundary is correctly included in both chunks it touches, whatever your segment count or weighting looks like.

Switching `mode` to **First/Last Frame** repurposes the first two character slots as **First Frame** and **Last Frame**, and greys out every other reference slot plus the reference video/audio row, since H3's First/Last Frame checkpoint has no reference-tag system for them to feed.

---

## Reference mode, in detail

Reference mode uses H3's `MiniMaxH3ReferenceToVideo` checkpoint and MiniMax's own full **six-section prompt format** (`subject_definitions` / `summary` / `retention_analysis` / `detailed_description` / `overall_soundscape` / `non_diegetic_music`), per their official prompt-writing guide. This is the mode for character-consistent generation across a full script, and for anything that needs more than one reference source at once — the node assembles all six sections for you from the timeline UI.

**Example** (Ref 1 filled with a photo of a woman with a ponytail, Location filled with a boardwalk photo):

```
subject_definitions:
<Subject 1> is a woman with a dark ponytail, wearing a cream trench coat (from `<Picture 1>`).
<Subject 2> is the setting, a wooden boardwalk beside the sea at golden hour (from `<Picture 2>`).

summary:
[reference generation] <Subject 1> walks along the boardwalk in <Subject 2>.

retention_analysis:
<Subject 1> (present throughout): fully_preserved - matches `<Picture 1>`.
<Subject 2> (present throughout): fully_preserved - matches `<Picture 2>`.

detailed_description:
[Shot 1] <Subject 1> walks slowly along the boardwalk in <Subject 2>, wind moving through her coat, warm low sun behind her.

overall_soundscape:
Gentle waves, distant gulls, a light breeze.

non_diegetic_music:
Soft, warm acoustic guitar, understated.
```

This scaffolding is built automatically from your Characters, Location, and CUT text — you only need to write the CUT prompt itself, plus optional per-slot descriptions/retention and the two global Soundscape/Music fields in Reference Settings.

---

## First/Last Frame mode, in detail

First/Last Frame mode uses H3's `MiniMaxH3ImageToVideo` checkpoint. Unlike Reference mode's soft conditioning, this is a genuine hard positional lock: your Ref 1 image is locked to frame 0, and your Ref 2 image (if present) is locked to the final frame. Either can be left empty to skip that lock.

This mode has no reference-tag system at all — write your CUT prompts as plain descriptive text describing the motion that should happen between the locked first and last frames. It's the right tool when you have a specific start pose and end pose you need the generation to hit exactly, rather than a general character-consistency need across an open-ended script.

---

## Reference video, in detail

Reference video (Reference mode only) lets you hand H3 an actual video clip as a reference, tagged `<Video N>` in the compiled prompt. Per H3's own documented task types, a reference video can be used for:

- **Video editing** — replace a person or element in the reference video while keeping the same scene, camera motion, and action. Pair a character reference image with a reference video of someone else performing the action you want, and describe the swap in your CUT text.
- **Video continuation** — extend the reference video's motion and camera movement onward, rather than starting fresh.
- **Structural / camera-move reference** — carry over the reference video's camera language (a push-in, a pan, a specific framing) into a new scene.

**Example — replacing a person while keeping the scene**, with a character reference image in Ref 1 and a reference video of someone else walking down a corridor dropped into the reference video slot:

```
`<Picture 1>` = a woman with a dark ponytail, wearing a cream trench coat
`<Video 1>` = the corridor scene and camera motion to keep, with the person to be replaced

CUT 1: `<Picture 1>` walks down the corridor from `<Video 1>`, matching the same pace, camera movement, and framing as the original — everything about the scene and camera stays the same, only the person changes.
```

---

## Reference video audio, in detail

Each reference video slot has its own **"Include this clip's audio (as its own reference)"** toggle. Off by default. When switched on, the node pulls that clip's own embedded audio track out and passes it to H3 as a separate, paired `<Audio N>` reference alongside that clip's `<Video N>` tag — this maps directly onto H3's real `ref_video_audios` input, which the stock ComfyUI node exposes but which isn't used unless you explicitly opt in per clip.

This is the mechanism behind a **"reperformance"**: the same spoken words, delivered in a different voice. To do it, combine a reference video (with its audio toggle on) for the words/timing/performance, with a separate standalone reference audio clip (see below) for the target voice, and describe the swap explicitly in your CUT text — H3 needs to be told which `<Audio N>` is the words source and which is the target voice.

Leave the toggle off if you only want the reference video's motion, and don't want its audio influencing the result at all — for example, if you're doing a silent motion/style transfer and don't want any spoken content or ambient sound bleeding through.

---

## Reference audio, in detail

Standalone reference audio clips (Reference mode only, independent of any reference video) are tagged `<Audio N>` and used for voice timbre, style, rhythm, and content reference — most commonly, voice cloning for dialogue.

**Example — simple voice-cloned dialogue**, with a character reference image in Ref 1 and a short clean voice sample dropped into a reference audio slot:

```
`<Picture 1>` = a woman with a dark ponytail
`<Audio 1>` = the voice to use for her dialogue

CUT 1: `<Picture 1>` looks directly at camera and says, in the voice from `<Audio 1>`, "I didn't think you'd actually come."
```

---

## Chunking and long-form scripts, in detail

MiniMax H3 tops out at roughly 15 seconds per single generation call. Set `duration_seconds` above `chunk_duration_seconds` and the node automatically:

1. Splits the total duration into chunks sized to `chunk_duration_seconds`, with the remainder folded into the final chunk rather than producing an oddly short trailing chunk.
2. Maps every CUT you've written into every chunk it overlaps in time, based on actual elapsed-time overlap — not a fixed per-chunk split — so a CUT that straddles a chunk boundary appears, correctly, in both chunks.
3. Carries a short tail of the previous chunk's own audio forward into the next chunk's reference audio, with an automatically-added continuity instruction, so score/ambience doesn't hard-reset at every chunk boundary.
4. In Reference mode, also carries the previous chunk's final frame forward as a continuity-anchor reference image (labelled accordingly in the compiled prompt), with a continuity instruction telling H3 to continue the same action and framing rather than cutting to a new take.

This is soft conditioning by default — H3 has no true hard pixel-lock across separate generation calls in Reference mode, so some visible seam at chunk boundaries is possible even with all of the above. **Hybrid Continuation** (below) is the harder-lock alternative for when that seam matters more than perfect reference-image fidelity on the affected chunk.

---

## Hybrid Continuation, in detail

Available only in Reference mode, and only with a First/Last-Frame checkpoint wired into `model_fl2va`. When switched on, every chunk after the first is generated using the First/Last-Frame checkpoint instead of the Reference-to-Video checkpoint, with the previous chunk's exact final frame locked in as that chunk's first frame — a genuine positional lock, not conditioning.

This trades away some of Reference mode's character-reference reinforcement on the affected chunks (since the First/Last-Frame checkpoint has no `<Picture N>` tag system to keep reinforcing identity from your original reference images) in exchange for eliminating the visible hard-cut seam at chunk boundaries. In practice this trade is often worth it, since the locked anchor frame itself already carries the correct identity forward from whichever chunk was generated in full Reference mode.

The first chunk of a render is always generated in full Reference mode regardless of this toggle, since there is no previous chunk to continue from yet.

---

## Switching between Reference and First/Last Frame checkpoints, in detail

`model` and `model_fl2va` are normally two separate, permanently-wired inputs. If you'd rather flip between Reference mode and First/Last Frame mode with one boolean instead of maintaining two full LoRA/attention-patch chains side by side, the companion `MuseModelRoute` node (bundled in this same repo) lets a single switch pick which checkpoint is actually active.

Wire your one shared LoRA/attention-patch chain's output into `MuseModelRoute`'s `model` input, then wire its two outputs so one feeds `model` and the other feeds `model_fl2va`. In the bundled example workflow, `false` routes the Reference (ref2va) checkpoint to `model`, and `true` routes the First/Last-Frame (fl2va) checkpoint to `model_fl2va`. Only one of the two Director sockets actually receives a real model on any given run — the other stays empty, which is safe because `model_fl2va` has always tolerated being empty, and `model` now does too, whenever it isn't the one currently needed.

**Important**: `MuseModelRoute`'s boolean and the Director node's own `mode` dropdown are two separate controls with nothing keeping them in sync automatically. If you switch one without the other, the node will either sample with the wrong checkpoint for the mode you're in, or — if `model` ends up empty while `mode` is still set to Reference — fail outright. Always change both together.

---

## Multi-speaker CUTs, in detail

Each CUT can have one or more speaking characters selected via chips in the timeline UI. Quoted dialogue in a CUT's text gets `(Sx)` speaker tags auto-attached to the right `<Subject N>` occurrences, with `(Sx)` numbers assigned in order of first appearance within the chunk. Standalone reference audio clips pair positionally with character slots (Ref Audio N ↔ Ref N) for voice-timbre reference, cited against the correct `(Sx)` automatically. With exactly one speaker selected on a CUT, untagged dialogue is auto-attributed to them; with two or more selected, tag the `<Subject N>` you mean directly in the CUT text — there's no safe way to guess which speaker owns an untagged line once more than one is selected.

---

## Seed Hunt, in detail

MiniMax H3 generation is expensive enough that finding out a render didn't follow the prompt, after paying full price for it, is a real cost. Seed Hunt lets you generate several different seeds of the same prompt cheaply, compare them, and only spend real compute refining the one that actually worked.

`candidate_1` always runs — it's the main `seed` run, at no extra cost, exactly as if this feature didn't exist. `candidate_2`, `candidate_3`, and `candidate_4` are three independent toggles; each one you switch on runs one extra full pass at `seed` plus a fixed offset (`+1,000,003` / `+2,000,006` / `+3,000,009`) and fills that candidate's own `candidate_N_images`/`candidate_N_audio` output. Turn on only as many as you actually want to pay for — one toggle costs one extra pass, not four.

With all three off (the default), the node behaves exactly like a plain single-generation run: `images`/`audio` are populated normally with the main result, and `candidate_1_images`/`candidate_1_audio` mirror it for free.

With any of `candidate_2`/`candidate_3`/`candidate_4` on:
- The main `images`/`audio` outputs are **intentionally blocked** — they don't mean "a picked result" while scouting, only during a normal single run, so nothing downstream can mistake an unpicked scout for a finished video.
- Each candidate you turned on gets its own populated `candidate_N` pair; the ones you left off stay empty.
- Render time scales with how many you turned on, not a fixed 4x.

**Recommended workflow**: set `megapixels` low for scouting, turn on however many candidates you actually want to compare, wire `candidate_1..4_images`/`audio` and `ref_images_used` into [Muse Minimax Refine](https://github.com/muse-collective-26/Muse-MiniMax-H3-Refine), pick the candidate that actually matches the prompt via its button selector, and let Refine do the expensive, high-resolution second pass on just that one.

---

## Credits & Licensing

**This project would not exist without MiniMax H3.** Every frame, every second of audio, and every reference-following behaviour this node relies on comes from MiniMax's own model — this repository is a thin, original scripting/timeline layer on top of it, nothing more. If you use this node, please extend MiniMax the same credit this README gives them: **Powered by MiniMax H3.**

- **Model:** MiniMax H3, developed and published by MiniMax — [huggingface.co/MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- **License:** MiniMax H3 Community License Agreement — [full text here](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). This is MiniMax's own license for their model and is separate from, and takes precedence over, anything in this repository regarding the model itself. Read it yourself before using the model — summarised here only for orientation, not as a substitute:
  - Displaying a "Powered by MiniMax H3" credit and adding an AI-generation identifier to output files are both described in the license as **encouraged**, not mandatory.
  - Distributing a NOTICE file alongside any redistribution of software built on MiniMax H3 **is** described as required — see [NOTICE](NOTICE) in this repo.
  - The license sets a revenue threshold above which separate written authorization from MiniMax is required for commercial use, and above which prominent "MiniMax H3" branding on a commercial product's UI becomes mandatory rather than encouraged.
  - The license also contains territory-specific language that may affect redistribution in certain jurisdictions. **Read this section yourself** in the official text linked above before redistributing — it was not something this project's own author verified in full before publishing this node, and it is exactly the kind of clause a summary can get wrong.
- **Official prompt-writing guide** (the source for the reference-tag conventions and task types documented above): [VIDEO_PROMPT_WRITING_GUIDE_ref_en.md](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)

**This repository's own original code** (the timeline UI, the chunking logic, the tag-numbering fix, the hybrid continuation feature — everything in this repo outside of calling MiniMax H3's own stock ComfyUI nodes) is licensed under the **MIT License** — see [LICENSE](LICENSE). The MIT license covers only that original code; it grants no rights whatsoever over the MiniMax H3 model itself, which remains governed entirely by MiniMax's own Community License Agreement linked above.

See [NOTICE](NOTICE) for the required third-party attribution notice.
