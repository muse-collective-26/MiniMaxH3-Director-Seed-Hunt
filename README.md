# Muse Minimax Director

**Timeline-based director node for MiniMax H3 in ComfyUI**

Built by [Muse Collective](https://musecollective.co.uk) — write a single flowing script broken into CUTs, drop in reference character images/video/audio, and let the node handle chunking, prompt-per-chunk splitting, and reference-tag numbering for you.

![ComfyUI Custom Node](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange?style=flat-square)
![MiniMax H3](https://img.shields.io/badge/MiniMax-H3-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

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
- **Sigma shift controls** exposed directly on the node, applied via the real `MiniMaxH3SigmaShift` node
- **Image + audio output**, not a bundled video file — wire straight into a standard Video Combine node alongside the rest of your pipeline
- **`compiled_prompt` output** — the exact, fully-resolved prompt text sent to H3 for every chunk, so you can see precisely what tags and continuity language the node generated

---

## Nodes included

| Node | Description |
|------|-------------|
| `MuseMinimaxDirector` | Timeline-based director for MiniMax H3 — the only node in this package |

---

## Requirements

- A recent ComfyUI install with the stock `MiniMaxH3ReferenceToVideo` / `MiniMaxH3ImageToVideo` / `MiniMaxH3SigmaShift` nodes available (these ship with ComfyUI core — no separate node pack needed for the model support itself, only for this timeline layer)
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
git clone https://github.com/muse-collective-26/MiniMaxH3-Director
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
| `duration_seconds` | FLOAT | Total output length. If this is longer than one H3 generation can produce in a single call, the node automatically splits the render into multiple chunks |
| `chunk_duration_seconds` | FLOAT | Target length per chunk when chunking is needed |
| `ref_image_size` | Combo | Resolution reference images are resized to before being sent to H3 |
| `hybrid_continuation` | BOOLEAN | Reference mode only. When on (and `model_fl2va` is connected), chunk boundaries are hard-locked via the First/Last Frame checkpoint instead of soft carry-over conditioning |
| `seed` | INT | Sampler seed |
| `steps` | INT | Sampler steps |
| `sampler_name` | Combo | Sampler algorithm |
| `scheduler` | Combo | Noise scheduler |
| `shift_video` | FLOAT | Sigma shift value applied to the video branch via `MiniMaxH3SigmaShift` |
| `shift_audio` | FLOAT | Sigma shift value applied to the audio branch via `MiniMaxH3SigmaShift` |
| `timeline_data` | Hidden | Populated automatically by the visual timeline editor UI — not meant to be edited by hand |

### Outputs

| Output | Type | Description |
|--------|------|--------------|
| `images` | IMAGE | Generated video frames — wire into a Video Combine node |
| `audio` | AUDIO | Generated/mixed audio track — wire into the same Video Combine node |
| `compiled_prompt` | STRING | The exact per-chunk prompt(s) actually sent to H3, including every resolved `<Picture N>` / `<Video N>` / `<Audio N>` label and any auto-generated continuity language. The single best debugging tool for this node — if a render doesn't look right, check this first |

---

## Using the timeline UI

The timeline editor has three parts:

- **Characters** — up to 9 character reference image slots (Ref 1 through Ref 9). Only filled slots get sent to H3, and they are packed densely in fill order — so if you only fill Ref 1 and Ref 3, they still become `<Picture 1>` and `<Picture 2>` in the compiled prompt with no gap, matching exactly how H3 itself will number them. Each slot has its own free-text description field, used to build the compiled prompt's `<Picture N> = ...` label line.
- **Location** — a single background/setting reference image slot, sent as the final `<Picture N>` after all filled character slots.
- **CUTs** — your script, written as a sequence of timed segments along the timeline. Each CUT has its own prompt text and a weight that controls how much of the total duration it covers. CUTs are mapped into chunks by actual time overlap, not by a fixed split — a CUT that spans a chunk boundary is correctly included in both chunks it touches, whatever your segment count or weighting looks like.

Switching `mode` to **First/Last Frame** repurposes the first two character slots as **First Frame** and **Last Frame**, and greys out every other reference slot plus the reference video/audio row, since H3's First/Last Frame checkpoint has no reference-tag system for them to feed.

---

## Reference mode, in detail

Reference mode uses H3's `MiniMaxH3ReferenceToVideo` checkpoint and its `<Picture N>` / `<Video N>` / `<Audio N>` tag system. This is the mode for character-consistent generation across a full script, and for anything that needs more than one reference source at once.

**Example — a simple character-reference prompt** (Ref 1 filled with a photo of a woman with a ponytail, Location filled with a boardwalk photo):

```
`<Picture 1>` = a woman with a dark ponytail, wearing a cream trench coat
`<Picture 2>` = the setting (a wooden boardwalk beside the sea at golden hour)

CUT 1: The woman from `<Picture 1>` walks slowly along the boardwalk from `<Picture 2>`, wind moving through her coat, warm low sun behind her.
```

This label/tag scaffolding is built automatically from your Characters and Location slots and their description fields — you only need to write the CUT text itself.

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
