# Muse Minimax Director

**Timeline-based director node for MiniMax H3 in ComfyUI**

Built by [Muse Collective](https://musecollective.co.uk) — write a single flowing script broken into CUTs, drop in reference character images/video/audio, and let the node handle chunking, prompt-per-chunk splitting, and reference-tag numbering for you.

![ComfyUI Custom Node](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange?style=flat-square)
![MiniMax H3](https://img.shields.io/badge/MiniMax-H3-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## What it does

MiniMax H3 is a strong omni-modal video model, but its native inputs are low-level: numbered `<Picture N>` / `<Video N>` / `<Audio N>` reference tags that have to line up exactly with iteration order, and a hard ~15 second ceiling per generation. `MuseMinimaxDirector` sits on top of the real, stock ComfyUI MiniMax H3 nodes and handles all of that bookkeeping so you can just write a script.

### Key features

- **Visual timeline editor** — a script made of CUTs (segments) laid out along a single timeline, each with its own prompt
- **Automatic chunking** — write one script of any length; the node splits it into H3-sized chunks itself, and every CUT's text lands in every chunk it actually spans, regardless of how the segment weights divide up
- **Two generation modes**, switchable with a single toggle in the node's Reference Settings:
  - **Reference mode** — up to 9 character reference images, plus reference video and reference audio, combined with soft conditioning
  - **First/Last Frame mode** — Ref 1 and Ref 2 become a locked first frame and last frame instead (other reference slots grey out, since H3 has no tag system in this mode)
- **Hybrid Continuation** (Reference mode only) — an optional toggle that hard-locks chunk-to-chunk transitions using the First/Last Frame checkpoint on the boundary frame, instead of relying on Reference mode's softer carry-over conditioning. Needs a First/Last Frame model wired into the node's `model_fl2va` input to use.
- **Reference video** — motion/style reference, or full person-swap-in-scene editing, with an optional per-clip toggle to also use that video's own embedded audio as a voice/timbre reference (for lipsync-style reperformance)
- **Reference audio** — standalone voice cloning / timbre reference, independent of any video
- **Correct `<Picture N>` / `<Video N>` / `<Audio N>` tag numbering** — built to match MiniMax H3's real assignment rule (iteration order over the reference dict, not slot position), so labels in the compiled prompt always match what H3 actually sees
- **Sigma shift controls** exposed directly on the node
- **Video + audio output** (not a bundled video file) — wire straight into a standard Video Combine node like the rest of your pipeline

---

## Nodes included

| Node | Description |
|------|-------------|
| `MuseMinimaxDirector` | Timeline-based director for MiniMax H3 — the only node in this package |

---

## Requirements

- A recent ComfyUI install with the stock `MiniMaxH3ReferenceToVideo` / `MiniMaxH3ImageToVideo` / `MiniMaxH3SigmaShift` nodes available (these ship with ComfyUI core — no separate node pack needed for the model itself)
- MiniMax H3 model weights, downloaded separately — see [Model setup](#model-setup)

### Python packages

```bash
pip install av
```

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

MiniMax H3 weights are not included in this repository and must be downloaded separately from the official source:

**https://huggingface.co/MiniMaxAI/MiniMax-H3**

You need the checkpoint(s) matching the mode(s) you plan to use:
- The **Reference-to-Video** checkpoint, wired into this node's `model` input — required for Reference mode
- The **Image-to-Video (First/Last Frame)** checkpoint, wired into this node's `model_fl2va` input — required for First/Last Frame mode, and optional (but needed) for the Hybrid Continuation toggle in Reference mode

Follow the official repo's own instructions for which files to download and where to place them, and check ComfyUI's built-in MiniMax H3 example workflow/template (if your ComfyUI version ships one) for the exact folder layout it expects — filenames and folder conventions can change between releases, so this README intentionally doesn't hardcode them.

You'll also need a matching CLIP text encoder and VAE/audio VAE, wired the same way as any other ComfyUI MiniMax H3 workflow.

---

## Node inputs

| Input | Type | Description |
|-------|------|--------------|
| `mode` | Combo | `Reference` or `First/Last Frame` |
| `model` | MODEL | MiniMax H3 Reference-to-Video checkpoint |
| `model_fl2va` | MODEL (optional) | MiniMax H3 First/Last-Frame checkpoint — required for First/Last Frame mode or Hybrid Continuation |
| `clip` | CLIP | MiniMax H3 text encoder |
| `vae` | VAE | MiniMax H3 video VAE |
| `audio_vae` | VAE | MiniMax H3 audio VAE — required in both modes |
| `aspect_ratio` / `megapixels` / `multiple` | — | Resolution controls |
| `duration_seconds` | FLOAT | Total output length — the node chunks automatically if this exceeds one H3 generation |
| `chunk_duration_seconds` | FLOAT | Target length per chunk |
| `hybrid_continuation` | BOOLEAN | Reference mode only — hard-locks chunk boundaries via `model_fl2va` |
| `seed` / `steps` / `sampler_name` / `scheduler` | — | Standard sampler controls |
| `shift_video` / `shift_audio` | — | Sigma shift values, applied via `MiniMaxH3SigmaShift` |
| `timeline_data` | Hidden | Populated by the visual timeline editor UI — not meant to be edited directly |

## Node outputs

| Output | Type | Description |
|--------|------|--------------|
| `images` | IMAGE | Generated video frames — wire into a Video Combine node |
| `audio` | AUDIO | Generated/mixed audio track |
| `compiled_prompt` | STRING | The exact per-chunk prompt(s) sent to H3, including resolved `<Picture N>` / `<Video N>` / `<Audio N>` labels — useful for debugging |

---

## Using the timeline

- **Characters** — up to 9 character reference image slots. Filled slots are packed densely into H3's `<Picture N>` tags in the order they're filled, so gaps in the slot list never create gaps or wrong numbers in the compiled prompt.
- **Location** — a single background/setting reference image.
- **CUTs** — write your script as a sequence of CUT segments along the timeline, each with its own prompt. Weight a CUT to change how much of the total duration it covers. CUTs are mapped into chunks by actual time overlap, so a CUT that spans a chunk boundary appears correctly in both chunks.
- **Reference video / audio** (Reference mode only) — drop in reference video clips and standalone reference audio clips. Each video clip has its own toggle to additionally include that clip's own audio as a voice/timbre reference.

Switching `mode` to **First/Last Frame** repurposes the first two character slots as **First Frame** and **Last Frame** and greys out the rest of the reference UI, since H3's First/Last Frame checkpoint has no reference-tag system to feed.

---

## Attribution

This node is **Powered by MiniMax H3**. MiniMax H3 model weights are a separate download under MiniMax's own [Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) — see [NOTICE](NOTICE). The original code in this repository is MIT licensed — see [LICENSE](LICENSE).
