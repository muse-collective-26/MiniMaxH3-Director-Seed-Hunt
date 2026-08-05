"""
Muse Minimax Director — an all-in-one timeline-driven Director node for MiniMax H3,
matching MuseDirectorSamplerV10Final's own philosophy: take the real models as inputs
(so they stay fully swappable upstream — this node never loads a checkpoint itself),
run the whole pipeline internally, hand back ready-to-use frames + audio. Outputs raw
IMAGE + AUDIO (not the newer VIDEO type) so it wires straight into a standard Video
Combine node, same as the rest of this project's other Director workflows.

Internally this chains exactly the nodes the official H3 reference template uses,
verified against the real source (comfy_extras/nodes_minimax_h3.py) rather than
guessed: conditioning via MiniMaxH3ReferenceToVideo or MiniMaxH3ImageToVideo depending
on mode, MiniMaxH3SigmaShift on the model, RandomNoise -> BasicGuider -> KSamplerSelect
-> BasicScheduler -> SamplerCustomAdvanced -> VAEDecode + VAEDecodeAudio.

Two real, confirmed facts worth being explicit about:
  - There is no negative/CFG conditioning anywhere in H3's own reference pipeline — it
    uses BasicGuider, which only ever takes one conditioning input. That's not a gap
    in this node, it's how the model is actually set up to run.
  - Both conditioning nodes build a joint audio+video latent internally regardless of
    mode, so audio_vae is required for final decode in both Reference and First/Last
    Frame mode, even though First/Last Frame mode never encodes reference audio.

Resolution matches the real ResolutionSelector node exactly (aspect_ratio x megapixels,
rounded to `multiple`) rather than a fixed table, since H3 documents six real supported
aspect ratios, not just 16:9.

CHUNKING: H3 is only trained/reliable up to ~15s per call (its own node source flags
longer as untested). Same situation the LTX Director's own chunking system existed to
solve — that was never native to LTX either, it was Muse's own orchestration on top.
Ported here the same way: split the requested total duration into chunk_duration_seconds
pieces (the final chunk absorbs whatever's left over, so it may be shorter — chunks are
NOT evenly redistributed, so a 10s chunk size actually gives 10s chunks, not some other
number), run one real H3 call per chunk, and carry continuity from chunk to chunk.

In Reference mode, continuity is two-pronged, both confirmed against the real
MiniMaxH3ReferenceToVideo node source rather than assumed:
  - Video: `ref_videos` accepts up to 3 reference *clips*, not just a still frame, so
    continuation chunks feed the previous chunk's own output back in as a reference
    video (slot 0, reserved) alongside any character/background images.
  - Audio: the previous chunk's own decoded audio (last ~4s) is fed back in too, via
    `ref_audios` slot 0 — otherwise each chunk invents its own score/ambience from
    scratch with zero awareness of what the previous chunk sounded like, producing an
    audible hard reset between chunks (observed directly in an early test render).
Both carried-over references get an explicit instruction appended to the prompt telling
the model to continue seamlessly (no cut, no new piece of music) rather than treating
the carry-over clips as just more generic reference material — a bare reference without
that instruction doesn't reliably read as "keep going from here" on its own.

First/Last Frame mode doesn't have a reference-clip mechanism at all, so its
continuation falls back to the previous chunk's last decoded frame as the next chunk's
first_frame, same idea as LTX's carry-frame (no equivalent audio carry-over exists for
this mode either — that model variant never encodes reference audio at all).

CUT blocks in the timeline are a prompt-authoring convenience (H3 has no chunk concept
in its own single-call architecture) — but they now double as the chunk-bucketing input:
each CUT's proportional position along the total duration decides which real chunk call
it gets compiled into, based on its own start time crossing a chunk boundary.
"""

import base64
import io as _io
import json
import logging
import math
import os

import av
import folder_paths
import numpy as np
import torch
from PIL import Image

from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ReferenceToVideo, MiniMaxH3ImageToVideo, MiniMaxH3SigmaShift, align_frame_count,
)
from comfy_extras.nodes_resolution import AspectRatio, ASPECT_RATIOS

log = logging.getLogger(__name__)

MODE_REFERENCE = "Reference (Omni) — up to 9 images, 3 videos, 3 audio"
MODE_FIRST_LAST = "First/Last Frame — zero, one, or two frame images"

MAX_CHARACTER_SLOTS = 9
ASPECT_RATIO_OPTIONS = [a.value for a in AspectRatio]


def _execute_comfy_node(node_class, **kwargs):
    """Invoke a ComfyUI node's main entrypoint, whether it is a comfy_api io.ComfyNode
    (classmethod 'execute') or a legacy node (instance method named by FUNCTION)."""
    if hasattr(node_class, "execute"):
        return node_class.execute(**kwargs)
    fn_name = getattr(node_class, "FUNCTION", None)
    instance = node_class()
    if fn_name and hasattr(instance, fn_name):
        return getattr(instance, fn_name)(**kwargs)
    raise RuntimeError(f"Could not determine how to execute node {node_class!r}")


def _unpack_node_result(out):
    """Normalise a node return (io.NodeOutput, tuple, list or dict) into a tuple of outputs."""
    if out is None:
        return ()
    for attr in ("result", "args", "values", "outputs"):
        if hasattr(out, attr):
            val = getattr(out, attr)
            if callable(val):
                try:
                    val = val()
                except Exception:
                    continue
            if isinstance(val, (tuple, list)):
                return tuple(val)
    if isinstance(out, (tuple, list)):
        return tuple(out)
    if isinstance(out, dict) and isinstance(out.get("result"), (tuple, list)):
        return tuple(out["result"])
    return (out,)


def _resolve_resolution(aspect_ratio: str, megapixels: float, multiple: int):
    """Exact port of the stock ResolutionSelector node's own formula."""
    w_ratio, h_ratio = ASPECT_RATIOS[AspectRatio(aspect_ratio)]
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
    width = round(w_ratio * scale / multiple) * multiple
    height = round(h_ratio * scale / multiple) * multiple
    return width, height


def _load_image_source(b64_or_url: str, filename: str = "") -> torch.Tensor:
    if not b64_or_url:
        return None
    try:
        b64_str = b64_or_url
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception as e:
        log.warning("[MuseMinimaxDirector] Could not decode reference image %s: %s", filename, e)
        return None


def _load_character_image(entry: dict):
    """Character/background reference images upload through the same mechanism as
    ref_video/ref_audio (a small file-path string in timeline_data), not embedded
    base64 — embedding full images inline blew past the browser's per-entry workflow
    draft-autosave budget (~750KB in the wild), silently losing unsaved edits every
    time the workflow was left and returned to. Falls back to legacy inline base64
    for character cards saved before this change."""
    if entry.get("file"):
        file_path = _resolve_path(entry["file"])
        if not file_path or not os.path.exists(file_path):
            log.warning("[MuseMinimaxDirector] Reference image not found: %s", entry.get("name", entry.get("file", "")))
            return None
        try:
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)
        except Exception as e:
            log.warning("[MuseMinimaxDirector] Could not load reference image %s: %s", entry.get("name", ""), e)
            return None
    if entry.get("image_b64"):
        return _load_image_source(entry["image_b64"], entry.get("name", ""))
    return None


def _parse_timeline(timeline_data: str) -> dict:
    try:
        data = json.loads(timeline_data) if timeline_data and timeline_data.strip() else {}
    except Exception as e:
        log.warning("[MuseMinimaxDirector] Could not parse timeline_data: %s", e)
        data = {}
    data.setdefault("characters", [])
    data.setdefault("segments", [])
    data.setdefault("style_line", "")
    data.setdefault("background", None)
    data.setdefault("refVideos", [])
    data.setdefault("refAudios", [])
    return data


def _resolve_path(rel: str) -> str:
    """Reference video/audio clips are uploaded through ComfyUI's own /upload/image
    endpoint (works for any file type despite the name) into the input dir, same
    mechanism LTX Director's audio/video tracks already use — so the same multi-base
    fallback lookup applies here."""
    if not rel:
        return ""
    input_dir = folder_paths.get_input_directory()
    for base in (input_dir, os.path.join(input_dir, "musedirector"), os.path.join(input_dir, "muse")):
        p = os.path.join(base, os.path.basename(rel))
        if os.path.exists(p):
            return p
    p = os.path.join(input_dir, rel)
    return p if os.path.exists(p) else ""


def _load_ref_video_tensor(entry: dict, max_frames: int = 200):
    """Decodes the [trimStartSec, trimEndSec) window of an uploaded reference video
    clip into an IMAGE tensor of frames, for H3's ref_videos input."""
    file_path = _resolve_path(entry.get("file", ""))
    if not file_path or not os.path.exists(file_path):
        log.warning("[MuseMinimaxDirector] Reference video not found: %s", entry.get("fileName", entry.get("file", "")))
        return None
    start_sec = float(entry.get("trimStartSec", 0) or 0)
    end_sec = entry.get("trimEndSec")
    frames = []
    try:
        with av.open(file_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if stream.time_base:
                seek_pts = int(max(0, start_sec - 0.5) / float(stream.time_base))
            else:
                seek_pts = int(max(0, start_sec - 0.5) * av.time_base)
            container.seek(seek_pts, stream=stream, backward=True)
            for frame in container.decode(stream):
                frame_time = frame.time
                if frame_time is None and frame.pts is not None and stream.time_base:
                    frame_time = float(frame.pts * stream.time_base)
                if frame_time is None:
                    frame_time = 0.0
                if frame_time < start_sec - 0.01:
                    continue
                if end_sec is not None and frame_time > float(end_sec):
                    break
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= max_frames:
                    break
    except Exception as exc:
        log.warning("[MuseMinimaxDirector] Reference video decode error (%s): %s", entry.get("fileName", ""), exc)
        return None
    if not frames:
        return None
    frames_np = np.array(frames, dtype=np.float32) / 255.0
    return torch.from_numpy(frames_np)


def _load_ref_audio_clip(entry: dict, target_sr: int = 44100):
    """Decodes the [trimStartSec, trimEndSec) window of an uploaded reference audio
    clip into an AUDIO dict, for H3's ref_audios input."""
    file_path = _resolve_path(entry.get("file", ""))
    if not file_path or not os.path.exists(file_path):
        log.warning("[MuseMinimaxDirector] Reference audio not found: %s", entry.get("fileName", entry.get("file", "")))
        return None
    start_sec = float(entry.get("trimStartSec", 0) or 0)
    end_sec = entry.get("trimEndSec")
    try:
        clip_frames = []
        with av.open(file_path) as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_sr)
            for frame in container.decode(stream):
                for rf in resampler.resample(frame):
                    clip_frames.append(torch.from_numpy(rf.to_ndarray()))
            for rf in resampler.resample(None):
                clip_frames.append(torch.from_numpy(rf.to_ndarray()))
        if not clip_frames:
            return None
        waveform = torch.cat(clip_frames, dim=1)  # [2, samples]
        start_sample = max(0, min(int(start_sec * target_sr), waveform.shape[1]))
        end_sample = waveform.shape[1] if end_sec is None else max(start_sample, min(int(float(end_sec) * target_sr), waveform.shape[1]))
        trimmed = waveform[:, start_sample:end_sample]
        if trimmed.shape[1] == 0:
            return None
        return {"waveform": trimmed.unsqueeze(0), "sample_rate": target_sr}
    except Exception as exc:
        log.warning("[MuseMinimaxDirector] Reference audio decode error (%s): %s", entry.get("fileName", ""), exc)
        return None


def _build_character_ref_images(tdata: dict):
    """Returns (ref_images_dict, picture_labels_list) for character slots only —
    background is handled separately per-chunk, since on continuation chunks it gets
    replaced by a continuity anchor (see the per-chunk loop in execute()).

    Densely packed in fill order (ref_image_0, ref_image_1, ... with no gaps), NOT
    keyed by the UI's fixed Ref-N slot index. This matters because MiniMaxH3ReferenceToVideo
    assigns <Picture N> tags purely by iteration order over ref_images.values() — it never
    inspects the "ref_image_N" key's own number at all (confirmed directly from its source).
    An earlier version of this function kept slot position fixed (e.g. background always at
    ref_image_9), which desynced the compiled prompt's tag numbers from what H3 actually
    assigned the moment any earlier slot was left empty — e.g. only Ref 1 filled meant H3 saw
    just two images total and labeled the second one <Picture 2>, while the old prompt text
    still called it <Picture 10>, a tag that didn't correspond to anything — silently breaking
    the background/setting reference on most real-world scenes (which rarely fill all 9 slots)."""
    characters = tdata.get("characters", [])[:MAX_CHARACTER_SLOTS]

    picture_labels = []
    ref_images = {}
    for ch in characters:
        if not ch or not (ch.get("file") or ch.get("image_b64")):
            continue
        tensor = _load_character_image(ch)
        if tensor is None:
            continue
        slot = len(ref_images)
        ref_images[f"ref_image_{slot}"] = tensor
        desc = (ch.get("description") or "").strip()
        label = f"`<Picture {slot + 1}>`"
        picture_labels.append(f"{label} = {desc}" if desc else label)

    return ref_images, picture_labels


def _build_chunk_prompt(style_line: str, picture_labels: list, audio_labels: list,
                         continuity_notes: list, segments: list) -> str:
    parts = []
    if style_line:
        parts.append(style_line)
    if picture_labels:
        parts.append("Reference images: " + "; ".join(picture_labels) + ".")
    if audio_labels:
        parts.append("Reference audio: " + "; ".join(audio_labels) + ".")
    if continuity_notes:
        parts.extend(continuity_notes)
    for i, seg in enumerate(segments):
        text = (seg.get("prompt") or "").strip()
        if not text:
            continue
        cut_label = seg.get("label") or f"CUT {i + 1}"
        weight = seg.get("duration_hint")
        if weight:
            cut_label += f" (~{weight}s)"
        parts.append(f"{cut_label}: {text}")
    return "\n\n".join(parts)


def _bucket_segments_into_chunks(tdata: dict, duration_seconds: float, chunk_duration_seconds: float):
    """Splits the timeline's CUT segments into per-chunk groups. Chunks run
    chunk_duration_seconds each, in order, with the final chunk absorbing whatever
    remains — so a chunk boundary lands where the user actually set chunk_duration_seconds
    (e.g. 15s total at a 10s chunk size gives 10s + 5s), rather than being silently
    redistributed into N equal-sized pieces (which previously turned that same 15s/10s
    case into two 7.5s chunks — nowhere near what "10s chunks" implies). A remainder
    under H3's own ~4s reliable minimum gets folded into the previous chunk instead of
    becoming an unreliably short trailing chunk. Returns (buckets, chunk_lengths) —
    chunk_lengths is a list of per-chunk durations in seconds, same length as buckets."""
    segments = tdata.get("segments", [])
    total_weight = sum(float(s.get("weight", 1) or 1) for s in segments) or 1.0

    chunk_size = max(0.5, chunk_duration_seconds)
    num_chunks = max(1, math.ceil(duration_seconds / chunk_size))

    bounds = []
    cursor = 0.0
    for i in range(num_chunks):
        end = duration_seconds if i == num_chunks - 1 else min(duration_seconds, cursor + chunk_size)
        bounds.append([cursor, end])
        cursor = end

    min_chunk_seconds = 4.0
    while len(bounds) > 1 and (bounds[-1][1] - bounds[-1][0]) < min_chunk_seconds:
        bounds[-2][1] = bounds[-1][1]
        bounds.pop()
    num_chunks = len(bounds)

    # Each CUT's own [start, end) time range from cumulative weight.
    seg_ranges = []
    seg_cursor = 0.0
    for seg in segments:
        seg_start = seg_cursor
        seg_cursor += (float(seg.get("weight", 1) or 1) / total_weight) * duration_seconds
        seg_ranges.append((seg_start, seg_cursor, seg))

    # A CUT goes into every chunk its time range overlaps, not just the one
    # containing its start — otherwise a single CUT spanning multiple chunks (e.g.
    # one CUT describing the whole clip, split into two H3 calls by chunking) left
    # every chunk after the first with no textual direction at all: just the style
    # line and the "continue seamlessly" instruction, nothing describing what should
    # actually happen, which reads on screen as a stall/repeat rather than progress.
    buckets = [[] for _ in range(num_chunks)]
    for seg_start, seg_end, seg in seg_ranges:
        matched = False
        for i, (b_start, b_end) in enumerate(bounds):
            if seg_start < b_end and seg_end > b_start:
                buckets[i].append(seg)
                matched = True
        if not matched:
            # Degenerate zero-length segment — fall back to whichever chunk
            # contains its start point so it's never silently dropped.
            chunk_idx = num_chunks - 1
            for i, (b_start, b_end) in enumerate(bounds):
                if seg_start < b_end:
                    chunk_idx = i
                    break
            buckets[chunk_idx].append(seg)

    chunk_lengths = [end - start for start, end in bounds]
    return buckets, chunk_lengths


class MuseMinimaxDirector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ([MODE_REFERENCE, MODE_FIRST_LAST], {"default": MODE_REFERENCE}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE", {"tooltip": "Needed for final audio decode in both modes — H3 always builds a joint audio+video latent internally, even in First/Last Frame mode."}),
                "aspect_ratio": (ASPECT_RATIO_OPTIONS, {"default": AspectRatio.WIDESCREEN_H.value}),
                "megapixels": ("FLOAT", {"default": 0.98, "min": 0.1, "max": 4.0, "step": 0.02}),
                "multiple": ("INT", {"default": 32, "min": 8, "max": 128, "step": 4, "advanced": True}),
                "duration_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Total length of the finished video. Automatically split into multiple H3 "
                               "generation calls if longer than chunk_duration_seconds, stitched together."}),
                "chunk_duration_seconds": ("FLOAT", {"default": 10.0, "min": 3.0, "max": 15.0, "step": 0.5,
                    "tooltip": "Length of each individual H3 call. H3's own trained range tops out around "
                               "15s per call — longer totals get split into chunks this long (the final "
                               "chunk absorbs whatever's left over, so it may be shorter). Reference mode: "
                               "each continuation chunk is fed the previous chunk's own last few frames "
                               "and last few seconds of audio as reference video/audio, plus an explicit "
                               "instruction to continue seamlessly rather than cut. First/Last Frame mode: "
                               "continuation falls back to the previous chunk's last frame only."}),
                "ref_image_size": (["match", "max"], {
                    "default": "match",
                    "tooltip": "'match' scales references down to the generation's pixel area (faster). "
                               "'max' keeps up to a 2048px short edge for stronger identity fidelity, but "
                               "reference tokens ride every sampling step so it's several times slower. "
                               "Reference (Omni) mode only.",
                }),
                "hybrid_continuation": ("BOOLEAN", {"default": False, "tooltip":
                    "Reference (Omni) mode only, needs model_fl2va connected. Reference mode's own "
                    "carry-over (ref_video/ref_audio) is a soft reference, not a hard lock — H3 can still "
                    "cut to a new composition at a chunk boundary despite it. When this is on, continuation "
                    "chunks (2nd onward) switch to a hard-locked first-frame anchor instead: the exact last "
                    "frame of the previous chunk, via the separate First/Last-Frame checkpoint's real "
                    "keyframe-lock mechanism. The first chunk always runs Reference (Omni) normally, so "
                    "character/background images still establish identity — continuation chunks just don't "
                    "get fresh reference-image reinforcement after that (the anchor frame itself already "
                    "carries the correct likeness forward, since it's real output from the reference-anchored "
                    "first chunk, not a blank start)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "sampler_name": (["res_multistep", "euler", "euler_ancestral", "dpmpp_2m"], {"default": "res_multistep"}),
                "scheduler": (["simple", "normal", "beta", "sgm_uniform"], {"default": "simple"}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01, "advanced": True}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01, "advanced": True}),
                "timeline_data": ("STRING", {"default": "{}", "multiline": False}),
            },
            "optional": {
                "model_fl2va": ("MODEL", {"tooltip": "Only used when hybrid_continuation is on. The separate "
                    "First/Last-Frame checkpoint (not the same weights as the main Reference/Omni model input) "
                    "— load it via its own loader, bypassed when hybrid_continuation is off, same idea as "
                    "switching between the two mode checkpoints."}),
                # Reference (Omni) mode only. Up to 3 reference videos and 3 reference audio
                # clips, uploaded and scrub-trimmed directly in the timeline UI (timeline_data's
                # refVideos/refAudios) rather than as graph sockets — same convention as the
                # character/background reference images.
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "compiled_prompt")
    FUNCTION = "execute"
    CATEGORY = "Muse Collective"

    def execute(self, mode, model, clip, vae, audio_vae, aspect_ratio, megapixels, multiple,
                duration_seconds, chunk_duration_seconds, ref_image_size, hybrid_continuation,
                seed, steps, sampler_name, scheduler, shift_video, shift_audio, timeline_data,
                model_fl2va=None):
        tdata = _parse_timeline(timeline_data)
        char_ref_images, char_picture_labels = _build_character_ref_images(tdata)
        background = tdata.get("background")
        # First/Last Frame mode has no sockets of its own — Ref 1 / Ref 2 (the same
        # upload UI every other reference uses) double as first_frame/last_frame. Read
        # these directly from the raw characters list by literal slot position, NOT via
        # char_ref_images — that dict is now densely fill-order-packed (see its own
        # docstring), so "ref_image_0" there means "whichever slot was filled first",
        # not "literally Ref 1". First/Last Frame mode has no <Picture N> tagging at all
        # (ImageToVideo takes first_frame/last_frame directly), so there's no tag-order
        # concern here — Ref 1 must always mean Ref 1.
        characters_raw = tdata.get("characters", [])
        first_frame = _load_character_image(characters_raw[0]) if len(characters_raw) > 0 and characters_raw[0] else None
        last_frame = _load_character_image(characters_raw[1]) if len(characters_raw) > 1 and characters_raw[1] else None
        # Background/continuity-anchor slot: the next free dense position after however
        # many character images actually made it into char_ref_images — NOT a fixed index
        # — same reasoning as _build_character_ref_images: H3 tags by iteration order, so
        # this must land wherever the character images actually stopped, not at a fixed
        # slot number that only happens to be correct when all 9 character slots are full.
        bg_index = len(char_ref_images)
        style_line = (tdata.get("style_line") or "").strip()
        width, height = _resolve_resolution(aspect_ratio, megapixels, multiple)

        buckets, chunk_lengths = _bucket_segments_into_chunks(tdata, duration_seconds, chunk_duration_seconds)
        num_chunks = len(buckets)

        log.info(
            "[MuseMinimaxDirector] mode=%s, %dx%d, %d chunk(s): %s (total %.1fs)",
            mode, width, height, num_chunks, ", ".join(f"~{c:.1f}s" for c in chunk_lengths), duration_seconds,
        )

        from nodes import NODE_CLASS_MAPPINGS
        RandomNoise = NODE_CLASS_MAPPINGS["RandomNoise"]
        BasicGuider = NODE_CLASS_MAPPINGS["BasicGuider"]
        KSamplerSelect = NODE_CLASS_MAPPINGS["KSamplerSelect"]
        BasicScheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]
        SamplerCustomAdvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]
        VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]
        VAEDecodeAudio = NODE_CLASS_MAPPINGS["VAEDecodeAudio"]

        # Sigma shift only depends on the model + the two shift values — same for every
        # chunk, so it only needs to run once rather than inside the loop.
        shifted_model = _unpack_node_result(_execute_comfy_node(
            MiniMaxH3SigmaShift, model=model, shift_video=shift_video, shift_audio=shift_audio,
        ))[0]
        shifted_model_fl2va = None
        use_hybrid = mode == MODE_REFERENCE and hybrid_continuation and model_fl2va is not None
        if use_hybrid:
            shifted_model_fl2va = _unpack_node_result(_execute_comfy_node(
                MiniMaxH3SigmaShift, model=model_fl2va, shift_video=shift_video, shift_audio=shift_audio,
            ))[0]
        elif mode == MODE_REFERENCE and hybrid_continuation and model_fl2va is None:
            log.warning("[MuseMinimaxDirector] hybrid_continuation is on but model_fl2va isn't connected — "
                        "falling back to normal Reference (Omni) continuity for every chunk.")
        sampler = _unpack_node_result(_execute_comfy_node(KSamplerSelect, sampler_name=sampler_name))[0]

        # User-provided reference videos/audio — uploaded and scrub-trimmed in the timeline
        # UI, decoded here from disk. Unchanging across chunks; both share slots with the
        # chunking carry-over below (reserved slot 0 on continuation chunks), so both are
        # merged per-chunk inside the loop rather than built once here.
        user_ref_videos = []  # list of (frames_tensor, paired_audio_dict_or_None)
        for entry in (tdata.get("refVideos") or [])[:3]:
            if not entry or not entry.get("file"):
                continue
            tensor = _load_ref_video_tensor(entry)
            if tensor is None:
                continue
            # Opt-in per slot: reuses _load_ref_audio_clip directly on the video file's own
            # embedded audio track (PyAV doesn't care whether the container is "video" or
            # "audio", it just reads whichever stream exists) — no separate decode path needed.
            paired_audio = _load_ref_audio_clip(entry) if entry.get("includeAudio") else None
            user_ref_videos.append((tensor, paired_audio))

        user_ref_audios = []  # list of (AUDIO dict, description) in slot order
        for entry in (tdata.get("refAudios") or [])[:3]:
            if not entry or not entry.get("file"):
                continue
            clip_audio = _load_ref_audio_clip(entry)
            if clip_audio is not None:
                user_ref_audios.append((clip_audio, (entry.get("description") or "").strip()))

        all_images = []
        all_waveform = None
        audio_sample_rate = None
        compiled_prompts = []
        prev_chunk_images = None
        prev_chunk_audio = None

        for chunk_idx, chunk_segments in enumerate(buckets):
            chunk_len_seconds = chunk_lengths[chunk_idx]
            chunk_length = align_frame_count(max(5, round(chunk_len_seconds * 24)))
            is_last_chunk = chunk_idx == num_chunks - 1

            chunk_ref_videos = None
            chunk_ref_audios = None
            chunk_ref_images = {}
            chunk_picture_labels = []
            audio_labels = []
            continuity_notes = []
            # Only actually hybrid-switches once there's a predecessor chunk to lock
            # onto — the first chunk always runs normal Reference (Omni), hybrid or not.
            use_hybrid_chunk = use_hybrid and prev_chunk_images is not None

            if use_hybrid_chunk:
                continuity_notes.append(
                    "This shot begins from the exact final frame of the previous shot, locked as the "
                    "starting frame — continue the ongoing action naturally from this pose and framing, "
                    "no restart, no new take."
                )
                # Hybrid chunks have no reference audio at all (MiniMaxH3ImageToVideo takes none),
                # so without any audio grounding H3 tends to hallucinate unprompted vocalization/
                # speech (confirmed via spectrogram on a real test render). Only suppress that when
                # this chunk's own CUT text doesn't actually call for dialogue — quoted text is the
                # existing convention for spoken lines, so a quote mark means the user wants speech
                # here and this note must not fight that.
                has_dialogue = any('"' in (seg.get("prompt") or "") for seg in chunk_segments)
                if not has_dialogue:
                    # No hardcoded example sounds here (an earlier version listed "footsteps" as an
                    # example and H3 took that literally even in a standing-still shot with no
                    # walking at all) — defer entirely to whatever the shot description below
                    # actually says, rather than suggesting specific sounds that may not apply.
                    continuity_notes.append(
                        "This chunk has no reference audio to ground it — keep the soundscape ambient "
                        "and grounded only in whatever is actually happening in the shot description "
                        "below, consistent with the previous shot's environment. No invented sound "
                        "effects or actions beyond what's described, and no dialogue or vocalization "
                        "unless the shot description explicitly includes spoken lines."
                    )
            elif mode == MODE_REFERENCE:
                # Carry-over continuity always claims slot 0 of both ref_videos and
                # ref_audios once a chunk has a predecessor — H3 allows 3 of each, so
                # user-provided references fill whatever slots are left after that.
                chunk_ref_videos = {}
                chunk_ref_video_audios = {}
                video_slot = 0
                video_continuity_tag = None
                # <Audio j> numbering: a reference video's own paired soundtrack gets tagged
                # before any standalone ref_audios (interleaved right before its <Video k> tag)
                # — confirmed from the real node's own ref_items build order — so this counter
                # has to run across both loops below, video-paired audio first.
                audio_tag_counter = 0
                if prev_chunk_images is not None:
                    chunk_ref_videos[f"ref_video_{video_slot}"] = prev_chunk_images
                    video_continuity_tag = f"<Video {video_slot + 1}>"
                    video_slot += 1
                for v, paired_audio in user_ref_videos:
                    if video_slot > 2:
                        log.warning("[MuseMinimaxDirector] ref_video slots full (3 max, one reserved for chunk "
                                    "carry-over) — dropping an extra user-provided reference video.")
                        break
                    chunk_ref_videos[f"ref_video_{video_slot}"] = v
                    if paired_audio is not None:
                        chunk_ref_video_audios[f"ref_video_audio_{video_slot}"] = paired_audio
                        audio_tag_counter += 1
                        audio_labels.append(f"`<Audio {audio_tag_counter}>` = the audio from `<Video {video_slot + 1}>`")
                    video_slot += 1

                chunk_ref_audios = {}
                audio_slot = 0
                carry_audio_tag = None
                if prev_chunk_audio is not None:
                    # Tail of the previous chunk's own decoded audio, not the whole thing —
                    # H3 treats every ref_audio as a short (2-15s) reference clip.
                    tail_sr = prev_chunk_audio["sample_rate"]
                    tail_samples = min(prev_chunk_audio["waveform"].shape[-1], int(4.0 * tail_sr))
                    tail_wave = prev_chunk_audio["waveform"][..., -tail_samples:]
                    chunk_ref_audios[f"ref_audio_{audio_slot}"] = {"waveform": tail_wave, "sample_rate": tail_sr}
                    audio_tag_counter += 1
                    carry_audio_tag = f"<Audio {audio_tag_counter}>"
                    audio_labels.append(f"`{carry_audio_tag}` = the tail end of the previous shot's own score/ambience")
                    audio_slot += 1
                for clip_audio, desc in user_ref_audios:
                    if audio_slot > 2:
                        log.warning("[MuseMinimaxDirector] ref_audio slots full (3 max, one reserved for chunk "
                                    "carry-over once a chunk has a predecessor) — dropping an extra reference audio clip.")
                        break
                    chunk_ref_audios[f"ref_audio_{audio_slot}"] = clip_audio
                    audio_tag_counter += 1
                    label = f"`<Audio {audio_tag_counter}>`"
                    audio_labels.append(f"{label} = {desc}" if desc else label)
                    audio_slot += 1

                if video_continuity_tag:
                    continuity_notes.append(
                        f"{video_continuity_tag} is the final moment of the immediately preceding shot — continue "
                        "the same action, camera framing, and motion seamlessly from exactly where it left off. "
                        "No cut, no restart, and no change of camera angle unless the shot description below "
                        "explicitly calls for one."
                    )
                if carry_audio_tag:
                    continuity_notes.append(
                        f"Continue the same music, score, and ambience from `{carry_audio_tag}` through this shot "
                        "rather than starting a new piece of music or resetting the soundscape."
                    )

                # Reference images: characters are always present; the fixed background
                # slot holds either the real background image (first chunk / no
                # predecessor) or, on continuation chunks, the previous chunk's own last
                # frame as an explicitly-labeled continuity anchor — a much more direct
                # "continue from exactly here" signal than the same frame sitting buried
                # inside the full ref_video, and auto-cross-referenced to the character's
                # own <Picture N> tag rather than making the user retype it per chunk.
                chunk_ref_images = dict(char_ref_images)
                chunk_picture_labels = list(char_picture_labels)
                if prev_chunk_images is not None:
                    last_frame_still = prev_chunk_images[-1:]
                    chunk_ref_images[f"ref_image_{bg_index}"] = last_frame_still
                    cross_ref = " `<Picture 1>` continues the same ongoing action directly from this point." if char_ref_images else ""
                    chunk_picture_labels.append(
                        f"`<Picture {bg_index + 1}>` = continuity anchor — the exact framing, pose, and camera "
                        f"angle at the end of the previous shot.{cross_ref} No cut, no new take."
                    )
                elif background and (background.get("file") or background.get("image_b64")):
                    tensor = _load_character_image(background)
                    if tensor is not None:
                        chunk_ref_images[f"ref_image_{bg_index}"] = tensor
                        bg_desc = (background.get("description") or "").strip()
                        chunk_picture_labels.append(
                            f"`<Picture {bg_index + 1}>` = the setting" + (f" ({bg_desc})" if bg_desc else "")
                        )

            chunk_prompt = _build_chunk_prompt(style_line, chunk_picture_labels, audio_labels, continuity_notes, chunk_segments)
            compiled_prompts.append(f"--- Chunk {chunk_idx + 1}/{num_chunks} (~{chunk_len_seconds:.1f}s) ---\n{chunk_prompt}")

            log.info("[MuseMinimaxDirector] chunk %d/%d, length=%d frames, video_carry=%s, audio_carry=%s, hybrid=%s",
                      chunk_idx + 1, num_chunks, chunk_length, prev_chunk_images is not None,
                      prev_chunk_audio is not None, use_hybrid_chunk)

            if use_hybrid_chunk:
                out = _execute_comfy_node(
                    MiniMaxH3ImageToVideo,
                    clip=clip, vae=vae, prompt=chunk_prompt,
                    width=width, height=height, length=chunk_length,
                    first_frame=prev_chunk_images[-1:], last_frame=None,
                )
                chunk_shifted_model = shifted_model_fl2va
            elif mode == MODE_REFERENCE:
                out = _execute_comfy_node(
                    MiniMaxH3ReferenceToVideo,
                    clip=clip, vae=vae, audio_vae=audio_vae, prompt=chunk_prompt,
                    width=width, height=height, length=chunk_length, ref_image_size=ref_image_size,
                    ref_images=chunk_ref_images if chunk_ref_images else None,
                    ref_videos=chunk_ref_videos if chunk_ref_videos else None,
                    ref_video_audios=chunk_ref_video_audios if chunk_ref_video_audios else None,
                    ref_audios=chunk_ref_audios if chunk_ref_audios else None,
                )
                chunk_shifted_model = shifted_model
            else:
                chunk_first = prev_chunk_images[-1:] if prev_chunk_images is not None else first_frame
                chunk_last = last_frame if is_last_chunk else None
                out = _execute_comfy_node(
                    MiniMaxH3ImageToVideo,
                    clip=clip, vae=vae, prompt=chunk_prompt,
                    width=width, height=height, length=chunk_length,
                    first_frame=chunk_first, last_frame=chunk_last,
                )
                chunk_shifted_model = shifted_model
            positive, latent = _unpack_node_result(out)[:2]

            noise = _unpack_node_result(_execute_comfy_node(RandomNoise, noise_seed=(seed + chunk_idx)))[0]
            guider = _unpack_node_result(_execute_comfy_node(BasicGuider, model=chunk_shifted_model, conditioning=positive))[0]
            sigmas = _unpack_node_result(_execute_comfy_node(
                BasicScheduler, model=chunk_shifted_model, scheduler=scheduler, steps=steps, denoise=1.0,
            ))[0]
            sampled = _unpack_node_result(_execute_comfy_node(
                SamplerCustomAdvanced, noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent,
            ))[0]

            chunk_images = _unpack_node_result(_execute_comfy_node(VAEDecode, samples=sampled, vae=vae))[0]
            chunk_audio = _unpack_node_result(_execute_comfy_node(VAEDecodeAudio, samples=sampled, vae=audio_vae))[0]

            all_images.append(chunk_images)
            waveform = chunk_audio["waveform"]
            if all_waveform is None:
                all_waveform = waveform
                audio_sample_rate = chunk_audio["sample_rate"]
            else:
                all_waveform = torch.cat([all_waveform, waveform], dim=-1)

            prev_chunk_images = chunk_images
            prev_chunk_audio = chunk_audio

        final_images = torch.cat(all_images, dim=0) if len(all_images) > 1 else all_images[0]
        final_audio = {"waveform": all_waveform, "sample_rate": audio_sample_rate}

        return (final_images, final_audio, "\n\n".join(compiled_prompts))


NODE_CLASS_MAPPINGS = {
    "MuseMinimaxDirector": MuseMinimaxDirector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuseMinimaxDirector": "Muse Minimax Director",
}
