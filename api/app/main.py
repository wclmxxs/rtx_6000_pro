from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import os
import re
import shutil
import socket
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import httpx
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator


def parse_comfy_urls() -> tuple[str, ...]:
    configured = os.getenv("COMFY_URLS") or os.getenv(
        "COMFY_URL", "http://127.0.0.1:8188"
    )
    urls = tuple(
        dict.fromkeys(item.strip().rstrip("/") for item in configured.split(",") if item.strip())
    )
    if not urls:
        raise RuntimeError("COMFY_URLS must contain at least one URL")
    return urls


COMFY_URLS = parse_comfy_urls()
INPUT_ROOT = Path(os.getenv("INPUT_ROOT", "/input")).resolve()
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "/data/minimax-h3")).resolve()
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data")).resolve()
OUTPUT_ROOT = DATA_ROOT / "outputs"
JOB_ROOT = DATA_ROOT / "jobs"
API_KEY = os.getenv("API_KEY", "")
ALLOW_REMOTE_MEDIA = os.getenv("ALLOW_REMOTE_MEDIA", "true").lower() in {
    "1",
    "true",
    "yes",
}
MAX_MEDIA_BYTES = int(os.getenv("MAX_MEDIA_BYTES", str(4 * 1024**3)))
REMOTE_MEDIA_HOST_ALLOWLIST = tuple(
    item.strip().lower()
    for item in os.getenv("REMOTE_MEDIA_HOST_ALLOWLIST", ".byted.org").split(",")
    if item.strip()
)
MODEL_NAME = "MiniMaxAI/MiniMax-H3"
FL_MODEL = "minimax_h3_fl2va_pruned_nvfp4.safetensors"
REF_MODEL = "minimax_h3_ref2va_pruned_nvfp4.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_ENABLED = os.getenv("TURBO_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
}
TURBO_LORA = os.getenv(
    "TURBO_LORA",
    "minimax_h3_turbo_v4_step600_ema.safetensors",
)
ASPECT_RATIOS = {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
CORS_ORIGINS = tuple(
    item.strip()
    for item in os.getenv("CORS_ORIGINS", "").split(",")
    if item.strip()
)
GPU_INDEX = int(os.getenv("GPU_INDEX", "0"))
GPU_UUID = os.getenv("GPU_UUID", "")
RELEASE_ID = os.getenv("RELEASE_ID", "unknown")


@dataclass(frozen=True)
class Worker:
    id: int
    url: str


WORKERS = tuple(Worker(id=index, url=url) for index, url in enumerate(COMFY_URLS))
WORKER_SELECTION_LOCK = asyncio.Lock()
NEXT_WORKER_INDEX = 0


@dataclass(frozen=True)
class SamplingProfile:
    name: str
    denoiser_evaluations: int
    sampler_name: str
    shift_video: float
    shift_audio: float
    lora_name: str | None = None
    lora_strength: float = 1.0
    turbo_sampler: bool = False


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image", "video", "video_audio", "audio"]
    uri: str
    role: Literal["keyframe", "reference"]
    frame_index: int | None = None
    start_time_seconds: float | None = Field(default=None, ge=0)


class Target(BaseModel):
    model_config = ConfigDict(extra="ignore")

    short_edge: Literal[480, 704, 768]
    aspect_ratio: str
    duration_seconds: float | None = None


class VideoRequest(BaseModel):
    # This mirrors SGLang's VideoGenerationsRequest: transport extensions are
    # accepted, while H3-specific validation below rejects fields H3 cannot use.
    model_config = ConfigDict(extra="allow")

    prompt: str
    model: str | None = None
    n: int | None = Field(default=1, ge=1, le=10)
    num_outputs_per_prompt: int | None = Field(default=None, ge=1, le=10)
    seconds: int | None = 4
    task: Literal["t2va", "fl2va", "ref2va"]
    conditions: list[Condition] = Field(default_factory=list)
    target: Target
    num_inference_steps: int | None = Field(default=None, ge=2, le=10000)
    flow_shift: float | None = Field(default=None, gt=0)
    audio_flow_shift: float | None = Field(default=None, gt=0)
    seed: int | list[int] | None = None
    quality: Literal["lossless", "high"] | None = None
    output_quality: str | None = "default"
    output_compression: int | None = None
    fps: int | None = None
    num_frames: int | None = None
    guidance_scale: float | None = None
    guidance_scale_2: float | None = None
    true_cfg_scale: float | None = None
    negative_prompt: str | None = None
    enable_frame_interpolation: bool = False
    enable_upscaling: bool = False
    output_mode: str | None = None

    @model_validator(mode="after")
    def validate_h3_contract(self) -> "VideoRequest":
        if not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self.model not in (None, MODEL_NAME):
            raise ValueError(f"model must be {MODEL_NAME!r}")
        if self.target.aspect_ratio not in ASPECT_RATIOS | {"auto"}:
            raise ValueError(
                "target.aspect_ratio must be 'auto' or one of "
                f"{sorted(ASPECT_RATIOS)}"
            )
        if self.fps is not None or self.num_frames is not None:
            raise ValueError(
                "fps and num_frames are not supported: MiniMax H3 derives "
                "timing from target.duration_seconds"
            )
        retired = {
            "guidance_scale": self.guidance_scale,
            "guidance_scale_2": self.guidance_scale_2,
            "true_cfg_scale": self.true_cfg_scale,
            "negative_prompt": self.negative_prompt,
        }
        for name, value in retired.items():
            if value is not None:
                raise ValueError(
                    f"{name} is not supported by MiniMax H3's distilled checkpoint"
                )
        if self.enable_frame_interpolation or self.enable_upscaling:
            raise ValueError(
                "MiniMax H3 does not support API interpolation or upscaling"
            )
        if self.output_mode not in (None, "decoded_files"):
            raise ValueError("output_mode must be 'decoded_files' when provided")
        # SGLang's audited high path is fail-closed to a specific 4xH200 shape.
        # It is not available through this single-card ComfyUI backend.
        if self.quality == "high":
            raise ValueError(
                "quality='high' is not available on this backend; use 'lossless'"
            )

        if self.task == "t2va":
            if self.conditions:
                raise ValueError("conditions must be empty for task 't2va'")
        elif self.task == "fl2va":
            if not 1 <= len(self.conditions) <= 2:
                raise ValueError("fl2va requires one or two keyframe conditions")
            signature: list[int] = []
            for index, condition in enumerate(self.conditions):
                if condition.type != "image" or condition.role != "keyframe":
                    raise ValueError(
                        f"conditions[{index}] must be image/keyframe for fl2va"
                    )
                if condition.start_time_seconds is not None:
                    raise ValueError(
                        f"conditions[{index}].start_time_seconds is not allowed"
                    )
                if condition.frame_index is None:
                    raise ValueError(
                        f"conditions[{index}].frame_index is required for fl2va"
                    )
                signature.append(condition.frame_index)
            if signature not in ([0], [-1], [0, -1]):
                raise ValueError(
                    "fl2va frame_index signature must be [0], [-1], or [0, -1]"
                )
        else:
            if not self.conditions:
                raise ValueError("ref2va requires at least one reference condition")
            image_count = video_count = audio_count = 0
            for index, condition in enumerate(self.conditions):
                if condition.role != "reference":
                    raise ValueError(
                        f"conditions[{index}].role must be 'reference' for ref2va"
                    )
                if condition.frame_index is not None:
                    raise ValueError(
                        f"conditions[{index}].frame_index is not allowed for ref2va"
                    )
                if (
                    condition.start_time_seconds is not None
                    and condition.type not in {"video", "video_audio"}
                ):
                    raise ValueError(
                        f"conditions[{index}].start_time_seconds is only allowed "
                        "for video references"
                    )
                image_count += condition.type == "image"
                video_count += condition.type in {"video", "video_audio"}
                audio_count += condition.type == "audio"
            if image_count > 9 or video_count > 3 or audio_count > 3:
                raise ValueError(
                    "ref2va supports at most 9 images, 3 videos, and 3 audios"
                )

        if self.target.duration_seconds is None and self.task != "ref2va":
            raise ValueError("target.duration_seconds is required")
        if self.target.duration_seconds is not None and not (
            4 <= self.target.duration_seconds <= 15
        ):
            raise ValueError("target.duration_seconds must be in [4, 15]")
        return self


class VideoResponse(BaseModel):
    id: str
    object: str = "video"
    model: str = MODEL_NAME
    status: str = "queued"
    progress: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    size: str = ""
    seconds: str = "4"
    quality: str = "standard"
    url: str | None = None
    remixed_from_video_id: str | None = None
    completed_at: int | None = None
    expires_at: int | None = None
    error: dict[str, Any] | None = None
    file_path: str | None = None
    file_paths: list[str] | None = None
    num_outputs: int | None = None
    peak_memory_mb: float | None = None
    queue_time_s: float | None = None
    inference_time_s: float | None = None
    worker_id: int | None = None
    action: dict[str, Any] | None = None


class VideoListResponse(BaseModel):
    data: list[VideoResponse]
    object: str = "list"


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def align_frame_count(frame_count: int) -> int:
    current = max(1, int(frame_count))
    return current + (5 - current) % 17


def format_seconds(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def sanitize_upload_name(filename: str | None) -> str:
    original = Path(filename or "upload.bin").name
    suffix = re.sub(r"[^A-Za-z0-9.]", "", Path(original).suffix)[:16]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(original).stem).strip("-_")
    return f"{(stem or 'upload')[:120]}{suffix or '.bin'}"


def sampling_profile_for(req: VideoRequest) -> SamplingProfile:
    # SGLang counts sigma grid points, including the terminal zero. ComfyUI's
    # BasicScheduler takes the number of denoiser evaluations instead.
    grid_points = req.num_inference_steps or 50
    denoiser_evaluations = max(1, grid_points - 1)

    lora_name: str | None = None
    profile_name = "base"
    sampler_name = "res_multistep"
    expected_video_shift: float | None = None
    expected_audio_shift: float | None = None

    turbo_sampler = False
    if (
        TURBO_ENABLED
        and req.task in {"t2va", "fl2va"}
        and grid_points in {5, 7, 9}
    ):
        # LarryVRH v4 step-600 EMA is trained for 4 NFE and recommended at
        # 6-8 NFE for better motion/detail. The custom node handles pruned bases
        # and MiniMax-H3's separate video/audio schedules.
        profile_name = f"larry-v4-{denoiser_evaluations}nfe"
        lora_name = TURBO_LORA
        sampler_name = "euler"
        turbo_sampler = True
        expected_video_shift = 12.0
        expected_audio_shift = 3.0

    if lora_name is not None:
        if req.flow_shift is not None and not math.isclose(
            req.flow_shift, expected_video_shift, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"{profile_name} requires flow_shift={expected_video_shift:g}"
            )
        if req.audio_flow_shift is not None and not math.isclose(
            req.audio_flow_shift, expected_audio_shift, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"{profile_name} requires audio_flow_shift={expected_audio_shift:g}"
            )

    return SamplingProfile(
        name=profile_name,
        denoiser_evaluations=denoiser_evaluations,
        sampler_name=sampler_name,
        shift_video=(
            expected_video_shift
            if expected_video_shift is not None
            else (req.flow_shift or 12.0)
        ),
        shift_audio=(
            expected_audio_shift
            if expected_audio_shift is not None
            else (req.audio_flow_shift or 3.0)
        ),
        lora_name=lora_name,
        turbo_sampler=turbo_sampler,
    )


def resolve_spatial_shape(short_edge: int, width: float, height: float) -> tuple[int, int]:
    ratio = float(width) / float(height)
    if not 0.25 <= ratio <= 4.0:
        raise ValueError("target aspect ratio must be within 1:4 and 4:1")
    if ratio >= 1:
        nominal_width, nominal_height = short_edge * ratio, float(short_edge)
    else:
        nominal_width, nominal_height = float(short_edge), short_edge / ratio
    # Keep SGLang's 768x1344 cap and use ComfyUI's validated 864x480 canvas
    # as the corresponding cap for the added 480 profile.
    max_long_edge = {480: 864, 704: 1248, 768: 1344}[short_edge]
    max_pixels = short_edge * max_long_edge
    if nominal_width * nominal_height > max_pixels:
        scale = math.sqrt(max_pixels / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale

    def nearest_32(value: float) -> int:
        return max(32, int(round(value / 32)) * 32)

    return nearest_32(nominal_width), nearest_32(nominal_height)


def parse_ratio(value: str) -> tuple[int, int]:
    left, right = value.split(":", 1)
    return int(left), int(right)


def run_ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def probe_media(path: Path, media_type: str) -> dict[str, Any]:
    if media_type == "image":
        with Image.open(path) as image:
            width, height = image.size
        return {"width": width, "height": height}
    facts = run_ffprobe(path)
    streams = facts.get("streams") or []
    visual = next((s for s in streams if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = float((facts.get("format") or {}).get("duration") or 0)
    return {
        "width": int((visual or {}).get("width") or 0),
        "height": int((visual or {}).get("height") or 0),
        "has_audio": has_audio,
        "duration": duration,
    }


def hostname_is_allowlisted(hostname: str) -> bool:
    normalized = hostname.rstrip(".").lower()
    return any(
        normalized == pattern.lstrip(".")
        or (pattern.startswith(".") and normalized.endswith(pattern))
        for pattern in REMOTE_MEDIA_HOST_ALLOWLIST
    )


def is_safe_remote_hostname(hostname: str) -> bool:
    if hostname_is_allowlisted(hostname):
        return True
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            return False
    return True


def safe_relative_to(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"media path must be inside {root}") from exc


async def download_remote(uri: str, destination: Path) -> None:
    if not ALLOW_REMOTE_MEDIA:
        raise ValueError("remote media URLs are disabled")
    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("media URI must use file://, http://, or https://")
    if not is_safe_remote_hostname(parsed.hostname):
        raise ValueError(
            "remote media host must be allowlisted or resolve only to public IP addresses"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    async with httpx.AsyncClient(timeout=120, follow_redirects=False) as client:
        current = uri
        for _ in range(6):
            current_parsed = urlparse(current)
            if not current_parsed.hostname or not is_safe_remote_hostname(
                current_parsed.hostname
            ):
                raise ValueError("redirect target is not an allowed remote host")
            async with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("remote media redirect has no location")
                    current = str(response.url.join(location))
                    continue
                response.raise_for_status()
                with destination.open("wb") as output:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_MEDIA_BYTES:
                            raise ValueError("remote media exceeds MAX_MEDIA_BYTES")
                        output.write(chunk)
                return
        raise ValueError("too many remote media redirects")


async def materialize_condition(
    condition: Condition, job_id: str, condition_index: int, duration: float | None
) -> tuple[str, dict[str, Any]]:
    parsed = urlparse(condition.uri)
    extension = Path(unquote(parsed.path)).suffix or {
        "image": ".png",
        "video": ".mp4",
        "video_audio": ".mp4",
        "audio": ".wav",
    }[condition.type]
    job_input = INPUT_ROOT / "sglang-bridge" / job_id
    job_input.mkdir(parents=True, exist_ok=True)

    if parsed.scheme == "file":
        source = Path(unquote(parsed.path)).resolve()
        safe_relative_to(source, MEDIA_ROOT)
        if not source.is_file():
            raise ValueError(f"media file does not exist: {condition.uri}")
        # MEDIA_ROOT and INPUT_ROOT normally point at the same host directory.
        try:
            relative = source.relative_to(MEDIA_ROOT)
            input_candidate = INPUT_ROOT / relative
            if input_candidate.is_file():
                local = input_candidate
            else:
                local = job_input / f"{condition_index}{extension}"
                shutil.copy2(source, local)
        except ValueError:
            local = job_input / f"{condition_index}{extension}"
            shutil.copy2(source, local)
    elif parsed.scheme in {"http", "https"}:
        local = job_input / f"{condition_index}{extension}"
        await download_remote(condition.uri, local)
    else:
        raise ValueError("condition URI must use file://, http://, or https://")

    media_type = condition.type
    probe_type = "video" if media_type in {"video", "video_audio"} else media_type
    facts = await asyncio.to_thread(probe_media, local, probe_type)
    if condition.type == "video_audio" and not facts.get("has_audio"):
        raise ValueError(
            f"conditions[{condition_index}] requires a video with an audio stream"
        )

    if condition.type in {"video", "video_audio"} and (
        condition.start_time_seconds or duration
    ):
        processed = job_input / f"{condition_index}-trimmed.mp4"
        command = ["ffmpeg", "-y", "-v", "error"]
        if condition.start_time_seconds:
            command += ["-ss", str(condition.start_time_seconds)]
        command += ["-i", str(local)]
        if duration:
            command += ["-t", str(duration)]
        command += [
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            "fps=24",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-ar",
            "32000",
            "-ac",
            "2",
            str(processed),
        ]
        await asyncio.to_thread(subprocess.run, command, check=True)
        local = processed
        facts = await asyncio.to_thread(probe_media, local, "video")

    relative_name = str(safe_relative_to(local, INPUT_ROOT))
    return relative_name, facts


def new_node(graph: dict[str, Any], class_type: str, inputs: dict[str, Any]) -> str:
    node_id = str(len(graph) + 1)
    graph[node_id] = {"class_type": class_type, "inputs": inputs}
    return node_id


def build_graph(
    req: VideoRequest,
    media: list[tuple[str, dict[str, Any]]],
    width: int,
    height: int,
    frame_count: int,
    seed: int,
    job_id: str,
    variant: int,
) -> dict[str, Any]:
    graph: dict[str, Any] = {}
    profile = sampling_profile_for(req)
    video_vae = new_node(graph, "VAELoader", {"vae_name": VIDEO_VAE})
    audio_vae = new_node(graph, "VAELoader", {"vae_name": AUDIO_VAE})
    unet = new_node(
        graph,
        "UNETLoader",
        {
            "unet_name": REF_MODEL if req.task == "ref2va" else FL_MODEL,
            "weight_dtype": "default",
        },
    )
    clip = new_node(
        graph,
        "CLIPLoader",
        {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"},
    )
    sampling_model = unet
    if profile.lora_name is not None:
        sampling_model = new_node(
            graph,
            "MiniMaxH3TurboLoRA",
            {
                "model": [unet, 0],
                "lora_name": profile.lora_name,
                "strength": profile.lora_strength,
                "low_vram": False,
            },
        )
    shifted_model = new_node(
        graph,
        "MiniMaxH3SigmaShift",
        {
            "model": [sampling_model, 0],
            "shift_video": profile.shift_video,
            "shift_audio": profile.shift_audio,
        },
    )

    if req.task in {"t2va", "fl2va"}:
        conditioning_inputs: dict[str, Any] = {
            "clip": [clip, 0],
            "vae": [video_vae, 0],
            "prompt": req.prompt,
            "width": width,
            "height": height,
            "length": frame_count,
        }
        for condition, (filename, _) in zip(req.conditions, media):
            loader = new_node(graph, "LoadImage", {"image": filename})
            if condition.frame_index == 0:
                conditioning_inputs["first_frame"] = [loader, 0]
            else:
                conditioning_inputs["last_frame"] = [loader, 0]
        conditioning = new_node(
            graph, "MiniMaxH3ImageToVideo", conditioning_inputs
        )
    else:
        ref_images: dict[str, Any] = {}
        ref_videos: dict[str, Any] = {}
        ref_video_audios: dict[str, Any] = {}
        ref_audios: dict[str, Any] = {}
        image_index = video_index = audio_index = 0
        for condition, (filename, _) in zip(req.conditions, media):
            if condition.type == "image":
                loader = new_node(graph, "LoadImage", {"image": filename})
                ref_images[f"ref_image_{image_index}"] = [loader, 0]
                image_index += 1
            elif condition.type in {"video", "video_audio"}:
                loader = new_node(graph, "LoadVideo", {"file": filename})
                components = new_node(
                    graph, "GetVideoComponents", {"video": [loader, 0]}
                )
                ref_videos[f"ref_video_{video_index}"] = [components, 0]
                # Comfy returns None for a silent soundtrack; the H3 node skips it.
                ref_video_audios[f"ref_video_audio_{video_index}"] = [components, 1]
                video_index += 1
            else:
                loader = new_node(graph, "LoadAudio", {"audio": filename})
                ref_audios[f"ref_audio_{audio_index}"] = [loader, 0]
                audio_index += 1
        conditioning_inputs = {
            "clip": [clip, 0],
            "vae": [video_vae, 0],
            "audio_vae": [audio_vae, 0],
            "prompt": req.prompt,
            "width": width,
            "height": height,
            "length": frame_count,
            "ref_image_size": "match",
        }
        if ref_images:
            conditioning_inputs["ref_images"] = ref_images
        if ref_videos:
            conditioning_inputs["ref_videos"] = ref_videos
            conditioning_inputs["ref_video_audios"] = ref_video_audios
        if ref_audios:
            conditioning_inputs["ref_audios"] = ref_audios
        conditioning = new_node(
            graph, "MiniMaxH3ReferenceToVideo", conditioning_inputs
        )

    noise = new_node(graph, "RandomNoise", {"noise_seed": seed})
    if profile.turbo_sampler:
        sampler = new_node(graph, "MiniMaxH3TurboSampler", {})
    else:
        sampler = new_node(
            graph, "KSamplerSelect", {"sampler_name": profile.sampler_name}
        )
    scheduler = new_node(
        graph,
        "BasicScheduler",
        {
            "model": [shifted_model, 0],
            "scheduler": "simple",
            "steps": profile.denoiser_evaluations,
            "denoise": 1.0,
        },
    )
    guider = new_node(
        graph,
        "BasicGuider",
        {"model": [shifted_model, 0], "conditioning": [conditioning, 0]},
    )
    sampled = new_node(
        graph,
        "SamplerCustomAdvanced",
        {
            "noise": [noise, 0],
            "guider": [guider, 0],
            "sampler": [sampler, 0],
            "sigmas": [scheduler, 0],
            "latent_image": [conditioning, 1],
        },
    )
    decoded_video = new_node(
        graph, "VAEDecode", {"samples": [sampled, 0], "vae": [video_vae, 0]}
    )
    decoded_audio = new_node(
        graph,
        "VAEDecodeAudio",
        {"samples": [sampled, 0], "vae": [audio_vae, 0]},
    )
    created = new_node(
        graph,
        "CreateVideo",
        {
            "images": [decoded_video, 0],
            "audio": [decoded_audio, 0],
            "fps": 24.0,
            "bit_depth": 8,
        },
    )
    new_node(
        graph,
        "SaveVideo",
        {
            "video": [created, 0],
            "filename_prefix": f"sglang-bridge/{job_id}/{variant}",
            "format": "mp4",
            "codec": "auto",
        },
    )
    return graph


def job_file(job_id: str) -> Path:
    return JOB_ROOT / f"{job_id}.json"


def save_job(job: dict[str, Any]) -> None:
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    destination = job_file(job["id"])
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2))
    temporary.replace(destination)


def load_job(job_id: str) -> dict[str, Any] | None:
    path = job_file(job_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def public_job(job: dict[str, Any]) -> VideoResponse:
    fields = VideoResponse.model_fields
    return VideoResponse(**{key: value for key, value in job.items() if key in fields})


def choose_worker_index(depths: list[int | None], cursor: int) -> int:
    available = [index for index, depth in enumerate(depths) if depth is not None]
    if not available:
        raise RuntimeError("no healthy ComfyUI workers")
    worker_count = len(depths)
    return min(
        available,
        key=lambda index: (
            depths[index],
            (index - cursor) % worker_count,
        ),
    )


async def read_worker_queue(
    client: httpx.AsyncClient, worker: Worker
) -> tuple[int, int]:
    response = await client.get(f"{worker.url}/queue")
    response.raise_for_status()
    queue = response.json()
    return len(queue.get("queue_running") or []), len(queue.get("queue_pending") or [])


async def select_worker(client: httpx.AsyncClient) -> Worker:
    global NEXT_WORKER_INDEX
    results = await asyncio.gather(
        *(read_worker_queue(client, worker) for worker in WORKERS),
        return_exceptions=True,
    )
    depths: list[int | None] = []
    for result in results:
        if isinstance(result, BaseException):
            depths.append(None)
        else:
            running, pending = result
            depths.append(running + pending)
    try:
        selected_index = choose_worker_index(depths, NEXT_WORKER_INDEX)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="No healthy ComfyUI workers are available"
        ) from exc
    NEXT_WORKER_INDEX = (selected_index + 1) % len(WORKERS)
    return WORKERS[selected_index]


def worker_for_job(job: dict[str, Any]) -> Worker:
    worker_url = str(job.get("_worker_url") or WORKERS[0].url).rstrip("/")
    for worker in WORKERS:
        if worker.url == worker_url:
            return worker
    return Worker(id=int(job.get("worker_id") or 0), url=worker_url)


async def copy_comfy_output(
    worker: Worker, item: dict[str, Any], destination: Path
) -> None:
    params = {
        "filename": item["filename"],
        "subfolder": item.get("subfolder", ""),
        "type": item.get("type", "output"),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "GET", f"{worker.url}/view", params=params
        ) as response:
            response.raise_for_status()
            with destination.open("wb") as output:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    output.write(chunk)


async def update_job(job: dict[str, Any]) -> None:
    prompt_ids: list[str] = job.get("_prompt_ids") or []
    if not prompt_ids:
        return
    worker = worker_for_job(job)
    histories: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=30) as client:
        queue_response = await client.get(f"{worker.url}/queue")
        queue_response.raise_for_status()
        queue = queue_response.json()
        running_ids = {
            entry[1] for entry in queue.get("queue_running", []) if len(entry) > 1
        }
        pending_ids = {
            entry[1] for entry in queue.get("queue_pending", []) if len(entry) > 1
        }
        for prompt_id in prompt_ids:
            response = await client.get(f"{worker.url}/history/{prompt_id}")
            response.raise_for_status()
            histories.append(response.json().get(prompt_id) or {})

    failed: str | None = None
    completed = 0
    for prompt_id, history in zip(prompt_ids, histories):
        status = history.get("status") or {}
        if status.get("status_str") == "error":
            messages = status.get("messages") or []
            failed = json.dumps(messages[-1] if messages else status, ensure_ascii=False)
            break
        if status.get("completed"):
            completed += 1

    if failed:
        completed_at = time.time()
        gpu_started_at = job.get("_gpu_started_at")
        job.update(
            {
                "status": "failed",
                "error": {"message": failed},
                "completed_at": int(completed_at),
                "inference_time_s": (
                    round(completed_at - gpu_started_at, 3)
                    if gpu_started_at is not None
                    else None
                ),
                "file_path": None,
                "file_paths": None,
                "num_outputs": None,
            }
        )
        save_job(job)
        return

    if completed == len(prompt_ids):
        paths: list[str] = []
        for variant, history in enumerate(histories):
            output_item: dict[str, Any] | None = None
            for node_output in (history.get("outputs") or {}).values():
                for item in node_output.get("images") or []:
                    if str(item.get("filename", "")).lower().endswith(".mp4"):
                        output_item = item
                        break
                if output_item:
                    break
            if output_item is None:
                raise RuntimeError(f"ComfyUI prompt {prompt_ids[variant]} has no MP4")
            destination = OUTPUT_ROOT / f"{job['id']}-{variant}.mp4"
            await copy_comfy_output(worker, output_item, destination)
            paths.append(str(destination.resolve()))
        completed_at = time.time()
        gpu_started_at = job.get("_gpu_started_at") or job.get(
            "_queued_at", job.get("_started_at", completed_at)
        )
        job.update(
            {
                "status": "completed",
                "progress": 100,
                "completed_at": int(completed_at),
                "file_path": paths[0],
                "file_paths": paths,
                "num_outputs": len(paths),
                "inference_time_s": round(completed_at - gpu_started_at, 3),
            }
        )
    elif running_ids.intersection(prompt_ids):
        if job.get("_gpu_started_at") is None:
            gpu_started_at = time.time()
            queued_at = job.get("_queued_at", job.get("_started_at", gpu_started_at))
            job["_gpu_started_at"] = gpu_started_at
            job["queue_time_s"] = round(max(0.0, gpu_started_at - queued_at), 3)
        job["status"] = "in_progress"
        job["progress"] = max(1, int(completed / len(prompt_ids) * 100))
    elif pending_ids.intersection(prompt_ids):
        job["status"] = "queued"
        job["progress"] = int(completed / len(prompt_ids) * 100)
    save_job(job)


async def monitor_jobs() -> None:
    while True:
        try:
            for path in JOB_ROOT.glob("*.json"):
                current_job: dict[str, Any] | None = None
                try:
                    current_job = json.loads(path.read_text())
                    if current_job.get("status") in {"queued", "in_progress"}:
                        await update_job(current_job)
                except httpx.HTTPError:
                    # A brief ComfyUI restart/network interruption is retryable.
                    continue
                except Exception as exc:
                    if current_job and current_job.get("id"):
                        current_job.update(
                            {
                                "status": "failed",
                                "error": {"message": str(exc)},
                                "file_path": None,
                                "file_paths": None,
                            }
                        )
                        save_job(current_job)
        except Exception:
            pass
        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    for directory in (INPUT_ROOT, MEDIA_ROOT, OUTPUT_ROOT, JOB_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    monitor = asyncio.create_task(monitor_jobs())
    try:
        yield
    finally:
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass


app = FastAPI(title="MiniMax H3 SGLang-compatible NVFP4 bridge", lifespan=lifespan)
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(CORS_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    if request.url.path.startswith(("/ic/", "/sync_infer")):
        messages = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", [])[1:])
            message = str(error.get("msg") or "invalid request")
            messages.append(f"{location}: {message}" if location else message)
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "invalid_request_error",
                    "message": "; ".join(messages),
                    "http_code": 400,
                }
            },
        )
    # SGLang's video endpoint lowers Pydantic/request validation failures to 400.
    return JSONResponse(
        status_code=400, content=jsonable_encoder({"detail": exc.errors()})
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if not request.url.path.startswith(("/ic/", "/sync_infer")):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    error_type = {
        400: "invalid_request_error",
        404: "not_found_error",
        409: "conflict_error",
        413: "payload_too_large_error",
        429: "rate_limit_error",
        502: "upstream_error",
        503: "upstream_unavailable_error",
        504: "timeout_error",
    }.get(exc.status_code, "internal_error")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": error_type,
                "message": str(exc.detail),
                "http_code": exc.status_code,
            }
        },
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if not request.url.path.startswith(("/ic/", "/sync_infer")):
        raise exc
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "message": f"{type(exc).__name__}: {exc}",
                "http_code": 500,
            }
        },
    )


@app.get("/healthz")
async def healthz(_: None = Depends(require_api_key)) -> dict[str, Any]:
    async def inspect(client: httpx.AsyncClient, worker: Worker) -> dict[str, Any]:
        try:
            stats_response, queue_response = await asyncio.gather(
                client.get(f"{worker.url}/system_stats"),
                client.get(f"{worker.url}/queue"),
            )
            stats_response.raise_for_status()
            queue_response.raise_for_status()
            stats = stats_response.json()
            queue = queue_response.json()
            devices = stats.get("devices") or []
            device = devices[0] if devices else {}
            return {
                "id": worker.id,
                "ok": True,
                "name": device.get("name"),
                "vram_total": device.get("vram_total"),
                "vram_free": device.get("vram_free"),
                "running": len(queue.get("queue_running") or []),
                "pending": len(queue.get("queue_pending") or []),
            }
        except (httpx.HTTPError, ValueError) as exc:
            return {"id": worker.id, "ok": False, "error": str(exc)}

    async with httpx.AsyncClient(timeout=15) as client:
        workers = await asyncio.gather(*(inspect(client, worker) for worker in WORKERS))
    healthy_workers = sum(bool(worker["ok"]) for worker in workers)
    if healthy_workers == 0:
        raise HTTPException(status_code=503, detail="No healthy ComfyUI workers")
    return {
        "ok": True,
        "worker_count": len(WORKERS),
        "healthy_workers": healthy_workers,
        "workers": workers,
        "gpu": {"index": GPU_INDEX, "uuid": GPU_UUID},
        "deployment": {
            "release_id": RELEASE_ID,
            "service_model": "MiniMax-H3",
            "base_model": MODEL_NAME,
            "checkpoint": "NVFP4",
            "attention": "SageAttention2",
            "turbo_lora": TURBO_LORA if TURBO_ENABLED else None,
        },
    }


@app.post("/v1/files")
async def upload_file(
    file: UploadFile = File(...),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    upload_id = str(uuid.uuid4())
    relative = Path("browser-uploads") / upload_id / sanitize_upload_name(file.filename)
    destination = INPUT_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_MEDIA_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds MAX_MEDIA_BYTES ({MAX_MEDIA_BYTES})",
                    )
                output.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    media_path = MEDIA_ROOT / relative
    return {
        "id": upload_id,
        "object": "file",
        "name": destination.name,
        "bytes": total,
        "content_type": file.content_type,
        "uri": media_path.as_uri(),
    }


@app.post("/v1/videos", response_model=VideoResponse)
async def create_video(
    req: VideoRequest,
    _: None = Depends(require_api_key),
) -> VideoResponse:
    job_id = str(uuid.uuid4())
    duration = req.target.duration_seconds
    media: list[tuple[str, dict[str, Any]]] = []
    worker: Worker | None = None
    try:
        for index, condition in enumerate(req.conditions):
            media.append(
                await materialize_condition(condition, job_id, index, duration)
            )
        if duration is None:
            duration_sources = [
                facts
                for condition, (_, facts) in zip(req.conditions, media)
                if condition.type in {"audio", "video", "video_audio"}
            ]
            if len(duration_sources) != 1:
                raise ValueError(
                    "target.duration_seconds is required unless ref2va has exactly "
                    "one audio-bearing reference"
                )
            duration = float(duration_sources[0].get("duration") or 0)
            if not 4 <= duration <= 15:
                raise ValueError("derived target duration must be in [4, 15]")

        if req.target.aspect_ratio == "auto":
            if req.task == "fl2va":
                ratio_facts = media[0][1]
                ratio_width, ratio_height = ratio_facts["width"], ratio_facts["height"]
            else:
                ratio_width, ratio_height = 16, 9
        else:
            ratio_width, ratio_height = parse_ratio(req.target.aspect_ratio)
        width, height = resolve_spatial_shape(
            req.target.short_edge, ratio_width, ratio_height
        )
        frame_count = align_frame_count(round(duration * 24))
        outputs = req.num_outputs_per_prompt or req.n or 1
        if isinstance(req.seed, list):
            if len(req.seed) != outputs:
                raise ValueError("seed list length must equal the number of outputs")
            seeds = req.seed
            if any(seed < 0 or seed > (1 << 63) - 1 for seed in seeds):
                raise ValueError("every seed must be in [0, 2^63-1]")
        else:
            base_seed = (
                req.seed
                if req.seed is not None
                else int.from_bytes(os.urandom(8)) & ((1 << 63) - 1)
            )
            if not 0 <= base_seed <= (1 << 63) - 1:
                raise ValueError("seed must be in [0, 2^63-1]")
            seeds = [base_seed + index for index in range(outputs)]
            if seeds[-1] > (1 << 63) - 1:
                raise ValueError("expanded seed exceeds 2^63-1")

        graphs = [
            build_graph(
                req,
                media,
                width,
                height,
                frame_count,
                seed,
                job_id,
                variant,
            )
            for variant, seed in enumerate(seeds)
        ]
        prompt_ids: list[str] = []
        async with httpx.AsyncClient(timeout=120) as client:
            # Keep selection and enqueue atomic across concurrent API requests.
            # The next request sees this job in the selected worker's queue and
            # naturally chooses another GPU when one is available.
            async with WORKER_SELECTION_LOCK:
                worker = await select_worker(client)
                for graph in graphs:
                    response = await client.post(
                        f"{worker.url}/prompt", json={"prompt": graph}
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("node_errors"):
                        raise ValueError(
                            json.dumps(payload["node_errors"], ensure_ascii=False)
                        )
                    prompt_ids.append(payload["prompt_id"])
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"Selected ComfyUI worker failed: {exc}"
        ) from exc

    if worker is None:
        raise HTTPException(status_code=503, detail="No ComfyUI worker selected")
    queued_at = time.time()
    now = int(queued_at)
    job: dict[str, Any] = {
        "id": job_id,
        "object": "video",
        "model": req.model or MODEL_NAME,
        "status": "queued",
        "progress": 0,
        "created_at": now,
        "size": f"{width}x{height}",
        "seconds": format_seconds(frame_count / 24),
        "quality": req.quality or "standard",
        "url": None,
        "remixed_from_video_id": None,
        "completed_at": None,
        "expires_at": None,
        "error": None,
        "file_path": None,
        "file_paths": None,
        "num_outputs": None,
        "peak_memory_mb": None,
        "queue_time_s": None,
        "inference_time_s": None,
        "worker_id": worker.id,
        "action": None,
        "_prompt_ids": prompt_ids,
        "_worker_url": worker.url,
        "_queued_at": queued_at,
        "_gpu_started_at": None,
        "_started_at": queued_at,
    }
    save_job(job)
    return public_job(job)


@app.get("/v1/videos", response_model=VideoListResponse)
async def list_videos(
    after: str | None = Query(default=None),
    limit: int | None = Query(default=None, ge=1, le=100),
    order: Literal["asc", "desc"] = Query(default="desc"),
    _: None = Depends(require_api_key),
) -> VideoListResponse:
    jobs = [json.loads(path.read_text()) for path in JOB_ROOT.glob("*.json")]
    jobs.sort(key=lambda item: item.get("created_at", 0), reverse=order == "desc")
    if after:
        positions = [index for index, item in enumerate(jobs) if item["id"] == after]
        if positions:
            jobs = jobs[positions[0] + 1 :]
    if limit:
        jobs = jobs[:limit]
    return VideoListResponse(data=[public_job(job) for job in jobs])


@app.get("/v1/videos/{video_id}", response_model=VideoResponse)
async def retrieve_video(
    video_id: str, _: None = Depends(require_api_key)
) -> VideoResponse:
    job = load_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    return public_job(job)


@app.delete("/v1/videos/{video_id}", response_model=VideoResponse)
async def delete_video(
    video_id: str, _: None = Depends(require_api_key)
) -> VideoResponse:
    job = load_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    # SGLang currently marks the stored record deleted; it does not abort work.
    for file_path in job.get("file_paths") or []:
        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError:
            pass
    job_file(video_id).unlink(missing_ok=True)
    job["status"] = "deleted"
    return public_job(job)


@app.get("/v1/videos/{video_id}/content")
async def download_video_content(
    video_id: str,
    variant: str | None = Query(default=None),
    _: None = Depends(require_api_key),
) -> FileResponse:
    job = load_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    if job.get("url"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Video has been uploaded to cloud storage. Please use the cloud URL: "
                f"{job['url']}"
            ),
        )
    if job.get("status") not in {"completed", "failed"}:
        raise HTTPException(status_code=404, detail="Generation is still in-progress")
    try:
        variant_index = 0 if variant is None else int(variant)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=404, detail=f"Video variant {variant} not found"
        ) from exc
    paths = job.get("file_paths") or []
    if not 0 <= variant_index < len(paths) or not Path(paths[variant_index]).is_file():
        raise HTTPException(
            status_code=404, detail=f"Video variant {variant} not found"
        )
    path = Path(paths[variant_index])
    return FileResponse(path=path, media_type="video/mp4", filename=path.name)
