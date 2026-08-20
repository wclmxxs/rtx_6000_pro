from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import main as core


BUSINESS_MODEL = "MiniMax-H3"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:30010").rstrip("/")
DEFAULT_NFE = int(os.getenv("DEFAULT_NFE", "8"))
SYNC_INFER_TIMEOUT_SECONDS = int(os.getenv("SYNC_INFER_TIMEOUT_SECONDS", "1800"))

router = APIRouter()


class MediaURL(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str

    @model_validator(mode="after")
    def validate_url(self) -> "MediaURL":
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be an absolute http(s) URL")
        return self


class ContentItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["text", "image_url", "video_url", "audio_url"]
    # Text roles such as "user" are accepted for gateway compatibility and
    # ignored. Media roles remain semantically validated below.
    role: str | None = None
    text: str | None = None
    image_url: MediaURL | None = None
    video_url: MediaURL | None = None
    audio_url: MediaURL | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "ContentItem":
        values = {
            "text": self.text,
            "image_url": self.image_url,
            "video_url": self.video_url,
            "audio_url": self.audio_url,
        }
        populated = [name for name, value in values.items() if value is not None]
        if populated != [self.type]:
            raise ValueError(f"type={self.type!r} requires only the {self.type} field")
        if self.type == "text":
            if not self.text or not self.text.strip():
                raise ValueError("text must be non-empty")
            return self

        expected_roles = {
            "image_url": {"first_frame", "last_frame", "reference_image"},
            "video_url": {"reference_video"},
            "audio_url": {"reference_audio"},
        }
        if self.role not in expected_roles[self.type]:
            allowed = ", ".join(sorted(expected_roles[self.type]))
            raise ValueError(f"type={self.type!r} requires role in [{allowed}]")
        return self


class GenerationRequest(BaseModel):
    # Gateways may add optional fields (for example aigc_watermark) that this
    # backend does not implement. Ignore them instead of rejecting the task.
    model_config = ConfigDict(extra="ignore")

    model: str
    content: list[ContentItem] = Field(min_length=1)
    resolution: Literal["768P", "704P"]
    duration: int = Field(ge=4, le=15)
    ratio: Literal["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"] | None = None
    num_inference_steps: int | None = Field(default=None, ge=1, le=50)
    seed: int | None = Field(default=None, ge=0, le=(1 << 63) - 1)

    @model_validator(mode="after")
    def validate_generation(self) -> "GenerationRequest":
        if self.model != BUSINESS_MODEL:
            raise ValueError(f"model must be {BUSINESS_MODEL!r}")
        if not any(item.type == "text" and item.text and item.text.strip() for item in self.content):
            raise ValueError("content must contain at least one non-empty text item")

        media = [item for item in self.content if item.type != "text"]
        if not media:
            if self.ratio is None or self.ratio == "adaptive":
                raise ValueError("text-only generation requires a non-adaptive ratio")
            return self

        keyframes = [item for item in media if item.role in {"first_frame", "last_frame"}]
        references = [item for item in media if item.role and item.role.startswith("reference_")]
        if keyframes and references:
            raise ValueError("first/last frames cannot be mixed with reference media")
        if len([item for item in keyframes if item.role == "first_frame"]) > 1:
            raise ValueError("at most one first_frame is allowed")
        if len([item for item in keyframes if item.role == "last_frame"]) > 1:
            raise ValueError("at most one last_frame is allowed")
        reference_image_count = sum(item.role == "reference_image" for item in references)
        reference_video_count = sum(item.role == "reference_video" for item in references)
        reference_audio_count = sum(item.role == "reference_audio" for item in references)
        if reference_image_count > 9:
            raise ValueError("at most 9 reference images are allowed")
        if reference_video_count > 3:
            raise ValueError("at most 3 reference videos are allowed")
        if reference_audio_count > 3:
            raise ValueError("at most 3 reference audios are allowed")
        if self.ratio is None:
            self.ratio = "adaptive"
        return self


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    task_id: str

    @model_validator(mode="after")
    def validate_query(self) -> "QueryRequest":
        if self.model != BUSINESS_MODEL:
            raise ValueError(f"model must be {BUSINESS_MODEL!r}")
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        return self


def _media_url(item: ContentItem) -> str:
    payload = {
        "image_url": item.image_url,
        "video_url": item.video_url,
        "audio_url": item.audio_url,
    }[item.type]
    assert payload is not None
    return payload.url


def to_core_request(request: GenerationRequest) -> core.VideoRequest:
    prompt = "\n".join(
        item.text.strip()
        for item in request.content
        if item.type == "text" and item.text
    )
    media = [item for item in request.content if item.type != "text"]
    conditions: list[dict[str, Any]] = []
    if not media:
        task = "t2va"
    elif any(item.role in {"first_frame", "last_frame"} for item in media):
        task = "fl2va"
        ordered = sorted(media, key=lambda item: 0 if item.role == "first_frame" else 1)
        for item in ordered:
            conditions.append(
                {
                    "type": "image",
                    "uri": _media_url(item),
                    "role": "keyframe",
                    "frame_index": 0 if item.role == "first_frame" else -1,
                }
            )
    else:
        task = "ref2va"
        type_map = {
            "image_url": "image",
            "video_url": "video",
            "audio_url": "audio",
        }
        for item in media:
            conditions.append(
                {
                    "type": type_map[item.type],
                    "uri": _media_url(item),
                    "role": "reference",
                }
            )

    nfe = request.num_inference_steps or DEFAULT_NFE
    payload = {
        "model": core.MODEL_NAME,
        "prompt": prompt,
        "seconds": request.duration,
        "task": task,
        "conditions": conditions,
        "target": {
            "short_edge": int(request.resolution.removesuffix("P")),
            "aspect_ratio": "auto" if request.ratio == "adaptive" else request.ratio,
            "duration_seconds": float(request.duration),
        },
        "num_outputs_per_prompt": 1,
        # Core/SGLang counts sigma grid points including terminal zero; the
        # business field counts actual denoiser evaluations (NFE).
        "num_inference_steps": nfe + 1,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": request.seed,
    }
    return core.VideoRequest.model_validate(payload)


def task_payload(job: dict[str, Any]) -> dict[str, Any]:
    business = job.get("_business") or {}
    status = {
        "queued": "queued",
        "in_progress": "running",
        "completed": "succeeded",
        "failed": "failed",
        "deleted": "cancelled",
        "expired": "expired",
    }.get(str(job.get("status")), "failed")
    created_at = int(job.get("created_at") or time.time())
    updated_at = int(
        job.get("expired_at")
        or job.get("completed_at")
        or job.get("_gpu_started_at")
        or job.get("_queued_at")
        or created_at
    )
    task: dict[str, Any] = {
        "id": job["id"],
        "model": BUSINESS_MODEL,
        "status": status,
        "created_at": created_at,
        "updated_at": updated_at,
        "resolution": business.get("resolution", "768P"),
        "duration": business.get("duration", 5),
        "ratio": business.get("ratio", "16:9"),
        "task_type": "generation",
        "modality": "video",
    }
    if status == "succeeded":
        task["content"] = {
            "url": job.get("url")
            or f"{PUBLIC_BASE_URL}/ic/capcut/edit_gateway/v2/video_generation/{job['id']}/content"
        }
    elif status == "failed":
        raw_error = job.get("error") or {}
        message = raw_error.get("message") if isinstance(raw_error, dict) else str(raw_error)
        task["error"] = {
            "type": "upstream_error",
            "message": message or "Video generation failed",
            "http_code": 500,
        }
    return task


async def submit(request: GenerationRequest) -> str:
    core_request = to_core_request(request)
    response = await core.create_video(core_request, None)
    job = core.load_job(response.id)
    if job is None:
        raise HTTPException(status_code=500, detail="task record was not created")
    job["_business"] = {
        "resolution": request.resolution,
        "duration": request.duration,
        "ratio": request.ratio,
        "nfe": request.num_inference_steps or DEFAULT_NFE,
    }
    core.save_job(job)
    return response.id


@router.post("/ic/capcut/edit_gateway/v2/video_generation")
async def video_generation(request: GenerationRequest) -> dict[str, str]:
    return {"task_id": await submit(request)}


@router.post("/ic/capcut/edit_gateway/v2/query/video_generation")
async def query_video_generation(request: QueryRequest) -> dict[str, Any]:
    job = core.load_job(request.task_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"task {request.task_id!r} not found")
    return {"task": task_payload(job)}


async def _sync_infer(request: GenerationRequest) -> dict[str, Any] | JSONResponse:
    task_id = await submit(request)
    deadline = time.monotonic() + SYNC_INFER_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        job = core.load_job(task_id)
        if job is None:
            raise HTTPException(status_code=500, detail="task record disappeared")
        task = task_payload(job)
        if task["status"] == "succeeded":
            return {"task": task}
        if task["status"] == "failed":
            return JSONResponse(status_code=500, content={"task": task})
        await asyncio.sleep(1)

    job = core.load_job(task_id) or {"id": task_id, "status": "failed"}
    task = task_payload(job)
    task["status"] = "failed"
    task["error"] = {
        "type": "timeout_error",
        "message": f"Video generation exceeded {SYNC_INFER_TIMEOUT_SECONDS} seconds",
        "http_code": 504,
    }
    return JSONResponse(status_code=504, content={"task": task})


@router.post("/sync_infer", response_model=None)
async def sync_infer(request: GenerationRequest) -> dict[str, Any] | JSONResponse:
    return await _sync_infer(request)


@router.post("/ic/capcut/edit_gateway/v2/sync_infer", response_model=None)
async def namespaced_sync_infer(
    request: GenerationRequest,
) -> dict[str, Any] | JSONResponse:
    return await _sync_infer(request)


@router.get("/ic/capcut/edit_gateway/v2/video_generation/{task_id}/content")
async def business_video_content(task_id: str) -> FileResponse:
    job = core.load_job(task_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"task {task_id!r} not found")
    if job.get("status") != "completed":
        if job.get("status") == "expired":
            raise HTTPException(status_code=410, detail="Video output has expired")
        raise HTTPException(status_code=409, detail=f"task is {job.get('status')}")
    paths = job.get("file_paths") or []
    if not paths or not Path(paths[0]).is_file():
        raise HTTPException(status_code=404, detail="generated video file is missing")
    return FileResponse(paths[0], media_type="video/mp4", filename=f"{task_id}.mp4")
