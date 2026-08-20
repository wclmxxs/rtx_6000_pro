import pytest
from pydantic import ValidationError

from app.main import (
    VideoRequest,
    align_frame_count,
    build_graph,
    choose_worker_index,
    resolve_spatial_shape,
    sanitize_upload_name,
    sampling_profile_for,
)


def payload(task="t2va", short_edge=768, aspect_ratio="16:9", conditions=None):
    return {
        "model": "MiniMaxAI/MiniMax-H3",
        "prompt": "test prompt",
        "seconds": 5,
        "task": task,
        "conditions": conditions or [],
        "target": {
            "short_edge": short_edge,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": 5.0,
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": 50,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": 1101,
    }


def test_resolution_extension():
    assert resolve_spatial_shape(480, 16, 9) == (864, 480)
    assert resolve_spatial_shape(768, 16, 9) == (1344, 768)


def test_frame_alignment():
    assert align_frame_count(round(5 * 24)) == 124
    assert align_frame_count(round(4 * 24)) == 107


def test_worker_selection_uses_lowest_depth_and_rotates_ties():
    assert choose_worker_index([0, 0, 0, 0], cursor=2) == 2
    assert choose_worker_index([1, 0, 2, 0], cursor=0) == 1
    assert choose_worker_index([1, 0, 2, 0], cursor=2) == 3


def test_worker_selection_skips_unhealthy_workers():
    assert choose_worker_index([None, 1, None, 0], cursor=0) == 3
    with pytest.raises(RuntimeError, match="no healthy"):
        choose_worker_index([None, None, None, None], cursor=0)


def test_upload_name_is_safe_and_keeps_extension():
    assert sanitize_upload_name("../../my reference 图.png") == "my-reference.png"
    assert sanitize_upload_name(None) == "upload.bin"


def test_t2va_contract():
    request = VideoRequest.model_validate(payload())
    assert request.task == "t2va"


def test_fl2va_signature():
    request = VideoRequest.model_validate(
        payload(
            task="fl2va",
            aspect_ratio="auto",
            conditions=[
                {
                    "type": "image",
                    "uri": "file:///data/minimax-h3/first.png",
                    "role": "keyframe",
                    "frame_index": 0,
                },
                {
                    "type": "image",
                    "uri": "file:///data/minimax-h3/last.png",
                    "role": "keyframe",
                    "frame_index": -1,
                },
            ],
        )
    )
    assert [item.frame_index for item in request.conditions] == [0, -1]


def test_rejects_unknown_short_edge():
    with pytest.raises(ValidationError):
        VideoRequest.model_validate(payload(short_edge=512))


def test_accepts_704_short_edge():
    request = VideoRequest.model_validate(payload(short_edge=704))
    assert request.target.short_edge == 704


def graph_nodes(graph, class_type):
    return [node for node in graph.values() if node["class_type"] == class_type]


def test_larry_6nfe_turbo_profile_and_graph():
    data = payload(short_edge=768)
    data["num_inference_steps"] = 7
    request = VideoRequest.model_validate(data)
    profile = sampling_profile_for(request)
    assert profile.name == "larry-v4-6nfe"
    assert profile.denoiser_evaluations == 6
    assert profile.sampler_name == "euler"
    assert profile.shift_video == 12.0
    assert profile.shift_audio == 3.0
    assert profile.turbo_sampler is True

    graph = build_graph(request, [], 1344, 768, 124, 1101, "job", 0)
    loras = graph_nodes(graph, "MiniMaxH3TurboLoRA")
    assert len(loras) == 1
    assert loras[0]["inputs"] == {
        "model": ["3", 0],
        "lora_name": "minimax_h3_turbo_v4_step600_ema.safetensors",
        "strength": 1.0,
        "low_vram": False,
    }
    assert len(graph_nodes(graph, "MiniMaxH3TurboSampler")) == 1
    assert not graph_nodes(graph, "KSamplerSelect")
    assert graph_nodes(graph, "BasicScheduler")[0]["inputs"]["steps"] == 6


def test_larry_turbo_supports_480_and_4_or_8_nfe():
    data = payload(short_edge=480)
    data["num_inference_steps"] = 5
    request = VideoRequest.model_validate(data)
    profile = sampling_profile_for(request)
    assert profile.name == "larry-v4-4nfe"
    assert profile.denoiser_evaluations == 4
    assert profile.shift_video == 12.0
    assert profile.shift_audio == 3.0
    assert profile.lora_name == "minimax_h3_turbo_v4_step600_ema.safetensors"

    data["num_inference_steps"] = 9
    request = VideoRequest.model_validate(data)
    assert sampling_profile_for(request).denoiser_evaluations == 8


def test_larry_turbo_rejects_wrong_video_shift():
    data = payload(short_edge=768)
    data["num_inference_steps"] = 7
    data["flow_shift"] = 6.0
    request = VideoRequest.model_validate(data)
    with pytest.raises(ValueError, match="requires flow_shift=12"):
        sampling_profile_for(request)


def test_ref2va_never_uses_fl2va_turbo_lora():
    data = payload(
        task="ref2va",
        short_edge=480,
        conditions=[
            {
                "type": "image",
                "uri": "file:///data/minimax-h3/reference.png",
                "role": "reference",
            }
        ],
    )
    data["num_inference_steps"] = 5
    request = VideoRequest.model_validate(data)
    assert sampling_profile_for(request).lora_name is None


def test_normal_step_count_keeps_base_sampler():
    data = payload(short_edge=480)
    data["num_inference_steps"] = 30
    request = VideoRequest.model_validate(data)
    profile = sampling_profile_for(request)
    assert profile.name == "base"
    assert profile.lora_name is None
    assert profile.sampler_name == "res_multistep"
    assert profile.denoiser_evaluations == 29
