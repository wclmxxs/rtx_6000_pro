from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.business import GenerationRequest, QueryRequest, task_payload, to_core_request


def text_item(text="A cinematic landscape"):
    return {"type": "text", "text": text}


def request(**overrides):
    payload = {
        "model": "MiniMax-H3",
        "content": [text_item()],
        "resolution": "768P",
        "duration": 5,
        "ratio": "16:9",
        "num_inference_steps": 6,
    }
    payload.update(overrides)
    return GenerationRequest.model_validate(payload)


def test_text_to_video_maps_business_nfe_to_sigma_points():
    core = to_core_request(request())
    assert core.task == "t2va"
    assert core.target.short_edge == 768
    assert core.num_inference_steps == 7


@pytest.mark.parametrize("model", [None, "", "wrong-model", "another-model"])
def test_generation_model_is_optional_and_ignored(model):
    overrides = {"model": model}
    generation = request(**overrides)
    core = to_core_request(generation)
    assert core.model == "MiniMaxAI/MiniMax-H3"


@pytest.mark.parametrize("model", [None, "", "wrong-model", "another-model"])
def test_query_model_is_optional_and_ignored(model):
    payload = {"task_id": "task-id"}
    if model is not None:
        payload["model"] = model
    query = QueryRequest.model_validate(payload)
    assert query.task_id == "task-id"


def test_text_role_and_unknown_gateway_fields_are_ignored():
    generation = GenerationRequest.model_validate(
        {
            "model": "MiniMax-H3",
            "content": [
                {
                    "type": "text",
                    "role": "user",
                    "text": "给小猫戴个帽子并跳舞",
                    "gateway_metadata": {"trace": "ignored"},
                }
            ],
            "resolution": "768P",
            "duration": 10,
            "ratio": "16:9",
            "num_inference_steps": 8,
            "aigc_watermark": False,
        }
    )
    core = to_core_request(generation)
    assert core.task == "t2va"
    assert core.prompt == "给小猫戴个帽子并跳舞"
    assert core.conditions == []
    assert "aigc_watermark" not in generation.model_dump()


def test_media_role_is_still_validated_after_text_role_is_relaxed():
    with pytest.raises(ValidationError, match="requires role"):
        request(
            content=[
                text_item(),
                {
                    "type": "image_url",
                    "role": "user",
                    "image_url": {"url": "https://example.com/first.png"},
                },
            ]
        )


def test_text_only_rejects_adaptive_ratio():
    with pytest.raises(ValidationError, match="non-adaptive"):
        request(ratio="adaptive")


def test_first_frame_maps_to_fl2va_and_adaptive():
    data = request(
        content=[
            text_item(),
            {
                "type": "image_url",
                "role": "first_frame",
                "image_url": {"url": "https://example.com/first.png"},
            },
        ],
        ratio="adaptive",
        resolution="704P",
        num_inference_steps=8,
    )
    core = to_core_request(data)
    assert core.task == "fl2va"
    assert core.target.short_edge == 704
    assert core.target.aspect_ratio == "auto"
    assert core.num_inference_steps == 9
    assert core.conditions[0].frame_index == 0


def test_reference_media_maps_to_ref2va():
    data = request(
        content=[
            text_item(),
            {
                "type": "image_url",
                "role": "reference_image",
                "image_url": {"url": "https://example.com/ref.png"},
            },
            {
                "type": "audio_url",
                "role": "reference_audio",
                "audio_url": {"url": "https://example.com/ref.wav"},
            },
        ],
        ratio="adaptive",
    )
    core = to_core_request(data)
    assert core.task == "ref2va"
    assert [item.type for item in core.conditions] == ["image", "audio"]


def test_reference_video_does_not_require_an_audio_track():
    data = request(
        content=[
            text_item(),
            {
                "type": "video_url",
                "role": "reference_video",
                "video_url": {"url": "https://example.com/ref.mp4"},
            },
        ],
        ratio="adaptive",
    )
    core = to_core_request(data)
    assert core.conditions[0].type == "video"


def test_rejects_mixed_keyframe_and_reference_media():
    with pytest.raises(ValidationError, match="cannot be mixed"):
        request(
            content=[
                text_item(),
                {
                    "type": "image_url",
                    "role": "first_frame",
                    "image_url": {"url": "https://example.com/first.png"},
                },
                {
                    "type": "image_url",
                    "role": "reference_image",
                    "image_url": {"url": "https://example.com/ref.png"},
                },
            ]
        )


def test_failed_task_returns_explicit_error():
    task = task_payload(
        {
            "id": "task-id",
            "status": "failed",
            "created_at": 100,
            "completed_at": 120,
            "error": {"message": "CUDA out of memory"},
            "_business": {"resolution": "768P", "duration": 5, "ratio": "16:9"},
        }
    )
    assert task["status"] == "failed"
    assert task["error"] == {
        "type": "upstream_error",
        "message": "CUDA out of memory",
        "http_code": 500,
    }


def test_expired_task_is_a_terminal_status_without_content():
    task = task_payload(
        {
            "id": "task-id",
            "status": "expired",
            "created_at": 100,
            "completed_at": 120,
            "expired_at": 200,
            "_business": {"resolution": "768P", "duration": 5, "ratio": "16:9"},
        }
    )
    assert task["status"] == "expired"
    assert task["updated_at"] == 200
    assert "content" not in task


def test_server_exposes_gateway_routes():
    from app.business import router

    paths = {route.path for route in router.routes if hasattr(route, "path")}
    assert "/ic/capcut/edit_gateway/v2/video_generation" in paths
    assert "/ic/capcut/edit_gateway/v2/query/video_generation" in paths
    assert "/sync_infer" in paths


def test_api_image_keeps_gateway_connections_alive():
    dockerfile = Path(__file__).parents[2] / "docker" / "Dockerfile.api"
    contents = dockerfile.read_text()
    assert '"--timeout-keep-alive", "120"' in contents


def test_gateway_validation_error_is_explicit_json():
    from app.server import app

    response = TestClient(app, raise_server_exceptions=False).post(
        "/ic/capcut/edit_gateway/v2/video_generation",
        json={
            "model": "wrong-model",
            "content": [{"type": "text", "text": "hello"}],
            "resolution": "invalid-resolution",
            "duration": 5,
            "ratio": "16:9",
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["http_code"] == 400


def test_gateway_not_found_error_is_explicit_json():
    from app.server import app

    response = TestClient(app, raise_server_exceptions=False).post(
        "/ic/capcut/edit_gateway/v2/query/video_generation",
        json={"model": "wrong-model", "task_id": "missing-task"},
    )
    assert response.status_code == 404
    assert response.json()["error"] == {
        "type": "not_found_error",
        "message": "task 'missing-task' not found",
        "http_code": 404,
    }
