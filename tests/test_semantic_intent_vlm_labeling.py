import json

import pytest

from data.semantic_intent_labels import EventInterval
from scripts.label_semantic_intent_events import (
    EventLabelRequest,
    build_event_label_prompt,
    parse_vlm_event_label,
)


def valid_response():
    return {
        "operation": "grasp",
        "theme_name": "black bowl",
        "relation": "held_by",
        "reference_name": "gripper",
        "constraint": None,
        "theme_grounding": {
            "camera": "image",
            "frame_offset": 4,
            "type": "box",
            "value": [0.1, 0.2, 0.7, 0.8],
        },
        "reference_grounding": {
            "camera": "wrist_image",
            "frame_offset": 4,
            "type": "point",
            "value": [0.5, 0.6],
        },
        "label_confidence": 0.84,
        "boundary_adjustment": None,
    }


def test_parse_vlm_event_label_preserves_raw_response_and_provenance():
    payload = valid_response()
    raw = json.dumps(payload)
    request = EventLabelRequest(
        episode_key="ep_000",
        interval=EventInterval(3, 12),
        source_frame_ids=(3, 7, 11),
        cameras=("image", "wrist_image"),
        gripper_summary="open -> closing",
    )

    event = parse_vlm_event_label(
        raw_response=raw,
        request=request,
        model_id="frozen-vlm-v1",
        prompt_version="semantic-intent-v1",
    )

    assert event.start == 3
    assert event.end == 12
    assert event.frame.theme_name == "black bowl"
    assert event.theme_grounding.kind == "box"
    assert event.provenance.raw_response == raw
    assert event.provenance.source_frame_ids == (3, 7, 11)


def test_parse_vlm_event_label_rejects_prose_wrapped_json():
    raw = "Here is the answer: " + json.dumps(valid_response())
    request = EventLabelRequest(
        episode_key="ep_000",
        interval=EventInterval(0, 8),
        source_frame_ids=(0, 4, 7),
        cameras=("image",),
        gripper_summary="open",
    )

    with pytest.raises(ValueError, match="valid JSON object"):
        parse_vlm_event_label(raw, request, "vlm", "v1")


def test_parse_vlm_event_label_requires_theme_grounding():
    payload = valid_response()
    payload["theme_grounding"] = None
    request = EventLabelRequest(
        episode_key="ep_000",
        interval=EventInterval(0, 8),
        source_frame_ids=(0, 4, 7),
        cameras=("image",),
        gripper_summary="open",
    )

    with pytest.raises(ValueError, match="theme_grounding"):
        parse_vlm_event_label(json.dumps(payload), request, "vlm", "v1")


def test_parse_vlm_event_label_rejects_unavailable_camera():
    payload = valid_response()
    payload["theme_grounding"]["camera"] = "nonexistent"
    request = EventLabelRequest(
        episode_key="ep_000",
        interval=EventInterval(0, 8),
        source_frame_ids=(0, 4, 7),
        cameras=("image", "wrist_image"),
        gripper_summary="open",
    )

    with pytest.raises(ValueError, match="camera"):
        parse_vlm_event_label(json.dumps(payload), request, "vlm", "v1")


def test_prompt_requires_only_immediate_next_subgoal_and_json():
    request = EventLabelRequest(
        episode_key="ep_000",
        interval=EventInterval(0, 8),
        source_frame_ids=(0, 4, 7),
        cameras=("image", "wrist_image"),
        gripper_summary="open -> closing",
    )

    prompt = build_event_label_prompt(request)

    assert "immediate next subgoal" in prompt
    assert "episode-level task goal" in prompt
    assert "JSON" in prompt
    assert "theme_grounding" in prompt
