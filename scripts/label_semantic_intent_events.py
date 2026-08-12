#!/usr/bin/env python3
"""Provider-independent frozen-VLM event labeling helpers.

The actual provider adapter should pass the returned prompt and event frames to
its multimodal API, then feed the unmodified text response into
``parse_vlm_event_label``. This module stores no credentials.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

from data.semantic_intent_labels import (
    EventInterval,
    EventLabel,
    EventProvenance,
    Grounding,
    SemanticFrame,
)


@dataclass(frozen=True)
class EventLabelRequest:
    episode_key: str
    interval: EventInterval
    source_frame_ids: Tuple[int, ...]
    cameras: Tuple[str, ...]
    gripper_summary: str

    def __post_init__(self) -> None:
        if not self.episode_key.strip():
            raise ValueError("episode_key must be non-empty")
        if not self.source_frame_ids:
            raise ValueError("source_frame_ids must not be empty")
        if not self.cameras:
            raise ValueError("cameras must not be empty")


def build_event_label_prompt(request: EventLabelRequest) -> str:
    """Return the versioned semantic labeling instruction for one event."""
    return f"""You label one manipulation event from a temporal visual window.
Predict only the immediate next subgoal represented by this event. Do not infer
or output an episode-level task goal or a task phase.

Event metadata:
- episode: {request.episode_key}
- half-open interval: [{request.interval.start}, {request.interval.end})
- provided source frame IDs: {list(request.source_frame_ids)}
- camera names: {list(request.cameras)}
- gripper command summary: {request.gripper_summary}

Return exactly one JSON object and no prose or markdown. Required schema:
{{
  "operation": "short operation label",
  "theme_name": "short open-vocabulary object noun phrase",
  "relation": "desired future-state predicate",
  "reference_name": "short noun phrase or null",
  "constraint": "short phrase or null",
  "theme_grounding": {{
    "camera": "one provided camera name",
    "frame_offset": 0,
    "type": "point or box",
    "value": ["2 normalized numbers for point or 4 for box"]
  }},
  "reference_grounding": null,
  "label_confidence": 0.0,
  "boundary_adjustment": null
}}

Grounding coordinates must be normalized to [0,1]. frame_offset is relative to
the event interval. The theme grounding is always required. Reference grounding
is required when a visible reference is named. Do not write a grammatical task
sentence; fill only the semantic fields.
"""


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _load_strict_json_object(raw_response: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("VLM response must be one valid JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("VLM response must be one valid JSON object")
    return payload


def _parse_grounding(
    value: Any,
    *,
    field_name: str,
    cameras: Sequence[str],
    event_length: int,
    required: bool,
) -> Grounding | None:
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object or null")

    camera_value = value.get("camera")
    if not isinstance(camera_value, str) or camera_value not in cameras:
        raise ValueError(f"{field_name} camera must be one of {tuple(cameras)}")

    frame_offset = value.get("frame_offset")
    if not isinstance(frame_offset, int) or not 0 <= frame_offset < event_length:
        raise ValueError(f"{field_name} frame_offset is outside the event")

    kind_value = value.get("type")
    if not isinstance(kind_value, str):
        raise ValueError(f"{field_name} type must be 'point' or 'box'")

    return Grounding(
        camera=camera_value,
        frame_offset=frame_offset,
        kind=kind_value,
        value=tuple(value.get("value", ())),
    )


def parse_vlm_event_label(
    raw_response: str,
    request: EventLabelRequest,
    model_id: str,
    prompt_version: str,
) -> EventLabel:
    """Parse and validate an unmodified VLM response for one event."""
    payload = _load_strict_json_object(raw_response)
    event_length = request.interval.end - request.interval.start

    theme_grounding = _parse_grounding(
        payload.get("theme_grounding"),
        field_name="theme_grounding",
        cameras=request.cameras,
        event_length=event_length,
        required=True,
    )
    assert theme_grounding is not None

    reference_name = payload.get("reference_name")
    reference_grounding = _parse_grounding(
        payload.get("reference_grounding"),
        field_name="reference_grounding",
        cameras=request.cameras,
        event_length=event_length,
        required=reference_name is not None,
    )

    confidence_value = payload.get("label_confidence")
    if not isinstance(confidence_value, (int, float)):
        raise ValueError("label_confidence must be numeric")

    return EventLabel(
        start=request.interval.start,
        end=request.interval.end,
        frame=SemanticFrame(
            operation=_required_string(payload, "operation"),
            theme_name=_required_string(payload, "theme_name"),
            relation=_required_string(payload, "relation"),
            reference_name=reference_name,
            constraint=payload.get("constraint"),
        ),
        theme_grounding=theme_grounding,
        reference_grounding=reference_grounding,
        confidence=float(confidence_value),
        provenance=EventProvenance(
            label_source="frozen_vlm",
            model_id=model_id,
            prompt_version=prompt_version,
            source_frame_ids=request.source_frame_ids,
            raw_response=raw_response,
        ),
    )
