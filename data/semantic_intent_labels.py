"""Validated sidecar records for event-level semantic-intent pseudo-labels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple


@dataclass(frozen=True)
class EventInterval:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("event interval requires 0 <= start < end")


@dataclass(frozen=True)
class SemanticFrame:
    operation: str
    theme_name: str
    relation: str
    reference_name: Optional[str] = None
    constraint: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in ("operation", "theme_name", "relation"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        for field_name in ("reference_name", "constraint"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be None or a non-empty string")


@dataclass(frozen=True)
class Grounding:
    camera: str
    frame_offset: int
    kind: str
    value: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.camera, str) or not self.camera.strip():
            raise ValueError("camera must be a non-empty string")
        if self.frame_offset < 0:
            raise ValueError("frame_offset must be non-negative")
        if self.kind not in {"point", "box"}:
            raise ValueError("grounding kind must be 'point' or 'box'")

        expected = 2 if self.kind == "point" else 4
        if len(self.value) != expected:
            word = "two" if expected == 2 else "four"
            raise ValueError(f"{self.kind} grounding must contain {word} coordinates")

        if any(not 0.0 <= float(coord) <= 1.0 for coord in self.value):
            raise ValueError("grounding coordinates must lie in [0, 1]")

        if self.kind == "box":
            x1, y1, x2, y2 = self.value
            if x2 <= x1 or y2 <= y1:
                raise ValueError("box must satisfy x2 > x1 and y2 > y1")


@dataclass(frozen=True)
class EventProvenance:
    label_source: str
    model_id: str
    prompt_version: str
    source_frame_ids: Tuple[int, ...]
    raw_response: str

    def __post_init__(self) -> None:
        for field_name in ("label_source", "model_id", "prompt_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if any(frame_id < 0 for frame_id in self.source_frame_ids):
            raise ValueError("source_frame_ids must be non-negative")
        if not isinstance(self.raw_response, str):
            raise ValueError("raw_response must be a string")


@dataclass(frozen=True)
class EventLabel:
    start: int
    end: int
    frame: SemanticFrame
    theme_grounding: Grounding
    reference_grounding: Optional[Grounding]
    confidence: float
    provenance: EventProvenance

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end <= self.start:
            raise ValueError("end must be greater than start for a half-open interval")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")


@dataclass(frozen=True)
class SemanticIntentSidecar:
    schema_version: int
    episode_key: str
    events: Tuple[EventLabel, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported semantic-intent schema_version")
        if not isinstance(self.episode_key, str) or not self.episode_key.strip():
            raise ValueError("episode_key must be a non-empty string")

        previous_end = -1
        for event in self.events:
            if event.start < previous_end:
                raise ValueError("semantic-intent events must not overlap")
            previous_end = event.end


def _grounding_from_dict(data: Optional[dict[str, Any]]) -> Optional[Grounding]:
    if data is None:
        return None
    return Grounding(
        camera=data["camera"],
        frame_offset=int(data["frame_offset"]),
        kind=data["kind"],
        value=tuple(float(x) for x in data["value"]),
    )


def _event_from_dict(data: dict[str, Any]) -> EventLabel:
    frame = data["frame"]
    provenance = data["provenance"]
    theme_grounding = _grounding_from_dict(data["theme_grounding"])
    if theme_grounding is None:
        raise ValueError("theme_grounding is required")

    return EventLabel(
        start=int(data["start"]),
        end=int(data["end"]),
        frame=SemanticFrame(**frame),
        theme_grounding=theme_grounding,
        reference_grounding=_grounding_from_dict(data.get("reference_grounding")),
        confidence=float(data["confidence"]),
        provenance=EventProvenance(
            label_source=provenance["label_source"],
            model_id=provenance["model_id"],
            prompt_version=provenance["prompt_version"],
            source_frame_ids=tuple(int(x) for x in provenance["source_frame_ids"]),
            raw_response=provenance["raw_response"],
        ),
    )


def semantic_sidecar_dir(h5_path: Path | str) -> Path:
    """Derive ``<dataset-stem>.semantic_intent/`` without a new CLI flag."""
    path = Path(h5_path)
    return path.with_name(f"{path.stem}.semantic_intent")


def event_for_timestep(
    sidecar: SemanticIntentSidecar,
    timestep: int,
) -> Optional[EventLabel]:
    """Return the event containing ``timestep`` under half-open semantics."""
    if timestep < 0:
        return None
    for event in sidecar.events:
        if event.start <= timestep < event.end:
            return event
        if event.start > timestep:
            break
    return None


def segment_gripper_events(
    commands: Sequence[float],
    *,
    threshold: float = 0.5,
    debounce: int = 2,
    min_event_length: int = 1,
) -> Tuple[EventInterval, ...]:
    """Split a command sequence at sustained binary gripper-state changes.

    Intervals are half-open. A candidate state must remain unchanged for
    ``debounce`` samples before its transition is accepted; the boundary is
    placed at the first sample of that sustained run.
    """
    if debounce < 1:
        raise ValueError("debounce must be at least one")
    if min_event_length < 1:
        raise ValueError("min_event_length must be at least one")
    if not commands:
        return ()

    states = [float(value) >= threshold for value in commands]
    boundaries = [0]
    stable_state = states[0]
    run_state = stable_state
    run_start = 0

    for index in range(1, len(states)):
        state = states[index]
        if state != run_state:
            run_state = state
            run_start = index

        run_length = index - run_start + 1
        if run_state != stable_state and run_length >= debounce:
            boundaries.append(run_start)
            stable_state = run_state

    boundaries.append(len(states))

    # Merge short intervals into their neighbors. Repeating until stable handles
    # adjacent short intervals without producing zero-length events.
    while len(boundaries) > 2:
        lengths = [b - a for a, b in zip(boundaries, boundaries[1:])]
        short_index = next(
            (i for i, length in enumerate(lengths) if length < min_event_length),
            None,
        )
        if short_index is None:
            break

        if short_index == 0:
            del boundaries[1]
        elif short_index == len(lengths) - 1:
            del boundaries[-2]
        else:
            # Remove the boundary that merges the short interval with the
            # longer adjacent interval. Ties merge backward deterministically.
            left_length = lengths[short_index - 1]
            right_length = lengths[short_index + 1]
            boundary_index = short_index if left_length >= right_length else short_index + 1
            del boundaries[boundary_index]

    return tuple(
        EventInterval(start, end)
        for start, end in zip(boundaries, boundaries[1:])
    )


def save_episode_labels(path: Path | str, sidecar: SemanticIntentSidecar) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(sidecar), indent=2, ensure_ascii=False) + "\n")


def load_episode_labels(path: Path | str) -> SemanticIntentSidecar:
    data = json.loads(Path(path).read_text())
    return SemanticIntentSidecar(
        schema_version=int(data["schema_version"]),
        episode_key=data["episode_key"],
        events=tuple(_event_from_dict(event) for event in data["events"]),
    )
