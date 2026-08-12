import json
from pathlib import Path

import pytest

from data.semantic_intent_labels import (
    EventLabel,
    EventProvenance,
    Grounding,
    SemanticFrame,
    SemanticIntentSidecar,
    load_episode_labels,
    save_episode_labels,
    segment_gripper_events,
    semantic_sidecar_dir,
    event_for_timestep,
)


def make_event() -> EventLabel:
    return EventLabel(
        start=3,
        end=12,
        frame=SemanticFrame(
            operation="grasp",
            theme_name="black bowl",
            relation="held_by",
            reference_name="gripper",
            constraint=None,
        ),
        theme_grounding=Grounding(
            camera="image",
            frame_offset=5,
            kind="box",
            value=(0.1, 0.2, 0.7, 0.8),
        ),
        reference_grounding=Grounding(
            camera="wrist_image",
            frame_offset=5,
            kind="point",
            value=(0.5, 0.6),
        ),
        confidence=0.84,
        provenance=EventProvenance(
            label_source="frozen_vlm",
            model_id="example-vlm",
            prompt_version="v1",
            source_frame_ids=(3, 7, 11),
            raw_response='{"operation":"grasp"}',
        ),
    )


def test_sidecar_round_trip_preserves_semantics_grounding_and_provenance(tmp_path):
    sidecar = SemanticIntentSidecar(
        schema_version=1,
        episode_key="ep_000",
        events=(make_event(),),
    )
    path = tmp_path / "ep_000.json"

    save_episode_labels(path, sidecar)
    loaded = load_episode_labels(path)

    assert loaded == sidecar
    raw = json.loads(path.read_text())
    assert raw["events"][0]["provenance"]["raw_response"] == '{"operation":"grasp"}'


def test_grounding_rejects_coordinates_outside_normalized_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Grounding(
            camera="image",
            frame_offset=0,
            kind="point",
            value=(1.2, 0.5),
        )


def test_grounding_rejects_wrong_coordinate_count_for_kind():
    with pytest.raises(ValueError, match="box.*four"):
        Grounding(
            camera="image",
            frame_offset=0,
            kind="box",
            value=(0.2, 0.3),
        )


def test_event_rejects_empty_semantic_fields():
    with pytest.raises(ValueError, match="theme_name"):
        SemanticFrame(
            operation="grasp",
            theme_name=" ",
            relation="held_by",
            reference_name="gripper",
            constraint=None,
        )


def test_event_interval_is_half_open_and_nonempty():
    event = make_event()
    with pytest.raises(ValueError, match="end.*start"):
        EventLabel(
            start=event.start,
            end=event.start,
            frame=event.frame,
            theme_grounding=event.theme_grounding,
            reference_grounding=event.reference_grounding,
            confidence=event.confidence,
            provenance=event.provenance,
        )


def test_sidecar_rejects_overlapping_events():
    event = make_event()
    overlapping = EventLabel(
        start=11,
        end=20,
        frame=event.frame,
        theme_grounding=event.theme_grounding,
        reference_grounding=event.reference_grounding,
        confidence=event.confidence,
        provenance=event.provenance,
    )

    with pytest.raises(ValueError, match="overlap"):
        SemanticIntentSidecar(
            schema_version=1,
            episode_key="ep_000",
            events=(event, overlapping),
        )


def test_segment_gripper_events_splits_on_debounced_state_changes():
    commands = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    events = segment_gripper_events(
        commands,
        threshold=0.5,
        debounce=2,
        min_event_length=1,
    )

    assert [(event.start, event.end) for event in events] == [
        (0, 3),
        (3, 6),
        (6, 8),
    ]


def test_segment_gripper_events_ignores_short_command_chatter():
    commands = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    events = segment_gripper_events(
        commands,
        threshold=0.5,
        debounce=2,
        min_event_length=1,
    )

    assert [(event.start, event.end) for event in events] == [(0, 6)]


def test_segment_gripper_events_merges_short_intervals():
    commands = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    events = segment_gripper_events(
        commands,
        threshold=0.5,
        debounce=2,
        min_event_length=3,
    )

    assert [(event.start, event.end) for event in events] == [
        (0, 4),
        (4, 7),
    ]


def test_semantic_sidecar_dir_is_auto_derived_from_dataset_filename(tmp_path):
    dataset = tmp_path / "libero_object.h5"
    assert semantic_sidecar_dir(dataset) == tmp_path / "libero_object.semantic_intent"


def test_event_for_timestep_uses_half_open_intervals():
    first = make_event()
    second = EventLabel(
        start=12,
        end=20,
        frame=first.frame,
        theme_grounding=first.theme_grounding,
        reference_grounding=first.reference_grounding,
        confidence=first.confidence,
        provenance=first.provenance,
    )
    sidecar = SemanticIntentSidecar(
        schema_version=1,
        episode_key="ep_000",
        events=(first, second),
    )

    assert event_for_timestep(sidecar, 11) == first
    assert event_for_timestep(sidecar, 12) == second
    assert event_for_timestep(sidecar, 20) is None
