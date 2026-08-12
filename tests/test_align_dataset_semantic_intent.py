import json

import h5py
import numpy as np

from data.align_dataset import ALIGNDataset
from data.semantic_intent_labels import (
    EventLabel,
    EventProvenance,
    Grounding,
    SemanticFrame,
    SemanticIntentSidecar,
    save_episode_labels,
    semantic_sidecar_dir,
)


def write_minimal_dataset(path):
    with h5py.File(path, "w") as h5:
        ep = h5.create_group("ep_000")
        frames = ep.create_group("frames")
        frames.create_dataset(
            "wrist_image",
            data=np.zeros((8, 8, 8, 3), dtype=np.uint8),
        )
        ep.create_dataset("poses", data=np.zeros((8, 6), dtype=np.float32))
        ep.create_dataset("actions", data=np.zeros((8, 7), dtype=np.float32))
        ep.create_dataset("texts", data=json.dumps(["episode task"]))


def write_sidecar(dataset_path):
    sidecar_path = semantic_sidecar_dir(dataset_path)
    event = EventLabel(
        start=0,
        end=8,
        frame=SemanticFrame(
            operation="grasp",
            theme_name="black bowl",
            relation="held_by",
            reference_name="gripper",
        ),
        theme_grounding=Grounding(
            camera="wrist_image",
            frame_offset=2,
            kind="point",
            value=(0.5, 0.5),
        ),
        reference_grounding=Grounding(
            camera="wrist_image",
            frame_offset=2,
            kind="point",
            value=(0.6, 0.6),
        ),
        confidence=0.9,
        provenance=EventProvenance(
            label_source="frozen_vlm",
            model_id="test-vlm",
            prompt_version="v1",
            source_frame_ids=(0, 4, 7),
            raw_response="{}",
        ),
    )
    save_episode_labels(
        sidecar_path / "ep_000.json",
        SemanticIntentSidecar(1, "ep_000", (event,)),
    )


def test_dataset_auto_detects_semantic_sidecar_and_returns_current_event(tmp_path):
    dataset_path = tmp_path / "sample.h5"
    write_minimal_dataset(dataset_path)
    write_sidecar(dataset_path)

    dataset = ALIGNDataset(
        str(dataset_path),
        camera="wrist_image",
        frames_per_ep=2,
        traj_window=2,
    )
    item = dataset[0]

    assert item["semantic_event"] is not None
    assert item["semantic_event"]["frame"]["theme_name"] == "black bowl"
    assert item["semantic_event"]["event_id"] == 0


def test_dataset_remains_backward_compatible_without_semantic_sidecar(tmp_path):
    dataset_path = tmp_path / "sample.h5"
    write_minimal_dataset(dataset_path)

    dataset = ALIGNDataset(
        str(dataset_path),
        camera="wrist_image",
        frames_per_ep=2,
        traj_window=2,
    )
    item = dataset[0]

    assert item["semantic_event"] is None
