"""A missing LFS checkout must fail with an actionable checkpoint error."""
import pytest
import torch

from models.task_context_observer import load_observer_bundle


def test_observer_lfs_pointer_reports_checkout_step(tmp_path):
    pointer = tmp_path / "observer_frozen.pt"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "a" * 64 + "\n"
        "size 18580208\n"
    )
    with pytest.raises(ValueError, match=r"Git LFS pointer.*git lfs pull"):
        load_observer_bundle(pointer)


def test_observer_bundle_loads_with_weights_only(tmp_path):
    checkpoint = tmp_path / "observer_frozen.pt"
    torch.save({"metadata": {"fps": 20.0}, "weight": torch.ones(2)}, checkpoint)
    bundle = load_observer_bundle(checkpoint)
    assert bundle["metadata"]["fps"] == 20.0
    torch.testing.assert_close(bundle["weight"], torch.ones(2))
