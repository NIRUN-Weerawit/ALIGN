"""Close-only mapping for X-VLA's predicted gripper opening."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "piper_xvla_binary_gripper.json"


@dataclass(frozen=True)
class BinaryGripperConfig:
    threshold_mm: float = 20.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold_mm) or not -70 <= self.threshold_mm <= 70:
            raise ValueError("gripper switch threshold must be finite and within [-70, 70] mm")

    @classmethod
    def from_dict(cls, payload: dict) -> "BinaryGripperConfig":
        if not isinstance(payload, dict) or set(payload) not in (
            {"threshold_mm"}, {"close_mm", "threshold_mm", "open_mm"},
        ):
            raise ValueError("gripper config needs threshold_mm")
        try:
            # Accept the old binary-mapping file, preserving its saved switch value.
            return cls(threshold_mm=float(payload["threshold_mm"]))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid gripper config: {exc}") from exc

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def validate_calibration(self, gripper_min_m: float, gripper_max_m: float) -> None:
        if not np.isfinite([gripper_min_m, gripper_max_m]).all() or gripper_min_m >= gripper_max_m:
            raise ValueError("invalid gripper normalization range")
        if not gripper_min_m <= 0.0 <= gripper_max_m or not gripper_min_m <= self.threshold_mm / 1000 <= gripper_max_m:
            raise ValueError(
                f"0 mm close target and switch threshold must be inside the model calibration "
                f"[{gripper_min_m * 1000:.3f}, {gripper_max_m * 1000:.3f}] mm"
            )


def load_binary_gripper_config(path: str | Path = DEFAULT_CONFIG_PATH) -> BinaryGripperConfig:
    return BinaryGripperConfig.from_dict(json.loads(Path(path).read_text()))
