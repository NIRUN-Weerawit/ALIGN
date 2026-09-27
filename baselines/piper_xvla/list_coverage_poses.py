"""List candidate drag-teach poses from retained episodes for the pose-coverage test."""
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def main() -> None:
    cache = torch.load("data/piper_replay/piper_xvla_prepared_cache.pt", map_location="cpu", weights_only=True)
    state = cache["state"].numpy()  # [cap, 8]: xyz + quat(xyzw) + normalized gripper
    episodes = cache["episode_index"].numpy()
    available = cache["available"].numpy()
    manifest = json.load(open("data/piper_replay/xvla_training_manifest.json"))
    gmin = manifest["gripper_normalization"]["raw_meters_min"]
    gmax = manifest["gripper_normalization"]["raw_meters_max"]

    rows = []
    for e in manifest["source_episode_ids"]:
        mask = available * (episodes == e)
        idx = np.where(mask)[0]
        if len(idx) < 100:
            continue
        for label, i in (("start", idx[5]), ("mid", idx[len(idx) // 2]), ("end", idx[-6])):
            s = state[i]
            xyz = s[:3]
            euler = Rotation.from_quat(s[3:7]).as_euler("xyz", degrees=True)
            grip_m = gmin + float(s[7]) * (gmax - gmin)
            rows.append((int(e), label, xyz, euler, grip_m))

    print(f"{'ep':>4} {'label':>5}  xyz (m)                          euler deg             grip_mm")
    for e, label, xyz, euler, grip_m in rows:
        if label == "mid" and e not in (12, 30, 60, 90, 112):
            continue
        print(
            f"{e:>4} {label:>5}  [{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}]   "
            f"[{euler[0]:7.1f},{euler[1]:7.1f},{euler[2]:7.1f}]  {grip_m * 1000:6.1f}"
        )


if __name__ == "__main__":
    main()
