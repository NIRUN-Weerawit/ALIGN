"""Camera preflight for Piper X-VLA collection.

Grabs one frame from each configured device, verifies it is a valid non-black
RGB image, and saves labeled frames plus a contact sheet under
camera_checks/preflight_<timestamp>/. Exits 0 when both cameras pass, 2 otherwise.

Read-only: touches no robot hardware.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "config" / "piper_cameras.json"


def grab_frame(device: str, warmup_frames: int = 5) -> np.ndarray | None:
    """Open a V4L2 device, drop warm-up frames, return one BGR frame or None."""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"FAIL {device}: cannot open")
        return None
    frame = None
    for _ in range(max(1, warmup_frames)):
        ok, frame = cap.read()
        if not ok:
            frame = None
    cap.release()
    return frame


def check_camera(device: str, role: str) -> tuple[np.ndarray | None, dict]:
    """Return (frame_or_None, result_dict)."""
    frame = grab_frame(device)
    result: dict = {"device": device, "role": role, "ok": False, "reason": ""}
    if frame is None:
        result["reason"] = "no frame captured"
        return None, result
    h, w = frame.shape[:2]
    brightness = float(frame.reshape(-1, 3).mean())
    std = float(frame.std())
    result.update(width=w, height=h, brightness=round(brightness, 1), contrast=round(std, 1))
    if brightness < 8.0:
        result["reason"] = f"frame is (nearly) black: mean={brightness:.1f}"
        return frame, result
    if std < 4.0:
        result["reason"] = f"frame has no texture (lens covered?): std={std:.1f}"
        return frame, result
    result["ok"] = True
    return frame, result


def annotate(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--global", dest="global_device", help="override global camera device")
    parser.add_argument("--wrist", dest="wrist_device", help="override wrist camera device")
    parser.add_argument("--out-dir", default=str(MODULE_DIR / "camera_checks"))
    args = parser.parse_args()

    config: dict = {}
    if Path(args.config).exists():
        config = json.loads(Path(args.config).read_text())
    global_device = args.global_device or config.get("global_camera", {}).get("device")
    wrist_device = args.wrist_device or config.get("wrist_camera", {}).get("device")
    if not global_device or not wrist_device:
        print(f"FAIL: missing camera devices (config={args.config})")
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"preflight_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    results = []
    all_ok = True
    for role, device in (("global", global_device), ("wrist", wrist_device)):
        frame, result = check_camera(device, role)
        detail = result.get("reason") or f"{result['width']}x{result['height']} mean={result['brightness']} std={result['contrast']}"
        print(f"{'PASS' if result['ok'] else 'FAIL'} {role} {device}: {detail}")
        if frame is not None:
            label = f"{role} {device}" + ("" if result["ok"] else f"  [{result['reason']}]")
            path = out_dir / f"{role}_{Path(device).name}.jpg"
            cv2.imwrite(str(path), annotate(frame, label))
            result["frame_path"] = str(path)
            panels.append(cv2.resize(annotate(frame, label), (640, 480)))
        if not result["ok"]:
            all_ok = False
        results.append(result)

    sheet_path = out_dir / "contact_sheet.jpg"
    if panels:
        cv2.imwrite(str(sheet_path), np.hstack(panels))
    summary = {
        "ok": all_ok,
        "global_device": global_device,
        "wrist_device": wrist_device,
        "results": results,
        "contact_sheet": str(sheet_path) if panels else None,
    }
    print(json.dumps(summary, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
