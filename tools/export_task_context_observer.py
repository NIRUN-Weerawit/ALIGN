"""Export an inference-only observer after a hash-bound confirmation information gate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from data.task_context_dataset import TaskContextEpisodes, file_hash
from data.task_context_prefix_dataset import TaskContextPrefixes
from training.train_task_context_encoder import load_checkpoint
from training.train_task_context_teacher import load_checkpoint as load_teacher


def export_observer(teacher_path, student_path, targets, cache, report_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Observer bundle exists: {output}")
    report = json.loads(Path(report_path).read_text())
    if report.get("split") != "confirmation" or not report.get("gates", {}).get("causal_information", {}).get("passed"):
        raise ValueError("Observer export requires a passing confirmation information gate")
    for name, path in (("teacher", teacher_path), ("full", student_path)):
        if report["checkpoints"][name]["sha256"] != file_hash(path):
            raise ValueError("Observer checkpoint changed since confirmation")
    source = TaskContextEpisodes(targets, "train")
    prefixes = TaskContextPrefixes(cache, "train")
    if report["manifest_sha256"] != prefixes.manifest["manifest_sha256"] or source.manifest["manifest_sha256"] != prefixes.manifest["target_manifest_sha256"]:
        raise ValueError("Observer normalization/target binding mismatch")
    teacher, _ = load_teacher(teacher_path)
    encoder, checkpoint = load_checkpoint(student_path)
    if encoder.config["history"] != "full" or checkpoint["config"]["teacher_sha256"] != file_hash(teacher_path):
        raise ValueError("Expected the confirmed full-prefix student")
    metadata = dict(teacher_sha256=file_hash(teacher_path), student_sha256=file_hash(student_path),
        confirmation_sha256=file_hash(report_path), target_manifest_sha256=source.manifest["manifest_sha256"],
        prefix_manifest_sha256=prefixes.manifest["manifest_sha256"],
        causal_information_gate=report["gates"]["causal_information"], history_advantage_gate=report["gates"]["history_advantage"],
        input_contract="Frozen normalized CLS/2x2 slots in manifest camera order; translation+two matrix columns+previous-gripper fallback; elapsed seconds.",
        cameras=source.manifest["cameras"], fps=source.manifest["fps"],
        limitations=report["limitations"],
        contract="Offline-validated causal representation pilot only. No action head, automatic takeover, or success probability.")
    bundle = dict(schema="align-causal-task-observer-v1",
        summary_config={k: teacher.config[k] for k in ("cameras", "raw_dim", "width")},
        encoder_config=encoder.config, summary=teacher.summary.state_dict(), encoder=encoder.state_dict(),
        state_mean=torch.from_numpy(np.array(source.norm["state_mean"], copy=True)),
        state_std=torch.from_numpy(np.array(source.norm["state_std"], copy=True)), metadata=metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(output)
    return dict(path=str(output), sha256=file_hash(output), **metadata)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "student", "targets", "cache", "report", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    print(json.dumps(export_observer(a.teacher, a.student, a.targets, a.cache, a.report, a.output), indent=2))
