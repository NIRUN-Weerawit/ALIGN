"""Portable frozen-summary + causal-encoder inference; no controller integration."""
from dataclasses import dataclass

import torch
from torch import nn

from models.task_context_teacher import DemonstrationSummary
from models.task_context_encoder import TaskContextEncoder, ContextStreamState


@dataclass
class ObservationStream:
    context: ContextStreamState
    last_time: torch.Tensor | None = None


class TaskContextObserver(nn.Module):
    def __init__(self, summary_config, encoder_config):
        super().__init__()
        self.summary_config = summary_config
        self.summary = DemonstrationSummary(**summary_config)
        self.encoder = TaskContextEncoder(**encoder_config)
        self.register_buffer("state_mean", torch.zeros(10))
        self.register_buffer("state_std", torch.ones(10))

    def new_stream(self, batch_size=1):
        return ObservationStream(self.encoder.new_stream(batch_size))

    @torch.no_grad()
    def observe(self, visual, geometric_state, elapsed_seconds, stream):
        """Normalized DINO CLS/2x2 slots B,V,5,D; raw geometric state B,10.

        elapsed_seconds is time since episode start, not a fraction of its length.
        Its clock must continue across later handover. A new stream resets it.
        """
        b = visual.shape[0]
        expected = (b, self.summary_config["cameras"], 5, self.summary_config["raw_dim"])
        if visual.shape != expected or geometric_state.shape != (b, 10) or elapsed_seconds.shape != (b,):
            raise ValueError("Invalid observation shape")
        if not all(torch.isfinite(x).all() for x in (visual, geometric_state, elapsed_seconds)):
            raise ValueError("An observation must have finite camera/state/time inputs")
        if (elapsed_seconds < 0).any() or (stream.last_time is not None and (elapsed_seconds <= stream.last_time).any()):
            raise ValueError("Observation times must increase within an episode")
        if b != stream.context.layers[0][0].shape[0]:
            raise ValueError("Stream batch size changed; allocate an independent stream")
        state = (geometric_state - self.state_mean) / self.state_std
        summary = self.summary(visual[:, None], state[:, None], elapsed_seconds[:, None])[:, 0]
        result = self.encoder.step(summary, stream.context)
        stream.last_time = elapsed_seconds.detach().clone()
        return result


def load_observer(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle["schema"] != "align-causal-task-observer-v1":
        raise ValueError("Unknown task observer bundle")
    model = TaskContextObserver(bundle["summary_config"], bundle["encoder_config"])
    model.summary.load_state_dict(bundle["summary"], strict=True)
    model.encoder.load_state_dict(bundle["encoder"], strict=True)
    model.state_mean.copy_(bundle["state_mean"])
    model.state_std.copy_(bundle["state_std"])
    return model.to(device).requires_grad_(False).eval(), bundle["metadata"]
