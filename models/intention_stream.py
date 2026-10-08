"""Episode-local observation caches for recurrent intention inference."""
from collections import deque

import torch


class IntentionStream:
    """Encode each real observation once, retaining observation-only Mamba state.

    Construct a new stream at every episode/reset. Initial repetition applies
    only to the action head's visual/state window, never to Mamba recurrence.
    """

    def __init__(self, model, history_size):
        if history_size < 1:
            raise ValueError("history_size must be positive")
        self.model = model
        self.history_size = history_size
        self.cache = None
        self.visual = deque(maxlen=history_size)
        self.state = deque(maxlen=history_size)

    @torch.no_grad()
    def observe(self, frames, robot_state, produce_intent=False):
        result = self.model.encode_step(frames, robot_state, self.cache,
                                        produce_intent=produce_intent)
        visual, state, hidden, self.cache = result[:4]
        if not self.visual:
            self.visual.extend([visual] * self.history_size)
            self.state.extend([state] * self.history_size)
        else:
            self.visual.append(visual)
            self.state.append(state)
        return {"z_v_pooled_seq": torch.stack(list(self.visual), dim=1),
                "z_s_seq": torch.stack(list(self.state), dim=1),
                "h_seq": hidden.unsqueeze(1),
                "intent_emb": result[4] if len(result) == 5 else None}
