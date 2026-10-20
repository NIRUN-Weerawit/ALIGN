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
        self.timestep = 0
        self.visual = deque(maxlen=history_size)
        self.state = deque(maxlen=history_size)

    @torch.no_grad()
    def observe(self, frames, robot_state, produce_intent=False, store_memory=False, camera_mask=None, state_mask=None):
        bank = getattr(self.model,"memory_module",None)
        # Cognitive raw writes need a readout even between action plans.
        produce_intent = produce_intent or (store_memory and getattr(bank,"_has_cognitive",False))
        produce_intent = produce_intent or getattr(self.model, 'task_context', None) is not None
        visibility = {}
        if camera_mask is not None:visibility['camera_mask'] = camera_mask
        if state_mask is not None:visibility['state_mask'] = state_mask
        result = self.model.encode_step(frames, robot_state, self.cache,
                                        produce_intent=produce_intent,**visibility)
        visual, state, hidden, self.cache = result[:4]
        B,device = visual.shape[0],visual.device
        observed = torch.ones(B,device=device,dtype=torch.bool)
        if camera_mask is not None and state_mask is not None:
            observed = (torch.as_tensor(camera_mask,device=device,dtype=torch.bool).reshape(B,-1).any(1) |
                        torch.as_tensor(state_mask,device=device,dtype=torch.bool).reshape(B))
        timestamp = torch.full((B,),float(self.timestep),device=device)
        self.timestep += 1
        if store_memory and bank is not None and hasattr(bank,"observe_only"):
            cognitive = (result[4] if len(result)==5 else None) if getattr(self.model,'task_context',None) is None else None
            bank.observe_only(visual,state,cognitive,observed_mask=observed,timestamp=timestamp)
        if not self.visual:
            self.visual.extend([visual] * self.history_size)
            self.state.extend([state] * self.history_size)
        else:
            self.visual.append(visual)
            self.state.append(state)
        return {"observed_mask":observed,"timestamp":timestamp,"z_v_pooled_seq": torch.stack(list(self.visual), dim=1),
                "z_s_seq": torch.stack(list(self.state), dim=1),
                "h_seq": hidden.unsqueeze(1),
                "intent_emb": result[4] if len(result) == 5 else None}

    def freeze_task(self):
        """Snapshot inferred task at requested handover; recurrence/execution continue."""
        branch = getattr(self.model, 'task_context', None)
        if branch is None:
            raise ValueError('Task snapshot requires the task-context branch')
        branch.freeze_task(self.cache)

    def resume_human(self):
        """Allow a human correction to update task inference without resetting history."""
        if self.cache is not None and getattr(self.model, 'task_context', None) is not None:
            self.cache.task_snapshot = None
