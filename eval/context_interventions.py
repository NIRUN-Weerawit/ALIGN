"""Evaluation-only interventions on the context diffusion head; no weight updates."""
import torch

ARMS = ('baseline', 'repeat_control', 'freeze_task', 'task_mean', 'task_swap',
        'same_task_swap', 'no_task', 'no_execution', 'no_context', 'visual_zero_control', 'no_temporal')


def intervene(head, build, visual, state, payload, arm, replacement=None):
    if arm not in ARMS:
        raise ValueError(f'Unknown intervention: {arm}')
    if payload.shape[-2:] != (getattr(head, 'context_count', head.context_tokens + 1), head.context_dim):
        raise ValueError('Unexpected context payload shape')
    if arm in ('freeze_task', 'task_mean', 'task_swap', 'same_task_swap'):
        if replacement is None or replacement.shape != payload[:, :head.context_tokens].shape:
            raise ValueError('Replacement must match task-token shape')
        changed = payload.clone()
        changed[:, :head.context_tokens] = replacement.to(payload)
        return build(visual, state, changed)
    if arm == 'visual_zero_control':
        return build(torch.zeros_like(visual), state, payload)
    condition = build(visual, state, payload)
    if arm == 'no_temporal' and not getattr(head, 'temporal_tokens', 0):
        raise ValueError('no_temporal requires a policy temporal token')
    if arm in ('no_task', 'no_execution', 'no_context', 'no_temporal') and hasattr(head, 'remove_context'):
        return head.remove_context(condition, task=arm in ('no_task', 'no_context'),
            execution=arm in ('no_execution', 'no_context'),
            temporal=arm in ('no_temporal', 'no_context'))
    components = None
    if arm in ('no_task', 'no_execution', 'no_context') and hasattr(head, 'conditioning_components'):
        components = head.conditioning_components(visual, state, payload)
    if arm in ('no_task', 'no_context'):
        task = (components[1] if components is not None else
                head.context_task_scale * head.context_task(payload[:, :head.context_tokens].flatten(1))[:, None])
        condition = condition - task
    if arm in ('no_execution', 'no_context'):
        execution = (components[2] if components is not None else
                     head.context_execution_scale * head.context_execution(payload[:, -1])[:, None])
        condition = condition - execution
    return condition


def select_donors(keys, groups, group, seed):
    """Deterministic train-only same/different task controls, without outcome selection."""
    import numpy as np
    rng = np.random.default_rng(seed)
    same = np.flatnonzero(groups == group)
    other_groups = np.unique(groups[groups != group])
    if not len(same) or not len(other_groups):
        raise ValueError('Need training donors from the same and a different task')
    other = np.flatnonzero(groups == rng.choice(other_groups))
    return int(rng.choice(same)), int(rng.choice(other))


class StreamingIntervention:
    """Temporarily intercept head conditioning, leaving encoder/observation caches intact.

    Requires exactly one action-head call per real observation (action_horizon=1).
    Noise is keyed by observation step, so every rollout arm gets the same draws.
    """
    def __init__(self, head, arm, switch_step, seed, replacement=None):
        if arm not in ARMS:raise ValueError('Unknown intervention')
        self.head,self.arm,self.switch_step,self.seed = head,arm,switch_step,seed
        self.replacement=replacement
        self.step=0
        self.snapshot=None
        self.audit=[]

    def __enter__(self):
        self.original=self.head._build_per_step_cond
        self.had_override='_build_per_step_cond' in self.head.__dict__
        self.head._build_per_step_cond=self.build
        return self

    def build(self, visual, state, payload=None):
        step=self.step
        self.step+=1
        torch.manual_seed(self.seed+step)
        if step==self.switch_step:self.snapshot=payload[:,:self.head.context_tokens].detach().clone()
        arm=self.arm if step>=self.switch_step else 'baseline'
        replacement=self.snapshot if arm=='freeze_task' else self.replacement
        condition=intervene(self.head,self.original,visual,state,payload,arm,replacement)
        if self.arm in ('baseline','repeat_control'):
            import hashlib
            self.audit.append({name:hashlib.sha256(value.detach().float().cpu().numpy().tobytes()).hexdigest()
                for name,value in dict(visual=visual,state=state,payload=payload,condition=condition).items()})
        return condition

    def __exit__(self,*exc):
        if self.had_override:self.head._build_per_step_cond=self.original
        else:del self.head._build_per_step_cond
