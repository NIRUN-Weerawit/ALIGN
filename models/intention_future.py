"""Training-only predictive supervision on final contextual intention embeddings."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class IntentionFuturePredictor(nn.Module):
    """Predict fixed future targets from intent alone; never consumes observations."""

    def __init__(self, intent_width, num_cameras, raw_dim, horizons, hidden_dim=256):
        super().__init__()
        self.horizons = tuple(horizons)
        self.visual_width = num_cameras * raw_dim
        self.output_width = 2*self.visual_width + 7
        self.trunk = nn.Sequential(nn.LayerNorm(intent_width),
            nn.Linear(intent_width, hidden_dim), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, self.output_width) for _ in horizons])

    def forward(self, intent, horizon_index):
        return self.heads[horizon_index](self.trunk(intent.flatten(1)))


def future_prediction_loss(model, intent, cls_targets, state_targets, lengths,
                           anchor_mask, action_count):
    """Sum valid horizon losses / (action anchors * configured horizon count).

    That denominator preserves the combined action/auxiliary objective under
    microbatch accumulation even when future-target coverage differs. Targets
    are detached, gathered only at real within-episode timesteps, and computed
    in FP32. No padding or target encoder receives auxiliary gradients.
    """
    predictor = getattr(model, 'intention_future_predictor', None)
    if predictor is None or intent is None:
        raise ValueError('Future prediction requires enabled intention embeddings/predictor')
    anchor_mask = torch.as_tensor(anchor_mask, device='cpu', dtype=torch.bool)
    lengths = torch.as_tensor(lengths, device='cpu', dtype=torch.long)
    if anchor_mask.shape != intent.shape[:2]:
        raise ValueError('Future anchor mask must match the causal intent sequence')
    metrics = {}
    total = intent.new_zeros((), dtype=torch.float32)
    for hi,horizon in enumerate(predictor.horizons):
        valid = anchor_mask & (torch.arange(intent.shape[1])[None] + horizon < lengths[:,None])
        episode,time = valid.nonzero(as_tuple=True)
        count = len(episode)
        metrics[f'pairs_h{horizon}'] = count
        if not count:continue
        episode,time = episode.to(intent.device),time.to(intent.device)
        with torch.autocast(device_type=intent.device.type, enabled=False):
            current_cls = F.layer_norm(cls_targets[episode,time].detach().float(),
                                      (cls_targets.shape[-1],)).flatten(1)
            future_cls = F.layer_norm(cls_targets[episode,time+horizon].detach().float(),
                                     (cls_targets.shape[-1],)).flatten(1)
            current_state = state_targets[episode,time].detach().float()
            delta = state_targets[episode,time+horizon].detach().float()-current_state
            # Pose angles use the shortest signed Euler displacement. Translation
            # scale is one stored position unit, angles pi radians, gripper one unit.
            angular = torch.atan2(torch.sin(delta[:,3:6]),torch.cos(delta[:,3:6]))/math.pi
            state_delta = torch.cat([delta[:,:3],angular,delta[:,6:7]],-1)
            predicted = predictor(intent[episode,time].float(),hi)
            absolute,change,state = torch.split(predicted,[predictor.visual_width,predictor.visual_width,7],-1)
            parts = dict(visual=F.smooth_l1_loss(absolute,future_cls,reduction='none').mean(-1),
                visual_delta=F.smooth_l1_loss(change,future_cls-current_cls,reduction='none').mean(-1),
                state_delta=F.smooth_l1_loss(state,state_delta,reduction='none').mean(-1))
            loss = sum(parts.values())/len(parts)
            persistence=(2*F.smooth_l1_loss(current_cls,future_cls,reduction='none').mean(-1)
                +F.smooth_l1_loss(torch.zeros_like(state_delta),state_delta,reduction='none').mean(-1))/3
            total = total + loss.sum()
            metrics[f'loss_sum_h{horizon}'] = loss.detach().sum()
            metrics[f'persistence_sum_h{horizon}'] = persistence.detach().sum()
            for key,value in parts.items():
                metrics[f'{key}_sum_h{horizon}'] = value.detach().sum()
    return total / max(1, action_count*len(predictor.horizons)), metrics


def freeze_intent_backbone(model):
    """Pilot: preserve existing encoders; train final projection and policy head."""
    if model.intention_encoder is None or model.intention_future_predictor is None:
        raise ValueError('Freezing the intention backbone requires the future objective')
    model.vision_encoder.requires_grad_(False)
    model.vision_patch_encoder.requires_grad_(False)
    model.state_encoder.requires_grad_(False)
    model.intention_encoder.requires_grad_(False)
    model.intention_encoder.intent_proj.requires_grad_(True)
    model._intent_backbone_frozen=True
    # Fixed weights must also supply fixed features: in particular, disable
    # state-encoder dropout even after the trainer calls model.train().
    model.train(model.training)


class FutureMetrics:
    def __init__(self, predictor, device):
        self.horizons=predictor.horizons
        keys=['anchors','action_sum','total_sum','aux_sum']
        keys += [f'{name}_h{h}' for h in self.horizons for name in
                 ['pairs','loss_sum','persistence_sum','visual_sum','visual_delta_sum','state_delta_sum']]
        self.values={key:torch.zeros((),device=device,dtype=torch.float64) for key in keys}

    def update(self, action, auxiliary, count, terms, weight):
        for key,value in terms.items():self.values[key]+=torch.as_tensor(value,device=self.values[key].device).detach()
        self.values['anchors']+=count
        self.values['action_sum']+=action.detach()*count
        self.values['aux_sum']+=auxiliary.detach()*count
        self.values['total_sum']+=(action.detach()+weight*auxiliary.detach())*count

    def finish(self):
        import torch.distributed as dist
        keys=sorted(self.values)
        totals=torch.stack([self.values[key] for key in keys])
        if dist.is_available() and dist.is_initialized():dist.all_reduce(totals)
        values=dict(zip(keys,totals.cpu().tolist()))
        n=max(1,values['anchors'])
        result=dict(action_loss=values['action_sum']/n,total_loss=values['total_sum']/n,
                    intent_future_loss=values['aux_sum']/n)
        for h in self.horizons:
            count=values[f'pairs_h{h}'];result[f'intent_future_pairs_h{h}']=count
            for key in ['loss','persistence','visual','visual_delta','state_delta']:
                result[f'intent_future_{key}_h{h}']=values[f'{key}_sum_h{h}']/max(1,count)
        return result


def add_future_arguments(parser):
    parser.add_argument('--intent-future-weight', type=float, default=0.,
        help='Opt-in auxiliary loss on final intent embeddings; 0 preserves existing training')
    parser.add_argument('--intent-future-horizons', type=int, nargs='+', default=[16,64],
        help='Positive future frame offsets; unavailable targets are excluded')
    parser.add_argument('--intent-future-hidden-dim', type=int, default=256)
    parser.add_argument('--freeze-intent-backbone', action='store_true',
        help='Pilot: freeze vision/state/Mamba, retaining trainable intent projection, future predictor and action head')


def validate_future_arguments(args):
    if not math.isfinite(args.intent_future_weight) or args.intent_future_weight < 0:
        raise ValueError('intent-future-weight must be finite and nonnegative')
    if not args.intent_future_horizons or min(args.intent_future_horizons)<1 or len(set(args.intent_future_horizons))!=len(args.intent_future_horizons):
        raise ValueError('Future horizons must be unique positive frame offsets')
    if args.intent_future_hidden_dim<1:raise ValueError('Future hidden dimension must be positive')
    if args.freeze_intent_backbone and args.intent_future_weight<=0:
        raise ValueError('freeze-intent-backbone requires positive intent-future-weight')
