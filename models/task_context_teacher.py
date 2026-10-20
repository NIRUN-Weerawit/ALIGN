"""Whole-episode task teacher and independently trained observation controls.

This is privileged completed-demonstration encoding, not causal intent inference.
No text, group ID, action target, endpoint target, or episode length ratio is an
encoder feature. Current-only and constant controls share the decoder interface.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from models.intention_head import DiffusionPolicyHead


class TimeFeatures(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.register_buffer("frequencies", torch.exp(torch.linspace(math.log(.05),math.log(5.),width//2)))

    def forward(self, times):
        value = times[...,None] * self.frequencies
        return torch.cat([value.sin(),value.cos()],-1)


class DemonstrationSummary(nn.Module):
    def __init__(self, cameras=2, raw_dim=768, width=256):
        super().__init__()
        self.cls = nn.Sequential(nn.Linear(cameras*raw_dim,256),nn.LayerNorm(256),nn.SiLU())
        self.patch = nn.Linear(raw_dim,64)
        self.state = nn.Sequential(nn.Linear(10,64),nn.SiLU())
        self.time = TimeFeatures(32)
        self.output = nn.Sequential(nn.Linear(256+cameras*4*64+64+32,width),nn.LayerNorm(width),nn.SiLU())

    def forward(self, visual, states, times):
        cls = self.cls(visual[:,:,:,0].flatten(2))
        patch = self.patch(visual[:,:,:,1:]).flatten(2)
        return self.output(torch.cat([cls,patch,self.state(states),self.time(times)],-1))


class TaskContextTeacher(nn.Module):
    def __init__(self, cameras=2, raw_dim=768, width=256, tokens=4, token_dim=128,
                 layers=2, heads=4, horizon=8, procedure_points=8, diffusion_width=64,
                 pooling="learned", decoder_style="flat"):
        super().__init__()
        if pooling not in ('learned','quarters'):raise ValueError('Unknown temporal pooling')
        if decoder_style not in ('flat','temporal'):raise ValueError('Unknown target decoder')
        self.config = dict(cameras=cameras,raw_dim=raw_dim,width=width,tokens=tokens,token_dim=token_dim,
                           layers=layers,heads=heads,horizon=horizon,procedure_points=procedure_points,diffusion_width=diffusion_width,pooling=pooling,decoder_style=decoder_style)
        self.summary = DemonstrationSummary(cameras,raw_dim,width)
        block = nn.TransformerEncoderLayer(width,heads,width*4,dropout=0.,batch_first=True,norm_first=True)
        self.encoder = nn.TransformerEncoder(block,layers,norm=nn.LayerNorm(width),enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(1,tokens,width)*.02)
        self.pool = nn.MultiheadAttention(width,heads,batch_first=True)
        self.readout = nn.Sequential(nn.Linear(width,token_dim),nn.LayerNorm(token_dim))
        flat = tokens*token_dim
        self.constant = nn.Parameter(torch.randn(1,tokens,token_dim)*.02)
        self.target_time = TimeFeatures(32)
        self.decoder = nn.Sequential(nn.Linear(flat+32,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        if decoder_style=='temporal':
            self.local_decoder=nn.Sequential(nn.Linear(token_dim,256),nn.SiLU())
        self.visual_decoder = nn.Linear(256,cameras*5*raw_dim)
        self.state_decoder = nn.Linear(256,10)
        self.group_projection = nn.Sequential(nn.Linear(flat,128),nn.SiLU(),nn.Linear(128,32))
        self.task_condition = nn.Sequential(nn.LayerNorm(flat),nn.Linear(flat,width))
        self.current_condition = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width))
        self.task_gate = nn.Parameter(torch.tensor(.1))
        self.regression = nn.Sequential(nn.Linear(width,256),nn.SiLU(),nn.Linear(256,horizon*7))
        self.diffusion = DiffusionPolicyHead(cond_dim=width,action_dim=7,hidden_dim=diffusion_width,
            num_train_timesteps=100,num_inference_steps=10,chunk_size=horizon,clip_denoised=True,loss_repeats=1)
        self.register_buffer("action_mean",torch.zeros(7))
        self.register_buffer("action_std",torch.ones(7))

    def encode(self, features, states, times, lengths, anchors, mode="full"):
        batch,length = states.shape[:2]
        valid = torch.arange(length,device=states.device)[None] < lengths[:,None]
        # Sanitize masked padding before any projection; NaN*0 is still NaN.
        features = torch.where(valid[:,:,None,None,None],features,torch.zeros_like(features))
        states = torch.where(valid[:,:,None],states,torch.zeros_like(states))
        times = torch.where(valid,times,torch.zeros_like(times))
        x = self.summary(features,states,times)
        index = torch.arange(batch,device=x.device)[:,None]
        current = x[index,anchors]
        count = anchors.shape[1]
        if mode == "constant":
            tokens = self.constant.expand(batch,count,-1,-1)
        elif mode == "full":
            hidden = self.encoder(x,src_key_padding_mask=~valid)
            query = self.queries.expand(batch,-1,-1)
            pool_mask=None
            if self.config['pooling']=='quarters':
                # A structural temporal prior for the privileged teacher: each
                # token reads one ordered interval. Never applied to a causal
                # student/current-only input, which has no episode duration.
                slot=(torch.arange(length,device=x.device)[None]*self.config['tokens']//lengths[:,None]).clamp_max(self.config['tokens']-1)
                pool_mask=slot[:,None].ne(torch.arange(self.config['tokens'],device=x.device)[None,:,None])
                # Short sequences still give every query one real observation.
                nearest=((torch.arange(self.config['tokens'],device=x.device)[None]+.5)*lengths[:,None]/self.config['tokens']).long()
                nearest=torch.minimum(nearest,lengths[:,None]-1)
                empty=(pool_mask | ~valid[:,None]).all(-1)
                bi,qi=empty.nonzero(as_tuple=True)
                pool_mask[bi,qi,nearest[bi,qi]]=False
                pool_mask=pool_mask.repeat_interleave(self.config['heads'],dim=0)
            pooled,_ = self.pool(query,hidden,hidden,key_padding_mask=~valid,attn_mask=pool_mask,need_weights=False)
            tokens = self.readout(pooled)[:,None].expand(-1,count,-1,-1)
        elif mode == "current":
            hidden = self.encoder(current.flatten(0,1)[:,None])
            query = self.queries.expand(batch*count,-1,-1)
            pooled,_ = self.pool(query,hidden,hidden,need_weights=False)
            tokens = self.readout(pooled).reshape(batch,count,self.config["tokens"],self.config["token_dim"])
        else:
            raise ValueError(f"Unknown teacher input mode: {mode}")
        return tokens,current

    def decode(self, tokens, current):
        batch,count = current.shape[:2]
        flat = tokens.flatten(0,1).flatten(1)
        points = self.config["procedure_points"]
        fractions = torch.cat([flat.new_ones(1),torch.linspace(0,1,points,device=flat.device)])
        query = self.target_time(fractions).expand(len(flat),-1,-1)
        hidden = self.decoder(torch.cat([flat[:,None].expand(-1,len(fractions),-1),query],-1))
        if self.config['decoder_style']=='temporal':
            # Time-indexed access makes local procedure evidence recoverable
            # without forcing the global MLP to learn token routing from scratch.
            centers=(torch.arange(self.config['tokens'],device=flat.device)+.5)/self.config['tokens']
            weights=(-((fractions[:,None]-centers[None])/.2).square()).softmax(-1)
            local=torch.einsum('qt,btd->bqd',weights,tokens.flatten(0,1))
            hidden=hidden+self.local_decoder(local)
        visual = self.visual_decoder(hidden).reshape(batch,count,points+1,self.config["cameras"],5,self.config["raw_dim"])
        state = self.state_decoder(hidden).reshape(batch,count,points+1,10)
        cond = self.current_condition(current.flatten(0,1)) + self.task_gate*self.task_condition(flat)
        regression = self.regression(cond).reshape(batch,count,self.config["horizon"],7)
        regression = regression*self.action_std + self.action_mean
        return dict(tokens=tokens,current=current,terminal=visual[:,:,0],procedure=visual[:,:,1:],
                    procedure_states=state[:,:,1:],regression=regression,condition=cond[:,None])

    def forward(self, batch, mode="full", tokens=None):
        encoded,current = self.encode(batch["features"],batch["states"],batch["times"],
                                      batch["lengths"],batch["anchors"],mode)
        return self.decode(encoded if tokens is None else tokens,current)


def group_contrastive(tokens, labels, temperature=.15):
    z = F.normalize(tokens.float(),dim=-1)
    logits = z@z.T/temperature
    eye = torch.eye(len(z),device=z.device,dtype=torch.bool)
    positive = labels[:,None].eq(labels[None]) & ~eye
    valid = positive.any(1)
    if not valid.any():
        return z.sum()*0
    log_probability = logits - torch.logsumexp(logits.masked_fill(eye,-1e9),dim=1,keepdim=True)
    return -(log_probability*positive).sum(1)[valid].div(positive.sum(1)[valid]).mean()


def teacher_loss(model, output, batch, *, diffusion_weight=1.):
    terminal = F.smooth_l1_loss(output["terminal"].float(),batch["terminal"][:,None].expand_as(output["terminal"]).float())
    procedure = F.smooth_l1_loss(output["procedure"].float(),batch["procedure"][:,None].expand_as(output["procedure"]).float())
    state = F.smooth_l1_loss(output["procedure_states"].float(),batch["procedure_states"][:,None].expand_as(output["procedure_states"]).float())
    regression = (((output["regression"].float()-batch["actions"])/model.action_std)**2).mean()
    tokens = output["tokens"].mean(1).flatten(1)
    group = group_contrastive(model.group_projection(tokens),batch["group"])
    variance = F.relu(.5-tokens.float().std(0,unbiased=False)).mean()
    diffusion = (model.diffusion.loss(batch["actions"].flatten(0,1),output["condition"])
                 if diffusion_weight else regression*0)
    loss = diffusion_weight*diffusion + regression + .5*terminal + .5*(procedure+state)/2 + .1*group + .01*variance
    return loss,dict(diffusion=diffusion,regression=regression,terminal=terminal,procedure=procedure,state=state,group=group,variance=variance)
