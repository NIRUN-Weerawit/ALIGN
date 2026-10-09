#!/usr/bin/env python3
"""Controlled capacity experiment: only past cue identifies the target action."""
import argparse,json,sys,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.memory_bank import EpisodicMemoryModule
from models.intention_head import DiffusionPolicyHead,FlowMatchingPolicyHead


def prepare(memory,labels,history):
    B=len(labels)
    memory.reset(B,labels.device)
    current=torch.ones(B,8,device=labels.device)
    cue=current.clone();cue[:,0]=1+2*(2*labels-1)
    state=torch.zeros(B,4,device=labels.device)
    memory.observe_only(cue,state,timestamp=torch.zeros(B,device=labels.device))
    for t in range(1,history):
        memory.observe_only(current,state,timestamp=torch.full((B,),float(t),device=labels.device))
    return current,state


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=400)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--patch',action='store_true')
    ap.add_argument('--value-preserving',action='store_true')
    ap.add_argument('--patch-temporal',action='store_true')
    ap.add_argument('--context-only',action='store_true')
    ap.add_argument('--checkpoint',type=Path,help='Evaluate saved benchmark weights without further training')
    ap.add_argument('--head',choices=['linear','diffusion','flow_matching'],default='linear')
    args=ap.parse_args()
    if args.steps < 1:ap.error('--steps must be positive')
    if (args.output/'results.json').exists():raise ValueError('Completed experiment exists; use a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(81)
    memory=EpisodicMemoryModule(8,0,4,bank_len=16,value_preserving=args.value_preserving,patch_temporal=args.patch_temporal,context_only=args.context_only,patch_dim=4 if args.patch else None).to(args.device)
    if args.head=='linear':
        head=nn.Linear(8,1)
    elif args.head=='diffusion':
        head=DiffusionPolicyHead(cond_dim=12,hidden_dim=16,time_dim=16,chunk_size=4,
                                 num_train_timesteps=100,loss_repeats=4,clip_denoised=True)
    else:
        head=FlowMatchingPolicyHead(cond_dim=12,hidden_dim=16,time_dim=16,chunk_size=4)
    head=head.to(args.device)
    if args.checkpoint:
        saved=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
        if saved.get('value_preserving',False) != args.value_preserving:
            raise ValueError('Saved benchmark value representation differs')
        if saved.get('patch_temporal',False) != args.patch_temporal:
            raise ValueError('Saved benchmark patch layout differs')
        if saved.get('context_only',False) != args.context_only:
            raise ValueError('Saved benchmark retrieval mode differs; specify its original --context-only setting')
        memory.load_state_dict(saved['memory']);head.load_state_dict(saved['head'])
    def condition(p,s):return torch.cat([p,s],dim=-1).unsqueeze(1)
    def prediction(p,s,seed):
        if args.head=='linear':return head(p).squeeze(-1)>0,None
        torch.manual_seed(seed)
        action=head.sample(condition(p,s))
        return action[:,:,:3].mean((1,2))>0,action[:,:,6].mean(1)>.5
    opt=torch.optim.AdamW(list(memory.parameters())+list(head.parameters()),lr=.003)
    start=time.monotonic()
    with (args.output/'training.jsonl').open('w',buffering=1) as log:
        for step in range(0 if args.checkpoint else args.steps):
            labels=torch.randint(0,2,(16,),device=args.device).float()
            history=[4,8,16,32][step%4]
            current,state=prepare(memory,labels,history)
            fused,fused_state,_=memory(current,state)
            if args.head=='linear':
                logits=head(fused).squeeze(-1)
                loss=nn.functional.binary_cross_entropy_with_logits(logits,labels)
            else:
                target=(labels[:,None,None]*2-1).expand(-1,4,7).clone()*.5
                target[:,:,6]=labels[:,None]
                loss=head.loss(target,condition(fused,fused_state))
            opt.zero_grad();loss.backward();opt.step()
            if step%50==0:
                log.write(json.dumps(dict(step=step,loss=loss.item(),head=args.head))+'\n')
                print('step',step,'loss',round(loss.item(),4),flush=True)
    results={}
    memory.eval();head.eval()
    with torch.no_grad():
        for history in [8,32,48]:
            counts={'correct':0,'shuffled':0,'bypass':0};grip_counts={k:0 for k in counts};total=0
            for batch in range(16):
                labels=(torch.arange(32,device=args.device)%2).float()
                current,state=prepare(memory,labels,history)
                original={k:getattr(memory,k).clone() for k in ['perceptual_bank','state_bank','cognitive_bank','timestamps','_count','_next_timestep']}
                fp,fs,_=memory(current,state)
                draw_seed=81+history*100+batch
                good,good_grip=prediction(fp,fs,draw_seed)
                for k,v in original.items():setattr(memory,k,v.roll(1,0))
                fp,fs,_=memory(current,state)
                wrong,wrong_grip=prediction(fp,fs,draw_seed)
                bypass,bypass_grip=prediction(current,state,draw_seed)
                for arm,pred in [('correct',good),('shuffled',wrong),('bypass',bypass)]:counts[arm]+=int((pred==labels).sum())
                if args.head!='linear':
                    for arm,pred in [('correct',good_grip),('shuffled',wrong_grip),('bypass',bypass_grip)]:grip_counts[arm]+=int((pred==labels).sum())
                total+=len(labels)
            results[str(history)]={k:v/total for k,v in counts.items()}
            if args.head!='linear':results[str(history)]['gripper_accuracy']={k:v/total for k,v in grip_counts.items()}
    report=dict(protocol='Same current observation/state for both labels; target is cue shown only at first observation. Wrong-bank donors have opposite cue.',
                training_steps=0 if args.checkpoint else args.steps,evaluated_checkpoint=str(args.checkpoint) if args.checkpoint else None,
                sampling="DDIM noise consistent with clipped clean estimate" if args.head=="diffusion" else args.head,patch=args.patch,value_preserving=args.value_preserving,patch_temporal=args.patch_temporal,context_only=args.context_only,head=args.head,results=results,seconds=time.monotonic()-start,
                caveat='Controlled mechanism capacity, not evidence of LIBERO policy benefit.')
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save(dict(memory=memory.state_dict(),head=head.state_dict(),value_preserving=args.value_preserving,patch_temporal=args.patch_temporal,context_only=args.context_only),args.output/'model.pt')
    print(json.dumps(report,indent=2),flush=True)
    if any(r['correct']<.95 or r['correct']-r['shuffled']<.2 for r in results.values()):
        raise RuntimeError('Memory did not pass the delayed-cue acceptance test')

if __name__=='__main__':main()
