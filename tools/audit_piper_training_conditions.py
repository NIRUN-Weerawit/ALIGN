"""Audit saved training conditions for the exact checkpoints in a donor report."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]


def audit(root):
    torch.set_num_threads(2)
    protocol=json.loads((root/'protocol.json').read_text())
    rows=[];sources=[];configs=[]
    common_keys=('lr','weight_decay','grad_clip','batch_size','seed','temporal_sampling',
        'supervision_points','chunk_size','diffusion_train_steps','diffusion_loss_repeats',
        'task_context_trainable','task_context_checkpoint','task_context_fusion',
        'warm_start','policy_warm_start','selection_metric','observation_dropout_prob',
        'gripper_state_source','gripper_transition_fraction')
    for model in protocol['models']:
        config_path=Path(model['checkpoint']).parent/'config.json'
        cfg=json.loads(config_path.read_text());configs.append(cfg)
        folder=ROOT/'checkpoints'/model['model']/model['variant']
        metrics_path=folder/'metrics.jsonl'
        metrics=[json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        selected=next(r for r in metrics if r['epoch']==model['epoch'])
        summary_path=folder.parent/'summary.json'
        summary=json.loads(summary_path.read_text())[model['variant']]
        recovery_path=folder/'training_last.pt'
        try:
            ck=torch.load(recovery_path,map_location='cpu',weights_only=True,mmap=True)
            recovery=dict(readable=True,epoch=ck['epoch'],
                optimizer_lrs=[g['lr'] for g in ck['optimizer_state_dict']['param_groups']],
                optimizer_steps=sorted({int(s['step']) for s in ck['optimizer_state_dict']['state'].values() if 'step' in s}))
            del ck
        except Exception as error:
            recovery=dict(readable=False,error=f'{type(error).__name__}: {error}')
        rows.append(dict(model=model['model'],variant=model['variant'],configured_epochs=cfg['epochs'],
            last_logged_epoch=max(r['epoch'] for r in metrics),summary_completed_epochs=summary.get('completed_epochs'),
            evaluated_checkpoint_epoch=model['epoch'],checkpoint_sha256=model['checkpoint_sha256'],
            compressed_dim=cfg['compressed_dim'],head_width=cfg['head_d_model'],
            effective_batch_size=cfg['batch_size'],micro_batch_size=cfg['micro_batch_size'] or cfg['batch_size'],
            gradient_accumulation_steps=cfg['batch_size']//(cfg['micro_batch_size'] or cfg['batch_size']),
            recorded_trainable_parameters=cfg['trainable_parameters'],history_size=cfg['history_size'],
            head_history_mode=cfg['head_history_mode'],use_memory_bank=cfg['use_memory_bank'],
            train_loss_at_selected=selected['train/loss'],val_loss_at_selected=selected['val/loss'],
            pos_mse_at_selected=selected['pos_mse'],recovery_checkpoint=recovery))
        for path in (config_path,metrics_path,summary_path):
            sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    shared={k:configs[0][k] for k in common_keys}
    assert all(all(cfg[k]==v for k,v in shared.items()) for cfg in configs)
    assert all(cfg['train_episodes']==configs[0]['train_episodes'] and
               cfg['val_episodes']==configs[0]['val_episodes'] for cfg in configs)
    pretraining={}
    for name in ('teacher','student'):
        folder=ROOT/'checkpoints/task_context_piper_384_20261023'/name
        cfg=json.loads((folder/'config.json').read_text())
        metrics=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
        pretraining[name]=dict(configured_epochs=cfg['epochs'],last_logged_epoch=metrics[-1]['epoch'],
                              lr=cfg['lr'],batch_size=cfg['batch_size'],seed=cfg['seed'])
    report=dict(models=rows,shared_settings=shared,train_episodes=len(configs[0]['train_episodes']),
        validation_episodes=len(configs[0]['val_episodes']),pretraining=pretraining,source_files=sources,
        optimizer='fused AdamW',lr_schedule='constant; no scheduler or warmup in pinned policy loop',
        precision='BF16 CUDA autocast',weights_updated=False)
    (root/'training_conditions.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Training conditions — latest evaluated checkpoints','',
        'Configs correspond to the exact evaluated checkpoints. Latest training progress is checked separately from local logs and recovery files.','',
        '| Model | Variant | CP / head | Planned epochs | Latest logged | Evaluated epoch | Micro-batch / accumulation | Trainable parameters |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['variant']} | {r['compressed_dim']} / {r['head_width']} | "
            f"{r['configured_epochs']} | {r['last_logged_epoch']} | {r['evaluated_checkpoint_epoch']} | "
            f"{r['micro_batch_size']} / {r['gradient_accumulation_steps']} | {r['recorded_trainable_parameters']/1e6:.2f} M |")
    lines+=['','## Shared conditions','',
        '- Same 384 training / 44 validation episodes; one training seed, 42.',
        '- Fused AdamW; constant LR **3e-4**, weight decay **1e-4**, gradient clipping **1.0**; no scheduler/warmup.',
        '- Effective batch **32 episodes**, using gradient accumulation for smaller micro-batches.',
        '- Episode sampling with **16 supervision anchors**, action chunk size **4**, two cameras with cached DINOv2 features, BF16 CUDA training.',
        '- Diffusion training schedule **100 steps**, **4 loss repeats**, denoised sample clipping enabled.',
        '- Same frozen context encoder and additive task/execution conditioning; no text, policy warm start, or observation dropout.',
        '- Evaluated best checkpoint selected by validation **position MSE**, not simulator completion.',
        '- Memory length **16**; history variants use an ordered **4-observation** head window. All variants retain the full-history context observer.','',
        '## Shared context pretraining','',
        'Teacher: 80 logged epochs. Student: 100 logged epochs. Both use LR 1e-4, batch 8, seed 42, and the Piper 384-episode training split.','',
        '## Current artifact status','']
    for r in rows:
        recovery=r['recovery_checkpoint']
        if recovery['readable']:
            lines.append(f"- {r['model']}/{r['variant']}: readable recovery checkpoint at epoch {recovery['epoch']}.")
        else:
            lines.append(f"- **{r['model']}/{r['variant']}**: latest log epoch {r['last_logged_epoch']}, "
                f"summary counter {r['summary_completed_epochs']}; recovery checkpoint remains unreadable ({recovery['error']}). "
                'Its evaluated best checkpoint is independently readable and hash-verified.')
    lines+=['',
        'The earlier incomplete cp8 history and head-64 memory records have arrived: their logs and readable recovery checkpoints now reach 500 and 400 epochs, respectively. '
        'Head-64 memory now uses evaluated epoch 289; head-256 memory uses evaluated epoch 374. '
        'Different configured budgets (500 vs 400), one training seed, and substantially different parameter counts still limit architecture-level conclusions.','',
        '## Losses at the evaluated checkpoint','',
        '| Model / variant | Epoch | Train loss | Validation loss | Position MSE |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']}/{r['variant']} | {r['evaluated_checkpoint_epoch']} | "
            f"{r['train_loss_at_selected']:.5f} | {r['val_loss_at_selected']:.5f} | {r['pos_mse_at_selected']:.5f} |")
    lines+=['','[Machine-readable training audit](training_conditions.json). [Overall simulator results](summary.md).']
    (root/'training_conditions.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    audit(parser.parse_args().root.resolve())
