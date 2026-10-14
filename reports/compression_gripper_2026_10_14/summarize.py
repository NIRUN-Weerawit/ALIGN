import json, pathlib, statistics, hashlib, subprocess
root=pathlib.Path('/home/whinnoy/ALIGN')
out=root/'reports/compression_gripper_2026_10_14'
probe={r['name']:r for r in json.loads((out/'transition_probe.json').read_text())}
rows=[]
for name in ['d8_uniform','d8_transition','d16_transition','d32_transition']:
 run=root/'checkpoints/compression_gripper_2026_10_14'/name/'no_intent_no_memory'
 config=json.loads((run/'config.json').read_text())
 metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
 sha=hashlib.sha256()
 with (run/'training_last.pt').open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):sha.update(chunk)
 rows.append(dict(name=name,width=config['compressed_dim'],transition_fraction=config['gripper_transition_fraction'],parameters=config['trainable_parameters'],final=metrics[-1],median_train_s=statistics.median(r['timing']['train_wall_s'] for r in metrics),median_epoch_s=statistics.median(r['timing']['total_wall_s'] for r in metrics),peak_gpu_gib=max(r['peak_gpu_gib'] for r in metrics),probe=probe[name]['counts'],checkpoint_sha256=sha.hexdigest()))
(out/'summary.json').write_text(json.dumps(rows,indent=2))
lines=['# Causal gripper and compression pilot — 2026-10-15','',
'## Changes','',
'- Shared gripper reader uses measured observations or previous executed action; missing measurements no longer make the cached loader feed constant zero. Initial state uses the episode attribute (default 0), and padded measured states carry the last value.',
'- Dense and sparse cached loaders share the corrected convention. Manifests record `gripper_state_source`; legacy manifests require explicit `legacy_zero` for compatible resume/evaluate-existing.',
'- Configurable `--gripper-transition-fraction` reserves training supervision for actual command changes while keeping the number of anchors fixed. Validation stays uniform. Default remains 0 (uniform); opt-in 0.25 is an experimental recall/false-switch tradeoff, not a proven universal improvement.',
'- New HDF5-backed tests cover measured precedence, causal fallback, crop boundaries, initial state, padding, sparse history, legacy mode, supervision budget and validation invariance. 75 tests passed across the relevant suites (74 initially, plus one new CLI test).',
'- Real-data verification: all 428 episodes / 52,042 frames satisfy input[t] == action[t-1] for t>0. There are 685 command changes and 32,401 nonzero causal input frames.',
'', '## Matched GPU pilots','',
'All runs: text enabled, no intent or memory, history 1, diffusion horizon 8, seed 42, same 384/44 episode split, 16 anchors/episode, batch 2 with microbatch 1, AdamW lr 0.0001. Each run used **10 shortened epochs of 64 microbatches: 320 optimizer updates, 640 episode presentations**. Full held-out validation runs every epoch. Results below use the final epoch, not independently selected best checkpoints. Width changes model capacity as well as compression. Existing unrelated GPU workloads were left running, so timing is approximate.',
'', '| Width | Transition fraction | Parameters | Position MSE | Rotation MSE | Chunk gripper accuracy | Switch recall | Stable false-switch rate | Median training s/epoch |',
'|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 p=r['probe'];f=r['final'];lines.append(f"| {r['width']} | {r['transition_fraction']} | {r['parameters']:,} | {f['pos_mse']:.4f} | {f['rot_mse']:.4f} | {f['grip_acc']:.1%} | {p['switch']['correct']}/{p['switch']['n']} ({p['switch']['accuracy']:.1%}) | {p['stable']['false_switch']/p['stable']['n']:.1%} | {r['median_train_s']:.2f} |")
lines += ['',
'The separate switch probe uses all 72 eligible true transitions plus uniform stable anchors across the 44 held-out episodes. It excludes frame zero because there is no prior command. Predictions use matched diffusion noise per episode, 10 sampling steps, recorded observations, and the corrected causal input. Stable false-switch rates use 688 anchors. This is teacher-forced prediction, not simulator task success.',
'', '## Interpretation and next decision','',
'1. The loader defect is fixed and independently verified. Existing trained weights must be retrained to learn the corrected input convention.',
'2. Transition supervision improves width-8 switch recall from 31/72 to 38/72, but increases false switches from 145/688 to 176/688 and reduces average chunk gripper accuracy. Keep the knob visible; evaluate timing tolerance and physical grasps before adopting it as a quality improvement. Uniform control is `--gripper-transition-fraction 0`.',
'3. Width 32 is the strongest early candidate for overall action error and average gripper accuracy; width 16 has no clear overall advantage. Width 32 does not beat width 8 on switch recall. It has 3.7x as many trainable parameters as width 8, and this comparison does not isolate a pure information-bottleneck effect.',
'4. These models are undertrained relative to previous 200-epoch policies. One short seed does not establish final convergence or rollout success. Compression defaults were not changed. Use widths 8 and 32 for a full-budget paired follow-up with uniform/transition controls, then standardized closed-loop evaluation.',
'5. These compression tests cover the text-only plain architecture. Flat memory attention grows much faster with width; the result is not a recommendation to increase memory dimensions blindly.',
'', '## Reproduction and artifacts','',
'Exact commands, return codes, and wall times: `sweep.json`. Logs and `run_sweep.py`, `probe.py`, `summarize.py` are in this directory. `summary.json` includes final checkpoint SHA256, metrics and timing. Checkpoints are in `checkpoints/compression_gripper_2026_10_14/<case>/no_intent_no_memory/`. `code_changes.diff` records the production patch. No simulator rollouts were run for these undertrained pilots.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
(out/'code_changes.diff').write_text(subprocess.check_output(['git','diff'],cwd=root,text=True))
print('\n'.join(lines[14:23]))
