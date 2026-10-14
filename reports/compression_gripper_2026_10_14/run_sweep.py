import subprocess, pathlib, json, time
root=pathlib.Path('/home/whinnoy/ALIGN')
report=root/'reports/compression_gripper_2026_10_14'
rows=[]
for dim,fraction,name in [(8,0.,'d8_uniform'),(8,.25,'d8_transition'),(16,.25,'d16_transition'),(32,.25,'d32_transition')]:
    out=root/'checkpoints/compression_gripper_2026_10_14'/name
    cmd=['/home/whinnoy/.venvs/align-ablation/bin/python','scripts/run_intention_ablation.py','--output',str(out),'--variants','no_intent_no_memory','--use-task-text','--compressed-dim',str(dim),'--gripper-transition-fraction',str(fraction),'--epochs','10','--max-steps','64','--batch-size','2','--micro-batch-size','1','--num-workers','0','--gpu-memory-fraction','.3','--plain-head-batch-size','16']
    start=time.time()
    with (report/f'{name}.log').open('w') as log:
        result=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    rows.append(dict(name=name,compressed_dim=dim,transition_fraction=fraction,returncode=result.returncode,wall_s=time.time()-start,command=cmd))
    (report/'sweep.json').write_text(json.dumps(rows,indent=2))
    print(name,result.returncode,flush=True)
