"""Head-only task-token interventions with per-observation noise and payload audits."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval.context_interventions import intervene

ARMS = ('normal', 'repeat', 'same_task', 'wrong_task', 'no_task')


def array_hash(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


class PassiveEnvironmentAudit:
    """Record observations returned by LIBERO without refreshing/replacing them."""
    def __init__(self, env, alternative=None):
        self.env_original = env
        self.initial = None
        self.steps = []
        self.goal_trace=[]
        self.alternative=alternative
        self.alternative_reason=None
        actual=getattr(env,'env',env)
        self.goal_env=actual
        if alternative:
            requested=alternative['definition']
            current=actual.parsed_problem
            goal=requested['goal_state']
            entities={arg for p in goal for arg in p[1:]}
            if requested['problem_name']!=current['problem_name']:
                self.alternative_reason='Different environment domain'
            elif any(len(p) not in (2,3) for p in goal):
                self.alternative_reason='Unsupported goal structure'
            elif not entities<=set(actual.object_states_dict):
                self.alternative_reason='Required objects or goal regions absent from actual scene'
            elif any(requested['regions'].get(x)!=current['regions'].get(x) for x in entities
                     if x in requested['regions'] or x in current['regions']):
                self.alternative_reason='Goal-region definitions differ'

    def observe_goal(self, step):
        if self.alternative and self.alternative_reason is None:
            value=all(self.goal_env._eval_predicate(p) for p in self.alternative['definition']['goal_state'])
            self.goal_trace.append(dict(after_action_step=step,success=bool(value)))

    def __getattr__(self, name):
        return getattr(self.env_original, name)

    def identity(self, obs):
        model = self.env_original.sim.model
        geometry = hashlib.sha256(b''.join(np.asarray(getattr(model, k)).tobytes()
            for k in ('body_pos', 'body_quat', 'geom_pos', 'geom_quat', 'geom_size'))).hexdigest()
        state = array_hash(self.env_original.sim.get_state().flatten())
        cameras = {k:array_hash(v) for k,v in obs.items() if k.endswith('_image')}
        return dict(geometry=geometry, state=state, cameras=cameras)

    def reset(self, *args, **kwargs):
        obs = self.env_original.reset(*args, **kwargs)
        self.initial = self.identity(obs)
        return obs

    def set_init_state(self, state):
        obs = self.env_original.set_init_state(state)
        self.initial = self.identity(obs)
        self.goal_trace=[]
        self.observe_goal(-1)
        return obs

    def step(self, action):
        result = self.env_original.step(action)
        self.steps.append(self.identity(result[0]))
        self.observe_goal(len(self.steps)-1)
        return result


class AuditedTaskIntervention:
    def __init__(self, head, arm, switch, seed, estimator, replacement=None):
        if arm not in ARMS:
            raise ValueError('Unknown task-token arm')
        if arm in ('same_task','wrong_task') and replacement is None:
            raise ValueError('A native training donor is required')
        self.head,self.arm,self.switch,self.seed = head,arm,int(switch),int(seed)
        self.estimator,self.replacement = estimator,replacement
        self.rows=[]
        self.task_before=[];self.task_supplied=[];self.execution=[];self.conditions=[]
        self.noise=[]

    def __enter__(self):
        self.original_build=self.head._build_per_step_cond
        self.original_sample=self.head.sample
        self.build_override='_build_per_step_cond' in self.head.__dict__
        self.sample_override='sample' in self.head.__dict__
        self.head._build_per_step_cond=self.build
        self.head.sample=self.sample
        return self

    def build(self, visual, state, payload=None):
        step=len(self.rows)
        torch.manual_seed((self.seed + step) % (2**63-1))
        active=step>=self.switch
        supplied=payload.detach().clone()
        arm='baseline'
        replacement=None
        if active and self.arm in ('same_task','wrong_task'):
            replacement=torch.as_tensor(self.replacement,device=payload.device,dtype=payload.dtype)[None]
            supplied[:,:-1]=replacement
            arm='same_task_swap' if self.arm=='same_task' else 'task_swap'
        elif active and self.arm=='no_task':
            arm='no_task'
        if arm=='no_task':
            current,_,execution=self.head.conditioning_components(visual,state,payload)
            condition=current+execution
        else:
            condition=intervene(self.head,self.original_build,visual,state,payload,arm,replacement)
        # The execution token and all current inputs must be untouched by swapping.
        if not torch.equal(supplied[:,-1],payload[:,-1]):
            raise ValueError('Execution token changed')
        self.rows.append(dict(step=step,noise_seed=(self.seed+step)%(2**63-1),
            task_contribution_active=not(active and self.arm=='no_task'),
            visual_sha256=array_hash(visual),state_sha256=array_hash(state),
            before_payload_sha256=array_hash(payload),condition_sha256=array_hash(condition)))
        self.task_before.append(payload[0,:-1].detach().float().cpu().numpy())
        self.task_supplied.append(supplied[0,:-1].detach().float().cpu().numpy())
        self.execution.append(payload[0,-1].detach().float().cpu().numpy())
        self.conditions.append(condition.detach().float().cpu().numpy())
        return condition

    def sample(self, cond, *args, **kwargs):
        # The pinned DDIM implementation draws this tensor first and has eta=0.
        # Copy the generator state, so auditing never advances the real sampler.
        device=cond.device
        generator=torch.Generator(device=device)
        state=torch.cuda.get_rng_state(device) if device.type=='cuda' else torch.get_rng_state()
        generator.set_state(state)
        noise=torch.randn(cond.shape[0],self.head.chunk_size,self.head.action_dim,
            device=device,dtype=torch.float32,generator=generator)
        self.noise.append(array_hash(noise))
        return self.original_sample(cond,*args,**kwargs)

    def __exit__(self,*exc):
        if self.build_override:self.head._build_per_step_cond=self.original_build
        else:del self.head._build_per_step_cond
        if self.sample_override:self.head.sample=self.original_sample
        else:del self.head.sample

    def save(self, path, actual_task, donor):
        if len(self.rows)!=len(self.noise):
            raise ValueError('Expected one DDIM sample per observation')
        before=np.asarray(self.task_before);supplied=np.asarray(self.task_supplied)
        pre=(self.estimator.predict(before[self.switch:self.switch+1])[0]
             if len(before)>self.switch else None)
        used=(self.estimator.predict(supplied[self.switch:self.switch+1])[0]
              if self.arm!='no_task' and pre is not None else None)
        if donor and used is not None:
            if used['predicted_task']!=donor['task']:
                raise ValueError('The supplied native vector no longer decodes to the intended donor task')
            if (used['predicted_task']==actual_task)!=(self.arm=='same_task'):
                raise ValueError('The donor does not satisfy the intended same/wrong task intervention')
        info=dict(arm=self.arm,actual_task=actual_task,handover=self.switch,
            task_at_handover_before=pre,task_at_handover_supplied=used,
            supplied_tokens_used=self.arm!='no_task',donor=donor,
            ground_truth_used_for_donor_selection=self.arm in ('same_task','wrong_task'),
            replacement_is_native_student=True if donor else None,
            task_removed_after_projection=self.arm=='no_task',weights_updated=False,
            note='No-task arrays record the unused native task latents; their task contribution is removed from conditioning.')
        np.savez_compressed(str(path)+'.npz',task_before=before,task_supplied=supplied,
            execution_tokens=np.asarray(self.execution),conditions=np.asarray(self.conditions),
            steps=np.arange(len(self.rows)),noise_sha256=np.asarray(self.noise),
            task_contribution_active=np.asarray([r['task_contribution_active'] for r in self.rows]))
        Path(str(path)+'.json').write_text(json.dumps(dict(summary=info,observations=self.rows),indent=2)+'\n')
        return info


def run_cli(job_path, evaluator_args):
    """Use the standard evaluator unchanged; wrap only the policy rollout/head."""
    from eval import eval_libero_v4_trajectory as ev
    from eval.task_estimation import TaskLabelEstimator
    import sys
    job=json.loads(Path(job_path).read_text())
    bank_path=Path(job['bank'])
    with np.load(bank_path,allow_pickle=False) as data:
        native=data['native_tokens'].copy()
    plan=json.loads(Path(job['plan']).read_text())
    estimator=TaskLabelEstimator(job['estimator'])
    output=Path(job['out_dir'])/'counterfactual'
    output.mkdir(exist_ok=True)
    original=ev.run_model_in_sim
    count=0
    episode_info={}

    def wrapped(**kwargs):
        nonlocal count
        key=job['episodes'][count];count+=1
        row=plan[str(job['seed'])][key]
        model=kwargs['model']
        model.requires_grad_(False).eval()
        if model.text_condition is not None or model.task_context is None:
            raise ValueError('Expected a non-text task-context policy')
        spec=job.get('policy_spec')
        if spec is None:
            if model.use_memory_bank or model.history_size!=1:
                raise ValueError('Expected the pinned cp4/no-memory context policy')
        elif (bool(model.use_memory_bank)!=spec['use_memory_bank'] or
              model.history_size!=spec['history_size'] or
              model.compressed_dim!=spec['compressed_dim']):
            raise ValueError('Loaded policy differs from its predeclared configuration')
        estimator.validate_branch(model.task_context)
        if kwargs['actual_task_name']!=row['actual_task']:
            raise ValueError('Episode order or task differs from the predeclared donor plan')
        switch=int(len(kwargs['expert_actions'])*kwargs['switch_at'])
        donor=row.get(job['arm'])
        replacement=native[donor['index']] if donor else None
        from libero.libero.envs.bddl_utils import robosuite_parse_problem
        wrong_task=row['wrong_task']['task']
        wrong_bddl=ev.get_bddl_path('libero_goal',wrong_task)
        alternative=dict(task=wrong_task,definition=robosuite_parse_problem(wrong_bddl))
        env=PassiveEnvironmentAudit(kwargs['env'],alternative)
        kwargs['env']=env
        if kwargs['action_horizon']!=1 or kwargs['task_context_mode']!='handover':
            raise ValueError('Require action horizon 1 and fixed handover')
        seed=ev.episode_random_seed(job['seed'],key)
        with AuditedTaskIntervention(model.intention_head,job['arm'],switch,seed,estimator,replacement) as hook:
            result=original(**kwargs)
        if len(hook.rows)!=result['n_steps']:
            raise ValueError('Expected one head call per real observation')
        info=hook.save(output/key,row['actual_task'],donor)
        info.update(initial_environment=env.initial,environment_steps=env.steps,
            expert_prefix_actions_sha256=array_hash(result['executed_actions'][:switch]),
            expert_prefix_poses_sha256=array_hash(result['sim_positions'][:switch]),
            alternative_goal=dict(task=wrong_task,available=env.alternative_reason is None,
                unavailable_reason=env.alternative_reason,goal_predicates=alternative['definition']['goal_state'],
                observations=env.goal_trace,
                true_before_handover=any(r['success'] for r in env.goal_trace if r['after_action_step']<switch),
                true_after_handover=any(r['success'] for r in env.goal_trace if r['after_action_step']>=switch)))
        (output/(key+'_audit.json')).write_text(json.dumps(info,indent=2)+'\n')
        episode_info[key]=dict(strategy='counterfactual_'+job['arm'],
            ground_truth_used_for_donor_selection=info['ground_truth_used_for_donor_selection'],
            task_contribution_active=job['arm']!='no_task',donor=donor,
            source='head conditioning intervention; original streaming encoder/cache unchanged')
        return result

    ev.run_model_in_sim=wrapped
    sys.argv=[ev.__file__]+list(evaluator_args)
    ev.main()
    if count!=len(job['episodes']):raise ValueError('Missing planned episodes')
    path=Path(job['out_dir'])/'result.json'
    report=json.loads(path.read_text())
    for row in report['episodes']:
        row['task_token_intervention']=episode_info[row['episode']]
    path.write_text(json.dumps(report,indent=2)+'\n')
