"""Plot saved successful-rollout task beliefs and training-only token geometry.

Uses actual saved task tokens. Execution curves use rollout traces when present;
otherwise a separate cached expert-demonstration reference is explicitly labeled.
Scores are uncalibrated and are never used to condition the action policy.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from data.task_context_dataset import file_hash
from eval.task_estimation import TaskLabelEstimator, canonical_task_name
from models.task_context_policy import TaskContextPolicyBranch

SHORT={'put the bowl on the plate':'Bowl → plate','put the wine bottle on the rack':'Wine → rack',
 'put the wine bottle on top of the cabinet':'Wine → cabinet','open the top drawer and put the bowl inside':'Top drawer + bowl',
 'put the bowl on top of the cabinet':'Bowl → cabinet','put the cream cheese in the bowl':'Cheese → bowl',
 'open the middle drawer of the cabinet':'Middle drawer','turn on the stove':'Turn on stove',
 'put the bowl on the stove':'Bowl → stove','push the plate to the front of the stove':'Push plate'}


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def first_stable(mask,steps,window):
    mask=np.asarray(mask,dtype=bool);steps=np.asarray(steps)
    if window<1:raise ValueError('Window must be positive')
    if len(mask)!=len(steps) or (np.diff(steps)!=1).any():raise ValueError('Expected contiguous observation steps')
    if len(mask)<window:return None
    candidates=np.flatnonzero(np.convolve(mask.astype(int),np.ones(window,dtype=int),'valid')==window)
    if not len(candidates):return None
    start=int(candidates[0])
    return dict(start_step=int(steps[start]),confirmed_step=int(steps[start+window-1]))


def fit_view(x,labels):
    shape=x.shape[-1]
    x=x.reshape(-1,shape).astype(np.float64)
    y=np.repeat(labels,len(x)//len(labels))
    scaler=StandardScaler().fit(x)
    z=scaler.transform(x)
    classifier=LogisticRegression(C=.1,class_weight='balanced',max_iter=3000,random_state=42).fit(z,y)
    centroids=np.stack([z[y==group].mean(0) for group in classifier.classes_])
    return scaler,classifier,unit(centroids)


def score_view(model,x):
    scaler,classifier,centroids=model
    z=scaler.transform(np.asarray(x,dtype=np.float64))
    return classifier.predict_proba(z),unit(z)@centroids.T


def signal_metrics(probability,similarity,actual,steps,window,threshold):
    prediction=probability.argmax(1)
    top=np.sort(probability,axis=1)
    geometry=similarity[:,actual]-np.max(np.delete(similarity,actual,axis=1),axis=1)
    correct=prediction==actual
    normalized_entropy=-(probability*np.log(np.maximum(probability,1e-15))).sum(1)/np.log(probability.shape[1])
    return dict(probability=probability,similarity=similarity,prediction=prediction,
        true_score=probability[:,actual],top_score=top[:,-1],margin=top[:,-1]-top[:,-2],
        entropy=normalized_entropy,geometry_gap=geometry,
        first_stable_correct=first_stable(correct,steps,window),
        first_stable_strong_correct=first_stable(correct&(top[:,-1]>=threshold),steps,window),
        first_stable_closest=first_stable(geometry>0,steps,window),
        strong_wrong_fraction=float(np.mean((top[:,-1]>=threshold)&~correct)))


def json_metrics(signal,indices):
    result={k:signal[k] for k in ('first_stable_correct','first_stable_strong_correct','first_stable_closest','strong_wrong_fraction')}
    result['checkpoints']={name:dict(actual_task_score=float(signal['true_score'][i]),
        top_score=float(signal['top_score'][i]),predicted_group=int(signal['prediction'][i]),
        geometry_gap=float(signal['geometry_gap'][i])) for name,i in indices.items() if i is not None}
    return result


def plot_all(cases,out,fps,threshold,window,names,reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    views=['bundle','actor','token_1','token_2','token_3','token_4']
    labels=['Continuing task bundle','Task bundle used by actor','Individual task token 1',
            'Individual task token 2','Individual task token 3','Individual task token 4']
    markers=[Line2D([0],[0],marker='o',color='white',markeredgecolor='black',ls='',label='Handover'),
             Line2D([0],[0],marker='*',color='#ffdd00',markeredgecolor='black',ls='',label='First successful action'),
             Line2D([0],[0],marker='|',color='black',ls='',markersize=10,label='Demonstration end')]
    rowlabels=[c['episode']+': '+SHORT.get(c['task'],c['task']) for c in cases]
    for kind in ('scores','certainty'):
        fig,axes=plt.subplots(3,2,figsize=(20,17),sharex=True,sharey=True,layout='constrained')
        for ax,view,label in zip(axes.flat,views,labels):
            values=[]
            for c in cases:
                s=c['signals'][view]
                if kind=='scores':values.append(s['true_score'])
                else:values.append(np.where(s['top_score']<threshold,0,np.where(s['prediction']==c['actual'],1,2)))
            x=np.full((len(values),max(len(value) for value in values)),np.nan)
            for i,value in enumerate(values):x[i,:len(value)]=value
            opts=dict(cmap='viridis',vmin=0,vmax=1) if kind=='scores' else dict(cmap=ListedColormap(['#d0d4d8','#009E73','#D55E00']),vmin=0,vmax=2)
            im=ax.imshow(x,aspect='auto',interpolation='nearest',extent=(-.5/fps,(x.shape[1]-.5)/fps,len(cases)-.5,-.5),**opts)
            for i,c in enumerate(cases):
                for point,marker,color in [(c['switch'],'o','white'),(c['success'],'*','#ffdd00'),(c['reference']-1,'|','black')]:
                    ax.scatter(point/fps,i,marker=marker,c=color,s=34,edgecolor='black' if marker!='|' else None,lw=.5,zorder=5)
            ax.set_title(label);ax.set_yticks(range(len(cases)),rowlabels,fontsize=8)
            ax.set_xlabel('Elapsed observation time (seconds, 20 Hz); blank = episode ended')
        if kind=='scores':fig.colorbar(im,ax=axes,label='Recorded task decoder score (uncalibrated)',shrink=.65,pad=.015)
        else:
            cb=fig.colorbar(im,ax=axes,ticks=[0,1,2],shrink=.65,pad=.015)
            cb.ax.set_yticklabels([f'Top score < {threshold:g}','Strong + correct','Strong + wrong'])
        fig.suptitle(f'{len(cases)} successful rollouts — '+('task evidence over time' if kind=='scores' else 'strong estimates can be correct or wrong')+
                     '\n'+reference,fontsize=15)
        fig.legend(handles=markers,loc='outside lower center',ncol=3)
        for ext in ('png','svg'):fig.savefig(out/f'successful_token_{kind}.{ext}',dpi=145)
        plt.close(fig)
    colors=plt.get_cmap('tab10').colors
    for c in cases:
        t=c['steps']/fps;bundle=c['signals']['bundle'];actual=c['actual']
        fig,axes=plt.subplots(4,1,figsize=(15,14),sharex=True,layout='constrained')
        for group,name in enumerate(names):
            axes[0].plot(t,bundle['probability'][:,group],color=colors[group],lw=2.4 if group==actual else 1,
                         alpha=1 if group==actual else .6,label=SHORT.get(name,name))
        axes[0].plot(t,c['signals']['actor']['true_score'],'k--',lw=1.5,label='Actor-used score for recorded task')
        axes[0].set(ylabel='Decoder score',ylim=(-.03,1.03),title='All ten task hypotheses from the continuing bundle; black dashed = actor snapshot')
        axes[0].legend(ncol=4,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.02))
        for i in range(4):
            axes[1].plot(t,c['signals'][f'token_{i+1}']['true_score'],label=f'Task token {i+1}',lw=1.6)
        if 'execution' in c['signals']:
            axes[1].plot(t,c['signals']['execution']['true_score'],label='Execution token',ls='--',c='black')
        axes[1].set(ylabel='Recorded-task score',ylim=(-.03,1.03),title='Separate training-only probes; score scales are not calibrated across probes')
        axes[1].legend(ncol=5,fontsize=9)
        for view,label in zip(['bundle','token_1','token_2','token_3','token_4'],['Bundle','Token 1','Token 2','Token 3','Token 4']):
            axes[2].plot(t,c['signals'][view]['geometry_gap'],label=label,lw=2 if view=='bundle' else 1,alpha=1 if view=='bundle' else .75)
        axes[2].axhline(0,color='black',lw=.8)
        axes[2].set(ylabel='True − best wrong similarity',title='Distance-based check: positive means the recorded task has the closest training centroid')
        axes[2].legend(ncol=5,fontsize=9)
        axes[3].plot(t,bundle['top_score'],label='Top task score',c='#0072B2')
        axes[3].plot(t,bundle['margin'],label='Top-two score margin',c='#CC79A7')
        axes[3].plot(t,1-bundle['entropy'],label='1 − normalized entropy',c='#009E73')
        axes[3].axhline(threshold,c='grey',ls=':',label=f'Display threshold {threshold:g}')
        strong=bundle['top_score']>=threshold
        axes[3].fill_between(t,0,1,where=strong&(bundle['prediction']==actual),color='#009E73',alpha=.12)
        axes[3].fill_between(t,0,1,where=strong&(bundle['prediction']!=actual),color='#D55E00',alpha=.18)
        axes[3].set(ylabel='Score / concentration',xlabel='Elapsed observation time (seconds, 20 Hz)',ylim=(-.03,1.03),
                    title='Green shade = strong correct estimate; orange = strong wrong estimate; neither is calibrated confidence')
        axes[3].legend(ncol=4,fontsize=9)
        for ax in axes:
            ax.axvline(c['switch']/fps,c='black',ls='--',lw=1)
            ax.axvline(c['success']/fps,c='#D55E00',ls='-',lw=1)
            ax.axvline((c['reference']-1)/fps,c='grey',ls=':',lw=1)
            ax.grid(alpha=.15)
        fig.suptitle(f"{c['episode']}: {c['task']} — successful rollout\n"
                     f"Dashed black: handover {c['switch']/fps:.2f}s; orange: success {c['success']/fps:.2f}s; "
                     f"dotted grey: demonstration end. Scores are uncalibrated.",fontsize=13)
        for ext in ('png','svg'):fig.savefig(out/f"{c['episode']}_task_evidence.{ext}",dpi=145)
        plt.close(fig)


def cached_plot(cases,dev,models,out,names):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    query_labels=['0%','10%','25%','40%','50%','70%','90%','1s','2s']
    views=['bundle','token_1','token_2','token_3','token_4','execution','summary','cls','states','time']
    labels=['Task bundle','Task token 1','Task token 2','Task token 3','Task token 4','Execution token',
            'Current observation summary','Current camera CLS','Current robot geometry','Elapsed time only']
    ix=[dev['keys'].tolist().index(c['episode']) for c in cases]
    actual=np.array([c['actual'] for c in cases])
    scores={};correct={}
    for view in views:
        if view=='bundle':x=dev['tokens'][ix].reshape(len(ix),9,-1)
        elif view.startswith('token_'):x=dev['tokens'][ix,:,int(view[-1])-1]
        elif view=='time':x=dev['times'][ix,:,None]
        else:x=dev[view][ix]
        p,_=score_view(models[view],x.reshape(-1,x.shape[-1]))
        p=p.reshape(len(ix),9,-1)
        scores[view]=p[np.arange(len(ix))[:,None],np.arange(9)[None],actual[:,None]]
        correct[view]=p.argmax(-1)==actual[:,None]
    fig,axes=plt.subplots(2,5,figsize=(22,15),sharey=True,layout='constrained')
    for ax,view,label in zip(axes.flat,views,labels):
        im=ax.imshow(scores[view],aspect='auto',vmin=0,vmax=1,cmap='viridis')
        ax.set_title(label,fontsize=11);ax.set_xticks(range(9),query_labels,rotation=65,fontsize=8)
        ax.axvline(6.5,color='white',lw=1,ls='--')
        ax.set_yticks(range(len(ix)),[c['episode']+': '+SHORT.get(c['task'],c['task']) for c in cases],fontsize=8)
    fig.colorbar(im,ax=axes,label='Recorded-task score (uncalibrated)',shrink=.65)
    fig.suptitle('Separate reference: cached expert demonstrations corresponding to the successful rollout episode IDs\n'
                 'These are demonstration observations, not simulator rollout execution-token traces. All probes fit 340 training episodes only.',fontsize=14)
    for ext in ('png','svg'):fig.savefig(out/f'expert_reference_other_tokens.{ext}',dpi=145)
    plt.close(fig)
    return dict(queries=query_labels,scores={k:v.tolist() for k,v in scores.items()},correct={k:v.tolist() for k,v in correct.items()})


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--token-probe',type=Path,required=True)
    p.add_argument('--task-estimator',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--score-threshold',type=float,default=.8)
    p.add_argument('--stable-seconds',type=float,default=1.)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output directory')
    if not 0<args.score_threshold<1 or args.stable_seconds<=0:raise ValueError('Invalid display threshold or stable duration')
    torch.set_num_threads(2)
    result=json.loads(args.result.read_text())
    if not result.get('evaluation_complete'):raise ValueError('Require a completed evaluation')
    checkpoint=Path(result['checkpoint']);checkpoint=checkpoint if checkpoint.is_absolute() else ROOT/checkpoint
    if file_hash(checkpoint)!=result['evaluation_protocol']['checkpoint_sha256']:raise ValueError('Policy hash mismatch')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);cfg=saved['config']
    branch=TaskContextPolicyBranch(config=cfg['task_context_config']).requires_grad_(False).eval()
    branch.load_state_dict({k[len('task_context.'):]:v for k,v in saved['model_state_dict'].items() if k.startswith('task_context.')},strict=True)
    estimator=TaskLabelEstimator(args.task_estimator);estimator.validate_branch(branch)
    if file_hash(args.task_estimator)!=result['evaluation_protocol']['task_estimator']['sha256']:raise ValueError('Decoder mismatch')
    train=dict(np.load(args.token_probe/'train_features.npz',allow_pickle=False))
    dev=dict(np.load(args.token_probe/'development_features.npz',allow_pickle=False))
    if file_hash(args.token_probe/'train_features.npz')!=estimator.metadata['train_features_sha256']:raise ValueError('Training feature hash mismatch')
    for arrays,part in ((train,'train'),(dev,'development')):
        if len(set(arrays['keys']))!=len(arrays['keys']) or set(arrays['keys'])!=set(cfg['task_context_split'][part+'_episodes']):
            raise ValueError('Protected split mismatch')
    success=[r for r in result['episodes'] if r.get('success') is True and r.get('policy_eligible',True)]
    if not success:raise ValueError('No successful policy episodes')
    if not set(r['episode'] for r in success)<=set(dev['keys']):raise ValueError('Success selection must be development-only')
    names=estimator.task_names;fps=float(branch.fps);window=max(1,round(fps*args.stable_seconds))
    x=train['tokens'];models={}
    with threadpool_limits(limits=2):
        for view in ['bundle','token_1','token_2','token_3','token_4','execution','summary','cls','states','time']:
            features=x.reshape(len(x),x.shape[1],-1) if view=='bundle' else (x[:,:,int(view[-1])-1] if view.startswith('token_') else
                train['times'][:,:,None] if view=='time' else train[view])
            models[view]=fit_view(features,train['groups'])
        cases=[];records=[];source_hashes={};max_score_error=0.
        for row in success:
            key=row['episode'];path=args.result.parent/f'{key}_task_estimates.npz';sidecar=path.with_suffix('.json')
            metadata=json.loads(sidecar.read_text())
            if metadata['summary']!=row['task_estimation']:raise ValueError(f'{key}: stale task sidecar')
            source_hashes[str(path)]=file_hash(path);source_hashes[str(sidecar)]=file_hash(sidecar)
            with np.load(path,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
            if arrays['task_names'].tolist()!=names or str(arrays['actual_task'])!=row['task_name']:raise ValueError('Task vocabulary mismatch')
            steps=arrays['steps'];actual=estimator.name_to_group[canonical_task_name(row['task_name'])]
            if len(steps)!=row['n_steps'] or not np.array_equal(steps,np.arange(row['n_steps'])):raise ValueError('Trace coverage mismatch')
            inferred,actor=arrays['inferred_tokens'],arrays['policy_tokens']
            p=estimator.probabilities(inferred);q=estimator.probabilities(actor)
            err=max(float(abs(p-arrays['probabilities'][:,0]).max()),float(abs(q-arrays['probabilities'][:,1]).max()))
            max_score_error=max(max_score_error,err)
            if err>1e-8:raise ValueError('Saved decoder scores do not match raw task tokens')
            norm=(inferred-estimator.latent_mean)/estimator.latent_std
            actor_norm=(actor-estimator.latent_mean)/estimator.latent_std
            signals={}
            for view,tokens,probability in [('bundle',norm,p),('actor',actor_norm,q)]:
                _,similarity=score_view(models['bundle'],tokens.reshape(len(tokens),-1))
                signals[view]=signal_metrics(probability,similarity,actual,steps,window,args.score_threshold)
            for i in range(4):
                probability,similarity=score_view(models[f'token_{i+1}'],norm[:,i])
                signals[f'token_{i+1}']=signal_metrics(probability,similarity,actual,steps,window,args.score_threshold)
            if 'execution_tokens' in arrays:
                probability,similarity=score_view(models['execution'],arrays['execution_tokens'])
                signals['execution']=signal_metrics(probability,similarity,actual,steps,window,args.score_threshold)
            switch=row['switch_step'];completion=row['first_success_step']
            peak_mode=(row.get('task_context_selection') or {}).get('strategy')=='highest_decoder_score'
            if (row['task_context_mode']=='handover' or row['task_context_mode']=='freeze' and not peak_mode) and not np.array_equal(actor[switch:],np.broadcast_to(actor[switch],actor[switch:].shape)):
                raise ValueError('Actor task snapshot is not constant after frozen handover')
            if peak_mode:
                if not all(k in arrays for k in ('snapshot_steps','snapshot_scores','snapshot_updated','candidate_scores')):
                    raise ValueError('Highest-score trace is missing snapshot provenance')
                chosen=np.maximum.accumulate(arrays['candidate_scores'])
                if not np.allclose(chosen,arrays['snapshot_scores'],rtol=0,atol=1e-12):
                    raise ValueError('Snapshot scores do not match the best observed score')
                if not np.array_equal(actor,inferred[arrays['snapshot_steps']]):
                    raise ValueError('Actor tokens do not match selected source observations')
            cases.append(dict(episode=key,task=row['task_name'],actual=actual,steps=steps,signals=signals,
                switch=switch,success=completion,reference=row['reference_steps']))
            indices=dict(handover=switch,before_first_success=completion,final=len(steps)-1)
            records.append(dict(episode=key,task=row['task_name'],actual_group=actual,switch_step=switch,
                first_success_step=completion,execution_rollout_available='execution' in signals,
                views={view:json_metrics(signal,indices) for view,signal in signals.items()}))
            print('ANALYZED',key,flush=True)
        args.output.mkdir(parents=True)
        protocol=dict(schema='align-successful-token-evidence-v1',reference=str(args.result),reference_sha256=file_hash(args.result),
            checkpoint_sha256=file_hash(checkpoint),estimator_sha256=file_hash(args.task_estimator),
            train_features_sha256=file_hash(args.token_probe/'train_features.npz'),development_features_sha256=file_hash(args.token_probe/'development_features.npz'),
            source_sha256=file_hash(__file__),episodes=[r['episode'] for r in success],
            selection='All successful, policy-eligible development episodes from the specified completed evaluation',fps=fps,
            display_threshold=args.score_threshold,stable_window_observations=window,probabilities_calibrated=False,
            confirmation_opened=False,neural_weights_updated=False,rollout_rerun=False,
            probe_training='340 train episodes; all nine declared prefixes pooled; balanced logistic regression C=0.1, fixed seed42',
            geometry='Cosine similarity to task centroids after training-only StandardScaler; positive true-vs-best-other gap means recorded task is nearest',
            stability='Retrospective start of a consecutive window; confirmed_step is when that window has actually been observed',
            saved_probability_max_reconstruction_error=max_score_error,trace_sha256=source_hashes)
        for view,(scaler,classifier,centroids) in models.items():
            np.savez_compressed(args.output/f'probe_{view}.npz',feature_mean=scaler.mean_,feature_scale=scaler.scale_,
                weight=classifier.coef_,bias=classifier.intercept_,classes=classifier.classes_,centroids=centroids)
        plot_all(cases,args.output,fps,args.score_threshold,window,names,args.result.parent.name)
        cached=cached_plot(cases,dev,models,args.output,names)
    summary={}
    for view in ['bundle','actor','token_1','token_2','token_3','token_4']:
        signals=[c['signals'][view] for c in cases]
        def strong_wrong(s,c,start,end):
            return bool(np.any((s['prediction'][start:end]!=c['actual']) &
                               (s['top_score'][start:end]>=args.score_threshold)))
        summary[view]=dict(correct_at_handover=sum(s['prediction'][c['switch']]==c['actual'] for s,c in zip(signals,cases)),
            correct_before_first_success=sum(s['prediction'][c['success']]==c['actual'] for s,c in zip(signals,cases)),
            strong_correct_at_handover=sum(s['prediction'][c['switch']]==c['actual'] and s['top_score'][c['switch']]>=args.score_threshold for s,c in zip(signals,cases)),
            stable_strong_correct_by_handover=sum(s['first_stable_strong_correct'] is not None and s['first_stable_strong_correct']['confirmed_step']<=c['switch'] for s,c in zip(signals,cases)),
            episodes_with_strong_wrong_estimate=sum(s['strong_wrong_fraction']>0 for s in signals),
            episodes_strong_wrong_before_handover=sum(strong_wrong(s,c,0,c['switch']) for s,c in zip(signals,cases)),
            episodes_strong_wrong_handover_through_success=sum(strong_wrong(s,c,c['switch'],c['success']+1) for s,c in zip(signals,cases)),
            episodes_strong_wrong_after_first_success=sum(strong_wrong(s,c,c['success']+1,len(c['steps'])) for s,c in zip(signals,cases)))
    report=dict(protocol=protocol,episodes=records,summary=summary,cached_expert_reference=cached)
    (args.output/'result.json').write_text(json.dumps(report,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else bool(x) if isinstance(x,np.bool_) else float(x))+'\n')
    with (args.output/'timings.csv').open('w') as f:
        fields=['episode','task','view','stable_correct_start_s','stable_strong_correct_start_s','strong_confirmed_s','closest_start_s','handover_s','first_success_s','strong_wrong_fraction']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c in cases:
            for view,s in c['signals'].items():
                def at(key,field='start_step'):return s[key][field]/fps if s[key] is not None else ''
                writer.writerow(dict(episode=c['episode'],task=c['task'],view=view,stable_correct_start_s=at('first_stable_correct'),
                    stable_strong_correct_start_s=at('first_stable_strong_correct'),strong_confirmed_s=at('first_stable_strong_correct','confirmed_step'),
                    closest_start_s=at('first_stable_closest'),handover_s=c['switch']/fps,first_success_s=c['success']/fps,strong_wrong_fraction=s['strong_wrong_fraction']))
    lines=['# Task evidence during successful policy episodes','',f"Reference: `{args.result}`. All {len(cases)} successful policy-eligible development episodes; {result['success_count']}/{len(result['episodes'])} task completions overall.",
        '',f"A strong estimate means top decoder score ≥ {args.score_threshold:g}; stable means {window} consecutive observations ({args.stable_seconds:g}s). These are display criteria, not calibrated certainty or takeover criteria.",
        '', '![Recorded-task scores](successful_token_scores.png)','', '![Strong correct versus strong wrong estimates](successful_token_certainty.png)','',
        '| Representation | Correct at handover | Strong correct at handover | Stable strong estimate confirmed by handover | Correct before successful action | Episodes with any strong wrong estimate |',
        '|---|---:|---:|---:|---:|---:|']
    columns=('correct_at_handover','strong_correct_at_handover','stable_strong_correct_by_handover',
             'correct_before_first_success','episodes_with_strong_wrong_estimate')
    for view,stats in summary.items():
        lines.append('| '+view+' | '+' | '.join(f'{stats[k]}/{len(cases)}' for k in columns)+' |')
    lines+=['','| Representation | Strong wrong before handover | Strong wrong from handover through first success | Strong wrong after first success |',
            '|---|---:|---:|---:|']
    for view,stats in summary.items():
        lines.append('| '+view+' | '+' | '.join(f'{stats[k]}/{len(cases)}' for k in
            ('episodes_strong_wrong_before_handover','episodes_strong_wrong_handover_through_success','episodes_strong_wrong_after_first_success'))+' |')
    lines+=['','The actor row shows the task tokens actually used. Legacy handover snapshots stay fixed; highest-score snapshots change only on a new score record. '
        'Continuing-estimator drift therefore does not imply the actor changed its task. '
        'The action that first completes the task follows the plotted observation at first_success_step. '
        'Selecting successes does not establish how these signals distinguish failures or predict success.', '',
        'The four individual-token classifiers are newly fitted diagnostic probes using training episodes only. The full bundle uses the exact decoder saved during evaluation. '
        'Separate probes can have different score scales. The centroid gap supplies an additional geometric comparison, not a probability.', '',
        f"Execution tokens were present in {sum(r['execution_rollout_available'] for r in records)}/{len(cases)} saved rollout traces. "
        'When absent, their rollout trajectory cannot be recovered from the task tokens. Future evaluations now log the execution token.', '',
        '## Other representations: separate expert-demonstration reference','',
        'The following figure uses the matching episode IDs from cached expert demonstrations. These are different observations from simulator rollouts; '
        'the execution, observation-summary, camera, robot-state and time curves must not be interpreted as rollout measurements.', '',
        '![Expert reference representations](expert_reference_other_tokens.png)','',
        '## Individual successful episodes','',
        '| Episode | Recorded task | First stable strong bundle estimate (window start → confirmation) | Handover | First success | Detailed curves |',
        '|---|---|---:|---:|---:|---|']
    for c in cases:
        timing=c['signals']['bundle']['first_stable_strong_correct']
        text=f"{timing['start_step']/fps:.2f}s → {timing['confirmed_step']/fps:.2f}s" if timing else 'Never'
        lines.append(f"| {c['episode']} | {c['task']} | {text} | {c['switch']/fps:.2f}s | {c['success']/fps:.2f}s | [Plot]({c['episode']}_task_evidence.png) |")
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2,default=int),flush=True)
    print('Saved',args.output,flush=True)


if __name__=='__main__':main()
