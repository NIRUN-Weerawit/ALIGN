"""Plot the consolidated normal/GT-donor simulator evaluation as exportable figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

BLUE='#3569B0'
ORANGE='#D77535'
INK='#243449'
MUTED='#596777'
GRID='#E7ECF2'


def model_label(row, details=True):
    family={'piper_context_1':'C1','piper_context_2':'C2',
        'piper_context_2_cp_dim_8':'C2','piper_context_4_cp_dim_8':'C4',
        'piper_context_4_cp_dim_8_head_256':'C4'}[row['model']]
    variant={'context_memory':'memory','context_no_memory':'no memory','context_history':'history'}[row['variant']]
    label=f"{family} · cp{row['policy_spec']['compressed_dim']} · {variant}"
    return label+(f"\nhead {row['head_d_model']} · epoch {row['epoch']}" if details else '')


def heatmap_label(row):
    family=model_label(row,False).split(' · ')[0]
    variant={'context_memory':'mem','context_no_memory':'no mem','context_history':'hist'}[row['variant']]
    return f"{family} cp{row['policy_spec']['compressed_dim']}\n{variant}\nh{row['head_d_model']}"


def style_axis(ax,xgrid=True):
    for name in ('top','right','left'):
        ax.spines[name].set_visible(False)
    ax.spines['bottom'].set_color('#B9C4D1')
    ax.tick_params(axis='both',length=0,labelcolor=INK,pad=8)
    if xgrid:
        ax.xaxis.grid(True,color=GRID,linewidth=.8)
        ax.set_axisbelow(True)


def title(fig,heading,subtitle):
    fig.text(.035,.965,heading,fontsize=18,fontweight='bold',color=INK,va='top')
    fig.text(.035,.92,subtitle,fontsize=10.5,color=MUTED,va='top')


def note(fig,text):
    fig.text(.035,.035,text,fontsize=9,color=MUTED,va='bottom',linespacing=1.5)


def legend(fig,location=(.30,.875)):
    fig.legend(handles=[Patch(color=BLUE,label='Normal inferred context'),
                        Patch(color=ORANGE,label='GT-selected correct-task donor')],
               loc='upper left',bbox_to_anchor=location,ncol=2,frameon=False,fontsize=10)


def success_rates(rows):
    fig,ax=plt.subplots(figsize=(13,8.8))
    fig.subplots_adjust(left=.30,right=.97,top=.82,bottom=.16)
    title(fig,'Task context and policy completion',
          'Eight latest evaluated checkpoints · 44 episodes × 3 evaluation seeds · 40% expert handover')
    legend(fig)
    y=np.arange(len(rows));height=.29
    for arm,offset,color in [('normal',-.17,BLUE),('same_task',.17,ORANGE)]:
        values=[100*r[arm]/r['trials'] for r in rows]
        ax.barh(y+offset,values,height=height,color=color,zorder=3)
        for i,(r,value) in enumerate(zip(rows,values)):
            ax.text(value-1.2,i+offset,f"{r[arm]}/{r['trials']} · {value:.1f}%",
                    ha='right',va='center',color='white',fontsize=9.5)
    ax.set_yticks(y,[model_label(r) for r in rows],fontsize=10)
    ax.set_xlim(0,100);ax.set_ylim(len(rows)-.55,-.6)
    ax.set_xlabel('Policy completion (%)',labelpad=10)
    style_axis(ax)
    note(fig,'Point estimates; paired uncertainty is shown in the donor-benefit figure. GT-selected donors are a privileged diagnostic.\n'
        'cp = visual feature compression dimension; every model uses four 128-D task tokens. One training seed; failed expert replays included.')
    return fig


def donor_benefit(rows):
    fig,ax=plt.subplots(figsize=(13,8.3))
    fig.subplots_adjust(left=.30,right=.735,top=.83,bottom=.17)
    title(fig,'Benefit from a correct-task donor',
          'Paired completion difference: GT-selected donor − normal inferred context')
    y=np.arange(len(rows))
    for i,r in enumerate(rows):
        effect=r['effect'];mean=100*effect['mean'];lo,hi=np.asarray(effect['interval_95'])*100
        assert lo<=mean<=hi
        ax.plot([lo,hi],[i,i],color=ORANGE,linewidth=2.6,zorder=3)
        ax.plot([lo,lo],[i-.07,i+.07],color=ORANGE,linewidth=1.5)
        ax.plot([hi,hi],[i-.07,i+.07],color=ORANGE,linewidth=1.5)
        ax.scatter([mean],[i],s=78,facecolors=ORANGE if lo>0 else 'white',
                   edgecolors=ORANGE,linewidths=2,zorder=4)
        ax.text(1.04,i,f'{mean:+.1f} pp   [{lo:+.1f}, {hi:+.1f}]',
                transform=ax.get_yaxis_transform(),va='center',fontsize=10,color=INK)
    ax.axvline(0,color='#8190A2',linestyle='--',linewidth=1.1)
    ax.set_yticks(y,[model_label(r) for r in rows],fontsize=10)
    ax.set_xlim(-1,27);ax.set_xticks(np.arange(0,26,5));ax.set_ylim(len(rows)-.55,-.6)
    ax.set_xlabel('Completion difference (percentage points)',labelpad=10)
    style_axis(ax)
    fig.legend(handles=[Line2D([],[],marker='o',color=ORANGE,linestyle='none',markersize=7,label='95% interval excludes zero'),
                        Line2D([],[],marker='o',color=ORANGE,markerfacecolor='white',linestyle='none',markersize=7,label='95% interval reaches zero')],
               loc='upper left',bbox_to_anchor=(.30,.885),frameon=False,ncol=2,fontsize=9.5)
    note(fig,'Intervals: 10,000 bootstrap resamples of 44 episode identities, keeping all three evaluation seeds within each cluster.\n'
        'These are repeated trials on known tasks; 132 trials are not 132 independent episodes. GT task labels select native training donors.')
    return fig


def identity_cohorts(rows):
    fig,axes=plt.subplots(1,2,figsize=(15,9),sharey=True)
    fig.subplots_adjust(left=.265,right=.98,top=.80,bottom=.17,wspace=.11)
    title(fig,'Where correct-task donors help',
          'Cohorts are defined by the normal token’s decoded identity before autonomous outcomes')
    legend(fig,location=(.265,.885))
    y=np.arange(len(rows))
    for ax,cohort,heading in zip(axes,['correct','incorrect'],
        ['Originally decoded correctly · 117 trials/model','Originally decoded incorrectly · 15 trials/model']):
        for arm,offset,color in [('normal',-.17,BLUE),('same_task',.17,ORANGE)]:
            for i,row in enumerate(rows):
                r=row['identity_cohorts'][cohort];value=100*r[arm]/r['trials']
                ax.barh(i+offset,value,height=.29,color=color,zorder=3)
                if value==0:
                    ax.scatter([0],[i+offset],color=color,s=9,zorder=4)
                    ax.text(1.2,i+offset,f"0/{r['trials']} · 0%",va='center',fontsize=9,color=INK)
                else:
                    ax.text(value-1.4,i+offset,f"{r[arm]}/{r['trials']} · {value:.1f}%",
                        va='center',ha='right',fontsize=9,color='white')
        ax.set_title(heading,fontsize=11.5,pad=15,color=INK)
        ax.set_xlim(0,103);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Policy completion (%)',labelpad=10)
        style_axis(ax)
    axes[0].set_yticks(y,[model_label(r) for r in rows],fontsize=9.5)
    axes[0].set_ylim(len(rows)-.55,-.6)
    note(fig,'Point estimates. The 15 incorrectly decoded trials span 7 episode identities across three evaluation seeds; the 117 correct trials span 40.\n'
        'A donor replaces the entire task bundle, including any episode-specific procedure information; it does not merely change a task label.')
    return fig


def checkpoint_updates(comparisons):
    changed=[r for r in comparisons if r['previous_epoch']!=r['new_epoch']]
    assert len(changed)==2
    fig,axes=plt.subplots(1,2,figsize=(12.6,6.8),sharey=True)
    fig.subplots_adjust(left=.085,right=.965,top=.76,bottom=.27,wspace=.20)
    title(fig,'Updated checkpoints versus earlier evaluations',
          'Same 44 episodes, three evaluation seeds, simulator initial states and GT-selected donor choices')
    legend(fig,location=(.085,.865))
    for ax,r in zip(axes,changed):
        width=256 if 'head_256' in r['model'] else 64
        for arm,color,marker in [('normal',BLUE,'o'),('same_task',ORANGE,'D')]:
            counts=[r['previous_'+arm],r['new_'+arm]]
            rates=np.asarray(counts)*100/r['trials']
            ax.plot([0,1],rates,color=color,marker=marker,markersize=8,linewidth=2.3)
            for x,value,count in zip([0,1],rates,counts):
                ax.annotate(f'{value:.1f}%\n{count}/{r["trials"]}',(x,value),
                    xytext=(0,13 if arm=='same_task' else -13),textcoords='offset points',
                    ha='center',va='bottom' if arm=='same_task' else 'top',fontsize=10,color=color)
        ax.set_title(f'Context 4 · cp8 · memory · head {width}',fontsize=12,color=INK,pad=17)
        ax.set_xticks([0,1],[f"Earlier best\nepoch {r['previous_epoch']}",f"Updated best\nepoch {r['new_epoch']}"])
        ax.set_xlim(-.25,1.25);ax.set_ylim(43,82);ax.set_yticks([45,50,55,60,65,70,75,80])
        normal=100*(r['new_normal']-r['previous_normal'])/r['trials']
        donor=100*(r['new_same_task']-r['previous_same_task'])/r['trials']
        ax.text(.5,-.24,f'Normal {normal:+.1f} pp  |  GT donor {donor:+.1f} pp',
                transform=ax.transAxes,ha='center',fontsize=10,color=INK)
        style_axis(ax,False);ax.yaxis.grid(True,color=GRID);ax.set_axisbelow(True)
    axes[0].set_ylabel('Policy completion (%)',labelpad=10)
    note(fig,'Lines connect two evaluated checkpoints, not a complete learning curve. Values are point estimates from one training seed.\n'
        'Best checkpoints were selected by validation position MSE. GT-selected donor performance is diagnostic, not deployment performance.')
    return fig


def task_heatmaps(rows):
    tasks=sorted(rows[0]['per_task'])
    assert all(sorted(r['per_task'])==tasks for r in rows)
    short={'open the middle drawer of the cabinet':'Open middle drawer',
        'open the top drawer and put the bowl inside':'Open top drawer + put bowl inside',
        'push the plate to the front of the stove':'Push plate to front of stove',
        'put the bowl on the plate':'Put bowl on plate','put the bowl on the stove':'Put bowl on stove',
        'put the bowl on top of the cabinet':'Put bowl on top of cabinet',
        'put the cream cheese in the bowl':'Put cream cheese in bowl',
        'put the wine bottle on the rack':'Put wine bottle on rack',
        'put the wine bottle on top of the cabinet':'Put wine bottle on top of cabinet',
        'turn on the stove':'Turn on stove'}
    ylabels=[]
    for task in tasks:
        values=[r['per_task'][task] for r in rows]
        assert len({v['trials'] for v in values})==1
        # Expert replay success can vary across simulator runs; show a range if needed.
        expert=[100*v['expert']/v['trials'] for v in values]
        text=f'{min(expert):.0f}%' if min(expert)==max(expert) else f'{min(expert):.0f}–{max(expert):.0f}%'
        ylabels.append(f"{short[task]}\nexpert {text} · {values[0]['trials']} trials/model")
    fig,axes=plt.subplots(1,2,figsize=(17,10),sharey=True)
    fig.subplots_adjust(left=.245,right=.93,top=.80,bottom=.17,wspace=.09)
    title(fig,'Completion by task and model',
          'Absolute success rates; expert replay completion is shown beside each task for context')
    for ax,arm,heading in zip(axes,['normal','same_task'],['Normal inferred task context','GT-selected correct-task donor']):
        matrix=np.asarray([[100*r['per_task'][task][arm]/r['per_task'][task]['trials'] for r in rows] for task in tasks])
        im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
        ax.set_title(heading,fontsize=12,color=INK,pad=16)
        ax.set_xticks(np.arange(len(rows)),[heatmap_label(r) for r in rows],fontsize=8.5)
        ax.set_yticks(np.arange(len(tasks)),ylabels,fontsize=9.5)
        ax.tick_params(length=0,pad=9)
        for i in range(len(tasks)):
            for j in range(len(rows)):
                ax.text(j,i,f'{matrix[i,j]:.0f}%',ha='center',va='center',fontsize=9,
                        color='white' if matrix[i,j]>=65 else INK)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_xticks(np.arange(-.5,len(rows),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(tasks),1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.2);ax.tick_params(which='minor',length=0)
    cax=fig.add_axes([.947,.17,.013,.63]);bar=fig.colorbar(im,cax=cax)
    bar.set_label('Policy completion (%)',fontsize=10,labelpad=11);bar.outline.set_visible(False)
    note(fig,'Each task has 9–15 repeated trials per model (3–5 episode identities × 3 evaluation seeds). Cells are rounded point estimates.\n'
        'No expert failures were excluded. cp denotes visual feature compression; h denotes action-head width. GT task labels are privileged.')
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();source=root/'result.json'
    data=json.loads(source.read_text());assert data['complete'] and len(data['models'])==8
    rows=data['models'];assert all(r['trials']==132 for r in rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':INK,
        'text.color':INK,'figure.facecolor':'white','axes.facecolor':'white',
        'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
    output=root/'plots';output.mkdir(exist_ok=True)
    jobs=[('success_rates','Success rates across all eight models',lambda:success_rates(rows)),
          ('paired_donor_benefit','Paired donor benefit with 95% confidence intervals',lambda:donor_benefit(rows)),
          ('task_identity_cohorts','Outcomes by original decoded task identity',lambda:identity_cohorts(rows)),
          ('checkpoint_updates','Updated versus earlier checkpoints',lambda:checkpoint_updates(data['previous_checkpoint_comparison'])),
          ('task_success_heatmaps','Per-task completion and expert replay context',lambda:task_heatmaps(rows))]
    figures=[]
    with PdfPages(output/'evaluation_figures.pdf') as pdf:
        for name,heading,build in jobs:
            fig=build();assert len(fig.axes)>=1
            for extension in ('png','svg','pdf'):
                fig.savefig(output/f'{name}.{extension}',dpi=220)
            pdf.savefig(fig)
            figures.append(dict(name=name,title=heading,files=[f'{name}.{ext}' for ext in ('png','svg','pdf')]))
            plt.close(fig)
    fields=['model','variant','epoch','normal_successes','donor_successes','trials','normal_percent',
            'donor_percent','gain_percentage_points','gain_ci_low','gain_ci_high']
    with (output/'model_comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in rows:
            lo,hi=r['effect']['interval_95']
            writer.writerow(dict(model=r['model'],variant=r['variant'],epoch=r['epoch'],
                normal_successes=r['normal'],donor_successes=r['same_task'],trials=r['trials'],
                normal_percent=100*r['normal']/r['trials'],donor_percent=100*r['same_task']/r['trials'],
                gain_percentage_points=100*r['effect']['mean'],gain_ci_low=100*lo,gain_ci_high=100*hi))
    manifest=dict(source_result=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),figures=figures,
        bundle='evaluation_figures.pdf',models=[dict(label=model_label(r),model=r['model'],variant=r['variant'],
        epoch=r['epoch'],checkpoint_sha256=r['checkpoint_sha256']) for r in rows],
        confidence_intervals='Paired episode-cluster bootstrap intervals copied from the audited evaluation report; other graphs display point estimates.',
        protocol='44 episodes × 3 evaluation seeds; 40% handover; live execution context; GT-selected native training donors')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    readme=['# Evaluation figures','',
        '[Download all five figures as one PDF](evaluation_figures.pdf) · [Numeric data](model_comparison.csv) · [Overall report](../summary.md)','',
        'Latest evaluated checkpoints, including updated memory epochs 289 and 374. All figures use the same 44 episodes and three evaluation seeds. '
        'GT-selected donors are a privileged diagnostic. Only the paired-benefit figure shows confidence intervals; other panels show point estimates.']
    for figure in figures:
        name=figure['name'];readme+=['',f"## {figure['title']}",'',f'![{figure["title"]}]({name}.png)','',
            f'[PNG]({name}.png) · [SVG]({name}.svg) · [PDF]({name}.pdf)']
    readme+=['','## Reproduce','',
        '```bash',f'/home/whinnoy/.venvs/align-ablation/bin/python tools/plot_correct_task_donor_evaluations.py --root {root}','```']
    (output/'README.md').write_text('\n'.join(readme)+'\n')
    summary=root/'summary.md';text=summary.read_text()
    start='<!-- evaluation-figures:start -->';end='<!-- evaluation-figures:end -->'
    section=f'{start}\n## Graphs\n\n[All five figures](plots/README.md) · [Download PDF bundle](plots/evaluation_figures.pdf)\n\n'
    section+='![Completion across eight models](plots/success_rates.png)\n\n![Paired donor benefit](plots/paired_donor_benefit.png)\n'+end+'\n\n'
    if start in text:
        before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+section+after.lstrip('\n')
    else:
        assert '## Per-seed completion counts' in text
        text=text.replace('## Per-seed completion counts',section+'## Per-seed completion counts',1)
    summary.write_text(text)
    print(json.dumps(dict(figures=len(figures),directory=str(output),bundle=str(output/'evaluation_figures.pdf'))))


if __name__=='__main__':main()
