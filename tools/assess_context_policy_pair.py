"""Audit two fixed dynamic-context evaluations and write paired result evidence.

Read-only with respect to evaluation inputs and checkpoints. No simulation,
training, checkpoint selection, or replay-based episode filtering is performed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def assess(start_path, prefix_path):
    paths = {'start': Path(start_path), 'prefix': Path(prefix_path)}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    rows = {name: {r['episode']: r for r in run['episodes']} for name, run in data.items()}
    keys = sorted(rows['start'])
    if set(keys) != set(rows['prefix']):
        raise ValueError('Runs use different episode sets')
    for name, run in data.items():
        if len(rows[name]) != len(run['episodes']):
            raise ValueError('Duplicate episode records')
        if not run['evaluation_complete'] or len(keys) != run['requested_episodes']:
            raise ValueError('Incomplete requested evaluation')
        if set(rows[name]) != set(run['evaluation_protocol']['episode_selection']['selected_episodes']):
            raise ValueError('Executed episodes differ from recorded selection')
        if run['n_evaluated_episodes'] != len(keys) or run['n_policy_episodes'] != len(keys):
            raise ValueError('Expected every selected episode to be evaluated and policy-eligible')
        if any(r['n_steps'] != run['evaluation_args']['max_steps'] for r in run['episodes']):
            raise ValueError('Episode did not complete the configured step budget')
        if any(r['status'] != 'evaluated' or r['task_context_mode'] != 'dynamic'
               for r in run['episodes']):
            raise ValueError('Every episode must have a dynamic policy rollout')
        if run['evaluation_protocol']['replay_filters_policy_episodes']:
            raise ValueError('Replay filtering is not allowed in this assessment')
        if run['evaluation_args']['task_context_mode'] != 'dynamic':
            raise ValueError('Expected dynamic task inference')
        if sum(bool(r['success']) for r in run['episodes']) != run['success_count']:
            raise ValueError('Summary success count differs from episode records')
    proto = data['start']['evaluation_protocol']
    for field in ('checkpoint_sha256', 'evaluator_sha256', 'protocol_sha256', 'episode_selection'):
        if proto[field] != data['prefix']['evaluation_protocol'][field]:
            raise ValueError(f'Run provenance differs: {field}')
    differences = {k: [data['start']['evaluation_args'].get(k), data['prefix']['evaluation_args'].get(k)]
        for k in set(data['start']['evaluation_args']) | set(data['prefix']['evaluation_args'])
        if data['start']['evaluation_args'].get(k) != data['prefix']['evaluation_args'].get(k)}
    if set(differences) - {'switch_at', 'out_dir'}:
        raise ValueError(f'Unexpected protocol differences: {differences}')
    if data['start']['evaluation_args']['switch_at'] != 0.:
        raise ValueError('First run must begin autonomously')
    if data['prefix']['evaluation_args']['switch_at'] <= 0.:
        raise ValueError('Second run must observe a human prefix')

    paired = []
    tasks = {}
    trace_hashes = {}
    for key in keys:
        a, b = rows['start'][key], rows['prefix'][key]
        if a['task_name'] != b['task_name'] or a['episode_seed'] != b['episode_seed']:
            raise ValueError(f'{key}: mismatched task or scene seed')
        if a['replay_check'] != b['replay_check']:
            raise ValueError(f'{key}: replay diagnostics differ')
        if a['success_before_handover'] or b['success_before_handover']:
            raise ValueError('This assessment expects no expert-completed episodes')
        if a['success'] and a['first_success_step'] < a['switch_step']:
            raise ValueError('Success credited before policy control')
        if b['success'] and b['first_success_step'] < b['switch_step']:
            raise ValueError('Success credited before policy control')
        files = {name: path.parent / f'{key}_rollout.npz' for name, path in paths.items()}
        with np.load(files['start'], allow_pickle=False) as ta, np.load(files['prefix'], allow_pickle=False) as tb:
            if not np.array_equal(ta['initial_sim_state'], tb['initial_sim_state']):
                raise ValueError(f'{key}: initial simulator states differ')
            if not np.array_equal(ta['replay_position_errors'], tb['replay_position_errors']):
                raise ValueError(f'{key}: expert replay traces differ')
            for name, trace, row in [('start', ta, a), ('prefix', tb, b)]:
                if any(not np.isfinite(trace[field]).all() for field in
                       ('executed_actions', 'predicted_actions', 'sim_positions', 'position_errors')):
                    raise ValueError(f'{key}: nonfinite {name} trace')
                if len(trace['executed_actions']) != row['n_steps']:
                    raise ValueError(f'{key}: trace length does not match report')
                if int(trace['switch_step']) != row['switch_step']:
                    raise ValueError(f'{key}: trace handover does not match report')
                if not np.isclose(trace['position_errors'].mean(), row['mean_error_model']):
                    raise ValueError(f'{key}: trace EEF error does not match report')
            boundary = b['switch_step']
            ea, eb = ta['position_errors'][boundary:], tb['position_errors'][boundary:]
            if not len(ea) or len(ea) != len(eb):
                raise ValueError(f'{key}: unmatched post-boundary reference windows')
            error_start, error_prefix = float(ea.mean()), float(eb.mean())
        trace_hashes[key] = {name: digest(path) for name, path in files.items()}
        paired.append(dict(episode=key, task=a['task_name'], episode_seed=a['episode_seed'],
            start_success=bool(a['success']), prefix_success=bool(b['success']),
            replay_valid=bool(a['replay_check']['valid']),
            replay_task_success=bool(a['replay_check']['task_success']),
            switch_step=boundary, start_error_on_common_window=error_start,
            prefix_error_on_common_window=error_prefix))
        task = tasks.setdefault(a['task_name'], dict(episodes=0, start_successes=0, prefix_successes=0))
        task['episodes'] += 1
        task['start_successes'] += int(a['success'])
        task['prefix_successes'] += int(b['success'])

    counts = Counter((r['start_success'], r['prefix_success']) for r in paired)
    subgroups = {}
    for field in ('replay_valid', 'replay_task_success'):
        subgroups[field] = {}
        for value in (False, True):
            selected = [r for r in paired if r[field] == value]
            subgroups[field][str(value).lower()] = dict(episodes=len(selected),
                start_successes=sum(r['start_success'] for r in selected),
                prefix_successes=sum(r['prefix_success'] for r in selected))
    summary = {}
    for name, run in data.items():
        summary[name] = dict(successes=run['success_count'], episodes=len(keys),
            success_rate=run['success_count'] / len(keys), expert_completed_episodes=run['expert_completed_episodes'],
            mean_error_all_reference_frames=run['aggregate']['mean_model_err'],
            mean_error_on_common_post_boundary_window=float(np.mean([
                r[f'{name}_error_on_common_window'] for r in paired])))
    return dict(schema='align-context-policy-paired-assessment-v1', inputs={name: dict(
        path=str(path), sha256=digest(path)) for name, path in paths.items()},
        checkpoint_sha256=proto['checkpoint_sha256'], evaluator_sha256=proto['evaluator_sha256'],
        protocol_sha256=proto['protocol_sha256'], episode_selection=proto['episode_selection'],
        setting_differences=differences, summary=summary,
        gain_percentage_points=100 * (summary['prefix']['success_rate'] - summary['start']['success_rate']),
        paired_successes=dict(both_succeed=counts[(True, True)], prefix_only=counts[(False, True)],
            start_only=counts[(True, False)], both_fail=counts[(False, False)]),
        per_task=tasks, replay_subgroups=subgroups, paired_episodes=paired, trace_sha256=trace_hashes,
        trace_verification=dict(paired_initial_states_identical=True, replay_errors_identical=True,
            all_actions_and_positions_finite=True, all_requested_episodes_evaluated=True),
        interpretation='Expert prefix changes both physical state and task evidence. This comparison does not isolate context-encoder, pretraining, long-history or memory benefits. One seed and ten known task groups; saved development episodes, not the earlier Piper list.')


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tasks = report['per_task']
    names = sorted(tasks)
    y = np.arange(len(names))
    fraction = report['setting_differences']['switch_at'][1]
    prefix_label = f'{100 * fraction:g}% expert prefix'
    fig, ax = plt.subplots(figsize=(12, 7.5))
    for offset, field, label, color in [(-.18, 'start_successes', 'Autonomous from start', '#0072B2'),
                                      (.18, 'prefix_successes', prefix_label, '#D55E00')]:
        values = [100 * tasks[name][field] / tasks[name]['episodes'] for name in names]
        ax.barh(y + offset, values, height=.32, color=color, label=label)
        for row, name, value in zip(y + offset, names, values):
            ax.text(value + 1, row, f"{tasks[name][field]}/{tasks[name]['episodes']}",
                    va='center', fontsize=9)
    ax.set_yticks(y, names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 109)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('Task success (%)')
    ax.legend(loc='upper right', frameon=False)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    a, b = report['summary']['start'], report['summary']['prefix']
    fig.suptitle(f"Context policy: {a['successes']}/{a['episodes']} ({100*a['success_rate']:.1f}%) from start; "
                 f"{b['successes']}/{b['episodes']} ({100*b['success_rate']:.1f}%) after expert prefix",
                 fontsize=13, fontweight='bold', y=.97)
    fig.text(.03, .025, 'Same 44 development episodes; dynamic task inference; seed 42; all replay outcomes included.\n'
             'The prefix supplies task evidence and physical progress. It does not isolate the benefit of the context encoder.', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .93))
    for extension in ('png', 'svg'):
        fig.savefig(output / f'task_success.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.start, args.prefix)
    result['analysis_source_sha256'] = digest(__file__)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    plot(result, args.output)
    print(json.dumps(dict(summary=result['summary'], paired=result['paired_successes'],
                         gain_percentage_points=result['gain_percentage_points']), indent=2))


if __name__ == '__main__':
    main()
