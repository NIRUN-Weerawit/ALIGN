"""Known-task separability of the exact frozen context encoder in a policy.

Diagnostic classifiers fit train episodes only. Context/policy weights are never
trained. Teacher tokens, future targets and task labels never enter the encoder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from data.task_context_dataset import file_hash, json_hash
from models.task_context_policy import TaskContextPolicyBranch


FRACTIONS = (0., .1, .25, .4, .5, .7, .9)
QUERIES = ([dict(name=f'fraction_{f:g}', fraction=f) for f in FRACTIONS] +
           [dict(name='second_1', seconds=1.), dict(name='second_2', seconds=2.)])


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def metrics(labels, prediction):
    return dict(accuracy=float(accuracy_score(labels, prediction)),
                balanced_accuracy=float(balanced_accuracy_score(labels, prediction)),
                confusion=confusion_matrix(labels, prediction).tolist())


def linear_probe(x, labels, y, target):
    # Match the pre-existing teacher-probe regularization; no development tuning.
    classifier = make_pipeline(StandardScaler(), LogisticRegression(
        C=.1, class_weight='balanced', max_iter=3000, random_state=42))
    classifier.fit(x, labels)
    prediction = classifier.predict(y)
    return dict(train=metrics(labels, classifier.predict(x)),
                development=metrics(target, prediction),
                solver_iterations=classifier[-1].n_iter_.tolist()), prediction


@torch.inference_mode()
def extract(branch, cache, manifest, split, output):
    values = {name: [] for name in ('keys', 'groups', 'tokens', 'execution', 'summary',
                                    'states', 'cls', 'times', 'mode', 'anchors', 'lengths')}
    encoder = branch.observer.encoder
    start = time.monotonic()
    causal_audit = None
    for index, key in enumerate(manifest['split'][split]):
        path = cache / f'{key}.npz'
        if file_hash(path) != manifest['files'][key]:
            raise ValueError(f'Cache hash differs: {key}')
        with np.load(path, allow_pickle=False) as row:
            n = len(row['summary'])
            anchors = np.array([int(n * q['fraction']) if 'fraction' in q else
                                int(branch.fps * q['seconds']) for q in QUERIES], dtype=np.int64)
            if anchors.max() >= n:
                raise ValueError(f'{key}: episode too short for all fixed-time queries')
            observed = int(anchors.max()) + 1
            x = torch.from_numpy(row['summary'][:observed].copy())[None]
            result = encoder(x, torch.tensor([observed]), torch.from_numpy(anchors)[None])
            native = result['tokens'][0]
            tokens = (native - encoder.latent_mean) / encoder.latent_std
            values['keys'].append(key)
            values['groups'].append(int(row['group']))
            values['tokens'].append(tokens.cpu().numpy())
            values['execution'].append(result['execution_token'][0, :, 0].cpu().numpy())
            values['mode'].append(result['mode'][0].cpu().numpy())
            values['anchors'].append(anchors)
            values['lengths'].append(n)
            for field in ('summary', 'states', 'cls', 'times'):
                values[field].append(row[field][anchors].copy())
            if split == 'development' and causal_audit is None:
                t = int(anchors[3])
                early = encoder(x[:, :t+1], torch.tensor([t+1]), torch.tensor([[t]]))['tokens'][0, 0]
                difference = float((early - native[3]).abs().max())
                if difference > 2e-5:
                    raise ValueError('Prefix/full-sequence readouts disagree')
                causal_audit = dict(episode=key, anchor=t, max_absolute_error=difference,
                                    contract='40% readout matches running only its observed prefix')
        if (index + 1) % 32 == 0 or index + 1 == len(manifest['split'][split]):
            print(f'EXTRACT {split} {index+1}/{len(manifest["split"][split])} '
                  f'{time.monotonic()-start:.1f}s', flush=True)
    arrays = {key: np.asarray(value) for key, value in values.items()}
    np.savez_compressed(output / f'{split}_features.npz', **arrays)
    return arrays, causal_audit


def feature_views(arrays, query):
    tokens = arrays['tokens'][:, query]
    return dict(task_bundle=tokens.reshape(len(tokens), -1),
        **{f'task_token_{i+1}': tokens[:, i] for i in range(tokens.shape[1])},
        execution_token=arrays['execution'][:, query],
        current_summary=arrays['summary'][:, query],
        current_cls=arrays['cls'][:, query],
        current_geometric_state=arrays['states'][:, query],
        elapsed_time=arrays['times'][:, query, None],
        mixture_mode_only=np.eye(4)[arrays['mode'][:, query]])


def geometric_separation(train, development, query):
    x = train['tokens'][:, query].reshape(len(train['keys']), -1)
    y = development['tokens'][:, query].reshape(len(development['keys']), -1)
    train_labels, labels = train['groups'], development['groups']
    classes = np.unique(train_labels)
    centroids = np.stack([x[train_labels == group].mean(0) for group in classes])
    cosine = unit(y) @ unit(x).T
    prediction_nn = train_labels[cosine.argmax(1)]
    prediction_centroid = classes[(unit(y) @ unit(centroids).T).argmax(1)]
    same = labels[:, None] == labels[None]
    triangle = np.triu(np.ones(same.shape, dtype=bool), 1)
    similarity = unit(y) @ unit(y).T
    distances = np.linalg.norm(y[:, None] - y[None], axis=-1)
    within, between = triangle & same, triangle & ~same
    result = dict(nearest_training_episode=metrics(labels, prediction_nn),
        nearest_train_task_centroid=metrics(labels, prediction_centroid),
        within_task_pairs=int(within.sum()), between_task_pairs=int(between.sum()),
        within_task_mean_cosine=float(similarity[within].mean()),
        between_task_mean_cosine=float(similarity[between].mean()),
        within_task_mean_l2=float(distances[within].mean()),
        between_task_mean_l2=float(distances[between].mean()),
        pair_similarity_auc=float(roc_auc_score(same[triangle], similarity[triangle])),
        train_centroid_cosine=(unit(centroids) @ unit(centroids).T).tolist(),
        distinct_development_vectors=int(len(np.unique(y, axis=0))))
    return result, similarity[within], similarity[between], centroids


def plots(output, result, train, development, similarities, primary_prediction):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    names = result['task_names']
    short = [name.replace('put the ', '').replace('open the ', '').replace('turn on the ', '')
             .replace(' of the cabinet', '').replace('to the front of the ', 'to ') for name in names]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for view, label, color in [('task_bundle', 'Task tokens (4 × 128)', '#0072B2'),
                              ('execution_token', 'Execution token', '#009E73'),
                              ('current_summary', 'Current summary', '#D55E00'),
                              ('current_geometric_state', 'Current geometry', '#CC79A7'),
                              ('elapsed_time', 'Elapsed time only', '#777777')]:
        scores = [result['queries'][q['name']]['linear_probes'][view]['development']['balanced_accuracy']
                  for q in QUERIES[:7]]
        axes[0, 0].plot(np.asarray(FRACTIONS)*100, scores, 'o-', label=label, color=color)
    axes[0, 0].axhline(.1, ls='--', color='black', lw=1, label='10-task chance')
    axes[0, 0].set(xlabel='Observed episode fraction (%)', ylabel='Balanced task accuracy',
                   title='Train-only linear probes; 44 development episodes', ylim=(0, 1.04))
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].legend(fontsize=8, loc='lower right')
    primary = result['queries']['fraction_0.4']
    confusion = np.asarray(primary['linear_probes']['task_bundle']['development']['confusion'])
    axes[0, 1].imshow(confusion, cmap='Blues')
    for i in range(len(names)):
        for j in range(len(names)):
            axes[0, 1].text(j, i, str(confusion[i, j]), ha='center', va='center', fontsize=8,
                            color='white' if confusion[i, j] > confusion.max()/2 else 'black')
    axes[0, 1].set(xticks=range(10), yticks=range(10), xlabel='Predicted task index',
                   ylabel='True task index', title='Task-token probe at 40% prefix (counts)')
    for values, label, color in zip(similarities, ['Same task', 'Different task'], ['#0072B2', '#D55E00']):
        axes[1, 0].hist(values, bins=np.linspace(-1, 1, 31), density=True,
                         alpha=.55, color=color, label=label)
    axes[1, 0].set(xlabel='Cosine similarity in train-normalized task space', ylabel='Density',
                   title='Development pairs at the same 40% prefix')
    axes[1, 0].legend()
    views = ['task_bundle'] + [f'task_token_{i}' for i in range(1, 5)]
    scores = [primary['linear_probes'][view]['development']['balanced_accuracy'] for view in views]
    axes[1, 1].bar(['All four', 'Token 1', 'Token 2', 'Token 3', 'Token 4'], scores,
                    color=['#0072B2']+['#56B4E9']*4)
    axes[1, 1].axhline(.1, color='black', linestyle='--', linewidth=1)
    axes[1, 1].set(ylabel='Balanced task accuracy', ylim=(0, 1.04),
                   title='Individual token probes at 40% prefix')
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('Does the deployed context representation distinguish known tasks?', fontsize=14)
    fig.text(.02, .015, '340 probe-training / 44 development episodes; frozen policy encoder; no confirmation. '
             'Task labels are probe targets only. Class decoding does not establish policy utility or scene-independent intent.', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .96))
    for extension in ('png', 'svg'):
        fig.savefig(output / f'task_token_separability.{extension}', dpi=170)
    plt.close(fig)

    x = train['tokens'][:, 3].reshape(len(train['keys']), -1)
    y = development['tokens'][:, 3].reshape(len(development['keys']), -1)
    projection = PCA(n_components=2, random_state=42).fit(x)
    points = projection.transform(y)
    centroids = projection.transform(np.stack([x[train['groups'] == i].mean(0) for i in range(10)]))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = plt.get_cmap('tab10')
    for group, name in enumerate(short):
        selected = development['groups'] == group
        ax.scatter(points[selected, 0], points[selected, 1], color=colors(group), label=f'{group}: {name}', s=40)
        ax.scatter(*centroids[group], color=colors(group), marker='x', s=100, linewidths=2)
    variance = projection.explained_variance_ratio_
    ax.set(xlabel=f'PC1 ({100*variance[0]:.1f}% train variance)',
           ylabel=f'PC2 ({100*variance[1]:.1f}% train variance)',
           title='40% task tokens: development episodes; × = training task centroid')
    ax.legend(loc='center left', bbox_to_anchor=(1, .5), fontsize=9)
    fig.text(.02, .015, 'PCA fitted on training episodes only. Two dimensions are an illustration, not a separability test.', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for extension in ('png', 'svg'):
        fig.savefig(output / f'task_token_projection.{extension}', dpi=170)
    plt.close(fig)
    return variance.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--observer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    if json_hash({k:v for k,v in manifest.items() if k != 'manifest_sha256'}) != manifest['manifest_sha256']:
        raise ValueError('Invalid prefix manifest')
    if file_hash(args.cache / 'normalization.npz') != manifest['normalization_sha256']:
        raise ValueError('Invalid cache normalization')
    checkpoint = torch.load(args.policy, map_location='cpu', weights_only=True, mmap=True)
    cfg = checkpoint['config']
    if cfg.get('task_context_trainable'):
        raise ValueError('This protocol targets the frozen pretrained-context policy')
    context = cfg['task_context_config']
    if context['metadata']['prefix_manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Policy and prefix cache are not bound')
    partitions = manifest['split']
    if cfg['task_context_split']['train_episodes'] != partitions['train'] or cfg['task_context_split']['development_episodes'] != partitions['development']:
        raise ValueError('Policy and cache splits differ')
    if set(partitions['train']) & set(partitions['development']):
        raise ValueError('Train/development overlap')
    branch = TaskContextPolicyBranch(config=context).requires_grad_(False).eval()
    prefix = 'task_context.'
    branch.load_state_dict({k[len(prefix):]:v for k,v in checkpoint['model_state_dict'].items()
                           if k.startswith(prefix)}, strict=True)
    bundle = torch.load(args.observer, map_location='cpu', weights_only=True)
    for section in ('encoder', 'summary'):
        own = getattr(branch.observer, section).state_dict()
        if any(not torch.equal(own[k], v) for k,v in bundle[section].items()):
            raise ValueError('Pretrained observer changed during policy training')
    for field in ('state_mean', 'state_std'):
        if not torch.equal(getattr(branch.observer, field), bundle[field]):
            raise ValueError('Observation normalization differs')
    args.output.mkdir(parents=True)
    protocol = dict(schema='align-task-token-separability-v1', policy=str(args.policy),
        policy_sha256=file_hash(args.policy), observer_sha256=file_hash(args.observer),
        prefix_manifest_sha256=manifest['manifest_sha256'], analysis_sha256=file_hash(__file__),
        splits={name:partitions[name] for name in ('train','development')}, confirmation_opened=False,
        queries=QUERIES, primary='fraction_0.4', probe=dict(type='StandardScaler + balanced LogisticRegression',
            C=.1, max_iter=3000, random_state=42, fitting='train episodes only; no development hyperparameter tuning'),
        coordinate_system='Selected top-weight task mode; native tokens normalized by encoder train-only teacher latent buffers',
        scope='Cached expert-demonstration observations, not saved on-policy simulator task estimates',
        labels='Known LIBERO task-group identity; text/IDs used for diagnostics only, never encoder inputs')
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    with threadpool_limits(limits=2):
        train, _ = extract(branch, args.cache, manifest, 'train', args.output)
        development, audit = extract(branch, args.cache, manifest, 'development', args.output)
        classes = np.unique(train['groups'])
        if not np.array_equal(classes, np.arange(10)) or not np.array_equal(np.unique(development['groups']), classes):
            raise ValueError('Expected ten known tasks represented in both splits')
        names = {}
        with h5py.File(args.data, 'r') as data:
            for key, group in zip(train['keys'], train['groups']):
                text = json.loads(data[str(key)]['texts'][()])[0]
                if int(group) in names and names[int(group)] != text:
                    raise ValueError('Task-group labels are inconsistent')
                names[int(group)] = text
        result = dict(protocol=protocol, task_names=[names[i] for i in range(10)],
            train_episodes=len(train['keys']), development_episodes=len(development['keys']),
            chance_balanced_accuracy=.1, encoder_bitwise_identical_to_observer=True,
            causal_audit=audit, queries={})
        primary_prediction = None
        for qi, query in enumerate(QUERIES):
            train_views, dev_views = feature_views(train, qi), feature_views(development, qi)
            row = dict(query=query, linear_probes={})
            for view, x in train_views.items():
                row['linear_probes'][view], prediction = linear_probe(x, train['groups'],
                    dev_views[view], development['groups'])
                if qi == 3 and view == 'task_bundle':
                    primary_prediction = prediction
            row['geometry'], within, between, _ = geometric_separation(train, development, qi)
            if qi == 3:
                primary_similarities = (within, between)
            result['queries'][query['name']] = row
            print('PROBE', query['name'], {view:round(value['development']['balanced_accuracy'],4)
                for view,value in row['linear_probes'].items()}, flush=True)
        # Fixed predictions and permuted development labels test label association;
        # this is a null reference, not a new fitted model or uncertainty interval.
        rng = np.random.default_rng(42)
        null = [float(balanced_accuracy_score(rng.permutation(development['groups']), primary_prediction))
                for _ in range(1000)]
        result['label_permutation_control'] = dict(repetitions=1000, seed=42,
            method='Shuffle development labels only; fitted primary predictions held fixed',
            mean_balanced_accuracy=float(np.mean(null)),
            central_95_percentile=np.quantile(null,[.025,.975]).tolist())
        result['pca_train_variance'] = plots(args.output, result, train, development,
                                             primary_similarities, primary_prediction)
    result['limitations'] = ['Known-task episode split, not new tasks or verified scene/operator-disjoint data.',
        'Post-hoc development diagnostic; these episodes already informed encoder/policy selection.',
        'Task IDs are coarse labels; decodability does not establish full context, hidden constraints or action utility.',
        'Current-scene/geometry/time controls can expose task-correlated cues; no proof of exclusive history use.',
        'Fraction anchors are analysis-only; fixed-time queries are included to reduce duration-selection confounding.',
        'Mixture mode index is not task identity; each of four learned tokens may carry overlapping information.',
        'Expert cached observations differ from autonomous on-policy observations; saved rollout traces lack task tokens/images.',
        'No confirmation opened and no policy/encoder weights updated.']
    (args.output / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print('DONE',args.output,flush=True)


if __name__ == '__main__':
    main()
