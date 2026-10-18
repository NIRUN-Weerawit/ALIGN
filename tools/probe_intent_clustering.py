#!/usr/bin/env python3
"""Frozen causal intention probe with episode-disjoint task decoding.

Uses cached DINO CLS features, exact encoder weights, and real per-frame states.
Task strings are analysis labels only, never encoder inputs. Saves trajectories,
original-space cluster metrics, train-only-fit linear probes, PCA/t-SNE figures,
and raw features. Action heads, vision backbones and memory banks are not run.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import mmap
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.align_dataset import ALIGNDataset
from models.align_model import RobotStateEncoder
from models.intention_encoder import IntentionEncoder


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_split(config, available):
    train = list(config.get('train_episodes', []))
    val = list(config.get('val_episodes', []))
    if not train or not val:
        raise ValueError('Checkpoint must record train_episodes and val_episodes')
    if len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
        raise ValueError('Probe split must contain unique, disjoint whole episodes')
    if (set(train) | set(val)) - set(available):
        raise ValueError('Checkpoint split contains episodes absent from dataset')
    return train, val


def make_encoder(config):
    accepted = inspect.signature(IntentionEncoder.__init__).parameters
    values = {key: value for key, value in config.items() if key in accepted}
    values.update(use_intent_tokens=True)
    return IntentionEncoder(**values)


def load_encoders(path, device, seed=42, random_control=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    config = checkpoint.get('config')
    if not isinstance(config, dict) or not config.get('use_intent_tokens'):
        raise ValueError('Checkpoint must have config and enabled intention tokens')
    if not config.get('use_history', True) or config.get('mamba_output_dim', 0) <= 0:
        raise ValueError('Checkpoint has no active intention encoder')
    weights = checkpoint['model_state_dict']
    weights = {key.removeprefix('module.'): value for key, value in weights.items()}
    torch.manual_seed(seed)
    state = RobotStateEncoder(input_dim=7, state_dim=config['state_dim'])
    encoder = make_encoder(config)
    fingerprint = hashlib.sha256()
    for prefix, module in [('state_encoder.', state), ('intention_encoder.', encoder)]:
        selected = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
        # Fail on any missing trained encoder weight; no tolerant random fallback.
        module.load_state_dict(selected, strict=True)
        for key, value in sorted(selected.items()):
            fingerprint.update((prefix + key).encode())
            fingerprint.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    control = None
    if random_control:
        torch.manual_seed(seed + 9021)
        control = make_encoder(config).to(device).eval()
    metadata = dict(checkpoint=str(Path(path).resolve()), epoch=checkpoint.get('epoch'),
        encoder_sha256=fingerprint.hexdigest(), checkpoint_bytes=Path(path).stat().st_size,
        trained_with_text=bool(config.get('use_text', False)),
        use_memory_bank=bool(config.get('use_memory_bank', False)), config=config)
    return encoder.to(device).eval(), state.to(device).eval(), control, metadata


@torch.inference_mode()
def prefix_readouts(encoder, cls, states, indices, device, precision='float32'):
    """Full causal history and a control that resets recurrence at each sample."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == 'cuda' and precision == 'bfloat16'):
        full = encoder.forward_sequence(cls[None], states[None])[0, indices]
        reset = encoder.forward_sequence(cls[indices, None], states[indices, None])[:, 0]
    # Concatenate token slots; never average distinct learned tokens together.
    return full.flatten(1).float().cpu().numpy(), reset.flatten(1).float().cpu().numpy()


@torch.inference_mode()
def collect(args):
    device = torch.device(args.device)
    before = time.perf_counter()
    encoder, state_encoder, random_encoder, metadata = load_encoders(
        args.checkpoint, device, args.seed, args.random_control)
    config = metadata['config']
    cameras = args.cameras or config.get('cameras', ['image', 'wrist_image'])
    if len(cameras) != config['num_cameras']:
        raise ValueError('Camera count must match checkpoint')
    dataset = ALIGNDataset(str(args.data), mode='head', cameras=cameras,
        traj_window=config['chunk_size'], dinov2_path=str(args.cache))
    train, val = validate_split(config, dataset._episode_keys)
    episodes = train + val
    positions = {key: i for i, key in enumerate(dataset._episode_keys)}
    source = args.gripper_state_source
    if source == 'auto':
        source = config.get('gripper_state_source') or ('legacy_zero' if 'variants' in config else 'causal')
    progress = np.linspace(0., 1., args.sample_points)
    features = {key: [] for key in ['intent', 'reset_intent', 'current_cls_state', 'state_only', 'history_mean']}
    if random_encoder is not None:
        features['random_encoder'] = []
    labels, indices_all, lengths = [], [], []
    raw_dim = config.get('raw_dim', 768)
    for index, key in enumerate(episodes):
        ep = positions[key]
        length = dataset._get_episode_length(ep)
        times = np.rint(progress * (length - 1)).astype(np.int64)
        frames = dataset._read_frames_dinov2_view(ep, 0, length)
        # Sparse CLS reads stride over large patch arrays. Disable mmap
        # readahead so the OS does not fetch every unused image patch too.
        backing = dataset._dinov2_memmaps[ep]._mmap
        if hasattr(backing, 'madvise') and hasattr(mmap, 'MADV_RANDOM'):
            backing.madvise(mmap.MADV_RANDOM)
        if frames.shape[1:] != (257 * len(cameras), raw_dim):
            raise ValueError(f'Unexpected cached token shape: {frames.shape}')
        # Read only CLS pages from the mmap; do not copy all image patches.
        cls = np.array(frames.reshape(length, len(cameras), 257, raw_dim)[:, :, -1], copy=True)
        del frames
        poses = dataset._read_poses(ep, 0, length)[:, :6]
        gripper = dataset._read_poses_gripper(ep, 0, length)
        if source == 'legacy_zero':
            gripper = np.zeros_like(gripper)
        states = np.concatenate([poses, gripper[:, None]], axis=1).astype(np.float32)
        cls_t = torch.from_numpy(cls).to(device).float()
        states_t = state_encoder(torch.from_numpy(states).to(device))
        full, reset = prefix_readouts(encoder, cls_t, states_t, times, device, args.precision)
        features['intent'].append(full)
        features['reset_intent'].append(reset)
        features['current_cls_state'].append(np.concatenate([cls[times].reshape(len(times), -1), states[times]], axis=1))
        features['state_only'].append(states[times])
        means = np.cumsum(cls.reshape(length, -1), axis=0, dtype=np.float64) / np.arange(1, length+1)[:, None]
        state_means = np.cumsum(states, axis=0, dtype=np.float64) / np.arange(1, length+1)[:, None]
        features['history_mean'].append(np.concatenate([means[times], state_means[times], states[times]], axis=1).astype(np.float32))
        if random_encoder is not None:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda' and args.precision == 'bfloat16'):
                random = random_encoder.forward_sequence(cls_t[None], states_t[None])[0, times]
            features['random_encoder'].append(random.flatten(1).float().cpu().numpy())
        labels.append(dataset._read_text(ep))
        indices_all.append(times)
        lengths.append(length)
        getattr(dataset, '_dinov2_memmaps', {}).pop(ep, None)
        if (index+1) % 50 == 0 or index+1 == len(episodes):
            print(f'EXTRACT {index+1}/{len(episodes)} ({time.perf_counter()-before:.1f}s)', flush=True)
    dataset.close()
    arrays = {key: np.stack(value) for key, value in features.items()}
    arrays.update(labels=np.asarray(labels), episodes=np.asarray(episodes), progress=progress,
        timesteps=np.stack(indices_all), lengths=np.asarray(lengths),
        train_mask=np.arange(len(episodes)) < len(train))
    np.savez_compressed(args.output / 'intent_embeddings.npz', **arrays)
    metadata.update(extraction_wall_s=time.perf_counter()-before, device=str(device),
        gripper_state_source=source, sample_points=args.sample_points, precision=args.precision,
        supplied_text_to_encoder=False, token_stage='raw intention encoder output before memory fusion',
        random_control='random Mamba/readout; same trained state encoder' if random_encoder else None,
        data=str(args.data.resolve()), cache=str(args.cache.resolve()),
        protocol='Whole-episode causal recurrence, per-frame states, no future observations; task strings used only as probe labels')
    save_json(args.output / 'extraction.json', metadata)
    return arrays, metadata


def fit_decoder(train_x, train_y, val_x, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    decoder = make_pipeline(StandardScaler(), LogisticRegression(
        C=.1, class_weight='balanced', max_iter=2000, solver='lbfgs', random_state=seed))
    decoder.fit(train_x, train_y)
    return decoder, decoder.predict(val_x)


def classification_metrics(truth, prediction):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    return dict(accuracy=float(accuracy_score(truth, prediction)),
                balanced_accuracy=float(balanced_accuracy_score(truth, prediction)))


def stratified_bootstrap_ci(truth, prediction, seed=42, repeats=2000):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(truth == task) for task in np.unique(truth)]
    scores = []
    for _ in range(repeats):
        scores.append(np.mean([np.mean(prediction[sampled := rng.choice(g, len(g), replace=True)] == truth[sampled]) for g in groups]))
    return np.quantile(scores, [.025, .975]).tolist()


def cosine_distance(a, b):
    from sklearn.preprocessing import normalize
    return 1. - normalize(a) @ normalize(b).T


def analyze(arrays, metadata, args):
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from threadpoolctl import threadpool_limits
    labels = LabelEncoder().fit(arrays['labels'])
    y = labels.transform(arrays['labels'])
    train = arrays['train_mask'].astype(bool)
    val = ~train
    if set(y[train]) != set(y[val]):
        raise ValueError('Every task must appear in both train and held-out episodes')
    classes = len(labels.classes_)
    features = [key for key in arrays if key in ['intent', 'reset_intent', 'current_cls_state', 'state_only', 'history_mean', 'random_encoder']]
    progress = arrays['progress']
    results = dict(chance_balanced_accuracy=1/classes, train_episodes=int(train.sum()),
        held_out_episodes=int(val.sum()), tasks=labels.classes_.tolist(), sklearn=sklearn.__version__,
        classifier='StandardScaler fitted only on training episodes + logistic regression C=0.1, balanced classes; no validation tuning',
        features={}, geometry=[], predictions={},
        caveats=['One checkpoint/seed; task decoding and geometric clusters are different tests.',
                 'Current appearance/state can disclose the task; separation alone is not evidence of goal inference or causal action use.',
                 'A token may encode phase and uncertainty as well as objective; monotonic collapse into one cluster is not required.',
                 'Features come from expert demonstration trajectories; late observations can reveal task consequences rather than predict future goals.',
                 '2-D projections are visualization only; quantitative metrics use original feature dimensions.',
                 'Episode progress is retrospective analysis, not an input to the encoder.'])
    with threadpool_limits(limits=args.cpu_threads):
        for feature in features:
            rows = []
            for t, fraction in enumerate(progress):
                decoder, predicted = fit_decoder(arrays[feature][train, t], y[train], arrays[feature][val, t], args.seed)
                row = dict(progress=float(fraction), **classification_metrics(y[val], predicted))
                if feature == 'intent':
                    transfer = decoder.predict(arrays['reset_intent'][val, t])
                    row['full_probe_on_reset_accuracy'] = classification_metrics(y[val], transfer)['balanced_accuracy']
                rows.append(row)
                if t == len(progress)-1:
                    rows[-1]['balanced_accuracy_95pct_episode_bootstrap'] = stratified_bootstrap_ci(y[val], predicted, args.seed)
                    results['predictions'][feature] = predicted.tolist()
            results['features'][feature] = rows
            print('DECODE', feature, 'early', rows[0]['balanced_accuracy'], 'late', rows[-1]['balanced_accuracy'], flush=True)
        full = arrays['intent']
        # A single final-position decoder tests whether task directions persist
        # across time; separate per-position fits alone cannot establish that.
        late_decoder, _ = fit_decoder(full[train, -1], y[train], full[val, -1], args.seed)
        results['final_decoder_across_progress'] = [
            dict(progress=float(fraction), **classification_metrics(
                y[val], late_decoder.predict(full[val, t])))
            for t, fraction in enumerate(progress)]
        scaler = StandardScaler().fit(full[train].reshape(-1, full.shape[-1]))
        scaled = scaler.transform(full.reshape(-1, full.shape[-1])).reshape(full.shape)
        late = progress >= .8
        late_centroids = np.stack([scaled[train & (y==task)][:, late].reshape(-1, scaled.shape[-1]).mean(0) for task in range(classes)])
        final_pred = np.asarray(results['predictions']['intent'])
        cm = confusion_matrix(y[val], final_pred, labels=np.arange(classes))
        results['confusion_matrix'] = cm.tolist()
        for t, fraction in enumerate(progress):
            x, xt = scaled[val, t], scaled[train, t]
            centroid = np.stack([xt[y[train]==task].mean(0) for task in range(classes)])
            distances = cosine_distance(x, centroid)
            within = float(distances[np.arange(val.sum()), y[val]].mean())
            between = float(cosine_distance(centroid, centroid)[np.triu_indices(classes, 1)].mean())
            late_dist = cosine_distance(x, late_centroids)[np.arange(val.sum()), y[val]]
            kmeans = KMeans(n_clusters=classes, n_init=10, random_state=args.seed).fit(xt)
            clusters = kmeans.predict(x)
            results['geometry'].append(dict(progress=float(fraction),
                cosine_silhouette=float(silhouette_score(x, y[val], metric='cosine')),
                adjusted_rand_index=float(adjusted_rand_score(y[val], clusters)),
                normalized_mutual_information=float(normalized_mutual_info_score(y[val], clusters)),
                within_task_cosine_distance=within, between_task_centroid_distance=between,
                within_between_ratio=within/between if between > 1e-12 else None,
                distance_to_train_late_task_centroid=float(late_dist.mean()),
                nearest_centroid_balanced_accuracy=classification_metrics(y[val], distances.argmin(1))['balanced_accuracy']))
        if args.permutations:
            rng = np.random.default_rng(args.seed)
            null = []
            for _ in range(args.permutations):
                _, prediction = fit_decoder(full[train, -1], rng.permutation(y[train]), full[val, -1], args.seed)
                null.append(classification_metrics(y[val], prediction)['balanced_accuracy'])
            observed = results['features']['intent'][-1]['balanced_accuracy']
            results['permutation_test'] = dict(permutations=args.permutations,
                null_mean=float(np.mean(null)), null_max=float(np.max(null)),
                p_value=float((1+np.count_nonzero(np.asarray(null)>=observed))/(1+len(null))))
        results['checkpoint'] = {key:metadata[key] for key in ['checkpoint','epoch','encoder_sha256','trained_with_text','gripper_state_source','token_stage']}
        save_json(args.output / 'results.json', results)
        plot_results(arrays, results, scaled, labels.classes_, args)
    return results


def plot_results(arrays, results, scaled, names, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    train = arrays['train_mask'].astype(bool)
    y = np.searchsorted(names, arrays['labels'])
    val = ~train
    progress = arrays['progress']*100
    colors = plt.get_cmap('tab10')
    task_ids = [f'T{i+1:02d}' for i in range(len(names))]
    def save(fig, name):
        fig.savefig(args.output / name, dpi=170, bbox_inches='tight');plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for feature, rows in results['features'].items():
        ax.plot(progress, [r['balanced_accuracy'] for r in rows], label=feature)
    ax.plot(progress, [r['balanced_accuracy'] for r in results['final_decoder_across_progress']],
            linestyle=':', color='black', label='intent: fixed final-position decoder')
    ax.axhline(results['chance_balanced_accuracy'], color='gray', linestyle='--', label='chance')
    ax.set(xlabel='Observed episode progress (%)', ylabel='Held-out balanced task accuracy', ylim=(0,1.03),
           title='Task information over causal observation history')
    ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,'task_decode_vs_progress.png')
    pca = PCA(n_components=2, random_state=args.seed).fit(scaled[train].reshape(-1,scaled.shape[-1]))
    projected = pca.transform(scaled.reshape(-1,scaled.shape[-1])).reshape(*scaled.shape[:2],2)
    fig, axes = plt.subplots(1,3,figsize=(15,4),sharex=True,sharey=True)
    for ax, index in zip(axes, [0,int(np.argmin(abs(arrays['progress']-.4))),len(progress)-1]):
        for task in range(len(names)):
            points=projected[val & (y==task),index]
            ax.scatter(points[:,0],points[:,1],s=32,color=colors(task),label=task_ids[task])
        ax.set_title(f'{progress[index]:.0f}% observed');ax.set_xlabel('PC1')
    axes[0].set_ylabel('PC2');axes[-1].legend(fontsize=7,ncol=2)
    fig.suptitle(f'Held-out intention tokens; PCA fit on training episodes ({pca.explained_variance_ratio_.sum():.1%} variance)')
    save(fig,'intent_pca_stages.png')
    fig, ax = plt.subplots(figsize=(8,6))
    for task in range(len(names)):
        for index in np.flatnonzero(val & (y==task)):
            path=projected[index];ax.plot(path[:,0],path[:,1],color=colors(task),alpha=.28,linewidth=.8)
            ax.scatter(*path[0],color=colors(task),s=8,alpha=.4)
            ax.scatter(*path[-1],color=colors(task),s=35,marker='^')
        ax.scatter([],[],color=colors(task),label=task_ids[task])
    ax.set(xlabel='PC1',ylabel='PC2',title='Held-out token evolution: dots=start, triangles=end')
    ax.legend(fontsize=8,ncol=2);save(fig,'intent_pca_trajectories.png')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for key,ax in zip(['cosine_silhouette','adjusted_rand_index','within_between_ratio','distance_to_train_late_task_centroid'],axes.flat):
        ax.plot(progress,[r[key] for r in results['geometry']],marker='.',markersize=3)
        ax.set(xlabel='Observed progress (%)',ylabel=key.replace('_',' '));ax.grid(alpha=.2)
    fig.suptitle('Original-space geometry on held-out episodes');fig.tight_layout();save(fig,'cluster_geometry.png')
    late = arrays['progress'] >= .8
    late_centroids = np.stack([scaled[train & (y == task)][:, late].reshape(
        -1, scaled.shape[-1]).mean(0) for task in range(len(names))])
    columns = min(5, len(names)); rows = (len(names) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.2*columns, 3*rows),
                             sharex=True, sharey=True, squeeze=False)
    convergence = {}
    for task, ax in enumerate(axes.flat):
        if task >= len(names):
            ax.set_visible(False)
            continue
        tokens = scaled[val & (y == task)]
        distances = cosine_distance(tokens.reshape(-1, tokens.shape[-1]), late_centroids)
        distances = distances.reshape(tokens.shape[0], tokens.shape[1], len(names))
        own = distances[:, :, task].mean(0)
        other = np.delete(distances, task, axis=-1).min(-1).mean(0)
        convergence[task_ids[task]] = dict(task=str(names[task]), episodes=len(tokens),
            own_late_centroid_distance=own.tolist(), nearest_other_late_centroid_distance=other.tolist())
        ax.plot(progress, own, label='own task')
        ax.plot(progress, other, linestyle='--', label='nearest other task')
        ax.set_title(f'{task_ids[task]} (n={len(tokens)})');ax.grid(alpha=.2)
        ax.set_xlabel('Observed progress (%)')
    axes[0, 0].legend(fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel('Cosine distance')
    fig.suptitle('Per-task approach to fixed late-training-task centroids; lower is closer')
    fig.tight_layout();save(fig, 'intent_task_convergence.png')
    save_json(args.output/'per_task_convergence.json', convergence)
    cm=np.asarray(results['confusion_matrix']);norm=cm/cm.sum(1,keepdims=True)
    fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if norm[i,j]>.5 else 'black',fontsize=8)
    ax.set(xticks=np.arange(len(names)),yticks=np.arange(len(names)),xticklabels=task_ids,yticklabels=task_ids,
           xlabel='Predicted task',ylabel='True task',title='Final intention task decoder: held-out episodes')
    fig.colorbar(im,ax=ax,label='Row-normalized fraction');save(fig,'task_confusion.png')
    if not args.no_tsne:
        final=scaled[:,-1]
        coords=TSNE(n_components=2,init='pca',learning_rate='auto',perplexity=min(30.,(len(final)-1)/3),random_state=args.seed).fit_transform(final)
        fig,ax=plt.subplots(figsize=(8,6))
        for task in range(len(names)):
            a=train & (y==task);b=val & (y==task)
            ax.scatter(coords[a,0],coords[a,1],color=colors(task),alpha=.2,s=12)
            ax.scatter(coords[b,0],coords[b,1],color=colors(task),edgecolor='black',linewidth=.5,s=50,label=task_ids[task])
        ax.set(title='Final-token t-SNE: faint=train, outlined=held-out\nJoint unlabeled projection; visualization only',xlabel='t-SNE 1',ylabel='t-SNE 2')
        ax.legend(ncol=2,fontsize=8);save(fig,'intent_tsne.png')
    save_json(args.output/'task_labels.json',dict(zip(task_ids,names.tolist())))


def write_report(results, args):
    lines=['# Intention task probe','',f"Checkpoint: `{results['checkpoint']['checkpoint']}`; epoch {results['checkpoint']['epoch']}.",
        f"Whole episodes: {results['train_episodes']} train / {results['held_out_episodes']} held-out; {len(results['tasks'])} tasks.",
        'Task strings are labels only. No instruction, action target, or future observation enters intention extraction.',
        'Raw intention outputs are probed before cognitive memory fusion. Encoder weights are frozen.',
        '', '| Feature | Start balanced accuracy | 40% | End |', '|---|---:|---:|---:|']
    for feature, rows in results['features'].items():
        mid=min(rows,key=lambda r:abs(r['progress']-.4))
        lines.append(f"| {feature} | {rows[0]['balanced_accuracy']:.1%} | {mid['balanced_accuracy']:.1%} | {rows[-1]['balanced_accuracy']:.1%} |")
    lines+=['',f"Chance balanced accuracy: {results['chance_balanced_accuracy']:.1%}.",
        f"Final intent bootstrap 95% interval: {results['features']['intent'][-1]['balanced_accuracy_95pct_episode_bootstrap']}.",
        'The dotted curve uses one decoder trained on final-position training tokens at every held-out progress position.',
        '', '| Geometry | Start | End |','|---|---:|---:|']
    for key in ['cosine_silhouette','adjusted_rand_index','within_between_ratio','distance_to_train_late_task_centroid']:
        a,b=results['geometry'][0][key],results['geometry'][-1][key]
        lines.append(f'| {key} | {a:.4f} | {b:.4f} |' if a is not None and b is not None else f'| {key} | {a} | {b} |')
    lines+=['','## Interpretation limits','']+['- '+c for c in results['caveats']]
    lines+=['- Historical command inputs follow the checkpoint convention: '+results['checkpoint']['gripper_state_source']+'.',
        '- random_encoder retains the same trained state encoder; it is a random Mamba/readout control, not a fully untrained policy.',
        '- Clustering does not establish that actions or memory retrieval causally use task information.',
        '- See intent_task_convergence.png and per_task_convergence.json for approach to own and nearest other fixed late-training-task centroids.',
        '', '## Task IDs','']
    lines += [f'- T{i+1:02d}: {task}' for i,task in enumerate(results['tasks'])]
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'data/libero_goal.h5')
    parser.add_argument('--cache',type=Path,default=ROOT/'data/libero_goal.dinov2')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cameras',nargs='+',default=None)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--precision',choices=['float32','bfloat16'],default='float32')
    parser.add_argument('--sample-points',type=int,default=21)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--permutations',type=int,default=99)
    parser.add_argument('--random-control',action='store_true')
    parser.add_argument('--gripper-state-source',choices=['auto','causal','legacy_zero'],default='auto')
    parser.add_argument('--reuse-embeddings',action='store_true')
    parser.add_argument('--no-tsne',action='store_true')
    args=parser.parse_args(argv)
    if args.sample_points<2 or args.cpu_threads<1 or args.permutations<0:
        parser.error('Need sample-points >=2, cpu-threads >=1, permutations >=0')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    if args.reuse_embeddings:
        with np.load(args.output/'intent_embeddings.npz') as saved:
            arrays={key:saved[key] for key in saved.files}
        metadata=json.loads((args.output/'extraction.json').read_text())
        if Path(metadata['checkpoint'])!=args.checkpoint.resolve():
            raise ValueError('Cached embeddings belong to a different checkpoint')
    else:
        arrays,metadata=collect(args)
    with warnings.catch_warnings(record=True) as caught:
        results=analyze(arrays,metadata,args)
    save_json(args.output/'analysis_warnings.json',[str(w.message) for w in caught])
    write_report(results,args)
    print('COMPLETE',args.output/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
