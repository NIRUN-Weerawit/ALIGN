"""Saved episode selection and explicit context handover rules for evaluation."""
from pathlib import Path


def saved_development_episodes(config):
    """Prefer the protected context split; never reconstruct it randomly."""
    split = config.get('task_context_split')
    if split is not None:
        names = ('train_episodes', 'development_episodes', 'confirmation_episodes')
        if not all(isinstance(split.get(name), list) for name in names):
            raise ValueError('Checkpoint has an incomplete protected episode split')
        flat = [key for name in names for key in split[name]]
        if not all(isinstance(key, str) for key in flat) or len(flat) != len(set(flat)):
            raise ValueError('Checkpoint episode partitions contain duplicates or overlap')
        partition = split.get('evaluation_partition', 'development')
        if partition not in ('development', 'confirmation'):
            raise ValueError('Unknown saved evaluation partition')
        keys = split[partition + '_episodes']
    else:
        keys = config.get('val_episodes')
    if keys is None:
        return None
    if not keys or not all(isinstance(key, str) for key in keys) or len(keys) != len(set(keys)):
        raise ValueError('Saved development episode keys must be nonempty, unique strings')
    return list(keys)


def select_evaluation_episodes(available, config, episode_file=None, limit=None, selection='development'):
    """Use saved development by default; comparison files expose split membership."""
    if selection not in ('development', 'comparison'):
        raise ValueError('Unknown episode selection mode')
    if selection == 'comparison' and not episode_file:
        raise ValueError('Comparison mode requires an explicit episode file')
    available = list(available)
    allowed = saved_development_episodes(config)
    if allowed is not None and not set(allowed) <= set(available):
        raise ValueError('Dataset is missing saved development episodes')
    split = config.get('task_context_split')
    if split is not None:
        expected = set().union(*(split[name] for name in
            ('train_episodes', 'development_episodes', 'confirmation_episodes')))
        if expected != set(available):
            raise ValueError('Dataset episodes differ from the checkpoint protected split')
    if episode_file:
        requested = [line.strip() for line in Path(episode_file).read_text().splitlines() if line.strip()]
        if not requested or len(requested) != len(set(requested)):
            raise ValueError('Episode file must contain nonempty, unique keys')
        unknown = set(requested) - set(available)
        if unknown:
            raise ValueError(f'Unknown evaluation episodes: {sorted(unknown)}')
        if allowed is not None and selection == 'development':
            outside = set(requested) - set(allowed)
            if outside:
                raise ValueError('Episode list includes training or confirmation episodes outside saved '
                                 f'development: {sorted(outside)}')
        keys = sorted(requested)
        source = ('explicit comparison episode file' if selection == 'comparison' else
                  'explicit subset of saved development' if allowed is not None else 'explicit episode file')
    elif allowed is not None:
        keys, source = sorted(allowed), 'saved development'
    else:
        keys, source = sorted(available), 'all dataset episodes (no saved split)'
    if limit is not None:
        if limit < 1:
            raise ValueError('n_episodes must be positive')
        keys = keys[:limit]
    counts = ({name:len(set(keys) & set(split[name])) for name in
        ('train_episodes', 'development_episodes', 'confirmation_episodes')} if split else None)
    return keys, dict(source=source, selection_mode=selection,
        saved_development_count=len(allowed) if allowed else None,
        selected_episodes=keys, split_membership=counts,
        held_out_for_this_checkpoint=(counts['train_episodes'] == 0) if counts else None,
        confirmation_opened=(counts['confirmation_episodes'] > 0) if counts else None,
        manifest_sha256=split.get('manifest_sha256') if split else None)


def context_handover_mode(requested, switch_step, expert_only=False):
    if requested not in ('auto', 'freeze', 'handover', 'dynamic'):
        raise ValueError('Unknown task context mode')
    if expert_only:
        return 'dynamic'
    if requested == 'handover' and switch_step == 0:
        raise ValueError('Cannot take a handover snapshot without a human prefix; use freeze or dynamic at switch-at 0')
    if requested == 'auto':
        return 'freeze' if switch_step > 0 else 'dynamic'
    return requested


def reset_simulation(env, initial_state=None):
    """Reset controller state, then optionally restore a shared simulator state."""
    obs = env.reset()
    if initial_state is not None:
        if not hasattr(env, 'set_init_state'):
            raise ValueError('Simulator must support set_init_state for paired replay')
        obs = env.set_init_state(initial_state.copy())
    return obs


def load_initial_sim_state(episode, embedded=None, sidecar=None):
    """Optional non-pickled NPZ maps the converted episode key to its exact state."""
    import numpy as np
    state, source = embedded, 'demonstration'
    if sidecar:
        with np.load(sidecar, allow_pickle=False) as states:
            if episode not in states:
                raise ValueError(f'Initial-state sidecar is missing {episode}')
            state = states[episode].copy()
        source = 'episode-keyed initial-state sidecar'
    if state is not None:
        state = np.asarray(state)
        if state.ndim != 1 or not state.size or not np.isfinite(state).all():
            raise ValueError(f'{episode}: initial simulator state must be a finite flat vector')
    return state, source if state is not None else 'captured seeded reset'


def task_succeeded(env, done, reward):
    """LIBERO has an explicit task predicate; horizon termination is not success."""
    if hasattr(env, 'check_success'):
        return bool(env.check_success())
    underlying = getattr(env, 'env', env)
    if hasattr(underlying, '_check_success'):
        return bool(underlying._check_success())
    # Generic test/adaptor fallback: positive sparse task reward, never `done`.
    return bool(reward > 0)


def assess_expert_replay(replay, tolerance):
    """Success and pre-action pose fidelity must both hold, independent of video."""
    import numpy as np
    errors = np.asarray(replay['errors'], dtype=float)
    finite = len(errors) > 0 and bool(np.isfinite(errors).all())
    mean = float(errors.mean()) if finite else None
    initial = float(errors[0]) if finite else None
    success = bool(replay['success'])
    valid = success and finite and mean <= tolerance and initial <= tolerance
    return dict(valid=valid, task_success=success, mean_position_error=mean,
        initial_position_error=initial, position_tolerance=tolerance,
        pose_alignment='observation before action',
        reason=None if valid else ('expert did not complete task' if not success else
            'expert replay does not match recorded EEF observations'))
