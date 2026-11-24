"""Pair offline datasets with caches and namespace overlapping episode IDs."""
from pathlib import Path


def dataset_sources(data, cache):
    paths = [data] if isinstance(data, (str, Path)) else list(data)
    caches = [cache] if isinstance(cache, (str, Path)) else list(cache)
    if not paths or len(paths) != len(caches):
        raise ValueError('Supply one --cache directory per --data file, in the same order')
    sources = [(Path(path).resolve(), Path(features).resolve()) for path, features in zip(paths, caches)]
    if len({path.stem for path, _ in sources}) != len(sources):
        raise ValueError('Dataset filenames must have distinct stems for episode namespaces')
    return sources


def episode_key(path, key):
    return f'{path.stem}__{key}'


class MultiEpisodeDataset:
    """Dispatch episode reads to the original HDF5 and feature cache."""
    def __init__(self, sources, **kwargs):
        from data.align_dataset import ALIGNDataset

        self.datasets = [ALIGNDataset(str(path), dinov2_path=str(cache), **kwargs)
                         for path, cache in sources]
        self.cameras = self.datasets[0].cameras
        self._episode_keys = []
        self.locations = []
        for (path, _), dataset in zip(sources, self.datasets):
            self._episode_keys.extend(episode_key(path, key) for key in dataset._episode_keys)
            self.locations.extend((dataset, idx) for idx in range(len(dataset._episode_keys)))

    def _get_episode_length(self, ep):
        dataset, local = self.locations[ep]
        return dataset._get_episode_length(local)

    def _read_text(self, ep):
        dataset, local = self.locations[ep]
        return dataset._read_text(local)

    def _read_frames_dinov2_view(self, ep, start, length):
        dataset, local = self.locations[ep]
        return dataset._read_frames_dinov2_view(local, start, length)

    _read_frames_dinov2 = _read_frames_dinov2_view

    def _read_poses(self, ep, start, length):
        dataset, local = self.locations[ep]
        return dataset._read_poses(local, start, length)

    def _read_poses_gripper(self, ep, start, length):
        dataset, local = self.locations[ep]
        return dataset._read_poses_gripper(local, start, length)

    def _read_actions(self, ep, start, length):
        dataset, local = self.locations[ep]
        return dataset._read_actions(local, start, length)

    def close(self):
        for dataset in self.datasets:
            dataset.close()


def open_datasets(data, cache, **kwargs):
    from data.align_dataset import ALIGNDataset

    sources = dataset_sources(data, cache)
    if len(sources) == 1:
        # Preserve single-file behavior and paths in existing manifests.
        path = data if isinstance(data, (str, Path)) else data[0]
        features = cache if isinstance(cache, (str, Path)) else cache[0]
        return ALIGNDataset(path, dinov2_path=features, **kwargs)
    return MultiEpisodeDataset(sources, **kwargs)
