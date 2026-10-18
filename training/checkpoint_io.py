"""Atomic checkpoints with optional fast staging and bounded background mirroring.

The training thread finishes each local save before continuing. The mirror only
reads immutable generation files, never live model/optimizer tensors or mutable
canonical checkpoint paths. Pending generations of the same destination coalesce.
"""
from collections import OrderedDict
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

import torch


def cpu_snapshot(value, memo=None):
    """Freeze tensors and metadata without altering dtype or optimizer state."""
    if memo is None:
        memo = {}
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device='cpu', copy=True)
    elif isinstance(value, dict):
        result = type(value)((key,cpu_snapshot(item,memo)) for key,item in value.items())
        if hasattr(value,'_metadata'):
            result._metadata = copy.deepcopy(value._metadata)
    elif isinstance(value, list):
        result = [cpu_snapshot(item,memo) for item in value]
    elif isinstance(value, tuple):
        result = tuple(cpu_snapshot(item,memo) for item in value)
    else:
        result = copy.deepcopy(value)
    memo[identity] = result
    return result


def latest_resume_checkpoint(paths):
    """Select by saved epoch, not mtime. mmap avoids reading optimizer storages."""
    found,errors = [],[]
    for path in dict.fromkeys(Path(p) for p in paths):
        if not path.exists():
            continue
        try:
            saved = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        except (RuntimeError,EOFError,OSError) as exc:
            errors.append((path,exc))
            continue
        found.append((int(saved['epoch']),path))
        del saved
    if errors and not found:
        raise RuntimeError(f'No complete resume checkpoint: {errors[0][0]}') from errors[0][1]
    for path,exc in errors:
        print(f'Ignoring incomplete resume candidate {path}: {exc}',flush=True)
    return max(found,key=lambda item:item[0])[1] if found else None


class CheckpointWriter:
    def __init__(self, output, staging_root=None):
        self.output = Path(output)
        self.local = None
        if staging_root is not None:
            digest = hashlib.sha256(str(self.output.resolve()).encode()).hexdigest()[:16]
            self.local = Path(staging_root)/digest/self.output.name
            self.local.mkdir(parents=True,exist_ok=True)
            if self.local.resolve()==self.output.resolve():
                raise ValueError('Checkpoint staging must differ from output')
        self._condition = threading.Condition()
        self._pending = OrderedDict()
        self._active = None
        self._failure = None
        self._closing = False
        self._thread = None
        self._last_mirrored = {}
        self._lock_file = None

    def __enter__(self):
        if self.local is not None:
            lock=(self.local/'.checkpoint-writer.lock').open('a')
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError:
                lock.close()
                raise RuntimeError(f'Another checkpoint writer owns {self.local}')
            self._lock_file=lock
            # A previous interrupted process may leave immutable generations.
            # Advertised canonical recovery files are separate hard links.
            try:
                for pattern in ['.*.ready','.*.tmp','.*.link']:
                    for path in self.local.glob(pattern):path.unlink()
            except BaseException:
                self._release_lock()
                raise
            self._thread = threading.Thread(target=self._mirror_loop,daemon=True,
                                            name='checkpoint-mirror')
            self._thread.start()
        return self

    def raise_if_failed(self):
        with self._condition:
            if self._failure is not None:
                raise RuntimeError(f'Checkpoint mirroring failed; local files retained in {self.local}') from self._failure

    def resume_path(self):
        paths = ([self.local/'training_last.pt'] if self.local is not None else [])
        return latest_resume_checkpoint(paths+[self.output/'training_last.pt'])

    def best_path(self):
        local = self.local/'intention_best.pt' if self.local is not None else None
        return local if local is not None and local.exists() else self.output/'intention_best.pt'

    @staticmethod
    def _estimated_bytes(payload):
        storages = {}
        def visit(value):
            if isinstance(value,torch.Tensor):
                storage=value.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
            elif isinstance(value,dict):
                for item in value.values():visit(item)
            elif isinstance(value,(tuple,list)):
                for item in value:visit(item)
        visit(payload)
        return sum(storages.values())+8*1024**2

    def save(self, filename, payload):
        """Publish a complete local checkpoint; NFS replacement runs separately."""
        self.raise_if_failed()
        if Path(filename).name != filename:
            raise ValueError('Checkpoint filename must be a basename')
        with self._condition:
            # Named latest/best checkpoints coalesce. Explicit per-epoch archives
            # cannot be dropped, so apply backpressure instead of filling disk.
            self._condition.wait_for(lambda:filename in self._pending or
                len(self._pending)<3 or self._failure is not None or self._closing)
            if self._closing:
                raise RuntimeError('Checkpoint writer is closed')
        self.raise_if_failed()
        directory = self.local if self.local is not None else self.output
        directory.mkdir(parents=True,exist_ok=True)
        token=uuid.uuid4().hex
        generation=directory/f'.{filename}.{token}.ready'
        temporary=directory/f'.{filename}.{token}.tmp'
        canonical=directory/filename
        if self.local is not None and shutil.disk_usage(directory).free < self._estimated_bytes(payload):
            raise OSError(f'Insufficient checkpoint staging space in {directory}; previous checkpoints retained')
        try:
            torch.save(payload,temporary)
            if self.local is None:
                temporary.replace(canonical)
                return
            temporary.replace(generation)
            # Hard links share storage. The canonical local recovery point is
            # replaced atomically, while any active mirror retains its inode.
            link=directory/f'.{filename}.{token}.link'
            try:
                os.link(generation,link)
                link.replace(canonical)
            finally:
                link.unlink(missing_ok=True)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        with self._condition:
            if self._closing:
                raise RuntimeError('Checkpoint writer is closed')
            previous=self._pending.pop(filename,None)
            if previous is not None:
                previous[0].unlink(missing_ok=True)
            self._pending[filename]=(generation,payload.get('epoch'))
            self._condition.notify_all()
        self.raise_if_failed()

    def _copy_atomic(self,source,destination):
        temporary=destination.with_name(destination.name+f'.{uuid.uuid4().hex}.tmp')
        try:
            shutil.copyfile(source,temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _release_lock(self):
        with self._condition:
            lock,self._lock_file=self._lock_file,None
        if lock is not None:lock.close()

    def _mirror_loop(self):
        try:self._mirror_work()
        finally:self._release_lock()

    def _mirror_work(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda:self._pending or self._closing)
                if not self._pending:
                    return
                filename,(source,epoch)=self._pending.popitem(last=False)
                self._active=filename
            started=time.perf_counter()
            try:
                self.output.mkdir(parents=True,exist_ok=True)
                self._copy_atomic(source,self.output/filename)
                source.unlink(missing_ok=True)
            except BaseException as exc:
                with self._condition:
                    self._failure=exc
                    self._active=None
                    self._condition.notify_all()
                return
            with self._condition:
                self._last_mirrored[filename]=dict(epoch=epoch,seconds=time.perf_counter()-started)
                self._active=None
                self._condition.notify_all()

    def status(self):
        self.raise_if_failed()
        with self._condition:
            return dict(staging_dir=str(self.local) if self.local is not None else None,
                        active=self._active,pending_files=len(self._pending),
                        last_mirrored=copy.deepcopy(self._last_mirrored))

    def close(self,wait=True):
        with self._condition:
            self._closing=True
            self._condition.notify_all()
        if self._thread is not None and wait:
            self._thread.join()
        self.raise_if_failed()

    def __exit__(self,exc_type,exc,tb):
        # On interruption do not wait for a blocked NFS server. Complete local
        # canonical files remain available for recovery; remote files are atomic.
        if exc_type is None:
            self.close()
        else:
            self.close(wait=False)
