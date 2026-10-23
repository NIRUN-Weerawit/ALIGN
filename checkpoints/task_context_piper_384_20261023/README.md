# Context teacher/student refit on the Piper split

384 training and 44 held-out episodes, exactly matching the Piper ablations.

See [protocol, results and verification](../../reports/task_context_piper_384_20261023/summary.md).

Use `observer_frozen.pt` for new action-policy training. Existing action heads require retraining for this new latent space.

On another machine, run `git lfs pull` after checking out this branch. The
checkpoint must contain a PyTorch binary (about 18 MB), not the small text
pointer beginning with `version https://git-lfs.github.com/spec/v1`.
